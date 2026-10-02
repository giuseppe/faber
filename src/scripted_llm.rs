/*
 * faber
 *
 * Copyright (C) 2025 Giuseppe Scrivano <giuseppe@scrivano.org>
 * faber is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * faber is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with faber.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

//! A scripted stand-in for a model, selected with `--model script:PATH`,
//! for testing agent workflows end to end - spawning, fan-out, tasks -
//! deterministically and without a model server.
//!
//! The script is a JSON list of rules. For each request, the first rule
//! whose `if` text appears in the latest message (a user turn, an injected
//! message, or a tool's result) gives the response: `reply` text, or a
//! `tool` call with `arguments`. Agents run concurrently, so rules match on
//! what was said rather than on the order of requests.
//!
//! ```json
//! [
//!   {"if": "triage", "tool": "fan_out",
//!    "arguments": {"items": ["1", "2"], "prompt": "check issue {item}"}},
//!   {"if": "check issue", "reply": "looks fine", "delay_ms": 200},
//!   {"if": "## 1.", "reply": "all checked"}
//! ]
//! ```
//!
//! `once: true` makes a rule apply only the first time it matches, and
//! `delay_ms` makes the response take that long (interruptibly). With no
//! matching rule, the reply says so, quoting the message.

use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use crate::openai::{
    Choice, FunctionCall, InterruptedError, Message, OpenAIResponse, ProgressInfo, ResponseMode,
    StatusUpdate, ToolCall, ToolsCollection, Usage, accumulate_usage, run_tool_calls,
};

/// The `--model` prefix that selects a script.
const PREFIX: &str = "script:";

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct Rule {
    /// Text the latest message must contain; matches anything if absent.
    #[serde(default, rename = "if")]
    when: Option<String>,
    #[serde(default)]
    reply: Option<String>,
    #[serde(default)]
    tool: Option<String>,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
    #[serde(default)]
    once: bool,
    #[serde(default)]
    delay_ms: u64,
}

pub fn is_scripted_model(model: &str) -> bool {
    model.starts_with(PREFIX)
}

/// Each script's rules, and which `once` rules were used up - per script
/// path, shared by every agent using it.
fn scripts() -> &'static Mutex<HashMap<String, (Vec<Rule>, Vec<bool>)>> {
    static SCRIPTS: OnceLock<Mutex<HashMap<String, (Vec<Rule>, Vec<bool>)>>> = OnceLock::new();
    SCRIPTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn load(path: &str) -> Result<Vec<Rule>, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("can't read model script {}: {}", path, e))?;
    let rules: Vec<Rule> =
        serde_json::from_str(&text).map_err(|e| format!("invalid model script {}: {}", path, e))?;
    for (i, rule) in rules.iter().enumerate() {
        if rule.reply.is_some() == rule.tool.is_some() {
            return Err(format!(
                "model script {}: rule {} needs either reply or tool",
                path,
                i + 1
            )
            .into());
        }
    }
    Ok(rules)
}

/// The rule for a request whose latest message is `last`, marking a `once`
/// rule as used.
fn pick(path: &str, last: &str) -> Result<Option<Rule>, Box<dyn Error>> {
    let mut scripts = scripts().lock().unwrap_or_else(|e| e.into_inner());
    if !scripts.contains_key(path) {
        let rules = load(path)?;
        let used = vec![false; rules.len()];
        scripts.insert(path.to_string(), (rules, used));
    }
    let (rules, used) = scripts.get_mut(path).expect("just inserted");
    for (i, rule) in rules.iter().enumerate() {
        if used[i] {
            continue;
        }
        if rule.when.as_deref().is_none_or(|w| last.contains(w)) {
            if rule.once {
                used[i] = true;
            }
            return Ok(Some(rule.clone()));
        }
    }
    Ok(None)
}

fn check_interrupted(
    ctrl_c_rx: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<(), InterruptedError> {
    if let Some(rx) = ctrl_c_rx {
        if rx.lock().map(|r| r.try_recv().is_ok()).unwrap_or(false) {
            return Err(InterruptedError::new("Interrupted by user"));
        }
    }
    Ok(())
}

fn report(mode: &ResponseMode, status: StatusUpdate, start: Instant) -> Result<(), Box<dyn Error>> {
    if let ResponseMode::Streaming {
        progress_handler, ..
    } = mode
    {
        progress_handler(&ProgressInfo {
            status,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })?;
    }
    Ok(())
}

pub fn post_request_scripted(
    messages: Vec<Message>,
    model: &str,
    tools_collection: &ToolsCollection,
    mode: ResponseMode,
    ctx: &crate::ToolContext,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    let path = &model[PREFIX.len()..];
    let start = Instant::now();
    let mut messages = messages;
    let mut turn_usage: Option<Usage> = None;
    let mut request = 0;
    loop {
        request += 1;
        if ctx.max_requests.is_some_and(|limit| request > limit) {
            return Err(crate::openai::request_budget_error(
                ctx.max_requests.unwrap_or(0),
            ));
        }
        check_interrupted(&ctrl_c_rx)?;
        report(&mode, StatusUpdate::Thinking, start)?;
        let last = messages
            .last()
            .and_then(|m| m.content.clone())
            .unwrap_or_default();
        let rule = pick(path, &last)?;
        if let Some(rule) = &rule {
            let deadline = Instant::now() + Duration::from_millis(rule.delay_ms);
            while Instant::now() < deadline {
                check_interrupted(&ctrl_c_rx)?;
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        let prompt_chars: usize = messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .map(|c| c.len())
            .sum();
        let message = match &rule {
            Some(Rule {
                tool: Some(tool),
                arguments,
                ..
            }) => Message {
                role: "assistant".to_string(),
                content: None,
                tool_call_id: None,
                name: None,
                tool_calls: Some(vec![ToolCall {
                    index: Some(0),
                    id: format!("call_script_{}", request),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: tool.clone(),
                        arguments: arguments
                            .clone()
                            .unwrap_or_else(|| serde_json::json!({}))
                            .to_string(),
                    },
                }]),
            },
            other => {
                let text = match other {
                    Some(rule) => rule.reply.clone().unwrap_or_default(),
                    None => format!(
                        "(no script rule matched: {})",
                        last.chars().take(200).collect::<String>()
                    ),
                };
                crate::openai::make_message("assistant", text)
            }
        };
        let usage = Usage {
            prompt_tokens: Some((prompt_chars / 4) as u32),
            completion_tokens: Some(
                (message.content.as_ref().map_or(20, |c| c.len()) / 4).max(1) as u32,
            ),
            total_tokens: None,
        };
        accumulate_usage(&mut turn_usage, Some(&usage));
        messages.push(message.clone());

        if let Some(calls) = &message.tool_calls {
            let results = run_tool_calls(tools_collection, calls, ctx, &mode, start)?;
            messages.extend(results);
            continue;
        }
        if let ResponseMode::Streaming { stream_handler, .. } = &mode {
            stream_handler(message.content.as_deref().unwrap_or(""))?;
            stream_handler("")?;
        }
        report(
            &mode,
            StatusUpdate::Complete {
                usage: Some(usage.clone()),
            },
            start,
        )?;
        return Ok(OpenAIResponse {
            error: None,
            choices: Some(vec![Choice {
                message,
                finish_reason: Some("stop".to_string()),
                native_finish_reason: None,
            }]),
            usage: Some(usage),
            turn_usage,
            history: messages,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{ToolItem, make_message};

    /// Writes `rules` to a fresh script file and returns its `--model`.
    pub(crate) fn script(name: &str, rules: serde_json::Value) -> String {
        let path =
            std::env::temp_dir().join(format!("faber_script_{}_{}.json", name, std::process::id()));
        std::fs::write(&path, rules.to_string()).unwrap();
        format!("{}{}", PREFIX, path.display())
    }

    fn shout(args: &String, _ctx: &crate::ToolContext) -> Result<String, Box<dyn Error>> {
        let v: serde_json::Value = serde_json::from_str(args)?;
        Ok(format!(
            "SHOUTED {}",
            v["text"].as_str().unwrap_or("").to_uppercase()
        ))
    }

    fn ask(model: &str, prompt: &str) -> Result<OpenAIResponse, Box<dyn Error>> {
        let mut tools = ToolsCollection::new();
        tools.insert(
            "shout".to_string(),
            ToolItem {
                callback: shout,
                schema: String::new(),
            },
        );
        let ctx = crate::ToolContext::new(|_: &str| {});
        post_request_scripted(
            vec![make_message("user", prompt.to_string())],
            model,
            &tools,
            ResponseMode::Complete,
            &ctx,
            None,
        )
    }

    fn answer(response: &OpenAIResponse) -> String {
        response.choices.as_ref().unwrap()[0]
            .message
            .content
            .clone()
            .unwrap()
    }

    #[test]
    fn test_rules_drive_tool_calls_and_replies() {
        let model = script(
            "basic",
            serde_json::json!([
                {"if": "greet", "tool": "shout", "arguments": {"text": "hello"}},
                {"if": "SHOUTED HELLO", "reply": "I shouted."},
            ]),
        );
        let response = ask(&model, "please greet").unwrap();
        assert_eq!(answer(&response), "I shouted.");
        // user, assistant tool call, tool result, final answer.
        assert_eq!(response.history.len(), 4);
        assert!(response.turn_usage.unwrap().prompt_tokens.unwrap() > 0);
        let fallback = answer(&ask(&model, "something else").unwrap());
        assert_eq!(fallback, "(no script rule matched: something else)");
    }

    #[test]
    fn test_once_rules_are_used_up() {
        let model = script(
            "once",
            serde_json::json!([
                {"if": "hi", "reply": "first", "once": true},
                {"if": "hi", "reply": "later"},
            ]),
        );
        assert_eq!(answer(&ask(&model, "hi").unwrap()), "first");
        assert_eq!(answer(&ask(&model, "hi").unwrap()), "later");
        assert_eq!(answer(&ask(&model, "hi").unwrap()), "later");
    }

    #[test]
    fn test_invalid_scripts_are_reported() {
        let model = script("invalid", serde_json::json!([{"if": "x"}]));
        assert!(
            ask(&model, "x")
                .unwrap_err()
                .to_string()
                .contains("needs either reply or tool")
        );
        assert!(ask("script:/nonexistent/faber.json", "x").is_err());
    }

    #[test]
    fn test_delay_is_interruptible() {
        let model = script(
            "slow",
            serde_json::json!([{"reply": "late", "delay_ms": 30000}]),
        );
        let (tx, rx) = mpsc::channel();
        let ctx = crate::ToolContext::new(|_: &str| {});
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            let _ = tx.send(());
        });
        let started = Instant::now();
        let result = post_request_scripted(
            vec![make_message("user", "go".to_string())],
            &model,
            &ToolsCollection::new(),
            ResponseMode::Complete,
            &ctx,
            Some(Arc::new(Mutex::new(rx))),
        );
        assert!(
            result
                .unwrap_err()
                .downcast_ref::<InterruptedError>()
                .is_some()
        );
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
