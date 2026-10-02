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

//! The `fan_out` tool: runs the same task for many items at once, each in
//! its own worker agent, and returns all their results together in one
//! tool result - unlike `spawn_agent`, whose results arrive later, one
//! message at a time.
//!
//! Workers only get read-only tools unless others are asked for by name:
//! workers writing to the same files concurrently could overwrite each
//! other's changes.

use serde::Deserialize;
use std::collections::VecDeque;
use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use crate::openai::{
    self, ContextLengthError, InterruptedError, ProgressInfo, ResponseMode, StatusUpdate,
    ToolsCollection, make_message, post_request_with_mode,
};
use crate::{SubAgentContext, final_response_text};
use faber::ToolContext;

const MAX_ITEMS: usize = 100;
const DEFAULT_MAX_PARALLEL: usize = 4;
const MAX_PARALLEL: usize = 16;

/// How many times a worker shortens its tool results to recover from a
/// context overflow before giving up.
const MAX_TOOL_RESULT_SHRINKS: usize = 3;

/// Smallest share of the output budget any one item's result gets.
const MIN_RESULT_CHARS: usize = 2000;

/// What workers get when `tools` isn't given: tools that only read.
const READ_ONLY_TOOLS: &[&str] = &[
    "read_file",
    "glob",
    "grep_in_current_directory",
    "lsp",
    "fetch_web_content",
    "github_issue",
    "github_issue_comments",
    "github_issues",
    "github_pull_request",
    "github_pull_request_patch",
    "github_pull_requests",
    "kb_search",
    "kb_read",
    "kb_list",
];

/// Tools a worker can never have: no nested agents, and no plan (workers
/// have no agent identity to keep one under).
const NEVER_FOR_WORKERS: &[&str] = &["spawn_agent", "fan_out", "plan_update", "plan_get"];

const WORKER_INSTRUCTIONS: &str = "You are a worker in a fan-out: the same task is being run for many items in parallel, \
and you handle exactly one of them. Work only on your item. Nobody will answer questions, so don't ask any. \
Finish with a concise, self-contained result - it's all the coordinating agent will see of your work.";

#[derive(Deserialize)]
struct Params {
    items: Vec<String>,
    prompt: String,
    #[serde(default)]
    max_parallel: Option<usize>,
    #[serde(default)]
    tools: Option<Vec<String>>,
}

/// The prompt for one item: `{item}` in `template` replaced by it, or the
/// item appended if there's no placeholder.
pub(crate) fn expand_prompt(template: &str, item: &str) -> String {
    if template.contains("{item}") {
        template.replace("{item}", item)
    } else {
        format!("{}\n\nItem: {}", template, item)
    }
}

/// The tools workers get: `requested` by name (each must exist and be
/// allowed), or else every available read-only tool.
pub(crate) fn worker_tools(
    available: &ToolsCollection,
    requested: Option<&[String]>,
) -> Result<ToolsCollection, String> {
    let names: Vec<String> = match requested {
        None => READ_ONLY_TOOLS
            .iter()
            .filter(|n| available.contains_key(**n))
            .map(|n| n.to_string())
            .collect(),
        Some(names) => {
            for name in names {
                if NEVER_FOR_WORKERS.contains(&name.as_str()) {
                    return Err(format!("workers can't use {}", name));
                }
                if !available.contains_key(name) {
                    return Err(format!("unknown tool '{}'", name));
                }
            }
            names.to_vec()
        }
    };
    Ok(names
        .into_iter()
        .map(|n| {
            let tool = available[&n].clone();
            (n, tool)
        })
        .collect())
}

/// All the results, in item order, each capped to its share of
/// `budget_chars`.
pub(crate) fn format_results(
    items: &[String],
    results: &[Option<Result<String, String>>],
    budget_chars: usize,
) -> String {
    let per_item = (budget_chars / items.len().max(1)).max(MIN_RESULT_CHARS);
    let mut out = String::new();
    for (i, (item, result)) in items.iter().zip(results).enumerate() {
        let body = match result {
            Some(Ok(text)) => openai::truncate_tool_output(text.trim(), per_item),
            Some(Err(e)) => format!("Error: {}", e),
            None => "Not run.".to_string(),
        };
        out.push_str(&format!("## {}. {}\n{}\n\n", i + 1, item, body));
    }
    out
}

/// Runs one worker to completion. `cancel` gets a message on Ctrl-C.
fn run_worker(
    sa_ctx: &SubAgentContext,
    ctx: &ToolContext,
    tools: &ToolsCollection,
    status_key: &str,
    prompt: String,
    cancel: mpsc::Receiver<()>,
) -> Result<String, Box<dyn Error>> {
    let mut worker_ctx = ToolContext::new(|_: &str| {});
    worker_ctx.db = ctx.db.clone();
    worker_ctx.context_window = ctx.context_window;
    // Streamed, although nothing is shown, so Ctrl-C is noticed between
    // chunks rather than only once a whole response has arrived.
    let mode = || ResponseMode::Streaming {
        stream_handler: Box::new(|_: &str| Ok(())),
        reasoning_handler: Box::new(|_: &str| Ok(())),
        progress_handler: {
            let status_bar = sa_ctx.status_bar.clone();
            let key = status_key.to_string();
            Box::new(move |progress: &ProgressInfo| {
                let status = match &progress.status {
                    StatusUpdate::Thinking => "Thinking".to_string(),
                    StatusUpdate::ToolStart { name, .. } => format!("Running {}", name),
                    StatusUpdate::SendingRequest { .. } => "Waiting for response".to_string(),
                    _ => return Ok(()),
                };
                status_bar.set_agent_status(&key, &status, false);
                Ok(())
            })
        },
    };
    let cancel = Some(Arc::new(Mutex::new(cancel)));
    let mut messages = vec![
        make_message("system", WORKER_INSTRUCTIONS.to_string()),
        make_message("user", prompt),
    ];
    // Like the chat, recover from a context overflow by shortening tool
    // results (keeping the worker's progress) and carrying on.
    let mut shrinks = 0;
    let result = loop {
        let result = post_request_with_mode(
            messages,
            tools,
            &sa_ctx.opts,
            mode(),
            &worker_ctx,
            cancel.clone(),
        );
        let Err(e) = &result else { break result };
        if e.downcast_ref::<InterruptedError>().is_some() {
            return Err(Box::new(InterruptedError::new(
                "Operation interrupted by user",
            )));
        }
        let shrunk = e
            .downcast_ref::<ContextLengthError>()
            .filter(|_| shrinks < MAX_TOOL_RESULT_SHRINKS)
            .and_then(|overflow| openai::shrink_tool_results(&overflow.history, &overflow.message));
        match shrunk {
            Some(shrunk) => {
                shrinks += 1;
                messages = shrunk;
            }
            None => break result,
        }
    };
    if let Some(usage) = result.as_ref().ok().and_then(|r| r.turn_usage.as_ref()) {
        sa_ctx
            .session_usage
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .record(usage);
    }
    final_response_text(result).map_err(|e| e.into())
}

/// entrypoint for the fan_out tool
pub(crate) fn tool_fan_out(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    let params: Params = serde_json::from_str(params_str)?;
    let sa_ctx = ctx
        .extra
        .as_ref()
        .and_then(|e| e.downcast_ref::<SubAgentContext>())
        .ok_or("fan_out is only available to the main chat agent")?;
    if params.items.is_empty() || params.items.len() > MAX_ITEMS {
        return Err(format!("give between 1 and {} items", MAX_ITEMS).into());
    }
    let tools = worker_tools(&sa_ctx.tools, params.tools.as_deref())?;
    let parallel = params
        .max_parallel
        .unwrap_or(DEFAULT_MAX_PARALLEL)
        .clamp(1, MAX_PARALLEL)
        .min(params.items.len());
    let total = params.items.len();

    let queue: Mutex<VecDeque<usize>> = Mutex::new((0..total).collect());
    let results: Mutex<Vec<Option<Result<String, String>>>> = Mutex::new(vec![None; total]);
    let cancelled = AtomicBool::new(false);
    let finished = AtomicBool::new(false);
    let done_count = AtomicUsize::new(0);
    // Every running worker's Ctrl-C channel. The watcher sets `cancelled`
    // before sending to these under the lock, and a worker checks
    // `cancelled` after registering under the same lock, so none can miss
    // the signal.
    let cancel_senders: Mutex<Vec<mpsc::Sender<()>>> = Mutex::new(Vec::new());

    ctx.println(&format!(
        "Fanning out to {} items, {} at a time...",
        total, parallel
    ));

    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !finished.load(Ordering::Relaxed) {
                let interrupted = ctx.interrupt.as_ref().is_some_and(|rx| {
                    rx.lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .try_recv()
                        .is_ok()
                });
                if interrupted {
                    cancelled.store(true, Ordering::Relaxed);
                    for tx in cancel_senders
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .iter()
                    {
                        let _ = tx.send(());
                    }
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });

        let workers: Vec<_> = (0..parallel)
            .map(|_| {
                scope.spawn(|| {
                    loop {
                        let Some(index) =
                            queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front()
                        else {
                            return;
                        };
                        let (tx, rx) = mpsc::channel();
                        cancel_senders
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(tx);
                        if cancelled.load(Ordering::Relaxed) {
                            return;
                        }
                        let item = &params.items[index];
                        let key = format!("fan-out {}/{}", index + 1, total);
                        sa_ctx.status_bar.set_agent_status(&key, "Starting", true);
                        let result = run_worker(
                            sa_ctx,
                            ctx,
                            &tools,
                            &key,
                            expand_prompt(&params.prompt, item),
                            rx,
                        );
                        sa_ctx.status_bar.clear_agent_status(&key);
                        let result = match result {
                            Err(e) if e.downcast_ref::<InterruptedError>().is_some() => return,
                            other => other.map_err(|e| e.to_string()),
                        };
                        let done = done_count.fetch_add(1, Ordering::Relaxed) + 1;
                        ctx.println(&format!(
                            "[{}/{}] {} {}",
                            done,
                            total,
                            if result.is_ok() { "done:" } else { "failed:" },
                            item
                        ));
                        results.lock().unwrap_or_else(|e| e.into_inner())[index] = Some(result);
                    }
                })
            })
            .collect();
        for worker in workers {
            let _ = worker.join();
        }
        finished.store(true, Ordering::Relaxed);
    });

    if cancelled.load(Ordering::Relaxed) {
        return Err(Box::new(InterruptedError::new(
            "Operation interrupted by user",
        )));
    }
    let results = results.into_inner().unwrap_or_else(|e| e.into_inner());
    Ok(format_results(
        &params.items,
        &results,
        ctx.max_tool_output_chars(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ToolItem;

    /// A chat-like context for fan_out against the built-in dummy model,
    /// with the Ctrl-C sender and the session usage it records into.
    fn dummy_chat_ctx() -> (
        ToolContext,
        mpsc::Sender<()>,
        Arc<Mutex<crate::SessionUsage>>,
    ) {
        let usage = Arc::new(Mutex::new(crate::SessionUsage::default()));
        let (ctrl_c_tx, ctrl_c_rx) = mpsc::channel();
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.interrupt = Some(Arc::new(Mutex::new(ctrl_c_rx)));
        ctx.extra = Some(Arc::new(SubAgentContext {
            tools: Arc::new(crate::initialize_tools(false, None)),
            opts: openai::Opts {
                max_tokens: None,
                model: "dummy".to_string(),
                endpoint: String::new(),
                tool_choice: None,
                api_key: None,
                max_retries: None,
                retry_base_delay_secs: None,
                parameters: Default::default(),
            },
            session_id: "test".to_string(),
            active_subagents: Arc::new(AtomicUsize::new(0)),
            status_bar: Arc::new(crate::status_bar::StatusBar::new()),
            session_usage: usage.clone(),
        }));
        (ctx, ctrl_c_tx, usage)
    }

    #[test]
    fn test_fan_out_runs_every_item_and_returns_results_in_order() {
        let (ctx, _ctrl_c, usage) = dummy_chat_ctx();
        let params = serde_json::json!({
            "items": ["one", "two", "three"],
            "prompt": "Describe {item}.",
            "max_parallel": 2,
        });
        let out = tool_fan_out(&params.to_string(), &ctx).unwrap();
        let positions: Vec<usize> = ["## 1. one", "## 2. two", "## 3. three"]
            .iter()
            .map(|h| out.find(h).unwrap_or_else(|| panic!("{h} missing: {out}")))
            .collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]));
        assert!(out.contains("Dummy response"), "{out}");
        assert_eq!(
            usage.lock().unwrap().turns,
            3,
            "every worker's tokens count: {out}"
        );
    }

    #[test]
    fn test_fan_out_stops_every_worker_on_ctrl_c() {
        let (ctx, ctrl_c, _) = dummy_chat_ctx();
        // "slow N" makes the dummy stream N paragraphs a word every 200ms -
        // minutes, unless interrupted.
        let params = serde_json::json!({
            "items": ["slow 100", "slow 100", "slow 100", "slow 100"],
            "prompt": "{item}",
            "max_parallel": 2,
        });
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(500));
            let _ = ctrl_c.send(());
        });
        let started = std::time::Instant::now();
        let err = tool_fan_out(&params.to_string(), &ctx).unwrap_err();
        assert!(err.downcast_ref::<InterruptedError>().is_some(), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn test_fan_out_needs_the_main_chat_context() {
        let ctx = ToolContext::new(|_: &str| {});
        let params = serde_json::json!({"items": ["a"], "prompt": "x"});
        assert!(tool_fan_out(&params.to_string(), &ctx).is_err());
    }

    #[test]
    fn test_expand_prompt() {
        assert_eq!(
            expand_prompt("Review {item} now", "a.rs"),
            "Review a.rs now"
        );
        assert_eq!(
            expand_prompt("Review it", "a.rs"),
            "Review it\n\nItem: a.rs"
        );
    }

    fn noop(_: &String, _: &ToolContext) -> Result<String, Box<dyn Error>> {
        Ok(String::new())
    }

    fn available(names: &[&str]) -> ToolsCollection {
        names
            .iter()
            .map(|n| {
                (
                    n.to_string(),
                    ToolItem {
                        callback: noop,
                        schema: String::new(),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn test_worker_tools_default_to_the_available_read_only_ones() {
        let tools = available(&[
            "read_file",
            "write_file",
            "lsp",
            "run_command",
            "spawn_agent",
        ]);
        let mut names: Vec<_> = worker_tools(&tools, None).unwrap().into_keys().collect();
        names.sort();
        assert_eq!(names, vec!["lsp", "read_file"]);
    }

    #[test]
    fn test_worker_tools_explicit_list_is_checked() {
        let tools = available(&["read_file", "write_file", "spawn_agent"]);
        let picked = worker_tools(&tools, Some(&["write_file".to_string()])).unwrap();
        assert_eq!(picked.len(), 1);
        assert!(worker_tools(&tools, Some(&["spawn_agent".to_string()])).is_err());
        assert!(worker_tools(&tools, Some(&["nope".to_string()])).is_err());
    }

    #[test]
    fn test_format_results_in_order_with_errors_and_caps() {
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let results = vec![
            Some(Ok("x".repeat(10_000))),
            Some(Err("boom".to_string())),
            None,
        ];
        let out = format_results(&items, &results, 6000);
        let a = out.find("## 1. a").unwrap();
        let b = out.find("## 2. b\nError: boom").unwrap();
        let c = out.find("## 3. c\nNot run.").unwrap();
        assert!(a < b && b < c);
        assert!(out.contains("characters omitted"), "a's result is capped");
    }
}
