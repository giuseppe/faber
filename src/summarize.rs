/*
 * swarmblabla
 *
 * Copyright (C) 2025 Giuseppe Scrivano <giuseppe@scrivano.org>
 * swarmblabla is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * swarmblabla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with swarmblabla.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

//! Conversation summarization, used by `/summarize` and to recover when a
//! request no longer fits in the model's context window.

use std::error::Error;
use std::fmt::Write;
use std::sync::{Arc, Mutex, mpsc};

use crate::openai::{
    ContextLengthError, Message, Opts, ResponseMode, ToolsCollection, make_message,
    post_request_with_mode,
};
use swarmblabla::ToolContext;

/// Marks the system message that carries a summary, so a later summary can
/// fold it in instead of treating it as a prompt to preserve.
pub const SUMMARY_PREFIX: &str = "Summary of the conversation so far:\n";

const SUMMARY_INSTRUCTIONS: &str = "You are summarizing a conversation between a user and an AI assistant so that the conversation can continue with a much shorter history.  \
Write a summary that lets the assistant carry on without the original messages.  Include: the user's goals and the task currently in progress, decisions made, \
facts learned (file paths, names, values, commands and their results), work already completed, and what remains to be done.  \
Be concise and factual, do not address the user, and do not call any tools.";

/// Longest rendering of a single message in the transcript sent for
/// summarization; tool output is the usual offender.
const MAX_MESSAGE_CHARS: usize = 2000;

/// How many times the transcript is halved when the summarization request
/// itself does not fit in the context window.
const MAX_SHRINK_ATTEMPTS: usize = 6;

pub fn is_summary(msg: &Message) -> bool {
    msg.role == "system"
        && msg
            .content
            .as_deref()
            .is_some_and(|c| c.starts_with(SUMMARY_PREFIX))
}

fn truncate(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((end, _)) => format!(
            "{}... [{} more characters]",
            &text[..end],
            text[end..].chars().count()
        ),
        None => text.to_string(),
    }
}

fn render_message(out: &mut String, msg: &Message) {
    let content = msg.content.as_deref().unwrap_or("");
    match msg.role.as_str() {
        "tool" => {
            let _ = writeln!(
                out,
                "[tool result: {}]\n{}\n",
                msg.name.as_deref().unwrap_or("unknown"),
                truncate(content, MAX_MESSAGE_CHARS)
            );
        }
        role => {
            let _ = writeln!(out, "[{}]", role);
            if !content.is_empty() {
                let _ = writeln!(out, "{}", truncate(content, MAX_MESSAGE_CHARS));
            }
            for call in msg.tool_calls.iter().flatten() {
                let _ = writeln!(
                    out,
                    "(called tool {} with {})",
                    call.function.name,
                    truncate(&call.function.arguments, MAX_MESSAGE_CHARS)
                );
            }
            out.push('\n');
        }
    }
}

fn render_transcript(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        if is_summary(msg) {
            let _ = writeln!(
                out,
                "[earlier summary]\n{}\n",
                msg.content
                    .as_deref()
                    .and_then(|c| c.strip_prefix(SUMMARY_PREFIX))
                    .unwrap_or("")
            );
        } else {
            render_message(&mut out, msg);
        }
    }
    out
}

fn request_summary(
    transcript: &str,
    opts: &Opts,
    ctrl_c_rx: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<String, Box<dyn Error>> {
    let messages = vec![
        make_message("system", SUMMARY_INSTRUCTIONS.to_string()),
        make_message(
            "user",
            format!("Summarize this conversation:\n\n{}", transcript),
        ),
    ];
    let ctx = ToolContext::new(|_: &str| {});
    let response = post_request_with_mode(
        messages,
        &ToolsCollection::new(),
        opts,
        ResponseMode::Complete,
        &ctx,
        ctrl_c_rx.clone(),
    )?;
    let summary = response
        .choices
        .as_ref()
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.message.content.as_deref())
        .map(str::trim)
        .unwrap_or("");
    if summary.is_empty() {
        return Err("The model returned an empty summary.".into());
    }
    Ok(summary.to_string())
}

/// Returns a shorter history for `messages`: the prompts (system messages)
/// followed by a model-written summary of everything else.
///
/// With `keep_last_user`, the most recent user message is re-appended
/// verbatim after the summary, so a request that was cut short by a context
/// overflow can simply be retried.
pub fn summarize_conversation(
    messages: &[Message],
    keep_last_user: bool,
    opts: &Opts,
    ctrl_c_rx: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<Vec<Message>, Box<dyn Error>> {
    let (prompts, conversation): (Vec<&Message>, Vec<&Message>) = messages
        .iter()
        .partition(|m| m.role == "system" && !is_summary(m));
    if conversation.is_empty() {
        return Err("Nothing to summarize.".into());
    }

    let pending_user = if keep_last_user {
        conversation
            .iter()
            .rev()
            .copied()
            .find(|m| m.role == "user")
    } else {
        None
    };

    // Earlier summaries stay in front; only the newer messages are ever
    // dropped when the transcript has to be shrunk.
    let (earlier, mut recent): (Vec<&Message>, Vec<&Message>) =
        conversation.into_iter().partition(|m| is_summary(m));

    let mut attempt = 0;
    let summary = loop {
        let transcript: Vec<Message> = earlier
            .iter()
            .chain(recent.iter())
            .map(|m| (*m).clone())
            .collect();
        match request_summary(&render_transcript(&transcript), opts, ctrl_c_rx) {
            Err(e)
                if e.downcast_ref::<ContextLengthError>().is_some()
                    && recent.len() > 1
                    && attempt < MAX_SHRINK_ATTEMPTS =>
            {
                attempt += 1;
                recent.drain(..recent.len() / 2);
            }
            result => break result?,
        }
    };

    let mut result: Vec<Message> = prompts.into_iter().cloned().collect();
    result.push(make_message(
        "system",
        format!("{}{}", SUMMARY_PREFIX, summary),
    ));
    if let Some(user) = pending_user {
        result.push((*user).clone());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{FunctionCall, ToolCall};

    fn dummy_opts() -> Opts {
        Opts {
            max_tokens: None,
            model: "dummy".to_string(),
            endpoint: String::new(),
            tool_choice: None,
            api_key: None,
            max_retries: None,
            retry_base_delay_secs: None,
            parameters: Default::default(),
        }
    }

    fn tool_call_message() -> Message {
        Message {
            role: "assistant".to_string(),
            content: None,
            tool_call_id: None,
            name: None,
            tool_calls: Some(vec![ToolCall {
                index: None,
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"a.rs"}"#.to_string(),
                },
            }]),
        }
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        assert_eq!(truncate("héllo", 10), "héllo");
        assert_eq!(truncate("héllo", 2), "hé... [3 more characters]");
    }

    #[test]
    fn transcript_shows_tool_calls_and_truncates_results() {
        let mut result = make_message("tool", "x".repeat(MAX_MESSAGE_CHARS + 50));
        result.name = Some("read_file".to_string());
        let transcript = render_transcript(&[
            make_message("user", "look at a.rs".to_string()),
            tool_call_message(),
            result,
        ]);
        assert!(transcript.contains("[user]\nlook at a.rs"));
        assert!(transcript.contains(r#"(called tool read_file with {"path":"a.rs"})"#));
        assert!(transcript.contains("[tool result: read_file]"));
        assert!(transcript.contains("... [50 more characters]"));
    }

    #[test]
    fn summary_replaces_conversation_but_keeps_prompts() {
        let messages = vec![
            make_message("system", "be helpful".to_string()),
            make_message("user", "first".to_string()),
            make_message("assistant", "answer".to_string()),
            make_message("user", "second".to_string()),
        ];
        let summarized = summarize_conversation(&messages, false, &dummy_opts(), &None).unwrap();
        assert_eq!(summarized.len(), 2);
        assert_eq!(summarized[0].content.as_deref(), Some("be helpful"));
        assert!(is_summary(&summarized[1]));
    }

    #[test]
    fn pending_user_message_is_kept_after_summary() {
        let messages = vec![
            make_message("user", "first".to_string()),
            make_message("assistant", "answer".to_string()),
            make_message("user", "second".to_string()),
        ];
        let summarized = summarize_conversation(&messages, true, &dummy_opts(), &None).unwrap();
        assert_eq!(summarized.len(), 2);
        assert!(is_summary(&summarized[0]));
        assert_eq!(summarized[1].role, "user");
        assert_eq!(summarized[1].content.as_deref(), Some("second"));
    }

    #[test]
    fn earlier_summary_is_folded_into_the_new_one() {
        let messages = vec![
            make_message("system", format!("{}old facts", SUMMARY_PREFIX)),
            make_message("user", "more".to_string()),
        ];
        let summarized = summarize_conversation(&messages, false, &dummy_opts(), &None).unwrap();
        assert_eq!(summarized.iter().filter(|m| is_summary(m)).count(), 1);
        assert!(
            !summarized[0]
                .content
                .as_deref()
                .unwrap()
                .contains("old facts")
        );
    }

    #[test]
    fn nothing_to_summarize() {
        let messages = vec![make_message("system", "be helpful".to_string())];
        assert!(summarize_conversation(&messages, false, &dummy_opts(), &None).is_err());
    }

    #[test]
    fn oversized_transcript_is_shrunk_until_it_fits() {
        // The dummy backend rejects requests above its context limit.
        let messages: Vec<Message> = (0..200)
            .map(|i| make_message("user", format!("{} {}", i, "y".repeat(MAX_MESSAGE_CHARS))))
            .collect();
        let summarized = summarize_conversation(&messages, false, &dummy_opts(), &None).unwrap();
        assert_eq!(summarized.len(), 1);
        assert!(is_summary(&summarized[0]));
    }
}
