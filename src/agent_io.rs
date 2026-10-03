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

//! What agents do, as a stream of `AgentEvent`s: the one place a front
//! end - the terminal, `faber serve`'s web UI - learns what an agent is
//! up to, whichever process runs it (a chat, a worker, a sub-agent).
//!
//! The request loop reports through a `ResponseMode`; `observed` makes
//! one that also hands everything to an `AgentOutput`. `EventLog` is the
//! output that records events in the database, where anyone sharing it
//! can follow them (`DbBackend::agent_events`).

use crate::db_backend::DbBackend;
use crate::openai::{ProgressInfo, ResponseMode, StatusUpdate};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Most characters of a tool's result kept in its `ToolEnd` event.
pub const MAX_EVENT_TOOL_OUTPUT_CHARS: usize = 2000;

/// Most events kept per agent: older ones are dropped as new ones come.
pub const MAX_EVENTS_PER_AGENT: i64 = 2000;

/// Something an agent did.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// A turn starts on this: the user's message, a task, a parent
    /// agent's prompt.
    Input {
        text: String,
    },
    /// Part of the answer, as it streams in.
    Text {
        text: String,
    },
    /// Part of the model's reasoning, as it streams in.
    Reasoning {
        text: String,
    },
    ToolStart {
        name: String,
        arguments: String,
    },
    /// A tool call finished, with (the start of) its result.
    ToolEnd {
        name: String,
        duration_ms: u64,
        output: String,
        /// The tool failed (see `tool_failed`).
        #[serde(default)]
        failed: bool,
    },
    /// The turn is over: how it went, and its outcome or error.
    TurnEnd {
        succeeded: bool,
        text: String,
    },
}

/// An event as recorded: which agent, for which task if any, and when.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AgentEventRow {
    /// Increasing: ask for the ones after the last one seen to follow.
    pub id: i64,
    pub agent: String,
    pub task_id: Option<i64>,
    pub at: String,
    pub event: AgentEvent,
}

/// Which recorded events to read (`DbBackend::agent_events`).
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct EventFilter {
    /// Only this agent's.
    #[serde(default)]
    pub agent: Option<String>,
    /// Only those of this task's runs.
    #[serde(default)]
    pub task_id: Option<i64>,
    /// The first `limit` after this id, oldest first - or, without it,
    /// the latest `limit`, oldest first too.
    #[serde(default)]
    pub after: Option<i64>,
    pub limit: usize,
}

/// Whether a tool's result says it failed: tool_call's `error: ...`, a
/// JSON object with an `error` (how several tools report one, e.g.
/// read_file's missing file), or a command that never ran or finished. A
/// command that ran and exited non-zero isn't a failed tool call.
pub fn tool_failed(output: &str) -> bool {
    let output = output.trim_start();
    if output.starts_with("error: ") {
        return true;
    }
    if !output.starts_with('{') {
        return false;
    }
    let Ok(json) = serde_json::from_str::<serde_json::Value>(output) else {
        return false;
    };
    let has_error = json
        .get("error")
        .is_some_and(|e| !e.is_null() && e != &serde_json::Value::String(String::new()));
    // run_command's result for a command that never ran or didn't finish:
    // no exit code - it couldn't be started, or was killed.
    let never_finished = json.get("success") == Some(&serde_json::Value::Bool(false))
        && json.get("exit_code").is_some_and(|c| c.is_null());
    has_error || never_finished
}

/// Why a failed tool call failed: "error: tool 'x' failed: why" gives
/// "why"; a JSON result its `error`, or a command's `stderr`.
pub fn tool_error(output: &str) -> String {
    let text = output.trim();
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
        for field in ["error", "stderr"] {
            match json.get(field) {
                Some(serde_json::Value::String(s)) if !s.trim().is_empty() => return s.clone(),
                Some(v) if !v.is_null() && !v.is_string() => return v.to_string(),
                _ => {}
            }
        }
    }
    let text = text.strip_prefix("error: ").unwrap_or(text);
    match text.split_once(" failed: ") {
        Some((tool, why)) if tool.starts_with("tool '") || tool.starts_with("MCP tool '") => {
            why.to_string()
        }
        _ => text.to_string(),
    }
}

/// Where an agent's events go.
pub trait AgentOutput: Send + Sync {
    fn emit(&self, event: AgentEvent);
    /// Everything emitted so far should reach its destination now - a
    /// response finished streaming.
    fn flush(&self) {}
}

/// Hands events to each of several outputs.
pub struct Tee(pub Vec<Arc<dyn AgentOutput>>);

impl AgentOutput for Tee {
    fn emit(&self, event: AgentEvent) {
        for output in &self.0 {
            output.emit(event.clone());
        }
    }

    fn flush(&self) {
        for output in &self.0 {
            output.flush();
        }
    }
}

/// `mode`, with what it's told also going to `output` as events. A
/// `Complete` mode becomes a streaming one that shows nothing.
pub fn observed(mode: ResponseMode, output: Arc<dyn AgentOutput>) -> ResponseMode {
    let (stream, reasoning, progress) = match mode {
        ResponseMode::Streaming {
            stream_handler,
            reasoning_handler,
            progress_handler,
        } => (stream_handler, reasoning_handler, progress_handler),
        ResponseMode::Complete => (
            Box::new(|_: &str| Ok(())) as _,
            Box::new(|_: &str| Ok(())) as _,
            Box::new(|_: &ProgressInfo| Ok(())) as _,
        ),
    };
    let (for_stream, for_reasoning) = (output.clone(), output.clone());
    ResponseMode::Streaming {
        stream_handler: Box::new(move |chunk: &str| {
            match chunk {
                // The end of the response.
                "" => for_stream.flush(),
                text => for_stream.emit(AgentEvent::Text {
                    text: text.to_string(),
                }),
            }
            stream(chunk)
        }),
        reasoning_handler: Box::new(move |chunk: &str| {
            if !chunk.is_empty() {
                for_reasoning.emit(AgentEvent::Reasoning {
                    text: chunk.to_string(),
                });
            }
            reasoning(chunk)
        }),
        progress_handler: Box::new(move |info: &ProgressInfo| {
            match &info.status {
                StatusUpdate::ToolStart { name, arguments } => output.emit(AgentEvent::ToolStart {
                    name: name.clone(),
                    arguments: arguments.clone(),
                }),
                StatusUpdate::ToolComplete {
                    name,
                    duration_ms,
                    output: result,
                } => output.emit(AgentEvent::ToolEnd {
                    name: name.clone(),
                    duration_ms: *duration_ms,
                    output: crate::openai::truncate_tool_output(
                        result,
                        MAX_EVENT_TOOL_OUTPUT_CHARS,
                    ),
                    failed: tool_failed(result),
                }),
                StatusUpdate::Complete { .. } => output.flush(),
                _ => {}
            }
            progress(info)
        }),
    }
}

/// How long streamed text may wait to be recorded, so a slow stream still
/// shows up as it goes.
const FLUSH_INTERVAL: Duration = Duration::from_millis(500);

/// Streamed text that waits for this many bytes is recorded right away.
const FLUSH_BYTES: usize = 4096;

/// Text or reasoning not recorded yet: streamed deltas are joined into
/// one event, so a response isn't a row per token.
struct Pending {
    reasoning: bool,
    text: String,
    since: Instant,
}

/// Records an agent's events in the database (`DbBackend::agent_events`).
/// Failing to is never the agent's problem: it's only logged.
pub struct EventLog {
    db: Arc<dyn DbBackend>,
    agent: String,
    task_id: Option<i64>,
    pending: Mutex<Option<Pending>>,
}

impl EventLog {
    pub fn new(db: Arc<dyn DbBackend>, agent: &str, task_id: Option<i64>) -> Arc<Self> {
        Arc::new(Self {
            db,
            agent: agent.to_string(),
            task_id,
            pending: Mutex::new(None),
        })
    }

    fn write(&self, events: &[AgentEvent]) {
        if events.is_empty() {
            return;
        }
        if let Err(e) = self
            .db
            .append_agent_events(&self.agent, self.task_id, events)
        {
            log::debug!("Couldn't record events of agent '{}': {}", self.agent, e);
        }
    }

    fn pending_event(pending: Pending) -> AgentEvent {
        if pending.reasoning {
            AgentEvent::Reasoning { text: pending.text }
        } else {
            AgentEvent::Text { text: pending.text }
        }
    }

    fn take_pending(&self) -> Option<AgentEvent> {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .map(Self::pending_event)
    }
}

impl AgentOutput for EventLog {
    fn emit(&self, event: AgentEvent) {
        let (reasoning, text) = match event {
            AgentEvent::Text { text } => (false, text),
            AgentEvent::Reasoning { text } => (true, text),
            other => {
                let mut events: Vec<AgentEvent> = self.take_pending().into_iter().collect();
                events.push(other);
                self.write(&events);
                return;
            }
        };
        let ready = {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            let mut ready = Vec::new();
            if pending.as_ref().is_some_and(|p| p.reasoning != reasoning) {
                ready.extend(pending.take().map(Self::pending_event));
            }
            let current = pending.get_or_insert_with(|| Pending {
                reasoning,
                text: String::new(),
                since: Instant::now(),
            });
            current.text.push_str(&text);
            if current.text.len() >= FLUSH_BYTES || current.since.elapsed() >= FLUSH_INTERVAL {
                ready.extend(pending.take().map(Self::pending_event));
            }
            ready
        };
        self.write(&ready);
    }

    fn flush(&self) {
        let pending: Vec<AgentEvent> = self.take_pending().into_iter().collect();
        self.write(&pending);
    }
}

impl Drop for EventLog {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_db::LocalDb;

    fn test_db() -> Arc<dyn DbBackend> {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        crate::db::initialize_db(&conn).unwrap();
        crate::db::create_agent(&conn, "a", "").unwrap();
        Arc::new(LocalDb::new(Arc::new(Mutex::new(conn))))
    }

    fn all(db: &Arc<dyn DbBackend>) -> Vec<AgentEvent> {
        db.agent_events(&EventFilter {
            limit: 100,
            ..Default::default()
        })
        .unwrap()
        .into_iter()
        .map(|row| row.event)
        .collect()
    }

    fn text(text: &str) -> AgentEvent {
        AgentEvent::Text {
            text: text.to_string(),
        }
    }

    #[test]
    fn test_streamed_text_is_joined_into_one_event() {
        let db = test_db();
        let log = EventLog::new(db.clone(), "a", Some(3));
        log.emit(text("Hel"));
        log.emit(text("lo"));
        assert!(all(&db).is_empty(), "held back until flushed");
        log.emit(AgentEvent::Reasoning {
            text: "hm".to_string(),
        });
        log.emit(AgentEvent::ToolStart {
            name: "glob".to_string(),
            arguments: "{}".to_string(),
        });
        log.emit(text("bye"));
        drop(log);
        assert_eq!(
            all(&db),
            vec![
                text("Hello"),
                AgentEvent::Reasoning {
                    text: "hm".to_string()
                },
                AgentEvent::ToolStart {
                    name: "glob".to_string(),
                    arguments: "{}".to_string()
                },
                text("bye"),
            ]
        );
        let rows = db
            .agent_events(&EventFilter {
                task_id: Some(3),
                limit: 10,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|r| r.agent == "a"));
    }

    #[test]
    fn test_tool_failed() {
        assert!(tool_failed("error: tool 'agent_wait' failed: nope"));
        assert!(tool_failed(r#"{"content":null,"error":"File not found"}"#));
        assert!(!tool_failed(r#"{"content":"x","error":null}"#));
        assert!(!tool_failed(
            r#"{"stdout":"","exit_code":1,"success":false}"#
        ));
        assert!(tool_failed(
            r#"{"stdout":"","stderr":"Failed to launch the sandbox","exit_code":null,"success":false}"#
        ));
        assert!(!tool_failed("the word error: appears later"));
        assert!(!tool_failed("{not json"));
    }

    #[test]
    fn test_tool_error() {
        assert_eq!(tool_error("error: tool 'agent_wait' failed: nope"), "nope");
        assert_eq!(
            tool_error(r#"{"content":null,"error":"File not found"}"#),
            "File not found"
        );
        assert_eq!(
            tool_error(r#"{"stdout":"","stderr":"no bwrap","exit_code":null,"success":false}"#),
            "no bwrap"
        );
        assert_eq!(
            tool_error("error: invalid tool requested: 'x'"),
            "invalid tool requested: 'x'"
        );
    }

    #[test]
    fn test_long_text_is_recorded_as_it_streams() {
        let db = test_db();
        let log = EventLog::new(db.clone(), "a", None);
        log.emit(text(&"x".repeat(FLUSH_BYTES)));
        assert_eq!(all(&db).len(), 1);
    }

    #[test]
    fn test_observed_mode_reports_what_the_loop_does() {
        struct Collect(Mutex<Vec<AgentEvent>>);
        impl AgentOutput for Collect {
            fn emit(&self, event: AgentEvent) {
                self.0.lock().unwrap().push(event);
            }
        }
        let collected = Arc::new(Collect(Mutex::new(Vec::new())));
        let shown = Arc::new(Mutex::new(String::new()));
        let shown_by_mode = shown.clone();
        let mode = observed(
            ResponseMode::Streaming {
                stream_handler: Box::new(move |chunk: &str| {
                    shown_by_mode.lock().unwrap().push_str(chunk);
                    Ok(())
                }),
                reasoning_handler: Box::new(|_: &str| Ok(())),
                progress_handler: Box::new(|_: &ProgressInfo| Ok(())),
            },
            collected.clone(),
        );
        let ResponseMode::Streaming {
            stream_handler,
            progress_handler,
            ..
        } = &mode
        else {
            panic!("not streaming");
        };
        stream_handler("hi").unwrap();
        stream_handler("").unwrap();
        progress_handler(&ProgressInfo {
            status: StatusUpdate::ToolComplete {
                name: "glob".to_string(),
                duration_ms: 5,
                output: "y".repeat(MAX_EVENT_TOOL_OUTPUT_CHARS * 2),
            },
            elapsed_ms: 0,
        })
        .unwrap();
        assert_eq!(*shown.lock().unwrap(), "hi", "still shown as before");
        let events = collected.0.lock().unwrap();
        assert_eq!(events[0], text("hi"));
        let AgentEvent::ToolEnd { output, .. } = &events[1] else {
            panic!("{:?}", events[1]);
        };
        assert!(output.chars().count() < MAX_EVENT_TOOL_OUTPUT_CHARS * 2);
    }
}
