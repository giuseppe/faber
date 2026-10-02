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

use std::any::Any;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, mpsc};

pub mod db;
pub mod db_backend;
pub mod dummy_llm;
pub mod github;
pub mod local_db;
pub mod mcp;
pub mod openai;
#[allow(dead_code)]
pub mod protocol;
pub mod scripted_llm;

/// Largest tool result, in characters, handed back to the model when the
/// context window is unknown (roughly 8k tokens).
pub const DEFAULT_MAX_TOOL_OUTPUT_CHARS: usize = 32_000;

/// Smallest cap `max_tool_output_chars_for_context` will ever pick, so a
/// tiny context window still leaves room for a useful result.
const MIN_MAX_TOOL_OUTPUT_CHARS: usize = 4_000;

/// Tool-result cap for a model with `context_window` tokens: about a
/// quarter of the window (at ~4 characters per token), so a single tool
/// result can never fill the context on its own.
pub fn max_tool_output_chars_for_context(context_window: u32) -> usize {
    (context_window as usize).max(MIN_MAX_TOOL_OUTPUT_CHARS)
}

/// What an agent reported as the outcome of its work, with the
/// `report_result` tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ReportedResult {
    pub succeeded: bool,
    pub summary: String,
    /// Structured data, shaped as `ToolContext::result_schema` asks.
    pub data: Option<serde_json::Value>,
}

impl ReportedResult {
    /// The result as text for whoever gets it: the summary, then the data.
    pub fn text(&self) -> String {
        match &self.data {
            Some(data) => format!(
                "{}\n```json\n{}\n```",
                self.summary.trim(),
                serde_json::to_string_pretty(data).unwrap_or_default()
            ),
            None => self.summary.trim().to_string(),
        }
    }
}

pub struct ToolContext {
    pub println: Box<dyn Fn(&str) + Send + Sync>,
    pub db: Option<Arc<dyn db_backend::DbBackend>>,
    pub agent_name: Option<String>,
    pub extra: Option<Arc<dyn Any + Send + Sync>>,
    pub mcp_manager: Option<Arc<mcp::McpManager>>,
    /// Set around a tool's own execution (see `openai::tool_call`'s caller)
    /// so a `println` closure that wants to visually box a tool's output
    /// can tell "this line came from inside a running tool" apart from any
    /// other use of the same closure. Ignored by callers that don't care.
    pub boxed: Arc<AtomicBool>,
    /// Cap on a tool result's size, in characters (see
    /// `max_tool_output_chars`); `None` uses the default.
    pub max_tool_output_chars: Option<usize>,
    /// The model's context window in tokens, when known: sizes tool
    /// results (unless `max_tool_output_chars` is set) and lets the tool
    /// loop trim old tool results before the conversation outgrows it.
    pub context_window: Option<u32>,
    /// Receives Ctrl-C while a request is running, for a long-running tool
    /// (e.g. `fan_out`) to notice it. Whoever takes the signal must stop
    /// and return `openai::InterruptedError`.
    pub interrupt: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
    /// Most requests to the model one call of the request loop may make -
    /// a budget for a sub-agent's work. `None`: no limit.
    pub max_requests: Option<usize>,
    /// Where `report_result` puts the outcome of this run of work (a
    /// sub-agent, fan-out worker or task), when someone's waiting for one.
    pub result_slot: Option<Arc<Mutex<Option<ReportedResult>>>>,
    /// The shape `report_result`'s data must have: a JSON object mapping
    /// each required field to its type ("string", "number", "integer",
    /// "boolean", "array" or "object").
    pub result_schema: Option<serde_json::Value>,
}

impl ToolContext {
    pub fn new<F>(println_fn: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        Self {
            println: Box::new(println_fn),
            db: None,
            agent_name: None,
            extra: None,
            mcp_manager: None,
            boxed: Arc::new(AtomicBool::new(false)),
            max_tool_output_chars: None,
            context_window: None,
            interrupt: None,
            max_requests: None,
            result_slot: None,
            result_schema: None,
        }
    }

    /// Largest tool result, in characters, to hand back to the model.
    pub fn max_tool_output_chars(&self) -> usize {
        self.max_tool_output_chars
            .or_else(|| self.context_window.map(max_tool_output_chars_for_context))
            .unwrap_or(DEFAULT_MAX_TOOL_OUTPUT_CHARS)
    }

    pub fn println(&self, msg: &str) {
        (self.println)(msg);
    }

    pub fn db(&self) -> Result<&dyn db_backend::DbBackend, Box<dyn std::error::Error>> {
        self.db.as_ref().map(|arc| arc.as_ref()).ok_or_else(|| {
            "Database not configured. Set 'db_path' in your config file."
                .to_string()
                .into()
        })
    }
}
