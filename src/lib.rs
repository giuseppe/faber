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
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, mpsc};

pub mod agent_io;
pub mod db;
pub mod db_backend;
pub mod dummy_llm;
pub mod github;
pub mod local_db;
pub mod mcp;
pub mod openai;
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
    /// The MCP servers' tools this agent gets (see `mcp::McpAccess`).
    pub mcp: Option<mcp::McpAccess>,
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
    /// What each file this agent read (or wrote) looked like then - so a
    /// write can be refused if someone else has changed the file since
    /// (see `check_file_unchanged`). Shared by the turns of one chat;
    /// `None` turns the check off (a scheduled tool call, with no model to
    /// read anything first).
    pub file_versions: Option<Arc<Mutex<HashMap<PathBuf, u64>>>>,
    /// Whether the agent these tools run for has the unsafe ones: what
    /// it may hand on to agents it makes (see `child_unsafe`).
    pub unsafe_tools: bool,
    /// The directory the agent works in - its own, if it has one (see
    /// `AgentConfig::cwd`), else the process's. Every tool goes by it.
    pub cwd: Option<PathBuf>,
    /// The task this work is for, if any: what it records is tagged with
    /// it, the agents it starts included, so a task shows all it did.
    pub task_id: Option<i64>,
    /// Called with the whole conversation after each round of tool calls,
    /// when it's whole - every call answered - to keep it somewhere: a
    /// task's run saves it, to be carried on if the run doesn't finish.
    /// Never handed on to the agents this one makes.
    pub checkpoint: Option<Arc<dyn Fn(&[openai::Message]) + Send + Sync>>,
}

/// A file's key in `ToolContext::file_versions`: absolute (relative to
/// `cwd`), `.` and `..` folded away.
fn file_key(cwd: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    let mut key = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                key.pop();
            }
            c => key.push(c),
        }
    }
    key
}

fn content_hash(content: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
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
            mcp: None,
            boxed: Arc::new(AtomicBool::new(false)),
            max_tool_output_chars: None,
            context_window: None,
            interrupt: None,
            max_requests: None,
            result_slot: None,
            result_schema: None,
            file_versions: Some(Arc::new(Mutex::new(HashMap::new()))),
            unsafe_tools: false,
            cwd: None,
            task_id: None,
            checkpoint: None,
        }
    }

    /// Hands `messages` to `checkpoint`, if there's one.
    pub fn checkpoint(&self, messages: &[openai::Message]) {
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint(messages);
        }
    }

    /// The directory this agent's tools work in.
    pub fn cwd(&self) -> PathBuf {
        match &self.cwd {
            Some(cwd) => cwd.clone(),
            None => std::env::current_dir().unwrap_or_default(),
        }
    }

    /// `path` as the tools take it: an absolute path inside the directory
    /// this agent works in becomes relative to it (models often give one);
    /// anything else is left as it is, for the tool to accept or refuse.
    pub fn tool_path(&self, path: &str) -> String {
        let given = Path::new(path);
        if !given.is_absolute() {
            return path.to_string();
        }
        let cwd = self.cwd();
        let bases = [cwd.clone(), cwd.canonicalize().unwrap_or(cwd)];
        for base in &bases {
            if let Ok(rest) = given.strip_prefix(base) {
                let rest = rest.to_string_lossy();
                return if rest.is_empty() {
                    ".".to_string()
                } else {
                    rest.into_owned()
                };
            }
        }
        path.to_string()
    }

    /// Remembers `content` as what this agent last saw of `path`.
    pub fn note_file_version(&self, path: &str, content: &[u8]) {
        if let Some(versions) = &self.file_versions {
            versions
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(file_key(&self.cwd(), path), content_hash(content));
        }
    }

    /// Before changing the existing file `path`, whose content is now
    /// `current`: refuses if this agent never read it, or if it changed
    /// since this agent last saw it - e.g. another agent, a command or the
    /// user edited it - so nobody's changes get silently overwritten.
    pub fn check_file_unchanged(&self, path: &str, current: &[u8]) -> Result<(), String> {
        let Some(versions) = &self.file_versions else {
            return Ok(());
        };
        let versions = versions.lock().unwrap_or_else(|e| e.into_inner());
        match versions.get(&file_key(&self.cwd(), path)) {
            None => Err(format!(
                "'{}' already exists and you haven't read it: read it first, so you don't \
                 overwrite something you haven't seen",
                path
            )),
            Some(seen) if *seen != content_hash(current) => Err(format!(
                "'{}' has changed since you read it (someone else edited it): read it again \
                 and redo your change on top of what's there now",
                path
            )),
            Some(_) => Ok(()),
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
