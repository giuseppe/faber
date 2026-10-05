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

//! `faber serve`'s A2A side: other agents' way in, over the
//! [A2A protocol](https://a2a-protocol.org/) (v0.3, JSON-RPC binding).
//! A translation layer over the database, next to the web API in `web.rs`
//! (which routes to it); the line protocol stays faber's own.
//!
//! - `GET /.well-known/agent-card.json`: the Agent Card, one skill per
//!   profile of the database, readable without a key.
//! - `POST /a2a`: JSON-RPC 2.0 - `message/send`, `tasks/get`,
//!   `tasks/cancel`.
//!
//! Callers authenticate with their own keys (`--a2a-keys-file`), which
//! work here and nowhere else; the server's own key doesn't work here. A
//! context (`contextId`) is an agent made from a profile, kept for the key
//! that opened it; each message is a task of that agent (see
//! `db::a2a_message_task`), run by whichever worker picks it up.

use crate::web::{EventStream, Request, Response};
use faber::agent_io::{AgentEvent, EventFilter};
use faber::db;
use rusqlite::Connection;
use serde_json::{Value, json};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A part (a message's text, a data part) over this many bytes is refused.
const MAX_PART_BYTES: usize = 512 << 10;

/// How long `message/send` with `blocking` waits for the task to finish.
const BLOCKING_WAIT: Duration = Duration::from_secs(60);

/// How often a task is looked at while waiting for it.
const POLL_EVERY: Duration = Duration::from_millis(250);

/// How often a stream looks for the task's new events.
const STREAM_POLL_EVERY: Duration = Duration::from_millis(200);

/// A stream with no new event for this long is closed, with nothing
/// final said: the client can resubscribe.
const STREAM_IDLE_LIMIT: Duration = Duration::from_secs(30 * 60);

/// How often a stream with nothing to say writes a comment, to notice
/// that its reader has gone.
const STREAM_KEEP_ALIVE: Duration = Duration::from_secs(15);

/// Most events one look at a task's events takes.
const STREAM_BATCH: usize = 100;

/// Appended to a prompt whose caller accepts only JSON.
const JSON_ONLY: &str =
    "Answer with JSON only: a single JSON object or array, no prose and no code fence.";

// JSON-RPC and A2A error codes.
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const TASK_NOT_FOUND: i64 = -32001;
const TASK_NOT_CANCELABLE: i64 = -32002;
const PUSH_NOT_SUPPORTED: i64 = -32003;
const CONTENT_TYPE_NOT_SUPPORTED: i64 = -32005;

/// The keys A2A callers authenticate with, by name.
#[derive(Debug, Default, Clone)]
pub struct A2aKeys {
    keys: Vec<(String, String)>,
}

impl A2aKeys {
    /// Reads the keys file's `text`: each line that isn't empty or a `#`
    /// comment is `NAME KEY`. Names are `[A-Za-z0-9_-]+` and unique, keys
    /// at least 32 characters and not the server's own (`server_key`).
    pub fn parse(text: &str, server_key: Option<&str>) -> Result<Self, String> {
        let mut keys: Vec<(String, String)> = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let at = |problem: &str| format!("line {}: {}", i + 1, problem);
            let mut words = line.split_whitespace();
            let (Some(name), Some(key), None) = (words.next(), words.next(), words.next()) else {
                return Err(at("expected NAME KEY"));
            };
            if !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return Err(at(&format!(
                    "name '{}': use letters, digits, '_' and '-'",
                    name
                )));
            }
            if keys.iter().any(|(n, _)| n == name) {
                return Err(at(&format!("name '{}' is used twice", name)));
            }
            if key.chars().count() < 32 {
                return Err(at(&format!("the key of '{}' is under 32 characters", name)));
            }
            if server_key.is_some_and(|server| crate::server::keys_match(key, server)) {
                return Err(at(&format!(
                    "the key of '{}' is the server's own: A2A callers need keys of their own",
                    name
                )));
            }
            keys.push((name.to_string(), key.to_string()));
        }
        Ok(Self { keys })
    }

    /// Reads the keys file `path`.
    pub fn read(path: &str, server_key: Option<&str>) -> Result<Self, Box<dyn Error>> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("can't read {}: {}", path, e))?;
        Ok(Self::parse(&text, server_key).map_err(|e| format!("{}: {}", path, e))?)
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The name of the caller whose key is `given`. Tries every key, in a
    /// time that doesn't depend on which, if any, matches.
    fn caller(&self, given: &str) -> Option<&str> {
        let mut found = None;
        for (name, key) in &self.keys {
            if crate::server::keys_match(given, key) {
                found = Some(name.as_str());
            }
        }
        found
    }
}

/// Answers `request` if it is for A2A, or `None` if it isn't. Without
/// keys configured, A2A isn't there at all: 404.
pub(crate) fn handle(
    request: &Request,
    db: &Arc<Mutex<Connection>>,
    keys: &A2aKeys,
) -> Option<Response> {
    let segments: Vec<&str> = request.path.split('/').filter(|s| !s.is_empty()).collect();
    let response = match segments.as_slice() {
        [".well-known", "agent-card.json"] => {
            if keys.is_empty() {
                Response::error(404, "not found")
            } else if request.method != "GET" {
                Response::error(405, "method not allowed")
            } else {
                agent_card(request, db)
            }
        }
        ["a2a"] => {
            if keys.is_empty() {
                Response::error(404, "not found")
            } else {
                rpc(request, db, keys)
            }
        }
        _ => return None,
    };
    Some(response)
}

/// A profile of the database: the only kind that's offered as a skill.
struct Skill {
    name: String,
    profile: crate::Profile,
}

/// The database's profiles that a worker can make an agent from.
fn skills(conn: &Connection) -> Result<Vec<Skill>, Box<dyn Error>> {
    Ok(db::list_profiles(conn)?
        .into_iter()
        .filter_map(|(name, settings)| {
            let profile = serde_json::from_value(settings).ok()?;
            Some(Skill { name, profile })
        })
        .collect())
}

fn agent_card(request: &Request, db: &Arc<Mutex<Connection>>) -> Response {
    let skills = match db.lock() {
        Ok(conn) => skills(&conn),
        Err(e) => return Response::error(500, &format!("DB lock: {}", e)),
    };
    let skills = match skills {
        Ok(skills) => skills,
        Err(e) => return Response::error(500, &e.to_string()),
    };
    let base = format!("http://{}", request.host.as_deref().unwrap_or("localhost"));
    Response::json(
        200,
        &json!({
            "protocolVersion": "0.3.0",
            "name": "faber",
            "description": "faber agents, one skill per profile.",
            "url": format!("{}/a2a", base),
            "preferredTransport": "JSONRPC",
            "version": env!("CARGO_PKG_VERSION"),
            "capabilities": {"streaming": true, "pushNotifications": false},
            "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
            "security": [{"bearer": []}],
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["text/plain", "application/json"],
            "skills": skills.iter().map(|skill| json!({
                "id": skill.name,
                "name": skill.name,
                "description": skill
                    .profile
                    .description
                    .clone()
                    .unwrap_or_else(|| format!("Profile {}", skill.name)),
                "tags": ["profile"],
            })).collect::<Vec<_>>(),
        }),
    )
}

/// A JSON-RPC error.
#[derive(Debug)]
struct RpcError {
    code: i64,
    message: String,
}

fn fail<T>(code: i64, message: impl Into<String>) -> Result<T, RpcError> {
    Err(RpcError {
        code,
        message: message.into(),
    })
}

impl From<Box<dyn Error>> for RpcError {
    fn from(e: Box<dyn Error>) -> Self {
        RpcError {
            code: -32603,
            message: format!("internal error: {}", e),
        }
    }
}

fn rpc_response(id: &Value, result: Result<Value, RpcError>) -> Response {
    Response::json(
        200,
        &match result {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err(e) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": e.code, "message": e.message},
            }),
        },
    )
}

/// A JSON-RPC request, checked: who sent it, its id, method and params.
struct RpcCall {
    caller: String,
    id: Value,
    method: String,
    params: Value,
}

/// Reads `POST /a2a`: the caller's key, then the JSON-RPC request. `Err`
/// is the response to send for one that is none.
fn parse_rpc(request: &Request, keys: &A2aKeys) -> Result<RpcCall, Response> {
    let given = request.authorization.as_deref().unwrap_or("");
    let given = given.strip_prefix("Bearer ").unwrap_or("");
    let Some(caller) = keys.caller(given) else {
        return Err(Response::error(401, "unauthorized"));
    };
    if request.method != "POST" {
        return Err(Response::error(405, "method not allowed"));
    }
    if !request
        .content_type
        .as_deref()
        .is_some_and(|t| t.starts_with("application/json"))
    {
        return Err(Response::error(415, "send Content-Type: application/json"));
    }
    let body: Value = match serde_json::from_slice(&request.body) {
        Ok(body) => body,
        Err(e) => {
            return Err(rpc_response(
                &Value::Null,
                fail(PARSE_ERROR, format!("not JSON: {}", e)),
            ));
        }
    };
    if body.is_array() {
        return Err(rpc_response(
            &Value::Null,
            fail(INVALID_REQUEST, "batches aren't supported"),
        ));
    }
    match (
        body.get("jsonrpc").and_then(Value::as_str),
        body.get("method").and_then(Value::as_str),
        body.get("id"),
    ) {
        (Some("2.0"), Some(method), Some(id @ (Value::String(_) | Value::Number(_)))) => {
            Ok(RpcCall {
                caller: caller.to_string(),
                id: id.clone(),
                method: method.to_string(),
                params: body.get("params").cloned().unwrap_or(Value::Null),
            })
        }
        _ => Err(rpc_response(
            &Value::Null,
            fail(INVALID_REQUEST, "not a JSON-RPC 2.0 request with an id"),
        )),
    }
}

/// `POST /a2a`.
fn rpc(request: &Request, db: &Arc<Mutex<Connection>>, keys: &A2aKeys) -> Response {
    match parse_rpc(request, keys) {
        Ok(rpc) => rpc_response(&rpc.id, call(&rpc.method, &rpc.params, &rpc.caller, db)),
        Err(response) => response,
    }
}

fn call(
    method: &str,
    params: &Value,
    caller: &str,
    db: &Arc<Mutex<Connection>>,
) -> Result<Value, RpcError> {
    match method {
        "message/send" => message_send(params, caller, db),
        "tasks/get" => tasks_get(params, caller, db),
        "tasks/cancel" => tasks_cancel(params, caller, db),
        m if m.starts_with("tasks/pushNotificationConfig/") => {
            fail(PUSH_NOT_SUPPORTED, "push notifications aren't supported")
        }
        _ => fail(METHOD_NOT_FOUND, format!("no method '{}'", method)),
    }
}

fn lock(db: &Arc<Mutex<Connection>>) -> Result<std::sync::MutexGuard<'_, Connection>, RpcError> {
    db.lock()
        .or_else(|e| fail(-32603, format!("DB lock: {}", e)))
}

/// What a message's parts say, as a prompt (see `prompt_from_parts`).
struct Prompt {
    text: String,
}

/// Which of the output modes a caller accepts we produce.
#[derive(Debug, PartialEq)]
enum Modes {
    /// Text, or no preference: as the agent answers.
    Any,
    /// JSON and not text: the agent is asked for JSON only.
    JsonOnly,
}

/// `configuration.acceptedOutputModes`: what we can produce of it, or
/// `-32005` for a list with nothing we produce.
fn accepted_modes(configuration: &Value) -> Result<Modes, RpcError> {
    let Some(list) = configuration
        .get("acceptedOutputModes")
        .and_then(Value::as_array)
        .filter(|l| !l.is_empty())
    else {
        return Ok(Modes::Any);
    };
    let accepts = |wanted: &[&str]| {
        list.iter()
            .filter_map(Value::as_str)
            .any(|m| wanted.contains(&m.trim().to_ascii_lowercase().as_str()))
    };
    let text = accepts(&["text/plain", "text/*", "*/*"]);
    let json = accepts(&["application/json", "application/*", "*/*"]);
    match (text, json) {
        (false, false) => fail(
            CONTENT_TYPE_NOT_SUPPORTED,
            "acceptedOutputModes has nothing this server produces: text/plain, application/json",
        ),
        (false, true) => Ok(Modes::JsonOnly),
        _ => Ok(Modes::Any),
    }
}

/// The prompt a message's parts make, in order: a `text` part's text as
/// is, a `data` part a fenced block, preceded by a line naming it if it
/// has `metadata.name`; joined by blank lines.
fn prompt_from_parts(message: &Value, modes: &Modes) -> Result<Prompt, RpcError> {
    let Some(parts) = message
        .get("parts")
        .and_then(Value::as_array)
        .filter(|p| !p.is_empty())
    else {
        return fail(INVALID_PARAMS, "the message has no parts");
    };
    let mut pieces = Vec::new();
    for part in parts {
        let kind = part.get("kind").and_then(Value::as_str).or_else(|| {
            ["text", "data", "file"]
                .into_iter()
                .find(|k| part.get(k).is_some())
        });
        match kind {
            Some("text") => {
                let Some(text) = part.get("text").and_then(Value::as_str) else {
                    return fail(INVALID_PARAMS, "a text part needs a string 'text'");
                };
                if text.len() > MAX_PART_BYTES {
                    return fail(INVALID_PARAMS, "a text part is too large");
                }
                pieces.push(text.to_string());
            }
            Some("data") => {
                let Some(data) = part.get("data") else {
                    return fail(INVALID_PARAMS, "a data part needs 'data'");
                };
                let data = data.to_string();
                if data.len() > MAX_PART_BYTES {
                    return fail(INVALID_PARAMS, "a data part is too large");
                }
                let name = part
                    .get("metadata")
                    .and_then(|m| m.get("name"))
                    .and_then(Value::as_str);
                let mut piece = String::new();
                if let Some(name) = name {
                    piece.push_str(&format!("Data ({}):\n", name));
                }
                piece.push_str(&format!("```json\n{}\n```", data));
                pieces.push(piece);
            }
            Some("file") => {
                return fail(CONTENT_TYPE_NOT_SUPPORTED, "file parts aren't supported");
            }
            _ => return fail(INVALID_PARAMS, "a part is neither text, data nor file"),
        }
    }
    let mut text = pieces.join("\n\n");
    if text.trim().is_empty() {
        return fail(INVALID_PARAMS, "the message is empty");
    }
    if *modes == Modes::JsonOnly {
        text.push_str("\n\n");
        text.push_str(JSON_ONLY);
    }
    Ok(Prompt { text })
}

/// Task `id` as the database knows it, with its context, if both are
/// `caller`'s - else `-32001`, as for one that doesn't exist.
fn owned_task(
    conn: &Connection,
    caller: &str,
    id: &Value,
) -> Result<(db::TaskRow, db::A2aContext), RpcError> {
    let missing = || RpcError {
        code: TASK_NOT_FOUND,
        message: "no such task".to_string(),
    };
    let id: i64 = id
        .as_str()
        .and_then(|s| s.parse().ok())
        .ok_or_else(missing)?;
    let task = db::get_task(conn, id)?.ok_or_else(missing)?;
    let context = task
        .continue_agent
        .as_deref()
        .and_then(|agent| {
            db::get_agent_data(conn, agent, db::A2A_CONTEXT_KEY)
                .ok()
                .flatten()
        })
        .and_then(|context| db::a2a_get_context(conn, &context).ok().flatten())
        .filter(|context| context.owner == caller)
        .ok_or_else(missing)?;
    Ok((task, context))
}

/// The A2A state of a task.
fn task_state(task: &db::TaskRow) -> &'static str {
    match (task.status.as_str(), task.last_outcome.as_deref()) {
        (db::TaskStatus::SCHEDULED | db::TaskStatus::HELD, _) => "submitted",
        (db::TaskStatus::RUNNING, _) => "working",
        (db::TaskStatus::DONE, Some("succeeded")) => "completed",
        (db::TaskStatus::DONE, Some("failed")) if was_stopped(task) => "canceled",
        (db::TaskStatus::DONE, Some("failed")) => "failed",
        _ => "unknown",
    }
}

/// Whether a failed task failed by being stopped (see
/// `run_prompt_task_headless` and `db::request_task_stop`).
fn was_stopped(task: &db::TaskRow) -> bool {
    task.last_result
        .as_deref()
        .is_some_and(|r| r == "stopped" || r.starts_with("stopped ("))
}

/// The latest of when a task started and last ran.
fn task_timestamp(task: &db::TaskRow) -> String {
    [&task.started_at, &task.last_run_at]
        .into_iter()
        .flatten()
        .filter_map(|t| db::parse_db_time(t))
        .max()
        .or_else(|| db::parse_db_time(&task.created_at))
        .unwrap_or_else(chrono::Utc::now)
        .to_rfc3339()
}

fn text_message(role: &str, id: String, text: &str, task: i64, context: &str) -> Value {
    json!({
        "kind": "message",
        "role": role,
        "messageId": id,
        "parts": [{"kind": "text", "text": text}],
        "taskId": task.to_string(),
        "contextId": context,
    })
}

/// The part an answer is: its JSON, if it is that (an object or array,
/// alone or as the one `json` fenced block), else its text.
fn answer_part(answer: &str) -> Value {
    let answer = answer.trim();
    let structured = |text: &str| {
        serde_json::from_str::<Value>(text)
            .ok()
            .filter(|v| v.is_object() || v.is_array())
    };
    if let Some(value) = structured(answer) {
        return json!({"kind": "data", "data": value});
    }
    if let Some(inner) = answer
        .strip_prefix("```json")
        .and_then(|rest| rest.strip_suffix("```"))
        .filter(|inner| !inner.contains("```"))
    {
        if let Ok(value) = serde_json::from_str::<Value>(inner.trim()) {
            return json!({"kind": "data", "data": value});
        }
    }
    json!({"kind": "text", "text": answer})
}

/// The whole answer of a done task: the final message its worker kept,
/// else the (cut short) result.
fn task_answer(conn: &Connection, task: &db::TaskRow) -> String {
    task.continue_agent
        .as_deref()
        .and_then(|agent| {
            db::get_agent_data(conn, agent, &db::a2a_answer_key(task.id))
                .ok()
                .flatten()
        })
        .or_else(|| task.last_result.clone())
        .unwrap_or_default()
}

/// A task as an A2A `Task`.
fn task_json(
    conn: &Connection,
    task: &db::TaskRow,
    context: &db::A2aContext,
    history_length: Option<usize>,
) -> Value {
    let state = task_state(task);
    let mut status = json!({"state": state, "timestamp": task_timestamp(task)});
    if state == "failed" {
        status["message"] = text_message(
            "agent",
            format!("{}-status", task.id),
            task.last_result.as_deref().unwrap_or("failed"),
            task.id,
            &context.id,
        );
    }
    let mut history = Vec::new();
    if let Some(sent) = db::get_agent_data(conn, &context.agent, &db::a2a_message_key(task.id))
        .ok()
        .flatten()
        .and_then(|m| serde_json::from_str::<Value>(&m).ok())
    {
        history.push(sent);
    }
    let mut artifacts = Vec::new();
    if state == "completed" {
        let answer = task_answer(conn, task);
        history.push(text_message(
            "agent",
            format!("{}-answer", task.id),
            &answer,
            task.id,
            &context.id,
        ));
        artifacts.push(json!({
            "artifactId": "result",
            "name": "result",
            "parts": [answer_part(&answer)],
        }));
    }
    if let Some(n) = history_length {
        let skip = history.len().saturating_sub(n);
        history.drain(..skip);
    }
    json!({
        "kind": "task",
        "id": task.id.to_string(),
        "contextId": context.id,
        "status": status,
        "history": history,
        "artifacts": artifacts,
    })
}

/// `historyLength` of a request's params or configuration.
fn history_length(value: &Value) -> Option<usize> {
    value
        .get("historyLength")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
}

/// Takes a message for its context's agent - a new context, if it names
/// none - as a task. Returns the task's id, its context, and the request's
/// `configuration`.
fn submit_message(
    params: &Value,
    caller: &str,
    db: &Arc<Mutex<Connection>>,
) -> Result<(i64, db::A2aContext, Value), RpcError> {
    let Some(message) = params.get("message").filter(|m| m.is_object()) else {
        return fail(INVALID_PARAMS, "missing 'message'");
    };
    if message.get("role").and_then(Value::as_str) != Some("user") {
        return fail(INVALID_PARAMS, "the message's role must be 'user'");
    }
    let configuration = params.get("configuration").cloned().unwrap_or(Value::Null);
    let modes = accepted_modes(&configuration)?;
    let prompt = prompt_from_parts(message, &modes)?;
    let skill = message
        .get("metadata")
        .and_then(|m| m.get("skill"))
        .and_then(Value::as_str);
    let given_context = message.get("contextId").and_then(Value::as_str);
    let given_task = message.get("taskId").and_then(Value::as_str);

    let (task_id, context) = {
        let conn = lock(db)?;
        let context = match (given_task, given_context) {
            (Some(task_id), given) => {
                let (task, context) = owned_task(&conn, caller, &json!(task_id))?;
                if task.status != db::TaskStatus::DONE {
                    return fail(
                        INVALID_PARAMS,
                        "task is still running; send a new message in its context",
                    );
                }
                if given.is_some_and(|c| c != context.id) {
                    return fail(INVALID_PARAMS, "the task isn't in that context");
                }
                check_skill(skill, &context)?;
                context
            }
            (None, Some(context_id)) => {
                let context = db::a2a_get_context(&conn, context_id)?
                    .filter(|c| c.owner == caller)
                    .ok_or_else(|| RpcError {
                        code: INVALID_PARAMS,
                        message: "unknown contextId".to_string(),
                    })?;
                check_skill(skill, &context)?;
                context
            }
            (None, None) => {
                let mut skills = skills(&conn)?;
                let chosen = match skill {
                    Some(name) => {
                        skills
                            .into_iter()
                            .find(|s| s.name == name)
                            .ok_or_else(|| RpcError {
                                code: INVALID_PARAMS,
                                message: format!("no skill '{}'", name),
                            })?
                    }
                    None if skills.len() == 1 => skills.remove(0),
                    None if skills.is_empty() => {
                        return fail(INVALID_PARAMS, "this server offers no skills");
                    }
                    None => return fail(INVALID_PARAMS, "say which skill in metadata.skill"),
                };
                db::a2a_create_context(
                    &conn,
                    caller,
                    &chosen.name,
                    &chosen.profile.a2a_agent_config(&chosen.name),
                )?
            }
        };
        let task = db::a2a_message_task(&context, &prompt.text, chrono::Utc::now());
        let task_id = db::create_task(&conn, &task)?;
        let mut sent = message.clone();
        sent["kind"] = json!("message");
        sent["taskId"] = json!(task_id.to_string());
        sent["contextId"] = json!(context.id);
        db::set_agent_data(
            &conn,
            &context.agent,
            &db::a2a_message_key(task_id),
            &sent.to_string(),
        )?;
        db::a2a_touch_context(&conn, &context.id)?;
        (task_id, context)
    };
    Ok((task_id, context, configuration))
}

fn message_send(
    params: &Value,
    caller: &str,
    db: &Arc<Mutex<Connection>>,
) -> Result<Value, RpcError> {
    let (task_id, context, configuration) = submit_message(params, caller, db)?;
    let blocking = configuration
        .get("blocking")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let started = Instant::now();
    loop {
        let conn = lock(db)?;
        let task = db::get_task(&conn, task_id)?.ok_or_else(|| RpcError {
            code: TASK_NOT_FOUND,
            message: "the task went away".to_string(),
        })?;
        if !blocking || task.status == db::TaskStatus::DONE || started.elapsed() >= BLOCKING_WAIT {
            return Ok(task_json(
                &conn,
                &task,
                &context,
                history_length(&configuration),
            ));
        }
        drop(conn);
        std::thread::sleep(POLL_EVERY);
    }
}

/// A `metadata.skill` that names a different profile than the context's
/// is refused.
fn check_skill(skill: Option<&str>, context: &db::A2aContext) -> Result<(), RpcError> {
    match skill {
        Some(skill) if skill != context.profile => fail(
            INVALID_PARAMS,
            format!(
                "that context is of skill '{}', not '{}'",
                context.profile, skill
            ),
        ),
        _ => Ok(()),
    }
}

fn tasks_get(params: &Value, caller: &str, db: &Arc<Mutex<Connection>>) -> Result<Value, RpcError> {
    let conn = lock(db)?;
    let (task, context) = owned_task(&conn, caller, params.get("id").unwrap_or(&Value::Null))?;
    Ok(task_json(&conn, &task, &context, history_length(params)))
}

fn tasks_cancel(
    params: &Value,
    caller: &str,
    db: &Arc<Mutex<Connection>>,
) -> Result<Value, RpcError> {
    let conn = lock(db)?;
    let (task, context) = owned_task(&conn, caller, params.get("id").unwrap_or(&Value::Null))?;
    if task.status == db::TaskStatus::DONE {
        return fail(TASK_NOT_CANCELABLE, "the task is done");
    }
    // Running: the worker stops it. Not yet: nobody will.
    if !db::request_task_stop(&conn, task.id)? {
        db::cancel_waiting_task(&conn, task.id)?;
    }
    let task = db::get_task(&conn, task.id)?.unwrap_or(task);
    Ok(task_json(&conn, &task, &context, None))
}

/// A `message/stream` or `tasks/resubscribe` request, checked and ready
/// to run: a response that goes on for as long as the task does.
pub(crate) struct Stream {
    id: Value,
    db: Arc<Mutex<Connection>>,
    task_id: i64,
    context: db::A2aContext,
    /// Carrying on from where the task is, not from its start.
    resume: bool,
}

/// Prepares `request` as a stream if it is one - `None` if it isn't, for
/// the usual answer. `Err` is the answer to one that can't be streamed (a
/// bad request, a task that isn't the caller's).
pub(crate) fn streaming(
    request: &Request,
    db: &Arc<Mutex<Connection>>,
    keys: &A2aKeys,
) -> Option<Result<Stream, Response>> {
    let path = request.path.trim_matches('/');
    if keys.is_empty() || path != "a2a" {
        return None;
    }
    let rpc = parse_rpc(request, keys).ok()?;
    let resume = match rpc.method.as_str() {
        "message/stream" => false,
        "tasks/resubscribe" => true,
        _ => return None,
    };
    let started = if resume {
        lock(db).and_then(|conn| {
            owned_task(
                &conn,
                &rpc.caller,
                rpc.params.get("id").unwrap_or(&Value::Null),
            )
            .map(|(task, context)| (task.id, context))
        })
    } else {
        submit_message(&rpc.params, &rpc.caller, db).map(|(task_id, context, _)| (task_id, context))
    };
    Some(match started {
        Ok((task_id, context)) => Ok(Stream {
            id: rpc.id,
            db: db.clone(),
            task_id,
            context,
            resume,
        }),
        Err(e) => Err(rpc_response(&rpc.id, Err(e))),
    })
}

impl Stream {
    fn send(&self, out: &mut EventStream, result: Value) -> std::io::Result<()> {
        out.event(&json!({"jsonrpc": "2.0", "id": self.id, "result": result}).to_string())
    }

    fn status_update(&self, state: &str, message: Option<Value>, last: bool) -> Value {
        let mut status = json!({"state": state, "timestamp": chrono::Utc::now().to_rfc3339()});
        if let Some(message) = message {
            status["message"] = message;
        }
        json!({
            "kind": "status-update",
            "taskId": self.task_id.to_string(),
            "contextId": self.context.id,
            "status": status,
            "final": last,
        })
    }

    fn artifact_update(&self, text: &str, append: bool, last_chunk: bool) -> Value {
        self.artifact_part_update(json!({"kind": "text", "text": text}), append, last_chunk)
    }

    fn artifact_part_update(&self, part: Value, append: bool, last_chunk: bool) -> Value {
        json!({
            "kind": "artifact-update",
            "taskId": self.task_id.to_string(),
            "contextId": self.context.id,
            "artifact": {"artifactId": "result", "name": "result", "parts": [part]},
            "append": append,
            "lastChunk": last_chunk,
        })
    }

    /// Streams the task: its state now, then what it does, until it's
    /// done. The events come from the database, so any worker's run shows.
    pub(crate) fn run(self, out: &mut EventStream) -> std::io::Result<()> {
        let broken = |e: RpcError| std::io::Error::other(e.message);
        let (task, mut last) = {
            let conn = lock(&self.db).map_err(broken)?;
            let task = db::get_task(&conn, self.task_id)
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .ok_or_else(|| std::io::Error::other("the task went away"))?;
            self.send(out, task_json(&conn, &task, &self.context, None))?;
            // Resubscribed, from now on.
            let last = if self.resume {
                self.events_after(&conn, None, 1)
                    .last()
                    .map_or(0, |event| event.id)
            } else {
                0
            };
            (task, last)
        };
        if task.status == db::TaskStatus::DONE && self.resume {
            return Ok(());
        }
        let mut announced = false;
        let mut text_sent = false;
        let mut quiet_since = Instant::now();
        let mut last_write = Instant::now();
        loop {
            // Done before the events are read: all that it did is in them.
            let (task, events) = {
                let conn = lock(&self.db).map_err(broken)?;
                let task = db::get_task(&conn, self.task_id)
                    .map_err(|e| std::io::Error::other(e.to_string()))?
                    .ok_or_else(|| std::io::Error::other("the task went away"))?;
                (task, self.events_after(&conn, Some(last), STREAM_BATCH))
            };
            if !events.is_empty() {
                quiet_since = Instant::now();
                last_write = Instant::now();
            }
            let more = events.len() >= STREAM_BATCH;
            for row in events {
                last = row.id;
                if !announced {
                    announced = true;
                    self.send(out, self.status_update("working", None, false))?;
                }
                match row.event {
                    AgentEvent::Text { text } => {
                        self.send(out, self.artifact_update(&text, text_sent, false))?;
                        text_sent = true;
                    }
                    AgentEvent::ToolStart { name, .. } => {
                        let message = text_message(
                            "agent",
                            format!("{}-tool-{}", self.task_id, row.id),
                            &format!("Running {}", name),
                            self.task_id,
                            &self.context.id,
                        );
                        self.send(out, self.status_update("working", Some(message), false))?;
                    }
                    // What the caller sent, the model's reasoning, a
                    // tool's result and the turn's end (the task's own
                    // state says how it went) aren't shown.
                    AgentEvent::Input { .. }
                    | AgentEvent::Reasoning { .. }
                    | AgentEvent::ToolEnd { .. }
                    | AgentEvent::TurnEnd { .. } => {}
                }
            }
            if more {
                continue;
            }
            if task.status == db::TaskStatus::DONE {
                return self.finish(out, &task, text_sent);
            }
            if quiet_since.elapsed() >= STREAM_IDLE_LIMIT {
                return Ok(());
            }
            if last_write.elapsed() >= STREAM_KEEP_ALIVE {
                out.keep_alive()?;
                last_write = Instant::now();
            }
            std::thread::sleep(STREAM_POLL_EVERY);
        }
    }

    /// The task's events after `after` (the latest `limit`, without it).
    fn events_after(
        &self,
        conn: &Connection,
        after: Option<i64>,
        limit: usize,
    ) -> Vec<faber::agent_io::AgentEventRow> {
        db::agent_events(
            conn,
            &EventFilter {
                agent: Some(self.context.agent.clone()),
                task_id: Some(self.task_id),
                after,
                limit,
            },
        )
        .unwrap_or_default()
    }

    /// The end: the result's last chunk, then the final state.
    fn finish(
        &self,
        out: &mut EventStream,
        task: &db::TaskRow,
        text_sent: bool,
    ) -> std::io::Result<()> {
        let state = task_state(task);
        if state == "completed" {
            if text_sent && !self.resume {
                // Nothing more to add: a chunk with no parts, which a client
                // appending parts to the artifact doesn't turn into an empty one.
                let mut last = self.artifact_update("", true, true);
                last["artifact"]["parts"] = json!([]);
                self.send(out, last)?;
            } else {
                // Nothing streamed (the answer was reported, not said), or
                // only its tail (resubscribed): all of it at once.
                let answer = {
                    let conn = lock(&self.db).map_err(|e| std::io::Error::other(e.message))?;
                    task_answer(&conn, task)
                };
                self.send(
                    out,
                    self.artifact_part_update(answer_part(&answer), false, true),
                )?;
            }
        }
        let message = (state == "failed").then(|| {
            text_message(
                "agent",
                format!("{}-status", task.id),
                task.last_result.as_deref().unwrap_or("failed"),
                task.id,
                &self.context.id,
            )
        });
        self.send(out, self.status_update(state, message, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web::read_request;

    const ALICE: &str = "alice-key-0123456789abcdef0123456789abcdef";
    const BOB: &str = "bob-key-0123456789abcdef0123456789abcdef";
    const SERVER: &str = "server-key-0123456789abcdef0123456789abcdef";

    fn keys() -> A2aKeys {
        A2aKeys::parse(
            &format!("# callers\nalice {}\n\nbob {}\n", ALICE, BOB),
            Some(SERVER),
        )
        .unwrap()
    }

    fn test_db() -> Arc<Mutex<Connection>> {
        let conn = Connection::open_in_memory().unwrap();
        db::initialize_db(&conn).unwrap();
        Arc::new(Mutex::new(conn))
    }

    /// Writes a model script (see `scripted_llm`) and returns its `--model`.
    fn model_script(name: &str, rules: Value) -> String {
        let path = std::env::temp_dir().join(format!(
            "faber_a2a_script_{}_{}.json",
            name,
            std::process::id()
        ));
        std::fs::write(&path, rules.to_string()).unwrap();
        format!("script:{}", path.display())
    }

    fn add_profile(db: &Arc<Mutex<Connection>>, name: &str, settings: Value) {
        db::set_profile(&db.lock().unwrap(), name, &settings).unwrap();
    }

    fn http(method: &str, path: &str, key: Option<&str>, body: &str) -> Request {
        let auth = key
            .map(|k| format!("Authorization: Bearer {}\r\n", k))
            .unwrap_or_default();
        let raw = format!(
            "{} {} HTTP/1.1\r\nHost: faber.test:9090\r\n{}Content-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{}",
            method,
            path,
            auth,
            body.len(),
            body
        );
        read_request(&mut raw.as_bytes()).unwrap()
    }

    fn answer(db: &Arc<Mutex<Connection>>, key: Option<&str>, body: &str) -> (u16, Value) {
        let request = http("POST", "/a2a", key, body);
        let response = handle(&request, db, &keys()).unwrap();
        (
            response.status,
            serde_json::from_slice(&response.body).unwrap_or(Value::Null),
        )
    }

    /// A JSON-RPC call by `key`: its `result`, or its error's code.
    fn rpc_as(
        db: &Arc<Mutex<Connection>>,
        key: &str,
        method: &str,
        params: Value,
    ) -> Result<Value, i64> {
        let body = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
        let (status, reply) = answer(db, Some(key), &body.to_string());
        assert_eq!(status, 200, "{reply}");
        assert_eq!(reply["id"], 1);
        match reply.get("error") {
            Some(e) => Err(e["code"].as_i64().unwrap()),
            None => Ok(reply["result"].clone()),
        }
    }

    fn send(db: &Arc<Mutex<Connection>>, key: &str, message: Value) -> Result<Value, i64> {
        rpc_as(db, key, "message/send", json!({"message": message}))
    }

    fn user(text: &str) -> Value {
        json!({
            "kind": "message", "role": "user", "messageId": "m1",
            "parts": [{"kind": "text", "text": text}],
        })
    }

    #[test]
    fn test_keys_file_is_checked() {
        let good = "# a comment\nalice-1 0123456789abcdef0123456789abcdef\n";
        assert!(A2aKeys::parse(good, None).is_ok());
        assert!(A2aKeys::parse("", None).unwrap().is_empty());
        let long = "0123456789abcdef0123456789abcdef";
        for (bad, why) in [
            ("alice", "NAME KEY"),
            (&format!("alice {} extra", long), "NAME KEY"),
            (&format!("al ice {}", long), "NAME KEY"),
            (&format!("al.ice {}", long), "name 'al.ice'"),
            ("alice short", "under 32"),
            (&format!("a {long}\na {long}x"), "used twice"),
            (&format!("a {SERVER}"), "server's own"),
        ] {
            let err = A2aKeys::parse(bad, Some(SERVER)).unwrap_err();
            assert!(err.contains(why), "{bad:?}: {err}");
        }
        let keys = keys();
        assert_eq!(keys.caller(ALICE), Some("alice"));
        assert_eq!(keys.caller(BOB), Some("bob"));
        assert_eq!(keys.caller(SERVER), None);
        assert_eq!(keys.caller(""), None);
    }

    #[test]
    fn test_nothing_is_there_without_keys() {
        let db = test_db();
        for (method, path) in [("GET", "/.well-known/agent-card.json"), ("POST", "/a2a")] {
            let request = http(method, path, Some(ALICE), "{}");
            let response = handle(&request, &db, &A2aKeys::default()).unwrap();
            assert_eq!(response.status, 404, "{path}");
        }
        assert!(handle(&http("GET", "/other", None, ""), &db, &keys()).is_none());
    }

    #[test]
    fn test_the_card_lists_the_databases_profiles() {
        let db = test_db();
        add_profile(&db, "reviewer", json!({"description": "reviews code"}));
        add_profile(&db, "plain", json!({}));
        let request = http("GET", "/.well-known/agent-card.json", None, "");
        let response = handle(&request, &db, &keys()).unwrap();
        assert_eq!(response.status, 200);
        let card: Value = serde_json::from_slice(&response.body).unwrap();
        // The fields the A2A schema requires of a card.
        for field in [
            "protocolVersion",
            "name",
            "description",
            "url",
            "version",
            "capabilities",
            "defaultInputModes",
            "defaultOutputModes",
            "skills",
        ] {
            assert!(card.get(field).is_some(), "{field}");
        }
        assert_eq!(card["protocolVersion"], "0.3.0");
        assert_eq!(card["url"], "http://faber.test:9090/a2a");
        assert_eq!(card["preferredTransport"], "JSONRPC");
        assert_eq!(card["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(card["capabilities"]["streaming"], true);
        assert_eq!(card["capabilities"]["pushNotifications"], false);
        assert_eq!(card["security"], json!([{"bearer": []}]));
        let skills = card["skills"].as_array().unwrap();
        assert_eq!(skills.len(), 2);
        for skill in skills {
            for field in ["id", "name", "description", "tags"] {
                assert!(skill.get(field).is_some(), "{field}");
            }
        }
        assert_eq!(skills[0]["id"], "plain");
        assert_eq!(skills[0]["description"], "Profile plain");
        assert_eq!(skills[1]["id"], "reviewer");
        assert_eq!(skills[1]["description"], "reviews code");
        // Only a GET.
        let request = http("POST", "/.well-known/agent-card.json", None, "{}");
        assert_eq!(handle(&request, &db, &keys()).unwrap().status, 405);
    }

    #[test]
    fn test_only_a2a_keys_get_in_and_requests_are_checked() {
        let db = test_db();
        let ping = json!({"jsonrpc": "2.0", "id": 1, "method": "tasks/get"}).to_string();
        assert_eq!(answer(&db, None, &ping).0, 401);
        assert_eq!(answer(&db, Some("wrong"), &ping).0, 401);
        assert_eq!(answer(&db, Some(SERVER), &ping).0, 401, "the server's own");
        let (status, reply) = answer(&db, Some(ALICE), "not json");
        assert_eq!(
            (status, reply["error"]["code"].as_i64()),
            (200, Some(-32700))
        );
        let (_, reply) = answer(&db, Some(ALICE), "[]");
        assert_eq!(reply["error"]["code"], -32600, "no batches");
        let (_, reply) = answer(&db, Some(ALICE), r#"{"id": 1, "method": "x"}"#);
        assert_eq!(reply["error"]["code"], -32600, "no jsonrpc");
        let (_, reply) = answer(&db, Some(ALICE), r#"{"jsonrpc": "2.0", "method": "x"}"#);
        assert_eq!(reply["error"]["code"], -32600, "no id");
        assert_eq!(rpc_as(&db, ALICE, "bogus/method", json!({})), Err(-32601));
        assert_eq!(
            rpc_as(&db, ALICE, "agent/getAuthenticatedExtendedCard", json!({})),
            Err(-32601)
        );
        assert_eq!(
            rpc_as(&db, ALICE, "tasks/pushNotificationConfig/set", json!({})),
            Err(-32003)
        );
        assert_eq!(
            rpc_as(&db, ALICE, "tasks/get", json!({"id": "7"})),
            Err(-32001)
        );
        assert_eq!(rpc_as(&db, ALICE, "tasks/get", json!({})), Err(-32001));
        let mut request = http("POST", "/a2a", Some(ALICE), &ping);
        request.content_type = Some("text/plain".to_string());
        assert_eq!(handle(&request, &db, &keys()).unwrap().status, 415);
    }

    #[test]
    fn test_a_message_is_checked_before_anything_is_made() {
        let db = test_db();
        // No skills at all.
        assert_eq!(send(&db, ALICE, user("hi")), Err(-32602));
        add_profile(&db, "one", json!({}));
        add_profile(&db, "two", json!({}));
        // Which of several: it has to say.
        assert_eq!(send(&db, ALICE, user("hi")), Err(-32602));
        let mut message = user("hi");
        message["metadata"] = json!({"skill": "nope"});
        assert_eq!(send(&db, ALICE, message.clone()), Err(-32602));
        let mut bad = message.clone();
        bad["metadata"] = json!({"skill": "one"});
        bad["role"] = json!("agent");
        assert_eq!(send(&db, ALICE, bad), Err(-32602));
        let mut file = message.clone();
        file["metadata"] = json!({"skill": "one"});
        file["parts"] = json!([{"kind": "file", "file": {"uri": "http://x/y"}}]);
        assert_eq!(send(&db, ALICE, file), Err(-32005));
        for parts in [
            json!([]),
            json!([{"kind": "text", "text": "  "}]),
            json!([{"kind": "x"}]),
        ] {
            let mut empty = message.clone();
            empty["metadata"] = json!({"skill": "one"});
            empty["parts"] = parts;
            assert_eq!(send(&db, ALICE, empty), Err(-32602));
        }
        let mut huge = message.clone();
        huge["metadata"] = json!({"skill": "one"});
        huge["parts"] = json!([{"kind": "text", "text": "x".repeat(MAX_PART_BYTES + 1)}]);
        assert_eq!(send(&db, ALICE, huge), Err(-32602));
        // None of it made a context, nor a task.
        let conn = db.lock().unwrap();
        assert!(db::list_tasks(&conn, None).unwrap().is_empty());
        assert!(db::list_agents(&conn).unwrap().is_empty());
        drop(conn);
        // Output modes we can't produce.
        let body = |modes: Value| {
            json!({"message": {"role": "user", "parts": [{"kind": "text", "text": "x"}],
                               "metadata": {"skill": "one"}},
                   "configuration": {"acceptedOutputModes": modes}})
        };
        assert_eq!(
            rpc_as(&db, ALICE, "message/send", body(json!(["image/png"]))),
            Err(-32005)
        );
        assert!(
            rpc_as(
                &db,
                ALICE,
                "message/send",
                body(json!(["application/json"]))
            )
            .is_ok()
        );
    }

    #[test]
    fn test_a_prompt_is_made_of_the_parts() {
        let message = json!({"parts": [
            {"kind": "text", "text": "Look at this"},
            {"kind": "data", "data": {"id": 3, "items": []}, "metadata": {"name": "orders"}},
            {"kind": "data", "data": [1]},
        ]});
        let prompt = prompt_from_parts(&message, &Modes::Any).unwrap().text;
        assert_eq!(
            prompt,
            "Look at this\n\nData (orders):\n```json\n{\"id\":3,\"items\":[]}\n```\n\n```json\n[1]\n```"
        );
        let prompt = prompt_from_parts(&message, &Modes::JsonOnly).unwrap().text;
        assert!(prompt.ends_with(JSON_ONLY), "{prompt}");
        let accepted = |modes: Value| accepted_modes(&json!({"acceptedOutputModes": modes}));
        assert_eq!(accepted(json!(["text/plain"])).ok(), Some(Modes::Any));
        assert_eq!(
            accepted(json!(["application/json"])).ok(),
            Some(Modes::JsonOnly)
        );
        assert_eq!(
            accepted(json!(["application/json", "text/plain"])).ok(),
            Some(Modes::Any)
        );
        assert_eq!(accepted(json!([])).ok(), Some(Modes::Any));
        assert!(accepted(json!(["image/png"])).is_err());
    }

    #[test]
    fn test_answers_are_data_when_they_are_json() {
        assert_eq!(
            answer_part("hello"),
            json!({"kind": "text", "text": "hello"})
        );
        assert_eq!(
            answer_part(" {\"a\": 1} \n"),
            json!({"kind": "data", "data": {"a": 1}})
        );
        assert_eq!(answer_part("[1, 2]")["kind"], "data");
        assert_eq!(
            answer_part("```json\n{\"a\": 1}\n```"),
            json!({"kind": "data", "data": {"a": 1}})
        );
        // Not alone, not a block of json, or not valid.
        for text in [
            "Here: ```json\n{\"a\": 1}\n```",
            "```json\n{\"a\": 1}\n```\nand more ```json\n{}\n```",
            "```\n{\"a\": 1}\n```",
            "```json\n{oops\n```",
            "42",
            "{\"a\": ",
        ] {
            assert_eq!(answer_part(text)["kind"], "text", "{text}");
        }
    }

    /// A worker running in this process until dropped.
    struct Worker {
        stop: crate::WorkerStop,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Worker {
        fn start(db: &Arc<Mutex<Connection>>, name: &str, unsafe_tools: bool) -> Self {
            let backend: Arc<dyn faber::db_backend::DbBackend> =
                Arc::new(faber::local_db::LocalDb::new(db.clone()));
            let mut opts = crate::Opts::default();
            opts.unsafe_tools = unsafe_tools;
            let spec = crate::WorkerSpec {
                agent: name.to_string(),
                parallel: None,
                profiles: None,
                wait_for_agent: false,
                tool_tasks: false,
                label: true,
            };
            let stop = crate::WorkerStop::default();
            let thread = {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let opts = crate::with_db_profiles(&opts, &Some(backend.clone()));
                    crate::run_worker(&opts, backend, &None, &spec, &stop).unwrap();
                })
            };
            Self {
                stop,
                thread: Some(thread),
            }
        }
    }

    impl Drop for Worker {
        fn drop(&mut self) {
            self.stop.stop();
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    /// `tasks/get` until the task is in a state `done` accepts.
    fn wait_for(db: &Arc<Mutex<Connection>>, key: &str, id: &Value, states: &[&str]) -> Value {
        let started = Instant::now();
        loop {
            let task = rpc_as(db, key, "tasks/get", json!({"id": id})).unwrap();
            if states.contains(&task["status"]["state"].as_str().unwrap()) {
                return task;
            }
            assert!(started.elapsed() < Duration::from_secs(30), "{task}");
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    #[test]
    fn test_a_message_is_answered_by_a_worker_and_followed_up() {
        let model = model_script(
            "e2e",
            json!([
                {"if": "as json", "reply": "```json\n{\"verdict\": \"fine\"}\n```"},
                {"if": "second", "reply": "two"},
                {"if": "first", "reply": "one"},
            ]),
        );
        let db = test_db();
        add_profile(
            &db,
            "talker",
            json!({"model": model, "description": "talks"}),
        );
        let _worker = Worker::start(&db, "w-e2e", false);

        let sent = send(&db, ALICE, user("first question")).unwrap();
        assert_eq!(sent["kind"], "task");
        assert_eq!(sent["status"]["state"], "submitted");
        let (id, context) = (sent["id"].clone(), sent["contextId"].clone());
        let done = wait_for(&db, ALICE, &id, &["completed", "failed"]);
        assert_eq!(done["status"]["state"], "completed", "{done}");
        assert_eq!(done["artifacts"][0]["artifactId"], "result");
        assert_eq!(
            done["artifacts"][0]["parts"][0],
            json!({"kind": "text", "text": "one"})
        );
        let history = done["history"].as_array().unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0]["role"], "user");
        assert_eq!(history[0]["parts"][0]["text"], "first question");
        assert_eq!(history[1]["role"], "agent");
        assert_eq!(history[1]["parts"][0]["text"], "one");
        let last = rpc_as(
            &db,
            ALICE,
            "tasks/get",
            json!({"id": id, "historyLength": 1}),
        )
        .unwrap();
        assert_eq!(last["history"].as_array().unwrap().len(), 1);
        assert_eq!(last["history"][0]["role"], "agent");

        // Another caller can't see it, nor use its context, nor cancel it.
        assert_eq!(
            rpc_as(&db, BOB, "tasks/get", json!({"id": id})),
            Err(-32001)
        );
        assert_eq!(
            rpc_as(&db, BOB, "tasks/cancel", json!({"id": id})),
            Err(-32001)
        );
        let mut intruder = user("second question");
        intruder["contextId"] = context.clone();
        assert_eq!(send(&db, BOB, intruder), Err(-32602));
        let mut intruder = user("second question");
        intruder["taskId"] = id.clone();
        assert_eq!(send(&db, BOB, intruder), Err(-32001));

        // A follow-up in the same context, blocking: the agent has
        // the first exchange.
        let mut follow_up = user("second question");
        follow_up["contextId"] = context.clone();
        let reply = rpc_as(
            &db,
            ALICE,
            "message/send",
            json!({"message": follow_up, "configuration": {"blocking": true}}),
        )
        .unwrap();
        assert_eq!(reply["contextId"], context);
        assert_eq!(reply["status"]["state"], "completed", "{reply}");
        assert_eq!(reply["artifacts"][0]["parts"][0]["text"], "two");
        {
            let conn = db.lock().unwrap();
            let ctx = db::a2a_get_context(&conn, context.as_str().unwrap())
                .unwrap()
                .unwrap();
            let said: Vec<String> = db::load_agent_messages::<Value>(&conn, &ctx.agent)
                .unwrap()
                .iter()
                .filter(|m| m["role"] == "user")
                .map(|m| m["content"].as_str().unwrap().to_string())
                .collect();
            assert_eq!(said, vec!["first question", "second question"]);
        }

        // By task id: a done task's context.
        let mut again = user("second question");
        again["taskId"] = id.clone();
        let reply = send(&db, ALICE, again).unwrap();
        assert_eq!(reply["contextId"], context);

        // A JSON answer is a data part, when asked for or not.
        let reply = rpc_as(
            &db,
            ALICE,
            "message/send",
            json!({"message": {"role": "user", "parts": [
                        {"kind": "text", "text": "as json please"},
                        {"kind": "data", "data": {"n": 1}, "metadata": {"name": "input"}}],
                        "metadata": {"skill": "talker"}},
                   "configuration": {"blocking": true,
                                     "acceptedOutputModes": ["application/json"]}}),
        )
        .unwrap();
        assert_eq!(reply["status"]["state"], "completed", "{reply}");
        assert_eq!(
            reply["artifacts"][0]["parts"][0],
            json!({"kind": "data", "data": {"verdict": "fine"}})
        );
        assert_ne!(reply["contextId"], context, "a new context");
    }

    #[test]
    fn test_a_task_in_progress_can_be_canceled_and_a_done_one_cant() {
        let model = model_script(
            "cancel",
            json!([
                {"if": "slow", "reply": "too late", "delay_ms": 20000},
                {"if": "", "reply": "quick"},
            ]),
        );
        let db = test_db();
        add_profile(&db, "talker", json!({"model": model}));

        // Before any worker takes it.
        let waiting = send(&db, ALICE, user("slow one")).unwrap();
        assert_eq!(waiting["status"]["state"], "submitted");
        let canceled = rpc_as(&db, ALICE, "tasks/cancel", json!({"id": waiting["id"]})).unwrap();
        assert_eq!(canceled["status"]["state"], "canceled", "{canceled}");
        assert_eq!(
            rpc_as(&db, ALICE, "tasks/cancel", json!({"id": waiting["id"]})),
            Err(-32002)
        );

        let _worker = Worker::start(&db, "w-cancel", false);
        let running = send(&db, ALICE, user("slow one")).unwrap();
        wait_for(&db, ALICE, &running["id"], &["working"]);
        // A task still running can't be sent a message by its id.
        let mut busy = user("hello");
        busy["taskId"] = running["id"].clone();
        assert_eq!(send(&db, ALICE, busy), Err(-32602));
        rpc_as(&db, ALICE, "tasks/cancel", json!({"id": running["id"]})).unwrap();
        let task = wait_for(
            &db,
            ALICE,
            &running["id"],
            &["canceled", "completed", "failed"],
        );
        assert_eq!(task["status"]["state"], "canceled", "{task}");
        assert_eq!(task["artifacts"], json!([]));

        let quick = send(&db, ALICE, user("fast one")).unwrap();
        let task = wait_for(&db, ALICE, &quick["id"], &["completed", "failed"]);
        assert_eq!(task["status"]["state"], "completed");
        assert_eq!(
            rpc_as(&db, ALICE, "tasks/cancel", json!({"id": quick["id"]})),
            Err(-32002)
        );
    }

    #[test]
    fn test_a_failed_task_says_why() {
        let db = test_db();
        add_profile(
            &db,
            "broken",
            json!({"model": "script:/nonexistent/script.json"}),
        );
        let _worker = Worker::start(&db, "w-fail", false);
        let sent = send(&db, ALICE, user("hi")).unwrap();
        let task = wait_for(&db, ALICE, &sent["id"], &["completed", "failed"]);
        assert_eq!(task["status"]["state"], "failed", "{task}");
        assert!(
            task["status"]["message"]["parts"][0]["text"]
                .as_str()
                .unwrap()
                .contains("script"),
            "{task}"
        );
    }

    /// Where a stream writes, for a test to read.
    #[derive(Clone, Default)]
    struct Pipe(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Pipe {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Pipe {
        /// The events written so far (`result` of each `data:` line).
        fn events(&self) -> Vec<Value> {
            let text = String::from_utf8(self.0.lock().unwrap().clone()).unwrap();
            let body = text.split_once("\r\n\r\n").map_or("", |(_, body)| body);
            body.lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .map(|l| {
                    let reply: Value = serde_json::from_str(l).unwrap();
                    assert_eq!(reply["jsonrpc"], "2.0");
                    reply["result"].clone()
                })
                .collect()
        }

        fn head(&self) -> String {
            let text = String::from_utf8(self.0.lock().unwrap().clone()).unwrap();
            text.split_once("\r\n\r\n")
                .map_or("", |(h, _)| h)
                .to_string()
        }
    }

    /// Runs the stream for `body` on a thread, writing to the pipe.
    fn start_stream(
        db: &Arc<Mutex<Connection>>,
        key: &str,
        body: Value,
    ) -> (Pipe, std::thread::JoinHandle<()>) {
        let request = http("POST", "/a2a", Some(key), &body.to_string());
        let stream = match streaming(&request, db, &keys()) {
            Some(Ok(stream)) => stream,
            Some(Err(response)) => panic!("{}", String::from_utf8_lossy(&response.body)),
            None => panic!("not a stream"),
        };
        let pipe = Pipe::default();
        let mut writer = pipe.clone();
        let thread = std::thread::spawn(move || {
            let mut events = EventStream::start(&mut writer).unwrap();
            stream.run(&mut events).unwrap();
        });
        (pipe, thread)
    }

    fn stream_request(method: &str, params: Value) -> Value {
        json!({"jsonrpc": "2.0", "id": "s1", "method": method, "params": params})
    }

    /// Waits until `pipe` has `count` events.
    fn wait_for_events(pipe: &Pipe, count: usize) -> Vec<Value> {
        let started = Instant::now();
        loop {
            let events = pipe.events();
            if events.len() >= count {
                return events;
            }
            assert!(started.elapsed() < Duration::from_secs(20), "{events:?}");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn run_events(db: &Arc<Mutex<Connection>>, task: i64, events: &[AgentEvent]) {
        let conn = db.lock().unwrap();
        let agent = db::get_task(&conn, task)
            .unwrap()
            .unwrap()
            .continue_agent
            .unwrap();
        db::append_agent_events(&conn, &agent, Some(task), events).unwrap();
    }

    fn text(text: &str) -> AgentEvent {
        AgentEvent::Text {
            text: text.to_string(),
        }
    }

    #[test]
    fn test_a_stream_follows_the_task_and_can_be_picked_up_again() {
        let db = test_db();
        add_profile(&db, "talker", json!({}));
        let message = json!({"role": "user", "parts": [{"kind": "text", "text": "hi"}]});
        let (pipe, stream) = start_stream(
            &db,
            ALICE,
            stream_request("message/stream", json!({"message": message})),
        );
        // First, the task as it is.
        let first = wait_for_events(&pipe, 1).remove(0);
        assert!(
            pipe.head().starts_with("HTTP/1.1 200 OK"),
            "{}",
            pipe.head()
        );
        assert!(pipe.head().contains("Content-Type: text/event-stream"));
        assert_eq!(first["kind"], "task");
        assert_eq!(first["status"]["state"], "submitted");
        let task: i64 = first["id"].as_str().unwrap().parse().unwrap();
        let context = first["contextId"].as_str().unwrap().to_string();
        assert!(
            db::claim_task(&db.lock().unwrap(), task, "ws").unwrap(),
            "due already"
        );

        run_events(
            &db,
            task,
            &[
                AgentEvent::Input {
                    text: "hi".to_string(),
                },
                text("Hel"),
                AgentEvent::Reasoning {
                    text: "hmm".to_string(),
                },
            ],
        );
        // working, and the first chunk.
        let events = wait_for_events(&pipe, 3);
        assert_eq!(events[1]["kind"], "status-update");
        assert_eq!(events[1]["status"]["state"], "working");
        assert_eq!(events[1]["final"], false);
        assert_eq!(events[2]["kind"], "artifact-update");
        assert_eq!(events[2]["append"], false);
        assert_eq!(events[2]["artifact"]["artifactId"], "result");
        assert_eq!(events[2]["artifact"]["parts"][0]["text"], "Hel");

        // Picked up again mid-run: it carries on from where it is.
        let (again, resubscribed) = start_stream(
            &db,
            ALICE,
            stream_request("tasks/resubscribe", json!({"id": task.to_string()})),
        );
        let first = wait_for_events(&again, 1).remove(0);
        assert_eq!(first["kind"], "task");
        assert_eq!(first["status"]["state"], "working");
        assert_eq!(first["contextId"], context.as_str());
        run_events(
            &db,
            task,
            &[
                AgentEvent::ToolStart {
                    name: "glob".to_string(),
                    arguments: "{}".to_string(),
                },
                AgentEvent::ToolEnd {
                    name: "glob".to_string(),
                    duration_ms: 1,
                    output: "x".to_string(),
                    failed: false,
                },
                text("lo"),
                AgentEvent::TurnEnd {
                    succeeded: true,
                    text: "Hello".to_string(),
                },
            ],
        );
        {
            let conn = db.lock().unwrap();
            let agent = db::get_task(&conn, task)
                .unwrap()
                .unwrap()
                .continue_agent
                .unwrap();
            db::set_agent_data(&conn, &agent, &db::a2a_answer_key(task), "Hello").unwrap();
            db::finish_task(
                &conn,
                task,
                "ws",
                &db::TaskOutcome {
                    succeeded: true,
                    exit_code: None,
                    result: "Hello".to_string(),
                },
            )
            .unwrap();
        }
        stream.join().unwrap();
        resubscribed.join().unwrap();

        let events = pipe.events();
        let kinds: Vec<(&str, Option<&str>)> = events
            .iter()
            .map(|e| (e["kind"].as_str().unwrap(), e["status"]["state"].as_str()))
            .collect();
        assert_eq!(
            kinds,
            vec![
                ("task", Some("submitted")),
                ("status-update", Some("working")),
                ("artifact-update", None),
                ("status-update", Some("working")), // Running glob
                ("artifact-update", None),
                ("artifact-update", None), // the last chunk, empty
                ("status-update", Some("completed")),
            ]
        );
        assert_eq!(
            events[3]["status"]["message"]["parts"][0]["text"],
            "Running glob"
        );
        let chunks: Vec<(&str, bool)> = events
            .iter()
            .filter(|e| e["kind"] == "artifact-update")
            .map(|e| {
                (
                    e["artifact"]["parts"][0]["text"].as_str().unwrap_or(""),
                    e["append"].as_bool().unwrap(),
                )
            })
            .collect();
        assert_eq!(chunks, vec![("Hel", false), ("lo", true), ("", true)]);
        assert_eq!(events[5]["artifact"]["parts"], json!([]), "no empty part");
        assert_eq!(events[4]["lastChunk"], false);
        assert_eq!(events[5]["lastChunk"], true);
        assert_eq!(events[6]["final"], true);
        assert_eq!(events[6]["taskId"], task.to_string());

        // The picked-up stream saw only what came after, then the whole
        // result in place of that tail.
        let events = again.events();
        let chunks: Vec<(&str, bool)> = events
            .iter()
            .filter(|e| e["kind"] == "artifact-update")
            .map(|e| {
                (
                    e["artifact"]["parts"][0]["text"].as_str().unwrap_or(""),
                    e["append"].as_bool().unwrap(),
                )
            })
            .collect();
        assert_eq!(chunks, vec![("lo", false), ("Hello", false)]);
        assert_eq!(events.last().unwrap()["final"], true);

        // A done task: its state, and the end.
        let (done, finished) = start_stream(
            &db,
            ALICE,
            stream_request("tasks/resubscribe", json!({"id": task.to_string()})),
        );
        finished.join().unwrap();
        let events = done.events();
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0]["status"]["state"], "completed");
        assert_eq!(events[0]["artifacts"][0]["parts"][0]["text"], "Hello");

        // Not another caller's, and not a bad request.
        let request = http(
            "POST",
            "/a2a",
            Some(BOB),
            &stream_request("tasks/resubscribe", json!({"id": task.to_string()})).to_string(),
        );
        let Some(Err(response)) = streaming(&request, &db, &keys()) else {
            panic!("bob got a stream");
        };
        let reply: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(
            (response.status, &reply["id"], &reply["error"]["code"]),
            (200, &json!("s1"), &json!(-32001))
        );
        let request = http(
            "POST",
            "/a2a",
            Some(ALICE),
            &stream_request(
                "message/stream",
                json!({"message": {"role": "user", "parts": []}}),
            )
            .to_string(),
        );
        let Some(Err(response)) = streaming(&request, &db, &keys()) else {
            panic!("an empty message got a stream");
        };
        assert_eq!(
            serde_json::from_slice::<Value>(&response.body).unwrap()["error"]["code"],
            -32602
        );
        // Other requests aren't streams.
        let request = http(
            "POST",
            "/a2a",
            Some(ALICE),
            &stream_request("tasks/get", json!({})).to_string(),
        );
        assert!(streaming(&request, &db, &keys()).is_none());
        let request = http(
            "POST",
            "/a2a",
            Some(SERVER),
            &stream_request("message/stream", json!({})).to_string(),
        );
        assert!(
            streaming(&request, &db, &keys()).is_none(),
            "left to say 401"
        );
    }

    #[test]
    fn test_a_failed_or_canceled_stream_says_so_in_the_end() {
        let db = test_db();
        add_profile(&db, "talker", json!({}));
        let message = json!({"role": "user", "parts": [{"kind": "text", "text": "hi"}]});
        let (pipe, stream) = start_stream(
            &db,
            ALICE,
            stream_request("message/stream", json!({"message": message})),
        );
        let task: i64 = wait_for_events(&pipe, 1)[0]["id"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        {
            let conn = db.lock().unwrap();
            db::claim_task(&conn, task, "ws").unwrap();
            db::finish_task(
                &conn,
                task,
                "ws",
                &db::TaskOutcome {
                    succeeded: false,
                    exit_code: None,
                    result: "no model".to_string(),
                },
            )
            .unwrap();
        }
        stream.join().unwrap();
        let events = pipe.events();
        let end = events.last().unwrap();
        assert_eq!(end["status"]["state"], "failed");
        assert_eq!(end["final"], true);
        assert_eq!(end["status"]["message"]["parts"][0]["text"], "no model");
        assert!(events.iter().all(|e| e["kind"] != "artifact-update"));
    }

    #[test]
    fn test_streaming_over_http_with_a_worker() {
        let model = model_script("stream", json!([{"if": "", "reply": "streamed answer"}]));
        let db = test_db();
        add_profile(&db, "talker", json!({"model": model}));
        let _worker = Worker::start(&db, "w-stream", false);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = {
            let db = db.clone();
            std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                crate::web::handle_http(stream, db, Some(SERVER), &crate::Profiles::new(), &keys())
                    .unwrap();
            })
        };
        let body = stream_request(
            "message/stream",
            json!({"message": {"role": "user", "parts": [{"kind": "text", "text": "go"}]}}),
        )
        .to_string();
        let mut client = std::net::TcpStream::connect(addr).unwrap();
        std::io::Write::write_all(
            &mut client,
            format!(
                "POST /a2a HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {}\r\n\
                 Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                ALICE,
                body.len(),
                body
            )
            .as_bytes(),
        )
        .unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        let mut reply = String::new();
        std::io::Read::read_to_string(&mut client, &mut reply).unwrap();
        server.join().unwrap();
        assert!(reply.starts_with("HTTP/1.1 200 OK"), "{reply}");
        assert!(reply.contains("Content-Type: text/event-stream"), "{reply}");
        let events: Vec<Value> = reply
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(|l| serde_json::from_str::<Value>(l).unwrap()["result"].clone())
            .collect();
        assert_eq!(events[0]["kind"], "task");
        let end = events.last().unwrap();
        assert_eq!(end["status"]["state"], "completed", "{events:?}");
        assert_eq!(end["final"], true);
        let said: String = events
            .iter()
            .filter(|e| e["kind"] == "artifact-update")
            .filter_map(|e| e["artifact"]["parts"][0]["text"].as_str())
            .collect();
        assert_eq!(said, "streamed answer");
    }
}
