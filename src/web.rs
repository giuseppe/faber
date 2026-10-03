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

//! `faber serve`'s HTTP side: a JSON API over the agents, their events
//! and the tasks, and the web UI (`ui/`) built on it. Served on the same
//! port as the line protocol - `server::serve_command` tells them apart by
//! the first byte. One request per connection.
//!
//! - `GET /api/agents`, `GET /api/agents/<name>` (with its settings),
//!   `POST /api/agents` (make one), `DELETE /api/agents/<name>`,
//!   `PATCH /api/agents/<name>/config` (change its settings - the user's
//!   to, unsafe tools and working directory included),
//!   `POST /api/agents/<name>/cwd`, `POST /api/agents/<name>/stop` (stop
//!   what it's doing), `POST /api/agents/<name>/clear` (its conversation)
//! - `GET /api/tools`: the tools an agent's list can name, with
//!   `?schemas=1` their descriptions and parameters too;
//!   `POST /api/tools/<name>/run`: run one now, as a tool task
//! - `POST /api/tasks/prune`, `POST /api/agents/cleanup`: remove finished
//!   tasks, and finished agents made by agents (`dry_run` to only see)
//! - `GET /api/events?agent=&task=&after=&limit=`
//! - `GET /api/tasks`, `GET /api/tasks/<id>`, `POST /api/tasks`,
//!   `PATCH /api/tasks/<id>` (change it, as `POST /api/tasks` takes it),
//!   `DELETE /api/tasks/<id>`,
//!   `POST /api/tasks/<id>/hold|release|enable|disable|run|stop`,
//!   `POST /api/tasks/<id>/assign`
//! - `GET /api/profiles` (the config file's and the database's),
//!   `POST /api/profiles`, `DELETE /api/profiles/<name>` (the database's),
//! - `GET /api/info` (the server's directory, for a
//!   new task's default)
//!
//! With an auth key, every `/api/` request needs `Authorization: Bearer
//! <key>`; the UI's own files don't, as they hold no data. Without one,
//! the API only answers to a `Host` that is `localhost` or an IP address,
//! so a web page can't reach it through a hostname it controls (DNS
//! rebinding). Either way, a request that changes anything must be
//! `Content-Type: application/json`, which a page on another origin can't
//! send without a CORS preflight - never granted here.

use faber::agent_io::EventFilter;
use faber::db;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

/// The largest request body accepted.
const MAX_BODY_BYTES: usize = 1 << 20;

/// The most header lines accepted.
const MAX_HEADERS: usize = 100;

/// The most events one `/api/events` request returns.
const MAX_EVENTS: usize = 1000;

struct Request {
    method: String,
    path: String,
    query: Vec<(String, String)>,
    authorization: Option<String>,
    host: Option<String>,
    content_type: Option<String>,
    body: Vec<u8>,
}

impl Request {
    fn query(&self, name: &str) -> Option<&str> {
        self.query
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

#[derive(Debug, PartialEq)]
struct Response {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
}

impl Response {
    fn json(status: u16, value: &serde_json::Value) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: value.to_string().into_bytes(),
        }
    }

    fn ok(value: impl Serialize) -> Self {
        match serde_json::to_value(value) {
            Ok(value) => Self::json(200, &value),
            Err(e) => Self::error(500, &e.to_string()),
        }
    }

    fn error(status: u16, message: &str) -> Self {
        Self::json(status, &serde_json::json!({ "error": message }))
    }

    fn file(content_type: &'static str, content: &'static str) -> Self {
        Self {
            status: 200,
            content_type,
            body: content.as_bytes().to_vec(),
        }
    }
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        302 => "Found",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        415 => "Unsupported Media Type",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        _ => "Internal Server Error",
    }
}

/// Decodes `%XX` escapes and `+` (a space, in a query string).
fn percent_decode(text: &str) -> String {
    let hex = |b: u8| (b as char).to_digit(16).map(|d| d as u8);
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' => {
                match (
                    bytes.get(i + 1).copied().and_then(hex),
                    bytes.get(i + 2).copied().and_then(hex),
                ) {
                    (Some(high), Some(low)) => {
                        out.push(high << 4 | low);
                        i += 2;
                    }
                    _ => out.push(b'%'),
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn parse_query(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|pair| {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            (percent_decode(k), percent_decode(v))
        })
        .collect()
}

/// Reads one HTTP/1.x request, or `Err` with the response to send for a
/// malformed one.
fn read_request(reader: &mut impl BufRead) -> Result<Request, Response> {
    let bad = |m: &str| Response::error(400, m);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|_| bad("unreadable request"))?;
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return Err(bad("malformed request line"));
    };
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut request = Request {
        method: method.to_string(),
        path: percent_decode(path),
        query: parse_query(query),
        authorization: None,
        host: None,
        content_type: None,
        body: Vec::new(),
    };
    let mut content_length = 0;
    for _ in 0..MAX_HEADERS {
        line.clear();
        reader
            .read_line(&mut line)
            .map_err(|_| bad("unreadable headers"))?;
        let header = line.trim_end();
        if header.is_empty() {
            if content_length > MAX_BODY_BYTES {
                return Err(Response::error(413, "request body too large"));
            }
            request.body = vec![0; content_length];
            reader
                .read_exact(&mut request.body)
                .map_err(|_| bad("truncated body"))?;
            return Ok(request);
        }
        let Some((name, value)) = header.split_once(':') else {
            return Err(bad("malformed header"));
        };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => {
                content_length = value.parse().map_err(|_| bad("bad Content-Length"))?;
            }
            "authorization" => request.authorization = Some(value.to_string()),
            "host" => request.host = Some(value.to_string()),
            "content-type" => request.content_type = Some(value.to_ascii_lowercase()),
            _ => {}
        }
    }
    Err(bad("too many headers"))
}

fn write_response(writer: &mut impl Write, response: &Response) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n",
        response.status,
        status_text(response.status),
        response.content_type,
        response.body.len()
    );
    if response.status == 302 {
        head.push_str("Location: /ui/\r\n");
    }
    head.push_str("\r\n");
    let mut out = head.into_bytes();
    out.extend_from_slice(&response.body);
    writer.write_all(&out)?;
    writer.flush()
}

/// Serves one HTTP request on `stream`.
pub fn handle_http(
    stream: TcpStream,
    db: Arc<Mutex<Connection>>,
    auth_key: Option<&str>,
    profiles: &crate::Profiles,
) -> Result<(), Box<dyn Error>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let response = match read_request(&mut reader) {
        Ok(request) => route(&request, &db, auth_key, profiles),
        Err(response) => response,
    };
    write_response(&mut writer, &response)?;
    Ok(())
}

fn route(
    request: &Request,
    db: &Arc<Mutex<Connection>>,
    auth_key: Option<&str>,
    profiles: &crate::Profiles,
) -> Response {
    let segments: Vec<&str> = request.path.split('/').filter(|s| !s.is_empty()).collect();
    match (request.method.as_str(), segments.as_slice()) {
        ("GET", []) | ("GET", ["ui"]) if !request.path.ends_with("ui/") => Response {
            status: 302,
            content_type: "text/plain",
            body: Vec::new(),
        },
        ("GET", ["ui"]) | ("GET", ["ui", "index.html"]) => {
            Response::file("text/html; charset=utf-8", include_str!("../ui/index.html"))
        }
        ("GET", ["ui", "app.js"]) => Response::file(
            "text/javascript; charset=utf-8",
            include_str!("../ui/app.js"),
        ),
        ("GET", ["ui", "style.css"]) => {
            Response::file("text/css; charset=utf-8", include_str!("../ui/style.css"))
        }
        (_, ["api", rest @ ..]) => {
            match auth_key {
                Some(key) => {
                    let expected = format!("Bearer {}", key);
                    if request.authorization.as_deref() != Some(expected.as_str()) {
                        return Response::error(401, "unauthorized");
                    }
                }
                None if !is_local_host(request.host.as_deref()) => {
                    return Response::error(
                        403,
                        "without --auth-key, the API only answers at localhost or an IP address",
                    );
                }
                None => {}
            }
            if request.method != "GET"
                && !request
                    .content_type
                    .as_deref()
                    .is_some_and(|t| t.starts_with("application/json"))
            {
                return Response::error(415, "send Content-Type: application/json");
            }
            let conn = match db.lock() {
                Ok(conn) => conn,
                Err(e) => return Response::error(500, &format!("DB lock: {}", e)),
            };
            match api(&request.method, rest, request, &conn, profiles) {
                Ok(response) => response,
                Err(e) => Response::error(400, &e.to_string()),
            }
        }
        _ => Response::error(404, "not found"),
    }
}

/// An agent's plan, as plan_update keeps it - `None` without one.
fn agent_plan(conn: &Connection, agent: &str) -> Result<Option<serde_json::Value>, Box<dyn Error>> {
    Ok(db::get_agent_data(conn, agent, crate::PLAN_DATA_KEY)?
        .and_then(|plan| serde_json::from_str::<serde_json::Value>(&plan).ok())
        .filter(|plan| plan.as_array().is_some_and(|steps| !steps.is_empty())))
}

/// The config an agent made from `profile` (the config file's, or else
/// the database's) gets: a copy of its settings, keeping what's not a
/// profile's to set - the agent's unsafe tools and working directory.
fn profile_config(
    conn: &Connection,
    profiles: &crate::Profiles,
    profile: &str,
    current: db::AgentConfig,
) -> Result<db::AgentConfig, String> {
    let settings = match profiles.get(profile) {
        Some(settings) => settings.clone(),
        None => {
            let stored = db::list_profiles(conn)
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|(name, _)| name == profile)
                .ok_or_else(|| format!("no profile '{}'", profile))?;
            serde_json::from_value(stored.1).map_err(|e| e.to_string())?
        }
    };
    let mut config = settings.agent_config(profile);
    config.unsafe_tools = current.unsafe_tools;
    config.cwd = current.cwd;
    Ok(config)
}

/// The settings of an agent the web UI edits.
fn editable_config(config: &db::AgentConfig) -> serde_json::Value {
    serde_json::json!({
        "model": config.model,
        "endpoint": config.endpoint,
        "system_prompt": config.system_prompt,
        "max_tokens": config.max_tokens,
        "context_window": config.context_window,
        "parameters": config.parameters,
        "tools": config.tools,
        "unsafe_tools": config.unsafe_tools == Some(true),
        "cwd": config.cwd,
        "profile": config.profile,
    })
}

/// `config` with `changes` made: each field given is set, or cleared with
/// `null` (or an empty string); others are left as they are. Coming from
/// the user, it may give or take the unsafe tools and set the working
/// directory - which agents can't do to each other.
fn apply_config(
    mut config: db::AgentConfig,
    changes: &serde_json::Map<String, serde_json::Value>,
) -> Result<db::AgentConfig, String> {
    let text = |value: &serde_json::Value, field: &str| -> Result<Option<String>, String> {
        match value {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::String(s) if s.trim().is_empty() => Ok(None),
            serde_json::Value::String(s) => Ok(Some(s.clone())),
            _ => Err(format!("'{}' must be a string", field)),
        }
    };
    let number = |value: &serde_json::Value, field: &str| -> Result<Option<u32>, String> {
        match value {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::String(s) if s.trim().is_empty() => Ok(None),
            v => v
                .as_u64()
                .filter(|n| *n > 0)
                .and_then(|n| u32::try_from(n).ok())
                .map(Some)
                .ok_or_else(|| format!("'{}' must be a positive number", field)),
        }
    };
    for (field, value) in changes {
        match field.as_str() {
            "model" => config.model = text(value, field)?,
            "endpoint" => config.endpoint = text(value, field)?,
            "system_prompt" => config.system_prompt = text(value, field)?,
            "max_tokens" => config.max_tokens = number(value, field)?,
            "context_window" => config.context_window = number(value, field)?,
            "cwd" => config.cwd = text(value, field)?.map(|d| absolute_dir(&d)).transpose()?,
            "unsafe_tools" => {
                config.unsafe_tools = match value {
                    serde_json::Value::Bool(b) => Some(*b),
                    serde_json::Value::Null => None,
                    _ => return Err("'unsafe_tools' must be true or false".to_string()),
                }
            }
            "parameters" => {
                config.parameters = match value {
                    serde_json::Value::Null => None,
                    serde_json::Value::Object(map) if map.is_empty() => None,
                    serde_json::Value::Object(map) => Some(map.clone()),
                    _ => return Err("'parameters' must be an object".to_string()),
                }
            }
            "tools" => {
                config.tools = match value {
                    serde_json::Value::Null => None,
                    serde_json::Value::Array(names) => {
                        let known = crate::initialize_tools(true, None);
                        let mut tools = Vec::new();
                        for name in names {
                            let name = name.as_str().ok_or("'tools' must list tool names")?;
                            if !known.contains_key(name) {
                                return Err(format!("no tool named '{}'", name));
                            }
                            tools.push(name.to_string());
                        }
                        Some(tools)
                    }
                    _ => return Err("'tools' must be a list of tool names, or null".to_string()),
                }
            }
            other => {
                return Err(format!(
                    "'{}' isn't a setting that can be changed here",
                    other
                ));
            }
        }
    }
    Ok(config)
}

/// Whether a `Host` header names this machine by IP address or as
/// `localhost` - no name someone else's DNS could point here.
fn is_local_host(host: Option<&str>) -> bool {
    let Some(host) = host else {
        return false;
    };
    let name = if let Some(rest) = host.strip_prefix('[') {
        // [::1]:9090
        return rest
            .split_once(']')
            .is_some_and(|(ip, _)| ip.parse::<std::net::Ipv6Addr>().is_ok());
    } else {
        host.rsplit_once(':').map_or(host, |(name, _)| name)
    };
    name.eq_ignore_ascii_case("localhost") || name.parse::<std::net::IpAddr>().is_ok()
}

/// An agent as `/api/agents` lists it.
#[derive(Serialize)]
struct ApiAgent {
    #[serde(flatten)]
    agent: db::AgentRow,
    live: bool,
    model: Option<String>,
    profile: Option<String>,
    /// Has the unsafe tools now: for good (`faber agents set --unsafe`, its
    /// settings, an agent that has them), or for the session running it
    /// (started with --unsafe-tools).
    unsafe_tools: bool,
    /// Where it works, if not where whatever runs it does.
    cwd: Option<String>,
    /// Its plan (plan_update), if it has one: its steps, each with what
    /// it is and its status (pending, in_progress, completed).
    plan: Option<serde_json::Value>,
}

/// A task as `/api/tasks` lists it: with the dependencies it's still
/// waiting on.
#[derive(Serialize)]
struct ApiTask {
    #[serde(flatten)]
    task: db::TaskRow,
    blocked_by: Vec<i64>,
}

/// A task to create, as `POST /api/tasks` takes it.
#[derive(Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
struct ApiNewTask {
    /// What to do: instructions for an agent, or with `tool`, a
    /// `{"tool": ..., "arguments": ...}` call.
    command: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool: bool,
    /// Run on this agent - or on a new agent made from `profile`; neither:
    /// on whichever agent picks it up first.
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    profile: Option<String>,
    /// Run once at this RFC 3339 time (default: now)...
    #[serde(default)]
    at: Option<String>,
    /// ...or on this 7-field cron schedule.
    #[serde(default)]
    cron: Option<String>,
    #[serde(default)]
    max_runs: Option<i64>,
    #[serde(default)]
    depends_on: Vec<i64>,
    #[serde(default)]
    hold: bool,
    /// The directory it runs in, absolute; default: its agent's.
    #[serde(default)]
    cwd: Option<String>,
}

/// `dir` as a working directory: absolute, and resolved if it's on this
/// machine - the agent running it may be on another.
fn absolute_dir(dir: &str) -> Result<String, String> {
    let path = std::path::Path::new(dir);
    if !path.is_absolute() {
        return Err(format!("'{}': give an absolute path", dir));
    }
    Ok(path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned())
}

fn non_empty(value: &Option<String>) -> Option<String> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(String::from)
}

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

impl ApiNewTask {
    fn into_new_task(self, now: chrono::DateTime<chrono::Utc>) -> Result<db::NewTask, String> {
        let command = self.command.trim().to_string();
        if command.is_empty() {
            return Err("the task's command is empty".to_string());
        }
        let tool_name = if self.tool {
            let call: serde_json::Value = serde_json::from_str(&command)
                .map_err(|e| format!("a tool task's command must be a JSON tool call: {}", e))?;
            Some(
                call.get("tool")
                    .and_then(|t| t.as_str())
                    .filter(|t| !t.is_empty())
                    .ok_or(
                        r#"a tool task's command must be {"tool": "<name>", "arguments": {...}}"#,
                    )?
                    .to_string(),
            )
        } else {
            None
        };
        let schedule = match (non_empty(&self.cron), non_empty(&self.at)) {
            (Some(expression), _) => db::TaskSchedule::Cron {
                expression,
                max_runs: self.max_runs,
            },
            (None, Some(at)) => db::TaskSchedule::Once { at },
            (None, None) => db::TaskSchedule::Once {
                at: now.to_rfc3339(),
            },
        };
        let name = non_empty(&self.name)
            .or(tool_name)
            .unwrap_or_else(|| first_line(&command, 40));
        let cwd = non_empty(&self.cwd)
            .map(|dir| absolute_dir(&dir))
            .transpose()?;
        Ok(db::NewTask {
            name,
            description: String::new(),
            kind: if self.tool {
                db::TaskKind::TOOL
            } else {
                db::TaskKind::PROMPT
            }
            .to_string(),
            command,
            agent_name: non_empty(&self.agent),
            schedule,
            held: self.hold,
            depends_on: self.depends_on,
            profile: non_empty(&self.profile),
            run_safe: false,
            cwd,
        })
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ApiTarget {
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    profile: Option<String>,
}

fn body_json<T: for<'de> Deserialize<'de>>(request: &Request) -> Result<T, Box<dyn Error>> {
    serde_json::from_slice(&request.body).map_err(|e| format!("bad request body: {}", e).into())
}

fn api(
    method: &str,
    path: &[&str],
    request: &Request,
    conn: &Connection,
    profiles: &crate::Profiles,
) -> Result<Response, Box<dyn Error>> {
    let task_id = |id: &str| -> Result<i64, Box<dyn Error>> {
        id.parse()
            .map_err(|_| format!("bad task id '{}'", id).into())
    };
    let now = chrono::Utc::now();
    Ok(match (method, path) {
        ("GET", ["agents"]) => {
            let mut agents = Vec::new();
            for agent in db::list_agents(conn)? {
                let config = db::get_agent_config(conn, &agent.name)?;
                agents.push(ApiAgent {
                    live: db::agent_session_is_live(&agent, now),
                    model: config.model.clone(),
                    profile: config.profile.clone(),
                    unsafe_tools: db::has_unsafe_tools(
                        &agent,
                        &config,
                        db::get_agent_data(conn, &agent.name, db::SESSION_UNSAFE_KEY)?.as_deref(),
                        now,
                    ),
                    cwd: config.cwd.clone(),
                    plan: agent_plan(conn, &agent.name)?,
                    agent,
                });
            }
            Response::ok(agents)
        }
        ("GET", ["agents", name]) => {
            let Some(agent) = db::get_agent(conn, name)? else {
                return Ok(Response::error(404, &format!("no agent named '{}'", name)));
            };
            let config = db::get_agent_config(conn, name)?;
            let children: Vec<String> = db::list_agents(conn)?
                .into_iter()
                .filter(|a| a.parent.as_deref() == Some(*name))
                .map(|a| a.name)
                .collect();
            Response::ok(serde_json::json!({
                "agent": ApiAgent {
                    live: db::agent_session_is_live(&agent, now),
                    model: config.model.clone(),
                    profile: config.profile.clone(),
                    unsafe_tools: db::has_unsafe_tools(
                        &agent,
                        &config,
                        db::get_agent_data(conn, &agent.name, db::SESSION_UNSAFE_KEY)?.as_deref(),
                        now,
                    ),
                    cwd: config.cwd.clone(),
                    plan: agent_plan(conn, &agent.name)?,
                    agent,
                },
                "children": children,
                "system_prompt": config.system_prompt,
                "tools": config.tools,
                "config": editable_config(&config),
                "messages": db::agent_message_count(conn, name)?,
            }))
        }
        ("POST", ["agents"]) => {
            #[derive(Deserialize)]
            struct Body {
                name: String,
                #[serde(default)]
                description: String,
                #[serde(default)]
                config: serde_json::Map<String, serde_json::Value>,
            }
            let body: Body = body_json(request)?;
            let name = body.name.trim();
            if name.is_empty() || name.chars().any(|c| c.is_whitespace() || c == '/') {
                return Err("give the agent a name without spaces or slashes".into());
            }
            if db::get_agent(conn, name)?.is_some() {
                return Err(format!("there is already an agent named '{}'", name).into());
            }
            // The settings first: a bad one makes nothing.
            let config = apply_config(db::AgentConfig::default(), &body.config)?;
            db::create_agent(conn, name, body.description.trim())?;
            db::set_agent_config(conn, name, &config)?;
            Response::json(201, &serde_json::json!({ "name": name }))
        }
        ("PATCH", ["agents", name, "config"]) => {
            if db::get_agent(conn, name)?.is_none() {
                return Ok(Response::error(404, &format!("no agent named '{}'", name)));
            }
            let mut changes: serde_json::Map<String, serde_json::Value> = body_json(request)?;
            // A profile's settings first, then the others given.
            let base = match changes.remove("profile") {
                Some(serde_json::Value::String(profile)) => {
                    profile_config(conn, profiles, &profile, db::get_agent_config(conn, name)?)?
                }
                None | Some(serde_json::Value::Null) => db::get_agent_config(conn, name)?,
                Some(_) => return Err("'profile' must be a profile's name".into()),
            };
            // Not a setting, but the user's to change too.
            let description = match changes.remove("description") {
                None => None,
                Some(serde_json::Value::String(d)) => Some(d.trim().to_string()),
                Some(serde_json::Value::Null) => Some(String::new()),
                Some(_) => return Err("'description' must be a string".into()),
            };
            let config = apply_config(base, &changes)?;
            db::set_agent_config(conn, name, &config)?;
            if let Some(description) = description {
                db::set_agent_description(conn, name, &description)?;
            }
            Response::ok(editable_config(&config))
        }
        ("DELETE", ["agents", name]) => {
            let Some(agent) = db::get_agent(conn, name)? else {
                return Ok(Response::error(404, &format!("no agent named '{}'", name)));
            };
            if db::agent_session_is_live(&agent, now) {
                return Err(format!("'{}' is running: stop it first", name).into());
            }
            db::delete_agent(conn, name)?;
            Response::ok(true)
        }
        ("POST", ["agents", name, "stop"]) => {
            let Some(agent) = db::get_agent(conn, name)? else {
                return Ok(Response::error(404, &format!("no agent named '{}'", name)));
            };
            if !db::agent_session_is_live(&agent, now) {
                return Err(format!("nothing is running '{}'", name).into());
            }
            // Whichever process runs it acts on it (handle_agent_stops).
            db::set_agent_data(conn, name, db::AGENT_STOP_KEY, &now.to_rfc3339())?;
            Response::ok(true)
        }
        ("POST", ["agents", name, "clear"]) => {
            let Some(agent) = db::get_agent(conn, name)? else {
                return Ok(Response::error(404, &format!("no agent named '{}'", name)));
            };
            if db::agent_session_is_live(&agent, now) {
                return Err(format!(
                    "'{}' is running: its session would keep its conversation (in a chat, use /clear)",
                    name
                )
                .into());
            }
            db::clear_agent_messages(conn, name)?;
            db::delete_agent_data(conn, name, crate::PLAN_DATA_KEY)?;
            Response::ok(true)
        }
        ("POST", ["agents", name, "messages"]) => {
            // A message from the user: a task for the agent, so it reaches
            // it however it runs - a chat, a worker - once it's free.
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Body {
                text: String,
            }
            let Some(agent) = db::get_agent(conn, name)? else {
                return Ok(Response::error(404, &format!("no agent named '{}'", name)));
            };
            let text = body_json::<Body>(request)?.text.trim().to_string();
            if text.is_empty() {
                return Err("the message is empty".into());
            }
            let id = db::create_task(conn, &db::message_task(name, &text, now))?;
            let mut reply = serde_json::json!({ "task_id": id });
            if !db::agent_session_is_live(&agent, now) {
                reply["note"] = format!(
                    "Nothing is running '{}' now: it gets the message once a chat or worker does.",
                    name
                )
                .into();
            }
            Response::json(201, &reply)
        }
        ("GET", ["agents", name, "messages"]) => {
            if db::get_agent(conn, name)?.is_none() {
                return Ok(Response::error(404, &format!("no agent named '{}'", name)));
            }
            let messages: Vec<serde_json::Value> = db::load_agent_messages(conn, name)?;
            let limit = request
                .query("limit")
                .and_then(|l| l.parse::<usize>().ok())
                .unwrap_or(300);
            let skip = messages.len().saturating_sub(limit);
            Response::ok(serde_json::json!({
                "total": messages.len(),
                "messages": &messages[skip..],
            }))
        }
        ("POST", ["tasks", "prune"]) => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Body {
                /// How long ago their last run, at least: e.g. 7d.
                older_than: String,
                #[serde(default)]
                dry_run: bool,
            }
            let body: Body = body_json(request)?;
            let age = crate::parse_duration(&body.older_than)
                .ok_or_else(|| format!("'{}': give an age like 30m, 12h, 7d", body.older_than))?;
            let pruned = db::prune_tasks(conn, now - age, body.dry_run)?;
            let ids: Vec<i64> = pruned.iter().map(|t| t.id).collect();
            Response::ok(serde_json::json!({ "tasks": ids }))
        }
        ("POST", ["agents", "cleanup"]) => {
            #[derive(Deserialize, Default)]
            #[serde(deny_unknown_fields)]
            struct Body {
                #[serde(default)]
                dry_run: bool,
            }
            let body: Body = if request.body.is_empty() {
                Body::default()
            } else {
                body_json(request)?
            };
            let names = db::finished_made_agents(&db::list_agents(conn)?, now);
            if !body.dry_run {
                for name in &names {
                    db::delete_agent(conn, name)?;
                }
            }
            Response::ok(serde_json::json!({ "agents": names }))
        }
        ("GET", ["tools"]) => {
            let tools = crate::initialize_tools(true, None);
            let mut names: Vec<&String> = tools.keys().collect();
            names.sort();
            if request.query("schemas").is_none() {
                return Ok(Response::ok(names));
            }
            let schemas: Vec<serde_json::Value> = names
                .iter()
                .filter_map(|name| {
                    serde_json::from_str::<serde_json::Value>(&tools[*name].schema).ok()
                })
                .map(|schema| {
                    serde_json::json!({
                        "name": schema["function"]["name"],
                        "description": schema["function"]["description"],
                        "parameters": schema["function"]["parameters"],
                    })
                })
                .collect();
            Response::ok(schemas)
        }
        ("POST", ["tools", name, "run"]) => {
            // As a tool task: run, with the tools it has, by whichever
            // chat or worker's scheduler takes it, its result recorded.
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Body {
                #[serde(default)]
                arguments: serde_json::Map<String, serde_json::Value>,
                /// Whose conversation gets the call and its result.
                #[serde(default)]
                agent: Option<String>,
            }
            if !crate::initialize_tools(true, None).contains_key(*name) {
                return Ok(Response::error(404, &format!("no tool named '{}'", name)));
            }
            let body: Body = body_json(request)?;
            let id = db::create_task(
                conn,
                &db::NewTask {
                    name: format!("Run {}", name),
                    description: String::new(),
                    kind: db::TaskKind::TOOL.to_string(),
                    command: serde_json::json!({"tool": name, "arguments": body.arguments})
                        .to_string(),
                    agent_name: non_empty(&body.agent),
                    schedule: db::TaskSchedule::Once {
                        at: now.to_rfc3339(),
                    },
                    held: false,
                    depends_on: Vec::new(),
                    profile: None,
                    run_safe: false,
                    cwd: None,
                },
            )?;
            let mut reply = serde_json::json!({ "task_id": id });
            let anyone_running = db::list_agents(conn)?
                .iter()
                .any(|a| db::agent_session_is_live(a, now));
            if !anyone_running {
                reply["note"] = "No chat or worker is running: it runs once one is.".into();
            }
            Response::json(201, &reply)
        }

        ("POST", ["agents", name, "cwd"]) => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Body {
                cwd: Option<String>,
            }
            if db::get_agent(conn, name)?.is_none() {
                return Ok(Response::error(404, &format!("no agent named '{}'", name)));
            }
            let cwd = non_empty(&body_json::<Body>(request)?.cwd)
                .map(|dir| absolute_dir(&dir))
                .transpose()?;
            let mut config = db::get_agent_config(conn, name)?;
            config.cwd = cwd.clone();
            db::set_agent_config(conn, name, &config)?;
            let here = cwd
                .as_deref()
                .is_none_or(|c| std::path::Path::new(c).is_dir());
            Response::ok(serde_json::json!({ "cwd": cwd, "exists_here": here }))
        }
        ("GET", ["events"]) => {
            let number = |name: &str| -> Result<Option<i64>, Box<dyn Error>> {
                request
                    .query(name)
                    .filter(|v| !v.is_empty())
                    .map(|v| {
                        v.parse()
                            .map_err(|_| format!("bad '{}': {}", name, v).into())
                    })
                    .transpose()
            };
            let filter = EventFilter {
                agent: request
                    .query("agent")
                    .filter(|a| !a.is_empty())
                    .map(String::from),
                task_id: number("task")?,
                after: number("after")?,
                limit: number("limit")?
                    .map_or(200, |l| l.max(1) as usize)
                    .min(MAX_EVENTS),
            };
            Response::ok(db::agent_events(conn, &filter)?)
        }
        ("GET", ["tasks"]) => {
            let tasks = db::list_tasks(conn, None)?;
            let mut blocked = db::unmet_dependencies(&tasks);
            let tasks: Vec<ApiTask> = tasks
                .into_iter()
                .map(|task| ApiTask {
                    blocked_by: blocked.remove(&task.id).unwrap_or_default(),
                    task,
                })
                .collect();
            Response::ok(tasks)
        }
        ("GET", ["tasks", id]) => match db::get_task(conn, task_id(id)?)? {
            Some(task) => {
                let all = db::list_tasks(conn, None)?;
                let blocked_by = db::unmet_dependencies(&all)
                    .remove(&task.id)
                    .unwrap_or_default();
                Response::ok(ApiTask { task, blocked_by })
            }
            None => Response::error(404, &format!("no task #{}", id)),
        },
        ("POST", ["tasks"]) => {
            let task = body_json::<ApiNewTask>(request)?.into_new_task(now)?;
            let id = db::create_task(conn, &task)?;
            Response::json(201, &serde_json::json!({ "id": id }))
        }
        ("PATCH", ["tasks", id]) => {
            let id = task_id(id)?;
            let task = body_json::<ApiNewTask>(request)?.into_new_task(now)?;
            if !db::update_task(conn, id, &task)? {
                return match db::get_task(conn, id)? {
                    None => Ok(Response::error(404, &format!("no task #{}", id))),
                    Some(_) => Err(format!("task #{} is running: stop it to change it", id).into()),
                };
            }
            Response::ok(true)
        }
        ("DELETE", ["tasks", id]) => {
            let id = task_id(id)?;
            if !db::delete_task(conn, id)? {
                return Ok(Response::error(404, &format!("no task #{}", id)));
            }
            Response::ok(true)
        }
        ("POST", ["tasks", id, action @ ("hold" | "release")]) => {
            let id = task_id(id)?;
            if !db::set_task_held(conn, id, *action == "hold")? {
                return Err(format!(
                    "task #{} can't be {}: it isn't {}",
                    id,
                    if *action == "hold" {
                        "held"
                    } else {
                        "released"
                    },
                    if *action == "hold" {
                        "waiting to run"
                    } else {
                        "held"
                    }
                )
                .into());
            }
            Response::ok(true)
        }
        (
            "POST",
            [
                "tasks",
                id,
                action @ ("enable" | "disable" | "run" | "stop"),
            ],
        ) => {
            let id = task_id(id)?;
            let changed = match *action {
                "run" => db::run_task_now(conn, id)?,
                "stop" => db::request_task_stop(conn, id)?,
                action => db::set_task_enabled(conn, id, action == "enable")?,
            };
            if !changed {
                let status = db::get_task(conn, id)?
                    .map(|t| t.status)
                    .ok_or_else(|| format!("no task #{}", id))?;
                let done = match *action {
                    "stop" => "stopped".to_string(),
                    "run" => "run".to_string(),
                    action => format!("{}d", action),
                };
                return Err(format!("task #{} can't be {}: it's {}", id, done, status).into());
            }
            Response::ok(true)
        }
        ("POST", ["tasks", id, "assign"]) => {
            let id = task_id(id)?;
            let target: ApiTarget = body_json(request)?;
            let (agent, profile) = (non_empty(&target.agent), non_empty(&target.profile));
            if let Some(agent) = &agent {
                if db::get_agent(conn, agent)?.is_none() {
                    return Err(format!("no agent named '{}'", agent).into());
                }
            }
            if !db::set_task_target(conn, id, agent.as_deref(), profile.as_deref())? {
                return Err(format!("task #{} has already been picked up, or is gone", id).into());
            }
            Response::ok(true)
        }
        ("GET", ["profiles"]) => {
            let mut listed: Vec<serde_json::Value> = profiles
                .iter()
                .map(|(name, profile)| {
                    serde_json::json!({"name": name, "source": "config", "settings": profile})
                })
                .collect();
            for (name, settings) in db::list_profiles(conn)? {
                if !profiles.contains_key(&name) {
                    listed.push(
                        serde_json::json!({"name": name, "source": "database", "settings": settings}),
                    );
                }
            }
            listed.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
            Response::ok(listed)
        }
        ("POST", ["profiles"]) => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Body {
                name: String,
                settings: serde_json::Value,
            }
            let body: Body = body_json(request)?;
            let name = body.name.trim().to_string();
            if profiles.contains_key(&name) {
                return Err(format!(
                    "profile '{}' is in faber serve's config file: change it there",
                    name
                )
                .into());
            }
            let profile: crate::Profile = serde_json::from_value(body.settings.clone())
                .map_err(|e| format!("bad profile settings: {}", e))?;
            crate::validate_profiles(&crate::Profiles::from([(name.clone(), profile)]))?;
            db::set_profile(conn, &name, &body.settings)?;
            Response::ok(serde_json::json!({ "name": name }))
        }
        ("DELETE", ["profiles", name]) => {
            if profiles.contains_key(*name) {
                return Err(format!(
                    "profile '{}' is in faber serve's config file: remove it there",
                    name
                )
                .into());
            }
            if !db::delete_profile(conn, name)? {
                return Ok(Response::error(404, &format!("no profile '{}'", name)));
            }
            Response::ok(true)
        }
        ("GET", ["info"]) => Response::ok(serde_json::json!({
            "cwd": std::env::current_dir().ok(),
        })),
        (
            _,
            [
                "agents" | "events" | "tasks" | "profiles" | "info" | "tools",
                ..,
            ],
        ) => Response::error(405, "method not allowed"),
        _ => Response::error(404, "not found"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faber::agent_io::AgentEvent;

    fn test_db() -> Arc<Mutex<Connection>> {
        let conn = Connection::open_in_memory().unwrap();
        db::initialize_db(&conn).unwrap();
        Arc::new(Mutex::new(conn))
    }

    fn request(method: &str, target: &str, body: &str) -> Request {
        let raw = format!(
            "{} {} HTTP/1.1\r\nHost: localhost:9090\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{}",
            method,
            target,
            body.len(),
            body
        );
        read_request(&mut raw.as_bytes()).unwrap()
    }

    fn call(db: &Arc<Mutex<Connection>>, method: &str, target: &str, body: &str) -> Response {
        let profiles = crate::Profiles::from([(
            "fast".to_string(),
            serde_json::from_value(serde_json::json!({"description": "quick", "model": "small"}))
                .unwrap(),
        )]);
        route(&request(method, target, body), db, None, &profiles)
    }

    fn json(response: &Response) -> serde_json::Value {
        serde_json::from_slice(&response.body).unwrap()
    }

    #[test]
    fn test_read_request_and_query() {
        let r = request("GET", "/api/events?agent=a%20b&after=3&x=1+2", "");
        assert_eq!((r.method.as_str(), r.path.as_str()), ("GET", "/api/events"));
        assert_eq!(r.query("agent"), Some("a b"));
        assert_eq!(r.query("after"), Some("3"));
        assert_eq!(r.query("x"), Some("1 2"));
        let r = request("POST", "/api/tasks", r#"{"command":"x"}"#);
        assert_eq!(r.body, br#"{"command":"x"}"#);
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%4"), "%4");
        assert_eq!(percent_decode("%zz"), "%zz");
    }

    #[test]
    fn test_api_refuses_requests_a_foreign_page_could_send() {
        let db = test_db();
        let mut r = request("POST", "/api/tasks", r#"{"command": "rm -rf"}"#);
        r.content_type = Some("text/plain".to_string());
        assert_eq!(route(&r, &db, None, &crate::Profiles::new()).status, 415);
        r.content_type = None;
        assert_eq!(route(&r, &db, None, &crate::Profiles::new()).status, 415);
        let mut r = request("GET", "/api/tasks", "");
        r.host = Some("evil.example:9090".to_string());
        assert_eq!(route(&r, &db, None, &crate::Profiles::new()).status, 403);
        // With a key, any host will do.
        r.authorization = Some("Bearer k".to_string());
        assert_eq!(
            route(&r, &db, Some("k"), &crate::Profiles::new()).status,
            200
        );
        assert!(
            db::list_tasks(&db.lock().unwrap(), None)
                .unwrap()
                .is_empty()
        );
        for host in [
            "localhost",
            "LOCALHOST:1",
            "127.0.0.1:9090",
            "[::1]:9090",
            "10.0.0.2",
        ] {
            assert!(is_local_host(Some(host)), "{}", host);
        }
        for host in ["evil.example", "localhost.evil.example:80", "[nope]:1", ""] {
            assert!(!is_local_host(Some(host)), "{}", host);
        }
        assert!(!is_local_host(None));
    }

    #[test]
    fn test_ui_is_served_and_root_redirects() {
        let db = test_db();
        assert_eq!(call(&db, "GET", "/", "").status, 302);
        assert_eq!(call(&db, "GET", "/ui", "").status, 302);
        let page = call(&db, "GET", "/ui/", "");
        assert_eq!(page.status, 200);
        assert!(page.content_type.starts_with("text/html"));
        assert_eq!(call(&db, "GET", "/ui/app.js", "").status, 200);
        assert_eq!(call(&db, "GET", "/nope", "").status, 404);
    }

    #[test]
    fn test_auth_key_guards_the_api_only() {
        let db = test_db();
        let r = request("GET", "/api/tasks", "");
        assert_eq!(
            route(&r, &db, Some("k"), &crate::Profiles::new()).status,
            401
        );
        assert_eq!(
            route(
                &request("GET", "/ui/", ""),
                &db,
                Some("k"),
                &crate::Profiles::new()
            )
            .status,
            200
        );
        let mut r = request("GET", "/api/tasks", "");
        r.authorization = Some("Bearer k".to_string());
        assert_eq!(
            route(&r, &db, Some("k"), &crate::Profiles::new()).status,
            200
        );
    }

    #[test]
    fn test_create_and_manage_tasks() {
        let db = test_db();
        // No target: any agent may take it.
        let r = call(
            &db,
            "POST",
            "/api/tasks",
            r#"{"command": "Fix the bug\nin x"}"#,
        );
        assert_eq!(r.status, 201, "{:?}", json(&r));
        let id = json(&r)["id"].as_i64().unwrap();
        let task = json(&call(&db, "GET", &format!("/api/tasks/{}", id), ""));
        assert_eq!(task["name"], "Fix the bug");
        assert_eq!(task["kind"], "prompt");
        assert!(task["agent_name"].is_null() && task["profile"].is_null());

        let r = call(
            &db,
            "POST",
            "/api/tasks",
            &format!(
                r#"{{"command": "then this", "profile": "fast", "depends_on": [{}], "hold": true}}"#,
                id
            ),
        );
        let second = json(&r)["id"].as_i64().unwrap();
        let tasks = json(&call(&db, "GET", "/api/tasks", ""));
        let listed = tasks
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["id"] == second)
            .unwrap();
        assert_eq!(listed["blocked_by"], serde_json::json!([id]));
        assert_eq!(listed["status"], "held");
        assert_eq!(listed["profile"], "fast");

        let path = format!("/api/tasks/{}", second);
        assert_eq!(
            call(&db, "POST", &format!("{}/release", path), "").status,
            200
        );
        assert_eq!(
            call(&db, "POST", &format!("{}/release", path), "").status,
            400
        );
        assert_eq!(call(&db, "POST", &format!("{}/hold", path), "").status, 200);
        assert_eq!(
            call(&db, "POST", &format!("{}/disable", path), "").status,
            200
        );
        assert_eq!(json(&call(&db, "GET", &path, ""))["status"], "disabled");
        assert_eq!(
            call(&db, "POST", &format!("{}/disable", path), "").status,
            400
        );
        assert_eq!(
            call(&db, "POST", &format!("{}/enable", path), "").status,
            200
        );
        assert_eq!(json(&call(&db, "GET", &path, ""))["status"], "scheduled");
        assert_eq!(call(&db, "POST", &format!("{}/hold", path), "").status, 200);
        assert_eq!(call(&db, "POST", &format!("{}/run", path), "").status, 200);
        assert_eq!(json(&call(&db, "GET", &path, ""))["status"], "scheduled");
        assert_eq!(call(&db, "POST", &format!("{}/hold", path), "").status, 200);
        db::create_agent(&db.lock().unwrap(), "bob", "").unwrap();
        let r = call(
            &db,
            "POST",
            &format!("{}/assign", path),
            r#"{"agent": "bob"}"#,
        );
        assert_eq!(r.status, 200, "{:?}", json(&r));
        let task = json(&call(&db, "GET", &path, ""));
        assert_eq!(
            (task["agent_name"].as_str(), task["profile"].as_str()),
            (Some("bob"), None)
        );
        assert_eq!(call(&db, "DELETE", &path, "").status, 200);
        assert_eq!(call(&db, "GET", &path, "").status, 404);
    }

    #[test]
    fn test_a_task_can_run_in_its_own_directory() {
        let db = test_db();
        let tmp = std::env::temp_dir();
        let body = serde_json::json!({"command": "count lines", "cwd": tmp}).to_string();
        let id = json(&call(&db, "POST", "/api/tasks", &body))["id"]
            .as_i64()
            .unwrap();
        let task = json(&call(&db, "GET", &format!("/api/tasks/{}", id), ""));
        assert_eq!(
            task["cwd"].as_str().map(std::path::PathBuf::from),
            Some(tmp.canonicalize().unwrap())
        );
        let r = call(
            &db,
            "POST",
            "/api/tasks",
            r#"{"command": "x", "cwd": "rel/dir"}"#,
        );
        assert_eq!(r.status, 400);
        assert!(json(&call(&db, "GET", "/api/info", ""))["cwd"].is_string());
    }

    #[test]
    fn test_tasks_can_be_changed_until_they_run() {
        let db = test_db();
        let r = call(
            &db,
            "POST",
            "/api/tasks",
            r#"{"command": "first", "hold": true}"#,
        );
        let id = json(&r)["id"].as_i64().unwrap();
        let other = json(&call(&db, "POST", "/api/tasks", r#"{"command": "other"}"#))["id"]
            .as_i64()
            .unwrap();
        let path = format!("/api/tasks/{}", id);
        let body = serde_json::json!({
            "command": "second", "name": "renamed", "cron": "0 0 9 * * * *",
            "depends_on": [other], "cwd": std::env::temp_dir(),
        });
        let r = call(&db, "PATCH", &path, &body.to_string());
        assert_eq!(r.status, 200, "{:?}", json(&r));
        let task = json(&call(&db, "GET", &path, ""));
        assert_eq!(task["command"], "second");
        assert_eq!(task["name"], "renamed");
        assert_eq!(task["task_type"], "cron");
        assert_eq!(task["depends_on"], serde_json::json!([other]));
        assert_eq!(task["status"], "held", "its state stays");
        // A cycle, through the other task, is refused.
        let cycle = serde_json::json!({"command": "other", "depends_on": [id]});
        let r = call(
            &db,
            "PATCH",
            &format!("/api/tasks/{}", other),
            &cycle.to_string(),
        );
        assert_eq!(r.status, 400);
        assert!(json(&r)["error"].as_str().unwrap().contains("itself"));
        // Running, it can't be changed.
        {
            let conn = db.lock().unwrap();
            db::create_agent(&conn, "w", "").unwrap();
            assert!(db::claim_agent(&conn, "w", "s").unwrap());
            assert!(db::claim_task(&conn, other, "s").unwrap());
        }
        let r = call(
            &db,
            "PATCH",
            &format!("/api/tasks/{}", other),
            r#"{"command": "x"}"#,
        );
        assert!(json(&r)["error"].as_str().unwrap().contains("running"));
        assert_eq!(
            call(&db, "PATCH", "/api/tasks/999", r#"{"command": "x"}"#).status,
            404
        );
    }

    #[test]
    fn test_bad_tasks_are_refused() {
        let db = test_db();
        for body in [
            r#"{"command": "  "}"#,
            r#"{"command": "x", "agent": "nobody"}"#,
            r#"{"command": "x", "tool": true}"#,
            r#"{"command": "x", "cron": "not cron"}"#,
            r#"{"command": "x", "depends_on": [42]}"#,
            r#"{"command": "x", "bogus": 1}"#,
        ] {
            let r = call(&db, "POST", "/api/tasks", body);
            assert_eq!(r.status, 400, "{}", body);
            assert!(json(&r)["error"].is_string());
        }
        let r = call(
            &db,
            "POST",
            "/api/tasks",
            r#"{"command": "{\"tool\": \"glob\", \"arguments\": {}}", "tool": true}"#,
        );
        assert_eq!(r.status, 201);
        let id = json(&r)["id"].as_i64().unwrap();
        assert_eq!(
            json(&call(&db, "GET", &format!("/api/tasks/{}", id), ""))["name"],
            "glob"
        );
    }

    #[test]
    fn test_the_user_messages_agents_and_reads_their_conversation() {
        let db = test_db();
        {
            let conn = db.lock().unwrap();
            db::create_agent(&conn, "boss", "").unwrap();
            db::save_agent_messages(
                &conn,
                "boss",
                &[
                    serde_json::json!({"role": "user", "content": "hi"}),
                    serde_json::json!({"role": "assistant", "content": "hello"}),
                ],
            )
            .unwrap();
        }
        let r = call(
            &db,
            "POST",
            "/api/agents/boss/messages",
            r#"{"text": "count the files"}"#,
        );
        assert_eq!(r.status, 201, "{:?}", json(&r));
        assert!(
            json(&r)["note"]
                .as_str()
                .unwrap()
                .contains("Nothing is running")
        );
        let id = json(&r)["task_id"].as_i64().unwrap();
        let task = json(&call(&db, "GET", &format!("/api/tasks/{}", id), ""));
        assert_eq!(task["agent_name"], "boss");
        assert_eq!(task["command"], "count the files");
        assert_eq!(
            call(&db, "POST", "/api/agents/boss/messages", r#"{"text": " "}"#).status,
            400
        );
        assert_eq!(
            call(
                &db,
                "POST",
                "/api/agents/nobody/messages",
                r#"{"text": "x"}"#
            )
            .status,
            404
        );
        let convo = json(&call(&db, "GET", "/api/agents/boss/messages?limit=1", ""));
        assert_eq!(convo["total"], 2);
        assert_eq!(convo["messages"][0]["content"], "hello");
    }

    #[test]
    fn test_profiles_are_kept_in_the_database_and_applied_to_agents() {
        let db = test_db();
        // The config file's, read-only here.
        let listed = json(&call(&db, "GET", "/api/profiles", ""));
        assert_eq!(listed[0]["name"], "fast");
        assert_eq!(listed[0]["source"], "config");
        let r = call(
            &db,
            "POST",
            "/api/profiles",
            r#"{"name": "fast", "settings": {}}"#,
        );
        assert!(json(&r)["error"].as_str().unwrap().contains("config file"));
        // The database's: made, listed, applied, removed.
        let deep = r#"{"name": "deep", "settings": {"model": "big", "system_prompt": "think", "tools": ["read_file"]}}"#;
        assert_eq!(call(&db, "POST", "/api/profiles", deep).status, 200);
        let listed = json(&call(&db, "GET", "/api/profiles", ""));
        assert_eq!(listed[0]["name"], "deep");
        assert_eq!(listed[0]["source"], "database");
        assert_eq!(listed[0]["settings"]["model"], "big");
        for bad in [
            r#"{"name": "x y", "settings": {}}"#,
            r#"{"name": "x", "settings": {"tools": ["nope"]}}"#,
            r#"{"name": "x", "settings": {"bogus": 1}}"#,
        ] {
            assert_eq!(call(&db, "POST", "/api/profiles", bad).status, 400, "{bad}");
        }
        {
            let conn = db.lock().unwrap();
            db::create_agent(&conn, "a", "").unwrap();
        }
        let r = call(
            &db,
            "PATCH",
            "/api/agents/a/config",
            r#"{"unsafe_tools": true, "cwd": "/tmp"}"#,
        );
        assert_eq!(r.status, 200);
        // A profile's settings, then what else is given; its unsafe tools
        // and directory stay the agent's.
        let r = call(
            &db,
            "PATCH",
            "/api/agents/a/config",
            r#"{"profile": "deep", "max_tokens": 99}"#,
        );
        let config = json(&r);
        assert_eq!(config["model"], "big");
        assert_eq!(config["profile"], "deep");
        assert_eq!(config["max_tokens"], 99);
        assert_eq!(config["unsafe_tools"], true);
        assert!(config["cwd"].as_str().is_some());
        let r = call(
            &db,
            "PATCH",
            "/api/agents/a/config",
            r#"{"profile": "fast"}"#,
        );
        assert_eq!(json(&r)["model"], "small", "the config file's too");
        let r = call(
            &db,
            "PATCH",
            "/api/agents/a/config",
            r#"{"profile": "nope"}"#,
        );
        assert_eq!(r.status, 400);
        assert_eq!(call(&db, "DELETE", "/api/profiles/deep", "").status, 200);
        assert_eq!(call(&db, "DELETE", "/api/profiles/deep", "").status, 404);
        assert_eq!(call(&db, "DELETE", "/api/profiles/fast", "").status, 400);
    }

    #[test]
    fn test_cleaning_up_finished_tasks_and_agents() {
        let db = test_db();
        {
            let conn = db.lock().unwrap();
            for name in ["mine", "boss", "done-helper", "busy-helper", "deep"] {
                db::create_agent(&conn, name, "").unwrap();
            }
            db::set_agent_parent(&conn, "done-helper", Some("boss")).unwrap();
            db::set_agent_parent(&conn, "busy-helper", Some("done-helper")).unwrap();
            db::set_agent_parent(&conn, "deep", Some("boss")).unwrap();
            // A sub-agent of a sub-agent is still at work.
            assert!(db::claim_agent(&conn, "busy-helper", "s").unwrap());
        }
        let r = call(&db, "POST", "/api/agents/cleanup", r#"{"dry_run": true}"#);
        assert_eq!(json(&r)["agents"], serde_json::json!(["deep"]));
        assert!(
            db::get_agent(&db.lock().unwrap(), "deep")
                .unwrap()
                .is_some(),
            "only looked"
        );
        call(&db, "POST", "/api/agents/cleanup", "");
        let names: Vec<String> = db::list_agents(&db.lock().unwrap())
            .unwrap()
            .into_iter()
            .map(|a| a.name)
            .collect();
        assert_eq!(names, ["boss", "busy-helper", "done-helper", "mine"]);

        let old = (chrono::Utc::now() - chrono::Duration::days(10)).to_rfc3339();
        let id = {
            let conn = db.lock().unwrap();
            let id = db::create_task(
                &conn,
                &db::NewTask {
                    name: "t".to_string(),
                    description: String::new(),
                    kind: db::TaskKind::PROMPT.to_string(),
                    command: "x".to_string(),
                    agent_name: None,
                    schedule: db::TaskSchedule::Once { at: old.clone() },
                    held: false,
                    depends_on: Vec::new(),
                    profile: None,
                    run_safe: false,
                    cwd: None,
                },
            )
            .unwrap();
            conn.execute(
                "UPDATE scheduled_tasks SET status = 'done', last_run_at = ?2 WHERE id = ?1",
                rusqlite::params![id, old],
            )
            .unwrap();
            id
        };
        let r = call(&db, "POST", "/api/tasks/prune", r#"{"older_than": "30d"}"#);
        assert_eq!(json(&r)["tasks"], serde_json::json!([]));
        let r = call(&db, "POST", "/api/tasks/prune", r#"{"older_than": "7d"}"#);
        assert_eq!(json(&r)["tasks"], serde_json::json!([id]));
        assert_eq!(
            call(&db, "POST", "/api/tasks/prune", r#"{"older_than": "soon"}"#).status,
            400
        );
    }

    #[test]
    fn test_tools_can_be_listed_and_run() {
        let db = test_db();
        let schemas = json(&call(&db, "GET", "/api/tools?schemas=1", ""));
        let kb = schemas
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "kb_search")
            .unwrap();
        assert!(kb["parameters"]["properties"]["query"].is_object());
        let r = call(
            &db,
            "POST",
            "/api/tools/kb_search/run",
            r#"{"arguments": {"query": "deploy"}}"#,
        );
        assert_eq!(r.status, 201, "{:?}", json(&r));
        assert!(
            json(&r)["note"]
                .as_str()
                .unwrap()
                .contains("No chat or worker")
        );
        let id = json(&r)["task_id"].as_i64().unwrap();
        let task = json(&call(&db, "GET", &format!("/api/tasks/{}", id), ""));
        assert_eq!(task["kind"], "tool");
        let command: serde_json::Value =
            serde_json::from_str(task["command"].as_str().unwrap()).unwrap();
        assert_eq!(
            command,
            serde_json::json!({"tool": "kb_search", "arguments": {"query": "deploy"}})
        );
        assert_eq!(call(&db, "POST", "/api/tools/nope/run", "{}").status, 404);
    }

    #[test]
    fn test_agents_are_stopped_and_cleared() {
        let db = test_db();
        {
            let conn = db.lock().unwrap();
            db::create_agent(&conn, "a", "").unwrap();
            db::save_agent_messages(
                &conn,
                "a",
                &[serde_json::json!({"role": "user", "content": "hi"})],
            )
            .unwrap();
        }
        let r = call(&db, "POST", "/api/agents/a/stop", "");
        assert!(
            json(&r)["error"]
                .as_str()
                .unwrap()
                .contains("nothing is running")
        );
        assert_eq!(call(&db, "POST", "/api/agents/a/clear", "").status, 200);
        assert_eq!(
            db::agent_message_count(&db.lock().unwrap(), "a").unwrap(),
            0
        );
        assert!(db::claim_agent(&db.lock().unwrap(), "a", "s").unwrap());
        assert_eq!(call(&db, "POST", "/api/agents/a/stop", "").status, 200);
        assert!(
            db::get_agent_data(&db.lock().unwrap(), "a", db::AGENT_STOP_KEY)
                .unwrap()
                .is_some()
        );
        let r = call(&db, "POST", "/api/agents/a/clear", "");
        assert!(json(&r)["error"].as_str().unwrap().contains("is running"));
    }

    #[test]
    fn test_agents_are_made_and_configured_by_the_user() {
        let db = test_db();
        db::create_agent(&db.lock().unwrap(), "boss", "").unwrap();
        // Settings, unsafe tools and all, are the user's to change here.
        let r = call(
            &db,
            "PATCH",
            "/api/agents/boss/config",
            r#"{"model": "qwen", "unsafe_tools": true, "max_tokens": 512, "tools": ["read_file", "glob"]}"#,
        );
        assert_eq!(r.status, 200, "{:?}", json(&r));
        let config = &json(&call(&db, "GET", "/api/agents/boss", ""))["config"];
        assert_eq!(config["model"], "qwen");
        assert_eq!(config["unsafe_tools"], true);
        assert_eq!(config["tools"], serde_json::json!(["read_file", "glob"]));
        let r = call(
            &db,
            "PATCH",
            "/api/agents/boss/config",
            r#"{"model": null, "tools": null, "description": "runs things"}"#,
        );
        let boss = json(&call(&db, "GET", "/api/agents/boss", ""));
        assert_eq!(boss["agent"]["description"], "runs things");
        assert_eq!(json(&r)["model"], serde_json::Value::Null);
        assert_eq!(json(&r)["max_tokens"], 512, "left alone");
        for bad in [
            r#"{"tools": ["nope"]}"#,
            r#"{"max_tokens": -1}"#,
            r#"{"cwd": "relative"}"#,
            r#"{"api_key": "/k"}"#,
        ] {
            let r = call(&db, "PATCH", "/api/agents/boss/config", bad);
            assert_eq!(r.status, 400, "{bad}");
        }
        let r = call(
            &db,
            "POST",
            "/api/agents",
            r#"{"name": "reviewer", "description": "reviews", "config": {"model": "big"}}"#,
        );
        assert_eq!(r.status, 201, "{:?}", json(&r));
        let reviewer = json(&call(&db, "GET", "/api/agents/reviewer", ""));
        assert_eq!(reviewer["config"]["model"], "big");
        let again = r#"{"name": "reviewer"}"#;
        assert_eq!(call(&db, "POST", "/api/agents", again).status, 400);
        assert_eq!(
            call(&db, "POST", "/api/agents", r#"{"name": "a b"}"#).status,
            400
        );
        assert_eq!(call(&db, "DELETE", "/api/agents/reviewer", "").status, 200);
        assert_eq!(call(&db, "GET", "/api/agents/reviewer", "").status, 404);
        let tools = json(&call(&db, "GET", "/api/tools", ""));
        assert!(tools.as_array().unwrap().iter().any(|t| t == "run_command"));
    }

    #[test]
    fn test_agents_and_events() {
        let db = test_db();
        {
            let conn = db.lock().unwrap();
            db::create_agent(&conn, "boss", "").unwrap();
            db::create_agent(&conn, "helper", "").unwrap();
            db::set_agent_parent(&conn, "helper", Some("boss")).unwrap();
            db::append_agent_events(
                &conn,
                "helper",
                Some(7),
                &[
                    AgentEvent::Input {
                        text: "go".to_string(),
                    },
                    AgentEvent::Text {
                        text: "done".to_string(),
                    },
                ],
            )
            .unwrap();
        }
        let agents = json(&call(&db, "GET", "/api/agents", ""));
        assert_eq!(agents.as_array().unwrap().len(), 2);
        assert_eq!(agents[1]["parent"], "boss");
        assert_eq!(agents[1]["live"], false);
        let boss = json(&call(&db, "GET", "/api/agents/boss", ""));
        assert_eq!(boss["children"], serde_json::json!(["helper"]));
        assert!(boss["agent"]["cwd"].is_null());
        assert!(boss["agent"]["plan"].is_null());
        db::set_agent_data(
            &db.lock().unwrap(),
            "boss",
            crate::PLAN_DATA_KEY,
            r#"[{"content": "read", "status": "completed"}, {"content": "write", "status": "in_progress"}]"#,
        )
        .unwrap();
        let agents = json(&call(&db, "GET", "/api/agents", ""));
        assert_eq!(agents[0]["plan"][1]["status"], "in_progress");
        let tmp = std::env::temp_dir();
        let r = call(
            &db,
            "POST",
            "/api/agents/boss/cwd",
            &serde_json::json!({ "cwd": tmp }).to_string(),
        );
        assert_eq!(json(&r)["exists_here"], true, "{:?}", json(&r));
        let boss = json(&call(&db, "GET", "/api/agents/boss", ""));
        assert_eq!(
            boss["agent"]["cwd"].as_str().map(std::path::PathBuf::from),
            Some(tmp.canonicalize().unwrap())
        );
        let r = call(
            &db,
            "POST",
            "/api/agents/boss/cwd",
            r#"{"cwd": "relative"}"#,
        );
        assert_eq!(r.status, 400);
        call(&db, "POST", "/api/agents/boss/cwd", r#"{"cwd": null}"#);
        assert!(json(&call(&db, "GET", "/api/agents/boss", ""))["agent"]["cwd"].is_null());
        assert_eq!(call(&db, "GET", "/api/agents/nobody", "").status, 404);

        let events = json(&call(&db, "GET", "/api/events?task=7", ""));
        assert_eq!(events.as_array().unwrap().len(), 2);
        assert_eq!(events[0]["event"]["type"], "input");
        let first = events[0]["id"].as_i64().unwrap();
        let later = json(&call(
            &db,
            "GET",
            &format!("/api/events?agent=helper&after={}", first),
            "",
        ));
        assert_eq!(later.as_array().unwrap().len(), 1);
        assert_eq!(later[0]["event"]["text"], "done");
        assert_eq!(call(&db, "GET", "/api/events?after=x", "").status, 400);
        assert_eq!(
            json(&call(&db, "GET", "/api/profiles", ""))[0]["name"],
            "fast"
        );
        assert_eq!(call(&db, "PUT", "/api/tasks", "").status, 405);
    }
}
