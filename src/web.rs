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
//! - `GET /api/agents`, `GET /api/agents/<name>`,
//!   `POST /api/agents/<name>/cwd` (where it works: the user's to set)
//! - `GET /api/events?agent=&task=&after=&limit=`
//! - `GET /api/tasks`, `GET /api/tasks/<id>`, `POST /api/tasks`,
//!   `DELETE /api/tasks/<id>`,
//!   `POST /api/tasks/<id>/hold|release|enable|disable|run`,
//!   `POST /api/tasks/<id>/assign`
//! - `GET /api/profiles`
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

/// A profile agents can be made from, as `faber serve`'s config has it.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct UiProfile {
    pub name: String,
    pub description: Option<String>,
    pub model: Option<String>,
}

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
    profiles: &[UiProfile],
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
    profiles: &[UiProfile],
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
    /// Given the unsafe tools for good (`faber agents set --unsafe`, or by
    /// an agent that has them). A session's own agent may have them too,
    /// for the session, if started with --unsafe-tools.
    unsafe_tools: bool,
    /// Where it works, if not where whatever runs it does.
    cwd: Option<String>,
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
    profiles: &[UiProfile],
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
                    model: config.model,
                    profile: config.profile,
                    unsafe_tools: config.unsafe_tools == Some(true),
                    cwd: config.cwd.clone(),
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
                    unsafe_tools: config.unsafe_tools == Some(true),
                    cwd: config.cwd.clone(),
                    agent,
                },
                "children": children,
                "system_prompt": config.system_prompt,
                "tools": config.tools,
                "messages": db::agent_message_count(conn, name)?,
            }))
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
            let cwd = match body_json::<Body>(request)?.cwd.as_deref().map(str::trim) {
                None | Some("") => None,
                Some(path) => {
                    let path = std::path::Path::new(path);
                    if !path.is_absolute() {
                        return Err("give an absolute path".into());
                    }
                    // The agent may run on another machine than this
                    // server: only resolved if it's here.
                    Some(
                        path.canonicalize()
                            .unwrap_or_else(|_| path.to_path_buf())
                            .to_string_lossy()
                            .into_owned(),
                    )
                }
            };
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
        ("POST", ["tasks", id, action @ ("enable" | "disable" | "run")]) => {
            let id = task_id(id)?;
            let changed = match *action {
                "run" => db::run_task_now(conn, id)?,
                action => db::set_task_enabled(conn, id, action == "enable")?,
            };
            if !changed {
                let status = db::get_task(conn, id)?
                    .map(|t| t.status)
                    .ok_or_else(|| format!("no task #{}", id))?;
                return Err(format!("task #{} can't be {}d: it's {}", id, action, status).into());
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
        ("GET", ["profiles"]) => Response::ok(profiles),
        (_, ["agents" | "events" | "tasks" | "profiles", ..]) => {
            Response::error(405, "method not allowed")
        }
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
        let profiles = [UiProfile {
            name: "fast".to_string(),
            description: Some("quick".to_string()),
            model: None,
        }];
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
        assert_eq!(route(&r, &db, None, &[]).status, 415);
        r.content_type = None;
        assert_eq!(route(&r, &db, None, &[]).status, 415);
        let mut r = request("GET", "/api/tasks", "");
        r.host = Some("evil.example:9090".to_string());
        assert_eq!(route(&r, &db, None, &[]).status, 403);
        // With a key, any host will do.
        r.authorization = Some("Bearer k".to_string());
        assert_eq!(route(&r, &db, Some("k"), &[]).status, 200);
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
        assert_eq!(route(&r, &db, Some("k"), &[]).status, 401);
        assert_eq!(
            route(&request("GET", "/ui/", ""), &db, Some("k"), &[]).status,
            200
        );
        let mut r = request("GET", "/api/tasks", "");
        r.authorization = Some("Bearer k".to_string());
        assert_eq!(route(&r, &db, Some("k"), &[]).status, 200);
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
