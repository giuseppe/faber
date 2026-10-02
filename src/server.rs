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

use faber::db;
use faber::protocol::{RpcRequest, RpcResponse};
use log::{info, warn};
use rusqlite::Connection;
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

pub fn serve_command(
    bind: &str,
    db_path: &str,
    auth_key: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let conn = Connection::open(db_path)?;
    db::initialize_db(&conn)?;
    let db = Arc::new(Mutex::new(conn));

    let listener = TcpListener::bind(bind)?;
    info!("Listening on {}", bind);
    println!("faber serve listening on {}", bind);

    for stream in listener.incoming() {
        let stream = stream?;
        let peer = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "unknown".into());
        info!("Client connected: {}", peer);
        let db = db.clone();
        let auth_key = auth_key.map(|s| s.to_string());
        std::thread::spawn(move || {
            if let Err(e) = handle_client(stream, db, auth_key.as_deref()) {
                warn!("Client {} error: {}", peer, e);
            }
            info!("Client {} disconnected", peer);
        });
    }
    Ok(())
}

fn handle_client(
    stream: TcpStream,
    db: Arc<Mutex<Connection>>,
    auth_key: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Every message is a small request/response exchange; don't let
    // Nagle's algorithm hold any of them back.
    stream.set_nodelay(true)?;
    let reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut authenticated = auth_key.is_none();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let request: RpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp =
                    RpcResponse::error(request_id_hint(&line), format!("invalid request: {}", e));
                send_response(&mut writer, &resp)?;
                continue;
            }
        };

        if !authenticated {
            let resp = if request.method == "auth" {
                let key = request.params["key"].as_str().unwrap_or("");
                if key == auth_key.unwrap_or("") {
                    authenticated = true;
                    RpcResponse::success(request.id, serde_json::json!(true))
                } else {
                    let r = RpcResponse::error(Some(request.id), "unauthorized".into());
                    send_response(&mut writer, &r)?;
                    return Ok(());
                }
            } else {
                let r = RpcResponse::error(Some(request.id), "auth required".into());
                send_response(&mut writer, &r)?;
                return Ok(());
            };
            send_response(&mut writer, &resp)?;
            continue;
        }

        let response = dispatch(&db, &request);
        send_response(&mut writer, &response)?;
    }
    Ok(())
}

/// The `id` of a line that isn't a valid request, if it's JSON with a
/// numeric `id` at all - so the client can still tell which request was
/// rejected. `None` (sent as `null`) otherwise.
fn request_id_hint(line: &str) -> Option<u64> {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()?
        .get("id")?
        .as_u64()
}

/// Sends `resp` as one line, in a single write: with the newline written
/// separately, Nagle's algorithm would hold it back until the client's
/// delayed ACK, adding ~40ms to every call.
fn send_response(writer: &mut TcpStream, resp: &RpcResponse) -> Result<(), Box<dyn Error>> {
    let mut line = serde_json::to_string(resp)?;
    line.push('\n');
    writer.write_all(line.as_bytes())?;
    writer.flush()?;
    Ok(())
}

fn dispatch(db: &Arc<Mutex<Connection>>, req: &RpcRequest) -> RpcResponse {
    match dispatch_inner(db, req) {
        Ok(value) => RpcResponse::success(req.id, value),
        Err(e) => RpcResponse::error(Some(req.id), e.to_string()),
    }
}

fn dispatch_inner(
    db: &Arc<Mutex<Connection>>,
    req: &RpcRequest,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let p = &req.params;
    macro_rules! str_param {
        ($name:expr) => {
            p[$name]
                .as_str()
                .ok_or_else(|| format!("missing param '{}'", $name))?
        };
    }
    macro_rules! opt_str_param {
        ($name:expr) => {
            p[$name].as_str()
        };
    }
    macro_rules! i64_param {
        ($name:expr) => {
            p[$name]
                .as_i64()
                .ok_or_else(|| format!("missing param '{}'", $name))?
        };
    }

    let conn = db.lock().map_err(|e| format!("DB lock: {}", e))?;

    match req.method.as_str() {
        "claim_agent" => {
            let v = db::claim_agent(&conn, str_param!("name"), str_param!("session_id"))?;
            Ok(serde_json::to_value(v)?)
        }
        "heartbeat_all" => {
            db::heartbeat_all(&conn, str_param!("session_id"))?;
            Ok(serde_json::json!(true))
        }
        "release_agent" => {
            db::release_agent(&conn, str_param!("name"), str_param!("session_id"))?;
            Ok(serde_json::json!(true))
        }
        "release_all_agents" => {
            db::release_all_agents(&conn, str_param!("session_id"))?;
            Ok(serde_json::json!(true))
        }
        "create_agent" => {
            db::create_agent(&conn, str_param!("name"), str_param!("description"))?;
            Ok(serde_json::json!(true))
        }
        "set_agent_activity" => {
            db::set_agent_activity(&conn, str_param!("agent"), str_param!("activity"))?;
            Ok(serde_json::json!(true))
        }
        "agent_lineage" => {
            let v = db::agent_lineage(&conn, str_param!("agent"))?;
            Ok(serde_json::to_value(v)?)
        }
        "set_agent_parent" => {
            db::set_agent_parent(&conn, str_param!("agent"), opt_str_param!("parent"))?;
            Ok(serde_json::json!(true))
        }
        "delete_agent" => {
            let v = db::delete_agent(&conn, str_param!("name"))?;
            Ok(serde_json::to_value(v)?)
        }
        "list_agents" => {
            let v = db::list_agents(&conn)?;
            Ok(serde_json::to_value(v)?)
        }
        "get_agent" => {
            let v = db::get_agent(&conn, str_param!("name"))?;
            Ok(serde_json::to_value(v)?)
        }
        "ensure_default_agent" => {
            db::ensure_default_agent(&conn)?;
            Ok(serde_json::json!(true))
        }
        "set_agent_data" => {
            db::set_agent_data(
                &conn,
                str_param!("agent"),
                str_param!("key"),
                str_param!("value"),
            )?;
            Ok(serde_json::json!(true))
        }
        "get_agent_data" => {
            let v = db::get_agent_data(&conn, str_param!("agent"), str_param!("key"))?;
            Ok(serde_json::to_value(v)?)
        }
        "delete_agent_data" => {
            let v = db::delete_agent_data(&conn, str_param!("agent"), str_param!("key"))?;
            Ok(serde_json::to_value(v)?)
        }
        "list_agent_data" => {
            let v = db::list_agent_data(&conn, str_param!("agent"))?;
            Ok(serde_json::to_value(v)?)
        }
        "get_agent_config" => {
            let v = db::get_agent_config(&conn, str_param!("agent_name"))?;
            Ok(serde_json::to_value(v)?)
        }
        "set_agent_config" => {
            let config: db::AgentConfig = serde_json::from_value(p["config"].clone())
                .map_err(|e| format!("bad param 'config': {}", e))?;
            db::set_agent_config(&conn, str_param!("agent_name"), &config)?;
            Ok(serde_json::json!(true))
        }
        "save_agent_messages" => {
            let msgs = p["messages"].as_array().ok_or("missing param 'messages'")?;
            db::save_agent_messages(&conn, str_param!("agent_name"), msgs)?;
            Ok(serde_json::json!(true))
        }
        "load_agent_messages" => {
            let v: Vec<serde_json::Value> =
                db::load_agent_messages(&conn, str_param!("agent_name"))?;
            Ok(serde_json::to_value(v)?)
        }
        "append_agent_message" => {
            let msg = &p["message"];
            if msg.is_null() {
                return Err("missing param 'message'".into());
            }
            db::append_agent_message(&conn, str_param!("agent_name"), msg)?;
            Ok(serde_json::json!(true))
        }
        "clear_agent_messages" => {
            db::clear_agent_messages(&conn, str_param!("agent_name"))?;
            Ok(serde_json::json!(true))
        }
        "agent_message_count" => {
            let v = db::agent_message_count(&conn, str_param!("agent_name"))?;
            Ok(serde_json::to_value(v)?)
        }
        "send_notification" => {
            let v = db::send_notification(
                &conn,
                str_param!("from_agent"),
                str_param!("to_agent"),
                str_param!("message"),
            )?;
            Ok(serde_json::to_value(v)?)
        }
        "poll_notifications_for_session" => {
            drop(conn);
            let conn = db.lock().map_err(|e| format!("DB lock: {}", e))?;
            let v = db::poll_notifications_for_session(&conn, str_param!("session_id"))?;
            Ok(serde_json::to_value(v)?)
        }
        "create_cron_task" => {
            let v = db::create_cron_task(
                &conn,
                str_param!("name"),
                str_param!("description"),
                str_param!("cron_expression"),
                str_param!("command"),
                opt_str_param!("agent_name"),
                p["max_runs"].as_i64(),
            )?;
            Ok(serde_json::to_value(v)?)
        }
        "create_oneshot_task" => {
            let v = db::create_oneshot_task(
                &conn,
                str_param!("name"),
                str_param!("description"),
                str_param!("run_at"),
                str_param!("command"),
                opt_str_param!("agent_name"),
            )?;
            Ok(serde_json::to_value(v)?)
        }
        "delete_task" => {
            let v = db::delete_task(&conn, i64_param!("task_id"))?;
            Ok(serde_json::to_value(v)?)
        }
        "list_tasks" => {
            let v = db::list_tasks(&conn, opt_str_param!("agent_name"))?;
            Ok(serde_json::to_value(v)?)
        }
        "get_task" => {
            let v = db::get_task(&conn, i64_param!("task_id"))?;
            Ok(serde_json::to_value(v)?)
        }
        "set_task_enabled" => {
            let enabled = p["enabled"].as_bool().ok_or("missing param 'enabled'")?;
            let v = db::set_task_enabled(&conn, i64_param!("task_id"), enabled)?;
            Ok(serde_json::to_value(v)?)
        }
        "set_task_held" => {
            let held = p["held"].as_bool().ok_or("missing param 'held'")?;
            let v = db::set_task_held(&conn, i64_param!("task_id"), held)?;
            Ok(serde_json::to_value(v)?)
        }
        "set_task_target" => {
            let v = db::set_task_target(
                &conn,
                i64_param!("task_id"),
                opt_str_param!("agent"),
                opt_str_param!("profile"),
            )?;
            Ok(serde_json::to_value(v)?)
        }
        "get_pending_tasks" => {
            let v = db::get_pending_tasks(&conn)?;
            Ok(serde_json::to_value(v)?)
        }
        "create_task" => {
            let task: db::NewTask = serde_json::from_value(p["task"].clone())
                .map_err(|e| format!("bad param 'task': {}", e))?;
            let v = db::create_task(&conn, &task)?;
            Ok(serde_json::to_value(v)?)
        }
        "prune_tasks" => {
            let older_than = db::parse_db_time(str_param!("older_than"))
                .ok_or("bad param 'older_than': not a timestamp")?;
            let dry_run = p["dry_run"].as_bool().unwrap_or(false);
            let v = db::prune_tasks(&conn, older_than, dry_run)?;
            Ok(serde_json::to_value(v)?)
        }
        "kb_write" => {
            let tags: Vec<String> = serde_json::from_value(p["tags"].clone()).unwrap_or_default();
            let v = db::kb_write(
                &conn,
                str_param!("title"),
                str_param!("body"),
                &tags,
                opt_str_param!("private_to"),
                opt_str_param!("author"),
            )?;
            Ok(serde_json::to_value(v)?)
        }
        "kb_get" | "kb_get_by_title" | "kb_search" | "kb_list" | "kb_delete" => {
            let viewer: db::KbViewer = serde_json::from_value(p["viewer"].clone())
                .map_err(|e| format!("bad param 'viewer': {}", e))?;
            let limit = p["limit"].as_u64().unwrap_or(10) as usize;
            Ok(match req.method.as_str() {
                "kb_get" => serde_json::to_value(db::kb_get(&conn, i64_param!("id"), &viewer)?)?,
                "kb_get_by_title" => {
                    serde_json::to_value(db::kb_get_by_title(&conn, str_param!("title"), &viewer)?)?
                }
                "kb_search" => serde_json::to_value(db::kb_search(
                    &conn,
                    str_param!("query"),
                    &viewer,
                    opt_str_param!("tag"),
                    limit,
                )?)?,
                "kb_list" => serde_json::to_value(db::kb_list(
                    &conn,
                    &viewer,
                    opt_str_param!("tag"),
                    limit,
                )?)?,
                _ => serde_json::to_value(db::kb_delete(&conn, i64_param!("id"), &viewer)?)?,
            })
        }
        "fail_tasks_with_failed_dependencies" => {
            let v = db::fail_tasks_with_failed_dependencies(&conn)?;
            Ok(serde_json::to_value(v)?)
        }
        "claim_task" => {
            let v = db::claim_task(&conn, i64_param!("task_id"), str_param!("session_id"))?;
            Ok(serde_json::to_value(v)?)
        }
        "finish_task" => {
            let outcome: db::TaskOutcome = serde_json::from_value(p["outcome"].clone())
                .map_err(|e| format!("bad param 'outcome': {}", e))?;
            let v = db::finish_task(
                &conn,
                i64_param!("task_id"),
                str_param!("session_id"),
                &outcome,
            )?;
            Ok(serde_json::to_value(v)?)
        }
        "gc_agents" => {
            let v = db::gc_agents(&conn)?;
            Ok(serde_json::to_value(v)?)
        }
        _ => Err(format!("unknown method: {}", req.method).into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote_db::RemoteDb;
    use faber::db_backend::DbBackend;

    /// A `RemoteDb` talking to the real `handle_client` over a local
    /// socket, backed by an in-memory database.
    fn connected_client() -> RemoteDb {
        let conn = Connection::open_in_memory().unwrap();
        db::initialize_db(&conn).unwrap();
        let db = Arc::new(Mutex::new(conn));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let _ = handle_client(stream, db, None);
        });
        RemoteDb::connect(&addr.to_string()).unwrap()
    }

    #[test]
    fn test_remote_calls_returning_null_succeed() {
        let client = connected_client();
        client.heartbeat_all("session").unwrap();
        client.create_agent("a", "desc").unwrap();
        assert!(client.get_agent("missing").unwrap().is_none());
    }

    #[test]
    fn test_concurrent_remote_calls_each_get_their_own_response() {
        let client = Arc::new(connected_client());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let client = client.clone();
                std::thread::spawn(move || {
                    for i in 0..25 {
                        let name = format!("agent-{}-{}", t, i);
                        client.create_agent(&name, "desc").unwrap();
                        client.heartbeat_all("session").unwrap();
                        let row = client.get_agent(&name).unwrap();
                        assert_eq!(row.map(|r| r.name), Some(name));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(client.list_agents().unwrap().len(), 8 * 25);
    }

    #[test]
    fn test_kb_over_a_remote_connection() {
        let client = connected_client();
        let (id, created) = client
            .kb_write(
                "Build",
                "cargo build --release",
                &["ci".to_string()],
                None,
                Some("bot"),
            )
            .unwrap();
        assert!(created);
        let viewer = db::KbViewer::Agent(None);
        let hits = client
            .kb_search("release build", &viewer, Some("ci"), 5)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].note.id, id);
        assert!(client.kb_get_by_title("BUILD", &viewer).unwrap().is_some());
        assert_eq!(
            client.kb_list(&db::KbViewer::User, None, 10).unwrap().len(),
            1
        );
        assert!(client.kb_delete(id, &viewer).unwrap());
        assert!(client.kb_get(id, &db::KbViewer::User).unwrap().is_none());
    }

    #[test]
    fn test_agent_config_and_task_targets_over_a_remote_connection() {
        let client = connected_client();
        client.create_agent("bob", "").unwrap();
        let config = db::AgentConfig {
            model: Some("qwen".to_string()),
            max_tokens: Some(512),
            tools: Some(vec!["read_file".to_string()]),
            profile: Some("fast".to_string()),
            ..Default::default()
        };
        client.set_agent_config("bob", &config).unwrap();
        assert_eq!(client.get_agent_config("bob").unwrap(), config);

        let at = chrono::Utc::now().to_rfc3339();
        let id = client
            .create_task(&db::NewTask {
                name: "t".to_string(),
                description: String::new(),
                kind: db::TaskKind::PROMPT.to_string(),
                command: "do it".to_string(),
                agent_name: None,
                schedule: db::TaskSchedule::Once { at },
                held: false,
                depends_on: Vec::new(),
                profile: Some("fast".to_string()),
            })
            .unwrap();
        assert_eq!(
            client.get_task(id).unwrap().unwrap().profile.as_deref(),
            Some("fast")
        );
        assert!(client.set_task_target(id, Some("bob"), None).unwrap());
        assert!(
            client
                .set_task_target(id, Some("bob"), Some("fast"))
                .is_err()
        );
        let task = client.get_task(id).unwrap().unwrap();
        assert_eq!(
            (task.agent_name.as_deref(), task.profile),
            (Some("bob"), None)
        );
    }

    #[test]
    fn test_kb_lineage_over_a_remote_connection() {
        let client = connected_client();
        client.create_agent("boss", "").unwrap();
        client.create_agent("helper", "").unwrap();
        client.set_agent_parent("helper", Some("boss")).unwrap();
        assert!(
            client.set_agent_parent("boss", Some("helper")).is_err(),
            "no loops"
        );
        client
            .kb_write("Secret", "boss only", &[], Some("boss"), None)
            .unwrap();
        let as_helper = db::KbViewer::Agent(Some("helper".to_string()));
        assert_eq!(
            client
                .kb_search("secret", &as_helper, None, 5)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            client
                .get_agent("helper")
                .unwrap()
                .unwrap()
                .parent
                .as_deref(),
            Some("boss")
        );
    }

    #[test]
    fn test_request_id_hint() {
        assert_eq!(request_id_hint(r#"{"id": 7, "method": 1}"#), Some(7));
        assert_eq!(request_id_hint(r#"{"id": "x"}"#), None);
        assert_eq!(request_id_hint("not json"), None);
    }
}
