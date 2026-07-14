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

use log::{info, warn};
use rusqlite::Connection;
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use swarmblabla::db;
use swarmblabla::protocol::{RpcRequest, RpcResponse};

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
    println!("swarmblabla serve listening on {}", bind);

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
                let resp = RpcResponse::error(0, format!("invalid request: {}", e));
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
                    let r = RpcResponse::error(request.id, "unauthorized".into());
                    send_response(&mut writer, &r)?;
                    return Ok(());
                }
            } else {
                let r = RpcResponse::error(request.id, "auth required".into());
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

fn send_response(writer: &mut TcpStream, resp: &RpcResponse) -> Result<(), Box<dyn Error>> {
    let json = serde_json::to_string(resp)?;
    writer.write_all(json.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn dispatch(db: &Arc<Mutex<Connection>>, req: &RpcRequest) -> RpcResponse {
    match dispatch_inner(db, req) {
        Ok(value) => RpcResponse::success(req.id, value),
        Err(e) => RpcResponse::error(req.id, e.to_string()),
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
        "get_pending_tasks" => {
            let v = db::get_pending_tasks(&conn)?;
            Ok(serde_json::to_value(v)?)
        }
        "mark_task_executed" => {
            db::mark_task_executed(
                &conn,
                i64_param!("task_id"),
                str_param!("task_type"),
                opt_str_param!("cron_expression"),
                p["max_runs"].as_i64(),
            )?;
            Ok(serde_json::json!(true))
        }
        "gc_agents" => {
            let v = db::gc_agents(&conn)?;
            Ok(serde_json::to_value(v)?)
        }
        _ => Err(format!("unknown method: {}", req.method).into()),
    }
}
