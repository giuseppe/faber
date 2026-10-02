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

use faber::db::{AgentConfig, AgentRow, NewTask, NotificationRow, TaskOutcome, TaskRow};
use faber::db_backend::DbBackend;
use faber::protocol::{RpcRequest, RpcResponse};
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Both directions of the connection, behind one lock: a request and the
/// line answering it must be one exchange, or two threads calling at once
/// could each read the other's response.
struct Connection {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    /// Set once a response didn't match its request: what's left on the
    /// stream can no longer be paired up with requests.
    desynced: bool,
}

pub struct RemoteDb {
    connection: Mutex<Connection>,
    next_id: AtomicU64,
}

impl RemoteDb {
    pub fn connect(addr: &str) -> Result<Self, Box<dyn Error>> {
        let stream = TcpStream::connect(addr)?;
        // Small request/response exchanges: don't let Nagle's algorithm
        // hold a request back waiting for an ACK.
        stream.set_nodelay(true)?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(Self {
            connection: Mutex::new(Connection {
                reader,
                writer: stream,
                desynced: false,
            }),
            next_id: AtomicU64::new(1),
        })
    }

    pub fn authenticate(&self, key: &str) -> Result<(), Box<dyn Error>> {
        let result = self.call("auth", serde_json::json!({"key": key}))?;
        if result.as_bool() == Some(true) {
            Ok(())
        } else {
            Err("authentication failed".into())
        }
    }

    fn call(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn Error>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = RpcRequest {
            id,
            method: method.to_string(),
            params,
        };
        // One write per request (see `send_response` in the server).
        let mut request_line = serde_json::to_string(&request)?;
        request_line.push('\n');

        let mut connection = self
            .connection
            .lock()
            .map_err(|e| format!("connection lock: {}", e))?;
        if connection.desynced {
            return Err("connection to the faber server is out of sync; restart faber".into());
        }
        connection.writer.write_all(request_line.as_bytes())?;
        connection.writer.flush()?;

        let mut line = String::new();
        if connection.reader.read_line(&mut line)? == 0 {
            return Err("the faber server closed the connection".into());
        }
        let response: RpcResponse = serde_json::from_str(&line)?;
        // A null id (the server couldn't parse the request) still answers
        // this line; only another request's id means the stream is off.
        if response.id.is_some_and(|response_id| response_id != id) {
            connection.desynced = true;
        }
        Ok(response.into_result(id)?)
    }
}

impl DbBackend for RemoteDb {
    fn claim_agent(&self, name: &str, session_id: &str) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "claim_agent",
            serde_json::json!({"name": name, "session_id": session_id}),
        )?;
        Ok(v.as_bool().unwrap_or(false))
    }

    fn heartbeat_all(&self, session_id: &str) -> Result<(), Box<dyn Error>> {
        self.call(
            "heartbeat_all",
            serde_json::json!({"session_id": session_id}),
        )?;
        Ok(())
    }

    fn release_agent(&self, name: &str, session_id: &str) -> Result<(), Box<dyn Error>> {
        self.call(
            "release_agent",
            serde_json::json!({"name": name, "session_id": session_id}),
        )?;
        Ok(())
    }

    fn release_all_agents(&self, session_id: &str) -> Result<(), Box<dyn Error>> {
        self.call(
            "release_all_agents",
            serde_json::json!({"session_id": session_id}),
        )?;
        Ok(())
    }

    fn create_agent(&self, name: &str, description: &str) -> Result<(), Box<dyn Error>> {
        self.call(
            "create_agent",
            serde_json::json!({"name": name, "description": description}),
        )?;
        Ok(())
    }

    fn delete_agent(&self, name: &str) -> Result<bool, Box<dyn Error>> {
        let v = self.call("delete_agent", serde_json::json!({"name": name}))?;
        Ok(v.as_bool().unwrap_or(false))
    }

    fn list_agents(&self) -> Result<Vec<AgentRow>, Box<dyn Error>> {
        let v = self.call("list_agents", serde_json::json!({}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn get_agent(&self, name: &str) -> Result<Option<AgentRow>, Box<dyn Error>> {
        let v = self.call("get_agent", serde_json::json!({"name": name}))?;
        if v.is_null() {
            Ok(None)
        } else {
            Ok(Some(serde_json::from_value(v)?))
        }
    }

    fn ensure_default_agent(&self) -> Result<(), Box<dyn Error>> {
        self.call("ensure_default_agent", serde_json::json!({}))?;
        Ok(())
    }

    fn set_agent_data(&self, agent: &str, key: &str, value: &str) -> Result<(), Box<dyn Error>> {
        self.call(
            "set_agent_data",
            serde_json::json!({"agent": agent, "key": key, "value": value}),
        )?;
        Ok(())
    }

    fn get_agent_data(&self, agent: &str, key: &str) -> Result<Option<String>, Box<dyn Error>> {
        let v = self.call(
            "get_agent_data",
            serde_json::json!({"agent": agent, "key": key}),
        )?;
        if v.is_null() {
            Ok(None)
        } else {
            Ok(Some(v.as_str().ok_or("expected string value")?.to_string()))
        }
    }

    fn delete_agent_data(&self, agent: &str, key: &str) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "delete_agent_data",
            serde_json::json!({"agent": agent, "key": key}),
        )?;
        Ok(v.as_bool().unwrap_or(false))
    }

    fn list_agent_data(&self, agent: &str) -> Result<Vec<(String, String)>, Box<dyn Error>> {
        let v = self.call("list_agent_data", serde_json::json!({"agent": agent}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn get_agent_config(&self, agent_name: &str) -> Result<AgentConfig, Box<dyn Error>> {
        let v = self.call(
            "get_agent_config",
            serde_json::json!({"agent_name": agent_name}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn save_agent_messages(
        &self,
        agent_name: &str,
        messages: &[serde_json::Value],
    ) -> Result<(), Box<dyn Error>> {
        self.call(
            "save_agent_messages",
            serde_json::json!({"agent_name": agent_name, "messages": messages}),
        )?;
        Ok(())
    }

    fn load_agent_messages(
        &self,
        agent_name: &str,
    ) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
        let v = self.call(
            "load_agent_messages",
            serde_json::json!({"agent_name": agent_name}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn append_agent_message(
        &self,
        agent_name: &str,
        message: &serde_json::Value,
    ) -> Result<(), Box<dyn Error>> {
        self.call(
            "append_agent_message",
            serde_json::json!({"agent_name": agent_name, "message": message}),
        )?;
        Ok(())
    }

    fn clear_agent_messages(&self, agent_name: &str) -> Result<(), Box<dyn Error>> {
        self.call(
            "clear_agent_messages",
            serde_json::json!({"agent_name": agent_name}),
        )?;
        Ok(())
    }

    fn agent_message_count(&self, agent_name: &str) -> Result<i64, Box<dyn Error>> {
        let v = self.call(
            "agent_message_count",
            serde_json::json!({"agent_name": agent_name}),
        )?;
        v.as_i64().ok_or("expected integer".into())
    }

    fn send_notification(
        &self,
        from_agent: &str,
        to_agent: &str,
        message: &str,
    ) -> Result<i64, Box<dyn Error>> {
        let v = self.call(
            "send_notification",
            serde_json::json!({"from_agent": from_agent, "to_agent": to_agent, "message": message}),
        )?;
        v.as_i64().ok_or("expected integer".into())
    }

    fn poll_notifications_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<NotificationRow>, Box<dyn Error>> {
        let v = self.call(
            "poll_notifications_for_session",
            serde_json::json!({"session_id": session_id}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn create_cron_task(
        &self,
        name: &str,
        description: &str,
        cron_expr: &str,
        command: &str,
        agent_name: Option<&str>,
        max_runs: Option<i64>,
    ) -> Result<i64, Box<dyn Error>> {
        let v = self.call(
            "create_cron_task",
            serde_json::json!({
                "name": name,
                "description": description,
                "cron_expression": cron_expr,
                "command": command,
                "agent_name": agent_name,
                "max_runs": max_runs,
            }),
        )?;
        v.as_i64().ok_or("expected integer".into())
    }

    fn create_oneshot_task(
        &self,
        name: &str,
        description: &str,
        run_at: &str,
        command: &str,
        agent_name: Option<&str>,
    ) -> Result<i64, Box<dyn Error>> {
        let v = self.call(
            "create_oneshot_task",
            serde_json::json!({
                "name": name,
                "description": description,
                "run_at": run_at,
                "command": command,
                "agent_name": agent_name,
            }),
        )?;
        v.as_i64().ok_or("expected integer".into())
    }

    fn delete_task(&self, task_id: i64) -> Result<bool, Box<dyn Error>> {
        let v = self.call("delete_task", serde_json::json!({"task_id": task_id}))?;
        Ok(v.as_bool().unwrap_or(false))
    }

    fn list_tasks(&self, agent_name: Option<&str>) -> Result<Vec<TaskRow>, Box<dyn Error>> {
        let v = self.call("list_tasks", serde_json::json!({"agent_name": agent_name}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn get_task(&self, task_id: i64) -> Result<Option<TaskRow>, Box<dyn Error>> {
        let v = self.call("get_task", serde_json::json!({"task_id": task_id}))?;
        if v.is_null() {
            Ok(None)
        } else {
            Ok(Some(serde_json::from_value(v)?))
        }
    }

    fn set_task_enabled(&self, task_id: i64, enabled: bool) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "set_task_enabled",
            serde_json::json!({"task_id": task_id, "enabled": enabled}),
        )?;
        Ok(v.as_bool().unwrap_or(false))
    }

    fn get_pending_tasks(&self) -> Result<Vec<TaskRow>, Box<dyn Error>> {
        let v = self.call("get_pending_tasks", serde_json::json!({}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn create_task(&self, task: &NewTask) -> Result<i64, Box<dyn Error>> {
        let v = self.call("create_task", serde_json::json!({"task": task}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn prune_tasks(&self, older_than: &str, dry_run: bool) -> Result<Vec<TaskRow>, Box<dyn Error>> {
        let v = self.call(
            "prune_tasks",
            serde_json::json!({"older_than": older_than, "dry_run": dry_run}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn claim_task(&self, task_id: i64, session_id: &str) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "claim_task",
            serde_json::json!({"task_id": task_id, "session_id": session_id}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn finish_task(
        &self,
        task_id: i64,
        session_id: &str,
        outcome: &TaskOutcome,
    ) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "finish_task",
            serde_json::json!({"task_id": task_id, "session_id": session_id, "outcome": outcome}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn gc_agents(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let v = self.call("gc_agents", serde_json::json!({}))?;
        Ok(serde_json::from_value(v)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// A server that answers every request with the given ids, in turn.
    fn fake_server(answer_ids: Vec<u64>) -> RemoteDb {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut writer = stream;
            let mut answer_ids = answer_ids.into_iter();
            loop {
                // Read the request before answering - or before hanging up,
                // so the client sees a clean end of stream, not a reset.
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                let Some(id) = answer_ids.next() else { return };
                let _ = writeln!(writer, r#"{{"id":{},"result":null}}"#, id);
            }
        });
        RemoteDb::connect(&addr.to_string()).unwrap()
    }

    #[test]
    fn test_mismatched_response_id_is_an_error_and_stops_the_connection() {
        let client = fake_server(vec![1, 999, 3]);
        client.heartbeat_all("s").unwrap();
        let err = client.heartbeat_all("s").unwrap_err().to_string();
        assert!(err.contains("doesn't match request id 2"), "{err}");
        // The server would answer the next one "correctly", but the
        // stream can't be trusted any more.
        let err = client.heartbeat_all("s").unwrap_err().to_string();
        assert!(err.contains("out of sync"), "{err}");
    }

    #[test]
    fn test_closed_connection_is_an_error() {
        let client = fake_server(vec![]);
        let err = client.heartbeat_all("s").unwrap_err().to_string();
        assert!(err.contains("closed the connection"), "{err}");
    }
}
