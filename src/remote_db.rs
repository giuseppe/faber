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

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use faber::agent_io::{AgentEvent, AgentEventRow, EventFilter};
use faber::artifacts::NewArtifact;
use faber::db::{
    AgentConfig, AgentRow, ArtifactRow, KbHit, KbNote, KbViewer, NewTask, NotificationRow,
    TaskConversation, TaskOutcome, TaskRow,
};
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

    fn set_agent_activity(&self, agent: &str, activity: &str) -> Result<(), Box<dyn Error>> {
        self.call(
            "set_agent_activity",
            serde_json::json!({"agent": agent, "activity": activity}),
        )?;
        Ok(())
    }

    fn agent_lineage(&self, agent: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let v = self.call("agent_lineage", serde_json::json!({"agent": agent}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn set_agent_parent(&self, agent: &str, parent: Option<&str>) -> Result<(), Box<dyn Error>> {
        self.call(
            "set_agent_parent",
            serde_json::json!({"agent": agent, "parent": parent}),
        )?;
        Ok(())
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

    fn get_agent_config(&self, agent_name: &str) -> Result<AgentConfig, Box<dyn Error>> {
        let v = self.call(
            "get_agent_config",
            serde_json::json!({"agent_name": agent_name}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn set_agent_config(
        &self,
        agent_name: &str,
        config: &AgentConfig,
    ) -> Result<(), Box<dyn Error>> {
        self.call(
            "set_agent_config",
            serde_json::json!({"agent_name": agent_name, "config": config}),
        )?;
        Ok(())
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

    fn set_task_held(&self, task_id: i64, held: bool) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "set_task_held",
            serde_json::json!({"task_id": task_id, "held": held}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn set_task_target(
        &self,
        task_id: i64,
        agent: Option<&str>,
        profile: Option<&str>,
    ) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "set_task_target",
            serde_json::json!({"task_id": task_id, "agent": agent, "profile": profile}),
        )?;
        Ok(serde_json::from_value(v)?)
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

    fn kb_write(
        &self,
        title: &str,
        body: &str,
        tags: &[String],
        private_to: Option<&str>,
        author: Option<&str>,
    ) -> Result<(i64, bool), Box<dyn Error>> {
        let v = self.call(
            "kb_write",
            serde_json::json!({"title": title, "body": body, "tags": tags,
                               "private_to": private_to, "author": author}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn kb_get(&self, id: i64, viewer: &KbViewer) -> Result<Option<KbNote>, Box<dyn Error>> {
        let v = self.call("kb_get", serde_json::json!({"id": id, "viewer": viewer}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn kb_get_by_title(
        &self,
        title: &str,
        viewer: &KbViewer,
    ) -> Result<Option<KbNote>, Box<dyn Error>> {
        let v = self.call(
            "kb_get_by_title",
            serde_json::json!({"title": title, "viewer": viewer}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn kb_search(
        &self,
        query: &str,
        viewer: &KbViewer,
        tag: Option<&str>,
        limit: usize,
    ) -> Result<Vec<KbHit>, Box<dyn Error>> {
        let v = self.call(
            "kb_search",
            serde_json::json!({"query": query, "viewer": viewer, "tag": tag, "limit": limit}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn kb_list(
        &self,
        viewer: &KbViewer,
        tag: Option<&str>,
        limit: usize,
    ) -> Result<Vec<KbNote>, Box<dyn Error>> {
        let v = self.call(
            "kb_list",
            serde_json::json!({"viewer": viewer, "tag": tag, "limit": limit}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn kb_delete(&self, id: i64, viewer: &KbViewer) -> Result<bool, Box<dyn Error>> {
        let v = self.call("kb_delete", serde_json::json!({"id": id, "viewer": viewer}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn fail_tasks_with_failed_dependencies(&self) -> Result<Vec<i64>, Box<dyn Error>> {
        let v = self.call("fail_tasks_with_failed_dependencies", serde_json::json!({}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn claim_task(&self, task_id: i64, session_id: &str) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "claim_task",
            serde_json::json!({"task_id": task_id, "session_id": session_id}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn release_task(&self, task_id: i64, session_id: &str) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "release_task",
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

    fn update_task(&self, task_id: i64, task: &NewTask) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "update_task",
            serde_json::json!({"task_id": task_id, "task": task}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn set_profile(&self, name: &str, settings: &serde_json::Value) -> Result<(), Box<dyn Error>> {
        self.call(
            "set_profile",
            serde_json::json!({"name": name, "settings": settings}),
        )?;
        Ok(())
    }

    fn delete_profile(&self, name: &str) -> Result<bool, Box<dyn Error>> {
        let v = self.call("delete_profile", serde_json::json!({"name": name}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn set_agent_description(
        &self,
        agent: &str,
        description: &str,
    ) -> Result<bool, Box<dyn Error>> {
        let v = self.call(
            "set_agent_description",
            serde_json::json!({"agent": agent, "description": description}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn list_profiles(&self) -> Result<Vec<(String, serde_json::Value)>, Box<dyn Error>> {
        let v = self.call("list_profiles", serde_json::json!({}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn request_task_stop(&self, task_id: i64) -> Result<bool, Box<dyn Error>> {
        let v = self.call("request_task_stop", serde_json::json!({"task_id": task_id}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn save_task_conversation(
        &self,
        task_id: i64,
        conversation: &TaskConversation,
    ) -> Result<(), Box<dyn Error>> {
        self.call(
            "save_task_conversation",
            serde_json::json!({"task_id": task_id, "conversation": conversation}),
        )?;
        Ok(())
    }

    fn load_task_conversation(
        &self,
        task_id: i64,
    ) -> Result<Option<TaskConversation>, Box<dyn Error>> {
        let v = self.call(
            "load_task_conversation",
            serde_json::json!({"task_id": task_id}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn task_conversation_config(
        &self,
        task_id: i64,
    ) -> Result<Option<AgentConfig>, Box<dyn Error>> {
        let v = self.call(
            "task_conversation_config",
            serde_json::json!({"task_id": task_id}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn run_task_now(&self, task_id: i64) -> Result<bool, Box<dyn Error>> {
        let v = self.call("run_task_now", serde_json::json!({"task_id": task_id}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn a2a_create_context(
        &self,
        owner: &str,
        profile: &str,
        config: &AgentConfig,
    ) -> Result<faber::db::A2aContext, Box<dyn Error>> {
        let v = self.call(
            "a2a_create_context",
            serde_json::json!({"owner": owner, "profile": profile, "config": config}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn a2a_get_context(&self, id: &str) -> Result<Option<faber::db::A2aContext>, Box<dyn Error>> {
        let v = self.call("a2a_get_context", serde_json::json!({"id": id}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn a2a_touch_context(&self, id: &str) -> Result<bool, Box<dyn Error>> {
        let v = self.call("a2a_touch_context", serde_json::json!({"id": id}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn a2a_expired_contexts(
        &self,
        before: &str,
    ) -> Result<Vec<faber::db::A2aContext>, Box<dyn Error>> {
        let v = self.call(
            "a2a_expired_contexts",
            serde_json::json!({"before": before}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn a2a_delete_context(&self, id: &str) -> Result<bool, Box<dyn Error>> {
        let v = self.call("a2a_delete_context", serde_json::json!({"id": id}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn gc_agents(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let v = self.call("gc_agents", serde_json::json!({}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn artifact_upload_begin(&self) -> Result<String, Box<dyn Error>> {
        let v = self.call("artifact_upload_begin", serde_json::json!({}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn artifact_upload_append(&self, upload: &str, data: &[u8]) -> Result<(), Box<dyn Error>> {
        self.call(
            "artifact_upload_append",
            serde_json::json!({"upload": upload, "data": BASE64.encode(data)}),
        )?;
        Ok(())
    }

    fn artifact_upload_finish(
        &self,
        upload: &str,
        artifact: &NewArtifact,
    ) -> Result<ArtifactRow, Box<dyn Error>> {
        let v = self.call(
            "artifact_upload_finish",
            serde_json::json!({"upload": upload, "artifact": artifact}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn list_artifacts(&self, task_id: i64) -> Result<Vec<ArtifactRow>, Box<dyn Error>> {
        let v = self.call("list_artifacts", serde_json::json!({"task_id": task_id}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn find_artifact(
        &self,
        task_id: i64,
        name: &str,
        run: Option<i64>,
    ) -> Result<Option<ArtifactRow>, Box<dyn Error>> {
        let v = self.call(
            "find_artifact",
            serde_json::json!({"task_id": task_id, "name": name, "run": run}),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    fn read_artifact(&self, id: i64, offset: u64, len: usize) -> Result<Vec<u8>, Box<dyn Error>> {
        let v = self.call(
            "read_artifact",
            serde_json::json!({"id": id, "offset": offset, "len": len}),
        )?;
        let data = v.as_str().ok_or("read_artifact: expected base64 text")?;
        Ok(BASE64.decode(data)?)
    }

    fn gc_artifacts(&self) -> Result<usize, Box<dyn Error>> {
        let v = self.call("gc_artifacts", serde_json::json!({}))?;
        Ok(serde_json::from_value(v)?)
    }

    fn append_agent_events(
        &self,
        agent: &str,
        task_id: Option<i64>,
        events: &[AgentEvent],
    ) -> Result<(), Box<dyn Error>> {
        self.call(
            "append_agent_events",
            serde_json::json!({"agent": agent, "task_id": task_id, "events": events}),
        )?;
        Ok(())
    }

    fn agent_events(&self, filter: &EventFilter) -> Result<Vec<AgentEventRow>, Box<dyn Error>> {
        let v = self.call("agent_events", serde_json::json!({"filter": filter}))?;
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
