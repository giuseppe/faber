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

use crate::agent_io::{AgentEvent, AgentEventRow, EventFilter};
use crate::artifacts::{ArtifactStore, NewArtifact};
use crate::db;
use crate::db::{
    AgentConfig, AgentRow, ArtifactRow, KbHit, KbNote, KbViewer, NewTask, NotificationRow,
    TaskConversation, TaskOutcome, TaskRow,
};
use crate::db_backend::DbBackend;
use std::error::Error;
use std::sync::{Arc, Mutex, MutexGuard};

pub struct LocalDb {
    conn: Arc<Mutex<rusqlite::Connection>>,
    artifacts: Option<Arc<ArtifactStore>>,
}

impl LocalDb {
    pub fn new(conn: Arc<Mutex<rusqlite::Connection>>) -> Self {
        Self {
            conn,
            artifacts: None,
        }
    }

    /// Keeps artifacts' contents in `store`: without one, they can't be
    /// saved or read.
    pub fn with_artifacts(mut self, store: Arc<ArtifactStore>) -> Self {
        self.artifacts = Some(store);
        self
    }

    fn artifacts(&self) -> Result<&ArtifactStore, Box<dyn Error>> {
        self.artifacts
            .as_deref()
            .ok_or_else(|| "no directory for artifacts is configured".into())
    }

    /// The connection, to use for one call. None leaves a transaction
    /// open between calls, so one still open is a leftover from a failure:
    /// it's rolled back, and said so - left open, every later write would
    /// go nowhere, silently, until faber restarted.
    fn lock(&self) -> Result<MutexGuard<'_, rusqlite::Connection>, Box<dyn Error>> {
        let conn = self.conn.lock().map_err(|e| format!("DB lock: {}", e))?;
        if !conn.is_autocommit() {
            log::error!(
                "the database connection was left inside a transaction; rolling it back \
                 so that writes are saved again"
            );
            conn.execute_batch("ROLLBACK")?;
        }
        Ok(conn)
    }
}

impl DbBackend for LocalDb {
    fn claim_agent(&self, name: &str, session_id: &str) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::claim_agent(&conn, name, session_id)
    }

    fn heartbeat_all(&self, session_id: &str) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::heartbeat_all(&conn, session_id)
    }

    fn release_agent(&self, name: &str, session_id: &str) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::release_agent(&conn, name, session_id)
    }

    fn release_all_agents(&self, session_id: &str) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::release_all_agents(&conn, session_id)
    }

    fn create_agent(&self, name: &str, description: &str) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::create_agent(&conn, name, description)
    }

    fn delete_agent(&self, name: &str) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::delete_agent(&conn, name)
    }

    fn list_agents(&self) -> Result<Vec<AgentRow>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::list_agents(&conn)
    }

    fn set_agent_activity(&self, agent: &str, activity: &str) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::set_agent_activity(&conn, agent, activity)
    }

    fn agent_lineage(&self, agent: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::agent_lineage(&conn, agent)
    }

    fn set_agent_parent(&self, agent: &str, parent: Option<&str>) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::set_agent_parent(&conn, agent, parent)
    }

    fn get_agent(&self, name: &str) -> Result<Option<AgentRow>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::get_agent(&conn, name)
    }

    fn ensure_default_agent(&self) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::ensure_default_agent(&conn)
    }

    fn set_agent_data(&self, agent: &str, key: &str, value: &str) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::set_agent_data(&conn, agent, key, value)
    }

    fn get_agent_data(&self, agent: &str, key: &str) -> Result<Option<String>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::get_agent_data(&conn, agent, key)
    }

    fn delete_agent_data(&self, agent: &str, key: &str) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::delete_agent_data(&conn, agent, key)
    }

    fn get_agent_config(&self, agent_name: &str) -> Result<AgentConfig, Box<dyn Error>> {
        let conn = self.lock()?;
        db::get_agent_config(&conn, agent_name)
    }

    fn set_agent_config(
        &self,
        agent_name: &str,
        config: &AgentConfig,
    ) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::set_agent_config(&conn, agent_name, config)
    }

    fn save_agent_messages(
        &self,
        agent_name: &str,
        messages: &[serde_json::Value],
    ) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::save_agent_messages(&conn, agent_name, messages)
    }

    fn load_agent_messages(
        &self,
        agent_name: &str,
    ) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::load_agent_messages(&conn, agent_name)
    }

    fn append_agent_message(
        &self,
        agent_name: &str,
        message: &serde_json::Value,
    ) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::append_agent_message(&conn, agent_name, message)
    }

    fn clear_agent_messages(&self, agent_name: &str) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::clear_agent_messages(&conn, agent_name)
    }

    fn agent_message_count(&self, agent_name: &str) -> Result<i64, Box<dyn Error>> {
        let conn = self.lock()?;
        db::agent_message_count(&conn, agent_name)
    }

    fn send_notification(
        &self,
        from_agent: &str,
        to_agent: &str,
        message: &str,
    ) -> Result<i64, Box<dyn Error>> {
        let conn = self.lock()?;
        db::send_notification(&conn, from_agent, to_agent, message)
    }

    fn poll_notifications_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<NotificationRow>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::poll_notifications_for_session(&conn, session_id)
    }

    fn delete_task(&self, task_id: i64) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::delete_task(&conn, task_id)
    }

    fn list_tasks(&self, agent_name: Option<&str>) -> Result<Vec<TaskRow>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::list_tasks(&conn, agent_name)
    }

    fn get_task(&self, task_id: i64) -> Result<Option<TaskRow>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::get_task(&conn, task_id)
    }

    fn set_task_enabled(&self, task_id: i64, enabled: bool) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::set_task_enabled(&conn, task_id, enabled)
    }

    fn set_task_held(&self, task_id: i64, held: bool) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::set_task_held(&conn, task_id, held)
    }

    fn set_task_target(
        &self,
        task_id: i64,
        agent: Option<&str>,
        profile: Option<&str>,
    ) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::set_task_target(&conn, task_id, agent, profile)
    }

    fn get_pending_tasks(&self) -> Result<Vec<TaskRow>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::get_pending_tasks(&conn)
    }

    fn create_task(&self, task: &NewTask) -> Result<i64, Box<dyn Error>> {
        let conn = self.lock()?;
        db::create_task(&conn, task)
    }

    fn prune_tasks(&self, older_than: &str, dry_run: bool) -> Result<Vec<TaskRow>, Box<dyn Error>> {
        let older_than = db::parse_db_time(older_than).ok_or("invalid prune cutoff time")?;
        let conn = self.lock()?;
        db::prune_tasks(&conn, older_than, dry_run)
    }

    fn fail_tasks_with_failed_dependencies(&self) -> Result<Vec<i64>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::fail_tasks_with_failed_dependencies(&conn)
    }

    fn claim_task(&self, task_id: i64, session_id: &str) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::claim_task(&conn, task_id, session_id)
    }

    fn release_task(&self, task_id: i64, session_id: &str) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::release_task(&conn, task_id, session_id)
    }

    fn finish_task(
        &self,
        task_id: i64,
        session_id: &str,
        outcome: &TaskOutcome,
    ) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::finish_task(&conn, task_id, session_id, outcome)
    }

    fn kb_write(
        &self,
        title: &str,
        body: &str,
        tags: &[String],
        private_to: Option<&str>,
        author: Option<&str>,
    ) -> Result<(i64, bool), Box<dyn Error>> {
        let conn = self.lock()?;
        db::kb_write(&conn, title, body, tags, private_to, author)
    }

    fn kb_get(&self, id: i64, viewer: &KbViewer) -> Result<Option<KbNote>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::kb_get(&conn, id, viewer)
    }

    fn kb_get_by_title(
        &self,
        title: &str,
        viewer: &KbViewer,
    ) -> Result<Option<KbNote>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::kb_get_by_title(&conn, title, viewer)
    }

    fn kb_search(
        &self,
        query: &str,
        viewer: &KbViewer,
        tag: Option<&str>,
        limit: usize,
    ) -> Result<Vec<KbHit>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::kb_search(&conn, query, viewer, tag, limit)
    }

    fn kb_list(
        &self,
        viewer: &KbViewer,
        tag: Option<&str>,
        limit: usize,
    ) -> Result<Vec<KbNote>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::kb_list(&conn, viewer, tag, limit)
    }

    fn kb_delete(&self, id: i64, viewer: &KbViewer) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::kb_delete(&conn, id, viewer)
    }

    fn update_task(&self, task_id: i64, task: &NewTask) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::update_task(&conn, task_id, task)
    }

    fn set_profile(&self, name: &str, settings: &serde_json::Value) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::set_profile(&conn, name, settings)
    }

    fn delete_profile(&self, name: &str) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::delete_profile(&conn, name)
    }

    fn set_agent_description(
        &self,
        agent: &str,
        description: &str,
    ) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::set_agent_description(&conn, agent, description)
    }

    fn list_profiles(&self) -> Result<Vec<(String, serde_json::Value)>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::list_profiles(&conn)
    }

    fn request_task_stop(&self, task_id: i64) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::request_task_stop(&conn, task_id)
    }

    fn save_task_conversation(
        &self,
        task_id: i64,
        conversation: &TaskConversation,
    ) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::save_task_conversation(&conn, task_id, conversation)
    }

    fn load_task_conversation(
        &self,
        task_id: i64,
    ) -> Result<Option<TaskConversation>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::load_task_conversation(&conn, task_id)
    }

    fn task_conversation_config(
        &self,
        task_id: i64,
    ) -> Result<Option<AgentConfig>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::task_conversation_config(&conn, task_id)
    }

    fn run_task_now(&self, task_id: i64) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::run_task_now(&conn, task_id)
    }

    fn a2a_create_context(
        &self,
        owner: &str,
        profile: &str,
        config: &AgentConfig,
    ) -> Result<db::A2aContext, Box<dyn Error>> {
        let conn = self.lock()?;
        db::a2a_create_context(&conn, owner, profile, config)
    }

    fn a2a_get_context(&self, id: &str) -> Result<Option<db::A2aContext>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::a2a_get_context(&conn, id)
    }

    fn a2a_touch_context(&self, id: &str) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::a2a_touch_context(&conn, id)
    }

    fn a2a_expired_contexts(&self, before: &str) -> Result<Vec<db::A2aContext>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::a2a_expired_contexts(&conn, before)
    }

    fn a2a_delete_context(&self, id: &str) -> Result<bool, Box<dyn Error>> {
        let conn = self.lock()?;
        db::a2a_delete_context(&conn, id)
    }

    fn gc_agents(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::gc_agents(&conn)
    }

    fn artifact_upload_begin(&self) -> Result<String, Box<dyn Error>> {
        self.artifacts()?.begin()
    }

    fn artifact_upload_append(&self, upload: &str, data: &[u8]) -> Result<(), Box<dyn Error>> {
        self.artifacts()?.append(upload, data)
    }

    fn artifact_upload_finish(
        &self,
        upload: &str,
        artifact: &NewArtifact,
    ) -> Result<ArtifactRow, Box<dyn Error>> {
        crate::artifacts::finish_upload(self.artifacts()?, upload, artifact, || self.lock())
    }

    fn list_artifacts(&self, task_id: i64) -> Result<Vec<ArtifactRow>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::list_artifacts(&conn, task_id)
    }

    fn find_artifact(
        &self,
        task_id: i64,
        name: &str,
        run: Option<i64>,
    ) -> Result<Option<ArtifactRow>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::find_artifact(&conn, task_id, name, run)
    }

    fn read_artifact(&self, id: i64, offset: u64, len: usize) -> Result<Vec<u8>, Box<dyn Error>> {
        let store = self.artifacts()?;
        let conn = self.lock()?;
        crate::artifacts::read_artifact(&conn, store, id, offset, len)
    }

    fn gc_artifacts(&self) -> Result<usize, Box<dyn Error>> {
        let Some(store) = &self.artifacts else {
            return Ok(0);
        };
        crate::artifacts::collect_garbage(store, || self.lock())
    }

    fn append_agent_events(
        &self,
        agent: &str,
        task_id: Option<i64>,
        events: &[AgentEvent],
    ) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::append_agent_events(&conn, agent, task_id, events)
    }

    fn agent_events(&self, filter: &EventFilter) -> Result<Vec<AgentEventRow>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::agent_events(&conn, filter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_transaction_left_open_is_rolled_back_so_writes_are_saved_again() {
        let path = std::env::temp_dir().join(format!("faber_open_tx_{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let conn = rusqlite::Connection::open(&path).unwrap();
        db::initialize_db(&conn).unwrap();
        let shared = Arc::new(Mutex::new(conn));
        let local = LocalDb::new(shared.clone());
        // What a failure partway through a transaction can leave behind.
        shared
            .lock()
            .unwrap()
            .execute_batch("BEGIN IMMEDIATE")
            .unwrap();
        local.create_agent("alice", "").unwrap();
        // Saved: anyone else reading the database sees it.
        let other = rusqlite::Connection::open(&path).unwrap();
        let seen: i64 = other
            .query_row(
                "SELECT count(*) FROM agents WHERE name = 'alice'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(seen, 1);
        drop(other);
        let _ = std::fs::remove_file(&path);
    }
}
