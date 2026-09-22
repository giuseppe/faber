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

use crate::db;
use crate::db::{AgentConfig, AgentRow, NotificationRow, TaskRow};
use crate::db_backend::DbBackend;
use std::error::Error;
use std::sync::{Arc, Mutex, MutexGuard};

pub struct LocalDb {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl LocalDb {
    pub fn new(conn: Arc<Mutex<rusqlite::Connection>>) -> Self {
        Self { conn }
    }

    fn lock(&self) -> Result<MutexGuard<'_, rusqlite::Connection>, Box<dyn Error>> {
        self.conn
            .lock()
            .map_err(|e| format!("DB lock: {}", e).into())
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

    fn list_agent_data(&self, agent: &str) -> Result<Vec<(String, String)>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::list_agent_data(&conn, agent)
    }

    fn get_agent_config(&self, agent_name: &str) -> Result<AgentConfig, Box<dyn Error>> {
        let conn = self.lock()?;
        db::get_agent_config(&conn, agent_name)
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

    fn create_cron_task(
        &self,
        name: &str,
        description: &str,
        cron_expr: &str,
        command: &str,
        agent_name: Option<&str>,
        max_runs: Option<i64>,
    ) -> Result<i64, Box<dyn Error>> {
        let conn = self.lock()?;
        db::create_cron_task(
            &conn,
            name,
            description,
            cron_expr,
            command,
            agent_name,
            max_runs,
        )
    }

    fn create_oneshot_task(
        &self,
        name: &str,
        description: &str,
        run_at: &str,
        command: &str,
        agent_name: Option<&str>,
    ) -> Result<i64, Box<dyn Error>> {
        let conn = self.lock()?;
        db::create_oneshot_task(&conn, name, description, run_at, command, agent_name)
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

    fn get_pending_tasks(&self) -> Result<Vec<TaskRow>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::get_pending_tasks(&conn)
    }

    fn mark_task_executed(
        &self,
        task_id: i64,
        task_type: &str,
        cron_expression: Option<&str>,
        max_runs: Option<i64>,
    ) -> Result<(), Box<dyn Error>> {
        let conn = self.lock()?;
        db::mark_task_executed(&conn, task_id, task_type, cron_expression, max_runs)
    }

    fn gc_agents(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let conn = self.lock()?;
        db::gc_agents(&conn)
    }
}
