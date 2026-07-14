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

use crate::db::{AgentConfig, AgentRow, NotificationRow, TaskRow};
use std::error::Error;

pub trait DbBackend: Send + Sync {
    fn claim_agent(&self, name: &str, session_id: &str) -> Result<bool, Box<dyn Error>>;
    fn heartbeat_all(&self, session_id: &str) -> Result<(), Box<dyn Error>>;
    fn release_agent(&self, name: &str, session_id: &str) -> Result<(), Box<dyn Error>>;
    fn release_all_agents(&self, session_id: &str) -> Result<(), Box<dyn Error>>;

    fn create_agent(&self, name: &str, description: &str) -> Result<(), Box<dyn Error>>;
    fn delete_agent(&self, name: &str) -> Result<bool, Box<dyn Error>>;
    fn list_agents(&self) -> Result<Vec<AgentRow>, Box<dyn Error>>;
    fn get_agent(&self, name: &str) -> Result<Option<AgentRow>, Box<dyn Error>>;
    fn ensure_default_agent(&self) -> Result<(), Box<dyn Error>>;

    fn set_agent_data(&self, agent: &str, key: &str, value: &str) -> Result<(), Box<dyn Error>>;
    fn get_agent_data(&self, agent: &str, key: &str) -> Result<Option<String>, Box<dyn Error>>;
    fn delete_agent_data(&self, agent: &str, key: &str) -> Result<bool, Box<dyn Error>>;
    fn list_agent_data(&self, agent: &str) -> Result<Vec<(String, String)>, Box<dyn Error>>;
    fn get_agent_config(&self, agent_name: &str) -> Result<AgentConfig, Box<dyn Error>>;

    fn save_agent_messages(
        &self,
        agent_name: &str,
        messages: &[serde_json::Value],
    ) -> Result<(), Box<dyn Error>>;
    fn load_agent_messages(
        &self,
        agent_name: &str,
    ) -> Result<Vec<serde_json::Value>, Box<dyn Error>>;
    fn append_agent_message(
        &self,
        agent_name: &str,
        message: &serde_json::Value,
    ) -> Result<(), Box<dyn Error>>;
    fn clear_agent_messages(&self, agent_name: &str) -> Result<(), Box<dyn Error>>;
    fn agent_message_count(&self, agent_name: &str) -> Result<i64, Box<dyn Error>>;

    fn send_notification(
        &self,
        from_agent: &str,
        to_agent: &str,
        message: &str,
    ) -> Result<i64, Box<dyn Error>>;
    fn poll_notifications_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<NotificationRow>, Box<dyn Error>>;

    fn create_cron_task(
        &self,
        name: &str,
        description: &str,
        cron_expr: &str,
        command: &str,
        agent_name: Option<&str>,
        max_runs: Option<i64>,
    ) -> Result<i64, Box<dyn Error>>;
    fn create_oneshot_task(
        &self,
        name: &str,
        description: &str,
        run_at: &str,
        command: &str,
        agent_name: Option<&str>,
    ) -> Result<i64, Box<dyn Error>>;
    fn delete_task(&self, task_id: i64) -> Result<bool, Box<dyn Error>>;
    fn list_tasks(&self, agent_name: Option<&str>) -> Result<Vec<TaskRow>, Box<dyn Error>>;
    fn get_task(&self, task_id: i64) -> Result<Option<TaskRow>, Box<dyn Error>>;
    fn set_task_enabled(&self, task_id: i64, enabled: bool) -> Result<bool, Box<dyn Error>>;
    fn get_pending_tasks(&self) -> Result<Vec<TaskRow>, Box<dyn Error>>;
    fn mark_task_executed(
        &self,
        task_id: i64,
        task_type: &str,
        cron_expression: Option<&str>,
        max_runs: Option<i64>,
    ) -> Result<(), Box<dyn Error>>;

    fn gc_agents(&self) -> Result<Vec<String>, Box<dyn Error>>;
}
