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

//! Agents defined for this process only (`--agent NAME:CONFIG`): their
//! settings are layered over what the database has for them whenever it's
//! read, and never written to it - so a session against a shared server
//! runs them as given without changing them for anyone else.

use faber::agent_io::{AgentEvent, AgentEventRow, EventFilter};
use faber::db::{
    AgentConfig, AgentRow, KbHit, KbNote, KbViewer, NewTask, NotificationRow, TaskOutcome, TaskRow,
};
use faber::db_backend::DbBackend;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

/// The settings a definition can give: those it gives (`Some`) replace the
/// stored ones. Not whether an agent is unsafe, nor where it works.
macro_rules! defined_fields {
    ($apply:ident) => {
        $apply!(model);
        $apply!(endpoint);
        $apply!(system_prompt);
        $apply!(api_key);
        $apply!(max_tokens);
        $apply!(context_window);
        $apply!(parameters);
        $apply!(tools);
        $apply!(mcp_servers);
    };
}

/// `stored` as the agent runs here: with what `defined` gives in place.
fn layered(mut stored: AgentConfig, defined: &AgentConfig) -> AgentConfig {
    macro_rules! take {
        ($f:ident) => {
            if defined.$f.is_some() {
                stored.$f = defined.$f.clone();
            }
        };
    }
    defined_fields!(take);
    stored
}

/// `config` as it may be stored: what `defined` gives left as `stored` has
/// it, so a config read here (layered) and written back stores none of it.
fn unlayered(mut config: AgentConfig, stored: &AgentConfig, defined: &AgentConfig) -> AgentConfig {
    macro_rules! keep {
        ($f:ident) => {
            if defined.$f.is_some() {
                config.$f = stored.$f.clone();
            }
        };
    }
    defined_fields!(keep);
    config
}

pub(crate) struct LocalAgents {
    pub(crate) inner: Arc<dyn DbBackend>,
    pub(crate) agents: HashMap<String, AgentConfig>,
}

impl DbBackend for LocalAgents {
    fn get_agent_config(&self, agent_name: &str) -> Result<AgentConfig, Box<dyn Error>> {
        let stored = self.inner.get_agent_config(agent_name)?;
        Ok(match self.agents.get(agent_name) {
            Some(defined) => layered(stored, defined),
            None => stored,
        })
    }

    fn set_agent_config(
        &self,
        agent_name: &str,
        config: &AgentConfig,
    ) -> Result<(), Box<dyn Error>> {
        match self.agents.get(agent_name) {
            Some(defined) => {
                let stored = self.inner.get_agent_config(agent_name)?;
                let config = unlayered(config.clone(), &stored, defined);
                self.inner.set_agent_config(agent_name, &config)
            }
            None => self.inner.set_agent_config(agent_name, config),
        }
    }

    fn claim_agent(&self, name: &str, session_id: &str) -> Result<bool, Box<dyn Error>> {
        self.inner.claim_agent(name, session_id)
    }

    fn heartbeat_all(&self, session_id: &str) -> Result<(), Box<dyn Error>> {
        self.inner.heartbeat_all(session_id)
    }

    fn release_agent(&self, name: &str, session_id: &str) -> Result<(), Box<dyn Error>> {
        self.inner.release_agent(name, session_id)
    }

    fn release_all_agents(&self, session_id: &str) -> Result<(), Box<dyn Error>> {
        self.inner.release_all_agents(session_id)
    }

    fn create_agent(&self, name: &str, description: &str) -> Result<(), Box<dyn Error>> {
        self.inner.create_agent(name, description)
    }

    fn delete_agent(&self, name: &str) -> Result<bool, Box<dyn Error>> {
        self.inner.delete_agent(name)
    }

    fn list_agents(&self) -> Result<Vec<AgentRow>, Box<dyn Error>> {
        self.inner.list_agents()
    }

    fn get_agent(&self, name: &str) -> Result<Option<AgentRow>, Box<dyn Error>> {
        self.inner.get_agent(name)
    }

    fn set_agent_activity(&self, agent: &str, activity: &str) -> Result<(), Box<dyn Error>> {
        self.inner.set_agent_activity(agent, activity)
    }

    fn agent_lineage(&self, agent: &str) -> Result<Vec<String>, Box<dyn Error>> {
        self.inner.agent_lineage(agent)
    }

    fn set_agent_parent(&self, agent: &str, parent: Option<&str>) -> Result<(), Box<dyn Error>> {
        self.inner.set_agent_parent(agent, parent)
    }

    fn ensure_default_agent(&self) -> Result<(), Box<dyn Error>> {
        self.inner.ensure_default_agent()
    }

    fn set_agent_data(&self, agent: &str, key: &str, value: &str) -> Result<(), Box<dyn Error>> {
        self.inner.set_agent_data(agent, key, value)
    }

    fn get_agent_data(&self, agent: &str, key: &str) -> Result<Option<String>, Box<dyn Error>> {
        self.inner.get_agent_data(agent, key)
    }

    fn delete_agent_data(&self, agent: &str, key: &str) -> Result<bool, Box<dyn Error>> {
        self.inner.delete_agent_data(agent, key)
    }

    fn save_agent_messages(
        &self,
        agent_name: &str,
        messages: &[serde_json::Value],
    ) -> Result<(), Box<dyn Error>> {
        self.inner.save_agent_messages(agent_name, messages)
    }

    fn load_agent_messages(
        &self,
        agent_name: &str,
    ) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
        self.inner.load_agent_messages(agent_name)
    }

    fn append_agent_message(
        &self,
        agent_name: &str,
        message: &serde_json::Value,
    ) -> Result<(), Box<dyn Error>> {
        self.inner.append_agent_message(agent_name, message)
    }

    fn clear_agent_messages(&self, agent_name: &str) -> Result<(), Box<dyn Error>> {
        self.inner.clear_agent_messages(agent_name)
    }

    fn agent_message_count(&self, agent_name: &str) -> Result<i64, Box<dyn Error>> {
        self.inner.agent_message_count(agent_name)
    }

    fn send_notification(
        &self,
        from_agent: &str,
        to_agent: &str,
        message: &str,
    ) -> Result<i64, Box<dyn Error>> {
        self.inner.send_notification(from_agent, to_agent, message)
    }

    fn poll_notifications_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<NotificationRow>, Box<dyn Error>> {
        self.inner.poll_notifications_for_session(session_id)
    }

    fn create_task(&self, task: &NewTask) -> Result<i64, Box<dyn Error>> {
        self.inner.create_task(task)
    }

    fn delete_task(&self, task_id: i64) -> Result<bool, Box<dyn Error>> {
        self.inner.delete_task(task_id)
    }

    fn list_tasks(&self, agent_name: Option<&str>) -> Result<Vec<TaskRow>, Box<dyn Error>> {
        self.inner.list_tasks(agent_name)
    }

    fn get_task(&self, task_id: i64) -> Result<Option<TaskRow>, Box<dyn Error>> {
        self.inner.get_task(task_id)
    }

    fn set_task_enabled(&self, task_id: i64, enabled: bool) -> Result<bool, Box<dyn Error>> {
        self.inner.set_task_enabled(task_id, enabled)
    }

    fn update_task(&self, task_id: i64, task: &NewTask) -> Result<bool, Box<dyn Error>> {
        self.inner.update_task(task_id, task)
    }

    fn set_profile(&self, name: &str, settings: &serde_json::Value) -> Result<(), Box<dyn Error>> {
        self.inner.set_profile(name, settings)
    }

    fn delete_profile(&self, name: &str) -> Result<bool, Box<dyn Error>> {
        self.inner.delete_profile(name)
    }

    fn set_agent_description(
        &self,
        agent: &str,
        description: &str,
    ) -> Result<bool, Box<dyn Error>> {
        self.inner.set_agent_description(agent, description)
    }

    fn list_profiles(&self) -> Result<Vec<(String, serde_json::Value)>, Box<dyn Error>> {
        self.inner.list_profiles()
    }

    fn request_task_stop(&self, task_id: i64) -> Result<bool, Box<dyn Error>> {
        self.inner.request_task_stop(task_id)
    }

    fn run_task_now(&self, task_id: i64) -> Result<bool, Box<dyn Error>> {
        self.inner.run_task_now(task_id)
    }

    fn set_task_held(&self, task_id: i64, held: bool) -> Result<bool, Box<dyn Error>> {
        self.inner.set_task_held(task_id, held)
    }

    fn set_task_target(
        &self,
        task_id: i64,
        agent: Option<&str>,
        profile: Option<&str>,
    ) -> Result<bool, Box<dyn Error>> {
        self.inner.set_task_target(task_id, agent, profile)
    }

    fn get_pending_tasks(&self) -> Result<Vec<TaskRow>, Box<dyn Error>> {
        self.inner.get_pending_tasks()
    }

    fn prune_tasks(&self, older_than: &str, dry_run: bool) -> Result<Vec<TaskRow>, Box<dyn Error>> {
        self.inner.prune_tasks(older_than, dry_run)
    }

    fn fail_tasks_with_failed_dependencies(&self) -> Result<Vec<i64>, Box<dyn Error>> {
        self.inner.fail_tasks_with_failed_dependencies()
    }

    fn claim_task(&self, task_id: i64, session_id: &str) -> Result<bool, Box<dyn Error>> {
        self.inner.claim_task(task_id, session_id)
    }

    fn finish_task(
        &self,
        task_id: i64,
        session_id: &str,
        outcome: &TaskOutcome,
    ) -> Result<bool, Box<dyn Error>> {
        self.inner.finish_task(task_id, session_id, outcome)
    }

    fn gc_agents(&self) -> Result<Vec<String>, Box<dyn Error>> {
        self.inner.gc_agents()
    }

    fn append_agent_events(
        &self,
        agent: &str,
        task_id: Option<i64>,
        events: &[AgentEvent],
    ) -> Result<(), Box<dyn Error>> {
        self.inner.append_agent_events(agent, task_id, events)
    }

    fn agent_events(&self, filter: &EventFilter) -> Result<Vec<AgentEventRow>, Box<dyn Error>> {
        self.inner.agent_events(filter)
    }

    fn kb_write(
        &self,
        title: &str,
        body: &str,
        tags: &[String],
        private_to: Option<&str>,
        author: Option<&str>,
    ) -> Result<(i64, bool), Box<dyn Error>> {
        self.inner.kb_write(title, body, tags, private_to, author)
    }

    fn kb_get(&self, id: i64, viewer: &KbViewer) -> Result<Option<KbNote>, Box<dyn Error>> {
        self.inner.kb_get(id, viewer)
    }

    fn kb_get_by_title(
        &self,
        title: &str,
        viewer: &KbViewer,
    ) -> Result<Option<KbNote>, Box<dyn Error>> {
        self.inner.kb_get_by_title(title, viewer)
    }

    fn kb_search(
        &self,
        query: &str,
        viewer: &KbViewer,
        tag: Option<&str>,
        limit: usize,
    ) -> Result<Vec<KbHit>, Box<dyn Error>> {
        self.inner.kb_search(query, viewer, tag, limit)
    }

    fn kb_list(
        &self,
        viewer: &KbViewer,
        tag: Option<&str>,
        limit: usize,
    ) -> Result<Vec<KbNote>, Box<dyn Error>> {
        self.inner.kb_list(viewer, tag, limit)
    }

    fn kb_delete(&self, id: i64, viewer: &KbViewer) -> Result<bool, Box<dyn Error>> {
        self.inner.kb_delete(id, viewer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defined_settings_are_used_but_not_stored() {
        let stored = AgentConfig {
            model: Some("stored".to_string()),
            system_prompt: Some("be brief".to_string()),
            cwd: Some("/work".to_string()),
            ..Default::default()
        };
        let defined = AgentConfig {
            model: Some("mine".to_string()),
            endpoint: Some("http://mine/v1".to_string()),
            ..Default::default()
        };
        let running = layered(stored.clone(), &defined);
        assert_eq!(running.model.as_deref(), Some("mine"));
        assert_eq!(running.endpoint.as_deref(), Some("http://mine/v1"));
        assert_eq!(running.system_prompt.as_deref(), Some("be brief"));

        // Written back with a change of its own: only that is stored.
        let mut changed = running;
        changed.cwd = Some("/elsewhere".to_string());
        let written = unlayered(changed, &stored, &defined);
        assert_eq!(written.model.as_deref(), Some("stored"));
        assert_eq!(written.endpoint, None);
        assert_eq!(written.cwd.as_deref(), Some("/elsewhere"));
    }
}
