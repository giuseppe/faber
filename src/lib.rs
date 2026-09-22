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

use std::any::Any;
use std::sync::Arc;

pub mod db;
pub mod db_backend;
pub mod dummy_llm;
pub mod github;
pub mod local_db;
pub mod mcp;
pub mod openai;
#[allow(dead_code)]
pub mod protocol;

pub struct ToolContext {
    pub println: Box<dyn Fn(&str) + Send + Sync>,
    pub db: Option<Arc<dyn db_backend::DbBackend>>,
    pub agent_name: Option<String>,
    pub extra: Option<Arc<dyn Any + Send + Sync>>,
    pub mcp_manager: Option<Arc<mcp::McpManager>>,
}

impl ToolContext {
    pub fn new<F>(println_fn: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        Self {
            println: Box::new(println_fn),
            db: None,
            agent_name: None,
            extra: None,
            mcp_manager: None,
        }
    }

    pub fn println(&self, msg: &str) {
        (self.println)(msg);
    }

    pub fn db(&self) -> Result<&dyn db_backend::DbBackend, Box<dyn std::error::Error>> {
        self.db.as_ref().map(|arc| arc.as_ref()).ok_or_else(|| {
            "Database not configured. Set 'db_path' in your config file."
                .to_string()
                .into()
        })
    }
}
