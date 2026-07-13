/*
 * codehawk
 *
 * Copyright (C) 2025 Giuseppe Scrivano <giuseppe@scrivano.org>
 * codehawk is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * codehawk is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with codehawk.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

use std::any::Any;
use std::sync::{Arc, Mutex};

pub mod db;
pub mod github;
pub mod openai;

pub struct ToolContext {
    pub println: Box<dyn Fn(&str) + Send + Sync>,
    pub db: Option<Arc<Mutex<rusqlite::Connection>>>,
    pub agent_name: Option<String>,
    pub extra: Option<Arc<dyn Any + Send + Sync>>,
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
        }
    }

    pub fn println(&self, msg: &str) {
        (self.println)(msg);
    }

    pub fn db_conn(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, rusqlite::Connection>, Box<dyn std::error::Error>> {
        self.db
            .as_ref()
            .ok_or_else(|| {
                "Database not configured. Set 'db_path' in your config file."
                    .to_string()
                    .into()
            })
            .and_then(|arc| {
                arc.lock()
                    .map_err(|e| format!("DB lock poisoned: {}", e).into())
            })
    }
}
