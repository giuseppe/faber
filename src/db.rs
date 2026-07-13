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

use cron::Schedule;
use rusqlite::{Connection, params};
use serde::Serialize;
use std::error::Error;
use std::str::FromStr;

#[derive(Serialize)]
pub struct AgentRow {
    pub name: String,
    pub description: String,
    pub created_at: String,
}

#[derive(Serialize)]
pub struct TaskRow {
    pub id: i64,
    pub agent_name: Option<String>,
    pub name: String,
    pub description: String,
    pub task_type: String,
    pub cron_expression: Option<String>,
    pub run_at: Option<String>,
    pub next_run_at: Option<String>,
    pub last_run_at: Option<String>,
    pub enabled: bool,
    pub created_at: String,
}

pub fn initialize_db(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS agents (
            name TEXT PRIMARY KEY NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS agent_data (
            agent_name TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (agent_name, key),
            FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            task_type TEXT NOT NULL CHECK(task_type IN ('cron', 'oneshot')),
            cron_expression TEXT,
            run_at TEXT,
            next_run_at TEXT,
            last_run_at TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_agent_data_agent ON agent_data(agent_name);
        CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON scheduled_tasks(next_run_at)
            WHERE enabled = 1;
        CREATE INDEX IF NOT EXISTS idx_tasks_agent ON scheduled_tasks(agent_name);
        ",
    )
}

// --- Agent CRUD ---

pub fn create_agent(
    conn: &Connection,
    name: &str,
    description: &str,
) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "INSERT INTO agents (name, description) VALUES (?1, ?2)",
        params![name, description],
    )
    .map_err(|e| match e {
        rusqlite::Error::SqliteFailure(ref err, _)
            if err.code == rusqlite::ffi::ErrorCode::ConstraintViolation =>
        {
            format!("Agent '{}' already exists", name).into()
        }
        other => Box::new(other) as Box<dyn Error>,
    })?;
    Ok(())
}

pub fn delete_agent(conn: &Connection, name: &str) -> Result<bool, Box<dyn Error>> {
    let rows = conn.execute("DELETE FROM agents WHERE name = ?1", params![name])?;
    Ok(rows > 0)
}

pub fn list_agents(conn: &Connection) -> Result<Vec<AgentRow>, Box<dyn Error>> {
    let mut stmt =
        conn.prepare("SELECT name, description, created_at FROM agents ORDER BY name")?;
    let rows = stmt.query_map([], |row| {
        Ok(AgentRow {
            name: row.get(0)?,
            description: row.get(1)?,
            created_at: row.get(2)?,
        })
    })?;
    let mut agents = Vec::new();
    for row in rows {
        agents.push(row?);
    }
    Ok(agents)
}

pub fn get_agent(conn: &Connection, name: &str) -> Result<Option<AgentRow>, Box<dyn Error>> {
    let mut stmt =
        conn.prepare("SELECT name, description, created_at FROM agents WHERE name = ?1")?;
    let mut rows = stmt.query_map(params![name], |row| {
        Ok(AgentRow {
            name: row.get(0)?,
            description: row.get(1)?,
            created_at: row.get(2)?,
        })
    })?;
    match rows.next() {
        Some(row) => Ok(Some(row?)),
        None => Ok(None),
    }
}

// --- Agent Data (key-value) ---

pub fn set_agent_data(
    conn: &Connection,
    agent: &str,
    key: &str,
    value: &str,
) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "INSERT INTO agent_data (agent_name, key, value, updated_at)
         VALUES (?1, ?2, ?3, datetime('now'))
         ON CONFLICT(agent_name, key) DO UPDATE SET value = ?3, updated_at = datetime('now')",
        params![agent, key, value],
    )?;
    Ok(())
}

pub fn get_agent_data(
    conn: &Connection,
    agent: &str,
    key: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let mut stmt =
        conn.prepare("SELECT value FROM agent_data WHERE agent_name = ?1 AND key = ?2")?;
    let mut rows = stmt.query_map(params![agent, key], |row| row.get::<_, String>(0))?;
    match rows.next() {
        Some(val) => Ok(Some(val?)),
        None => Ok(None),
    }
}

pub fn delete_agent_data(
    conn: &Connection,
    agent: &str,
    key: &str,
) -> Result<bool, Box<dyn Error>> {
    let rows = conn.execute(
        "DELETE FROM agent_data WHERE agent_name = ?1 AND key = ?2",
        params![agent, key],
    )?;
    Ok(rows > 0)
}

pub fn list_agent_data(
    conn: &Connection,
    agent: &str,
) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let mut stmt =
        conn.prepare("SELECT key, value FROM agent_data WHERE agent_name = ?1 ORDER BY key")?;
    let rows = stmt.query_map(params![agent], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut data = Vec::new();
    for row in rows {
        data.push(row?);
    }
    Ok(data)
}

// --- Scheduled Tasks ---

fn row_to_task(row: &rusqlite::Row) -> rusqlite::Result<TaskRow> {
    Ok(TaskRow {
        id: row.get(0)?,
        agent_name: row.get(1)?,
        name: row.get(2)?,
        description: row.get(3)?,
        task_type: row.get(4)?,
        cron_expression: row.get(5)?,
        run_at: row.get(6)?,
        next_run_at: row.get(7)?,
        last_run_at: row.get(8)?,
        enabled: row.get::<_, i32>(9)? != 0,
        created_at: row.get(10)?,
    })
}

const TASK_COLUMNS: &str = "id, agent_name, name, description, task_type, cron_expression, run_at, next_run_at, last_run_at, enabled, created_at";

pub fn create_cron_task(
    conn: &Connection,
    name: &str,
    description: &str,
    cron_expr: &str,
    agent_name: Option<&str>,
) -> Result<i64, Box<dyn Error>> {
    let schedule = Schedule::from_str(cron_expr)
        .map_err(|e| format!("Invalid cron expression '{}': {}", cron_expr, e))?;

    let next_run = schedule
        .upcoming(chrono::Utc)
        .next()
        .map(|dt| dt.to_rfc3339());

    conn.execute(
        "INSERT INTO scheduled_tasks (name, description, task_type, cron_expression, next_run_at, agent_name)
         VALUES (?1, ?2, 'cron', ?3, ?4, ?5)",
        params![name, description, cron_expr, next_run, agent_name],
    )?;

    Ok(conn.last_insert_rowid())
}

pub fn create_oneshot_task(
    conn: &Connection,
    name: &str,
    description: &str,
    run_at: &str,
    agent_name: Option<&str>,
) -> Result<i64, Box<dyn Error>> {
    // Validate the datetime
    chrono::DateTime::parse_from_rfc3339(run_at)
        .map_err(|e| format!("Invalid RFC 3339 datetime '{}': {}", run_at, e))?;

    conn.execute(
        "INSERT INTO scheduled_tasks (name, description, task_type, run_at, next_run_at, agent_name)
         VALUES (?1, ?2, 'oneshot', ?3, ?3, ?4)",
        params![name, description, run_at, agent_name],
    )?;

    Ok(conn.last_insert_rowid())
}

pub fn delete_task(conn: &Connection, task_id: i64) -> Result<bool, Box<dyn Error>> {
    let rows = conn.execute(
        "DELETE FROM scheduled_tasks WHERE id = ?1",
        params![task_id],
    )?;
    Ok(rows > 0)
}

pub fn list_tasks(
    conn: &Connection,
    agent_name: Option<&str>,
) -> Result<Vec<TaskRow>, Box<dyn Error>> {
    let mut tasks = Vec::new();
    match agent_name {
        Some(agent) => {
            let sql = format!(
                "SELECT {} FROM scheduled_tasks WHERE agent_name = ?1 ORDER BY id",
                TASK_COLUMNS
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params![agent], row_to_task)?;
            for row in rows {
                tasks.push(row?);
            }
        }
        None => {
            let sql = format!("SELECT {} FROM scheduled_tasks ORDER BY id", TASK_COLUMNS);
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map([], row_to_task)?;
            for row in rows {
                tasks.push(row?);
            }
        }
    }
    Ok(tasks)
}

pub fn get_task(conn: &Connection, task_id: i64) -> Result<Option<TaskRow>, Box<dyn Error>> {
    let sql = format!("SELECT {} FROM scheduled_tasks WHERE id = ?1", TASK_COLUMNS);
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query_map(params![task_id], row_to_task)?;
    match rows.next() {
        Some(row) => Ok(Some(row?)),
        None => Ok(None),
    }
}

pub fn set_task_enabled(
    conn: &Connection,
    task_id: i64,
    enabled: bool,
) -> Result<bool, Box<dyn Error>> {
    let rows = conn.execute(
        "UPDATE scheduled_tasks SET enabled = ?1 WHERE id = ?2",
        params![enabled as i32, task_id],
    )?;
    Ok(rows > 0)
}

pub fn get_pending_tasks(conn: &Connection) -> Result<Vec<TaskRow>, Box<dyn Error>> {
    let now = chrono::Utc::now().to_rfc3339();
    let sql = format!(
        "SELECT {} FROM scheduled_tasks WHERE enabled = 1 AND next_run_at <= ?1 ORDER BY next_run_at",
        TASK_COLUMNS
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params![now], row_to_task)?;
    let mut tasks = Vec::new();
    for row in rows {
        tasks.push(row?);
    }
    Ok(tasks)
}
