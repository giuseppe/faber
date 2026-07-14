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

use cron::Schedule;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
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
    pub command: String,
    pub max_runs: Option<i64>,
    pub run_count: i64,
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

        CREATE TABLE IF NOT EXISTS agent_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            seq INTEGER NOT NULL,
            message_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_agent TEXT NOT NULL,
            to_agent TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (from_agent) REFERENCES agents(name) ON DELETE CASCADE,
            FOREIGN KEY (to_agent) REFERENCES agents(name) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_agent_data_agent ON agent_data(agent_name);
        CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON scheduled_tasks(next_run_at)
            WHERE enabled = 1;
        CREATE INDEX IF NOT EXISTS idx_tasks_agent ON scheduled_tasks(agent_name);
        CREATE INDEX IF NOT EXISTS idx_agent_messages_agent_seq ON agent_messages(agent_name, seq);
        CREATE INDEX IF NOT EXISTS idx_notifications_to_agent ON notifications(to_agent);
        ",
    )?;

    let _ = conn
        .execute_batch("ALTER TABLE scheduled_tasks ADD COLUMN command TEXT NOT NULL DEFAULT '';");
    let _ =
        conn.execute_batch("ALTER TABLE scheduled_tasks ADD COLUMN max_runs INTEGER DEFAULT NULL;");
    let _ = conn.execute_batch(
        "ALTER TABLE scheduled_tasks ADD COLUMN run_count INTEGER NOT NULL DEFAULT 0;",
    );

    let _ = conn.execute_batch("ALTER TABLE agents ADD COLUMN session_id TEXT DEFAULT NULL;");
    let _ = conn.execute_batch("ALTER TABLE agents ADD COLUMN heartbeat_at TEXT DEFAULT NULL;");

    Ok(())
}

// --- Session ownership ---

pub fn claim_agent(
    conn: &Connection,
    name: &str,
    session_id: &str,
) -> Result<bool, Box<dyn Error>> {
    let rows = conn.execute(
        "UPDATE agents SET session_id = ?2, heartbeat_at = datetime('now')
         WHERE name = ?1 AND (session_id IS NULL OR session_id = ?2
         OR heartbeat_at IS NULL OR heartbeat_at < datetime('now', '-10 seconds'))",
        params![name, session_id],
    )?;
    Ok(rows > 0)
}

pub fn heartbeat_all(conn: &Connection, session_id: &str) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "UPDATE agents SET heartbeat_at = datetime('now') WHERE session_id = ?1",
        params![session_id],
    )?;
    Ok(())
}

pub fn release_agent(
    conn: &Connection,
    name: &str,
    session_id: &str,
) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "UPDATE agents SET session_id = NULL, heartbeat_at = NULL
         WHERE name = ?1 AND session_id = ?2",
        params![name, session_id],
    )?;
    Ok(())
}

pub fn release_all_agents(conn: &Connection, session_id: &str) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "UPDATE agents SET session_id = NULL, heartbeat_at = NULL WHERE session_id = ?1",
        params![session_id],
    )?;
    Ok(())
}

pub fn poll_notifications_for_session(
    conn: &Connection,
    session_id: &str,
) -> Result<Vec<NotificationRow>, Box<dyn Error>> {
    conn.execute("BEGIN IMMEDIATE", [])?;
    let result = (|| -> Result<Vec<NotificationRow>, Box<dyn Error>> {
        let mut stmt = conn.prepare(
            "SELECT n.id, n.from_agent, n.to_agent, n.message, n.created_at
             FROM notifications n
             INNER JOIN agents a ON n.to_agent = a.name
             WHERE a.session_id = ?1 AND a.heartbeat_at >= datetime('now', '-10 seconds')
             ORDER BY n.id",
        )?;
        let rows = stmt.query_map(params![session_id], |row| {
            Ok(NotificationRow {
                id: row.get(0)?,
                from_agent: row.get(1)?,
                to_agent: row.get(2)?,
                message: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        let mut notifications = Vec::new();
        for row in rows {
            notifications.push(row?);
        }
        drop(stmt);
        if !notifications.is_empty() {
            let ids: Vec<String> = notifications.iter().map(|n| n.id.to_string()).collect();
            conn.execute(
                &format!("DELETE FROM notifications WHERE id IN ({})", ids.join(",")),
                [],
            )?;
        }
        Ok(notifications)
    })();
    match result {
        Ok(notifications) => {
            conn.execute("COMMIT", [])?;
            Ok(notifications)
        }
        Err(e) => {
            let _ = conn.execute("ROLLBACK", []);
            Err(e)
        }
    }
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
        command: row.get(11)?,
        max_runs: row.get(12)?,
        run_count: row.get(13)?,
    })
}

const TASK_COLUMNS: &str = "id, agent_name, name, description, task_type, cron_expression, run_at, next_run_at, last_run_at, enabled, created_at, command, max_runs, run_count";

pub fn create_cron_task(
    conn: &Connection,
    name: &str,
    description: &str,
    cron_expr: &str,
    command: &str,
    agent_name: Option<&str>,
    max_runs: Option<i64>,
) -> Result<i64, Box<dyn Error>> {
    let schedule = Schedule::from_str(cron_expr)
        .map_err(|e| format!("Invalid cron expression '{}': {}", cron_expr, e))?;

    let next_run = schedule
        .upcoming(chrono::Utc)
        .next()
        .map(|dt| dt.to_rfc3339());

    conn.execute(
        "INSERT INTO scheduled_tasks (name, description, task_type, cron_expression, next_run_at, agent_name, command, max_runs)
         VALUES (?1, ?2, 'cron', ?3, ?4, ?5, ?6, ?7)",
        params![name, description, cron_expr, next_run, agent_name, command, max_runs],
    )?;

    Ok(conn.last_insert_rowid())
}

pub fn create_oneshot_task(
    conn: &Connection,
    name: &str,
    description: &str,
    run_at: &str,
    command: &str,
    agent_name: Option<&str>,
) -> Result<i64, Box<dyn Error>> {
    chrono::DateTime::parse_from_rfc3339(run_at)
        .map_err(|e| format!("Invalid RFC 3339 datetime '{}': {}", run_at, e))?;

    conn.execute(
        "INSERT INTO scheduled_tasks (name, description, task_type, run_at, next_run_at, agent_name, command)
         VALUES (?1, ?2, 'oneshot', ?3, ?3, ?4, ?5)",
        params![name, description, run_at, agent_name, command],
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

pub fn mark_task_executed(
    conn: &Connection,
    task_id: i64,
    task_type: &str,
    cron_expression: Option<&str>,
    max_runs: Option<i64>,
) -> Result<(), Box<dyn Error>> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE scheduled_tasks SET run_count = run_count + 1 WHERE id = ?1",
        params![task_id],
    )?;

    if task_type == "cron" {
        if let Some(expr) = cron_expression {
            let schedule = Schedule::from_str(expr)?;
            let next_run = schedule
                .upcoming(chrono::Utc)
                .next()
                .map(|dt| dt.to_rfc3339());
            conn.execute(
                "UPDATE scheduled_tasks SET last_run_at = ?1, next_run_at = ?2 WHERE id = ?3",
                params![now, next_run, task_id],
            )?;
        }
    } else {
        conn.execute(
            "UPDATE scheduled_tasks SET last_run_at = ?1, enabled = 0 WHERE id = ?2",
            params![now, task_id],
        )?;
    }

    if let Some(max) = max_runs {
        conn.execute(
            "UPDATE scheduled_tasks SET enabled = 0 WHERE id = ?1 AND run_count >= ?2",
            params![task_id, max],
        )?;
    }

    Ok(())
}

// --- Agent Config ---

pub struct AgentConfig {
    pub model: Option<String>,
    pub endpoint: Option<String>,
    pub system_prompt: Option<String>,
}

pub fn get_agent_config(
    conn: &Connection,
    agent_name: &str,
) -> Result<AgentConfig, Box<dyn Error>> {
    Ok(AgentConfig {
        model: get_agent_data(conn, agent_name, "config:model")?,
        endpoint: get_agent_data(conn, agent_name, "config:endpoint")?,
        system_prompt: get_agent_data(conn, agent_name, "config:system_prompt")?,
    })
}

// --- Agent Messages ---

pub fn ensure_default_agent(conn: &Connection) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "INSERT OR IGNORE INTO agents (name, description) VALUES ('default', 'Default agent')",
        [],
    )?;
    Ok(())
}

pub fn save_agent_messages<T: Serialize>(
    conn: &Connection,
    agent_name: &str,
    messages: &[T],
) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "DELETE FROM agent_messages WHERE agent_name = ?1",
        params![agent_name],
    )?;
    let mut stmt = conn.prepare(
        "INSERT INTO agent_messages (agent_name, seq, message_json) VALUES (?1, ?2, ?3)",
    )?;
    for (i, msg) in messages.iter().enumerate() {
        let json = serde_json::to_string(msg)?;
        stmt.execute(params![agent_name, i as i64, json])?;
    }
    Ok(())
}

pub fn load_agent_messages<T: for<'de> Deserialize<'de>>(
    conn: &Connection,
    agent_name: &str,
) -> Result<Vec<T>, Box<dyn Error>> {
    let mut stmt =
        conn.prepare("SELECT message_json FROM agent_messages WHERE agent_name = ?1 ORDER BY seq")?;
    let rows = stmt.query_map(params![agent_name], |row| row.get::<_, String>(0))?;
    let mut messages = Vec::new();
    for row in rows {
        let json = row?;
        let msg: T = serde_json::from_str(&json)?;
        messages.push(msg);
    }
    Ok(messages)
}

pub fn append_agent_message<T: Serialize>(
    conn: &Connection,
    agent_name: &str,
    message: &T,
) -> Result<(), Box<dyn Error>> {
    let json = serde_json::to_string(message)?;
    conn.execute(
        "INSERT INTO agent_messages (agent_name, seq, message_json)
         VALUES (?1, COALESCE((SELECT MAX(seq) FROM agent_messages WHERE agent_name = ?1), -1) + 1, ?2)",
        params![agent_name, json],
    )?;
    Ok(())
}

pub fn clear_agent_messages(conn: &Connection, agent_name: &str) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "DELETE FROM agent_messages WHERE agent_name = ?1",
        params![agent_name],
    )?;
    Ok(())
}

pub fn agent_message_count(conn: &Connection, agent_name: &str) -> Result<i64, Box<dyn Error>> {
    let mut stmt = conn.prepare("SELECT COUNT(*) FROM agent_messages WHERE agent_name = ?1")?;
    let count: i64 = stmt.query_row(params![agent_name], |row| row.get(0))?;
    Ok(count)
}

// --- Notifications ---

#[allow(dead_code)]
pub struct NotificationRow {
    pub id: i64,
    pub from_agent: String,
    pub to_agent: String,
    pub message: String,
    pub created_at: String,
}

pub fn send_notification(
    conn: &Connection,
    from_agent: &str,
    to_agent: &str,
    message: &str,
) -> Result<i64, Box<dyn Error>> {
    conn.execute(
        "INSERT INTO notifications (from_agent, to_agent, message) VALUES (?1, ?2, ?3)",
        params![from_agent, to_agent, message],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn gc_agents(conn: &Connection) -> Result<Vec<String>, Box<dyn Error>> {
    let mut stmt = conn.prepare(
        "SELECT name FROM agents
         WHERE name != 'default'
         AND (session_id IS NULL OR heartbeat_at IS NULL
              OR heartbeat_at < datetime('now', '-10 seconds'))",
    )?;
    let names: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    drop(stmt);
    if !names.is_empty() {
        conn.execute(
            "DELETE FROM agents
             WHERE name != 'default'
             AND (session_id IS NULL OR heartbeat_at IS NULL
                  OR heartbeat_at < datetime('now', '-10 seconds'))",
            [],
        )?;
    }
    Ok(names)
}
