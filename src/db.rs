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

use cron::Schedule;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::str::FromStr;

#[derive(Serialize, Deserialize)]
pub struct AgentRow {
    pub name: String,
    pub description: String,
    pub created_at: String,
    pub session_id: Option<String>,
    pub heartbeat_at: Option<String>,
}

#[derive(Serialize, Deserialize)]
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
    /// `scheduled`, `running`, `done` or `disabled` - see `TaskStatus`.
    pub status: String,
    pub created_at: String,
    pub command: String,
    pub max_runs: Option<i64>,
    pub run_count: i64,
    /// The session running it, while `running`.
    pub claimed_by: Option<String>,
    /// When the current (or last) run started.
    pub started_at: Option<String>,
    /// `succeeded` or `failed`, for the last finished run.
    pub last_outcome: Option<String>,
    pub last_exit_code: Option<i64>,
    /// The last run's output (truncated), or why it failed.
    pub last_result: Option<String>,
}

/// A task's lifecycle:
///
/// - `scheduled`: waiting for `next_run_at`, then picked up by whichever
///   session's scheduler claims it first (`claim_task`).
/// - `running`: claimed by `claimed_by`. If that session stops
///   heartbeating, the claim is abandoned and the task can be claimed again.
/// - back to `scheduled` after a cron run (`finish_task`), or `done` after
///   a one-shot run or a cron task's last allowed run (`max_runs`).
/// - `disabled`: switched off by the user; switching it back on makes it
///   `scheduled` again.
///
/// Whether a run succeeded is separate (`last_outcome`): a cron task whose
/// last run failed is still `scheduled` for its next one.
pub struct TaskStatus;

impl TaskStatus {
    pub const SCHEDULED: &'static str = "scheduled";
    pub const RUNNING: &'static str = "running";
    pub const DONE: &'static str = "done";
    pub const DISABLED: &'static str = "disabled";
}

/// How a task's run went, recorded by `finish_task`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TaskOutcome {
    pub succeeded: bool,
    pub exit_code: Option<i64>,
    pub result: String,
}

/// SQL condition: the task's claim belongs to a session that's no longer
/// heartbeating (or no longer exists) - its run was abandoned. Sessions
/// heartbeat every 5 seconds.
const CLAIM_ABANDONED: &str = "claimed_by IS NULL OR claimed_by NOT IN (
    SELECT session_id FROM agents
    WHERE session_id IS NOT NULL AND heartbeat_at >= datetime('now', '-30 seconds'))";

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
            status TEXT NOT NULL DEFAULT 'scheduled'
                CHECK(status IN ('scheduled', 'running', 'done', 'disabled')),
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
        CREATE INDEX IF NOT EXISTS idx_tasks_agent ON scheduled_tasks(agent_name);
        CREATE INDEX IF NOT EXISTS idx_agent_messages_agent_seq ON agent_messages(agent_name, seq);
        CREATE INDEX IF NOT EXISTS idx_notifications_to_agent ON notifications(to_agent);

        CREATE TABLE IF NOT EXISTS readline_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        ",
    )?;

    let _ = conn
        .execute_batch("ALTER TABLE scheduled_tasks ADD COLUMN command TEXT NOT NULL DEFAULT '';");
    let _ =
        conn.execute_batch("ALTER TABLE scheduled_tasks ADD COLUMN max_runs INTEGER DEFAULT NULL;");
    let _ = conn.execute_batch(
        "ALTER TABLE scheduled_tasks ADD COLUMN run_count INTEGER NOT NULL DEFAULT 0;",
    );

    for column in [
        "claimed_by TEXT DEFAULT NULL",
        "started_at TEXT DEFAULT NULL",
        "last_outcome TEXT DEFAULT NULL",
        "last_exit_code INTEGER DEFAULT NULL",
        "last_result TEXT DEFAULT NULL",
    ] {
        let _ = conn.execute_batch(&format!(
            "ALTER TABLE scheduled_tasks ADD COLUMN {};",
            column
        ));
    }
    migrate_task_enabled_to_status(conn)?;
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_tasks_due ON scheduled_tasks(status, next_run_at);",
    )?;

    let _ = conn.execute_batch("ALTER TABLE agents ADD COLUMN session_id TEXT DEFAULT NULL;");
    let _ = conn.execute_batch("ALTER TABLE agents ADD COLUMN heartbeat_at TEXT DEFAULT NULL;");

    Ok(())
}

fn table_has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, rusqlite::Error> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", table))?;
    let names = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for name in names {
        if name? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Replaces the old `enabled` flag of databases created before tasks had a
/// `status`: enabled tasks are `scheduled`; disabled ones are `done` if
/// they had run their course (a one-shot that ran, a cron task at its
/// `max_runs`), `disabled` otherwise.
fn migrate_task_enabled_to_status(conn: &Connection) -> Result<(), rusqlite::Error> {
    if !table_has_column(conn, "scheduled_tasks", "enabled")? {
        return Ok(());
    }
    conn.execute_batch(
        "BEGIN;
         ALTER TABLE scheduled_tasks ADD COLUMN status TEXT NOT NULL DEFAULT 'scheduled'
             CHECK(status IN ('scheduled', 'running', 'done', 'disabled'));
         UPDATE scheduled_tasks SET status = CASE
             WHEN enabled = 1 THEN 'scheduled'
             WHEN run_count > 0 AND (task_type = 'oneshot'
                 OR (max_runs IS NOT NULL AND run_count >= max_runs)) THEN 'done'
             ELSE 'disabled' END;
         DROP INDEX IF EXISTS idx_tasks_next_run;
         ALTER TABLE scheduled_tasks DROP COLUMN enabled;
         COMMIT;",
    )
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
    let tx = conn.unchecked_transaction()?;
    tx.execute(
        "DELETE FROM scheduled_tasks WHERE agent_name = ?1",
        params![name],
    )?;
    let rows = tx.execute("DELETE FROM agents WHERE name = ?1", params![name])?;
    tx.commit()?;
    Ok(rows > 0)
}

fn row_to_agent(row: &rusqlite::Row) -> rusqlite::Result<AgentRow> {
    Ok(AgentRow {
        name: row.get(0)?,
        description: row.get(1)?,
        created_at: row.get(2)?,
        session_id: row.get(3)?,
        heartbeat_at: row.get(4)?,
    })
}

const AGENT_COLUMNS: &str = "name, description, created_at, session_id, heartbeat_at";

pub fn list_agents(conn: &Connection) -> Result<Vec<AgentRow>, Box<dyn Error>> {
    let sql = format!("SELECT {} FROM agents ORDER BY name", AGENT_COLUMNS);
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], row_to_agent)?;
    let mut agents = Vec::new();
    for row in rows {
        agents.push(row?);
    }
    Ok(agents)
}

pub fn get_agent(conn: &Connection, name: &str) -> Result<Option<AgentRow>, Box<dyn Error>> {
    let sql = format!("SELECT {} FROM agents WHERE name = ?1", AGENT_COLUMNS);
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query_map(params![name], row_to_agent)?;
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
        status: row.get(9)?,
        created_at: row.get(10)?,
        command: row.get(11)?,
        max_runs: row.get(12)?,
        run_count: row.get(13)?,
        claimed_by: row.get(14)?,
        started_at: row.get(15)?,
        last_outcome: row.get(16)?,
        last_exit_code: row.get(17)?,
        last_result: row.get(18)?,
    })
}

const TASK_COLUMNS: &str = "id, agent_name, name, description, task_type, cron_expression, run_at, next_run_at, last_run_at, status, created_at, command, max_runs, run_count, claimed_by, started_at, last_outcome, last_exit_code, last_result";

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
    // Disabling a running task lets the run finish but keeps it disabled
    // (see `finish_task`); a task that's `done` stays done.
    let rows = if enabled {
        conn.execute(
            "UPDATE scheduled_tasks SET status = 'scheduled' WHERE id = ?1 AND status = 'disabled'",
            params![task_id],
        )?
    } else {
        conn.execute(
            "UPDATE scheduled_tasks SET status = 'disabled'
             WHERE id = ?1 AND status IN ('scheduled', 'running')",
            params![task_id],
        )?
    };
    Ok(rows > 0)
}

pub fn get_pending_tasks(conn: &Connection) -> Result<Vec<TaskRow>, Box<dyn Error>> {
    let now = chrono::Utc::now().to_rfc3339();
    let sql = format!(
        "SELECT {} FROM scheduled_tasks
         WHERE next_run_at <= ?1
           AND (status = 'scheduled' OR (status = 'running' AND ({})))
         ORDER BY next_run_at",
        TASK_COLUMNS, CLAIM_ABANDONED
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params![now], row_to_task)?;
    let mut tasks = Vec::new();
    for row in rows {
        tasks.push(row?);
    }
    Ok(tasks)
}

/// Claims a due task for `session_id` to run. Returns false if it isn't due
/// any more or another session got to it first - the check and the claim
/// are one statement, so only one session can ever win. A task whose
/// claim was abandoned (see `CLAIM_ABANDONED`) can be claimed again.
pub fn claim_task(
    conn: &Connection,
    task_id: i64,
    session_id: &str,
) -> Result<bool, Box<dyn Error>> {
    let now = chrono::Utc::now().to_rfc3339();
    let sql = format!(
        "UPDATE scheduled_tasks SET status = 'running', claimed_by = ?2, started_at = ?3
         WHERE id = ?1 AND next_run_at <= ?3
           AND (status = 'scheduled' OR (status = 'running' AND ({})))",
        CLAIM_ABANDONED
    );
    let rows = conn.execute(&sql, params![task_id, session_id, now])?;
    Ok(rows > 0)
}

/// Records the outcome of a run of a task claimed by `session_id`, and
/// moves it on: a cron task is `scheduled` for its next run, a one-shot
/// task or a cron task that reached `max_runs` is `done`. A task disabled
/// while it ran stays disabled. Returns false (recording nothing) if the
/// claim isn't `session_id`'s any more.
pub fn finish_task(
    conn: &Connection,
    task_id: i64,
    session_id: &str,
    outcome: &TaskOutcome,
) -> Result<bool, Box<dyn Error>> {
    let Some(task) = get_task(conn, task_id)? else {
        return Ok(false);
    };
    let runs = task.run_count + 1;
    let next_run = match (&task.task_type[..], &task.cron_expression) {
        ("cron", Some(expr)) => Schedule::from_str(expr)?
            .upcoming(chrono::Utc)
            .next()
            .map(|dt| dt.to_rfc3339()),
        _ => None,
    };
    let finished = next_run.is_none() || task.max_runs.is_some_and(|max| runs >= max);
    let status = if finished {
        TaskStatus::DONE
    } else {
        TaskStatus::SCHEDULED
    };
    let rows = conn.execute(
        "UPDATE scheduled_tasks SET
             status = CASE WHEN status = 'running' THEN ?3 ELSE status END,
             run_count = run_count + 1,
             last_run_at = ?4,
             next_run_at = COALESCE(?5, next_run_at),
             last_outcome = ?6,
             last_exit_code = ?7,
             last_result = ?8,
             claimed_by = NULL
         WHERE id = ?1 AND claimed_by = ?2",
        params![
            task_id,
            session_id,
            status,
            chrono::Utc::now().to_rfc3339(),
            next_run,
            if outcome.succeeded {
                "succeeded"
            } else {
                "failed"
            },
            outcome.exit_code,
            outcome.result,
        ],
    )?;
    Ok(rows > 0)
}

// --- Agent Config ---

#[derive(Serialize, Deserialize)]
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
    let tx = conn.unchecked_transaction()?;
    tx.execute(
        "DELETE FROM agent_messages WHERE agent_name = ?1",
        params![agent_name],
    )?;
    let mut stmt = tx.prepare(
        "INSERT INTO agent_messages (agent_name, seq, message_json) VALUES (?1, ?2, ?3)",
    )?;
    for (i, msg) in messages.iter().enumerate() {
        let json = serde_json::to_string(msg)?;
        stmt.execute(params![agent_name, i as i64, json])?;
    }
    drop(stmt);
    tx.commit()?;
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

#[derive(Serialize, Deserialize)]
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
    // Both columns are foreign keys into `agents`; checking first turns a
    // cryptic "FOREIGN KEY constraint failed" (no indication which key, or
    // why) into a message that actually says what's wrong.
    if get_agent(conn, to_agent)?.is_none() {
        return Err(format!(
            "Agent '{}' does not exist. Use agent_list to see available agents.",
            to_agent
        )
        .into());
    }
    if get_agent(conn, from_agent)?.is_none() {
        return Err(format!("Sending agent '{}' no longer exists", from_agent).into());
    }
    conn.execute(
        "INSERT INTO notifications (from_agent, to_agent, message) VALUES (?1, ?2, ?3)",
        params![from_agent, to_agent, message],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn gc_agents(conn: &Connection) -> Result<Vec<String>, Box<dyn Error>> {
    let tx = conn.unchecked_transaction()?;
    let mut stmt = tx.prepare(
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
        tx.execute(
            "DELETE FROM agents
             WHERE name != 'default'
             AND (session_id IS NULL OR heartbeat_at IS NULL
                  OR heartbeat_at < datetime('now', '-10 seconds'))",
            [],
        )?;
    }
    tx.commit()?;
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        initialize_db(&conn).unwrap();
        conn
    }

    #[test]
    fn test_create_and_get_agent() {
        let conn = test_db();
        create_agent(&conn, "alice", "Test agent").unwrap();
        let agent = get_agent(&conn, "alice").unwrap().unwrap();
        assert_eq!(agent.name, "alice");
        assert_eq!(agent.description, "Test agent");
    }

    #[test]
    fn test_create_duplicate_agent_fails() {
        let conn = test_db();
        create_agent(&conn, "alice", "first").unwrap();
        let result = create_agent(&conn, "alice", "second");
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_agent() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        assert!(delete_agent(&conn, "alice").unwrap());
        assert!(get_agent(&conn, "alice").unwrap().is_none());
    }

    #[test]
    fn test_delete_nonexistent_agent() {
        let conn = test_db();
        assert!(!delete_agent(&conn, "nobody").unwrap());
    }

    #[test]
    fn test_list_agents() {
        let conn = test_db();
        create_agent(&conn, "bob", "").unwrap();
        create_agent(&conn, "alice", "").unwrap();
        let agents = list_agents(&conn).unwrap();
        let names: Vec<&str> = agents.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["alice", "bob"]);
    }

    #[test]
    fn test_agent_data_crud() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();

        set_agent_data(&conn, "alice", "color", "blue").unwrap();
        assert_eq!(
            get_agent_data(&conn, "alice", "color").unwrap(),
            Some("blue".to_string())
        );

        set_agent_data(&conn, "alice", "color", "red").unwrap();
        assert_eq!(
            get_agent_data(&conn, "alice", "color").unwrap(),
            Some("red".to_string())
        );

        assert!(delete_agent_data(&conn, "alice", "color").unwrap());
        assert!(get_agent_data(&conn, "alice", "color").unwrap().is_none());
    }

    #[test]
    fn test_list_agent_data() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        set_agent_data(&conn, "alice", "b_key", "val_b").unwrap();
        set_agent_data(&conn, "alice", "a_key", "val_a").unwrap();
        let data = list_agent_data(&conn, "alice").unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].0, "a_key");
        assert_eq!(data[1].0, "b_key");
    }

    #[test]
    fn test_agent_data_cascade_on_delete() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        set_agent_data(&conn, "alice", "k", "v").unwrap();
        delete_agent(&conn, "alice").unwrap();
        let data = list_agent_data(&conn, "alice").unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_save_and_load_agent_messages() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "hello"}),
            serde_json::json!({"role": "assistant", "content": "hi"}),
        ];
        save_agent_messages(&conn, "alice", &msgs).unwrap();
        let loaded: Vec<serde_json::Value> = load_agent_messages(&conn, "alice").unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0]["content"], "hello");
        assert_eq!(loaded[1]["content"], "hi");
    }

    #[test]
    fn test_save_agent_messages_replaces() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        let msgs1 = vec![serde_json::json!({"role": "user", "content": "first"})];
        save_agent_messages(&conn, "alice", &msgs1).unwrap();
        let msgs2 = vec![serde_json::json!({"role": "user", "content": "second"})];
        save_agent_messages(&conn, "alice", &msgs2).unwrap();
        let loaded: Vec<serde_json::Value> = load_agent_messages(&conn, "alice").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0]["content"], "second");
    }

    #[test]
    fn test_append_agent_message() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        append_agent_message(&conn, "alice", &serde_json::json!({"role": "user"})).unwrap();
        append_agent_message(&conn, "alice", &serde_json::json!({"role": "assistant"})).unwrap();
        assert_eq!(agent_message_count(&conn, "alice").unwrap(), 2);
    }

    #[test]
    fn test_clear_agent_messages() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        append_agent_message(&conn, "alice", &serde_json::json!({"role": "user"})).unwrap();
        clear_agent_messages(&conn, "alice").unwrap();
        assert_eq!(agent_message_count(&conn, "alice").unwrap(), 0);
    }

    #[test]
    fn test_claim_and_release_agent() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();

        assert!(claim_agent(&conn, "alice", "session-1").unwrap());
        let agent = get_agent(&conn, "alice").unwrap().unwrap();
        assert_eq!(agent.session_id, Some("session-1".to_string()));

        assert!(!claim_agent(&conn, "alice", "session-2").unwrap());

        release_agent(&conn, "alice", "session-1").unwrap();
        let agent = get_agent(&conn, "alice").unwrap().unwrap();
        assert!(agent.session_id.is_none());

        assert!(claim_agent(&conn, "alice", "session-2").unwrap());
    }

    #[test]
    fn test_release_all_agents() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        create_agent(&conn, "bob", "").unwrap();
        claim_agent(&conn, "alice", "s1").unwrap();
        claim_agent(&conn, "bob", "s1").unwrap();
        release_all_agents(&conn, "s1").unwrap();
        let alice = get_agent(&conn, "alice").unwrap().unwrap();
        let bob = get_agent(&conn, "bob").unwrap().unwrap();
        assert!(alice.session_id.is_none());
        assert!(bob.session_id.is_none());
    }

    #[test]
    fn test_send_notification() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        create_agent(&conn, "bob", "").unwrap();
        let id = send_notification(&conn, "alice", "bob", "hello bob").unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_send_notification_unknown_recipient_gives_a_clear_error() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        let err = send_notification(&conn, "alice", "does-not-exist", "hi").unwrap_err();
        let message = err.to_string();
        // Must name the actual problem, not leak the raw SQL constraint
        // error ("FOREIGN KEY constraint failed") that INSERT would
        // otherwise fail with.
        assert!(message.contains("does-not-exist"));
        assert!(message.contains("does not exist"));
        assert!(!message.to_lowercase().contains("constraint"));
    }

    #[test]
    fn test_send_notification_unknown_sender_gives_a_clear_error() {
        let conn = test_db();
        create_agent(&conn, "bob", "").unwrap();
        let err = send_notification(&conn, "ghost", "bob", "hi").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("ghost"));
        assert!(!message.to_lowercase().contains("constraint"));
    }

    #[test]
    fn test_send_notification_unknown_recipient_does_not_insert_a_row() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        assert!(send_notification(&conn, "alice", "does-not-exist", "hi").is_err());
        claim_agent(&conn, "alice", "s1").unwrap();
        let notifs = poll_notifications_for_session(&conn, "s1").unwrap();
        assert!(notifs.is_empty());
    }

    #[test]
    fn test_poll_notifications_for_session() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        create_agent(&conn, "bob", "").unwrap();
        claim_agent(&conn, "bob", "s1").unwrap();
        send_notification(&conn, "alice", "bob", "msg1").unwrap();
        send_notification(&conn, "alice", "bob", "msg2").unwrap();

        let notifs = poll_notifications_for_session(&conn, "s1").unwrap();
        assert_eq!(notifs.len(), 2);
        assert_eq!(notifs[0].message, "msg1");
        assert_eq!(notifs[1].message, "msg2");

        let notifs2 = poll_notifications_for_session(&conn, "s1").unwrap();
        assert!(notifs2.is_empty());
    }

    #[test]
    fn test_create_oneshot_task() {
        let conn = test_db();
        let run_at = chrono::Utc::now().to_rfc3339();
        let id = create_oneshot_task(&conn, "task1", "desc", &run_at, "echo hi", None).unwrap();
        let task = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(task.name, "task1");
        assert_eq!(task.task_type, "oneshot");
        assert_eq!(task.status, TaskStatus::SCHEDULED);
    }

    #[test]
    fn test_create_cron_task() {
        let conn = test_db();
        let id = create_cron_task(
            &conn,
            "cron1",
            "every sec",
            "* * * * * * *",
            "echo hi",
            None,
            None,
        )
        .unwrap();
        let task = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(task.name, "cron1");
        assert_eq!(task.task_type, "cron");
    }

    #[test]
    fn test_create_cron_task_invalid_expression() {
        let conn = test_db();
        let result = create_cron_task(&conn, "bad", "", "not a cron", "", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_task() {
        let conn = test_db();
        let run_at = chrono::Utc::now().to_rfc3339();
        let id = create_oneshot_task(&conn, "t", "", &run_at, "", None).unwrap();
        assert!(delete_task(&conn, id).unwrap());
        assert!(get_task(&conn, id).unwrap().is_none());
    }

    fn ok(result: &str) -> TaskOutcome {
        TaskOutcome {
            succeeded: true,
            exit_code: Some(0),
            result: result.to_string(),
        }
    }

    /// A session that's alive: it holds an agent and has just heartbeated.
    fn live_session(conn: &Connection, session: &str) {
        let agent = format!("agent-of-{}", session);
        create_agent(conn, &agent, "").unwrap();
        assert!(claim_agent(conn, &agent, session).unwrap());
    }

    fn due_oneshot(conn: &Connection) -> i64 {
        let past = (chrono::Utc::now() - chrono::Duration::seconds(60)).to_rfc3339();
        create_oneshot_task(conn, "t", "", &past, "", None).unwrap()
    }

    fn make_due(conn: &Connection, id: i64) {
        let past = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        conn.execute(
            "UPDATE scheduled_tasks SET next_run_at = ?1 WHERE id = ?2",
            params![past, id],
        )
        .unwrap();
    }

    #[test]
    fn test_new_task_is_scheduled() {
        let conn = test_db();
        let id = due_oneshot(&conn);
        let task = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::SCHEDULED);
        assert_eq!(task.last_outcome, None);
    }

    #[test]
    fn test_oneshot_lifecycle_and_exclusive_claim() {
        let conn = test_db();
        live_session(&conn, "s1");
        live_session(&conn, "s2");
        let id = due_oneshot(&conn);

        assert!(claim_task(&conn, id, "s1").unwrap());
        assert!(
            !claim_task(&conn, id, "s2").unwrap(),
            "s1 is still running it"
        );
        assert!(get_pending_tasks(&conn).unwrap().is_empty());
        let task = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::RUNNING);
        assert_eq!(task.claimed_by.as_deref(), Some("s1"));
        assert!(task.started_at.is_some());

        assert!(finish_task(&conn, id, "s1", &ok("hello")).unwrap());
        let task = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::DONE);
        assert_eq!(task.run_count, 1);
        assert_eq!(task.claimed_by, None);
        assert_eq!(task.last_outcome.as_deref(), Some("succeeded"));
        assert_eq!(task.last_exit_code, Some(0));
        assert_eq!(task.last_result.as_deref(), Some("hello"));
        assert!(task.last_run_at.is_some());
        assert!(
            !claim_task(&conn, id, "s2").unwrap(),
            "done tasks don't run again"
        );
    }

    #[test]
    fn test_cron_task_is_rescheduled_until_max_runs() {
        let conn = test_db();
        live_session(&conn, "s");
        let id = create_cron_task(&conn, "c", "", "0 0 * * * * *", "", None, Some(2)).unwrap();
        make_due(&conn, id);

        assert!(claim_task(&conn, id, "s").unwrap());
        finish_task(&conn, id, "s", &ok("")).unwrap();
        let task = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::SCHEDULED);
        assert!(task.next_run_at.unwrap() > chrono::Utc::now().to_rfc3339());
        assert!(!claim_task(&conn, id, "s").unwrap(), "not due again yet");

        make_due(&conn, id);
        assert!(claim_task(&conn, id, "s").unwrap());
        finish_task(&conn, id, "s", &ok("")).unwrap();
        let task = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::DONE);
        assert_eq!(task.run_count, 2);
    }

    #[test]
    fn test_failed_run_keeps_cron_task_scheduled() {
        let conn = test_db();
        live_session(&conn, "s");
        let id = create_cron_task(&conn, "c", "", "0 0 * * * * *", "", None, None).unwrap();
        make_due(&conn, id);
        claim_task(&conn, id, "s").unwrap();
        let failed = TaskOutcome {
            succeeded: false,
            exit_code: Some(2),
            result: "boom".to_string(),
        };
        finish_task(&conn, id, "s", &failed).unwrap();
        let task = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::SCHEDULED);
        assert_eq!(task.last_outcome.as_deref(), Some("failed"));
        assert_eq!(task.last_exit_code, Some(2));
    }

    #[test]
    fn test_abandoned_claim_can_be_taken_over() {
        let conn = test_db();
        live_session(&conn, "alive");
        let id = due_oneshot(&conn);
        // Claimed by a session that never heartbeated (e.g. it crashed).
        assert!(claim_task(&conn, id, "gone").unwrap());
        assert_eq!(get_pending_tasks(&conn).unwrap().len(), 1);
        assert!(claim_task(&conn, id, "alive").unwrap());
        // The old session's late finish is ignored.
        assert!(!finish_task(&conn, id, "gone", &ok("late")).unwrap());
        assert!(finish_task(&conn, id, "alive", &ok("")).unwrap());
        assert_eq!(get_task(&conn, id).unwrap().unwrap().run_count, 1);
    }

    #[test]
    fn test_set_task_enabled() {
        let conn = test_db();
        live_session(&conn, "s");
        let id = due_oneshot(&conn);
        assert!(set_task_enabled(&conn, id, false).unwrap());
        assert_eq!(
            get_task(&conn, id).unwrap().unwrap().status,
            TaskStatus::DISABLED
        );
        assert!(!claim_task(&conn, id, "s").unwrap());
        assert!(
            !set_task_enabled(&conn, id, false).unwrap(),
            "already disabled"
        );
        assert!(set_task_enabled(&conn, id, true).unwrap());
        assert_eq!(
            get_task(&conn, id).unwrap().unwrap().status,
            TaskStatus::SCHEDULED
        );

        // Disabled while running: the run finishes, the task stays disabled.
        assert!(claim_task(&conn, id, "s").unwrap());
        assert!(set_task_enabled(&conn, id, false).unwrap());
        finish_task(&conn, id, "s", &ok("")).unwrap();
        let task = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::DISABLED);
        assert_eq!(task.run_count, 1);
    }

    #[test]
    fn test_done_task_cannot_be_re_enabled() {
        let conn = test_db();
        live_session(&conn, "s");
        let id = due_oneshot(&conn);
        claim_task(&conn, id, "s").unwrap();
        finish_task(&conn, id, "s", &ok("")).unwrap();
        assert!(!set_task_enabled(&conn, id, true).unwrap());
        assert_eq!(
            get_task(&conn, id).unwrap().unwrap().status,
            TaskStatus::DONE
        );
    }

    #[test]
    fn test_migration_from_enabled_flag() {
        let conn = Connection::open_in_memory().unwrap();
        // The table as it was before tasks had a status.
        conn.execute_batch(
            "CREATE TABLE scheduled_tasks (
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
                command TEXT NOT NULL DEFAULT '',
                max_runs INTEGER DEFAULT NULL,
                run_count INTEGER NOT NULL DEFAULT 0
             );
             CREATE INDEX idx_tasks_next_run ON scheduled_tasks(next_run_at) WHERE enabled = 1;
             INSERT INTO scheduled_tasks (name, task_type, enabled, run_count)
                 VALUES ('active', 'cron', 1, 3);
             INSERT INTO scheduled_tasks (name, task_type, enabled, run_count)
                 VALUES ('ran once', 'oneshot', 0, 1);
             INSERT INTO scheduled_tasks (name, task_type, enabled, run_count, max_runs)
                 VALUES ('used up', 'cron', 0, 5, 5);
             INSERT INTO scheduled_tasks (name, task_type, enabled, run_count)
                 VALUES ('switched off', 'cron', 0, 1);",
        )
        .unwrap();

        initialize_db(&conn).unwrap();
        // Running it again on the migrated database is a no-op.
        initialize_db(&conn).unwrap();

        let statuses: Vec<(String, String)> = list_tasks(&conn, None)
            .unwrap()
            .into_iter()
            .map(|t| (t.name, t.status))
            .collect();
        assert_eq!(
            statuses,
            [
                ("active", "scheduled"),
                ("ran once", "done"),
                ("used up", "done"),
                ("switched off", "disabled"),
            ]
            .map(|(n, s)| (n.to_string(), s.to_string()))
        );
        assert!(!table_has_column(&conn, "scheduled_tasks", "enabled").unwrap());
    }

    #[test]
    fn test_delete_agent_cascades_tasks() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        create_cron_task(&conn, "t", "", "* * * * * * *", "", Some("alice"), None).unwrap();
        delete_agent(&conn, "alice").unwrap();
        let tasks = list_tasks(&conn, Some("alice")).unwrap();
        assert!(tasks.is_empty());
    }

    #[test]
    fn test_get_agent_config() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        set_agent_data(&conn, "alice", "config:model", "gpt-4").unwrap();
        set_agent_data(&conn, "alice", "config:system_prompt", "be nice").unwrap();
        let config = get_agent_config(&conn, "alice").unwrap();
        assert_eq!(config.model, Some("gpt-4".to_string()));
        assert_eq!(config.system_prompt, Some("be nice".to_string()));
        assert!(config.endpoint.is_none());
    }

    #[test]
    fn test_ensure_default_agent() {
        let conn = test_db();
        ensure_default_agent(&conn).unwrap();
        ensure_default_agent(&conn).unwrap();
        let agent = get_agent(&conn, "default").unwrap().unwrap();
        assert_eq!(agent.name, "default");
    }

    #[test]
    fn test_gc_agents_removes_dormant() {
        let conn = test_db();
        create_agent(&conn, "dormant", "").unwrap();
        let removed = gc_agents(&conn).unwrap();
        assert_eq!(removed, vec!["dormant"]);
        assert!(get_agent(&conn, "dormant").unwrap().is_none());
    }

    #[test]
    fn test_gc_agents_preserves_default() {
        let conn = test_db();
        ensure_default_agent(&conn).unwrap();
        let removed = gc_agents(&conn).unwrap();
        assert!(removed.is_empty());
        assert!(get_agent(&conn, "default").unwrap().is_some());
    }

    #[test]
    fn test_list_tasks_filter_by_agent() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        create_agent(&conn, "bob", "").unwrap();
        create_cron_task(&conn, "t1", "", "* * * * * * *", "", Some("alice"), None).unwrap();
        create_cron_task(&conn, "t2", "", "* * * * * * *", "", Some("bob"), None).unwrap();
        let alice_tasks = list_tasks(&conn, Some("alice")).unwrap();
        assert_eq!(alice_tasks.len(), 1);
        assert_eq!(alice_tasks[0].name, "t1");
        let all_tasks = list_tasks(&conn, None).unwrap();
        assert_eq!(all_tasks.len(), 2);
    }

    #[test]
    fn test_get_pending_tasks() {
        let conn = test_db();
        let past = (chrono::Utc::now() - chrono::Duration::seconds(60)).to_rfc3339();
        create_oneshot_task(&conn, "due", "", &past, "echo", None).unwrap();
        let future = (chrono::Utc::now() + chrono::Duration::seconds(3600)).to_rfc3339();
        create_oneshot_task(&conn, "later", "", &future, "echo", None).unwrap();
        let pending = get_pending_tasks(&conn).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].name, "due");
    }

    #[test]
    fn test_readline_history_table_created() {
        let conn = test_db();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM readline_history", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }
}
