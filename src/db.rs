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
    /// The agent that spawned this one, for a sub-agent.
    #[serde(default)]
    pub parent: Option<String>,
    /// What it's doing, or last did: "thinking", "running read_file",
    /// "idle", "finished: ...", ...
    #[serde(default)]
    pub activity: Option<String>,
    /// When `activity` last changed.
    #[serde(default)]
    pub activity_at: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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
    /// `tool` or `prompt` - see `TaskKind`.
    pub kind: String,
    /// Tasks that must be done, successfully, before this one is due.
    #[serde(default)]
    pub depends_on: Vec<i64>,
    /// For a prompt task: run it on a new agent made from this profile (see
    /// `config.json`'s "profiles"), rather than on `agent_name` or on any
    /// agent. Never set together with `agent_name`.
    #[serde(default)]
    pub profile: Option<String>,
    /// Made by an agent without unsafe tools: it runs on one without them
    /// too, whoever picks it up - so a safe agent can't get work done by
    /// an unsafe one through a task.
    #[serde(default)]
    pub run_safe: bool,
    /// The directory it runs in, over its agent's own (absolute).
    #[serde(default)]
    pub cwd: Option<String>,
    /// Asked to stop (`request_task_stop`) while running: the session
    /// running it interrupts it.
    #[serde(default)]
    pub stop_requested: bool,
}

/// What a task's `command` is:
///
/// - `tool`: a `{"tool": ..., "arguments": ...}` call, run directly by the
///   scheduler of any chat session, no LLM involved.
/// - `prompt`: an instruction for an agent. It's picked up by a chat
///   session waiting at its prompt - only one whose agent is `agent_name`,
///   if set - and run as a turn of that agent's conversation.
pub struct TaskKind;

impl TaskKind {
    pub const TOOL: &'static str = "tool";
    pub const PROMPT: &'static str = "prompt";
}

/// When a new task runs.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TaskSchedule {
    /// Once, at this RFC 3339 time.
    Once { at: String },
    /// On a 7-field cron schedule, optionally at most `max_runs` times.
    Cron {
        expression: String,
        max_runs: Option<i64>,
    },
}

/// A task to create with `create_task`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NewTask {
    pub name: String,
    pub description: String,
    /// `TaskKind::TOOL` or `TaskKind::PROMPT`.
    pub kind: String,
    pub command: String,
    pub agent_name: Option<String>,
    pub schedule: TaskSchedule,
    /// Create it `held` rather than `scheduled`.
    #[serde(default)]
    pub held: bool,
    /// Tasks that must be done, successfully, first (see `DEPENDENCIES_MET`).
    #[serde(default)]
    pub depends_on: Vec<i64>,
    /// Run it on a new agent made from this profile (prompt tasks only;
    /// not together with `agent_name`).
    #[serde(default)]
    pub profile: Option<String>,
    /// See `TaskRow::run_safe`.
    #[serde(default)]
    pub run_safe: bool,
    /// See `TaskRow::cwd`.
    #[serde(default)]
    pub cwd: Option<String>,
}

/// A task's lifecycle:
///
/// - `scheduled`: waiting for `next_run_at`, then picked up by whichever
///   session's scheduler claims it first (`claim_task`).
/// - `running`: claimed by `claimed_by`. If that session stops
///   heartbeating, the claim is abandoned and the task can be claimed again.
/// - back to `scheduled` after a cron run (`finish_task`), or `done` after
///   a one-shot run or a cron task's last allowed run (`max_runs`).
/// - `held`: scheduled, but not to be picked up by anyone until released
///   (`set_task_held`) - e.g. created ahead of time, to start on a go
///   signal. Released, it's `scheduled` again, and runs right away if it
///   came due meanwhile.
/// - `disabled`: switched off by the user; switching it back on makes it
///   `scheduled` again.
///
/// Whether a run succeeded is separate (`last_outcome`): a cron task whose
/// last run failed is still `scheduled` for its next one.
pub struct TaskStatus;

impl TaskStatus {
    pub const SCHEDULED: &'static str = "scheduled";
    pub const HELD: &'static str = "held";
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

/// SQL condition: every task in `scheduled_tasks.depends_on` is done, and
/// its last run succeeded. Dependencies can only be on tasks that existed
/// when the task was created (`create_task` checks), so they never loop.
const DEPENDENCIES_MET: &str = "NOT EXISTS (
    SELECT 1 FROM json_each(COALESCE(scheduled_tasks.depends_on, '[]')) AS dependency
    LEFT JOIN scheduled_tasks AS required ON required.id = dependency.value
    WHERE required.id IS NULL
       OR required.status != 'done'
       OR COALESCE(required.last_outcome, '') != 'succeeded')";

/// SQL condition: the task's claim belongs to a session that's no longer
/// heartbeating (or no longer exists) - its run was abandoned. Sessions
/// heartbeat every 5 seconds.
const CLAIM_ABANDONED: &str = "claimed_by IS NULL OR claimed_by NOT IN (
    SELECT session_id FROM agents
    WHERE session_id IS NOT NULL AND heartbeat_at >= datetime('now', '-30 seconds'))";

/// The columns and constraints of `scheduled_tasks`, as created fresh and
/// as rebuilt by `rebuild_task_table_if_needed`.
const TASK_TABLE_DEFINITION: &str = "
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
        CHECK(status IN ('scheduled', 'held', 'running', 'done', 'disabled')),
    kind TEXT NOT NULL DEFAULT 'tool' CHECK(kind IN ('tool', 'prompt')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    command TEXT NOT NULL DEFAULT '',
    max_runs INTEGER DEFAULT NULL,
    run_count INTEGER NOT NULL DEFAULT 0,
    claimed_by TEXT DEFAULT NULL,
    started_at TEXT DEFAULT NULL,
    last_outcome TEXT DEFAULT NULL,
    last_exit_code INTEGER DEFAULT NULL,
    last_result TEXT DEFAULT NULL,
    depends_on TEXT DEFAULT NULL,
    profile TEXT DEFAULT NULL,
    run_safe INTEGER NOT NULL DEFAULT 0,
    cwd TEXT DEFAULT NULL,
    stop_requested INTEGER NOT NULL DEFAULT 0,
    CHECK(agent_name IS NULL OR profile IS NULL),
    FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE SET NULL";

/// Every column of `TASK_TABLE_DEFINITION`, for copying rows across.
const TASK_TABLE_COLUMNS: &str = "id, agent_name, name, description, task_type, cron_expression, run_at, next_run_at, last_run_at, status, kind, created_at, command, max_runs, run_count, claimed_by, started_at, last_outcome, last_exit_code, last_result, depends_on, profile, run_safe, cwd, stop_requested";

/// Brings an existing `scheduled_tasks` table to `TASK_TABLE_DEFINITION`
/// when its constraints are older: its `status` one predates the `held`
/// state, or it lacks the one keeping `agent_name` and `profile` apart. SQLite can't
/// change a CHECK constraint in place, so the table is rebuilt: a new one
/// is created, the rows copied over, and it takes the old one's place, all
/// in one transaction. The id counter is carried over too, so ids of
/// deleted tasks are never handed out again.
fn rebuild_task_table_if_needed(conn: &Connection) -> Result<(), rusqlite::Error> {
    let sql: String = conn.query_row(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'scheduled_tasks'",
        [],
        |row| row.get(0),
    )?;
    if sql.contains("'held'") && sql.contains("profile IS NULL") {
        return Ok(());
    }
    let next_id: i64 = conn
        .query_row(
            "SELECT seq FROM sqlite_sequence WHERE name = 'scheduled_tasks'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    // Foreign key enforcement can't change inside a transaction, and has
    // to be off while the table is swapped.
    conn.execute_batch("PRAGMA foreign_keys = OFF;")?;
    let result = conn.execute_batch(&format!(
        "BEGIN;
         CREATE TABLE scheduled_tasks_new ({definition});
         INSERT INTO scheduled_tasks_new ({columns}) SELECT {columns} FROM scheduled_tasks;
         DROP TABLE scheduled_tasks;
         ALTER TABLE scheduled_tasks_new RENAME TO scheduled_tasks;
         UPDATE sqlite_sequence SET seq = MAX(seq, {next_id}) WHERE name = 'scheduled_tasks';
         COMMIT;",
        definition = TASK_TABLE_DEFINITION,
        columns = TASK_TABLE_COLUMNS,
        next_id = next_id,
    ));
    if result.is_err() {
        let _ = conn.execute_batch("ROLLBACK;");
    }
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;
    result
}

pub fn initialize_db(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;
    conn.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS scheduled_tasks ({});",
        TASK_TABLE_DEFINITION
    ))?;
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
        "depends_on TEXT DEFAULT NULL",
        "profile TEXT DEFAULT NULL",
        "run_safe INTEGER NOT NULL DEFAULT 0",
        "cwd TEXT DEFAULT NULL",
        "stop_requested INTEGER NOT NULL DEFAULT 0",
        "kind TEXT NOT NULL DEFAULT 'tool' CHECK(kind IN ('tool', 'prompt'))",
    ] {
        let _ = conn.execute_batch(&format!(
            "ALTER TABLE scheduled_tasks ADD COLUMN {};",
            column
        ));
    }
    migrate_task_enabled_to_status(conn)?;
    rebuild_task_table_if_needed(conn)?;
    // Tasks made before prompt tasks existed whose command is plain text
    // could only fail; they're instructions for an agent. Ones that already
    // ran are left as they were.
    conn.execute_batch(
        "UPDATE scheduled_tasks
         SET kind = 'prompt',
             command = CASE WHEN trim(command) = '' THEN description ELSE command END
         WHERE kind = 'tool' AND status IN ('scheduled', 'disabled')
           AND trim(CASE WHEN trim(command) = '' THEN description ELSE command END) != ''
           AND COALESCE(
                 CASE WHEN json_valid(CASE WHEN trim(command) = '' THEN description ELSE command END)
                      THEN json_type(CASE WHEN trim(command) = '' THEN description ELSE command END, '$.tool')
                 END, '') != 'text';",
    )?;
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_tasks_due ON scheduled_tasks(status, next_run_at);
         CREATE INDEX IF NOT EXISTS idx_tasks_agent ON scheduled_tasks(agent_name);",
    )?;

    let _ = conn.execute_batch("ALTER TABLE agents ADD COLUMN session_id TEXT DEFAULT NULL;");
    let _ = conn.execute_batch("ALTER TABLE agents ADD COLUMN heartbeat_at TEXT DEFAULT NULL;");
    // A sub-agent whose parent is deleted (sub-agents are, once done)
    // becomes a top-level agent rather than going too: its own sub-agents
    // may still be running.
    let _ = conn.execute_batch(
        "ALTER TABLE agents ADD COLUMN parent TEXT DEFAULT NULL
             REFERENCES agents(name) ON DELETE SET NULL;",
    );
    let _ = conn.execute_batch("ALTER TABLE agents ADD COLUMN activity TEXT DEFAULT NULL;");
    let _ = conn.execute_batch("ALTER TABLE agents ADD COLUMN activity_at TEXT DEFAULT NULL;");

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS agent_events (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             agent TEXT NOT NULL,
             task_id INTEGER,
             at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
             event TEXT NOT NULL,
             FOREIGN KEY (agent) REFERENCES agents(name) ON DELETE CASCADE
         );
         CREATE INDEX IF NOT EXISTS idx_agent_events_agent ON agent_events(agent, id);
         CREATE INDEX IF NOT EXISTS idx_agent_events_task ON agent_events(task_id, id);",
    )?;

    initialize_kb(conn)?;
    Ok(())
}

/// The knowledge base: `kb_notes`, plus `kb_fts`, an FTS5 full-text index
/// over their title, body and tags that triggers keep in step with it.
/// Titles are unique per scope (shared, or one agent's private notes),
/// ignoring case, so writing a note under an existing title updates it.
fn initialize_kb(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS kb_notes (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             title TEXT NOT NULL,
             body TEXT NOT NULL,
             tags TEXT NOT NULL DEFAULT '',
             agent_name TEXT,
             created_by TEXT,
             created_at TEXT NOT NULL DEFAULT (datetime('now')),
             updated_at TEXT NOT NULL DEFAULT (datetime('now')),
             FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE CASCADE
         );
         CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_title
             ON kb_notes(lower(title), COALESCE(agent_name, ''));
         CREATE INDEX IF NOT EXISTS idx_kb_updated ON kb_notes(updated_at);
         CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
             title, body, tags,
             content = 'kb_notes', content_rowid = 'id',
             tokenize = 'porter unicode61'
         );
         CREATE TRIGGER IF NOT EXISTS kb_notes_ai AFTER INSERT ON kb_notes BEGIN
             INSERT INTO kb_fts(rowid, title, body, tags)
                 VALUES (new.id, new.title, new.body, new.tags);
         END;
         CREATE TRIGGER IF NOT EXISTS kb_notes_ad AFTER DELETE ON kb_notes BEGIN
             INSERT INTO kb_fts(kb_fts, rowid, title, body, tags)
                 VALUES ('delete', old.id, old.title, old.body, old.tags);
         END;
         CREATE TRIGGER IF NOT EXISTS kb_notes_au AFTER UPDATE ON kb_notes BEGIN
             INSERT INTO kb_fts(kb_fts, rowid, title, body, tags)
                 VALUES ('delete', old.id, old.title, old.body, old.tags);
             INSERT INTO kb_fts(rowid, title, body, tags)
                 VALUES (new.id, new.title, new.body, new.tags);
         END;",
    )
}

/// Opens an existing database without changing it in any way: no file is
/// created if it's missing, no schema is created or migrated, and nothing
/// can be written. For commands that only look (`faber tasks`), so that
/// pointing one at the wrong path is an error instead of quietly creating
/// an empty database that then looks like there's nothing in it.
pub fn open_read_only(path: &str) -> Result<Connection, Box<dyn Error>> {
    let shown = std::path::absolute(path)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.to_string());
    if !std::path::Path::new(path).is_file() {
        return Err(format!("no database at {}", shown).into());
    }
    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let not_faber = || format!("{} isn't a faber database", shown);
    // A file that isn't SQLite at all only fails once it's read.
    if !table_has_column(&conn, "scheduled_tasks", "id").map_err(|_| not_faber())? {
        return Err(not_faber().into());
    }
    if !table_has_column(&conn, "scheduled_tasks", "status")?
        || !table_has_column(&conn, "scheduled_tasks", "kind")?
        || !table_has_column(&conn, "agents", "activity")?
        || !table_has_column(&conn, "scheduled_tasks", "profile")?
    {
        return Err(format!(
            "{} is from an older version of faber: open it once with `faber chat` to upgrade it",
            shown
        )
        .into());
    }
    Ok(conn)
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
        parent: row.get(5)?,
        activity: row.get(6)?,
        activity_at: row.get(7)?,
    })
}

const AGENT_COLUMNS: &str =
    "name, description, created_at, session_id, heartbeat_at, parent, activity, activity_at";

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
        kind: row.get(19)?,
        depends_on: row
            .get::<_, Option<String>>(20)?
            .and_then(|deps| serde_json::from_str(&deps).ok())
            .unwrap_or_default(),
        profile: row.get(21)?,
        run_safe: row.get(22)?,
        cwd: row.get(23)?,
        stop_requested: row.get(24)?,
    })
}

const TASK_COLUMNS: &str = "id, agent_name, name, description, task_type, cron_expression, run_at, next_run_at, last_run_at, status, created_at, command, max_runs, run_count, claimed_by, started_at, last_outcome, last_exit_code, last_result, kind, depends_on, profile, run_safe, cwd, stop_requested";

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
        "INSERT INTO scheduled_tasks (name, description, task_type, cron_expression, next_run_at, agent_name, command, max_runs, kind)
         VALUES (?1, ?2, 'cron', ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            name,
            description,
            cron_expr,
            next_run,
            agent_name,
            effective_command(command, description),
            max_runs,
            kind_for_command(effective_command(command, description))
        ],
    )?;

    Ok(conn.last_insert_rowid())
}

/// Whether `command` is a `{"tool": "<name>", ...}` call.
pub fn is_tool_call(command: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(command)
        .ok()
        .and_then(|v| v.get("tool")?.as_str().map(|t| !t.is_empty()))
        .unwrap_or(false)
}

/// The kind of task `command` makes: a tool call runs as one, anything else
/// is an instruction for an agent. An empty command stays a `tool` task,
/// which fails saying so when it runs.
fn kind_for_command(command: &str) -> &'static str {
    if command.trim().is_empty() || is_tool_call(command) {
        TaskKind::TOOL
    } else {
        TaskKind::PROMPT
    }
}

/// What a task runs: its command, or its description if the command is
/// empty (as the scheduler has always done).
fn effective_command<'a>(command: &'a str, description: &'a str) -> &'a str {
    if command.trim().is_empty() {
        description
    } else {
        command
    }
}

/// Creates a task of either kind (see `TaskKind`), returning its id.
pub fn create_task(conn: &Connection, task: &NewTask) -> Result<i64, Box<dyn Error>> {
    if task.kind != TaskKind::TOOL && task.kind != TaskKind::PROMPT {
        return Err(format!("unknown task kind '{}': use tool or prompt", task.kind).into());
    }
    if let Some(agent) = &task.agent_name {
        if get_agent(conn, agent)?.is_none() {
            return Err(format!("no agent named '{}'", agent).into());
        }
    }
    check_task_target(
        &task.kind,
        task.agent_name.as_deref(),
        task.profile.as_deref(),
    )?;
    for dependency in &task.depends_on {
        if get_task(conn, *dependency)?.is_none() {
            return Err(format!("no task #{} to depend on", dependency).into());
        }
    }
    let depends_on = if task.depends_on.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&task.depends_on)?)
    };
    let (task_type, cron_expression, run_at, next_run_at, max_runs) = match &task.schedule {
        TaskSchedule::Once { at } => {
            chrono::DateTime::parse_from_rfc3339(at)
                .map_err(|e| format!("Invalid RFC 3339 datetime '{}': {}", at, e))?;
            ("oneshot", None, Some(at.clone()), Some(at.clone()), None)
        }
        TaskSchedule::Cron {
            expression,
            max_runs,
        } => {
            let next = Schedule::from_str(expression)
                .map_err(|e| format!("Invalid cron expression '{}': {}", expression, e))?
                .upcoming(chrono::Utc)
                .next()
                .map(|dt| dt.to_rfc3339());
            ("cron", Some(expression.clone()), None, next, *max_runs)
        }
    };
    conn.execute(
        "INSERT INTO scheduled_tasks
             (name, description, kind, task_type, cron_expression, run_at, next_run_at,
              agent_name, command, max_runs, status, depends_on, profile, run_safe, cwd)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        params![
            task.name,
            task.description,
            task.kind,
            task_type,
            cron_expression,
            run_at,
            next_run_at,
            task.agent_name,
            task.command,
            max_runs,
            if task.held {
                TaskStatus::HELD
            } else {
                TaskStatus::SCHEDULED
            },
            depends_on,
            task.profile,
            task.run_safe,
            task.cwd
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Checks where a task of `kind` is to run: on `agent`, on a new agent made
/// from `profile`, or (neither) on any agent - not both, and a profile only
/// for a prompt task, as only those run on a model.
fn check_task_target(
    kind: &str,
    agent: Option<&str>,
    profile: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    match profile {
        Some(_) if agent.is_some() => {
            Err("a task runs either on an agent or on a profile, not both".into())
        }
        Some(p) if p.trim().is_empty() => Err("empty profile name".into()),
        Some(_) if kind != TaskKind::PROMPT => {
            Err("only prompt tasks can run on a profile: a tool task needs no model".into())
        }
        _ => Ok(()),
    }
}

/// Changes where a task runs: on `agent`, on a new agent made from
/// `profile`, or (neither) on whichever agent picks it up first. A running
/// or done task is left alone - it's been picked up already - and so is a
/// missing one; returns false for those.
pub fn set_task_target(
    conn: &Connection,
    task_id: i64,
    agent: Option<&str>,
    profile: Option<&str>,
) -> Result<bool, Box<dyn Error>> {
    let Some(task) = get_task(conn, task_id)? else {
        return Ok(false);
    };
    if let Some(agent) = agent {
        if get_agent(conn, agent)?.is_none() {
            return Err(format!("no agent named '{}'", agent).into());
        }
    }
    check_task_target(&task.kind, agent, profile)
        .map_err(|e| format!("task #{}: {}", task_id, e))?;
    let rows = conn.execute(
        "UPDATE scheduled_tasks SET agent_name = ?2, profile = ?3
         WHERE id = ?1 AND status NOT IN ('running', 'done')",
        params![task_id, agent, profile],
    )?;
    Ok(rows > 0)
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
        "INSERT INTO scheduled_tasks (name, description, task_type, run_at, next_run_at, agent_name, command, kind)
         VALUES (?1, ?2, 'oneshot', ?3, ?3, ?4, ?5, ?6)",
        params![
            name,
            description,
            run_at,
            agent_name,
            effective_command(command, description),
            kind_for_command(effective_command(command, description))
        ],
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
             WHERE id = ?1 AND status IN ('scheduled', 'held', 'running')",
            params![task_id],
        )?
    };
    Ok(rows > 0)
}

/// Asks the session running a task to stop it. Returns false if it isn't
/// running. If nobody is running it any more - its session died - it's
/// ended right here instead, as failed.
pub fn request_task_stop(conn: &Connection, task_id: i64) -> Result<bool, Box<dyn Error>> {
    let abandoned = format!(
        "UPDATE scheduled_tasks SET
             status = CASE WHEN task_type = 'cron' THEN 'scheduled' ELSE 'done' END,
             claimed_by = NULL, stop_requested = 0,
             last_outcome = 'failed', last_result = 'stopped (nobody was running it any more)',
             last_run_at = ?2
         WHERE id = ?1 AND status = 'running' AND ({})",
        CLAIM_ABANDONED
    );
    if conn.execute(
        &abandoned,
        params![task_id, chrono::Utc::now().to_rfc3339()],
    )? > 0
    {
        return Ok(true);
    }
    let rows = conn.execute(
        "UPDATE scheduled_tasks SET stop_requested = 1 WHERE id = ?1 AND status = 'running'",
        params![task_id],
    )?;
    Ok(rows > 0)
}

/// Makes a task due right away, whatever it was waiting for - its time, a
/// release (`held`), being enabled (`disabled`) - or, `done`, runs it
/// again. Dependencies still have to be met. A running task is left
/// alone; returns false for it, or a missing one.
pub fn run_task_now(conn: &Connection, task_id: i64) -> Result<bool, Box<dyn Error>> {
    let rows = conn.execute(
        "UPDATE scheduled_tasks SET status = 'scheduled', next_run_at = ?2
         WHERE id = ?1 AND status IN ('scheduled', 'held', 'disabled', 'done')",
        params![task_id, chrono::Utc::now().to_rfc3339()],
    )?;
    Ok(rows > 0)
}

/// Puts a `scheduled` task on hold (`held`), or releases a held one back
/// to `scheduled`. Returns false if the task wasn't in the state to start
/// from - a running task is already assigned and can't be held.
pub fn set_task_held(conn: &Connection, task_id: i64, held: bool) -> Result<bool, Box<dyn Error>> {
    let (from, to) = if held {
        (TaskStatus::SCHEDULED, TaskStatus::HELD)
    } else {
        (TaskStatus::HELD, TaskStatus::SCHEDULED)
    };
    let rows = conn.execute(
        "UPDATE scheduled_tasks SET status = ?3 WHERE id = ?1 AND status = ?2",
        params![task_id, from, to],
    )?;
    Ok(rows > 0)
}

pub fn get_pending_tasks(conn: &Connection) -> Result<Vec<TaskRow>, Box<dyn Error>> {
    let now = chrono::Utc::now().to_rfc3339();
    let sql = format!(
        "SELECT {} FROM scheduled_tasks
         WHERE next_run_at <= ?1
           AND (status = 'scheduled' OR (status = 'running' AND ({})))
           AND {}
         ORDER BY next_run_at",
        TASK_COLUMNS, CLAIM_ABANDONED, DEPENDENCIES_MET
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params![now], row_to_task)?;
    let mut tasks = Vec::new();
    for row in rows {
        tasks.push(row?);
    }
    Ok(tasks)
}

/// Reads a timestamp as stored in the database: RFC 3339 (`next_run_at`,
/// `last_run_at`, ...) or SQLite's `datetime('now')` UTC format
/// (`created_at`).
pub fn parse_db_time(text: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if let Ok(t) = chrono::DateTime::parse_from_rfc3339(text) {
        return Some(t.with_timezone(&chrono::Utc));
    }
    chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S")
        .ok()
        .map(|t| t.and_utc())
}

/// Deletes `done` tasks that last ran (or, if they never did, were
/// created) before `older_than`, returning them. With `dry_run`, only
/// returns what would be deleted. Scheduled, running and disabled tasks
/// are never touched.
pub fn prune_tasks(
    conn: &Connection,
    older_than: chrono::DateTime<chrono::Utc>,
    dry_run: bool,
) -> Result<Vec<TaskRow>, Box<dyn Error>> {
    let old: Vec<TaskRow> = list_tasks(conn, None)?
        .into_iter()
        .filter(|t| t.status == TaskStatus::DONE)
        .filter(|t| {
            t.last_run_at
                .as_deref()
                .or(Some(t.created_at.as_str()))
                .and_then(parse_db_time)
                .is_some_and(|when| when < older_than)
        })
        .collect();
    if !dry_run {
        let tx = conn.unchecked_transaction()?;
        for task in &old {
            tx.execute(
                "DELETE FROM scheduled_tasks WHERE id = ?1 AND status = 'done'",
                params![task.id],
            )?;
        }
        tx.commit()?;
    }
    Ok(old)
}

/// Fails every scheduled task with a dependency that failed or no longer
/// exists - it can never become due - returning the ids it failed. Run
/// repeatedly (the scheduler does, every poll), this fails whole chains.
pub fn fail_tasks_with_failed_dependencies(conn: &Connection) -> Result<Vec<i64>, Box<dyn Error>> {
    let mut stmt = conn.prepare(
        "SELECT scheduled_tasks.id, dependency.value, required.id IS NULL
         FROM scheduled_tasks, json_each(COALESCE(scheduled_tasks.depends_on, '[]')) AS dependency
         LEFT JOIN scheduled_tasks AS required ON required.id = dependency.value
         WHERE scheduled_tasks.status = 'scheduled'
           AND (required.id IS NULL
                OR (required.status = 'done' AND required.last_outcome = 'failed'))",
    )?;
    let blocked: Vec<(i64, i64, bool)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<Result<_, _>>()?;
    let mut failed = Vec::new();
    for (task, dependency, missing) in blocked {
        if failed.contains(&task) {
            continue;
        }
        let reason = if missing {
            format!("dependency #{} no longer exists", dependency)
        } else {
            format!("dependency #{} failed", dependency)
        };
        let rows = conn.execute(
            "UPDATE scheduled_tasks SET status = 'done', last_outcome = 'failed',
                 last_result = ?2, last_run_at = ?3
             WHERE id = ?1 AND status = 'scheduled'",
            params![task, reason, chrono::Utc::now().to_rfc3339()],
        )?;
        if rows > 0 {
            failed.push(task);
        }
    }
    Ok(failed)
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
        "UPDATE scheduled_tasks SET status = 'running', claimed_by = ?2, started_at = ?3,
             stop_requested = 0
         WHERE id = ?1 AND next_run_at <= ?3
           AND (status = 'scheduled' OR (status = 'running' AND ({})))
           AND {}",
        CLAIM_ABANDONED, DEPENDENCIES_MET
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
             claimed_by = NULL,
             stop_requested = 0
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

/// Records what `agent` is doing now (see `AgentRow::activity`).
pub fn set_agent_activity(
    conn: &Connection,
    agent: &str,
    activity: &str,
) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "UPDATE agents SET activity = ?2, activity_at = datetime('now') WHERE name = ?1",
        params![agent, activity],
    )?;
    Ok(())
}

/// SQL for `agent` (bound to `?1`) and its ancestors, nearest first, as
/// rows of `lineage(name, depth)`. Depth-bounded, so it ends even if
/// parents ever formed a loop (`set_agent_parent` refuses to make one).
const LINEAGE_CTE: &str = "WITH RECURSIVE lineage(name, depth) AS (
        SELECT ?1, 0
        UNION
        SELECT agents.parent, lineage.depth + 1
        FROM agents JOIN lineage ON agents.name = lineage.name
        WHERE agents.parent IS NOT NULL AND lineage.depth < 64
    )";

/// `agent` and its ancestors - its parent, its parent's parent... -
/// nearest first.
pub fn agent_lineage(conn: &Connection, agent: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let sql = format!(
        "{} SELECT name FROM lineage GROUP BY name ORDER BY MIN(depth)",
        LINEAGE_CTE
    );
    let names = conn
        .prepare(&sql)?
        .query_map(params![agent], |row| row.get(0))?
        .collect::<Result<Vec<String>, _>>()?;
    Ok(names)
}

/// Records `parent` as the agent that spawned `agent` (`None`: makes it a
/// top-level agent). Refused if `parent` is `agent` itself or one of its
/// descendants, which would make a loop.
pub fn set_agent_parent(
    conn: &Connection,
    agent: &str,
    parent: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = parent {
        if agent_lineage(conn, parent)?.iter().any(|a| a == agent) {
            return Err(format!(
                "'{}' can't be a sub-agent of '{}': '{}' is '{}' itself or one of its sub-agents",
                agent, parent, parent, agent
            )
            .into());
        }
        if get_agent(conn, parent)?.is_none() {
            return Err(format!("no agent named '{}'", parent).into());
        }
    }
    let rows = conn.execute(
        "UPDATE agents SET parent = ?2 WHERE name = ?1",
        params![agent, parent],
    )?;
    if rows == 0 {
        return Err(format!("no agent named '{}'", agent).into());
    }
    Ok(())
}

// --- Knowledge base ---

/// A knowledge base note.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KbNote {
    pub id: i64,
    pub title: String,
    pub body: String,
    pub tags: Vec<String>,
    /// The agent it's private to; `None` for a note every agent sees.
    pub agent_name: Option<String>,
    /// The agent (or user) that last wrote it.
    pub created_by: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A `kb_search` result: the note, and a snippet of where it matched with
/// the matching words in [brackets].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KbHit {
    pub note: KbNote,
    pub snippet: String,
}

/// Tags normalized for storage: lowercase, deduplicated, and wrapped in
/// commas (`,deploy,ci,`) so that one can be matched with a plain LIKE.
fn tags_to_db(tags: &[String]) -> String {
    let mut normalized: Vec<String> = tags
        .iter()
        .map(|t| t.trim().trim_start_matches('#').to_lowercase())
        .filter(|t| !t.is_empty() && !t.contains(','))
        .collect();
    normalized.sort();
    normalized.dedup();
    if normalized.is_empty() {
        String::new()
    } else {
        format!(",{},", normalized.join(","))
    }
}

fn tags_from_db(tags: &str) -> Vec<String> {
    tags.split(',')
        .filter(|t| !t.is_empty())
        .map(String::from)
        .collect()
}

const KB_COLUMNS: &str = "kb_notes.id, kb_notes.title, kb_notes.body, kb_notes.tags, \
     kb_notes.agent_name, kb_notes.created_by, kb_notes.created_at, kb_notes.updated_at";

fn row_to_kb_note(row: &rusqlite::Row) -> rusqlite::Result<KbNote> {
    Ok(KbNote {
        id: row.get(0)?,
        title: row.get(1)?,
        body: row.get(2)?,
        tags: tags_from_db(&row.get::<_, String>(3)?),
        agent_name: row.get(4)?,
        created_by: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
    })
}

/// Who's looking at the knowledge base.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum KbViewer {
    /// The user, through `faber kb`: sees every note, private ones too.
    User,
    /// An agent (`None`: one without an identity): sees shared notes, its
    /// own private ones, and those of every agent above it - the one that
    /// spawned it, and so on up. A sub-agent's own private notes aren't
    /// seen by the agents above it.
    Agent(Option<String>),
}

impl KbViewer {
    /// The SQL condition for notes this viewer sees, and the value to bind
    /// to its `?1` (always referenced, so the parameter count is fixed).
    fn condition(&self) -> (String, Option<&str>) {
        match self {
            KbViewer::User => ("(?1 IS NULL OR 1)".to_string(), None),
            KbViewer::Agent(None) => (
                "(kb_notes.agent_name IS NULL AND ?1 IS NULL)".to_string(),
                None,
            ),
            KbViewer::Agent(Some(agent)) => (
                format!(
                    "(kb_notes.agent_name IS NULL OR kb_notes.agent_name IN ({} SELECT name FROM lineage))",
                    LINEAGE_CTE
                ),
                Some(agent.as_str()),
            ),
        }
    }
}

/// Writes a note: creates it, or - if one with the same title (ignoring
/// case) already exists in the same scope - replaces its body and tags.
/// `private_to` makes it visible to that agent only. Returns the note's id
/// and whether it was newly created.
pub fn kb_write(
    conn: &Connection,
    title: &str,
    body: &str,
    tags: &[String],
    private_to: Option<&str>,
    author: Option<&str>,
) -> Result<(i64, bool), Box<dyn Error>> {
    let title = title.trim();
    if title.is_empty() {
        return Err("a note needs a title".into());
    }
    if body.trim().is_empty() {
        return Err("a note needs a body".into());
    }
    if let Some(agent) = private_to {
        if get_agent(conn, agent)?.is_none() {
            return Err(format!("no agent named '{}'", agent).into());
        }
    }
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM kb_notes
             WHERE lower(title) = lower(?1) AND COALESCE(agent_name, '') = COALESCE(?2, '')",
            params![title, private_to],
            |row| row.get(0),
        )
        .ok();
    match existing {
        Some(id) => {
            conn.execute(
                "UPDATE kb_notes SET title = ?2, body = ?3, tags = ?4, created_by = ?5,
                     updated_at = datetime('now')
                 WHERE id = ?1",
                params![id, title, body, tags_to_db(tags), author],
            )?;
            Ok((id, false))
        }
        None => {
            conn.execute(
                "INSERT INTO kb_notes (title, body, tags, agent_name, created_by)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![title, body, tags_to_db(tags), private_to, author],
            )?;
            Ok((conn.last_insert_rowid(), true))
        }
    }
}

/// The note with this id, if `viewer` may see it.
pub fn kb_get(
    conn: &Connection,
    id: i64,
    viewer: &KbViewer,
) -> Result<Option<KbNote>, Box<dyn Error>> {
    let (visible, who) = viewer.condition();
    let sql = format!(
        "SELECT {} FROM kb_notes WHERE {} AND id = ?2",
        KB_COLUMNS, visible
    );
    let mut rows = conn
        .prepare(&sql)?
        .query_map(params![who, id], row_to_kb_note)?
        .collect::<Vec<_>>();
    Ok(rows.pop().transpose()?)
}

/// The note with this title (ignoring case) that `viewer` sees: a private
/// one before a shared one, and the viewer's own before its parent's, and
/// so on up.
pub fn kb_get_by_title(
    conn: &Connection,
    title: &str,
    viewer: &KbViewer,
) -> Result<Option<KbNote>, Box<dyn Error>> {
    let owners: Vec<Option<String>> = match viewer {
        KbViewer::User => {
            // Any note with the title: shared first, then by owner.
            let sql = format!(
                "SELECT {} FROM kb_notes WHERE lower(title) = lower(?1)
                 ORDER BY agent_name IS NOT NULL, agent_name LIMIT 1",
                KB_COLUMNS
            );
            let mut rows = conn
                .prepare(&sql)?
                .query_map(params![title.trim()], row_to_kb_note)?
                .collect::<Vec<_>>();
            return Ok(rows.pop().transpose()?);
        }
        KbViewer::Agent(None) => vec![None],
        KbViewer::Agent(Some(agent)) => agent_lineage(conn, agent)?
            .into_iter()
            .map(Some)
            .chain([None])
            .collect(),
    };
    let sql = format!(
        "SELECT {} FROM kb_notes
         WHERE lower(title) = lower(?1) AND COALESCE(agent_name, '') = COALESCE(?2, '')",
        KB_COLUMNS
    );
    for owner in owners {
        let mut rows = conn
            .prepare(&sql)?
            .query_map(params![title.trim(), owner], row_to_kb_note)?
            .collect::<Vec<_>>();
        if let Some(note) = rows.pop().transpose()? {
            return Ok(Some(note));
        }
    }
    Ok(None)
}

/// Words too common to say anything about which note is meant. Left out
/// of a search, so that "how do I deploy a release?" doesn't match every
/// note with an "a" in it.
const KB_STOP_WORDS: &[&str] = &[
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "should", "so",
    "that", "the", "this", "to", "us", "was", "we", "what", "when", "where", "which", "who", "why",
    "with", "you", "your",
];

/// An FTS5 query for free text: its words, minus stop words (unless that
/// leaves none), each quoted - so nothing in it is read as FTS5 syntax -
/// and OR-ed, leaving it to the ranking to put notes matching more of them
/// first. `None` if there are no words at all.
pub fn kb_fts_query(text: &str) -> Option<String> {
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let meaningful: Vec<&str> = words
        .iter()
        .copied()
        .filter(|w| !KB_STOP_WORDS.contains(&w.to_lowercase().as_str()))
        .collect();
    let words = if meaningful.is_empty() {
        words
    } else {
        meaningful
    };
    (!words.is_empty()).then(|| {
        words
            .iter()
            .map(|w| format!("\"{}\"", w))
            .collect::<Vec<_>>()
            .join(" OR ")
    })
}

/// Ranked full-text search over the notes `viewer` sees, optionally only
/// those tagged `tag`. Matches in the title count most, then tags, then
/// the body.
pub fn kb_search(
    conn: &Connection,
    query: &str,
    viewer: &KbViewer,
    tag: Option<&str>,
    limit: usize,
) -> Result<Vec<KbHit>, Box<dyn Error>> {
    let (visible, who) = viewer.condition();
    let fts = kb_fts_query(query).ok_or("the search has no words in it")?;
    let sql = format!(
        "SELECT {}, snippet(kb_fts, 1, '[', ']', '…', 16)
         FROM kb_fts JOIN kb_notes ON kb_notes.id = kb_fts.rowid
         WHERE kb_fts MATCH ?2 AND {} AND (?3 IS NULL OR kb_notes.tags LIKE '%,' || ?3 || ',%')
         ORDER BY bm25(kb_fts, 10.0, 1.0, 5.0)
         LIMIT ?4",
        KB_COLUMNS, visible
    );
    let tag = tag.map(|t| t.trim().trim_start_matches('#').to_lowercase());
    let hits = conn
        .prepare(&sql)?
        .query_map(params![who, fts, tag, limit as i64], |row| {
            Ok(KbHit {
                note: row_to_kb_note(row)?,
                snippet: row.get(8)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(hits)
}

/// The notes `viewer` sees, most recently updated first, optionally only
/// those tagged `tag`.
pub fn kb_list(
    conn: &Connection,
    viewer: &KbViewer,
    tag: Option<&str>,
    limit: usize,
) -> Result<Vec<KbNote>, Box<dyn Error>> {
    let (visible, who) = viewer.condition();
    let sql = format!(
        "SELECT {} FROM kb_notes
         WHERE {} AND (?2 IS NULL OR tags LIKE '%,' || ?2 || ',%')
         ORDER BY updated_at DESC, id DESC LIMIT ?3",
        KB_COLUMNS, visible
    );
    let tag = tag.map(|t| t.trim().trim_start_matches('#').to_lowercase());
    let notes = conn
        .prepare(&sql)?
        .query_map(params![who, tag, limit as i64], row_to_kb_note)?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(notes)
}

/// Deletes a note `viewer` may see. Returns whether one was deleted.
pub fn kb_delete(conn: &Connection, id: i64, viewer: &KbViewer) -> Result<bool, Box<dyn Error>> {
    let (visible, who) = viewer.condition();
    let rows = conn.execute(
        &format!("DELETE FROM kb_notes WHERE {} AND id = ?2", visible),
        params![who, id],
    )?;
    Ok(rows > 0)
}

// --- Agent Config ---

/// An agent's own settings, overriding the session's: set one at a time
/// (`agent_configure`), or all at once from a profile when the agent is
/// made from one. Each field is an `agent_data` key, `config:<field>`.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
pub struct AgentConfig {
    pub model: Option<String>,
    pub endpoint: Option<String>,
    pub system_prompt: Option<String>,
    /// Path to the file holding the API key.
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub context_window: Option<u32>,
    /// Request parameters (`temperature`, `reasoning_effort`, ...), layered
    /// over the session's.
    #[serde(default)]
    pub parameters: Option<serde_json::Map<String, serde_json::Value>>,
    /// The only tools it may use, of those the session has.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// The profile it was made from, if any - for showing only: the
    /// settings were copied when it was made.
    #[serde(default)]
    pub profile: Option<String>,
    /// Whether it gets the unsafe tools (see `--unsafe-tools`). An agent
    /// made by another always has it set; one that doesn't is unsafe only
    /// in a session started with `--unsafe-tools`.
    #[serde(default)]
    pub unsafe_tools: Option<bool>,
    /// The directory it works in, absolute: what its tools' relative paths
    /// mean, where its commands run, all its sandbox can write to. Unset,
    /// the process's. Only the user - or an agent with the unsafe tools -
    /// can point an agent somewhere else; one an agent makes works where
    /// its maker does.
    #[serde(default)]
    pub cwd: Option<String>,
}

const AGENT_CONFIG_FIELDS: [&str; 11] = [
    "model",
    "endpoint",
    "system_prompt",
    "api_key",
    "max_tokens",
    "context_window",
    "parameters",
    "tools",
    "profile",
    "unsafe_tools",
    "cwd",
];

pub fn get_agent_config(
    conn: &Connection,
    agent_name: &str,
) -> Result<AgentConfig, Box<dyn Error>> {
    let mut fields = serde_json::Map::new();
    for field in AGENT_CONFIG_FIELDS {
        let Some(text) = get_agent_data(conn, agent_name, &format!("config:{}", field))? else {
            continue;
        };
        // Strings are stored as they are, everything else as JSON; a value
        // that doesn't parse as what the field takes is ignored.
        let value = match field {
            "max_tokens" | "context_window" | "parameters" | "tools" | "unsafe_tools" => {
                match serde_json::from_str(&text) {
                    Ok(value) => value,
                    Err(_) => continue,
                }
            }
            _ => serde_json::Value::String(text),
        };
        fields.insert(field.to_string(), value);
    }
    let mut config = AgentConfig::default();
    for (field, value) in fields {
        let mut one = serde_json::Map::new();
        one.insert(field, value);
        if let Ok(parsed) = serde_json::from_value::<AgentConfig>(one.into()) {
            config = merge_agent_config(config, parsed);
        }
    }
    Ok(config)
}

/// `base` with every field `over` sets replaced.
fn merge_agent_config(base: AgentConfig, over: AgentConfig) -> AgentConfig {
    AgentConfig {
        model: over.model.or(base.model),
        endpoint: over.endpoint.or(base.endpoint),
        system_prompt: over.system_prompt.or(base.system_prompt),
        api_key: over.api_key.or(base.api_key),
        max_tokens: over.max_tokens.or(base.max_tokens),
        context_window: over.context_window.or(base.context_window),
        parameters: over.parameters.or(base.parameters),
        tools: over.tools.or(base.tools),
        profile: over.profile.or(base.profile),
        unsafe_tools: over.unsafe_tools.or(base.unsafe_tools),
        cwd: over.cwd.or(base.cwd),
    }
}

/// Replaces all of `agent_name`'s config with `config`: fields it doesn't
/// set are cleared. One transaction, so nobody sees half of it.
pub fn set_agent_config(
    conn: &Connection,
    agent_name: &str,
    config: &AgentConfig,
) -> Result<(), Box<dyn Error>> {
    let serde_json::Value::Object(fields) = serde_json::to_value(config)? else {
        return Err("agent config isn't an object".into());
    };
    let tx = conn.unchecked_transaction()?;
    for field in AGENT_CONFIG_FIELDS {
        let key = format!("config:{}", field);
        match fields.get(field) {
            None | Some(serde_json::Value::Null) => {
                delete_agent_data(&tx, agent_name, &key)?;
            }
            Some(serde_json::Value::String(text)) => set_agent_data(&tx, agent_name, &key, text)?,
            Some(value) => set_agent_data(&tx, agent_name, &key, &value.to_string())?,
        }
    }
    tx.commit()?;
    Ok(())
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

/// Whether an agent's chat session (or the session running it, for a
/// sub-agent) is still heartbeating.
pub fn agent_session_is_live(agent: &AgentRow, now: chrono::DateTime<chrono::Utc>) -> bool {
    agent.session_id.is_some()
        && agent
            .heartbeat_at
            .as_deref()
            .and_then(parse_db_time)
            .is_some_and(|t| now - t < chrono::Duration::seconds(30))
}

/// Each task's dependencies that aren't done successfully yet - among
/// `all` tasks; one that no longer exists counts as unmet.
pub fn unmet_dependencies(all: &[TaskRow]) -> std::collections::HashMap<i64, Vec<i64>> {
    let met = |id: &i64| {
        all.iter().any(|t| {
            t.id == *id
                && t.status == TaskStatus::DONE
                && t.last_outcome.as_deref() == Some("succeeded")
        })
    };
    all.iter()
        .filter_map(|t| {
            let unmet: Vec<i64> = t.depends_on.iter().copied().filter(|d| !met(d)).collect();
            (!unmet.is_empty()).then_some((t.id, unmet))
        })
        .collect()
}

/// Records `events` of `agent` (for `task_id`'s run, if any), dropping
/// its oldest beyond `agent_io::MAX_EVENTS_PER_AGENT`.
pub fn append_agent_events(
    conn: &Connection,
    agent: &str,
    task_id: Option<i64>,
    events: &[AgentEvent],
) -> Result<(), Box<dyn Error>> {
    let tx = conn.unchecked_transaction()?;
    {
        let mut insert =
            tx.prepare("INSERT INTO agent_events (agent, task_id, event) VALUES (?1, ?2, ?3)")?;
        for event in events {
            insert.execute(params![agent, task_id, serde_json::to_string(event)?])?;
        }
    }
    tx.execute(
        "DELETE FROM agent_events WHERE agent = ?1 AND id <= (
             SELECT id FROM agent_events WHERE agent = ?1
             ORDER BY id DESC LIMIT 1 OFFSET ?2)",
        params![agent, crate::agent_io::MAX_EVENTS_PER_AGENT],
    )?;
    tx.commit()?;
    Ok(())
}

/// The recorded events `filter` picks, oldest first.
pub fn agent_events(
    conn: &Connection,
    filter: &EventFilter,
) -> Result<Vec<AgentEventRow>, Box<dyn Error>> {
    let mut sql = "SELECT id, agent, task_id, at, event FROM agent_events WHERE 1".to_string();
    let mut values: Vec<rusqlite::types::Value> = Vec::new();
    if let Some(agent) = &filter.agent {
        values.push(agent.clone().into());
        sql.push_str(&format!(" AND agent = ?{}", values.len()));
    }
    if let Some(task_id) = filter.task_id {
        values.push(task_id.into());
        sql.push_str(&format!(" AND task_id = ?{}", values.len()));
    }
    if let Some(after) = filter.after {
        values.push(after.into());
        sql.push_str(&format!(" AND id > ?{}", values.len()));
    }
    // Without `after`, the latest ones: newest first here, reversed below.
    sql.push_str(if filter.after.is_some() {
        " ORDER BY id ASC"
    } else {
        " ORDER BY id DESC"
    });
    values.push((filter.limit as i64).into());
    sql.push_str(&format!(" LIMIT ?{}", values.len()));
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(values), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<i64>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
        ))
    })?;
    let mut events = Vec::new();
    for row in rows {
        let (id, agent, task_id, at, event) = row?;
        // An event of a kind this faber doesn't know: skip it.
        let Ok(event) = serde_json::from_str(&event) else {
            continue;
        };
        events.push(AgentEventRow {
            id,
            agent,
            task_id,
            at,
            event,
        });
    }
    if filter.after.is_none() {
        events.reverse();
    }
    Ok(events)
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
    fn test_request_task_stop() {
        let conn = test_db();
        let past = (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
        let id = create_oneshot_task(&conn, "t", "", &past, "x", None).unwrap();
        assert!(!request_task_stop(&conn, id).unwrap(), "not running");
        // Running in a live session: it's asked to stop.
        create_agent(&conn, "w", "").unwrap();
        assert!(claim_agent(&conn, "w", "s").unwrap());
        assert!(claim_task(&conn, id, "s").unwrap());
        assert!(request_task_stop(&conn, id).unwrap());
        assert!(get_task(&conn, id).unwrap().unwrap().stop_requested);
        let outcome = TaskOutcome {
            succeeded: false,
            exit_code: None,
            result: "stopped".to_string(),
        };
        assert!(finish_task(&conn, id, "s", &outcome).unwrap());
        assert!(!get_task(&conn, id).unwrap().unwrap().stop_requested);
        // Running in a session that's gone: ended right away.
        let id = create_oneshot_task(&conn, "t2", "", &past, "x", None).unwrap();
        assert!(claim_task(&conn, id, "dead-session").unwrap());
        assert!(request_task_stop(&conn, id).unwrap());
        let task = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(
            (task.status.as_str(), task.last_outcome.as_deref()),
            ("done", Some("failed"))
        );
    }

    #[test]
    fn test_run_task_now_makes_any_waiting_or_finished_task_due() {
        let conn = test_db();
        let later = (chrono::Utc::now() + chrono::Duration::days(1)).to_rfc3339();
        let id = create_oneshot_task(&conn, "t", "", &later, "x", None).unwrap();
        assert!(get_pending_tasks(&conn).unwrap().is_empty());
        for status in ["scheduled", "held", "disabled", "done"] {
            conn.execute(
                "UPDATE scheduled_tasks SET status = ?2, next_run_at = ?3 WHERE id = ?1",
                params![id, status, later],
            )
            .unwrap();
            assert!(run_task_now(&conn, id).unwrap(), "{}", status);
            let due = get_pending_tasks(&conn).unwrap();
            assert_eq!(due.len(), 1, "{}", status);
            assert_eq!(due[0].status, "scheduled");
        }
        assert!(claim_task(&conn, id, "s").unwrap());
        assert!(!run_task_now(&conn, id).unwrap(), "running");
        assert!(!run_task_now(&conn, 999).unwrap());
    }

    #[test]
    fn test_agent_events_are_kept_per_agent_and_read_in_order() {
        let conn = test_db();
        create_agent(&conn, "a", "").unwrap();
        create_agent(&conn, "b", "").unwrap();
        let text = |t: &str| AgentEvent::Text {
            text: t.to_string(),
        };
        append_agent_events(&conn, "a", Some(1), &[text("1"), text("2")]).unwrap();
        append_agent_events(&conn, "b", None, &[text("3")]).unwrap();
        append_agent_events(&conn, "a", None, &[text("4")]).unwrap();
        let read = |filter: EventFilter| -> Vec<AgentEvent> {
            agent_events(&conn, &filter)
                .unwrap()
                .into_iter()
                .map(|r| r.event)
                .collect()
        };
        let a = |after, limit| EventFilter {
            agent: Some("a".to_string()),
            after,
            limit,
            ..Default::default()
        };
        assert_eq!(read(a(None, 10)), vec![text("1"), text("2"), text("4")]);
        // The latest ones, still oldest first.
        assert_eq!(read(a(None, 2)), vec![text("2"), text("4")]);
        let first = agent_events(&conn, &a(None, 10)).unwrap()[0].id;
        assert_eq!(read(a(Some(first), 1)), vec![text("2")]);
        let task = EventFilter {
            task_id: Some(1),
            limit: 10,
            ..Default::default()
        };
        assert_eq!(read(task), vec![text("1"), text("2")]);
        // An unknown agent's events can't be recorded; a deleted one's go.
        assert!(append_agent_events(&conn, "nobody", None, &[text("x")]).is_err());
        delete_agent(&conn, "a").unwrap();
        assert_eq!(
            read(EventFilter {
                limit: 10,
                ..Default::default()
            }),
            vec![text("3")]
        );
    }

    #[test]
    fn test_agent_events_are_capped_per_agent() {
        let conn = test_db();
        create_agent(&conn, "a", "").unwrap();
        let events: Vec<AgentEvent> = (0..crate::agent_io::MAX_EVENTS_PER_AGENT + 5)
            .map(|i| AgentEvent::Text {
                text: i.to_string(),
            })
            .collect();
        append_agent_events(&conn, "a", None, &events).unwrap();
        let kept = agent_events(
            &conn,
            &EventFilter {
                limit: 10_000,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(kept.len() as i64, crate::agent_io::MAX_EVENTS_PER_AGENT);
        assert_eq!(
            kept[0].event,
            AgentEvent::Text {
                text: "5".to_string()
            }
        );
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

    fn temp_db_path(name: &str) -> String {
        let path =
            std::env::temp_dir().join(format!("faber_test_{}_{}.db", name, std::process::id()));
        let _ = std::fs::remove_file(&path);
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn test_open_read_only_never_creates_a_database() {
        let path = temp_db_path("missing");
        let err = open_read_only(&path).unwrap_err().to_string();
        assert!(err.starts_with("no database at "), "{err}");
        assert!(!std::path::Path::new(&path).exists());
    }

    #[test]
    fn test_open_read_only_reads_but_cannot_write() {
        let path = temp_db_path("current");
        let conn = Connection::open(&path).unwrap();
        initialize_db(&conn).unwrap();
        create_oneshot_task(&conn, "t", "", &chrono::Utc::now().to_rfc3339(), "", None).unwrap();
        drop(conn);

        let conn = open_read_only(&path).unwrap();
        assert_eq!(list_tasks(&conn, None).unwrap().len(), 1);
        assert!(conn.execute("DELETE FROM scheduled_tasks", []).is_err());
        drop(conn);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_open_read_only_leaves_old_and_foreign_databases_alone() {
        let path = temp_db_path("old");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE scheduled_tasks (id INTEGER PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 1);",
        )
        .unwrap();
        drop(conn);
        let err = open_read_only(&path).unwrap_err().to_string();
        assert!(err.contains("older version of faber"), "{err}");
        let conn = Connection::open(&path).unwrap();
        assert!(
            !table_has_column(&conn, "scheduled_tasks", "status").unwrap(),
            "not migrated"
        );
        drop(conn);
        std::fs::remove_file(&path).unwrap();

        let path = temp_db_path("foreign");
        Connection::open(&path)
            .unwrap()
            .execute_batch("CREATE TABLE notes (body TEXT);")
            .unwrap();
        let err = open_read_only(&path).unwrap_err().to_string();
        assert!(err.contains("isn't a faber database"), "{err}");
        std::fs::remove_file(&path).unwrap();

        let path = temp_db_path("text");
        std::fs::write(&path, "just some text, not SQLite\n".repeat(100)).unwrap();
        let err = open_read_only(&path).unwrap_err().to_string();
        assert!(err.contains("isn't a faber database"), "{err}");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_create_task_kinds_and_validation() {
        let conn = test_db();
        let at = chrono::Utc::now().to_rfc3339();
        let task = |kind: &str| NewTask {
            name: "n".to_string(),
            description: String::new(),
            kind: kind.to_string(),
            command: "do it".to_string(),
            agent_name: None,
            schedule: TaskSchedule::Once { at: at.clone() },
            held: false,
            depends_on: Vec::new(),
            profile: None,
            run_safe: false,
            cwd: None,
        };
        let id = create_task(&conn, &task("prompt")).unwrap();
        let row = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(row.kind, TaskKind::PROMPT);
        assert_eq!(row.task_type, "oneshot");
        assert_eq!(row.status, TaskStatus::SCHEDULED);
        // The old creation functions still make tool tasks.
        let id = create_oneshot_task(&conn, "t", "", &at, "", None).unwrap();
        assert_eq!(get_task(&conn, id).unwrap().unwrap().kind, TaskKind::TOOL);

        assert!(create_task(&conn, &task("shell")).is_err());
        let mut cron = task("prompt");
        cron.schedule = TaskSchedule::Cron {
            expression: "not cron".to_string(),
            max_runs: None,
        };
        assert!(create_task(&conn, &cron).is_err());
    }

    #[test]
    fn test_plain_text_commands_make_prompt_tasks() {
        let conn = test_db();
        let at = chrono::Utc::now().to_rfc3339();
        let kind = |id: i64| get_task(&conn, id).unwrap().unwrap().kind;
        let joke = create_oneshot_task(&conn, "j", "", &at, "Tell the user a joke", None).unwrap();
        assert_eq!(kind(joke), TaskKind::PROMPT);
        let tool = create_oneshot_task(
            &conn,
            "t",
            "",
            &at,
            r#"{"tool":"glob","arguments":{}}"#,
            None,
        )
        .unwrap();
        assert_eq!(kind(tool), TaskKind::TOOL);
        let cron = create_cron_task(
            &conn,
            "c",
            "Summarize the news",
            "0 0 9 * * * *",
            "",
            None,
            None,
        )
        .unwrap();
        let row = get_task(&conn, cron).unwrap().unwrap();
        assert_eq!(row.kind, TaskKind::PROMPT);
        assert_eq!(
            row.command, "Summarize the news",
            "the description stands in for the command"
        );
        assert!(!is_tool_call(r#"{"tool": ""}"#));
        assert!(!is_tool_call("[1, 2]"));
    }

    #[test]
    fn test_scheduled_plain_text_tool_tasks_become_prompt_tasks() {
        let conn = test_db();
        let at = chrono::Utc::now().to_rfc3339();
        let insert = |name: &str, command: &str, status: &str| {
            conn.execute(
                "INSERT INTO scheduled_tasks (name, description, task_type, run_at, next_run_at, command, kind, status)
                 VALUES (?1, 'from the description', 'oneshot', ?2, ?2, ?3, 'tool', ?4)",
                params![name, at, command, status],
            )
            .unwrap();
        };
        insert("text", "Tell a joke", "scheduled");
        insert("empty", "", "disabled");
        insert("call", r#"{"tool":"glob"}"#, "scheduled");
        insert("ran", "Tell a joke", "done");
        initialize_db(&conn).unwrap();
        let kinds: Vec<(String, String, String)> = list_tasks(&conn, None)
            .unwrap()
            .into_iter()
            .map(|t| (t.name, t.kind, t.command))
            .collect();
        let expected = [
            ("text", "prompt", "Tell a joke"),
            ("empty", "prompt", "from the description"),
            ("call", "tool", r#"{"tool":"glob"}"#),
            ("ran", "tool", "Tell a joke"),
        ]
        .map(|(a, b, c)| (a.to_string(), b.to_string(), c.to_string()));
        assert_eq!(kinds, expected);
    }

    #[test]
    fn test_prune_tasks_removes_only_old_done_tasks() {
        let conn = test_db();
        let now = chrono::Utc::now();
        let at = now.to_rfc3339();
        let make = |name: &str, status: &str, last_run: Option<chrono::Duration>| {
            let id = create_oneshot_task(&conn, name, "", &at, "x", None).unwrap();
            conn.execute(
                "UPDATE scheduled_tasks SET status = ?1, last_run_at = ?2 WHERE id = ?3",
                params![status, last_run.map(|ago| (now - ago).to_rfc3339()), id],
            )
            .unwrap();
        };
        make("old done", "done", Some(chrono::Duration::days(10)));
        make("recent done", "done", Some(chrono::Duration::hours(1)));
        make("old disabled", "disabled", Some(chrono::Duration::days(10)));
        make(
            "old scheduled",
            "scheduled",
            Some(chrono::Duration::days(10)),
        );

        let cutoff = now - chrono::Duration::days(7);
        let would = prune_tasks(&conn, cutoff, true).unwrap();
        assert_eq!(
            would.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            ["old done"]
        );
        assert_eq!(
            list_tasks(&conn, None).unwrap().len(),
            4,
            "dry run deletes nothing"
        );

        let pruned = prune_tasks(&conn, cutoff, false).unwrap();
        assert_eq!(pruned.len(), 1);
        let left: Vec<String> = list_tasks(&conn, None)
            .unwrap()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(left, ["recent done", "old disabled", "old scheduled"]);
    }

    #[test]
    fn test_held_task_is_not_picked_up_until_released() {
        let conn = test_db();
        live_session(&conn, "s");
        let id = due_oneshot(&conn);

        assert!(set_task_held(&conn, id, true).unwrap());
        assert_eq!(
            get_task(&conn, id).unwrap().unwrap().status,
            TaskStatus::HELD
        );
        assert!(get_pending_tasks(&conn).unwrap().is_empty());
        assert!(!claim_task(&conn, id, "s").unwrap());
        assert!(!set_task_held(&conn, id, true).unwrap(), "already held");

        assert!(set_task_held(&conn, id, false).unwrap());
        assert_eq!(
            get_task(&conn, id).unwrap().unwrap().status,
            TaskStatus::SCHEDULED
        );
        assert!(
            claim_task(&conn, id, "s").unwrap(),
            "due, so it runs right away"
        );
        assert!(
            !set_task_held(&conn, id, true).unwrap(),
            "running tasks can't be held"
        );
        assert!(!set_task_held(&conn, id, false).unwrap(), "nor released");
    }

    #[test]
    fn test_held_task_can_be_disabled_and_created_held() {
        let conn = test_db();
        let id = due_oneshot(&conn);
        set_task_held(&conn, id, true).unwrap();
        assert!(set_task_enabled(&conn, id, false).unwrap());
        assert_eq!(
            get_task(&conn, id).unwrap().unwrap().status,
            TaskStatus::DISABLED
        );

        let held = create_task(
            &conn,
            &NewTask {
                name: "later".to_string(),
                description: String::new(),
                kind: TaskKind::PROMPT.to_string(),
                command: "go".to_string(),
                agent_name: None,
                schedule: TaskSchedule::Once {
                    at: chrono::Utc::now().to_rfc3339(),
                },
                held: true,
                depends_on: Vec::new(),
                profile: None,
                run_safe: false,
                cwd: None,
            },
        )
        .unwrap();
        assert_eq!(
            get_task(&conn, held).unwrap().unwrap().status,
            TaskStatus::HELD
        );
    }

    #[test]
    fn test_task_table_rebuilt_to_allow_held() {
        let conn = Connection::open_in_memory().unwrap();
        initialize_db(&conn).unwrap();
        // Recreate the table as it was before `held`: same columns, but a
        // status constraint without it.
        conn.execute_batch(&format!(
            "PRAGMA foreign_keys = OFF;
             DROP TABLE scheduled_tasks;
             CREATE TABLE scheduled_tasks ({});
             PRAGMA foreign_keys = ON;",
            TASK_TABLE_DEFINITION.replace("'scheduled', 'held',", "'scheduled',")
        ))
        .unwrap();
        create_agent(&conn, "bob", "").unwrap();
        let at = chrono::Utc::now().to_rfc3339();
        let keep = create_oneshot_task(&conn, "keep", "d", &at, "say hi", Some("bob")).unwrap();
        let gone = create_oneshot_task(&conn, "gone", "", &at, "x", None).unwrap();
        delete_task(&conn, gone).unwrap();
        assert!(
            set_task_held(&conn, keep, true).is_err(),
            "old constraint rejects held"
        );

        initialize_db(&conn).unwrap();

        let task = get_task(&conn, keep).unwrap().unwrap();
        assert_eq!(
            (task.name.as_str(), task.agent_name.as_deref()),
            ("keep", Some("bob"))
        );
        assert_eq!(task.kind, TaskKind::PROMPT);
        assert!(set_task_held(&conn, keep, true).unwrap());
        // Ids of deleted tasks aren't handed out again.
        let next = create_oneshot_task(&conn, "next", "", &at, "x", None).unwrap();
        assert!(next > gone, "{next} <= {gone}");
        // Indexes and the agent foreign key are back.
        let indexes: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'scheduled_tasks'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(
            indexes.contains(&"idx_tasks_due".to_string()),
            "{indexes:?}"
        );
        assert!(
            indexes.contains(&"idx_tasks_agent".to_string()),
            "{indexes:?}"
        );
        delete_agent(&conn, "bob").unwrap();
        assert!(
            get_task(&conn, keep).unwrap().is_none(),
            "deleting an agent still deletes its tasks"
        );
    }

    fn tags(list: &[&str]) -> Vec<String> {
        list.iter().map(|t| t.to_string()).collect()
    }

    fn agent(name: &str) -> KbViewer {
        KbViewer::Agent(Some(name.to_string()))
    }

    fn titles(hits: &[KbHit]) -> Vec<&str> {
        hits.iter().map(|h| h.note.title.as_str()).collect()
    }

    #[test]
    fn test_kb_write_creates_then_updates_by_title() {
        let conn = test_db();
        let (id, created) = kb_write(
            &conn,
            "Deploy steps",
            "run make deploy",
            &tags(&["Ops", "#deploy", "ops"]),
            None,
            Some("bob"),
        )
        .unwrap();
        assert!(created);
        let note = kb_get(&conn, id, &KbViewer::User).unwrap().unwrap();
        assert_eq!(note.tags, ["deploy", "ops"], "normalized and deduplicated");
        assert_eq!(note.created_by.as_deref(), Some("bob"));

        let (again, created) =
            kb_write(&conn, "deploy STEPS", "run make release", &[], None, None).unwrap();
        assert_eq!((again, created), (id, false));
        let note = kb_get(&conn, id, &KbViewer::User).unwrap().unwrap();
        assert_eq!(note.body, "run make release");
        assert!(note.tags.is_empty());

        assert!(kb_write(&conn, "  ", "x", &[], None, None).is_err());
        assert!(kb_write(&conn, "t", " ", &[], None, None).is_err());
        assert!(kb_write(&conn, "t", "x", &[], Some("nobody"), None).is_err());
    }

    #[test]
    fn test_kb_private_notes_are_scoped() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        create_agent(&conn, "bob", "").unwrap();
        kb_write(&conn, "Prefs", "shared prefs", &[], None, None).unwrap();
        let (alices, created) = kb_write(
            &conn,
            "Prefs",
            "alice likes tabs",
            &[],
            Some("alice"),
            Some("alice"),
        )
        .unwrap();
        assert!(created, "same title, different scope: a separate note");

        let seen = |viewer: &KbViewer| kb_list(&conn, viewer, None, 10).unwrap().len();
        assert_eq!(seen(&agent("alice")), 2);
        assert_eq!(seen(&agent("bob")), 1);
        assert_eq!(seen(&KbViewer::Agent(None)), 1);
        assert_eq!(seen(&KbViewer::User), 2);
        assert_eq!(
            kb_get_by_title(&conn, "prefs", &agent("alice"))
                .unwrap()
                .unwrap()
                .body,
            "alice likes tabs",
            "an agent's own note comes before the shared one"
        );
        assert!(kb_get(&conn, alices, &agent("bob")).unwrap().is_none());
        assert!(
            !kb_delete(&conn, alices, &agent("bob")).unwrap(),
            "can't delete what it can't see"
        );
        assert!(
            kb_search(&conn, "tabs", &agent("bob"), None, 10)
                .unwrap()
                .is_empty()
        );

        delete_agent(&conn, "alice").unwrap();
        assert_eq!(
            seen(&KbViewer::User),
            1,
            "an agent's private notes go with it"
        );
        assert!(
            kb_search(&conn, "tabs", &KbViewer::User, None, 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_kb_search_ranks_titles_first_and_stems_words() {
        let conn = test_db();
        kb_write(
            &conn,
            "Release checklist",
            "Before deploying, bump the version.",
            &tags(&["release"]),
            None,
            None,
        )
        .unwrap();
        kb_write(
            &conn,
            "Deploying to staging",
            "Use the staging cluster.",
            &tags(&["ops"]),
            None,
            None,
        )
        .unwrap();
        kb_write(&conn, "Coffee machine", "Descale monthly.", &[], None, None).unwrap();

        let hits = kb_search(&conn, "how do we deploy?", &KbViewer::User, None, 10).unwrap();
        assert_eq!(titles(&hits), ["Deploying to staging", "Release checklist"]);
        assert!(
            hits[1].snippet.contains("[deploying]"),
            "{}",
            hits[1].snippet
        );

        let hits = kb_search(&conn, "deploy", &KbViewer::User, Some("#Release"), 10).unwrap();
        assert_eq!(titles(&hits), ["Release checklist"]);
        assert_eq!(
            kb_search(&conn, "deploy", &KbViewer::User, None, 1)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_kb_search_index_follows_updates_and_deletes() {
        let conn = test_db();
        let (id, _) = kb_write(&conn, "Build", "uses make", &[], None, None).unwrap();
        kb_write(&conn, "Build", "uses cargo", &[], None, None).unwrap();
        assert!(
            kb_search(&conn, "make", &KbViewer::User, None, 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            kb_search(&conn, "cargo", &KbViewer::User, None, 10)
                .unwrap()
                .len(),
            1
        );
        assert!(kb_delete(&conn, id, &KbViewer::User).unwrap());
        assert!(
            kb_search(&conn, "cargo", &KbViewer::User, None, 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_kb_search_survives_query_syntax_in_free_text() {
        let conn = test_db();
        kb_write(
            &conn,
            "Quoting",
            "say \"hello\" NEAR the door",
            &[],
            None,
            None,
        )
        .unwrap();
        for query in [
            "\"hello",
            "hello*",
            "NEAR(hello door)",
            "title:hello",
            "-hello",
            "hello AND OR NOT",
            "a.b/c's",
        ] {
            assert!(
                kb_search(&conn, query, &KbViewer::User, None, 10).is_ok(),
                "query {query:?} failed"
            );
        }
        assert!(
            kb_search(&conn, "?!", &KbViewer::User, None, 10).is_err(),
            "no words"
        );
        assert_eq!(
            kb_fts_query("deploy steps?").unwrap(),
            "\"deploy\" OR \"steps\""
        );
        assert_eq!(
            kb_fts_query("How do I deploy a release?").unwrap(),
            "\"deploy\" OR \"release\""
        );
        assert_eq!(
            kb_fts_query("who are we").unwrap(),
            "\"who\" OR \"are\" OR \"we\""
        );
    }

    /// root <- middle <- leaf, each with a private note, plus a shared one.
    fn agent_tree(conn: &Connection) {
        for (agent, parent) in [
            ("root", None),
            ("middle", Some("root")),
            ("leaf", Some("middle")),
        ] {
            create_agent(conn, agent, "").unwrap();
            set_agent_parent(conn, agent, parent).unwrap();
            kb_write(
                conn,
                &format!("{} notes", agent),
                "private stuff",
                &[],
                Some(agent),
                Some(agent),
            )
            .unwrap();
        }
        kb_write(conn, "team notes", "shared stuff", &[], None, None).unwrap();
    }

    fn visible(conn: &Connection, agent: &str) -> Vec<String> {
        let mut titles: Vec<String> =
            kb_list(conn, &KbViewer::Agent(Some(agent.to_string())), None, 50)
                .unwrap()
                .into_iter()
                .map(|n| n.title)
                .collect();
        titles.sort();
        titles
    }

    #[test]
    fn test_kb_private_notes_are_inherited_down_not_up() {
        let conn = test_db();
        agent_tree(&conn);
        assert_eq!(
            agent_lineage(&conn, "leaf").unwrap(),
            ["leaf", "middle", "root"]
        );
        assert_eq!(
            visible(&conn, "leaf"),
            ["leaf notes", "middle notes", "root notes", "team notes"]
        );
        assert_eq!(
            visible(&conn, "middle"),
            ["middle notes", "root notes", "team notes"]
        );
        assert_eq!(visible(&conn, "root"), ["root notes", "team notes"]);
        // Search follows the same rule.
        let found = kb_search(
            &conn,
            "private",
            &KbViewer::Agent(Some("middle".to_string())),
            None,
            50,
        )
        .unwrap();
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_kb_deleting_a_middle_agent_keeps_the_rest() {
        let conn = test_db();
        agent_tree(&conn);
        delete_agent(&conn, "middle").unwrap();
        // Its private notes go; its sub-agent is now top-level; shared stays.
        assert_eq!(visible(&conn, "leaf"), ["leaf notes", "team notes"]);
        assert_eq!(get_agent(&conn, "leaf").unwrap().unwrap().parent, None);
        assert_eq!(visible(&conn, "root"), ["root notes", "team notes"]);
    }

    #[test]
    fn test_kb_title_lookup_prefers_the_nearest_note() {
        let conn = test_db();
        agent_tree(&conn);
        kb_write(&conn, "Plan", "root's", &[], Some("root"), None).unwrap();
        kb_write(&conn, "Plan", "middle's", &[], Some("middle"), None).unwrap();
        kb_write(&conn, "Plan", "everyone's", &[], None, None).unwrap();
        let body = |agent: &str| {
            kb_get_by_title(&conn, "plan", &KbViewer::Agent(Some(agent.to_string())))
                .unwrap()
                .unwrap()
                .body
        };
        assert_eq!(body("leaf"), "middle's");
        assert_eq!(body("middle"), "middle's");
        assert_eq!(body("root"), "root's");
        assert_eq!(
            kb_get_by_title(&conn, "plan", &KbViewer::Agent(None))
                .unwrap()
                .unwrap()
                .body,
            "everyone's"
        );
    }

    #[test]
    fn test_set_agent_parent_refuses_loops() {
        let conn = test_db();
        agent_tree(&conn);
        assert!(set_agent_parent(&conn, "root", Some("leaf")).is_err());
        assert!(set_agent_parent(&conn, "root", Some("root")).is_err());
        assert!(set_agent_parent(&conn, "leaf", Some("nobody")).is_err());
        assert!(set_agent_parent(&conn, "nobody", Some("root")).is_err());
        // Moving a sub-agent elsewhere in the tree, or to the top, is fine.
        set_agent_parent(&conn, "leaf", Some("root")).unwrap();
        assert_eq!(agent_lineage(&conn, "leaf").unwrap(), ["leaf", "root"]);
        set_agent_parent(&conn, "leaf", None).unwrap();
        assert_eq!(agent_lineage(&conn, "leaf").unwrap(), ["leaf"]);
    }

    fn due_task_after(conn: &Connection, name: &str, depends_on: Vec<i64>) -> i64 {
        create_task(
            conn,
            &NewTask {
                name: name.to_string(),
                description: String::new(),
                kind: TaskKind::PROMPT.to_string(),
                command: format!("do {}", name),
                agent_name: None,
                schedule: TaskSchedule::Once {
                    at: (chrono::Utc::now() - chrono::Duration::seconds(5)).to_rfc3339(),
                },
                held: false,
                depends_on,
                profile: None,
                run_safe: false,
                cwd: None,
            },
        )
        .unwrap()
    }

    fn pending_names(conn: &Connection) -> Vec<String> {
        get_pending_tasks(conn)
            .unwrap()
            .into_iter()
            .map(|t| t.name)
            .collect()
    }

    fn run_task(conn: &Connection, id: i64, succeeded: bool) {
        assert!(
            claim_task(conn, id, "s").unwrap(),
            "task {id} should be claimable"
        );
        let outcome = TaskOutcome {
            succeeded,
            exit_code: None,
            result: String::new(),
        };
        assert!(finish_task(conn, id, "s", &outcome).unwrap());
    }

    #[test]
    fn test_dependent_tasks_wait_for_their_dependencies() {
        let conn = test_db();
        live_session(&conn, "s");
        let a = due_task_after(&conn, "a", vec![]);
        let b = due_task_after(&conn, "b", vec![a]);
        let c = due_task_after(&conn, "c", vec![b]);
        assert_eq!(get_task(&conn, c).unwrap().unwrap().depends_on, vec![b]);

        assert_eq!(pending_names(&conn), ["a"]);
        assert!(
            !claim_task(&conn, b, "s").unwrap(),
            "the claim checks dependencies too"
        );
        run_task(&conn, a, true);
        assert_eq!(pending_names(&conn), ["b"]);
        run_task(&conn, b, true);
        assert_eq!(pending_names(&conn), ["c"]);
        assert!(
            fail_tasks_with_failed_dependencies(&conn)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_a_failed_dependency_fails_the_whole_chain() {
        let conn = test_db();
        live_session(&conn, "s");
        let a = due_task_after(&conn, "a", vec![]);
        let b = due_task_after(&conn, "b", vec![a]);
        let c = due_task_after(&conn, "c", vec![b]);
        run_task(&conn, a, false);
        assert_eq!(fail_tasks_with_failed_dependencies(&conn).unwrap(), [b]);
        assert_eq!(fail_tasks_with_failed_dependencies(&conn).unwrap(), [c]);
        let c = get_task(&conn, c).unwrap().unwrap();
        assert_eq!(
            (c.status.as_str(), c.last_outcome.as_deref()),
            ("done", Some("failed"))
        );
        assert_eq!(
            c.last_result.as_deref(),
            Some(format!("dependency #{} failed", b).as_str())
        );
        assert!(pending_names(&conn).is_empty());
    }

    #[test]
    fn test_a_deleted_or_unknown_dependency() {
        let conn = test_db();
        let a = due_task_after(&conn, "a", vec![]);
        let b = due_task_after(&conn, "b", vec![a]);
        delete_task(&conn, a).unwrap();
        assert_eq!(fail_tasks_with_failed_dependencies(&conn).unwrap(), [b]);
        assert_eq!(
            get_task(&conn, b).unwrap().unwrap().last_result.as_deref(),
            Some(format!("dependency #{} no longer exists", a).as_str())
        );
        let err = create_task(
            &conn,
            &NewTask {
                name: "x".to_string(),
                description: String::new(),
                kind: TaskKind::PROMPT.to_string(),
                command: "x".to_string(),
                agent_name: None,
                schedule: TaskSchedule::Once {
                    at: chrono::Utc::now().to_rfc3339(),
                },
                held: false,
                depends_on: vec![999],
                profile: None,
                run_safe: false,
                cwd: None,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("no task #999"), "{err}");
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
    fn test_agent_config_round_trip() {
        let conn = test_db();
        create_agent(&conn, "alice", "").unwrap();
        let mut parameters = serde_json::Map::new();
        parameters.insert("reasoning_effort".to_string(), "low".into());
        parameters.insert("temperature".to_string(), 0.2.into());
        let config = AgentConfig {
            model: Some("qwen".to_string()),
            endpoint: Some("http://gpu:8080/v1".to_string()),
            system_prompt: Some("be brief".to_string()),
            api_key: Some("/keys/k".to_string()),
            max_tokens: Some(4096),
            context_window: Some(32768),
            parameters: Some(parameters),
            tools: Some(vec!["read_file".to_string()]),
            profile: Some("fast".to_string()),
            unsafe_tools: Some(true),
            cwd: Some("/work".to_string()),
        };
        set_agent_config(&conn, "alice", &config).unwrap();
        assert_eq!(get_agent_config(&conn, "alice").unwrap(), config);
        // Replacing it clears what the new one doesn't set.
        let smaller = AgentConfig {
            model: Some("other".to_string()),
            ..Default::default()
        };
        set_agent_config(&conn, "alice", &smaller).unwrap();
        assert_eq!(get_agent_config(&conn, "alice").unwrap(), smaller);
        // A value that doesn't parse is ignored rather than failing.
        set_agent_data(&conn, "alice", "config:max_tokens", "lots").unwrap();
        assert_eq!(get_agent_config(&conn, "alice").unwrap().max_tokens, None);
    }

    #[test]
    fn test_task_target_agent_profile_or_any() {
        let conn = test_db();
        create_agent(&conn, "bob", "").unwrap();
        let at = chrono::Utc::now().to_rfc3339();
        let task = |kind: &str, agent: Option<&str>, profile: Option<&str>| NewTask {
            name: "t".to_string(),
            description: String::new(),
            kind: kind.to_string(),
            command: "do it".to_string(),
            agent_name: agent.map(String::from),
            schedule: TaskSchedule::Once { at: at.clone() },
            held: false,
            depends_on: Vec::new(),
            profile: profile.map(String::from),
            run_safe: false,
            cwd: None,
        };
        let id = create_task(&conn, &task(TaskKind::PROMPT, None, Some("fast"))).unwrap();
        assert_eq!(
            get_task(&conn, id).unwrap().unwrap().profile.as_deref(),
            Some("fast")
        );
        assert!(create_task(&conn, &task(TaskKind::PROMPT, Some("bob"), Some("fast"))).is_err());
        assert!(create_task(&conn, &task(TaskKind::TOOL, None, Some("fast"))).is_err());
        // And the table itself refuses both.
        assert!(
            conn.execute(
                "UPDATE scheduled_tasks SET agent_name = 'bob' WHERE id = ?1",
                params![id]
            )
            .is_err()
        );

        assert!(set_task_target(&conn, id, Some("bob"), None).unwrap());
        let row = get_task(&conn, id).unwrap().unwrap();
        assert_eq!(
            (row.agent_name.as_deref(), row.profile),
            (Some("bob"), None)
        );
        assert!(set_task_target(&conn, id, None, None).unwrap());
        assert_eq!(get_task(&conn, id).unwrap().unwrap().agent_name, None);
        assert!(set_task_target(&conn, id, Some("nobody"), None).is_err());
        assert!(set_task_target(&conn, id, Some("bob"), Some("fast")).is_err());
        assert!(!set_task_target(&conn, 999, None, None).unwrap());
        let tool = create_task(&conn, &task(TaskKind::TOOL, None, None)).unwrap();
        assert!(set_task_target(&conn, tool, None, Some("fast")).is_err());
        // A running task has been picked up already.
        assert!(claim_task(&conn, id, "s").unwrap());
        assert!(!set_task_target(&conn, id, None, Some("fast")).unwrap());
        let outcome = TaskOutcome {
            succeeded: true,
            exit_code: None,
            result: String::new(),
        };
        assert!(finish_task(&conn, id, "s", &outcome).unwrap());
        assert!(!set_task_target(&conn, id, None, None).unwrap(), "done");
    }

    #[test]
    fn test_task_table_rebuilt_to_keep_agent_and_profile_apart() {
        let conn = Connection::open_in_memory().unwrap();
        initialize_db(&conn).unwrap();
        // The table as it was before profiles: no column, no constraint.
        conn.execute_batch(&format!(
            "PRAGMA foreign_keys = OFF;
             DROP TABLE scheduled_tasks;
             CREATE TABLE scheduled_tasks ({});
             PRAGMA foreign_keys = ON;",
            TASK_TABLE_DEFINITION
                .replace("    profile TEXT DEFAULT NULL,\n", "")
                .replace("    CHECK(agent_name IS NULL OR profile IS NULL),\n", "")
        ))
        .unwrap();
        assert!(!table_has_column(&conn, "scheduled_tasks", "profile").unwrap());
        create_agent(&conn, "bob", "").unwrap();
        let at = chrono::Utc::now().to_rfc3339();
        let keep = create_oneshot_task(&conn, "keep", "", &at, "say hi", Some("bob")).unwrap();

        initialize_db(&conn).unwrap();

        let task = get_task(&conn, keep).unwrap().unwrap();
        assert_eq!(
            (task.agent_name.as_deref(), task.profile),
            (Some("bob"), None)
        );
        assert!(
            conn.execute(
                "UPDATE scheduled_tasks SET profile = 'fast' WHERE id = ?1",
                params![keep]
            )
            .is_err(),
            "the constraint is in place"
        );
        assert!(set_task_target(&conn, keep, None, Some("fast")).unwrap());
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
