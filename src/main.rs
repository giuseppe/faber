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

mod dummy_llm;
mod github;
mod openai;
mod remote_db;
mod server;
mod status_bar;
mod summarize;

use faber::db;

use clap::{Parser, Subcommand};
use console::Style;
use env_logger::Env;
use log::{debug, trace, warn};
use pathrs::{Root, flags::OpenFlags};
use prettytable::{Cell, Row, Table, format};
use rustyline::Editor;
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::{History, MemHistory, SearchDirection, SearchResult};
use rustyline::validate::Validator;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::fs::Permissions;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use github::{
    get_github_issue, get_github_issue_comments, get_github_issues, get_github_pull_request,
    get_github_pull_request_patch, get_github_pull_requests,
};
use openai::{
    FunctionCall, InterruptedError, Message, OpenAIResponse, ProgressInfo, ResponseMode,
    StatusUpdate, ToolCall, ToolCallback, ToolItem, ToolsCollection, list_models_from_endpoint,
    make_message, post_request, post_request_with_mode, tool_call,
};
use std::collections::HashMap;

struct AgentState {
    name: String,
    messages: Vec<Message>,
}

struct SubAgentContext {
    tools: Arc<ToolsCollection>,
    opts: openai::Opts,
    session_id: String,
    active_subagents: Arc<std::sync::atomic::AtomicUsize>,
    status_bar: Arc<status_bar::StatusBar>,
}

const CHAT_COMMANDS: &[&str] = &[
    "/help",
    "/quit",
    "/clear",
    "/show",
    "/limit",
    "/backtrace",
    "/summarize",
    "/system",
    "/agents",
    "/create-agent",
    "/select-agent",
    "/delete-agent",
    "/mcp-refresh",
    "/tools",
];

struct ChatHelper {
    agent_names: Arc<Mutex<Vec<String>>>,
}

impl Completer for ChatHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        if !line.starts_with('/') {
            return Ok((0, vec![]));
        }

        let input = &line[..pos];

        if input.starts_with("/select-agent ") || input.starts_with("/delete-agent ") {
            let prefix_end = input.find(' ').unwrap() + 1;
            let name_prefix = &input[prefix_end..];
            let mut candidates = Vec::new();
            if let Ok(names) = self.agent_names.lock() {
                for name in names.iter() {
                    if name.starts_with(name_prefix) {
                        candidates.push(Pair {
                            display: name.clone(),
                            replacement: name.clone(),
                        });
                    }
                }
            }
            return Ok((prefix_end, candidates));
        }

        let mut candidates = Vec::new();
        for cmd in CHAT_COMMANDS {
            if cmd.starts_with(input) {
                candidates.push(Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                });
            }
        }
        Ok((0, candidates))
    }
}

impl Hinter for ChatHelper {
    type Hint = String;
}
impl Highlighter for ChatHelper {}
impl Validator for ChatHelper {}
impl rustyline::Helper for ChatHelper {}

struct DbHistory {
    conn: Option<Arc<Mutex<rusqlite::Connection>>>,
    mem: MemHistory,
    max_len: usize,
    ignore_space: bool,
    ignore_dups: bool,
    row_id: usize,
}

impl DbHistory {
    fn new(conn: Option<Arc<Mutex<rusqlite::Connection>>>) -> Self {
        let row_id = if let Some(ref c) = conn {
            if let Ok(c) = c.lock() {
                c.query_row(
                    "SELECT COALESCE(MAX(id), 0) FROM readline_history",
                    [],
                    |r| r.get::<_, usize>(0),
                )
                .unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };
        Self {
            conn,
            mem: MemHistory::new(),
            max_len: 1000,
            ignore_space: false,
            ignore_dups: true,
            row_id,
        }
    }

    fn ignore(&self, line: &str) -> bool {
        if self.max_len == 0 {
            return true;
        }
        if line.is_empty() || (self.ignore_space && line.starts_with(' ')) {
            return true;
        }
        false
    }

    fn is_dup(&self, line: &str) -> bool {
        if !self.ignore_dups {
            return false;
        }
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let last: Option<String> = c
                    .query_row(
                        "SELECT entry FROM readline_history ORDER BY id DESC LIMIT 1",
                        [],
                        |r| r.get(0),
                    )
                    .ok();
                return last.as_deref() == Some(line);
            }
        }
        false
    }

    fn enforce_max_len(&self) {
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let _ = c.execute(
                    "DELETE FROM readline_history WHERE id IN (
                        SELECT id FROM readline_history ORDER BY id ASC
                        LIMIT MAX(0, (SELECT COUNT(*) FROM readline_history) - ?1)
                    )",
                    rusqlite::params![self.max_len],
                );
            }
        }
    }
}

impl History for DbHistory {
    fn get(
        &self,
        index: usize,
        dir: SearchDirection,
    ) -> rustyline::Result<Option<SearchResult<'_>>> {
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let rowid = index + 1;
                let (query, param) = match dir {
                    SearchDirection::Forward => (
                        "SELECT id, entry FROM readline_history WHERE id >= ?1 ORDER BY id ASC LIMIT 1",
                        rowid,
                    ),
                    SearchDirection::Reverse => (
                        "SELECT id, entry FROM readline_history WHERE id <= ?1 ORDER BY id DESC LIMIT 1",
                        rowid,
                    ),
                };
                let result = c.query_row(query, rusqlite::params![param], |r| {
                    let id: usize = r.get(0)?;
                    let entry: String = r.get(1)?;
                    Ok((id, entry))
                });
                return match result {
                    Ok((id, entry)) => Ok(Some(SearchResult {
                        entry: std::borrow::Cow::Owned(entry),
                        idx: id - 1,
                        pos: 0,
                    })),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(rustyline::error::ReadlineError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e,
                    ))),
                };
            }
        }
        self.mem.get(index, dir)
    }

    fn add(&mut self, line: &str) -> rustyline::Result<bool> {
        if self.ignore(line) {
            return Ok(false);
        }
        if let Some(ref c) = self.conn {
            if self.is_dup(line) {
                return Ok(false);
            }
            if let Ok(c) = c.lock() {
                match c.execute(
                    "INSERT INTO readline_history (entry) VALUES (?1)",
                    rusqlite::params![line],
                ) {
                    Ok(_) => {
                        self.row_id = c.last_insert_rowid() as usize;
                        drop(c);
                        self.enforce_max_len();
                        return Ok(true);
                    }
                    Err(e) => {
                        warn!("Failed to save history entry: {}", e);
                    }
                }
            }
            return Ok(false);
        }
        self.mem.add(line)
    }

    fn add_owned(&mut self, line: String) -> rustyline::Result<bool> {
        self.add(&line)
    }

    fn len(&self) -> usize {
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                return c
                    .query_row("SELECT COUNT(*) FROM readline_history", [], |r| {
                        r.get::<_, usize>(0)
                    })
                    .unwrap_or(self.row_id);
            }
        }
        self.mem.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn set_max_len(&mut self, len: usize) -> rustyline::Result<()> {
        self.max_len = len;
        self.enforce_max_len();
        self.mem.set_max_len(len)
    }

    fn ignore_dups(&mut self, yes: bool) -> rustyline::Result<()> {
        self.ignore_dups = yes;
        self.mem.ignore_dups(yes)
    }

    fn ignore_space(&mut self, yes: bool) {
        self.ignore_space = yes;
        self.mem.ignore_space(yes);
    }

    fn save(&mut self, _path: &std::path::Path) -> rustyline::Result<()> {
        Ok(())
    }

    fn append(&mut self, _path: &std::path::Path) -> rustyline::Result<()> {
        Ok(())
    }

    fn load(&mut self, _path: &std::path::Path) -> rustyline::Result<()> {
        Ok(())
    }

    fn clear(&mut self) -> rustyline::Result<()> {
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let _ = c.execute("DELETE FROM readline_history", []);
                self.row_id = 0;
                return Ok(());
            }
        }
        self.mem.clear()
    }

    fn search(
        &self,
        term: &str,
        start: usize,
        dir: SearchDirection,
    ) -> rustyline::Result<Option<SearchResult<'_>>> {
        if term.is_empty() || start >= self.len() {
            return Ok(None);
        }
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let rowid = start + 1;
                let escaped = term
                    .replace('\\', "\\\\")
                    .replace('%', "\\%")
                    .replace('_', "\\_");
                let pattern = format!("%{}%", escaped);
                let (query, param) = match dir {
                    SearchDirection::Forward => (
                        "SELECT id, entry FROM readline_history WHERE entry LIKE ?1 ESCAPE '\\' AND id >= ?2 ORDER BY id ASC LIMIT 1",
                        rowid,
                    ),
                    SearchDirection::Reverse => (
                        "SELECT id, entry FROM readline_history WHERE entry LIKE ?1 ESCAPE '\\' AND id <= ?2 ORDER BY id DESC LIMIT 1",
                        rowid,
                    ),
                };
                let result = c.query_row(query, rusqlite::params![pattern, param], |r| {
                    let id: usize = r.get(0)?;
                    let entry: String = r.get(1)?;
                    Ok((id, entry))
                });
                return match result {
                    Ok((id, entry)) => {
                        let pos = entry.find(term).unwrap_or(0);
                        Ok(Some(SearchResult {
                            entry: std::borrow::Cow::Owned(entry),
                            idx: id - 1,
                            pos,
                        }))
                    }
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(rustyline::error::ReadlineError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e,
                    ))),
                };
            }
        }
        self.mem.search(term, start, dir)
    }

    fn starts_with(
        &self,
        term: &str,
        start: usize,
        dir: SearchDirection,
    ) -> rustyline::Result<Option<SearchResult<'_>>> {
        if term.is_empty() || start >= self.len() {
            return Ok(None);
        }
        if let Some(ref c) = self.conn {
            if let Ok(c) = c.lock() {
                let rowid = start + 1;
                let escaped = term
                    .replace('\\', "\\\\")
                    .replace('%', "\\%")
                    .replace('_', "\\_");
                let pattern = format!("{}%", escaped);
                let (query, param) = match dir {
                    SearchDirection::Forward => (
                        "SELECT id, entry FROM readline_history WHERE entry LIKE ?1 ESCAPE '\\' AND id >= ?2 ORDER BY id ASC LIMIT 1",
                        rowid,
                    ),
                    SearchDirection::Reverse => (
                        "SELECT id, entry FROM readline_history WHERE entry LIKE ?1 ESCAPE '\\' AND id <= ?2 ORDER BY id DESC LIMIT 1",
                        rowid,
                    ),
                };
                let result = c.query_row(query, rusqlite::params![pattern, param], |r| {
                    let id: usize = r.get(0)?;
                    let entry: String = r.get(1)?;
                    Ok((id, entry))
                });
                return match result {
                    Ok((id, entry)) => Ok(Some(SearchResult {
                        entry: std::borrow::Cow::Owned(entry),
                        idx: id - 1,
                        pos: term.len(),
                    })),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(rustyline::error::ReadlineError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e,
                    ))),
                };
            }
        }
        self.mem.starts_with(term, start, dir)
    }
}

#[derive(Clone)]
struct ChatPrinter {
    printer: Arc<Mutex<Option<Box<dyn rustyline::ExternalPrinter + Send>>>>,
    direct_mode: Arc<AtomicBool>,
}

const AGENT_COLORS: &[console::Color] = &[
    console::Color::Cyan,
    console::Color::Green,
    console::Color::Yellow,
    console::Color::Magenta,
    console::Color::Blue,
    console::Color::Red,
];

const AGENT_ANSI_CODES: &[u8] = &[36, 32, 33, 35, 34, 31];

fn agent_color_hash(name: &str) -> usize {
    name.bytes()
        .fold(0usize, |acc, b| acc.wrapping_add(b as usize))
}

fn agent_style(name: &str) -> Style {
    Style::new()
        .fg(AGENT_COLORS[agent_color_hash(name) % AGENT_COLORS.len()])
        .bold()
}

fn agent_ansi_code(name: &str) -> u8 {
    AGENT_ANSI_CODES[agent_color_hash(name) % AGENT_ANSI_CODES.len()]
}

impl ChatPrinter {
    fn new() -> Self {
        Self {
            printer: Arc::new(Mutex::new(None)),
            direct_mode: Arc::new(AtomicBool::new(true)),
        }
    }

    fn set_printer(&self, p: Box<dyn rustyline::ExternalPrinter + Send>) {
        if let Ok(mut guard) = self.printer.lock() {
            *guard = Some(p);
        }
    }

    fn set_direct_mode(&self, direct: bool) {
        self.direct_mode.store(direct, Ordering::Relaxed);
    }

    fn println(&self, msg: &str) {
        if !self.direct_mode.load(Ordering::Relaxed) {
            if let Ok(mut guard) = self.printer.lock() {
                if let Some(ref mut p) = *guard {
                    let _ = p.print(format!("{}\n", msg));
                    return;
                }
            }
        }
        status_bar::write_stderr(&format!("{}\n", msg));
    }

    fn println_agent(&self, agent_name: &str, msg: &str) {
        let style = agent_style(agent_name);
        let header = style.apply_to(format!("── {} ──", agent_name));
        self.println(&format!("{}", header));
        for line in msg.lines() {
            self.println(&format!("  {}", line));
        }
        self.println("");
    }
}

const DEFAULT_ENDPOINT: &str = "http://localhost:8080";
const DEFAULT_MODEL: &str = "google/gemini-2.5-pro";

use faber::ToolContext;
use faber::db_backend::DbBackend;
use faber::local_db::LocalDb;

/// Parse parameter strings in NAME=VALUE format into a HashMap
fn parse_parameters(
    param_strings: &[String],
) -> Result<HashMap<String, serde_json::Value>, Box<dyn Error>> {
    let mut parameters = HashMap::new();

    for param in param_strings {
        if let Some((key, value)) = param.split_once('=') {
            let key = key.trim().to_string();
            let value_str = value.trim();

            // Try to parse as different types
            let json_value = if let Ok(num) = value_str.parse::<f64>() {
                if num.is_finite() {
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(num)
                            .unwrap_or_else(|| serde_json::Number::from(0)),
                    )
                } else {
                    serde_json::Value::String(value_str.to_string())
                }
            } else if let Ok(bool_val) = value_str.parse::<bool>() {
                serde_json::Value::Bool(bool_val)
            } else if value_str == "null" {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(value_str.to_string())
            };

            debug!("Parsed parameter: {} = {:?}", key, json_value);
            parameters.insert(key, json_value);
        } else {
            return Err(
                format!("Invalid parameter format: '{}'. Expected NAME=VALUE", param).into(),
            );
        }
    }

    if !parameters.is_empty() {
        debug!(
            "Using {} custom parameters: {:?}",
            parameters.len(),
            parameters
        );
    }

    Ok(parameters)
}

fn append_tool(tools: &mut ToolsCollection, name: String, callback: ToolCallback, schema: String) {
    let item = ToolItem {
        callback: callback,
        schema: schema,
    };
    debug!("Adding tool: {}", name);
    tools.insert(name, item);
}

/// entrypoint for the delete_path tool
fn tool_delete_path(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        path: String,
    }
    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Remove path: {}", params.path);
    let root = Root::open(".")?;

    let path = PathBuf::from(&params.path);

    root.remove_all(path)?;
    Ok("deleted".to_owned())
}

/// entrypoint for the read_file tool
fn tool_read_file(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        path: String,
    }

    #[derive(Serialize)]
    struct ReadFileResult {
        content: Option<String>,
        error: Option<String>,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Reading file: {}", params.path);

    let root = Root::open(".")?;
    let path = PathBuf::from(&params.path);
    let file = root.open_subpath(path, OpenFlags::O_RDONLY);

    let result = match file {
        Ok(mut file) => {
            let mut contents = String::new();
            match file.read_to_string(&mut contents) {
                Ok(_) => ReadFileResult {
                    content: Some(contents),
                    error: None,
                },
                Err(e) => ReadFileResult {
                    content: None,
                    error: Some(format!("Failed to read file: {}", e)),
                },
            }
        }
        Err(e) => ReadFileResult {
            content: None,
            error: Some(format!("File not found or cannot be opened: {}", e)),
        },
    };

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// Show diff between old and new content using system diff tool with file descriptors
fn show_diff(ctx: &ToolContext, old_content: &str, new_content: &str, file_path: &str) {
    use std::io::Write;
    use std::os::unix::io::{AsRawFd, FromRawFd};
    use std::process::Command;

    // Create temporary files using O_TMPFILE in current directory
    // Use rustix crate for system calls since it's available through pathrs
    let old_fd_result = rustix::fs::openat(
        rustix::fs::CWD,
        ".",
        rustix::fs::OFlags::TMPFILE | rustix::fs::OFlags::RDWR,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    );

    let new_fd_result = rustix::fs::openat(
        rustix::fs::CWD,
        ".",
        rustix::fs::OFlags::TMPFILE | rustix::fs::OFlags::RDWR,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    );

    let (old_fd, new_fd) = match (old_fd_result, new_fd_result) {
        (Ok(old), Ok(new)) => (old, new),
        _ => {
            debug!("Failed to create temporary file descriptors");
            return;
        }
    };

    let write_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Seek;
        use std::mem::ManuallyDrop;

        let mut old_file =
            ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(old_fd.as_raw_fd()) });
        let mut new_file =
            ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(new_fd.as_raw_fd()) });

        old_file.write_all(old_content.as_bytes())?;
        new_file.write_all(new_content.as_bytes())?;
        old_file.seek(std::io::SeekFrom::Start(0))?;
        new_file.seek(std::io::SeekFrom::Start(0))?;

        Ok(())
    })();

    if let Err(e) = write_result {
        debug!("Failed to write to temporary file descriptors: {}", e);
        return;
    }

    // Run diff command using /proc/self/fd/ paths
    let old_path = format!("/proc/self/fd/{}", old_fd.as_raw_fd());
    let new_path = format!("/proc/self/fd/{}", new_fd.as_raw_fd());

    let diff_result = Command::new("diff")
        .arg("--color=always")
        .arg("-Naur")
        .arg("--label")
        .arg(&format!("a/{}", file_path))
        .arg("--label")
        .arg(&format!("b/{}", file_path))
        .arg(&old_path)
        .arg(&new_path)
        .output();

    // File descriptors will be automatically closed when old_fd and new_fd go out of scope

    match diff_result {
        Ok(output) => {
            // diff returns 0 for no differences, 1 for differences, >1 for errors
            if output.status.code() == Some(0) {
                ctx.println("   No differences detected");
            } else if output.status.code() == Some(1) {
                ctx.println("   Changes:");
                let diff_output = String::from_utf8_lossy(&output.stdout);
                for line in diff_output.lines() {
                    ctx.println(&format!("   {}", line));
                }
            } else {
                debug!("diff command failed with status: {:?}", output.status);
            }
        }
        Err(e) => {
            debug!("Failed to run diff command: {}", e);
        }
    }
}

/// entrypoint for the write_file tool
fn tool_write_file(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        path: String,
        content: String,
        #[serde(default = "default_file_mode")]
        mode: String,
        #[serde(default)]
        offset: Option<u64>,
        #[serde(default)]
        length: Option<u64>,
        #[serde(default)]
        start_line: Option<u64>,
        #[serde(default)]
        end_line: Option<u64>,
        #[serde(default)]
        old_content: Option<String>,
        #[serde(default)]
        replace_all: Option<bool>,
    }

    #[derive(Serialize)]
    struct WriteFileResult {
        path: String,
        bytes_written: usize,
        mode: String,
        created: bool,
        operation: String,
        message: String,
    }

    fn default_file_mode() -> String {
        "0644".to_string()
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "write_file received params: path='{}', content_length={}, mode='{}'",
        params.path,
        params.content.len(),
        params.mode
    );

    let file_mode = if params.mode.starts_with("0o") {
        u32::from_str_radix(&params.mode[2..], 8)
    } else if params.mode.starts_with("0x") {
        u32::from_str_radix(&params.mode[2..], 16)
    } else if params.mode.starts_with("0") && params.mode.len() > 1 {
        u32::from_str_radix(&params.mode[1..], 8)
    } else {
        params.mode.parse::<u32>()
    }
    .map_err(|_| {
        format!(
            "Invalid file mode format: '{}'. Expected octal (0644, 0o644), hex (0x1a4), or decimal",
            params.mode
        )
    })?;

    debug!("Parsed file mode: {} (octal: {:o})", file_mode, file_mode);

    let has_offset = params.offset.is_some();
    let has_start_line = params.start_line.is_some();
    let has_old_content = params.old_content.is_some();

    if has_old_content && (has_offset || has_start_line) {
        return Err("Cannot combine old_content with offset or start_line parameters".into());
    }
    if has_offset && has_start_line {
        return Err("Cannot combine offset with start_line parameters".into());
    }
    if has_offset && params.length.is_none() {
        return Err("offset requires length to be specified".into());
    }
    if has_start_line && params.end_line.is_none() {
        return Err("start_line requires end_line to be specified".into());
    }

    let is_partial = has_offset || has_start_line || has_old_content;

    let root = Root::open(".")?;
    let path_buf = PathBuf::from(&params.path);

    if is_partial {
        let mut file = root
            .open_subpath(&params.path, OpenFlags::O_RDONLY)
            .map_err(|_| format!("File '{}' must exist for partial writes", params.path))?;
        let mut existing_bytes = Vec::new();
        file.read_to_end(&mut existing_bytes)?;
        drop(file);

        let (new_bytes, operation) = if has_old_content {
            let old_text = params.old_content.as_ref().unwrap();
            let existing_str = String::from_utf8(existing_bytes.clone())
                .map_err(|_| "File contains invalid UTF-8, cannot use search-and-replace mode")?;

            if old_text == &params.content {
                return Err("old_content and content are identical, nothing to replace".into());
            }

            let match_count = existing_str.matches(old_text.as_str()).count();
            if match_count == 0 {
                return Err(format!("old_content not found in file '{}'", params.path).into());
            }

            let replace_all = params.replace_all.unwrap_or(false);
            if match_count > 1 && !replace_all {
                return Err(format!(
                    "old_content matches {} times in '{}'. Set replace_all=true or provide more context to make it unique",
                    match_count, params.path
                ).into());
            }

            let new_str = if replace_all {
                existing_str.replace(old_text.as_str(), &params.content)
            } else {
                existing_str.replacen(old_text.as_str(), &params.content, 1)
            };
            (new_str.into_bytes(), "patch_search_replace")
        } else if has_offset {
            let offset = params.offset.unwrap() as usize;
            let length = params.length.unwrap() as usize;

            if offset > existing_bytes.len() {
                return Err(format!(
                    "offset {} is beyond file size {}",
                    offset,
                    existing_bytes.len()
                )
                .into());
            }
            if offset + length > existing_bytes.len() {
                return Err(format!(
                    "offset ({}) + length ({}) = {} exceeds file size {}",
                    offset,
                    length,
                    offset + length,
                    existing_bytes.len()
                )
                .into());
            }

            let mut new_bytes =
                Vec::with_capacity(existing_bytes.len() - length + params.content.len());
            new_bytes.extend_from_slice(&existing_bytes[..offset]);
            new_bytes.extend_from_slice(params.content.as_bytes());
            new_bytes.extend_from_slice(&existing_bytes[offset + length..]);
            (new_bytes, "patch_offset")
        } else {
            let existing_str = String::from_utf8(existing_bytes.clone())
                .map_err(|_| "File contains invalid UTF-8, cannot use line-number mode")?;

            let lines: Vec<&str> = existing_str.lines().collect();
            let start = params.start_line.unwrap() as usize;
            let end = params.end_line.unwrap() as usize;

            if start == 0 {
                return Err("start_line is 1-based, cannot be 0".into());
            }
            if end < start {
                return Err(format!("end_line ({}) must be >= start_line ({})", end, start).into());
            }
            if start > lines.len() {
                return Err(format!(
                    "start_line {} exceeds file line count {}",
                    start,
                    lines.len()
                )
                .into());
            }
            if end > lines.len() {
                return Err(
                    format!("end_line {} exceeds file line count {}", end, lines.len()).into(),
                );
            }

            let mut new_lines: Vec<&str> = Vec::new();
            new_lines.extend_from_slice(&lines[..start - 1]);
            for line in params.content.lines() {
                new_lines.push(line);
            }
            new_lines.extend_from_slice(&lines[end..]);

            let mut new_str = new_lines.join("\n");
            if existing_str.ends_with('\n') {
                new_str.push('\n');
            }
            (new_str.into_bytes(), "patch_lines")
        };

        let old_content_for_diff = String::from_utf8(existing_bytes).ok();

        let mut file = root.open_subpath(&params.path, OpenFlags::O_WRONLY | OpenFlags::O_TRUNC)?;
        {
            use std::os::unix::io::AsRawFd;
            let mode = rustix::fs::Mode::from_raw_mode(file_mode);
            let _ = rustix::fs::fchmod(
                unsafe { rustix::fd::BorrowedFd::borrow_raw(file.as_raw_fd()) },
                mode,
            );
        }
        file.write_all(&new_bytes)?;

        let bytes_written = new_bytes.len();
        let op_display = match operation {
            "patch_offset" => "PATCH (byte offset)",
            "patch_lines" => "PATCH (line range)",
            _ => "PATCH (search & replace)",
        };

        let result = WriteFileResult {
            path: params.path.clone(),
            bytes_written,
            mode: format!("{:o}", file_mode),
            created: false,
            operation: operation.to_string(),
            message: format!("File '{}' patched successfully", params.path),
        };

        ctx.println(&format!("\u{1f4dd} {}", result.message));
        ctx.println(&format!("   Path: {}", result.path));
        ctx.println(&format!("   Bytes written: {}", result.bytes_written));
        ctx.println(&format!("   File mode: {}", result.mode));
        ctx.println(&format!("   Operation: {}", op_display));

        if let Some(old_str) = old_content_for_diff {
            if let Ok(new_str) = std::str::from_utf8(&new_bytes) {
                show_diff(ctx, &old_str, new_str, &params.path);
            }
        }

        let json_result = serde_json::to_string(&result)?;
        return Ok(json_result);
    }

    // Full write mode (existing behavior)
    let existing_content = match root.open_subpath(&params.path, OpenFlags::O_RDONLY) {
        Ok(mut file) => {
            let mut content = Vec::new();
            match file.read_to_end(&mut content) {
                Ok(_) => Some(content),
                Err(_) => None,
            }
        }
        Err(_) => None,
    };

    let (created, mut file) =
        match root.open_subpath(&params.path, OpenFlags::O_WRONLY | OpenFlags::O_TRUNC) {
            Ok(file) => {
                debug!("File '{}' already exists, will overwrite", params.path);
                use std::os::unix::io::AsRawFd;
                let mode = rustix::fs::Mode::from_raw_mode(file_mode);
                let _ = rustix::fs::fchmod(
                    unsafe { rustix::fd::BorrowedFd::borrow_raw(file.as_raw_fd()) },
                    mode,
                );
                (false, file)
            }
            Err(e) => {
                debug!("File '{}' does not exist ({}), will create", params.path, e);

                if let Some(parent) = path_buf.parent() {
                    if !parent.as_os_str().is_empty() {
                        debug!("Creating parent directories for: {}", parent.display());
                        root.mkdir_all(parent, &Permissions::from_mode(0o755))?;
                        debug!("Parent directories created successfully");
                    }
                }

                let permissions = Permissions::from_mode(file_mode);
                debug!("Using file permissions: {:o}", permissions.mode());

                let new_file = root.create_file(
                    &params.path,
                    OpenFlags::O_WRONLY | OpenFlags::O_CREAT,
                    &permissions,
                )?;
                (true, new_file)
            }
        };

    let bytes_written = params.content.len();
    debug!("Writing {} bytes to file: {}", bytes_written, params.path);

    file.write_all(params.content.as_bytes())?;

    let result = WriteFileResult {
        path: params.path.clone(),
        bytes_written,
        mode: format!("{:o}", file_mode),
        created,
        operation: if created {
            "create".to_string()
        } else {
            "overwrite".to_string()
        },
        message: if created {
            format!("File '{}' created successfully", params.path)
        } else {
            format!("File '{}' overwritten successfully", params.path)
        },
    };

    debug!(
        "write_file completed successfully: {} bytes written to '{}' with mode {:o}",
        bytes_written, params.path, file_mode
    );

    ctx.println(&format!("\u{1f4dd} {}", result.message));
    ctx.println(&format!("   Path: {}", result.path));
    ctx.println(&format!("   Bytes written: {}", result.bytes_written));
    ctx.println(&format!("   File mode: {}", result.mode));
    ctx.println(&format!(
        "   Operation: {}",
        if result.created {
            "CREATE"
        } else {
            "OVERWRITE"
        }
    ));

    if !created {
        if let Some(old_content_bytes) = existing_content {
            if std::str::from_utf8(params.content.as_bytes()).is_ok() {
                let old_content = String::from_utf8(old_content_bytes);
                match old_content {
                    Ok(old_content) => {
                        show_diff(ctx, &old_content, &params.content, &params.path);
                    }
                    Err(_) => {}
                }
            }
        }
    }

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// entrypoint for the glob tool
fn tool_glob(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Params {
        pattern: String,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;
    let glob_pattern = &params.pattern;

    // Security check: reject patterns that try to escape current directory
    if glob_pattern.contains("../") || glob_pattern.starts_with('/') {
        return Err("Pattern cannot access parent directories or absolute paths".into());
    }

    trace!("Executing glob pattern: {}", glob_pattern);

    let mut files = Vec::new();
    for entry in glob::glob(glob_pattern)? {
        match entry {
            Ok(path) => {
                // Additional security check: ensure resolved path is within current directory
                let canonical_path = path.canonicalize().unwrap_or(path.clone());
                let current_dir = std::env::current_dir()?;

                if canonical_path.starts_with(&current_dir) {
                    files.push(path.to_string_lossy().to_string());
                } else {
                    debug!("Skipping file outside current directory: {:?}", path);
                }
            }
            Err(e) => {
                debug!("Error processing glob entry: {}", e);
            }
        }
    }

    files.sort();
    let result = files.join("\n");
    debug!(
        "Successfully matched {} files with pattern '{}'",
        files.len(),
        glob_pattern
    );
    Ok(result)
}

/// entrypoint for the fetch_web_content tool
fn tool_fetch_web_content(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        url: String,
        #[serde(default)]
        timeout_seconds: Option<u64>,
        #[serde(default)]
        follow_redirects: Option<bool>,
        #[serde(default)]
        max_content_length: Option<usize>,
    }

    #[derive(Serialize)]
    struct WebContentResult {
        url: String,
        status_code: u16,
        content_type: Option<String>,
        content: String,
        content_length: usize,
        final_url: Option<String>,
        headers: std::collections::HashMap<String, String>,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Fetching web content from URL: {}", params.url);

    // Validate URL
    if !params.url.starts_with("http://") && !params.url.starts_with("https://") {
        return Err("URL must start with http:// or https://".into());
    }

    let timeout_seconds = params.timeout_seconds.unwrap_or(30);
    let follow_redirects = params.follow_redirects.unwrap_or(true);
    let max_content_length = params.max_content_length.unwrap_or(10_000_000);

    ctx.println(&format!("🌐 Fetching content from: {}", params.url));
    ctx.println(&format!("   Timeout: {}s", timeout_seconds));
    ctx.println(&format!("   Follow redirects: {}", follow_redirects));
    ctx.println(&format!(
        "   Max content length: {} bytes",
        max_content_length
    ));

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_seconds))
        .redirect(if follow_redirects {
            reqwest::redirect::Policy::limited(10)
        } else {
            reqwest::redirect::Policy::none()
        })
        .user_agent("faber/0.1.0")
        .build()?;

    let response = client.get(&params.url).send()?;

    let status_code = response.status().as_u16();
    let final_url = if response.url().as_str() != params.url {
        Some(response.url().to_string())
    } else {
        None
    };

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Collect headers
    let mut headers = std::collections::HashMap::new();
    for (key, value) in response.headers() {
        if let Ok(value_str) = value.to_str() {
            headers.insert(key.to_string(), value_str.to_string());
        }
    }

    let content = if response.status().is_success() {
        let content_bytes = response.bytes()?;
        if content_bytes.len() > max_content_length {
            return Err(format!(
                "Content too large: {} bytes (max: {})",
                content_bytes.len(),
                max_content_length
            )
            .into());
        }

        // Try to decode as UTF-8, fallback to lossy conversion
        match std::str::from_utf8(&content_bytes) {
            Ok(text) => text.to_string(),
            Err(_) => String::from_utf8_lossy(&content_bytes).to_string(),
        }
    } else {
        format!("HTTP Error: {}", response.status())
    };

    let content_length = content.len();

    let result = WebContentResult {
        url: params.url.clone(),
        status_code,
        content_type,
        content: content.clone(),
        content_length,
        final_url,
        headers,
    };

    // Display result using context callback
    ctx.println(&format!("📄 Content fetched successfully"));
    ctx.println(&format!("   Status: {}", status_code));
    if let Some(ref ct) = result.content_type {
        ctx.println(&format!("   Content-Type: {}", ct));
    }
    if let Some(ref final_url) = result.final_url {
        ctx.println(&format!("   Final URL: {}", final_url));
    }
    ctx.println(&format!("   Content length: {} bytes", content_length));

    // Show a preview of the content
    let preview_lines: Vec<&str> = content.lines().take(5).collect();
    if !preview_lines.is_empty() {
        ctx.println("   Content preview:");
        for line in preview_lines {
            let truncated = if line.len() > 100 {
                let mut end = 97;
                while end > 0 && !line.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}...", &line[..end])
            } else {
                line.to_string()
            };
            ctx.println(&format!("     {}", truncated));
        }
        if content.lines().count() > 5 {
            ctx.println("     ... (content truncated in preview)");
        }
    }

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// entrypoint for the run_command tool
fn tool_run_command(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        command: String,
        args: Option<Vec<String>>,
    }

    #[derive(Serialize)]
    struct CommandResult {
        stdout: String,
        stderr: String,
        exit_code: Option<i32>,
        success: bool,
    }

    debug!("run_command received params: {}", params_str);

    // Try normal parsing first, fallback to manual parsing if it fails
    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    let mut cmd = if params.args.is_some() || !params.command.contains(' ') {
        let mut c = Command::new(&params.command);
        if let Some(ref args) = params.args {
            c.args(args);
        }
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(&params.command);
        c
    };

    trace!("Executing command: {:?}", cmd);

    let result = match cmd.output() {
        Ok(output) => {
            let stdout = String::from_utf8(output.stdout)
                .unwrap_or_else(|_| "<non-utf8 output>".to_string());
            let stderr = String::from_utf8(output.stderr)
                .unwrap_or_else(|_| "<non-utf8 output>".to_string());
            let exit_code = output.status.code();
            let success = output.status.success();

            if success {
                debug!("Successfully run command {}", params.command);
            } else {
                debug!(
                    "Command {} failed with exit code {:?}",
                    params.command, exit_code
                );
            }

            // Display output directly using context callback
            if success {
                ctx.println("✅ Command executed successfully:");
            } else {
                ctx.println(&format!(
                    "❌ Command failed (exit code: {}):",
                    exit_code.unwrap_or(-1)
                ));
            }
            if !stdout.is_empty() {
                ctx.println("STDOUT:");
                for line in stdout.lines() {
                    ctx.println(&format!("  {}", line));
                }
            }
            if !stderr.is_empty() {
                ctx.println("STDERR:");
                for line in stderr.lines() {
                    ctx.println(&format!("  {}", line));
                }
            }
            if stdout.is_empty() && stderr.is_empty() {
                ctx.println("(no output)");
            }
            CommandResult {
                stdout,
                stderr,
                exit_code,
                success,
            }
        }
        Err(e) => {
            debug!("Failed to execute command {}: {}", params.command, e);
            ctx.println(&format!("ERROR: Failed to execute command: {}", e));
            CommandResult {
                stdout: String::new(),
                stderr: format!("Failed to execute command: {}", e),
                exit_code: None,
                success: false,
            }
        }
    };

    // Output is now displayed directly above during command execution

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// entrypoint for the grep_in_current_directory tool
fn tool_grep_in_current_directory(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        pattern: String,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    let mut cmd = Command::new("grep");
    cmd.arg("-r");
    cmd.arg("-n");
    cmd.arg("-e").arg(&params.pattern);

    debug!(
        "Grepping for pattern '{}' in current directory",
        params.pattern
    );

    trace!("Executing grep command: {:?}", cmd);
    let output = cmd.output()?;
    // Grep returns 1 if no lines were selected, 0 if lines were selected, >1 for errors.
    // We consider no lines selected as a valid, empty result, not an error for the tool.
    if !output.status.success() && output.status.code() != Some(1) {
        let stderr = String::from_utf8(output.stderr)?;
        let err_msg = format!(
            "grep command failed with status {:?}. Stderr: {}",
            output.status, stderr
        );
        let err: Box<dyn Error> = err_msg.into();
        return Err(err);
    }

    let r = String::from_utf8(output.stdout)?;
    debug!(
        "Grep command successfully executed for pattern '{}'",
        params.pattern
    );

    // Display grep results directly using context callback
    ctx.println(&format!(
        "🔍 Grep results for pattern '{}':",
        params.pattern
    ));

    if r.is_empty() {
        ctx.println("(no matches found)");
    } else {
        let line_count = r.lines().count();
        ctx.println(&format!("Found {} matches:", line_count));
        for line in r.lines().take(10) {
            // Show first 10 matches
            ctx.println(&format!("  {}", line));
        }
        if line_count > 10 {
            ctx.println(&format!("  ... and {} more matches", line_count - 10));
        }
    }

    Ok(r)
}

/// entrypoint for the github_issue tool
fn tool_github_issue(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        issue: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Fetching GitHub issue: {}/{}", params.repo, params.issue);
    let issue = get_github_issue(&params.repo, params.issue)?;

    let s = serde_json::to_string(&issue)?;
    Ok(s)
}

/// entrypoint for the github_issue_comments tool
fn tool_github_issue_comments(
    params_str: &String,
    _ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        issue: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub issue comments: {}/{}",
        params.repo, params.issue
    );
    let comments = get_github_issue_comments(&params.repo, params.issue)?;
    let s = serde_json::to_string(&comments)?;
    Ok(s)
}

/// entrypoint for the github_issues tool
fn tool_github_issues(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        days: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub issues from repository {} for the last {} days",
        params.repo, params.days
    );
    let issues = get_github_issues(&params.repo, params.days)?;
    let s = serde_json::to_string(&issues)?;
    Ok(s)
}

/// entrypoint for the github_pull_requests tool
fn tool_github_pull_requests(
    params_str: &String,
    _ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        days: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub pull requests from repository {} for the last {} days",
        params.repo, params.days
    );
    let pull_requests = get_github_pull_requests(&params.repo, params.days)?;
    let s = serde_json::to_string(&pull_requests)?;
    Ok(s)
}

/// entrypoint for the github_pull_request tool
fn tool_github_pull_request(
    params_str: &String,
    _ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        pull_request: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub PR: {}/{}",
        params.repo, params.pull_request
    );
    let pr = get_github_pull_request(&params.repo, params.pull_request)?;
    let s = serde_json::to_string(&pr)?;
    Ok(s)
}

/// entrypoint for the github_pull_request_patch tool
fn tool_github_pull_request_patch(
    params_str: &String,
    _ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        pull_request: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub PR patch: {}/{}",
        params.repo, params.pull_request
    );
    let pr = get_github_pull_request_patch(&params.repo, params.pull_request)?;
    Ok(pr)
}

fn tool_agent_create(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
        #[serde(default)]
        description: Option<String>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    db.create_agent(&params.name, params.description.as_deref().unwrap_or(""))?;
    ctx.println(&format!("Agent '{}' created successfully", params.name));
    let result = serde_json::json!({"status": "created", "name": params.name});
    Ok(result.to_string())
}

fn tool_agent_delete(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    let deleted = db.delete_agent(&params.name)?;
    if deleted {
        ctx.println(&format!("Agent '{}' deleted", params.name));
        let result = serde_json::json!({"status": "deleted", "name": params.name});
        Ok(result.to_string())
    } else {
        let result = serde_json::json!({"status": "not_found", "name": params.name});
        Ok(result.to_string())
    }
}

fn tool_agent_list(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    let _ = params_str;
    let db = ctx.db()?;
    let agents = db.list_agents()?;
    ctx.println(&format!("Found {} agent(s)", agents.len()));
    for a in &agents {
        ctx.println(&format!("  - {} ({})", a.name, a.description));
    }
    Ok(serde_json::to_string(&agents)?)
}

fn tool_agent_get(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    match db.get_agent(&params.name)? {
        Some(agent) => {
            ctx.println(&format!("Agent '{}': {}", agent.name, agent.description));
            Ok(serde_json::to_string(&agent)?)
        }
        None => {
            let result = serde_json::json!({"error": format!("Agent '{}' not found", params.name)});
            Ok(result.to_string())
        }
    }
}

fn tool_agent_data_set(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        agent: String,
        key: String,
        value: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    db.set_agent_data(&params.agent, &params.key, &params.value)?;
    ctx.println(&format!(
        "Set data for agent '{}': {} = {}",
        params.agent, params.key, params.value
    ));
    let result = serde_json::json!({"status": "ok", "agent": params.agent, "key": params.key});
    Ok(result.to_string())
}

fn tool_agent_data_get(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        agent: String,
        key: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    match db.get_agent_data(&params.agent, &params.key)? {
        Some(value) => {
            ctx.println(&format!(
                "Agent '{}' data: {} = {}",
                params.agent, params.key, value
            ));
            let result =
                serde_json::json!({"agent": params.agent, "key": params.key, "value": value});
            Ok(result.to_string())
        }
        None => {
            let result =
                serde_json::json!({"agent": params.agent, "key": params.key, "value": null});
            Ok(result.to_string())
        }
    }
}

fn tool_agent_data_delete(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        agent: String,
        key: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    let deleted = db.delete_agent_data(&params.agent, &params.key)?;
    if deleted {
        ctx.println(&format!(
            "Deleted key '{}' for agent '{}'",
            params.key, params.agent
        ));
    }
    let result = serde_json::json!({"deleted": deleted, "agent": params.agent, "key": params.key});
    Ok(result.to_string())
}

fn tool_agent_data_list(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        agent: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    let data = db.list_agent_data(&params.agent)?;
    ctx.println(&format!(
        "Agent '{}' has {} data entries",
        params.agent,
        data.len()
    ));
    for (k, v) in &data {
        ctx.println(&format!("  {} = {}", k, v));
    }
    let entries: Vec<serde_json::Value> = data
        .into_iter()
        .map(|(k, v)| serde_json::json!({"key": k, "value": v}))
        .collect();
    Ok(serde_json::to_string(&entries)?)
}

fn tool_task_create_cron(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
        cron_expression: String,
        #[serde(default)]
        command: Option<String>,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        agent_name: Option<String>,
        #[serde(default)]
        max_runs: Option<i64>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let agent = params.agent_name.as_deref().or(ctx.agent_name.as_deref());
    let db = ctx.db()?;
    let id = db.create_cron_task(
        &params.name,
        params.description.as_deref().unwrap_or(""),
        &params.cron_expression,
        params.command.as_deref().unwrap_or(""),
        agent,
        params.max_runs,
    )?;
    ctx.println(&format!(
        "Created cron task '{}' (id={}) with schedule '{}'",
        params.name, id, params.cron_expression
    ));
    let result = serde_json::json!({"status": "created", "id": id, "name": params.name});
    Ok(result.to_string())
}

fn tool_task_create_oneshot(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
        #[serde(default)]
        run_at: Option<String>,
        #[serde(default)]
        delay_seconds: Option<u64>,
        #[serde(default)]
        command: Option<String>,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        agent_name: Option<String>,
    }
    let params: Params = serde_json::from_str(params_str)?;

    let run_at = match (params.run_at, params.delay_seconds) {
        (Some(t), _) => t,
        (None, Some(secs)) => {
            let when = chrono::Utc::now() + chrono::Duration::seconds(secs as i64);
            when.to_rfc3339()
        }
        (None, None) => {
            return Err("Either 'run_at' or 'delay_seconds' must be provided".into());
        }
    };

    let agent = params.agent_name.as_deref().or(ctx.agent_name.as_deref());
    let db = ctx.db()?;
    let id = db.create_oneshot_task(
        &params.name,
        params.description.as_deref().unwrap_or(""),
        &run_at,
        params.command.as_deref().unwrap_or(""),
        agent,
    )?;
    ctx.println(&format!(
        "Created one-shot task '{}' (id={}) scheduled at '{}'",
        params.name, id, run_at
    ));
    let result =
        serde_json::json!({"status": "created", "id": id, "name": params.name, "run_at": run_at});
    Ok(result.to_string())
}

fn tool_task_delete(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        id: i64,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    let deleted = db.delete_task(params.id)?;
    if deleted {
        ctx.println(&format!("Deleted task id={}", params.id));
    }
    let result = serde_json::json!({"deleted": deleted, "id": params.id});
    Ok(result.to_string())
}

fn tool_task_list(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        #[serde(default)]
        agent_name: Option<String>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    let tasks = db.list_tasks(params.agent_name.as_deref())?;
    ctx.println(&format!("Found {} task(s)", tasks.len()));
    for t in &tasks {
        let status = if t.enabled { "enabled" } else { "disabled" };
        ctx.println(&format!(
            "  [{}] {} - {} ({})",
            t.id, t.name, t.task_type, status
        ));
    }
    Ok(serde_json::to_string(&tasks)?)
}

fn tool_task_get(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        id: i64,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    match db.get_task(params.id)? {
        Some(task) => {
            ctx.println(&format!(
                "Task [{}]: {} ({})",
                task.id, task.name, task.task_type
            ));
            Ok(serde_json::to_string(&task)?)
        }
        None => {
            let result = serde_json::json!({"error": format!("Task id={} not found", params.id)});
            Ok(result.to_string())
        }
    }
}

fn tool_task_set_enabled(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        id: i64,
        enabled: bool,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let db = ctx.db()?;
    let updated = db.set_task_enabled(params.id, params.enabled)?;
    let status = if params.enabled {
        "enabled"
    } else {
        "disabled"
    };
    if updated {
        ctx.println(&format!("Task id={} {}", params.id, status));
    }
    let result =
        serde_json::json!({"updated": updated, "id": params.id, "enabled": params.enabled});
    Ok(result.to_string())
}

fn tool_task_pending(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    let _ = params_str;
    let db = ctx.db()?;
    let tasks = db.get_pending_tasks()?;
    ctx.println(&format!("Found {} pending task(s)", tasks.len()));
    for t in &tasks {
        ctx.println(&format!(
            "  [{}] {} - {} (next: {})",
            t.id,
            t.name,
            t.task_type,
            t.next_run_at.as_deref().unwrap_or("N/A")
        ));
    }
    Ok(serde_json::to_string(&tasks)?)
}

fn tool_send_message(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        to: String,
        message: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let from = ctx.agent_name.as_deref().unwrap_or("default");
    let db = ctx.db()?;
    let id = db.send_notification(from, &params.to, &params.message)?;
    ctx.println(&format!(
        "Message sent to agent '{}' (notification #{})",
        params.to, id
    ));
    let result = serde_json::json!({"status": "sent", "id": id, "from": from, "to": params.to});
    Ok(result.to_string())
}

fn tool_spawn_agent(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
        prompt: String,
    }
    let params: Params = serde_json::from_str(params_str)?;

    let sa_ctx = ctx
        .extra
        .as_ref()
        .and_then(|e| e.downcast_ref::<SubAgentContext>())
        .ok_or("Sub-agent context not configured")?;

    let tools = sa_ctx.tools.clone();
    let mut agent_opts = sa_ctx.opts.clone();
    let db = ctx.db.clone().ok_or("Database required for sub-agents")?;
    let parent_agent = ctx.agent_name.clone().unwrap_or("default".to_string());

    let agent_name = params.name.clone();
    let prompt = params.prompt.clone();

    let agent_config = {
        if db.get_agent(&agent_name)?.is_none() {
            db.create_agent(&agent_name, &format!("Sub-agent: {}", agent_name))?;
        }
        let _ = db.claim_agent(&agent_name, &sa_ctx.session_id);
        db.get_agent_config(&agent_name)?
    };

    if let Some(ref model) = agent_config.model {
        agent_opts.model = model.clone();
    }
    if let Some(ref endpoint) = agent_config.endpoint {
        agent_opts.endpoint = openai::normalize_endpoint(endpoint);
    }

    let mut messages: Vec<Message> = vec![make_message(
        "system",
        format!(
            "You are a sub-agent named '{}'. Complete the task and return a concise result.",
            agent_name
        ),
    )];
    if let Some(ref sp) = agent_config.system_prompt {
        messages.push(make_message("system", sp.clone()));
    }
    messages.push(make_message("user", prompt.clone()));

    ctx.println(&format!("Spawning sub-agent '{}'...", agent_name));

    let active_counter = sa_ctx.active_subagents.clone();
    active_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let sub_status_bar = sa_ctx.status_bar.clone();
    let agent_name_for_status = agent_name.clone();
    sub_status_bar.set_agent_status(&agent_name_for_status, "Running", true);
    sub_status_bar.set_agent_color(
        &agent_name_for_status,
        agent_ansi_code(&agent_name_for_status),
    );

    let session_id_for_sub = sa_ctx.session_id.clone();
    let mcp_for_sub = ctx.mcp_manager.clone();
    std::thread::spawn(move || {
        struct SubAgentGuard {
            status_bar: Arc<status_bar::StatusBar>,
            counter: Arc<std::sync::atomic::AtomicUsize>,
            db: Arc<dyn DbBackend>,
            name: String,
            session_id: String,
        }
        impl Drop for SubAgentGuard {
            fn drop(&mut self) {
                self.status_bar.clear_agent_status(&self.name);
                self.counter
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                let _ = self.db.release_agent(&self.name, &self.session_id);
            }
        }
        let _guard = SubAgentGuard {
            status_bar: sub_status_bar,
            counter: active_counter,
            db: db.clone(),
            name: agent_name_for_status,
            session_id: session_id_for_sub,
        };

        let mut sub_ctx = ToolContext::new(|_: &str| {});
        sub_ctx.db = Some(db.clone());
        sub_ctx.mcp_manager = mcp_for_sub;
        let result = post_request_with_mode(
            messages,
            &tools,
            &agent_opts,
            ResponseMode::Complete,
            &sub_ctx,
            None,
        );
        let response_text = match result {
            Ok(resp) => {
                if let Some(ref choices) = resp.choices {
                    choices
                        .first()
                        .and_then(|c| c.message.content.clone())
                        .unwrap_or_else(|| "(no response)".to_string())
                } else if let Some(ref err) = resp.error {
                    format!("Error: {}", err.message)
                } else {
                    "(empty response)".to_string()
                }
            }
            Err(e) => format!("Error: {}", e),
        };
        let notification = serde_json::json!({
            "type": "subagent_result",
            "agent": agent_name,
            "prompt": prompt,
            "response": response_text,
        });
        let _ = db.send_notification(&agent_name, &parent_agent, &notification.to_string());
    });

    let result =
        serde_json::json!({"status": "spawned", "agent": params.name, "prompt": params.prompt});
    Ok(result.to_string())
}

/// entrypoint for the patch_file tool
///
/// Applies a batch of search-and-replace edits to an existing file.  The file
/// is opened once (read-write) through `Root`, every edit is applied in
/// memory and validated before anything is written, and only the bytes that
/// actually changed are written back, without truncating the file first.
fn tool_patch_file(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;
    use std::os::unix::fs::FileExt;

    #[derive(Deserialize)]
    struct Edit {
        old_content: String,
        new_content: String,
        #[serde(default)]
        replace_all: bool,
    }

    #[derive(Deserialize)]
    struct Params {
        path: String,
        edits: Vec<Edit>,
    }

    #[derive(Serialize)]
    struct PatchFileResult {
        path: String,
        edits_applied: usize,
        replacements: usize,
        bytes_before: usize,
        bytes_after: usize,
        bytes_written: usize,
        message: String,
    }

    let params: Params = serde_json::from_str::<Params>(params_str)?;

    debug!(
        "patch_file received params: path='{}', edits={}",
        params.path,
        params.edits.len()
    );

    if params.edits.is_empty() {
        return Err("edits must contain at least one edit".into());
    }

    let root = Root::open(".")?;
    let file = root
        .open_subpath(&params.path, OpenFlags::O_RDWR)
        .map_err(|e| format!("File '{}' must exist and be writable: {}", params.path, e))?;
    if !file.metadata()?.is_file() {
        return Err(format!("'{}' is not a regular file", params.path).into());
    }

    let mut original = Vec::new();
    (&file).read_to_end(&mut original)?;
    let mut text = String::from_utf8(original.clone())
        .map_err(|_| format!("File '{}' contains invalid UTF-8", params.path))?;

    let mut replacements = 0;
    for (i, edit) in params.edits.iter().enumerate() {
        let n = i + 1;
        if edit.old_content.is_empty() {
            return Err(format!("edit {}: old_content must not be empty", n).into());
        }
        if edit.old_content == edit.new_content {
            return Err(format!("edit {}: old_content and new_content are identical", n).into());
        }

        let first = text
            .find(edit.old_content.as_str())
            .ok_or_else(|| format!("edit {}: old_content not found in '{}'", n, params.path))?;

        if edit.replace_all {
            replacements += text.matches(edit.old_content.as_str()).count();
            text = text.replace(edit.old_content.as_str(), &edit.new_content);
        } else {
            // Look for a second match starting after the first character of
            // the first one, so overlapping matches count as ambiguous too.
            let skip = first + edit.old_content.chars().next().map_or(1, char::len_utf8);
            if text[skip..].contains(edit.old_content.as_str()) {
                return Err(format!(
                    "edit {}: old_content matches multiple times in '{}'. Set replace_all=true or provide more context to make it unique",
                    n, params.path
                )
                .into());
            }
            text.replace_range(first..first + edit.old_content.len(), &edit.new_content);
            replacements += 1;
        }
    }

    let new_bytes = text.as_bytes();

    // Only rewrite the region between the first and the last differing byte.
    let prefix = original
        .iter()
        .zip(new_bytes)
        .take_while(|(a, b)| a == b)
        .count();
    let suffix = if original.len() == new_bytes.len() {
        original[prefix..]
            .iter()
            .rev()
            .zip(new_bytes[prefix..].iter().rev())
            .take_while(|(a, b)| a == b)
            .count()
    } else {
        0
    };
    let end = new_bytes.len() - suffix;

    if prefix < end {
        file.write_all_at(&new_bytes[prefix..end], prefix as u64)?;
    }
    if new_bytes.len() != original.len() {
        file.set_len(new_bytes.len() as u64)?;
    }

    let result = PatchFileResult {
        path: params.path.clone(),
        edits_applied: params.edits.len(),
        replacements,
        bytes_before: original.len(),
        bytes_after: new_bytes.len(),
        bytes_written: end.saturating_sub(prefix),
        message: format!("File '{}' patched successfully", params.path),
    };

    ctx.println(&format!("\u{1f4dd} {}", result.message));
    ctx.println(&format!("   Path: {}", result.path));
    ctx.println(&format!(
        "   Edits: {} ({} replacements)",
        result.edits_applied, result.replacements
    ));
    ctx.println(&format!(
        "   Bytes: {} -> {} ({} written)",
        result.bytes_before, result.bytes_after, result.bytes_written
    ));

    if let Ok(old_str) = String::from_utf8(original) {
        show_diff(ctx, &old_str, &text, &params.path);
    }

    Ok(serde_json::to_string(&result)?)
}

fn initialize_tools(unsafe_tools: bool, allowed: Option<&[String]>) -> ToolsCollection {
    let mut tools: ToolsCollection = ToolsCollection::new();

    append_tool(
        &mut tools,
        "github_pull_request".to_string(),
        tool_github_pull_request,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_pull_request",
                "description": "Get information about a github pull request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. owner/repo"
                        },
                        "pull_request": {
                            "type": "number",
                            "description": "number of the pull request"
                        }
                    },
                    "required": [
                        "repo",
                        "pull_request"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_pull_request_patch".to_string(),
        tool_github_pull_request_patch,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_pull_request_patch",
                "description": "Get the raw patch for a github pull request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. owner/repo"
                        },
                        "pull_request": {
                            "type": "number",
                            "description": "number of the pull request"
                        }
                    },
                    "required": [
                        "repo",
                        "pull_request"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "read_file".to_string(),
        tool_read_file,
        r#"
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Get the content of a file stored in the repository. Returns JSON with 'content' field containing file content on success, or 'error' field with error message on failure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the repository, e.g. src/main.rs"
                        }
                    },
                    "required": [
                        "path"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "delete_path".to_string(),
        tool_delete_path,
        r#"
        {
            "type": "function",
            "function": {
                "name": "delete_path",
                "description": "Delete a file or a directory under the current directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the current directory"
                        }
                    },
                    "required": [
                        "path"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "write_file".to_string(),
        tool_write_file,
        r#"
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or replace the content of a file, or partially edit it. For full writes, provide path and content. For partial edits, use one of: (1) offset+length for byte-level patching, (2) start_line+end_line for line-range replacement, or (3) old_content for search-and-replace. Only one partial edit mode can be used at a time. Partial edits require the file to already exist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the repository, e.g. src/main.rs"
                        },
                        "content": {
                            "type": "string",
                            "description": "the file content for full writes, or the replacement text for partial edits"
                        },
                        "mode": {
                            "type": "string",
                            "description": "file permissions mode in octal format (e.g., '0644', '0755', '0600'). Defaults to '0644' for regular files. Use '0755' for executable files.",
                            "default": "0644"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "byte offset (0-based) to start the partial write. Requires 'length'. Cannot be combined with start_line or old_content."
                        },
                        "length": {
                            "type": "integer",
                            "description": "number of bytes to replace starting from 'offset'. The replaced region is substituted with 'content'."
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based line number to start replacing. Requires 'end_line'. Cannot be combined with offset or old_content."
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "1-based line number to stop replacing (inclusive). Lines from start_line to end_line are replaced with 'content'."
                        },
                        "old_content": {
                            "type": "string",
                            "description": "exact text to find in the file and replace with 'content'. Must match exactly once unless replace_all is true. Cannot be combined with offset or start_line."
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "if true, replace all occurrences of old_content. Defaults to false (requires unique match)."
                        }
                    },
                    "required": [
                        "path",
                        "content"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "patch_file".to_string(),
        tool_patch_file,
        r#"
        {
            "type": "function",
            "function": {
                "name": "patch_file",
                "description": "Apply one or more search-and-replace edits to an existing file in a single call. Prefer this over write_file for changing several places in a file: the edits are applied in order (each one sees the result of the previous ones), validated together, and either all applied or none, and only the changed bytes are rewritten. Each old_content must match exactly once unless replace_all is true.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the existing file under the repository, e.g. src/main.rs"
                        },
                        "edits": {
                            "type": "array",
                            "description": "the edits to apply, in order",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_content": {
                                        "type": "string",
                                        "description": "exact, non-empty text to find in the file"
                                    },
                                    "new_content": {
                                        "type": "string",
                                        "description": "text that replaces old_content (may be empty to delete it)"
                                    },
                                    "replace_all": {
                                        "type": "boolean",
                                        "description": "if true, replace every occurrence of old_content. Defaults to false (requires a unique match)."
                                    }
                                },
                                "required": [
                                    "old_content",
                                    "new_content"
                                ],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": [
                        "path",
                        "edits"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "glob".to_string(),
        tool_glob,
        r#"
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files matching a glob pattern in the current directory. Supports patterns like '*.rs', '**/*.txt', 'src/**/*.rs', etc. Cannot access parent directories or absolute paths for security.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The glob pattern to match files against (e.g., '*.rs', '**/*.txt', 'src/**/*.rs'). Cannot contain '../' or start with '/' for security."
                        }
                    },
                    "required": [
                        "pattern"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "grep_in_current_directory".to_string(),
        tool_grep_in_current_directory,
        r#"
        {
            "type": "function",
            "function": {
                "name": "grep_in_current_directory",
                "description": "Grep for a pattern in the current directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The pattern to search for."
                        }
                    },
                    "required": [
                        "pattern"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_issue".to_string(),
        tool_github_issue,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_issue",
                "description": "Get the github issue description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. owner/repo"
                        },
                        "issue": {
                            "type": "number",
                            "description": "number of the github issue"
                        }
                    },
                    "required": [
                        "repo",
                        "issue"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_issue_comments".to_string(),
        tool_github_issue_comments,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_issue_comments",
                "description": "Get the comments associated with the github issue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. owner/repo"
                        },
                        "issue": {
                            "type": "number",
                            "description": "number of the github issue"
                        }
                    },
                    "required": [
                        "repo",
                        "issue"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_issues".to_string(),
        tool_github_issues,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_issues",
                "description": "Get issues from a GitHub repository that have been updated within the last specified number of days.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. owner/repo"
                        },
                        "days": {
                            "type": "number",
                            "description": "number of days to look back for updated issues"
                        }
                    },
                    "required": [
                        "repo",
                        "days"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_pull_requests".to_string(),
        tool_github_pull_requests,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_pull_requests",
                "description": "Get pull requests from a GitHub repository that have been updated within the last specified number of days.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. owner/repo"
                        },
                        "days": {
                            "type": "number",
                            "description": "number of days to look back for updated pull requests"
                        }
                    },
                    "required": [
                        "repo",
                        "days"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_create".to_string(),
        tool_agent_create,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_create",
                "description": "Create a new named agent in the database for organizing tasks and storing data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Unique name for the agent"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of the agent's purpose"
                        }
                    },
                    "required": [
                        "name"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_delete".to_string(),
        tool_agent_delete,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_delete",
                "description": "Delete an agent and all its associated data from the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the agent to delete"
                        }
                    },
                    "required": [
                        "name"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_list".to_string(),
        tool_agent_list,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_list",
                "description": "List all agents in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_get".to_string(),
        tool_agent_get,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_get",
                "description": "Get details of a specific agent by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the agent to retrieve"
                        }
                    },
                    "required": [
                        "name"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_data_set".to_string(),
        tool_agent_data_set,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_data_set",
                "description": "Store a key-value pair for an agent. Overwrites existing value if key exists.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent"
                        },
                        "key": {
                            "type": "string",
                            "description": "The key to store"
                        },
                        "value": {
                            "type": "string",
                            "description": "The value to associate with the key"
                        }
                    },
                    "required": [
                        "agent",
                        "key",
                        "value"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_data_get".to_string(),
        tool_agent_data_get,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_data_get",
                "description": "Retrieve the value for a key stored for an agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent"
                        },
                        "key": {
                            "type": "string",
                            "description": "The key to retrieve"
                        }
                    },
                    "required": [
                        "agent",
                        "key"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_data_delete".to_string(),
        tool_agent_data_delete,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_data_delete",
                "description": "Delete a key-value pair for an agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent"
                        },
                        "key": {
                            "type": "string",
                            "description": "The key to delete"
                        }
                    },
                    "required": [
                        "agent",
                        "key"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_data_list".to_string(),
        tool_agent_data_list,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_data_list",
                "description": "List all key-value pairs stored for an agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent"
                        }
                    },
                    "required": [
                        "agent"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_create_cron".to_string(),
        tool_task_create_cron,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_create_cron",
                "description": "Create a recurring scheduled task using a cron expression. Uses 7-field cron format: 'sec min hour day_of_month month day_of_week year'. Example: '0 30 9 * * Mon-Fri *' means 9:30 AM every weekday. The command is sent to the AI when the task fires.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the task"
                        },
                        "cron_expression": {
                            "type": "string",
                            "description": "7-field cron expression: sec min hour day_of_month month day_of_week year"
                        },
                        "command": {
                            "type": "string",
                            "description": "JSON tool call to execute when the task fires, e.g. {\"tool\": \"run_command\", \"arguments\": {\"command\": \"echo hello\"}}"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of what the task does"
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "Optional agent to associate the task with"
                        },
                        "max_runs": {
                            "type": "integer",
                            "description": "Maximum number of times the task will run before auto-disabling. Omit for unlimited."
                        }
                    },
                    "required": [
                        "name",
                        "cron_expression",
                        "command"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_create_oneshot".to_string(),
        tool_task_create_oneshot,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_create_oneshot",
                "description": "Create a one-shot scheduled task that runs once. Use delay_seconds for relative timing (preferred) or run_at for absolute. The command is sent to the AI when the task fires.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the task"
                        },
                        "command": {
                            "type": "string",
                            "description": "JSON tool call to execute when the task fires, e.g. {\"tool\": \"run_command\", \"arguments\": {\"command\": \"echo hello\"}}"
                        },
                        "delay_seconds": {
                            "type": "number",
                            "description": "Number of seconds from now to fire the task (preferred over run_at)"
                        },
                        "run_at": {
                            "type": "string",
                            "description": "Absolute time in RFC 3339 format (e.g. '2025-12-31T23:59:59Z'). Use delay_seconds instead when possible."
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of what the task does"
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "Optional agent to associate the task with"
                        }
                    },
                    "required": [
                        "name",
                        "command"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_delete".to_string(),
        tool_task_delete,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_delete",
                "description": "Delete a scheduled task by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "number",
                            "description": "ID of the task to delete"
                        }
                    },
                    "required": [
                        "id"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_list".to_string(),
        tool_task_list,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_list",
                "description": "List all scheduled tasks, optionally filtered by agent name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Optional agent name to filter tasks by"
                        }
                    },
                    "required": [],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_get".to_string(),
        tool_task_get,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_get",
                "description": "Get details of a specific scheduled task by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "number",
                            "description": "ID of the task to retrieve"
                        }
                    },
                    "required": [
                        "id"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_set_enabled".to_string(),
        tool_task_set_enabled,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_set_enabled",
                "description": "Enable or disable a scheduled task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "number",
                            "description": "ID of the task"
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "true to enable, false to disable"
                        }
                    },
                    "required": [
                        "id",
                        "enabled"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_pending".to_string(),
        tool_task_pending,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_pending",
                "description": "List all enabled scheduled tasks whose next_run_at is in the past (i.e. tasks that are due to run).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "send_message".to_string(),
        tool_send_message,
        r#"
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a message to another agent. The message will appear in the target agent's session and trigger a response.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Name of the target agent to send the message to"
                        },
                        "message": {
                            "type": "string",
                            "description": "The message content to send"
                        }
                    },
                    "required": [
                        "to",
                        "message"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "spawn_agent".to_string(),
        tool_spawn_agent,
        r#"
        {
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": "Spawn a sub-agent to work on a task in parallel. The sub-agent runs in the background and its result will be delivered asynchronously. You can spawn multiple sub-agents at once for parallel work.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the sub-agent (used for display and identification)"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The task or question for the sub-agent to work on"
                        }
                    },
                    "required": [
                        "name",
                        "prompt"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    if !unsafe_tools {
        if let Some(allowed_list) = allowed {
            tools.retain(|name, _| allowed_list.iter().any(|a| a == name));
        }
        return tools;
    }

    append_tool(
        &mut tools,
        "fetch_web_content".to_string(),
        tool_fetch_web_content,
        r#"
        {
            "type": "function",
            "function": {
                "name": "fetch_web_content",
                "description": "Fetch content from a web URL. Returns JSON with URL, status code, content type, headers, and the actual content. Supports HTTP/HTTPS URLs with configurable timeout and redirect handling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The HTTP/HTTPS URL to fetch content from"
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "description": "Request timeout in seconds (default: 30)"
                        },
                        "follow_redirects": {
                            "type": "boolean",
                            "description": "Whether to follow HTTP redirects (default: true)"
                        },
                        "max_content_length": {
                            "type": "number",
                            "description": "Maximum content length in bytes (default: 10MB)"
                        }
                    },
                    "required": [
                        "url"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "run_command".to_string(),
        tool_run_command,
        r#"
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a command and return results in JSON format. Returns {\"stdout\": string, \"stderr\": string, \"exit_code\": number|null, \"success\": boolean}. Commands do not fail on non-zero exit codes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute, it must be the name of the executable only, e.g. /usr/bin/ls"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Additional arguments to pass to the command after the executable path, e.g. [\"-l\", \"-a\"]"
                        }
                    },
                    "required": [
                        "command"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    if let Some(allowed_list) = allowed {
        tools.retain(|name, _| allowed_list.iter().any(|a| a == name));
    }

    debug!("Tools initialization completed with {} tools", tools.len());
    tools
}

fn add_tools_prompt(messages: &mut Vec<Message>, use_tools: bool) {
    let prompts = if use_tools {
        vec![
            "Use the available tools as much as possible to find a solution.  Iterate until the problem is solved.  Terminate only when you are sure to have found the solution, if a tool fails, analyze the failure, fix the issue and call again the tool.  Never ask to run commands manually or ask for permissions, just run the tool.",
            "When you've completed the task, you must terminate immediately the execution, do not explain your choices multiple times.",
        ]
    } else {
        vec![
            "You have access to various tools for file operations and code analysis.  Only use these tools when the user explicitly asks for file operations, code analysis, or repository interactions.  For simple questions, conversations, or general requests, respond directly without using tools.",
        ]
    };

    for prompt in prompts {
        messages.push(make_message("system", prompt.to_string()));
    }
}

fn initialize_chat_messages(tools: &ToolsCollection, _opts: &Opts) -> Vec<Message> {
    let mut messages: Vec<Message> = vec![];

    let use_tools = !tools.is_empty();
    add_tools_prompt(&mut messages, use_tools);

    debug!("Initialized chat with {} system messages", messages.len());
    messages
}

/// Sends a prompt to the OpenAI API and prints the AI's response to standard output.
fn post_request_and_print_output(
    prompt: &String,
    system_prompts: Option<Vec<String>>,
    opts: &Opts,
    db: Option<Arc<dyn DbBackend>>,
    mcp_manager: Option<Arc<faber::mcp::McpManager>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Prompt: {}", prompt);

    let model = opts.model.clone().unwrap_or(DEFAULT_MODEL.to_string());
    debug!("Using model: {}", model);

    let parameters = parse_parameters(&opts.parameter)?;

    let endpoint = opts
        .endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    let normalized_endpoint = openai::normalize_endpoint(&endpoint);

    let openai_opts = openai::Opts {
        max_tokens: opts.max_tokens,
        model: model,
        endpoint: normalized_endpoint,
        tool_choice: opts.tool_choice.clone(),
        api_key: opts.api_key.clone(),
        max_retries: None,
        retry_base_delay_secs: None,
        parameters,
    };

    let allowed_tools = if opts.tools.is_empty() {
        None
    } else {
        Some(opts.tools.clone())
    };
    let tools = match opts.no_tools {
        true => {
            debug!("Tools are disabled");
            ToolsCollection::new()
        }
        false => {
            debug!("Initializing tools for AI request");
            initialize_tools(opts.unsafe_tools, allowed_tools.as_deref())
        }
    };

    let mut messages = initialize_chat_messages(&tools, opts);

    if let Some(ref sys_prompts) = system_prompts {
        debug!("Using {} system prompts", sys_prompts.len());
        for sp in sys_prompts {
            messages.push(make_message("system", sp.clone()));
        }
    }
    messages.push(make_message("user", prompt.clone()));

    let mut tool_context = ToolContext::new(|msg: &str| {
        println!("{}", msg);
    });
    tool_context.db = db;
    tool_context.mcp_manager = mcp_manager;

    let response: OpenAIResponse = post_request(messages, &tools, &openai_opts, &tool_context)?;

    if let Some(choices) = response.choices {
        debug!("Received {} choices in response", choices.len());
        if let Some(choice) = choices.first() {
            if let Some(content) = &choice.message.content {
                println!("{}", content);
            }
        }
    } else {
        warn!("No choices received in the AI response");
    }
    Ok(())
}

/// Sends the concatenated content of specified files as a prompt to the AI.
fn prompt_command(
    prompt: &String,
    files: &Vec<String>,
    opts: &Opts,
    db: Option<Arc<dyn DbBackend>>,
    mcp_manager: Option<Arc<faber::mcp::McpManager>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Executing prompt command with {} files", files.len());
    let mut system_prompts: Vec<String> = vec![];

    for file in files {
        debug!("Reading file for prompt context: {}", file);
        let contents = fs::read_to_string(file)?;
        system_prompts.push(contents);
    }
    post_request_and_print_output(prompt, Some(system_prompts), opts, db, mcp_manager)
}

enum ChatCommand {
    Help,
    Quit,
    Clear,
    Show,
    Limit(usize),
    Backtrace(usize),
    Summarize,
    System(String),
    Agents,
    CreateAgent(String),
    SelectAgent(String),
    DeleteAgent(String),
    McpRefresh,
    Tools,
    Message(String),
    Empty,
    Invalid(String),
}

fn parse_chat_command(line: &str) -> ChatCommand {
    if line.is_empty() {
        return ChatCommand::Empty;
    }

    let normalized = if line.starts_with('\\') {
        format!("/{}", &line[1..])
    } else {
        line.to_string()
    };

    if normalized == "/help" {
        return ChatCommand::Help;
    }
    if normalized == "/quit" {
        return ChatCommand::Quit;
    }
    if normalized == "/clear" {
        return ChatCommand::Clear;
    }
    if normalized == "/show" {
        return ChatCommand::Show;
    }
    if normalized.starts_with("/limit ") {
        let parts: Vec<&str> = normalized.split_whitespace().collect();
        if parts.len() == 2 {
            if let Ok(n) = parts[1].parse::<usize>() {
                return ChatCommand::Limit(n);
            }
        }
        return ChatCommand::Invalid("Usage: /limit <number_of_messages>".to_string());
    }
    if normalized.starts_with("/backtrace ") {
        let parts: Vec<&str> = normalized.split_whitespace().collect();
        if parts.len() == 2 {
            if let Ok(n) = parts[1].parse::<usize>() {
                return ChatCommand::Backtrace(n);
            }
        }
        return ChatCommand::Invalid("Usage: /backtrace <number_of_messages>".to_string());
    }
    if normalized == "/summarize" {
        return ChatCommand::Summarize;
    }
    if normalized.starts_with("/system ") {
        let system_message = normalized
            .strip_prefix("/system ")
            .unwrap_or("")
            .to_string();
        if !system_message.is_empty() {
            return ChatCommand::System(system_message);
        }
        return ChatCommand::Invalid("Usage: /system <message>".to_string());
    }

    if normalized == "/agents" {
        return ChatCommand::Agents;
    }
    if normalized.starts_with("/create-agent ") {
        let name = normalized
            .strip_prefix("/create-agent ")
            .unwrap()
            .trim()
            .to_string();
        if name.is_empty() {
            return ChatCommand::Invalid("Usage: /create-agent <name>".to_string());
        }
        return ChatCommand::CreateAgent(name);
    }
    if normalized.starts_with("/select-agent ") {
        let name = normalized
            .strip_prefix("/select-agent ")
            .unwrap()
            .trim()
            .to_string();
        if name.is_empty() {
            return ChatCommand::Invalid("Usage: /select-agent <name>".to_string());
        }
        return ChatCommand::SelectAgent(name);
    }
    if normalized.starts_with("/delete-agent ") {
        let name = normalized
            .strip_prefix("/delete-agent ")
            .unwrap()
            .trim()
            .to_string();
        if name.is_empty() {
            return ChatCommand::Invalid("Usage: /delete-agent <name>".to_string());
        }
        return ChatCommand::DeleteAgent(name);
    }
    if normalized == "/mcp-refresh" {
        return ChatCommand::McpRefresh;
    }
    if normalized == "/tools" {
        return ChatCommand::Tools;
    }

    if normalized.starts_with('/') {
        return ChatCommand::Invalid(format!("Unknown command: {}", normalized));
    }

    ChatCommand::Message(line.to_string())
}

fn build_openai_opts(opts: &Opts, agent_config: &db::AgentConfig) -> openai::Opts {
    let model = agent_config
        .model
        .clone()
        .or_else(|| opts.model.clone())
        .unwrap_or(DEFAULT_MODEL.to_string());
    let endpoint = agent_config
        .endpoint
        .clone()
        .or_else(|| opts.endpoint.clone())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    let normalized_endpoint = openai::normalize_endpoint(&endpoint);
    let parameters = parse_parameters(&opts.parameter).unwrap_or_default();

    openai::Opts {
        max_tokens: opts.max_tokens,
        model,
        endpoint: normalized_endpoint,
        tool_choice: opts.tool_choice.clone(),
        api_key: opts.api_key.clone(),
        max_retries: None,
        retry_base_delay_secs: None,
        parameters,
    }
}

fn initialize_agent_messages(
    tools: &ToolsCollection,
    opts: &Opts,
    agent_config: &db::AgentConfig,
) -> Vec<Message> {
    let mut messages = initialize_chat_messages(tools, opts);
    if let Some(ref prompt) = agent_config.system_prompt {
        messages.push(make_message("system", prompt.clone()));
    }
    messages
}

fn handle_chat_command(
    command: ChatCommand,
    active_agent: &mut AgentState,
    tools: &ToolsCollection,
    opts: &Opts,
    chat_pb: &ChatPrinter,
    db: &Option<Arc<dyn DbBackend>>,
    openai_opts: &mut openai::Opts,
    prompt_text: &Arc<Mutex<String>>,
    agent_names: &Arc<Mutex<Vec<String>>>,
    session_id: &str,
    mcp_manager: &Option<Arc<faber::mcp::McpManager>>,
    status_bar: &status_bar::StatusBar,
) -> Result<bool, Box<dyn Error>> {
    let messages = &mut active_agent.messages;
    match command {
        ChatCommand::Help => {
            chat_pb.println("Available commands:");
            chat_pb.println("  /help                  Show this help message");
            chat_pb.println("  /quit                  Exit the chat session");
            chat_pb
                .println("  /clear                 Clear chat history and restore system prompts");
            chat_pb.println("  /show                  Show current chat history");
            chat_pb.println("  /limit <n>             Keep only the last n messages");
            chat_pb.println("  /backtrace <n>         Remove the last n messages");
            chat_pb.println("  /summarize             Replace the chat history with a summary");
            chat_pb.println("  /system <message>      Add a system message to the conversation");
            chat_pb.println("  /agents                List all agents");
            chat_pb.println("  /create-agent <name>   Create a new agent");
            chat_pb.println("  /select-agent <name>   Switch to an existing agent");
            chat_pb.println("  /delete-agent <name>   Delete an agent");
            chat_pb.println("  /mcp-refresh           Refresh MCP tool definitions");
            chat_pb.println("  /tools                 List all available tools");
            Ok(true)
        }
        ChatCommand::Quit => Ok(false),
        ChatCommand::Clear => {
            *messages = initialize_chat_messages(tools, opts);
            if let Some(db) = db {
                db.clear_agent_messages(&active_agent.name)?;
            }
            chat_pb.println("Chat history cleared and system prompts restored.");
            Ok(true)
        }
        ChatCommand::Show => {
            if messages.is_empty() {
                chat_pb.println("Chat history is empty.");
            } else {
                chat_pb.println("Current chat history:");
                for (i, msg) in messages.iter().enumerate() {
                    chat_pb.println(&format!(
                        "{}: [{}] {}",
                        i + 1,
                        msg.role,
                        msg.content.as_ref().unwrap_or(&"<no content>".to_string())
                    ));
                    if let Some(tool_calls) = &msg.tool_calls {
                        if !tool_calls.is_empty() {
                            chat_pb.println("  Tool Calls:");
                            for (j, tool_call) in tool_calls.iter().enumerate() {
                                chat_pb.println(&format!(
                                    "    {}.{}: {} ({})",
                                    i + 1,
                                    j + 1,
                                    tool_call.function.name,
                                    tool_call.id
                                ));
                                chat_pb.println(&format!(
                                    "      Args: {}",
                                    tool_call.function.arguments
                                ));
                            }
                        }
                    }
                }
            }
            Ok(true)
        }
        ChatCommand::Limit(n) => {
            if n == 0 {
                chat_pb.println("Limit cannot be zero. Clearing history instead.");
                *messages = initialize_chat_messages(tools, opts);
                if let Some(db) = db {
                    db.clear_agent_messages(&active_agent.name)?;
                }
            } else if messages.len() > n {
                *messages = messages.split_off(messages.len() - n);
                chat_pb.println(&format!("Chat history limited to the last {} messages.", n));
            } else {
                chat_pb.println(&format!(
                    "Chat history is already within the limit of {}.",
                    n
                ));
            }
            Ok(true)
        }
        ChatCommand::Backtrace(n) => {
            if n == 0 {
                chat_pb.println("Backtrace steps must be a positive number.");
            } else if n > messages.len() {
                chat_pb.println(&format!(
                    "Cannot go back {} steps, history has only {} messages. Clearing history.",
                    n,
                    messages.len()
                ));
                *messages = initialize_chat_messages(tools, opts);
                if let Some(db) = db {
                    db.clear_agent_messages(&active_agent.name)?;
                }
            } else {
                messages.truncate(messages.len() - n);
                chat_pb.println(&format!("Went back {} steps in chat history.", n));
            }
            Ok(true)
        }
        ChatCommand::System(system_message) => {
            messages.push(make_message("system", system_message));
            chat_pb.println("System message added to conversation.");
            Ok(true)
        }
        ChatCommand::Agents => {
            if let Some(db) = db {
                let agents = db.list_agents()?;
                if agents.is_empty() {
                    chat_pb.println("No agents.");
                } else {
                    let mut table = Table::new();
                    table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
                    table.set_titles(Row::new(vec![
                        Cell::new("Name"),
                        Cell::new("Description"),
                        Cell::new("Messages"),
                        Cell::new("Active"),
                    ]));
                    for agent in &agents {
                        let count = db.agent_message_count(&agent.name)?;
                        let active = if agent.name == active_agent.name {
                            "* (this session)"
                        } else if agent.session_id.is_some()
                            && agent.heartbeat_at.as_ref().is_some_and(|hb| {
                                chrono::NaiveDateTime::parse_from_str(hb, "%Y-%m-%d %H:%M:%S")
                                    .is_ok_and(|dt| {
                                        chrono::Utc::now()
                                            .naive_utc()
                                            .signed_duration_since(dt)
                                            .num_seconds()
                                            < 10
                                    })
                            })
                        {
                            "* (other session)"
                        } else {
                            ""
                        };
                        table.add_row(Row::new(vec![
                            Cell::new(&agent.name),
                            Cell::new(&agent.description),
                            Cell::new(&count.to_string()),
                            Cell::new(active),
                        ]));
                    }
                    chat_pb.println(&table.to_string());
                }
            } else {
                chat_pb.println("Database not configured.");
            }
            Ok(true)
        }
        ChatCommand::CreateAgent(name) => {
            if let Some(db) = db {
                match db.create_agent(&name, "") {
                    Ok(_) => {
                        if let Ok(mut names) = agent_names.lock() {
                            if !names.contains(&name) {
                                names.push(name.clone());
                            }
                        }
                        chat_pb.println(&format!(
                            "Agent '{}' created. Use /select-agent {} to switch.",
                            name, name
                        ));
                    }
                    Err(e) => {
                        chat_pb.println(&format!("Failed to create agent: {}", e));
                    }
                }
            } else {
                chat_pb.println("Database not configured.");
            }
            Ok(true)
        }
        ChatCommand::SelectAgent(name) => {
            if name == active_agent.name {
                chat_pb.println(&format!("Already on agent '{}'.", name));
                return Ok(true);
            }
            if let Some(db) = db {
                if db.get_agent(&name)?.is_none() {
                    chat_pb.println(&format!(
                        "Agent '{}' not found. Use /create-agent {} first.",
                        name, name
                    ));
                    return Ok(true);
                }

                if !db.claim_agent(&name, session_id)? {
                    chat_pb.println(&format!(
                        "Agent '{}' is already claimed by another session.",
                        name
                    ));
                    return Ok(true);
                }

                let msgs: Vec<serde_json::Value> = active_agent
                    .messages
                    .iter()
                    .map(|m| serde_json::to_value(m).unwrap())
                    .collect();
                db.save_agent_messages(&active_agent.name, &msgs)?;
                db.release_agent(&active_agent.name, session_id)?;

                let agent_config = db.get_agent_config(&name)?;
                *openai_opts = build_openai_opts(opts, &agent_config);

                let vals = db.load_agent_messages(&name)?;
                let loaded: Vec<Message> = vals
                    .into_iter()
                    .map(|v| serde_json::from_value(v).unwrap())
                    .collect();
                active_agent.name = name.clone();
                if loaded.is_empty() {
                    active_agent.messages = initialize_agent_messages(tools, opts, &agent_config);
                } else {
                    active_agent.messages = loaded;
                }

                *prompt_text.lock().map_err(|e| format!("lock: {}", e))? =
                    format!("{}", agent_style(&name).apply_to(format!("{}> ", name)));

                status_bar.set_color(agent_ansi_code(&name));

                chat_pb.println(&format!("Switched to agent '{}'.\n", name));
            } else {
                chat_pb.println("Database not configured.");
            }
            Ok(true)
        }
        ChatCommand::DeleteAgent(name) => {
            if name == "default" {
                chat_pb.println("Cannot delete the default agent.");
                return Ok(true);
            }
            if name == active_agent.name {
                chat_pb.println("Cannot delete the active agent. Switch first with /select-agent.");
                return Ok(true);
            }
            if let Some(db) = db {
                if db.delete_agent(&name)? {
                    if let Ok(mut names) = agent_names.lock() {
                        names.retain(|n| n != &name);
                    }
                    chat_pb.println(&format!("Agent '{}' deleted.", name));
                } else {
                    chat_pb.println(&format!("Agent '{}' not found.", name));
                }
            } else {
                chat_pb.println("Database not configured.");
            }
            Ok(true)
        }
        ChatCommand::McpRefresh => {
            if let Some(mcp) = mcp_manager {
                match mcp.refresh() {
                    Ok(count) => {
                        chat_pb.println(&format!("MCP tools refreshed: {} tools available", count));
                    }
                    Err(e) => {
                        chat_pb.println(&format!("MCP refresh failed: {}", e));
                    }
                }
            } else {
                chat_pb.println("No MCP servers configured.");
            }
            Ok(true)
        }
        ChatCommand::Tools => {
            let mut names: Vec<&String> = tools.keys().collect();
            names.sort();
            if !names.is_empty() {
                chat_pb.println("Built-in tools:");
                for name in &names {
                    chat_pb.println(&format!("  {}", name));
                }
            }
            if let Some(mcp) = mcp_manager {
                let schemas = mcp.get_tool_schemas();
                if !schemas.is_empty() {
                    chat_pb.println("MCP tools:");
                    let mut mcp_names: Vec<String> = schemas
                        .iter()
                        .filter_map(|s| {
                            s.get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .map(|n| n.to_string())
                        })
                        .collect();
                    mcp_names.sort();
                    for name in &mcp_names {
                        chat_pb.println(&format!("  {}", name));
                    }
                }
            }
            if names.is_empty()
                && mcp_manager
                    .as_ref()
                    .map_or(true, |m| m.get_tool_schemas().is_empty())
            {
                chat_pb.println("No tools available.");
            }
            Ok(true)
        }
        ChatCommand::Summarize | ChatCommand::Message(_) => Ok(false),
        ChatCommand::Empty => Ok(true),
        ChatCommand::Invalid(error_msg) => {
            chat_pb.println(&error_msg);
            Ok(true)
        }
    }
}

/// Extracts a live preview from a tool call's JSON arguments while they're
/// still streaming in: the `key=value` pairs that have already arrived in
/// full, in order, formatted as `key=value, key2=value2`. Stops at the
/// first key or value that isn't complete yet (still returning whatever
/// came before it) and returns `None` if nothing is complete yet, so the
/// status line can show real progress ("path=\"Cargo.toml\"") instead of a
/// meaningless byte count, without ever displaying broken-looking partial
/// JSON.
fn preview_partial_tool_arguments(args_json: &str) -> Option<String> {
    let bytes = args_json.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }

    // Scans a JSON string starting at its opening quote, returning the
    // index just past the closing quote, or None if it isn't closed yet.
    fn scan_string(bytes: &[u8], start: usize) -> Option<usize> {
        let mut i = start + 1;
        let mut escaped = false;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' if !escaped => escaped = true,
                b'"' if !escaped => return Some(i + 1),
                _ => escaped = false,
            }
            i += 1;
        }
        None
    }

    // Scans a nested object/array value starting at its opening brace or
    // bracket, returning the index just past its matching close, or None
    // if it isn't closed yet.
    fn scan_nested(bytes: &[u8], start: usize) -> Option<usize> {
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        for (offset, &b) in bytes[start..].iter().enumerate() {
            if in_string {
                match b {
                    b'\\' if !escaped => escaped = true,
                    b'"' if !escaped => in_string = false,
                    _ => escaped = false,
                }
                continue;
            }
            match b {
                b'"' => in_string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(start + offset + 1);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        i
    }

    let mut pairs = Vec::new();
    let mut i = skip_ws(bytes, 1); // past the opening '{'
    while i < bytes.len() && bytes[i] == b'"' {
        let key_end = match scan_string(bytes, i) {
            Some(end) => end,
            None => break,
        };
        let key = &args_json[i + 1..key_end - 1];

        i = skip_ws(bytes, key_end);
        if bytes.get(i) != Some(&b':') {
            break;
        }
        i = skip_ws(bytes, i + 1);
        if i >= bytes.len() {
            break;
        }

        let (value, value_end) = match bytes[i] {
            b'"' => match scan_string(bytes, i) {
                Some(end) => {
                    let raw = &args_json[i..end];
                    let display = serde_json::from_str::<String>(raw)
                        .map(|s| format!("{:?}", s))
                        .unwrap_or_else(|_| raw.to_string());
                    (display, end)
                }
                None => break,
            },
            b'{' | b'[' => match scan_nested(bytes, i) {
                Some(end) => (args_json[i..end].to_string(), end),
                None => break,
            },
            _ => {
                // A number, bool or null: only complete once followed by an
                // actual delimiter, never just because the buffer ends here
                // (more digits could still be on the way).
                let start = i;
                let mut j = i;
                while j < bytes.len() && !matches!(bytes[j], b',' | b'}' | b' ' | b'\t' | b'\n') {
                    j += 1;
                }
                if j >= bytes.len() {
                    break;
                }
                (args_json[start..j].to_string(), j)
            }
        };
        pairs.push(format!("{}={}", key, value));

        i = skip_ws(bytes, value_end);
        if bytes.get(i) == Some(&b',') {
            i = skip_ws(bytes, i + 1);
        } else {
            // Either the object just closed, or something unexpected
            // follows; either way there's nothing more to safely extract.
            break;
        }
    }

    if pairs.is_empty() {
        None
    } else {
        // Guarantee single-line output regardless of caller: a nested
        // object/array value is copied through verbatim and could be
        // pretty-printed with real newlines, and a string value with an
        // unescaped control character (invalid JSON, but tolerated by the
        // scanner above) would otherwise carry a literal newline straight
        // into the status line.
        let joined = pairs.join(", ");
        Some(
            joined
                .replace('\n', " ")
                .replace('\r', " ")
                .replace('\t', " "),
        )
    }
}

fn format_tool_arguments(args_json: &str) -> String {
    let clean_args = args_json
        .replace('\n', " ")
        .replace('\r', " ")
        .replace('\t', " ");
    if clean_args.len() > 50 {
        let mut end = 47;
        while end > 0 && !clean_args.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}... ({})", &clean_args[..end], clean_args.len())
    } else {
        clean_args
    }
}

/// Returns a stream handler that prints `style`d text one line at a time; an
/// empty chunk flushes the unfinished line.
fn line_streamer(
    printer: ChatPrinter,
    style: Style,
) -> impl Fn(&str) -> Result<(), Box<dyn Error>> {
    let buffer = Mutex::new(String::new());
    move |chunk: &str| {
        if let Ok(mut buffer) = buffer.lock() {
            buffer.push_str(chunk);
            if chunk.is_empty() {
                if !buffer.is_empty() {
                    printer.println(&style.apply_to(&*buffer).to_string());
                    buffer.clear();
                }
            } else if buffer.contains('\n') {
                let mut lines: Vec<&str> = buffer.split('\n').collect();
                let remaining = lines.pop().unwrap_or("").to_string();
                for line in lines {
                    printer.println(&style.apply_to(line).to_string());
                }
                *buffer = remaining;
            }
        }
        Ok(())
    }
}

fn create_response_mode(
    printer: ChatPrinter,
    status_bar: Arc<status_bar::StatusBar>,
    agent_name: String,
) -> ResponseMode {
    let tool_active = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicBool::new(false));

    let answer = line_streamer(printer.clone(), Style::new().cyan());
    let tool_active_for_stream = tool_active.clone();

    let printer_for_progress = printer.clone();
    let tool_active_for_progress = tool_active;
    let completed_for_progress = completed;

    ResponseMode::Streaming {
        stream_handler: Box::new(move |chunk: &str| {
            if !chunk.is_empty() && tool_active_for_stream.load(Ordering::Relaxed) {
                return Ok(());
            }
            answer(chunk)
        }),
        reasoning_handler: Box::new(line_streamer(printer, Style::new().color256(244).italic())),
        progress_handler: Box::new(move |progress_info: &ProgressInfo| {
            if completed_for_progress.load(Ordering::Relaxed) {
                return Ok(());
            }
            match &progress_info.status {
                StatusUpdate::Thinking => {
                    status_bar.set_agent_status(&agent_name, "Thinking", true);
                }
                StatusUpdate::ToolAccumulating { name, arguments } => {
                    // The arguments are still being streamed in, so they're
                    // incomplete/invalid JSON at this point; showing them raw
                    // (as format_tool_arguments does once they're complete)
                    // would just look like a broken tool call. Show whatever
                    // key/value pairs have already arrived in full instead,
                    // falling back to a byte count until the first one lands.
                    let status = match preview_partial_tool_arguments(arguments) {
                        Some(preview) => {
                            format!("Preparing {}({})", name, format_tool_arguments(&preview))
                        }
                        None => format!("Preparing {} ({} chars)", name, arguments.len()),
                    };
                    status_bar.set_agent_status(&agent_name, &status, false);
                }
                StatusUpdate::ToolStart { name, arguments } => {
                    tool_active_for_progress.store(true, Ordering::Relaxed);
                    let formatted_args = format_tool_arguments(arguments);
                    status_bar.set_agent_status(
                        &agent_name,
                        &format!("Running {}({})", name, formatted_args),
                        true,
                    );
                }
                StatusUpdate::ToolComplete {
                    name,
                    arguments,
                    duration_ms,
                } => {
                    let duration_secs = *duration_ms as f64 / 1000.0;
                    tool_active_for_progress.store(false, Ordering::Relaxed);
                    let formatted_args = format_tool_arguments(arguments);
                    printer_for_progress.println(&format!(
                        "Tool {}({}) completed in {:.1}s",
                        name, formatted_args, duration_secs
                    ));
                }
                StatusUpdate::StreamProcessing {
                    bytes_read,
                    chunks_processed,
                } => {
                    status_bar.set_agent_status(
                        &agent_name,
                        &format!(
                            "Streaming ({} bytes, {} chunks)",
                            bytes_read, chunks_processed
                        ),
                        false,
                    );
                }
                StatusUpdate::Continuing => {
                    // Same wait as the initial request (blocked on the
                    // network for the model's next turn); keep the wording
                    // consistent instead of using a less descriptive label.
                    status_bar.set_agent_status(&agent_name, "Waiting for response", true);
                }
                StatusUpdate::Complete { usage } => {
                    completed_for_progress.store(true, Ordering::Relaxed);
                    let elapsed_secs = progress_info.elapsed_ms as f64 / 1000.0;
                    status_bar.clear_agent_status(&agent_name);

                    if let Some(usage) = usage {
                        let mut parts = Vec::new();
                        if let Some(input_tokens) = usage.prompt_tokens {
                            parts.push(format!("Input: {}", input_tokens));
                        }
                        if let Some(output_tokens) = usage.completion_tokens {
                            parts.push(format!("Output: {}", output_tokens));
                        }
                        if let Some(total_tokens) = usage.total_tokens {
                            parts.push(format!("Total: {}", total_tokens));
                        }
                        if !parts.is_empty() {
                            printer_for_progress.println(&format!(
                                "Complete | {} | {:.1}s",
                                parts.join(" > "),
                                elapsed_secs
                            ));
                        } else {
                            printer_for_progress
                                .println(&format!("Complete | {:.1}s", elapsed_secs));
                        }
                    } else {
                        printer_for_progress.println(&format!("Complete | {:.1}s", elapsed_secs));
                    }
                }
            }
            Ok(())
        }),
    }
}

fn execute_ai_request(
    messages: Vec<Message>,
    tools: &ToolsCollection,
    openai_opts: &openai::Opts,
    mode: ResponseMode,
    tool_context: &ToolContext,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
    signal_handler_active: &Arc<AtomicBool>,
    status_bar: &Arc<status_bar::StatusBar>,
    chat_pb: &ChatPrinter,
    agent_name: &str,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    signal_handler_active.store(true, Ordering::Relaxed);

    let response = match post_request_with_mode(
        messages,
        tools,
        openai_opts,
        mode,
        tool_context,
        ctrl_c_rx.clone(),
    ) {
        Ok(response) => {
            signal_handler_active.store(false, Ordering::Relaxed);
            if let Some(ref ctrl_c_rx) = ctrl_c_rx {
                if let Ok(receiver) = ctrl_c_rx.lock() {
                    while receiver.try_recv().is_ok() {}
                }
            }
            response
        }
        Err(e) => {
            signal_handler_active.store(false, Ordering::Relaxed);
            if let Some(ref ctrl_c_rx) = ctrl_c_rx {
                if let Ok(receiver) = ctrl_c_rx.lock() {
                    while receiver.try_recv().is_ok() {}
                }
            }
            status_bar.clear_agent_status(agent_name);
            if e.downcast_ref::<InterruptedError>().is_some() {
                chat_pb.println("Operation interrupted. Type your next message or /quit to exit.");
            }
            return Err(e);
        }
    };

    status_bar.clear_agent_status(agent_name);
    Ok(response)
}

/// Tells the user when the model stopped because it ran out of tokens rather
/// than because it was done, which otherwise looks like a complete answer.
fn warn_if_truncated(response: &OpenAIResponse, chat_pb: &ChatPrinter) {
    let finish_reason = response
        .choices
        .as_ref()
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.finish_reason.as_deref());
    if finish_reason == Some("length") {
        chat_pb.println(
            "Warning: the response was cut off (finish reason: length).  The token limit or \
             the context window was reached; try /summarize, /clear or fewer tools.",
        );
    }
}

fn save_agent_history(db: &Option<Arc<dyn DbBackend>>, agent: &AgentState) {
    if let Some(db) = db {
        let msgs: Vec<serde_json::Value> = agent
            .messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        let _ = db.save_agent_messages(&agent.name, &msgs);
    }
}

/// Replaces the agent's history with a summary of `source` (the current
/// history, or the one recovered from a failed request) and saves it.
fn summarize_agent_history(
    agent: &mut AgentState,
    source: &[Message],
    keep_last_user: bool,
    openai_opts: &openai::Opts,
    db: &Option<Arc<dyn DbBackend>>,
    ctrl_c_rx: &Arc<Mutex<mpsc::Receiver<()>>>,
    signal_handler_active: &Arc<AtomicBool>,
    status_bar: &Arc<status_bar::StatusBar>,
    chat_pb: &ChatPrinter,
) -> Result<(), Box<dyn Error>> {
    status_bar.set_agent_status(&agent.name, "Summarizing", true);
    signal_handler_active.store(true, Ordering::Relaxed);
    let result = summarize::summarize_conversation(
        source,
        keep_last_user,
        openai_opts,
        &Some(ctrl_c_rx.clone()),
    );
    signal_handler_active.store(false, Ordering::Relaxed);
    if let Ok(receiver) = ctrl_c_rx.lock() {
        while receiver.try_recv().is_ok() {}
    }
    status_bar.clear_agent_status(&agent.name);

    let summarized = result.map_err(|e| {
        if e.downcast_ref::<InterruptedError>().is_some() {
            chat_pb.println("Operation interrupted. Type your next message or /quit to exit.");
        }
        e
    })?;
    chat_pb.println(&format!(
        "Conversation summarized: {} messages -> {}.",
        source.len(),
        summarized.len()
    ));
    if let Some(summary) = summarized.iter().find(|m| summarize::is_summary(m)) {
        chat_pb.println(summary.content.as_deref().unwrap_or(""));
    }
    agent.messages = summarized;
    save_agent_history(db, agent);
    Ok(())
}

/// Runs `request` on a copy of the agent's history.  If the request fails
/// because the conversation no longer fits in the model's context window, the
/// history is summarized and the request is run once more.
fn request_with_summary_fallback<T>(
    agent: &mut AgentState,
    openai_opts: &openai::Opts,
    db: &Option<Arc<dyn DbBackend>>,
    ctrl_c_rx: &Arc<Mutex<mpsc::Receiver<()>>>,
    signal_handler_active: &Arc<AtomicBool>,
    status_bar: &Arc<status_bar::StatusBar>,
    chat_pb: &ChatPrinter,
    mut request: impl FnMut(Vec<Message>) -> Result<T, Box<dyn Error>>,
) -> Result<T, Box<dyn Error>> {
    let err = match request(agent.messages.clone()) {
        Err(e) => e,
        ok => return ok,
    };
    let history = match err.downcast_ref::<openai::ContextLengthError>() {
        Some(overflow) => overflow.history.clone(),
        None => return Err(err),
    };

    chat_pb.println("Context length exceeded, summarizing the conversation and retrying...");
    summarize_agent_history(
        agent,
        &history,
        true,
        openai_opts,
        db,
        ctrl_c_rx,
        signal_handler_active,
        status_bar,
        chat_pb,
    )
    .map_err(|e| -> Box<dyn Error> {
        if e.downcast_ref::<InterruptedError>().is_some() {
            e
        } else {
            format!("{} (summarization failed: {})", err, e).into()
        }
    })?;
    request(agent.messages.clone())
}

fn format_tool_output(output: &str) -> String {
    if let Ok(obj) = serde_json::from_str::<serde_json::Value>(output) {
        let mut parts = Vec::new();
        if let Some(stdout) = obj.get("stdout").and_then(|v| v.as_str()) {
            let s = stdout.trim();
            if !s.is_empty() {
                parts.push(s.to_string());
            }
        }
        if let Some(stderr) = obj.get("stderr").and_then(|v| v.as_str()) {
            let s = stderr.trim();
            if !s.is_empty() {
                parts.push(format!("stderr: {}", s));
            }
        }
        if parts.is_empty() {
            output.trim().to_string()
        } else {
            parts.join("\n")
        }
    } else {
        output.trim().to_string()
    }
}

fn execute_scheduled_command(
    command: &str,
    tools: &ToolsCollection,
    db: &Option<Arc<dyn DbBackend>>,
) -> Option<(Message, Message)> {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(command);
    let (tool_name, arguments) = match parsed {
        Ok(obj) => {
            let name = obj
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args = match obj.get("arguments") {
                Some(a) if a.is_string() => a.as_str().unwrap().to_string(),
                Some(a) => a.to_string(),
                None => "{}".to_string(),
            };
            (name, args)
        }
        Err(_) => return None,
    };

    if tool_name.is_empty() {
        return None;
    }

    let call_id = format!("scheduled-task-{}", chrono::Utc::now().timestamp_millis());

    let tc = ToolCall {
        index: None,
        id: call_id.clone(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: tool_name.clone(),
            arguments,
        },
    };

    let mut tool_context = ToolContext::new(|_: &str| {});
    tool_context.db = db.clone();

    let tool_msg = match tool_call(tools, &tc, &tool_context) {
        Ok(msg) => msg,
        Err(e) => Message {
            role: "tool".to_string(),
            content: Some(format!("error: {}", e)),
            tool_call_id: Some(call_id),
            name: Some(tool_name),
            tool_calls: None,
        },
    };

    let assistant_msg = Message {
        role: "assistant".to_string(),
        content: None,
        tool_call_id: None,
        name: None,
        tool_calls: Some(vec![tc]),
    };

    Some((assistant_msg, tool_msg))
}

fn scheduler_loop(
    db: Arc<dyn DbBackend>,
    tools: Arc<ToolsCollection>,
    tx: mpsc::Sender<(Option<String>, String, Message, Message)>,
) {
    let db_opt: Option<Arc<dyn DbBackend>> = Some(db.clone());
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let tasks = match db.get_pending_tasks() {
            Ok(t) => t,
            Err(_) => continue,
        };

        for task in tasks {
            let command = if task.command.is_empty() {
                task.description.clone()
            } else {
                task.command.clone()
            };

            if let Some((assistant_msg, tool_msg)) =
                execute_scheduled_command(&command, &tools, &db_opt)
            {
                let _ = tx.send((task.agent_name.clone(), command, assistant_msg, tool_msg));
            }

            let _ = db.mark_task_executed(
                task.id,
                &task.task_type,
                task.cron_expression.as_deref(),
                task.max_runs,
            );
        }
    }
}

/// Interactive session
fn chat_command(
    opts: &Opts,
    db: Option<Arc<dyn DbBackend>>,
    db_conn: Option<Arc<Mutex<rusqlite::Connection>>>,
    mcp_manager: Option<Arc<faber::mcp::McpManager>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Executing chat command");

    let status_bar = Arc::new(status_bar::StatusBar::new());
    let chat_pb = ChatPrinter::new();

    // Create rustyline editor with history
    let initial_agent_names = if let Some(ref db) = db {
        db.list_agents()?.iter().map(|a| a.name.clone()).collect()
    } else {
        vec!["default".to_string()]
    };
    let agent_names = Arc::new(Mutex::new(initial_agent_names));
    let helper = ChatHelper {
        agent_names: agent_names.clone(),
    };
    let config = rustyline::Config::builder().auto_add_history(true).build();
    let history = DbHistory::new(db_conn);
    let mut rl = Editor::with_history(config, history)?;
    rl.set_helper(Some(helper));

    let allowed_tools = if opts.tools.is_empty() {
        None
    } else {
        Some(opts.tools.clone())
    };
    let tools = match opts.no_tools {
        true => {
            debug!("Tools are disabled");
            ToolsCollection::new()
        }
        false => {
            debug!("Initializing tools for AI request");
            initialize_tools(opts.unsafe_tools, allowed_tools.as_deref())
        }
    };

    let session_id = format!(
        "{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_millis()
    );
    let active_subagents = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let initial_agent_name = opts.agent.clone().unwrap_or_else(|| "default".to_string());

    let agent_config = if let Some(ref db) = db {
        db.ensure_default_agent()?;
        if initial_agent_name != "default" {
            if db.get_agent(&initial_agent_name)?.is_none() {
                db.create_agent(&initial_agent_name, "")?;
            }
        }
        if !db.claim_agent(&initial_agent_name, &session_id)? {
            return Err(format!(
                "Agent '{}' is already claimed by another session.",
                initial_agent_name
            )
            .into());
        }
        db.get_agent_config(&initial_agent_name)?
    } else {
        db::AgentConfig {
            model: None,
            endpoint: None,
            system_prompt: None,
        }
    };

    let initial_messages = if let Some(ref db) = db {
        let vals = db.load_agent_messages(&initial_agent_name)?;
        let loaded: Vec<Message> = vals
            .into_iter()
            .map(|v| serde_json::from_value(v).unwrap())
            .collect();
        if loaded.is_empty() {
            initialize_agent_messages(&tools, opts, &agent_config)
        } else {
            loaded
        }
    } else {
        initialize_agent_messages(&tools, opts, &agent_config)
    };

    let mut active_agent = AgentState {
        name: initial_agent_name.clone(),
        messages: initial_messages,
    };
    status_bar.set_color(agent_ansi_code(&active_agent.name));

    let mut openai_opts = build_openai_opts(opts, &agent_config);
    debug!("Using model: {}", openai_opts.model);

    let prompt_text = Arc::new(Mutex::new(format!(
        "{}",
        agent_style(&initial_agent_name).apply_to(format!("{}> ", initial_agent_name))
    )));
    let session_id = Arc::new(session_id);

    let (ctrl_c_tx, ctrl_c_rx) = mpsc::channel();
    let ctrl_c_rx = Arc::new(Mutex::new(ctrl_c_rx));

    let signal_handler_active = Arc::new(AtomicBool::new(false));

    let ctrl_c_tx_clone = ctrl_c_tx.clone();
    let signal_handler_active_clone = signal_handler_active.clone();
    ctrlc::set_handler(move || {
        if signal_handler_active_clone.load(Ordering::Relaxed) {
            debug!("Received SIGINT (Ctrl-C) - sending interrupt signal");
            let _ = ctrl_c_tx_clone.send(());
        }
    })
    .expect("Error setting up Ctrl-C handler");

    let (task_tx, task_rx) = mpsc::channel::<(Option<String>, String, Message, Message)>();
    if let Some(ref scheduler_db) = db {
        let heartbeat_db = scheduler_db.clone();
        let heartbeat_session = session_id.clone();

        let scheduler_db = scheduler_db.clone();
        let scheduler_tools = Arc::new(tools.clone());
        std::thread::spawn(move || {
            scheduler_loop(scheduler_db, scheduler_tools, task_tx);
        });
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(5));
                let _ = heartbeat_db.heartbeat_all(&heartbeat_session);
            }
        });
    }

    let tools_arc = Arc::new(tools.clone());

    let (input_tx, input_rx) = mpsc::channel::<Result<String, rustyline::error::ReadlineError>>();
    let (ready_tx, ready_rx) = mpsc::channel::<()>();

    chat_pb.set_printer(Box::new(rl.create_external_printer()?));

    let prompt_text_clone = prompt_text.clone();
    std::thread::spawn(move || {
        loop {
            if ready_rx.recv().is_err() {
                break;
            }
            let prompt = prompt_text_clone
                .lock()
                .map(|p| p.clone())
                .unwrap_or_else(|_| "> ".to_string());
            let result = rl.readline(&prompt);
            let is_eof = matches!(result, Err(rustyline::error::ReadlineError::Eof));
            let _ = input_tx.send(result);
            if is_eof {
                break;
            }
        }
    });

    let mut prompt_shown = false;

    loop {
        while let Ok((task_agent, command, assistant_msg, tool_msg)) = task_rx.try_recv() {
            let target = task_agent.as_deref().unwrap_or("default");

            chat_pb.println_agent(target, &format!("Scheduled task fired: {}", command));
            if target != active_agent.name {
                if let Some(ref db) = db {
                    let assistant_val = serde_json::to_value(&assistant_msg).unwrap();
                    let tool_val = serde_json::to_value(&tool_msg).unwrap();
                    let _ = db.append_agent_message(target, &assistant_val);
                    let _ = db.append_agent_message(target, &tool_val);
                }
            }
            if let Some(ref output) = tool_msg.content {
                chat_pb.println_agent(target, &format_tool_output(output));
            }
            if target == active_agent.name {
                active_agent.messages.push(assistant_msg);
                active_agent.messages.push(tool_msg);
            }
        }

        let mut pending_injections: Vec<String> = Vec::new();
        if let Some(ref db) = db {
            if let Ok(notifications) = db.poll_notifications_for_session(&session_id) {
                for notif in notifications {
                    let display_msg = if let Ok(obj) =
                        serde_json::from_str::<serde_json::Value>(&notif.message)
                    {
                        if obj.get("type").and_then(|v| v.as_str()) == Some("subagent_result") {
                            if let Some(agent) = obj.get("agent").and_then(|v| v.as_str()) {
                                status_bar.clear_agent_status(agent);
                                let _ = db.delete_agent(agent);
                                if let Ok(mut names) = agent_names.lock() {
                                    names.retain(|n| n != agent);
                                }
                            }
                            obj.get("response")
                                .and_then(|v| v.as_str())
                                .unwrap_or(&notif.message)
                                .to_string()
                        } else {
                            notif.message.clone()
                        }
                    } else {
                        notif.message.clone()
                    };
                    chat_pb.println_agent(&notif.from_agent, &display_msg);
                    pending_injections.push(format!(
                        "[Message from agent '{}']: {}",
                        notif.from_agent, notif.message
                    ));
                }
            }
        }
        let pending_injection: Option<String> = if pending_injections.is_empty() {
            None
        } else {
            Some(pending_injections.join("\n"))
        };

        let is_injected = pending_injection.is_some();
        let line = if let Some(injected) = pending_injection {
            injected
        } else {
            if !prompt_shown {
                let count = active_subagents.load(std::sync::atomic::Ordering::Relaxed);
                if count > 0 {
                    chat_pb.println(&format!("  {} subagent(s) running in background...", count));
                }
                chat_pb.set_direct_mode(false);
                status_bar.pause();
                let _ = ready_tx.send(());
                prompt_shown = true;
            }

            match input_rx.recv_timeout(Duration::from_millis(200)) {
                Ok(Ok(line)) => {
                    status_bar.resume();
                    chat_pb.set_direct_mode(true);
                    line.trim().to_string()
                }
                Ok(Err(rustyline::error::ReadlineError::Interrupted)) => {
                    status_bar.resume();
                    chat_pb.set_direct_mode(true);
                    prompt_shown = false;
                    continue;
                }
                Ok(Err(rustyline::error::ReadlineError::Eof)) => {
                    status_bar.resume();
                    chat_pb.set_direct_mode(true);
                    if let Some(ref db) = db {
                        let _ = db.release_all_agents(&session_id);
                    }
                    return Ok(());
                }
                Ok(Err(err)) => {
                    status_bar.resume();
                    chat_pb.set_direct_mode(true);
                    return Err(Box::new(err));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    status_bar.resume();
                    if let Some(ref db) = db {
                        let _ = db.release_all_agents(&session_id);
                    }
                    return Ok(());
                }
            }
        };

        if !is_injected {
            prompt_shown = false;
        }

        debug!("User input: '{}' (length: {})", line, line.len());

        let command = parse_chat_command(&line);
        let handled = match handle_chat_command(
            command,
            &mut active_agent,
            &tools,
            opts,
            &chat_pb,
            &db,
            &mut openai_opts,
            &prompt_text,
            &agent_names,
            &session_id,
            &mcp_manager,
            &status_bar,
        ) {
            Ok(handled) => handled,
            Err(e) => {
                chat_pb.println(&format!("Error: {}", e));
                continue;
            }
        };
        match handled {
            true => continue,
            false => {
                if let ChatCommand::Quit = parse_chat_command(&line) {
                    if let Some(ref db) = db {
                        let msgs: Vec<serde_json::Value> = active_agent
                            .messages
                            .iter()
                            .map(|m| serde_json::to_value(m).unwrap())
                            .collect();
                        let _ = db.save_agent_messages(&active_agent.name, &msgs);
                        let _ = db.release_all_agents(&session_id);
                    }
                    return Ok(());
                }
                if let ChatCommand::Message(user_message) = parse_chat_command(&line) {
                    active_agent
                        .messages
                        .push(make_message("user", user_message));

                    let mut tool_context = ToolContext::new(|_: &str| {});
                    tool_context.db = db.clone();
                    tool_context.agent_name = Some(active_agent.name.clone());
                    tool_context.mcp_manager = mcp_manager.clone();
                    tool_context.extra = Some(Arc::new(SubAgentContext {
                        tools: tools_arc.clone(),
                        opts: openai_opts.clone(),
                        session_id: session_id.to_string(),
                        active_subagents: active_subagents.clone(),
                        status_bar: status_bar.clone(),
                    }));

                    if is_injected {
                        match request_with_summary_fallback(
                            &mut active_agent,
                            &openai_opts,
                            &db,
                            &ctrl_c_rx,
                            &signal_handler_active,
                            &status_bar,
                            &chat_pb,
                            |messages| {
                                post_request_with_mode(
                                    messages,
                                    &tools,
                                    &openai_opts,
                                    ResponseMode::Complete,
                                    &tool_context,
                                    None,
                                )
                            },
                        ) {
                            Ok(response) => {
                                warn_if_truncated(&response, &chat_pb);
                                active_agent.messages = response.history;
                                if let Some(ref choices) = response.choices {
                                    if let Some(content) =
                                        choices.first().and_then(|c| c.message.content.as_ref())
                                    {
                                        chat_pb.println_agent(&active_agent.name, content);
                                    }
                                }
                                save_agent_history(&db, &active_agent);
                            }
                            Err(e) => {
                                if e.downcast_ref::<InterruptedError>().is_none() {
                                    chat_pb
                                        .println(&format!("Error processing notification: {}", e));
                                }
                            }
                        }
                    } else {
                        let printer_for_tool = chat_pb.clone();
                        tool_context.println = Box::new(move |msg: &str| {
                            printer_for_tool.println(msg);
                        });

                        let agent_name = active_agent.name.clone();
                        match request_with_summary_fallback(
                            &mut active_agent,
                            &openai_opts,
                            &db,
                            &ctrl_c_rx,
                            &signal_handler_active,
                            &status_bar,
                            &chat_pb,
                            |messages| {
                                let mode = create_response_mode(
                                    chat_pb.clone(),
                                    status_bar.clone(),
                                    agent_name.clone(),
                                );
                                status_bar.set_agent_status(
                                    &agent_name,
                                    "Waiting for response",
                                    true,
                                );
                                execute_ai_request(
                                    messages,
                                    &tools,
                                    &openai_opts,
                                    mode,
                                    &tool_context,
                                    Some(ctrl_c_rx.clone()),
                                    &signal_handler_active,
                                    &status_bar,
                                    &chat_pb,
                                    &agent_name,
                                )
                            },
                        ) {
                            Ok(response) => {
                                warn_if_truncated(&response, &chat_pb);
                                active_agent.messages = response.history;
                                save_agent_history(&db, &active_agent);
                            }
                            Err(e) => {
                                if e.downcast_ref::<InterruptedError>().is_none() {
                                    chat_pb.println(&format!("Error: {}", e));
                                }
                                continue;
                            }
                        }
                    }
                }
                if let ChatCommand::Summarize = parse_chat_command(&line) {
                    let history = active_agent.messages.clone();
                    if let Err(e) = summarize_agent_history(
                        &mut active_agent,
                        &history,
                        false,
                        &openai_opts,
                        &db,
                        &ctrl_c_rx,
                        &signal_handler_active,
                        &status_bar,
                        &chat_pb,
                    ) {
                        if e.downcast_ref::<InterruptedError>().is_none() {
                            chat_pb.println(&format!("Error: {}", e));
                        }
                    }
                }
            }
        }
    }
}

// ModelInfo and ModelsApiResponse structs are removed from here.

fn gc_command(opts: &Opts) -> Result<(), Box<dyn Error>> {
    let db_path = opts
        .db_path
        .as_ref()
        .ok_or("No db_path configured. Set 'db_path' in your config file.")?;
    let conn = rusqlite::Connection::open(db_path)?;
    db::initialize_db(&conn)?;
    let db = LocalDb::new(Arc::new(Mutex::new(conn)));
    let removed = db.gc_agents()?;
    if removed.is_empty() {
        println!("No dormant agents to clean up.");
    } else {
        println!("Removed {} agent(s):", removed.len());
        for name in &removed {
            println!("  {}", name);
        }
    }
    Ok(())
}

fn list_tools_command(
    mcp_manager: &Option<Arc<faber::mcp::McpManager>>,
) -> Result<(), Box<dyn Error>> {
    let safe_tools = initialize_tools(false, None);
    let all_tools = initialize_tools(true, None);

    let mut names: Vec<&String> = all_tools.keys().collect();
    names.sort();

    for name in names {
        let marker = if safe_tools.contains_key(name) {
            ""
        } else {
            " [unsafe]"
        };
        println!("  {}{}", name, marker);
    }

    if let Some(mcp) = mcp_manager {
        let schemas = mcp.get_tool_schemas();
        if !schemas.is_empty() {
            println!("MCP tools:");
            let mut mcp_names: Vec<String> = schemas
                .iter()
                .filter_map(|s| {
                    s.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .map(|n| n.to_string())
                })
                .collect();
            mcp_names.sort();
            for name in &mcp_names {
                println!("  {}", name);
            }
        }
    }

    Ok(())
}

fn list_models_command(opts: &Opts) -> Result<(), Box<dyn Error>> {
    let base_endpoint = opts
        .endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());

    let models_endpoint = if base_endpoint.ends_with("/chat/completions") {
        base_endpoint.replace("/chat/completions", "/models")
    } else if base_endpoint.ends_with("/") {
        format!("{}models", base_endpoint)
    } else {
        format!("{}/models", base_endpoint)
    };

    match list_models_from_endpoint(&models_endpoint, opts.api_key.as_ref()) {
        Ok(models) => {
            if models.is_empty() {
                println!("No models found.");
            } else {
                let source = if opts.endpoint.is_some() {
                    "custom endpoint"
                } else {
                    "default endpoint"
                };
                println!("Available models from {}:", source);
                let mut table = Table::new();
                table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
                table.set_titles(Row::new(vec![
                    Cell::new("ID"),
                    Cell::new("Name"),
                    Cell::new("Context Length"),
                    Cell::new("Prompt_USD/1M"),
                    Cell::new("Compl_USD/1M"),
                    Cell::new("Supported Parameters"),
                ]));

                for model in models {
                    let prompt_price_str = model
                        .pricing
                        .as_ref()
                        .map(|p| p.prompt.as_str())
                        .unwrap_or("N/A");
                    let completion_price_str = model
                        .pricing
                        .as_ref()
                        .map(|p| p.completion.as_str())
                        .unwrap_or("N/A");

                    let prompt_price = match prompt_price_str.parse::<f64>() {
                        Ok(p) => format!("{:.6}", p),
                        Err(_) => prompt_price_str.to_string(),
                    };
                    let completion_price = match completion_price_str.parse::<f64>() {
                        Ok(c) => format!("{:.6}", c),
                        Err(_) => completion_price_str.to_string(),
                    };
                    let supported_parameters = model
                        .supported_parameters
                        .unwrap_or_else(|| vec![])
                        .join(",");

                    table.add_row(Row::new(vec![
                        Cell::new(&model.id),
                        Cell::new(&model.name.unwrap_or("".to_string())),
                        Cell::new(&model.context_length.unwrap_or(0).to_string()),
                        Cell::new(&prompt_price),
                        Cell::new(&completion_price),
                        Cell::new(&supported_parameters),
                    ]));
                }
                table.printstd();
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[derive(Parser, Debug, Serialize, Deserialize)]
#[clap(version = env!("CARGO_PKG_VERSION"))]
#[serde(default)]
struct Opts {
    #[clap(short = 'c', long = "config")]
    #[serde(skip)]
    /// Path to JSON configuration file
    ///
    /// If not specified, will automatically use 'config.json'
    /// from current directory if it exists.
    ///
    /// Example config file:
    /// {
    ///   "model": "granite",
    ///   "api_key": "~/.path/to/key",
    ///   "parameters": ["temperature=0.7"]
    /// }
    ///
    /// CLI arguments override config file values.
    config: Option<String>,
    #[clap(short, long)]
    /// Override the maximum number of tokens to generate
    max_tokens: Option<u32>,
    #[clap(long)]
    /// Specify the AI model to use
    model: Option<String>,
    #[clap(long)]
    /// Override the endpoint URL to use
    endpoint: Option<String>,
    #[clap(long)]
    /// Inhibit usage of any tool
    no_tools: bool,
    /// Enable unsafe tools
    #[clap(long)]
    unsafe_tools: bool,
    #[clap(long, value_delimiter = ',')]
    /// Only enable the specified tools (comma-separated). Overrides --unsafe-tools
    tools: Vec<String>,
    #[clap(long)]
    /// Control when tools are used: "auto" (default), "none", "required"
    tool_choice: Option<String>,
    #[clap(long)]
    /// Override the file path to read API key from
    api_key: Option<String>,
    #[clap(long)]
    /// Set model parameters in NAME=VALUE format (e.g., --parameter temperature=0.7 --parameter top_p=0.9)
    parameter: Vec<String>,
    #[clap(long)]
    /// Path to the SQLite database file for persistent storage (agents, tasks, memory)
    db_path: Option<String>,
    #[clap(long)]
    /// Start chat session with this agent instead of 'default'
    agent: Option<String>,
    #[clap(long)]
    /// Connect to a remote faber server instead of using a local database
    server: Option<String>,
    #[clap(long)]
    /// Pre-shared key for server authentication
    server_key: Option<String>,
    #[clap(long)]
    /// Read server key from file (first line)
    server_key_file: Option<String>,

    #[clap(skip)]
    #[serde(default)]
    mcp_servers: HashMap<String, faber::mcp::McpServerConfig>,

    #[clap(subcommand)]
    #[serde(skip)]
    command: CliCommand,

    #[clap()]
    #[serde(skip)]
    args: Vec<String>,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            config: None,
            max_tokens: None,
            model: None,
            endpoint: None,
            no_tools: false,
            unsafe_tools: false,
            tools: Vec::new(),
            tool_choice: None,
            api_key: None,
            parameter: Vec::new(),
            db_path: None,
            agent: None,
            server: None,
            server_key: None,
            server_key_file: None,
            mcp_servers: HashMap::new(),
            command: CliCommand::Chat {},
            args: Vec::new(),
        }
    }
}

impl Opts {
    /// Load configuration from a JSON file
    fn load_from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        debug!("Loading configuration from file: {}", path);

        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file '{}': {}", path, e))?;

        let config: Opts = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse config file '{}': {}", path, e))?;

        debug!("Successfully loaded configuration from {}", path);
        Ok(config)
    }

    /// Merge this config with CLI options, giving precedence to CLI options
    fn merge_with_config(&mut self, config: Opts) {
        debug!("Merging configuration file with CLI options");

        // Only use config values if CLI didn't provide them
        if self.max_tokens.is_none() {
            self.max_tokens = config.max_tokens;
        }

        if self.model.is_none() {
            self.model = config.model;
        }

        if self.endpoint.is_none() {
            self.endpoint = config.endpoint;
        }

        if !self.no_tools && config.no_tools {
            self.no_tools = true;
        }

        if !self.unsafe_tools && config.unsafe_tools {
            self.unsafe_tools = true;
        }

        if self.tool_choice.is_none() {
            self.tool_choice = config.tool_choice;
        }

        if self.api_key.is_none() {
            self.api_key = config.api_key;
        }

        // Merge parameters - config parameters are added first, then CLI parameters
        if !config.parameter.is_empty() {
            let mut merged_params = config.parameter;
            merged_params.extend(self.parameter.clone());
            self.parameter = merged_params;
        }

        if self.db_path.is_none() {
            self.db_path = config.db_path;
        }

        if self.server.is_none() {
            self.server = config.server;
        }

        if self.server_key.is_none() {
            self.server_key = config.server_key;
        }

        if self.server_key_file.is_none() {
            self.server_key_file = config.server_key_file;
        }

        if self.mcp_servers.is_empty() {
            self.mcp_servers = config.mcp_servers;
        }

        debug!("Configuration merge completed");
    }
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Pass a request to the AI model and print its response
    Prompt {
        /// Prompt command to pass to the AI model
        prompt: String,
        /// List of files that are loaded and used as system context
        files: Vec<String>,
    },

    /// Interactive session
    Chat {},

    /// List available models from the configured endpoint
    Models {},

    /// List all available tools
    ListTools {},

    /// Remove dormant agents (no active session)
    Gc {},

    /// Start a server exposing the DB API over TCP
    Serve {
        /// Address to bind to
        #[clap(long, default_value = "127.0.0.1:9090")]
        bind: String,
        /// Pre-shared key for client authentication
        #[clap(long)]
        auth_key: Option<String>,
        /// Read auth key from file (first line)
        #[clap(long)]
        auth_key_file: Option<String>,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let env = Env::new()
        .filter_or("RUST_LOG", "warning")
        .write_style_or("LOG_STYLE", "always");
    env_logger::Builder::from_env(env).init();

    // Parse command line arguments
    let mut opts = Opts::parse();
    debug!("Command line options parsed");

    // Load and merge configuration file
    let config_path = match &opts.config {
        Some(path) => Some(path.clone()),
        None => {
            // Check for default config.json in current directory
            let default_config = "config.json";
            if std::path::Path::new(default_config).exists() {
                debug!("Found default config file: {}", default_config);
                Some(default_config.to_string())
            } else {
                None
            }
        }
    };

    if let Some(config_file) = config_path {
        match Opts::load_from_file(&config_file) {
            Ok(config) => {
                debug!("Loaded configuration file: {}", config_file);
                opts.merge_with_config(config);
            }
            Err(e) => {
                return Err(format!("Configuration file error: {}", e).into());
            }
        }
    }

    // Reset the model to use if an endpoint was provided
    if opts.model.is_none() && opts.endpoint.is_some() {
        opts.model = Some("".to_string());
    }

    let mut db_conn_for_history: Option<Arc<Mutex<rusqlite::Connection>>> = None;
    let db_connection: Option<Arc<dyn DbBackend>> = if let Some(ref server_addr) = opts.server {
        debug!("Connecting to remote server at: {}", server_addr);
        let remote = remote_db::RemoteDb::connect(server_addr)?;
        let key = match (&opts.server_key, &opts.server_key_file) {
            (Some(k), _) => Some(k.clone()),
            (_, Some(f)) => {
                let content = std::fs::read_to_string(f)?;
                Some(content.lines().next().unwrap_or("").to_string())
            }
            _ => None,
        };
        if let Some(ref k) = key {
            remote.authenticate(k)?;
        }
        Some(Arc::new(remote))
    } else if let Some(ref db_path) = opts.db_path {
        debug!("Opening SQLite database at: {}", db_path);
        let conn = rusqlite::Connection::open(db_path)?;
        db::initialize_db(&conn)?;
        let conn = Arc::new(Mutex::new(conn));
        db_conn_for_history = Some(conn.clone());
        Some(Arc::new(LocalDb::new(conn)))
    } else {
        debug!("No db_path configured, database tools will be unavailable");
        None
    };

    let mcp_manager: Option<Arc<faber::mcp::McpManager>> = if !opts.mcp_servers.is_empty() {
        match faber::mcp::McpManager::new(opts.mcp_servers.clone()) {
            Ok(mgr) => Some(Arc::new(mgr)),
            Err(e) => {
                return Err(format!("Failed to initialize MCP servers: {}", e).into());
            }
        }
    } else {
        None
    };

    // Execute the chosen command
    let result = match &opts.command {
        CliCommand::Prompt { prompt, files } => prompt_command(
            &prompt,
            &files,
            &opts,
            db_connection.clone(),
            mcp_manager.clone(),
        ),
        CliCommand::Chat {} => chat_command(
            &opts,
            db_connection.clone(),
            db_conn_for_history.clone(),
            mcp_manager.clone(),
        ),
        CliCommand::Models {} => list_models_command(&opts),
        CliCommand::ListTools {} => list_tools_command(&mcp_manager),
        CliCommand::Gc {} => gc_command(&opts),
        CliCommand::Serve {
            bind,
            auth_key,
            auth_key_file,
        } => {
            let db_path = opts
                .db_path
                .as_ref()
                .ok_or("--db-path is required for serve command")?;
            let key = match (auth_key, auth_key_file) {
                (Some(k), _) => Some(k.clone()),
                (_, Some(f)) => {
                    let content = std::fs::read_to_string(f)?;
                    Some(content.lines().next().unwrap_or("").to_string())
                }
                _ => None,
            };
            server::serve_command(bind, db_path, key.as_deref())
        }
    };

    if let Some(ref mcp) = mcp_manager {
        mcp.shutdown();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyline::history::{History, SearchDirection};

    #[test]
    fn test_parse_parameters_number() {
        let params = parse_parameters(&["temperature=0.7".to_string()]).unwrap();
        assert_eq!(params["temperature"], serde_json::json!(0.7));
    }

    #[test]
    fn test_parse_parameters_integer() {
        let params = parse_parameters(&["count=42".to_string()]).unwrap();
        assert_eq!(params["count"], serde_json::json!(42.0));
    }

    #[test]
    fn test_parse_parameters_bool() {
        let params = parse_parameters(&["stream=true".to_string()]).unwrap();
        assert_eq!(params["stream"], serde_json::json!(true));
    }

    #[test]
    fn test_parse_parameters_null() {
        let params = parse_parameters(&["val=null".to_string()]).unwrap();
        assert_eq!(params["val"], serde_json::Value::Null);
    }

    #[test]
    fn test_parse_parameters_string() {
        let params = parse_parameters(&["name=hello world".to_string()]).unwrap();
        assert_eq!(params["name"], serde_json::json!("hello world"));
    }

    #[test]
    fn test_parse_parameters_nan_becomes_string() {
        let params = parse_parameters(&["val=NaN".to_string()]).unwrap();
        assert_eq!(params["val"], serde_json::json!("NaN"));
    }

    #[test]
    fn test_parse_parameters_infinity_becomes_string() {
        let params = parse_parameters(&["val=inf".to_string()]).unwrap();
        assert_eq!(params["val"], serde_json::json!("inf"));

        let params = parse_parameters(&["val=Infinity".to_string()]).unwrap();
        assert_eq!(params["val"], serde_json::json!("Infinity"));
    }

    #[test]
    fn test_parse_parameters_invalid_format() {
        let result = parse_parameters(&["noequals".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_parameters_empty() {
        let params = parse_parameters(&[]).unwrap();
        assert!(params.is_empty());
    }

    #[test]
    fn test_parse_chat_command_help() {
        assert!(matches!(parse_chat_command("/help"), ChatCommand::Help));
    }

    #[test]
    fn test_parse_chat_command_quit() {
        assert!(matches!(parse_chat_command("/quit"), ChatCommand::Quit));
    }

    #[test]
    fn test_parse_chat_command_clear() {
        assert!(matches!(parse_chat_command("/clear"), ChatCommand::Clear));
    }

    #[test]
    fn test_parse_chat_command_show() {
        assert!(matches!(parse_chat_command("/show"), ChatCommand::Show));
    }

    #[test]
    fn test_parse_chat_command_limit() {
        match parse_chat_command("/limit 5") {
            ChatCommand::Limit(n) => assert_eq!(n, 5),
            _ => panic!("expected Limit"),
        }
    }

    #[test]
    fn test_parse_chat_command_limit_invalid() {
        assert!(matches!(
            parse_chat_command("/limit abc"),
            ChatCommand::Invalid(_)
        ));
    }

    #[test]
    fn test_parse_chat_command_backtrace() {
        match parse_chat_command("/backtrace 3") {
            ChatCommand::Backtrace(n) => assert_eq!(n, 3),
            _ => panic!("expected Backtrace"),
        }
    }

    #[test]
    fn test_parse_chat_command_system() {
        match parse_chat_command("/system be nice") {
            ChatCommand::System(msg) => assert_eq!(msg, "be nice"),
            _ => panic!("expected System"),
        }
    }

    #[test]
    fn test_parse_chat_command_agents() {
        assert!(matches!(parse_chat_command("/agents"), ChatCommand::Agents));
    }

    #[test]
    fn test_parse_chat_command_create_agent() {
        match parse_chat_command("/create-agent bob") {
            ChatCommand::CreateAgent(name) => assert_eq!(name, "bob"),
            _ => panic!("expected CreateAgent"),
        }
    }

    #[test]
    fn test_parse_chat_command_select_agent() {
        match parse_chat_command("/select-agent bob") {
            ChatCommand::SelectAgent(name) => assert_eq!(name, "bob"),
            _ => panic!("expected SelectAgent"),
        }
    }

    #[test]
    fn test_parse_chat_command_delete_agent() {
        match parse_chat_command("/delete-agent bob") {
            ChatCommand::DeleteAgent(name) => assert_eq!(name, "bob"),
            _ => panic!("expected DeleteAgent"),
        }
    }

    #[test]
    fn test_parse_chat_command_mcp_refresh() {
        assert!(matches!(
            parse_chat_command("/mcp-refresh"),
            ChatCommand::McpRefresh
        ));
    }

    #[test]
    fn test_parse_chat_command_tools() {
        assert!(matches!(parse_chat_command("/tools"), ChatCommand::Tools));
    }

    #[test]
    fn test_parse_chat_command_message() {
        match parse_chat_command("hello world") {
            ChatCommand::Message(msg) => assert_eq!(msg, "hello world"),
            _ => panic!("expected Message"),
        }
    }

    #[test]
    fn test_parse_chat_command_empty() {
        assert!(matches!(parse_chat_command(""), ChatCommand::Empty));
    }

    #[test]
    fn test_parse_chat_command_unknown() {
        assert!(matches!(
            parse_chat_command("/unknown"),
            ChatCommand::Invalid(_)
        ));
    }

    #[test]
    fn test_parse_chat_command_backslash_prefix() {
        assert!(matches!(parse_chat_command("\\help"), ChatCommand::Help));
    }

    #[test]
    fn test_format_tool_arguments_short() {
        assert_eq!(
            format_tool_arguments(r#"{"path":"foo"}"#),
            r#"{"path":"foo"}"#
        );
    }

    #[test]
    fn test_format_tool_arguments_long_truncated() {
        let long_args = "a".repeat(100);
        let result = format_tool_arguments(&long_args);
        assert!(result.contains("..."));
        assert!(result.contains("(100)"));
    }

    #[test]
    fn test_format_tool_arguments_newlines_removed() {
        let args = "line1\nline2\tline3";
        let result = format_tool_arguments(args);
        assert!(!result.contains('\n'));
        assert!(!result.contains('\t'));
    }

    #[test]
    fn test_preview_partial_tool_arguments_nothing_yet() {
        assert_eq!(preview_partial_tool_arguments(""), None);
        assert_eq!(preview_partial_tool_arguments("{"), None);
        assert_eq!(preview_partial_tool_arguments(r#"{"pa"#), None);
        assert_eq!(preview_partial_tool_arguments("not json"), None);
    }

    #[test]
    fn test_preview_partial_tool_arguments_incomplete_string_value() {
        // The key is complete but the value's closing quote hasn't arrived.
        assert_eq!(preview_partial_tool_arguments(r#"{"path":"Cargo.t"#), None);
    }

    #[test]
    fn test_preview_partial_tool_arguments_single_complete_pair() {
        // Complete even though the outer object hasn't closed yet - this is
        // the common single-argument case (e.g. read_file) where waiting
        // for a top-level comma would never show anything.
        assert_eq!(
            preview_partial_tool_arguments(r#"{"path":"Cargo.toml"#),
            None
        );
        assert_eq!(
            preview_partial_tool_arguments(r#"{"path":"Cargo.toml""#),
            Some(r#"path="Cargo.toml""#.to_string())
        );
    }

    #[test]
    fn test_preview_partial_tool_arguments_keeps_earlier_pairs_when_later_one_is_incomplete() {
        let partial = r#"{"path":"Cargo.toml","old_content":"fo"#;
        assert_eq!(
            preview_partial_tool_arguments(partial),
            Some(r#"path="Cargo.toml""#.to_string())
        );
    }

    #[test]
    fn test_preview_partial_tool_arguments_multiple_complete_pairs() {
        let complete = r#"{"path":"Cargo.toml","replace_all":true}"#;
        assert_eq!(
            preview_partial_tool_arguments(complete),
            Some(r#"path="Cargo.toml", replace_all=true"#.to_string())
        );
    }

    #[test]
    fn test_preview_partial_tool_arguments_number_bool_null() {
        assert_eq!(
            preview_partial_tool_arguments(r#"{"n":42,"ok":false,"x":null,"#),
            Some("n=42, ok=false, x=null".to_string())
        );
        // A number isn't complete just because the buffer ends right after
        // it - more digits could still be coming.
        assert_eq!(preview_partial_tool_arguments(r#"{"n":4"#), None);
    }

    #[test]
    fn test_preview_partial_tool_arguments_nested_value() {
        let partial = r#"{"options":{"a":1,"b":2},"path":"x"#;
        assert_eq!(
            preview_partial_tool_arguments(partial),
            Some(r#"options={"a":1,"b":2}"#.to_string())
        );
    }

    #[test]
    fn test_preview_partial_tool_arguments_escaped_quote_in_string() {
        let args = r#"{"content":"say \"hi\""}"#;
        assert_eq!(
            preview_partial_tool_arguments(args),
            Some(r#"content="say \"hi\"""#.to_string())
        );
    }

    #[test]
    fn test_preview_partial_tool_arguments_never_produces_multiple_lines() {
        // A value with an embedded real newline (e.g. a pretty-printed
        // nested object, or an unescaped control character some server
        // tolerates) must never turn into an actual line break on the
        // status line.
        let with_newline_in_nested_value = "{\"options\":{\"a\":\n1}}";
        let preview = preview_partial_tool_arguments(with_newline_in_nested_value).unwrap();
        assert!(
            !preview.contains('\n'),
            "preview contained a newline: {:?}",
            preview
        );

        let with_raw_newline_in_string = "{\"content\":\"line1\nline2\"}";
        if let Some(preview) = preview_partial_tool_arguments(with_raw_newline_in_string) {
            assert!(
                !preview.contains('\n'),
                "preview contained a newline: {:?}",
                preview
            );
        }
    }

    #[test]
    fn test_agent_color_hash_deterministic() {
        let hash1 = agent_color_hash("alice");
        let hash2 = agent_color_hash("alice");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_agent_color_hash_different_agents() {
        let hash1 = agent_color_hash("alice");
        let hash2 = agent_color_hash("bob");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_agent_ansi_code_in_range() {
        for name in &["alice", "bob", "charlie", "default", "xyz"] {
            let code = agent_ansi_code(name);
            assert!(AGENT_ANSI_CODES.contains(&code));
        }
    }

    fn test_db_conn() -> Arc<Mutex<rusqlite::Connection>> {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        db::initialize_db(&conn).unwrap();
        Arc::new(Mutex::new(conn))
    }

    #[test]
    fn test_db_history_add_and_get() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("first").unwrap();
        hist.add("second").unwrap();
        let result = hist.get(0, SearchDirection::Forward).unwrap().unwrap();
        assert_eq!(result.entry.as_ref(), "first");
        let result = hist.get(1, SearchDirection::Forward).unwrap().unwrap();
        assert_eq!(result.entry.as_ref(), "second");
    }

    #[test]
    fn test_db_history_len() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        assert_eq!(hist.len(), 0);
        assert!(hist.is_empty());
        hist.add("first").unwrap();
        assert_eq!(hist.len(), 1);
        assert!(!hist.is_empty());
    }

    #[test]
    fn test_db_history_ignore_empty() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("").unwrap();
        assert_eq!(hist.len(), 0);
    }

    #[test]
    fn test_db_history_ignore_space() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.ignore_space(true);
        hist.add(" hidden").unwrap();
        assert_eq!(hist.len(), 0);
    }

    #[test]
    fn test_db_history_ignore_dups() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("same").unwrap();
        hist.add("same").unwrap();
        assert_eq!(hist.len(), 1);
    }

    #[test]
    fn test_db_history_search() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("cargo build").unwrap();
        hist.add("cargo test").unwrap();
        hist.add("git status").unwrap();

        let result = hist
            .search("cargo", 0, SearchDirection::Forward)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "cargo build");

        let result = hist
            .search("cargo", 2, SearchDirection::Reverse)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "cargo test");
    }

    #[test]
    fn test_db_history_search_with_wildcards() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("test_file").unwrap();
        hist.add("testXfile").unwrap();

        let result = hist
            .search("test_file", 0, SearchDirection::Forward)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "test_file");
    }

    #[test]
    fn test_db_history_starts_with() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("cargo build").unwrap();
        hist.add("cargo test").unwrap();
        hist.add("git status").unwrap();

        let result = hist
            .starts_with("cargo", 0, SearchDirection::Forward)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "cargo build");

        let result = hist
            .starts_with("git", 0, SearchDirection::Forward)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "git status");
    }

    #[test]
    fn test_db_history_starts_with_wildcard_in_term() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("100% done").unwrap();
        hist.add("something else").unwrap();

        let result = hist
            .starts_with("100%", 0, SearchDirection::Forward)
            .unwrap()
            .unwrap();
        assert_eq!(result.entry.as_ref(), "100% done");
    }

    #[test]
    fn test_db_history_clear() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("first").unwrap();
        hist.add("second").unwrap();
        hist.clear().unwrap();
        assert_eq!(hist.len(), 0);
    }

    #[test]
    fn test_db_history_max_len() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.set_max_len(3).unwrap();
        for i in 0..5 {
            hist.add(&format!("entry{}", i)).unwrap();
        }
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_db_history_reverse_get() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("first").unwrap();
        hist.add("second").unwrap();
        hist.add("third").unwrap();

        let result = hist.get(2, SearchDirection::Reverse).unwrap().unwrap();
        assert_eq!(result.entry.as_ref(), "third");
    }

    #[test]
    fn test_db_history_mem_fallback() {
        let mut hist = DbHistory::new(None);
        hist.add("memory only").unwrap();
        assert_eq!(hist.len(), 1);
        let result = hist.get(0, SearchDirection::Forward).unwrap().unwrap();
        assert_eq!(result.entry.as_ref(), "memory only");
    }

    #[test]
    fn test_db_history_search_not_found() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("hello").unwrap();
        let result = hist
            .search("nonexistent", 0, SearchDirection::Forward)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_db_history_search_empty_term() {
        let conn = test_db_conn();
        let mut hist = DbHistory::new(Some(conn));
        hist.add("hello").unwrap();
        let result = hist.search("", 0, SearchDirection::Forward).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_format_tool_output_json() {
        let output = r#"{"stdout":"hello\n","stderr":"","exit_code":0,"success":true}"#;
        let result = format_tool_output(output);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_format_tool_output_json_with_stderr() {
        let output = r#"{"stdout":"out","stderr":"err","exit_code":1,"success":false}"#;
        let result = format_tool_output(output);
        assert!(result.contains("out"));
        assert!(result.contains("stderr: err"));
    }

    #[test]
    fn test_format_tool_output_plain_text() {
        let result = format_tool_output("just plain text");
        assert_eq!(result, "just plain text");
    }

    #[test]
    fn test_initialize_tools_safe() {
        let tools = initialize_tools(false, None);
        assert!(tools.contains_key("read_file"));
        assert!(tools.contains_key("write_file"));
        assert!(!tools.contains_key("run_command"));
        assert!(!tools.contains_key("fetch_web_content"));
    }

    #[test]
    fn test_initialize_tools_unsafe() {
        let tools = initialize_tools(true, None);
        assert!(tools.contains_key("read_file"));
        assert!(tools.contains_key("run_command"));
        assert!(tools.contains_key("fetch_web_content"));
    }

    #[test]
    fn test_initialize_tools_allowed_filter() {
        let allowed = vec!["read_file".to_string(), "write_file".to_string()];
        let tools = initialize_tools(true, Some(&allowed));
        assert!(tools.contains_key("read_file"));
        assert!(tools.contains_key("write_file"));
        assert!(!tools.contains_key("run_command"));
        assert!(!tools.contains_key("glob"));
    }

    fn patch(path: &str, edits: serde_json::Value) -> Result<serde_json::Value, Box<dyn Error>> {
        let params = serde_json::json!({ "path": path, "edits": edits });
        tool_patch_file(&params.to_string(), &test_ctx()).map(|r| serde_json::from_str(&r).unwrap())
    }

    #[test]
    fn test_patch_file_registered_as_safe_tool() {
        let tools = initialize_tools(false, None);
        assert!(tools.contains_key("patch_file"));
    }

    #[test]
    fn test_patch_file_multiple_edits() {
        let path = "_test_pf_multi.tmp";
        write_test_file(path, "one\ntwo\nthree\n");
        let res = patch(
            path,
            serde_json::json!([
                {"old_content": "one", "new_content": "1"},
                {"old_content": "three", "new_content": "THREE!"}
            ]),
        )
        .unwrap();
        assert_eq!(res["edits_applied"], 2);
        assert_eq!(res["replacements"], 2);
        assert_eq!(read_test_file(path), "1\ntwo\nTHREE!\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_edits_apply_in_order() {
        let path = "_test_pf_order.tmp";
        write_test_file(path, "alpha\n");
        patch(
            path,
            serde_json::json!([
                {"old_content": "alpha", "new_content": "beta"},
                {"old_content": "beta", "new_content": "gamma"}
            ]),
        )
        .unwrap();
        assert_eq!(read_test_file(path), "gamma\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_is_atomic() {
        let path = "_test_pf_atomic.tmp";
        write_test_file(path, "keep me\nchange me\n");
        let err = patch(
            path,
            serde_json::json!([
                {"old_content": "change me", "new_content": "changed"},
                {"old_content": "missing", "new_content": "x"}
            ]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("edit 2"));
        assert_eq!(read_test_file(path), "keep me\nchange me\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_ambiguous_match() {
        let path = "_test_pf_ambiguous.tmp";
        write_test_file(path, "aa bb aa\n");
        let err = patch(
            path,
            serde_json::json!([{"old_content": "aa", "new_content": "x"}]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("multiple times"));
        assert_eq!(read_test_file(path), "aa bb aa\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_overlapping_match_is_ambiguous() {
        let path = "_test_pf_overlap.tmp";
        write_test_file(path, "aaa\n");
        assert!(
            patch(
                path,
                serde_json::json!([{"old_content": "aa", "new_content": "x"}])
            )
            .is_err()
        );
        cleanup(path);
    }

    #[test]
    fn test_patch_file_replace_all() {
        let path = "_test_pf_all.tmp";
        write_test_file(path, "aa bb aa cc aa\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "aa", "new_content": "X", "replace_all": true}]),
        )
        .unwrap();
        assert_eq!(res["replacements"], 3);
        assert_eq!(read_test_file(path), "X bb X cc X\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_shrinks_and_grows() {
        let path = "_test_pf_resize.tmp";
        write_test_file(path, "head\nmiddle part\ntail\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "middle part\n", "new_content": ""}]),
        )
        .unwrap();
        assert_eq!(res["bytes_after"], 10);
        assert_eq!(read_test_file(path), "head\ntail\n");
        patch(
            path,
            serde_json::json!([{"old_content": "head\n", "new_content": "a much longer head\n"}]),
        )
        .unwrap();
        assert_eq!(read_test_file(path), "a much longer head\ntail\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_writes_only_changed_bytes() {
        let path = "_test_pf_minimal.tmp";
        write_test_file(path, "prefix XXXX suffix\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "XXXX", "new_content": "YYYY"}]),
        )
        .unwrap();
        assert_eq!(res["bytes_written"], 4);
        assert_eq!(read_test_file(path), "prefix YYYY suffix\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_preserves_mode() {
        use std::os::unix::fs::PermissionsExt;
        let path = "_test_pf_mode.tmp";
        write_test_file(path, "#!/bin/sh\necho hi\n");
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
        patch(
            path,
            serde_json::json!([{"old_content": "hi", "new_content": "ho"}]),
        )
        .unwrap();
        let mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755);
        cleanup(path);
    }

    #[test]
    fn test_patch_file_unicode() {
        let path = "_test_pf_unicode.tmp";
        write_test_file(path, "héllo wörld\n");
        patch(
            path,
            serde_json::json!([{"old_content": "wörld", "new_content": "monde"}]),
        )
        .unwrap();
        assert_eq!(read_test_file(path), "héllo monde\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_rejects_bad_input() {
        let path = "_test_pf_bad.tmp";
        write_test_file(path, "content\n");
        assert!(patch(path, serde_json::json!([])).is_err());
        assert!(
            patch(
                path,
                serde_json::json!([{"old_content": "", "new_content": "x"}])
            )
            .is_err()
        );
        assert!(
            patch(
                path,
                serde_json::json!([{"old_content": "content", "new_content": "content"}])
            )
            .is_err()
        );
        assert!(
            patch(
                "_test_pf_missing.tmp",
                serde_json::json!([{"old_content": "a", "new_content": "b"}])
            )
            .is_err()
        );
        assert_eq!(read_test_file(path), "content\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_rejects_invalid_utf8_and_directories() {
        let path = "_test_pf_binary.tmp";
        std::fs::write(path, [0xff, 0xfe, b'a']).unwrap();
        assert!(
            patch(
                path,
                serde_json::json!([{"old_content": "a", "new_content": "b"}])
            )
            .is_err()
        );
        cleanup(path);
        assert!(
            patch(
                "src",
                serde_json::json!([{"old_content": "a", "new_content": "b"}])
            )
            .is_err()
        );
    }

    #[test]
    fn test_patch_file_cannot_escape_the_root() {
        let outside = "../_test_pf_outside.tmp";
        std::fs::write(outside, "outside\n").unwrap();
        let res = patch(
            outside,
            serde_json::json!([{"old_content": "outside", "new_content": "hacked"}]),
        );
        assert_eq!(std::fs::read_to_string(outside).unwrap(), "outside\n");
        assert!(res.is_err());
        let _ = std::fs::remove_file(outside);
    }

    #[test]
    fn test_patch_file_schema_and_dispatch() {
        let tools = initialize_tools(false, None);
        let schema: serde_json::Value = serde_json::from_str(&tools["patch_file"].schema).unwrap();
        assert_eq!(schema["function"]["name"], "patch_file");
        assert_eq!(
            schema["function"]["parameters"]["required"],
            serde_json::json!(["path", "edits"])
        );
        assert_eq!(
            schema["function"]["parameters"]["properties"]["edits"]["items"]["required"],
            serde_json::json!(["old_content", "new_content"])
        );

        let path = "_test_pf_dispatch.tmp";
        write_test_file(path, "before\n");
        let call = ToolCall {
            index: None,
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "patch_file".to_string(),
                arguments: serde_json::json!({
                    "path": path,
                    "edits": [{"old_content": "before", "new_content": "after"}]
                })
                .to_string(),
            },
        };
        let msg = tool_call(&tools, &call, &test_ctx()).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.name.as_deref(), Some("patch_file"));
        assert!(msg.content.unwrap().contains("patched successfully"));
        assert_eq!(read_test_file(path), "after\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_malformed_params() {
        let ctx = test_ctx();
        for params in [
            "not json",
            r#"{"path": "x"}"#,
            r#"{"edits": []}"#,
            r#"{"path": "x", "edits": "nope"}"#,
            r#"{"path": "x", "edits": [{"old_content": "a"}]}"#,
        ] {
            assert!(
                tool_patch_file(&params.to_string(), &ctx).is_err(),
                "{}",
                params
            );
        }
    }

    #[test]
    fn test_patch_file_replace_all_not_found() {
        let path = "_test_pf_all_missing.tmp";
        write_test_file(path, "abc\n");
        let err = patch(
            path,
            serde_json::json!([{"old_content": "zzz", "new_content": "x", "replace_all": true}]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not found"));
        assert_eq!(read_test_file(path), "abc\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_replace_all_deletes() {
        let path = "_test_pf_all_delete.tmp";
        write_test_file(path, "a-b-c-d\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "-", "new_content": "", "replace_all": true}]),
        )
        .unwrap();
        assert_eq!(res["bytes_after"], 5);
        assert_eq!(read_test_file(path), "abcd\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_edits_that_cancel_out_write_nothing() {
        let path = "_test_pf_cancel.tmp";
        write_test_file(path, "one two\n");
        let res = patch(
            path,
            serde_json::json!([
                {"old_content": "one", "new_content": "1"},
                {"old_content": "1", "new_content": "one"}
            ]),
        )
        .unwrap();
        assert_eq!(res["edits_applied"], 2);
        assert_eq!(res["bytes_written"], 0);
        assert_eq!(read_test_file(path), "one two\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_same_length_edits_write_only_the_changed_span() {
        let path = "_test_pf_span.tmp";
        write_test_file(path, "AAAA----------------BBBB\n");
        let res = patch(
            path,
            serde_json::json!([
                {"old_content": "AAAA", "new_content": "aaaa"},
                {"old_content": "BBBB", "new_content": "bbbb"}
            ]),
        )
        .unwrap();
        // From the first to the last changed byte, not the whole file.
        assert_eq!(res["bytes_written"], 24);
        assert_eq!(read_test_file(path), "aaaa----------------bbbb\n");

        let res = patch(
            path,
            serde_json::json!([{"old_content": "aaaa", "new_content": "AAAA"}]),
        )
        .unwrap();
        assert_eq!(res["bytes_written"], 4);
        cleanup(path);
    }

    #[test]
    fn test_patch_file_growing_edit_writes_from_first_change_to_end() {
        let path = "_test_pf_grow_tail.tmp";
        write_test_file(path, "keep keep keep X tail\n");
        let res = patch(
            path,
            serde_json::json!([{"old_content": "X", "new_content": "XYZ"}]),
        )
        .unwrap();
        assert_eq!(res["bytes_written"], 8);
        assert_eq!(read_test_file(path), "keep keep keep XYZ tail\n");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_multiline_and_crlf() {
        let path = "_test_pf_crlf.tmp";
        write_test_file(path, "fn a() {\r\n    1\r\n}\r\nfn b() {\r\n    2\r\n}\r\n");
        patch(
            path,
            serde_json::json!([{
                "old_content": "fn a() {\r\n    1\r\n}\r\n",
                "new_content": "fn a() {\r\n    42\r\n}\r\n"
            }]),
        )
        .unwrap();
        assert_eq!(
            read_test_file(path),
            "fn a() {\r\n    42\r\n}\r\nfn b() {\r\n    2\r\n}\r\n"
        );
        cleanup(path);
    }

    #[test]
    fn test_patch_file_empty_file_cannot_match() {
        let path = "_test_pf_empty.tmp";
        write_test_file(path, "");
        assert!(
            patch(
                path,
                serde_json::json!([{"old_content": "a", "new_content": "b"}])
            )
            .is_err()
        );
        assert_eq!(read_test_file(path), "");
        cleanup(path);
    }

    #[test]
    fn test_patch_file_follows_symlink_inside_root() {
        let target = "_test_pf_sym_target.tmp";
        let link = "_test_pf_sym_link.tmp";
        write_test_file(target, "linked content\n");
        let _ = std::fs::remove_file(link);
        std::os::unix::fs::symlink(target, link).unwrap();
        patch(
            link,
            serde_json::json!([{"old_content": "linked", "new_content": "patched"}]),
        )
        .unwrap();
        assert_eq!(read_test_file(target), "patched content\n");
        cleanup(link);
        cleanup(target);
    }

    #[test]
    fn test_patch_file_symlinks_cannot_escape_the_root() {
        let outside = std::env::temp_dir().join("_test_pf_symlink_outside.tmp");
        std::fs::write(&outside, "outside\n").unwrap();

        // Absolute and relative links pointing outside the working directory
        // are resolved relative to the root, never to the real filesystem.
        let abs_link = "_test_pf_abs_link.tmp";
        let rel_link = "_test_pf_rel_link.tmp";
        let _ = std::fs::remove_file(abs_link);
        let _ = std::fs::remove_file(rel_link);
        std::os::unix::fs::symlink(&outside, abs_link).unwrap();
        std::os::unix::fs::symlink("../../../../../../../../../../tmp", rel_link).unwrap();

        let edits = serde_json::json!([{"old_content": "outside", "new_content": "hacked"}]);
        assert!(patch(abs_link, edits.clone()).is_err());
        assert!(patch(rel_link, edits).is_err());
        assert_eq!(std::fs::read_to_string(&outside).unwrap(), "outside\n");

        cleanup(abs_link);
        cleanup(rel_link);
        let _ = std::fs::remove_file(&outside);
    }

    #[test]
    fn test_patch_file_large_file() {
        let path = "_test_pf_large.tmp";
        let line = "The quick brown fox jumps over the lazy dog.\n";
        let original = line.repeat(20_000);
        write_test_file(path, &original);
        let res = patch(
            path,
            serde_json::json!([
                {"old_content": "quick", "new_content": "slow", "replace_all": true},
                {"old_content": "lazy", "new_content": "energetic", "replace_all": true}
            ]),
        )
        .unwrap();
        assert_eq!(res["replacements"], 40_000);
        assert_eq!(
            read_test_file(path),
            "The slow brown fox jumps over the energetic dog.\n".repeat(20_000)
        );
        cleanup(path);
    }

    fn test_ctx() -> ToolContext {
        ToolContext::new(|_| {})
    }

    fn write_test_file(name: &str, content: &str) {
        std::fs::write(name, content).unwrap();
    }

    fn read_test_file(name: &str) -> String {
        std::fs::read_to_string(name).unwrap()
    }

    fn cleanup(name: &str) {
        let _ = std::fs::remove_file(name);
    }

    #[test]
    fn test_write_file_full_create() {
        let path = "_test_wf_full_create.tmp";
        cleanup(path);
        let params = serde_json::json!({
            "path": path,
            "content": "hello world\n"
        });
        let result = tool_write_file(&params.to_string(), &test_ctx()).unwrap();
        let res: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(res["created"], true);
        assert_eq!(res["operation"], "create");
        assert_eq!(res["bytes_written"], 12);
        assert_eq!(read_test_file(path), "hello world\n");
        cleanup(path);
    }

    #[test]
    fn test_write_file_full_overwrite() {
        let path = "_test_wf_full_overwrite.tmp";
        write_test_file(path, "old content\n");
        let params = serde_json::json!({
            "path": path,
            "content": "new content\n"
        });
        let result = tool_write_file(&params.to_string(), &test_ctx()).unwrap();
        let res: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(res["created"], false);
        assert_eq!(res["operation"], "overwrite");
        assert_eq!(read_test_file(path), "new content\n");
        cleanup(path);
    }

    #[test]
    fn test_write_file_search_replace() {
        let path = "_test_wf_search_replace.tmp";
        write_test_file(path, "line one\nline two\nline three\n");
        let params = serde_json::json!({
            "path": path,
            "content": "line TWO",
            "old_content": "line two"
        });
        let result = tool_write_file(&params.to_string(), &test_ctx()).unwrap();
        let res: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(res["operation"], "patch_search_replace");
        assert_eq!(read_test_file(path), "line one\nline TWO\nline three\n");
        cleanup(path);
    }

    #[test]
    fn test_write_file_search_replace_all() {
        let path = "_test_wf_search_replace_all.tmp";
        write_test_file(path, "aaa bbb aaa ccc aaa\n");
        let params = serde_json::json!({
            "path": path,
            "content": "XXX",
            "old_content": "aaa",
            "replace_all": true
        });
        let result = tool_write_file(&params.to_string(), &test_ctx()).unwrap();
        let res: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(res["operation"], "patch_search_replace");
        assert_eq!(read_test_file(path), "XXX bbb XXX ccc XXX\n");
        cleanup(path);
    }

    #[test]
    fn test_write_file_search_replace_not_found() {
        let path = "_test_wf_sr_notfound.tmp";
        write_test_file(path, "hello world\n");
        let params = serde_json::json!({
            "path": path,
            "content": "replacement",
            "old_content": "nonexistent"
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
        cleanup(path);
    }

    #[test]
    fn test_write_file_search_replace_multiple_without_flag() {
        let path = "_test_wf_sr_multi.tmp";
        write_test_file(path, "foo bar foo baz foo\n");
        let params = serde_json::json!({
            "path": path,
            "content": "X",
            "old_content": "foo"
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("matches 3 times"));
        cleanup(path);
    }

    #[test]
    fn test_write_file_search_replace_identical() {
        let path = "_test_wf_sr_identical.tmp";
        write_test_file(path, "hello world\n");
        let params = serde_json::json!({
            "path": path,
            "content": "hello",
            "old_content": "hello"
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("identical"));
        cleanup(path);
    }

    #[test]
    fn test_write_file_byte_offset() {
        let path = "_test_wf_byte_offset.tmp";
        write_test_file(path, "Hello, World!");
        let params = serde_json::json!({
            "path": path,
            "content": "Rust",
            "offset": 7,
            "length": 5
        });
        let result = tool_write_file(&params.to_string(), &test_ctx()).unwrap();
        let res: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(res["operation"], "patch_offset");
        assert_eq!(read_test_file(path), "Hello, Rust!");
        cleanup(path);
    }

    #[test]
    fn test_write_file_byte_offset_expand() {
        let path = "_test_wf_byte_offset_expand.tmp";
        write_test_file(path, "ABCDE");
        let params = serde_json::json!({
            "path": path,
            "content": "XXXX",
            "offset": 1,
            "length": 2
        });
        let result = tool_write_file(&params.to_string(), &test_ctx()).unwrap();
        assert_eq!(read_test_file(path), "AXXXXDE");
        let res: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(res["bytes_written"], 7);
        cleanup(path);
    }

    #[test]
    fn test_write_file_byte_offset_out_of_bounds() {
        let path = "_test_wf_byte_oob.tmp";
        write_test_file(path, "short");
        let params = serde_json::json!({
            "path": path,
            "content": "x",
            "offset": 100,
            "length": 1
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("beyond file size"));
        cleanup(path);
    }

    #[test]
    fn test_write_file_byte_offset_length_exceeds() {
        let path = "_test_wf_byte_len_oob.tmp";
        write_test_file(path, "short");
        let params = serde_json::json!({
            "path": path,
            "content": "x",
            "offset": 3,
            "length": 10
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exceeds file size")
        );
        cleanup(path);
    }

    #[test]
    fn test_write_file_line_range() {
        let path = "_test_wf_line_range.tmp";
        write_test_file(path, "line1\nline2\nline3\nline4\nline5\n");
        let params = serde_json::json!({
            "path": path,
            "content": "NEW2\nNEW3",
            "start_line": 2,
            "end_line": 3
        });
        let result = tool_write_file(&params.to_string(), &test_ctx()).unwrap();
        let res: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(res["operation"], "patch_lines");
        assert_eq!(read_test_file(path), "line1\nNEW2\nNEW3\nline4\nline5\n");
        cleanup(path);
    }

    #[test]
    fn test_write_file_line_range_single_line() {
        let path = "_test_wf_line_single.tmp";
        write_test_file(path, "aaa\nbbb\nccc\n");
        let params = serde_json::json!({
            "path": path,
            "content": "BBB",
            "start_line": 2,
            "end_line": 2
        });
        let _result = tool_write_file(&params.to_string(), &test_ctx()).unwrap();
        assert_eq!(read_test_file(path), "aaa\nBBB\nccc\n");
        cleanup(path);
    }

    #[test]
    fn test_write_file_line_range_out_of_bounds() {
        let path = "_test_wf_line_oob.tmp";
        write_test_file(path, "one\ntwo\n");
        let params = serde_json::json!({
            "path": path,
            "content": "x",
            "start_line": 1,
            "end_line": 10
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exceeds file line count")
        );
        cleanup(path);
    }

    #[test]
    fn test_write_file_line_range_zero_start() {
        let path = "_test_wf_line_zero.tmp";
        write_test_file(path, "one\ntwo\n");
        let params = serde_json::json!({
            "path": path,
            "content": "x",
            "start_line": 0,
            "end_line": 1
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("1-based"));
        cleanup(path);
    }

    #[test]
    fn test_write_file_line_range_end_before_start() {
        let path = "_test_wf_line_inv.tmp";
        write_test_file(path, "one\ntwo\nthree\n");
        let params = serde_json::json!({
            "path": path,
            "content": "x",
            "start_line": 3,
            "end_line": 1
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be >="));
        cleanup(path);
    }

    #[test]
    fn test_write_file_conflicting_params_old_content_and_offset() {
        let path = "_test_wf_conflict1.tmp";
        write_test_file(path, "content");
        let params = serde_json::json!({
            "path": path,
            "content": "x",
            "old_content": "con",
            "offset": 0,
            "length": 3
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cannot combine"));
        cleanup(path);
    }

    #[test]
    fn test_write_file_conflicting_params_offset_and_start_line() {
        let path = "_test_wf_conflict2.tmp";
        write_test_file(path, "content\n");
        let params = serde_json::json!({
            "path": path,
            "content": "x",
            "offset": 0,
            "length": 1,
            "start_line": 1,
            "end_line": 1
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cannot combine"));
        cleanup(path);
    }

    #[test]
    fn test_write_file_offset_without_length() {
        let path = "_test_wf_no_len.tmp";
        write_test_file(path, "content");
        let params = serde_json::json!({
            "path": path,
            "content": "x",
            "offset": 0
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires length"));
        cleanup(path);
    }

    #[test]
    fn test_write_file_start_line_without_end_line() {
        let path = "_test_wf_no_end.tmp";
        write_test_file(path, "content\n");
        let params = serde_json::json!({
            "path": path,
            "content": "x",
            "start_line": 1
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires end_line")
        );
        cleanup(path);
    }

    #[test]
    fn test_write_file_partial_on_nonexistent_file() {
        let path = "_test_wf_nofile.tmp";
        cleanup(path);
        let params = serde_json::json!({
            "path": path,
            "content": "x",
            "old_content": "y"
        });
        let result = tool_write_file(&params.to_string(), &test_ctx());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must exist"));
    }
}
