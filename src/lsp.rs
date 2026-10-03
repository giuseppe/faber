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

//! A minimal Language Server Protocol client behind the `lsp` tool:
//! go-to-definition, references, hover, document/workspace symbols and
//! diagnostics, from whichever language server handles a file's extension.
//!
//! Servers are started on first use and kept running for the session, one
//! per (server, project root). Unless `--unsafe-tools` is set they run in
//! the same bubblewrap sandbox as the LaTeX toolchain (whole filesystem
//! read-only, only the project directory writable, no network): language
//! servers can run project code - rust-analyzer builds `build.rs` scripts
//! and proc macros - and that must not become a way around `run_command`'s
//! sandbox.

use log::{debug, trace, warn};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use faber::ToolContext;

/// How long a single request may take. Generous: the first request to a
/// server that's still loading the project can take a while.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// How long `diagnostics` waits for the server to publish results for a
/// file it was just sent.
const DIAGNOSTICS_TIMEOUT: Duration = Duration::from_secs(20);

/// How long a request waits for the server to finish loading/indexing the
/// project (reported through `$/progress`) before going ahead anyway.
const INDEXING_WAIT: Duration = Duration::from_secs(30);

/// Most locations/symbols listed in one result.
const MAX_RESULTS: usize = 200;

/// A language server faber knows how to start, and which files it handles.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ServerSpec {
    /// The name it's configured under in `lsp_servers`.
    name: String,
    /// Alternatives in order of preference; the first installed is used.
    commands: Vec<Vec<String>>,
    extensions: Vec<String>,
    /// The LSP `languageId` for its files, if not the built-in default.
    language_id: Option<String>,
    /// Sent as `initializationOptions` and as the answer to the server's
    /// `workspace/configuration` requests.
    settings: Value,
}

/// A server as written in the config file's `lsp_servers`:
///
/// ```json
/// "lsp_servers": {
///   "zig": {"command": ["zls"], "extensions": ["zig"]},
///   "python": {"command": ["pylsp"], "extensions": ["py"]},
///   "go": null
/// }
/// ```
///
/// An entry named like a built-in server (`rust`, `c`, `python`, `go`,
/// `typescript`) replaces it, `null` removes it, and any other name adds a
/// server.
#[derive(Debug, Clone, PartialEq, serde::Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// The server's executable (a name looked up on PATH, or a path) and
    /// its arguments.
    pub command: Vec<String>,
    /// File extensions it handles, without the dot.
    pub extensions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language_id: Option<String>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub settings: Value,
}

fn builtin(name: &str, commands: &[&[&str]], extensions: &[&str]) -> ServerSpec {
    ServerSpec {
        name: name.to_string(),
        commands: commands
            .iter()
            .map(|c| c.iter().map(|a| a.to_string()).collect())
            .collect(),
        extensions: extensions.iter().map(|e| e.to_string()).collect(),
        language_id: None,
        settings: Value::Null,
    }
}

fn builtin_servers() -> Vec<ServerSpec> {
    vec![
        builtin("rust", &[&["rust-analyzer"]], &["rs"]),
        builtin(
            "c",
            &[&["clangd"]],
            &["c", "h", "cc", "cpp", "cxx", "hh", "hpp", "hxx"],
        ),
        builtin(
            "python",
            &[&["pyright-langserver", "--stdio"], &["pylsp"]],
            &["py"],
        ),
        builtin("go", &[&["gopls"]], &["go"]),
        builtin(
            "typescript",
            &[&["typescript-language-server", "--stdio"]],
            &["ts", "tsx", "js", "jsx", "mjs", "cjs"],
        ),
    ]
}

/// Applies the config file's `lsp_servers` to `servers` (see
/// `ServerConfig`), checking that every entry makes sense and that no two
/// servers claim the same extension.
pub(crate) fn merge_servers(
    mut servers: Vec<ServerSpec>,
    overrides: &HashMap<String, Option<ServerConfig>>,
) -> Result<Vec<ServerSpec>, String> {
    let mut names: Vec<&String> = overrides.keys().collect();
    names.sort();
    for name in names {
        servers.retain(|s| &s.name != name);
        let Some(config) = &overrides[name] else {
            continue;
        };
        if config.command.first().is_none_or(|c| c.is_empty()) {
            return Err(format!("lsp_servers.{}: \"command\" is empty", name));
        }
        if config.extensions.is_empty() {
            return Err(format!("lsp_servers.{}: \"extensions\" is empty", name));
        }
        servers.push(ServerSpec {
            name: name.clone(),
            commands: vec![config.command.clone()],
            extensions: config
                .extensions
                .iter()
                .map(|e| e.trim_start_matches('.').to_ascii_lowercase())
                .collect(),
            language_id: config.language_id.clone(),
            settings: config.settings.clone(),
        });
    }
    let mut owner: HashMap<&str, &str> = HashMap::new();
    for server in &servers {
        for ext in &server.extensions {
            if let Some(other) = owner.insert(ext, &server.name) {
                return Err(format!(
                    "lsp_servers: both \"{}\" and \"{}\" handle .{} files",
                    other, server.name, ext
                ));
            }
        }
    }
    Ok(servers)
}

static SERVERS: OnceLock<Vec<ServerSpec>> = OnceLock::new();

/// Installs the config file's `lsp_servers`; called once at startup,
/// before any tool runs. Without it, the built-in servers are used.
pub(crate) fn configure(overrides: &HashMap<String, Option<ServerConfig>>) -> Result<(), String> {
    let servers = merge_servers(builtin_servers(), overrides)?;
    let _ = SERVERS.set(servers);
    Ok(())
}

fn servers() -> &'static [ServerSpec] {
    SERVERS.get_or_init(builtin_servers)
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
}

fn find_server<'a>(servers: &'a [ServerSpec], path: &Path) -> Option<&'a ServerSpec> {
    let ext = extension(path)?;
    servers.iter().find(|s| s.extensions.contains(&ext))
}

/// The LSP `languageId` for a file: the server's configured one, or the
/// standard one for well-known extensions, or else the extension itself.
fn language_id(spec: &ServerSpec, path: &Path) -> String {
    if let Some(id) = &spec.language_id {
        return id.clone();
    }
    let ext = extension(path).unwrap_or_default();
    match ext.as_str() {
        "rs" => "rust",
        "c" | "h" => "c",
        "cc" | "cpp" | "cxx" | "hh" | "hpp" | "hxx" => "cpp",
        "py" => "python",
        "go" => "go",
        "ts" => "typescript",
        "tsx" => "typescriptreact",
        "jsx" => "javascriptreact",
        "js" | "mjs" | "cjs" => "javascript",
        other => other,
    }
    .to_string()
}

/// Resolves the first of `spec`'s commands that's installed: a name is
/// looked up on PATH, a path is taken as is if it exists.
fn resolve_server_command(spec: &ServerSpec) -> Result<(PathBuf, Vec<String>), String> {
    for command in &spec.commands {
        let program = if command[0].contains('/') {
            Some(PathBuf::from(&command[0])).filter(|p| p.is_file())
        } else {
            crate::latex_kitty::resolve_on_path(&command[0])
        };
        if let Some(program) = program {
            return Ok((program, command[1..].to_vec()));
        }
    }
    Err(format!(
        "the \"{}\" language server isn't installed (tried: {})",
        spec.name,
        spec.commands
            .iter()
            .map(|c| c[0].as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// The bubblewrap arguments a sandboxed server runs under: the LaTeX
/// toolchain's sandbox (read-only root so toolchains in `$HOME` still work,
/// only `root` writable, no network), which also dies with faber.
pub(crate) fn sandboxed_server_args(root: &str, program: &str, args: &[String]) -> Vec<String> {
    crate::latex_kitty::whole_root_ro_bwrap_args(root, program, Some(args))
}

const UNRESERVED: &[u8] = b"-._~/";

pub(crate) fn path_to_uri(path: &Path) -> String {
    let mut uri = String::from("file://");
    for &b in path.as_os_str().as_encoded_bytes() {
        if b.is_ascii_alphanumeric() || UNRESERVED.contains(&b) {
            uri.push(b as char);
        } else {
            uri.push_str(&format!("%{:02X}", b));
        }
    }
    uri
}

pub(crate) fn uri_to_path(uri: &str) -> Option<PathBuf> {
    let encoded = uri.strip_prefix("file://")?.as_bytes();
    let mut bytes = Vec::with_capacity(encoded.len());
    let mut i = 0;
    while i < encoded.len() {
        if encoded[i] == b'%' && i + 2 < encoded.len() {
            let hex = std::str::from_utf8(&encoded[i + 1..i + 3]).ok()?;
            bytes.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            bytes.push(encoded[i]);
            i += 1;
        }
    }
    Some(PathBuf::from(String::from_utf8(bytes).ok()?))
}

/// Writes one JSON-RPC message with its `Content-Length` header.
fn write_message(out: &mut impl Write, message: &Value) -> std::io::Result<()> {
    let body = serde_json::to_string(message)?;
    write!(out, "Content-Length: {}\r\n\r\n{}", body.len(), body)?;
    out.flush()
}

/// Reads one JSON-RPC message; `None` at end of stream.
fn read_message(input: &mut impl BufRead) -> Result<Option<Value>, Box<dyn Error>> {
    let mut content_length = None;
    loop {
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = Some(value.trim().parse::<usize>()?);
            }
        }
    }
    let length = content_length.ok_or("LSP message without Content-Length")?;
    let mut body = vec![0; length];
    input.read_exact(&mut body)?;
    Ok(Some(serde_json::from_slice(&body)?))
}

type PendingReplies = HashMap<u64, mpsc::Sender<Result<Value, Value>>>;

/// Diagnostics per URI, with a counter bumped on every publish so a caller
/// can wait for results newer than its last change.
#[derive(Default)]
struct DiagnosticsStore {
    by_uri: HashMap<String, Vec<Value>>,
    generation: u64,
    generation_by_uri: HashMap<String, u64>,
}

struct Shared {
    pending: Mutex<PendingReplies>,
    diagnostics: Mutex<DiagnosticsStore>,
    diagnostics_changed: Condvar,
    /// `$/progress` tokens that have begun but not ended - the server is
    /// busy loading or indexing while this is non-empty.
    progress: Mutex<HashSet<String>>,
    progress_changed: Condvar,
    /// Answers `workspace/configuration` requests.
    settings: Value,
}

struct OpenDocument {
    version: i64,
    text: String,
}

pub(crate) struct Client {
    name: String,
    spec: ServerSpec,
    root: PathBuf,
    child: Mutex<Child>,
    stdin: Arc<Mutex<ChildStdin>>,
    next_id: AtomicU64,
    shared: Arc<Shared>,
    documents: Mutex<HashMap<String, OpenDocument>>,
}

/// The result for a request the server sent us, or `None` if unsupported.
/// Only what servers commonly need to get going is handled.
fn answer_server_request(
    method: &str,
    params: &Value,
    shared: &Shared,
    root: &Path,
) -> Option<Value> {
    Some(match method {
        "workspace/configuration" => {
            let count = params["items"].as_array().map_or(0, |i| i.len());
            Value::Array(vec![shared.settings.clone(); count])
        }
        "workspace/workspaceFolders" => json!([{
            "uri": path_to_uri(root),
            "name": root.file_name().and_then(|n| n.to_str()).unwrap_or("root"),
        }]),
        "window/workDoneProgress/create"
        | "client/registerCapability"
        | "client/unregisterCapability"
        | "window/showMessageRequest" => Value::Null,
        _ => return None,
    })
}

fn reader_loop(
    stdout: impl Read,
    stdin: Arc<Mutex<ChildStdin>>,
    shared: Arc<Shared>,
    root: PathBuf,
) {
    let mut input = BufReader::new(stdout);
    loop {
        let message = match read_message(&mut input) {
            Ok(Some(message)) => message,
            Ok(None) => break,
            Err(e) => {
                warn!("LSP: bad message from server: {}", e);
                break;
            }
        };
        trace!("LSP <- {}", message);
        let method = message.get("method").and_then(|m| m.as_str());
        match (message.get("id"), method) {
            (Some(id), Some(method)) => {
                let reply = match answer_server_request(method, &message["params"], &shared, &root)
                {
                    Some(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                    None => json!({"jsonrpc": "2.0", "id": id, "error": {
                        "code": -32601,
                        "message": format!("unsupported: {}", method),
                    }}),
                };
                let mut stdin = stdin.lock().unwrap_or_else(|e| e.into_inner());
                let _ = write_message(&mut *stdin, &reply);
            }
            (Some(id), None) => {
                let Some(id) = id.as_u64() else { continue };
                let sender = shared
                    .pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(&id);
                if let Some(sender) = sender {
                    let reply = match message.get("error") {
                        Some(error) => Err(error.clone()),
                        None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
                    };
                    let _ = sender.send(reply);
                }
            }
            (None, Some("textDocument/publishDiagnostics")) => {
                let params = &message["params"];
                let Some(uri) = params["uri"].as_str() else {
                    continue;
                };
                let diagnostics = params["diagnostics"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                let mut store = shared.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
                store.generation += 1;
                let generation = store.generation;
                store.by_uri.insert(uri.to_string(), diagnostics);
                store.generation_by_uri.insert(uri.to_string(), generation);
                shared.diagnostics_changed.notify_all();
            }
            (None, Some("$/progress")) => {
                let params = &message["params"];
                let token = params["token"].to_string();
                let mut progress = shared.progress.lock().unwrap_or_else(|e| e.into_inner());
                match params["value"]["kind"].as_str() {
                    Some("begin") => {
                        progress.insert(token);
                    }
                    Some("end") => {
                        progress.remove(&token);
                        shared.progress_changed.notify_all();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    // The server is gone: fail everything still waiting instead of letting
    // it time out.
    let mut pending = shared.pending.lock().unwrap_or_else(|e| e.into_inner());
    for (_, sender) in pending.drain() {
        let _ = sender.send(Err(json!({"message": "language server exited"})));
    }
}

/// Spawns processes from one long-lived thread. A sandboxed server dies
/// with the thread that started it (bwrap's `--die-with-parent` is
/// `PR_SET_PDEATHSIG`, which tracks the parent *thread*), and tools can run
/// on short-lived threads - parallel tool calls, sub-agents.
fn spawn_from_long_lived_thread(mut command: Command) -> std::io::Result<Child> {
    type Job = (Command, mpsc::Sender<std::io::Result<Child>>);
    static SPAWNER: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
    let spawner = SPAWNER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        std::thread::spawn(move || {
            for (mut command, reply) in rx {
                let _ = reply.send(command.spawn());
            }
        });
        Mutex::new(tx)
    });
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let (reply_tx, reply_rx) = mpsc::channel();
    spawner
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .send((command, reply_tx))
        .map_err(|_| std::io::Error::other("LSP spawner thread is gone"))?;
    reply_rx
        .recv()
        .map_err(|_| std::io::Error::other("LSP spawner thread is gone"))?
}

impl Client {
    fn start(spec: &ServerSpec, root: &Path, sandboxed: bool) -> Result<Client, Box<dyn Error>> {
        let (program, args) = resolve_server_command(spec)?;
        let name = program
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("language server")
            .to_string();
        let root_str = root
            .to_str()
            .ok_or("project directory is not valid UTF-8")?;
        let program_str = program.to_str().ok_or("server path is not valid UTF-8")?;
        let mut command = if sandboxed {
            let mut c = Command::new("bwrap");
            c.args(sandboxed_server_args(root_str, program_str, &args));
            c
        } else {
            let mut c = Command::new(&program);
            c.args(&args);
            c
        };
        command.current_dir(root);
        debug!("Starting language server: {:?}", command);
        let mut child = spawn_from_long_lived_thread(command).map_err(|e| {
            if sandboxed {
                format!(
                    "couldn't start {} in the sandbox (bwrap): {} - is bubblewrap installed?",
                    name, e
                )
            } else {
                format!("couldn't start {}: {}", name, e)
            }
        })?;
        let stdin = Arc::new(Mutex::new(child.stdin.take().ok_or("no stdin")?));
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let shared = Arc::new(Shared {
            pending: Mutex::new(HashMap::new()),
            diagnostics: Mutex::new(DiagnosticsStore::default()),
            diagnostics_changed: Condvar::new(),
            progress: Mutex::new(HashSet::new()),
            progress_changed: Condvar::new(),
            settings: spec.settings.clone(),
        });
        {
            let stdin = stdin.clone();
            let shared = shared.clone();
            let root = root.to_path_buf();
            std::thread::spawn(move || reader_loop(stdout, stdin, shared, root));
        }
        let client = Client {
            name,
            spec: spec.clone(),
            root: root.to_path_buf(),
            child: Mutex::new(child),
            stdin,
            next_id: AtomicU64::new(1),
            shared,
            documents: Mutex::new(HashMap::new()),
        };
        client.initialize()?;
        Ok(client)
    }

    fn initialize(&self) -> Result<(), Box<dyn Error>> {
        let root_uri = path_to_uri(&self.root);
        let mut params = json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "workspaceFolders": [{
                "uri": root_uri,
                "name": self.root.file_name().and_then(|n| n.to_str()).unwrap_or("root"),
            }],
            "capabilities": {
                "general": {"positionEncodings": ["utf-16"]},
                "window": {"workDoneProgress": true},
                "workspace": {"configuration": true, "workspaceFolders": true, "symbol": {}},
                "textDocument": {
                    "synchronization": {"didSave": true},
                    "definition": {"linkSupport": true},
                    "references": {},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": true},
                    "publishDiagnostics": {"relatedInformation": false},
                },
            },
        });
        if !self.spec.settings.is_null() {
            params["initializationOptions"] = self.spec.settings.clone();
        }
        self.request_once("initialize", params)
            .map_err(|e| format!("{} failed to initialize: {}", self.name, e))?;
        self.notify("initialized", json!({}))?;
        Ok(())
    }

    fn is_alive(&self) -> bool {
        matches!(
            self.child
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .try_wait(),
            Ok(None)
        )
    }

    fn notify(&self, method: &str, params: Value) -> Result<(), Box<dyn Error>> {
        let message = json!({"jsonrpc": "2.0", "method": method, "params": params});
        trace!("LSP -> {}", message);
        let mut stdin = self.stdin.lock().unwrap_or_else(|e| e.into_inner());
        write_message(&mut *stdin, &message)?;
        Ok(())
    }

    fn request_once(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel();
        self.shared
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, tx);
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        trace!("LSP -> {}", message);
        {
            let mut stdin = self.stdin.lock().unwrap_or_else(|e| e.into_inner());
            write_message(&mut *stdin, &message).map_err(|e| e.to_string())?;
        }
        match rx.recv_timeout(REQUEST_TIMEOUT) {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => Err(error.to_string()),
            Err(_) => {
                self.shared
                    .pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(&id);
                Err(format!(
                    "{} did not answer {} within {}s",
                    self.name,
                    method,
                    REQUEST_TIMEOUT.as_secs()
                ))
            }
        }
    }

    /// Sends a request, retrying a few times if the server says the
    /// content changed or it cancelled the request - what servers answer
    /// while they're still loading the project.
    fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        let mut attempt = 0;
        loop {
            match self.request_once(method, params.clone()) {
                Err(e) if attempt < 5 && (e.contains("-32801") || e.contains("-32802")) => {
                    attempt += 1;
                    std::thread::sleep(Duration::from_millis(500 * attempt));
                }
                result => return result,
            }
        }
    }

    /// Waits (up to `INDEXING_WAIT`) for the server to finish whatever it
    /// reported through `$/progress`. Returns whether it's still busy.
    fn wait_until_idle(&self) -> bool {
        let deadline = Instant::now() + INDEXING_WAIT;
        let mut progress = self
            .shared
            .progress
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while !progress.is_empty() {
            let now = Instant::now();
            if now >= deadline {
                return true;
            }
            progress = self
                .shared
                .progress_changed
                .wait_timeout(progress, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        false
    }

    /// Makes sure the server has `path`'s current content from disk -
    /// opening it the first time, and sending the new text whenever it has
    /// changed since (the model may have edited it in between). Returns the
    /// file's URI and text, and whether anything was sent.
    fn sync(&self, path: &Path) -> Result<(String, String, bool), Box<dyn Error>> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("can't read {}: {}", path.display(), e))?;
        let uri = path_to_uri(path);
        let mut documents = self.documents.lock().unwrap_or_else(|e| e.into_inner());
        let sent = match documents.get_mut(&uri) {
            None => {
                self.notify(
                    "textDocument/didOpen",
                    json!({"textDocument": {
                        "uri": uri,
                        "languageId": language_id(&self.spec, path),
                        "version": 1,
                        "text": text,
                    }}),
                )?;
                documents.insert(
                    uri.clone(),
                    OpenDocument {
                        version: 1,
                        text: text.clone(),
                    },
                );
                true
            }
            Some(doc) if doc.text != text => {
                doc.version += 1;
                doc.text = text.clone();
                self.notify(
                    "textDocument/didChange",
                    json!({
                        "textDocument": {"uri": uri, "version": doc.version},
                        "contentChanges": [{"text": text}],
                    }),
                )?;
                self.notify(
                    "textDocument/didSave",
                    json!({"textDocument": {"uri": uri}}),
                )?;
                true
            }
            Some(_) => false,
        };
        Ok((uri, text, sent))
    }

    fn diagnostics_generation(&self, uri: &str) -> Option<u64> {
        let store = self
            .shared
            .diagnostics
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        store.generation_by_uri.get(uri).copied()
    }

    /// Diagnostics for `uri`, waiting for a publish newer than `after`
    /// (`None`: any). The bool says whether that wait timed out.
    fn wait_for_diagnostics(&self, uri: &str, after: Option<u64>) -> (Vec<Value>, bool) {
        let deadline = Instant::now() + DIAGNOSTICS_TIMEOUT;
        let mut store = self
            .shared
            .diagnostics
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        loop {
            let current = store.generation_by_uri.get(uri).copied();
            if current.is_some() && current > after {
                return (store.by_uri.get(uri).cloned().unwrap_or_default(), false);
            }
            let now = Instant::now();
            if now >= deadline {
                return (store.by_uri.get(uri).cloned().unwrap_or_default(), true);
            }
            store = self
                .shared
                .diagnostics_changed
                .wait_timeout(store, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.lock().unwrap_or_else(|e| e.into_inner()).kill();
    }
}

/// Running clients, by (server command, project root).
fn clients() -> &'static Mutex<HashMap<(String, PathBuf), Arc<Client>>> {
    static CLIENTS: OnceLock<Mutex<HashMap<(String, PathBuf), Arc<Client>>>> = OnceLock::new();
    CLIENTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The running client for `path`'s language in the current directory's
/// project, started if needed (or restarted if it died).
fn client_for(path: &Path, root: &Path, sandboxed: bool) -> Result<Arc<Client>, Box<dyn Error>> {
    let spec = find_server(servers(), path).ok_or_else(|| {
        format!(
            "no language server is configured for {} (supported extensions: {}; more can be added with \"lsp_servers\" in the config file)",
            path.display(),
            servers()
                .iter()
                .flat_map(|s| s.extensions.iter())
                .map(|e| format!(".{}", e))
                .collect::<Vec<_>>()
                .join(" ")
        )
    })?;
    let root = root.to_path_buf();
    let key = (spec.name.clone(), root.clone());
    let mut clients = clients().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(client) = clients.get(&key) {
        if client.is_alive() {
            return Ok(client.clone());
        }
        warn!("LSP: {} exited, restarting it", client.name);
        clients.remove(&key);
    }
    let client = Arc::new(Client::start(spec, &root, sandboxed)?);
    clients.insert(key, client.clone());
    Ok(client)
}

/// 0-based LSP line/character for a 1-based `line` and either the
/// `symbol` text on it (its first occurrence) or a 1-based `column`.
/// Characters are counted in UTF-16 code units, the LSP default.
pub(crate) fn position_in(
    text: &str,
    line: u32,
    symbol: Option<&str>,
    column: Option<u32>,
) -> Result<Value, String> {
    let line_text = text
        .lines()
        .nth(line.checked_sub(1).ok_or("line is 1-based")? as usize)
        .ok_or_else(|| format!("line {} is past the end of the file", line))?;
    let byte = match (symbol, column) {
        (Some(symbol), _) => line_text.find(symbol).ok_or_else(|| {
            format!(
                "'{}' does not appear on line {}: {}",
                symbol,
                line,
                line_text.trim()
            )
        })?,
        (None, Some(column)) => line_text
            .char_indices()
            .nth(column.checked_sub(1).ok_or("column is 1-based")? as usize)
            .map_or(line_text.len(), |(i, _)| i),
        (None, None) => return Err("give the symbol to look up (or a column)".to_string()),
    };
    let character: usize = line_text[..byte].chars().map(char::len_utf16).sum();
    Ok(json!({"line": line - 1, "character": character}))
}

const SYMBOL_KINDS: &[&str] = &[
    "file",
    "module",
    "namespace",
    "package",
    "class",
    "method",
    "property",
    "field",
    "constructor",
    "enum",
    "interface",
    "function",
    "variable",
    "constant",
    "string",
    "number",
    "boolean",
    "array",
    "object",
    "key",
    "null",
    "enum member",
    "struct",
    "event",
    "operator",
    "type parameter",
];

fn symbol_kind(kind: &Value) -> &'static str {
    kind.as_u64()
        .and_then(|k| SYMBOL_KINDS.get((k as usize).wrapping_sub(1)))
        .copied()
        .unwrap_or("symbol")
}

/// `path` relative to `root` when it's inside it.
fn display_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// A result location as `path:line:column: <source line>`.
fn format_location(uri: &str, range: &Value, root: &Path) -> String {
    let line = range["start"]["line"].as_u64().unwrap_or(0);
    let character = range["start"]["character"].as_u64().unwrap_or(0);
    let Some(path) = uri_to_path(uri) else {
        return format!("{}:{}:{}", uri, line + 1, character + 1);
    };
    let source = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| t.lines().nth(line as usize).map(|l| l.trim().to_string()))
        .unwrap_or_default();
    format!(
        "{}:{}:{}: {}",
        display_path(&path, root),
        line + 1,
        character + 1,
        source
    )
}

/// Definition/reference results come as a Location, a list of them, or a
/// list of LocationLinks.
pub(crate) fn format_locations(result: &Value, root: &Path) -> Vec<String> {
    let items = match result {
        Value::Null => vec![],
        Value::Array(items) => items.clone(),
        single => vec![single.clone()],
    };
    items
        .iter()
        .filter_map(|item| {
            let uri = item["uri"].as_str().or(item["targetUri"].as_str())?;
            let range = if item.get("targetSelectionRange").is_some() {
                &item["targetSelectionRange"]
            } else {
                &item["range"]
            };
            Some(format_location(uri, range, root))
        })
        .collect()
}

pub(crate) fn format_hover(result: &Value) -> String {
    fn part(value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            Value::Object(o) => o
                .get("value")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            Value::Array(items) => items.iter().map(part).collect::<Vec<_>>().join("\n\n"),
            _ => String::new(),
        }
    }
    part(&result["contents"]).trim().to_string()
}

/// Document symbols come either as a tree of DocumentSymbols or a flat
/// list of SymbolInformation.
pub(crate) fn format_document_symbols(result: &Value) -> Vec<String> {
    fn walk(symbols: &[Value], depth: usize, out: &mut Vec<String>) {
        for s in symbols {
            let range = if s.get("location").is_some() {
                &s["location"]["range"]
            } else {
                &s["range"]
            };
            let start = range["start"]["line"].as_u64().unwrap_or(0) + 1;
            let end = range["end"]["line"].as_u64().unwrap_or(0) + 1;
            let detail = s["detail"]
                .as_str()
                .filter(|d| !d.is_empty())
                .map(|d| format!(" {}", d))
                .unwrap_or_default();
            out.push(format!(
                "{}{} {}{} (lines {}-{})",
                "  ".repeat(depth),
                symbol_kind(&s["kind"]),
                s["name"].as_str().unwrap_or("?"),
                detail,
                start,
                end
            ));
            if let Some(children) = s["children"].as_array() {
                walk(children, depth + 1, out);
            }
        }
    }
    let mut out = Vec::new();
    if let Some(symbols) = result.as_array() {
        walk(symbols, 0, &mut out);
    }
    out
}

pub(crate) fn format_workspace_symbols(result: &Value, root: &Path) -> Vec<String> {
    result
        .as_array()
        .map(|symbols| {
            symbols
                .iter()
                .map(|s| {
                    let location = &s["location"];
                    let place = match location["uri"].as_str() {
                        Some(uri) if location.get("range").is_some() => {
                            let path = uri_to_path(uri).unwrap_or_else(|| PathBuf::from(uri));
                            let line = location["range"]["start"]["line"].as_u64().unwrap_or(0);
                            format!("{}:{}", display_path(&path, root), line + 1)
                        }
                        Some(uri) => uri_to_path(uri)
                            .map(|p| display_path(&p, root))
                            .unwrap_or_else(|| uri.to_string()),
                        None => String::new(),
                    };
                    let container = s["containerName"]
                        .as_str()
                        .filter(|c| !c.is_empty())
                        .map(|c| format!(" (in {})", c))
                        .unwrap_or_default();
                    format!(
                        "{} {}{} - {}",
                        symbol_kind(&s["kind"]),
                        s["name"].as_str().unwrap_or("?"),
                        container,
                        place
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn format_diagnostics(diagnostics: &[Value], path: &str) -> Vec<String> {
    diagnostics
        .iter()
        .map(|d| {
            let severity = match d["severity"].as_u64() {
                Some(1) => "error",
                Some(2) => "warning",
                Some(3) => "info",
                Some(4) => "hint",
                _ => "diagnostic",
            };
            let code = match &d["code"] {
                Value::String(c) => format!(" [{}]", c),
                Value::Number(c) => format!(" [{}]", c),
                _ => String::new(),
            };
            let source = d["source"]
                .as_str()
                .map(|s| format!(" ({})", s))
                .unwrap_or_default();
            format!(
                "{}:{}:{}: {}: {}{}{}",
                path,
                d["range"]["start"]["line"].as_u64().unwrap_or(0) + 1,
                d["range"]["start"]["character"].as_u64().unwrap_or(0) + 1,
                severity,
                d["message"].as_str().unwrap_or("").trim(),
                code,
                source
            )
        })
        .collect()
}

fn limited(mut lines: Vec<String>, what: &str) -> String {
    if lines.is_empty() {
        return format!("No {}.", what);
    }
    let total = lines.len();
    lines.truncate(MAX_RESULTS);
    let mut out = lines.join("\n");
    if total > MAX_RESULTS {
        out.push_str(&format!(
            "\n[{} more {} not shown]",
            total - MAX_RESULTS,
            what
        ));
    }
    out
}

#[derive(Deserialize)]
struct Params {
    action: String,
    path: String,
    #[serde(default)]
    line: Option<u32>,
    #[serde(default)]
    symbol: Option<String>,
    #[serde(default)]
    column: Option<u32>,
    #[serde(default)]
    query: Option<String>,
}

/// The file an `lsp` call is about: relative to the current directory,
/// which it must stay inside (servers are rooted there).
fn project_file(cwd: &Path, path: &str) -> Result<PathBuf, String> {
    let relative = Path::new(path);
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| c == std::path::Component::ParentDir)
    {
        return Err(format!(
            "path '{}' must be relative to the current directory, without '..'",
            path
        ));
    }
    Ok(cwd.join(relative))
}

pub(crate) fn run(
    params_str: &str,
    ctx: &ToolContext,
    sandboxed: bool,
) -> Result<String, Box<dyn Error>> {
    let params: Params = serde_json::from_str(params_str)?;
    let cwd = ctx.cwd();
    let path = project_file(&cwd, &params.path)?;
    let client = client_for(&path, &cwd, sandboxed)?;
    let root = client.root.clone();
    let before = client.diagnostics_generation(&path_to_uri(&path));
    let (uri, text, sent) = client.sync(&path)?;
    let busy = client.wait_until_idle();
    let text_document = json!({"uri": uri});
    let position = || -> Result<Value, String> {
        position_in(
            &text,
            params.line.ok_or("give the line (1-based) of the symbol")?,
            params.symbol.as_deref(),
            params.column,
        )
    };

    let result = match params.action.as_str() {
        "definition" => {
            let result = client.request(
                "textDocument/definition",
                json!({"textDocument": text_document, "position": position()?}),
            )?;
            limited(format_locations(&result, &root), "definitions found")
        }
        "references" => {
            let result = client.request(
                "textDocument/references",
                json!({
                    "textDocument": text_document,
                    "position": position()?,
                    "context": {"includeDeclaration": true},
                }),
            )?;
            limited(format_locations(&result, &root), "references found")
        }
        "hover" => {
            let result = client.request(
                "textDocument/hover",
                json!({"textDocument": text_document, "position": position()?}),
            )?;
            let hover = format_hover(&result);
            if hover.is_empty() {
                "No information.".to_string()
            } else {
                hover
            }
        }
        "symbols" => {
            let result = client.request(
                "textDocument/documentSymbol",
                json!({"textDocument": text_document}),
            )?;
            limited(format_document_symbols(&result), "symbols found")
        }
        "workspace_symbols" => {
            let query = params
                .query
                .as_deref()
                .ok_or("give the query (a symbol name or part of one)")?;
            let result = client.request("workspace/symbol", json!({"query": query}))?;
            limited(format_workspace_symbols(&result, &root), "symbols found")
        }
        "diagnostics" => {
            // A file that was already open and unchanged may not get a new
            // publish, so just take what's there.
            let (diagnostics, timed_out) =
                client.wait_for_diagnostics(&uri, if sent { before } else { None });
            let mut out = limited(
                format_diagnostics(&diagnostics, &params.path),
                "problems reported",
            );
            if timed_out {
                out.push_str(&format!(
                    "\n(Note: {} didn't report diagnostics for this file within {}s; this may be incomplete.)",
                    client.name,
                    DIAGNOSTICS_TIMEOUT.as_secs()
                ));
            }
            out
        }
        other => {
            return Err(format!(
                "unknown action '{}': use definition, references, hover, symbols, workspace_symbols or diagnostics",
                other
            )
            .into());
        }
    };

    let mut result = result;
    if busy {
        result.push_str(&format!(
            "\n(Note: {} is still loading the project, so this may be incomplete - try again shortly.)",
            client.name
        ));
    }
    ctx.println(&format!(
        "{} {} {}",
        client.name, params.action, params.path
    ));
    for line in result.lines().take(10) {
        ctx.println(&format!("  {}", line));
    }
    if result.lines().count() > 10 {
        ctx.println(&format!("  ... and {} more", result.lines().count() - 10));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_server_by_extension() {
        let servers = builtin_servers();
        let rs = find_server(&servers, Path::new("src/main.rs")).unwrap();
        assert_eq!(rs.commands[0][0], "rust-analyzer");
        let cpp = find_server(&servers, Path::new("a/B.HPP")).unwrap();
        assert_eq!(cpp.commands[0][0], "clangd");
        assert!(find_server(&servers, Path::new("README.md")).is_none());
        assert!(find_server(&servers, Path::new("Makefile")).is_none());
    }

    fn overrides(json: Value) -> HashMap<String, Option<ServerConfig>> {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn test_merge_servers_adds_replaces_and_removes() {
        let servers = merge_servers(
            builtin_servers(),
            &overrides(json!({
                "zig": {"command": ["zls"], "extensions": [".ZIG"]},
                "python": {"command": ["/opt/bin/pylsp", "-v"], "extensions": ["py", "pyi"],
                           "settings": {"pylsp": {"plugins": {}}}},
                "go": null,
            })),
        )
        .unwrap();

        let zig = find_server(&servers, Path::new("a.zig")).unwrap();
        assert_eq!(zig.commands, vec![vec!["zls".to_string()]]);
        assert_eq!(language_id(zig, Path::new("a.zig")), "zig");

        let py = find_server(&servers, Path::new("x.pyi")).unwrap();
        assert_eq!(py.name, "python");
        assert_eq!(
            py.commands,
            vec![vec!["/opt/bin/pylsp".to_string(), "-v".to_string()]]
        );
        assert_eq!(py.settings, json!({"pylsp": {"plugins": {}}}));
        assert_eq!(language_id(py, Path::new("x.py")), "python");

        assert!(find_server(&servers, Path::new("main.go")).is_none());
        // Untouched built-ins stay.
        assert!(find_server(&servers, Path::new("main.rs")).is_some());
    }

    #[test]
    fn test_merge_servers_rejects_conflicts_and_empty_entries() {
        let err = merge_servers(
            builtin_servers(),
            &overrides(json!({"ccls": {"command": ["ccls"], "extensions": ["cpp"]}})),
        )
        .unwrap_err();
        assert!(err.contains("\"c\" and \"ccls\" handle .cpp"), "{err}");
        // Replacing the built-in instead of adding a second server is fine.
        assert!(
            merge_servers(
                builtin_servers(),
                &overrides(json!({
                    "c": null,
                    "ccls": {"command": ["ccls"], "extensions": ["c", "cpp"]},
                })),
            )
            .is_ok()
        );
        let err = merge_servers(
            builtin_servers(),
            &overrides(json!({"x": {"command": [], "extensions": ["x"]}})),
        )
        .unwrap_err();
        assert!(err.contains("\"command\" is empty"), "{err}");
    }

    #[test]
    fn test_server_config_rejects_unknown_fields() {
        let result: Result<HashMap<String, Option<ServerConfig>>, _> = serde_json::from_value(
            json!({"x": {"command": ["x"], "extensions": ["x"], "extension": ["y"]}}),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_uri_round_trip_with_special_characters() {
        let path = Path::new("/home/me/my project/ä#1.rs");
        let uri = path_to_uri(path);
        assert_eq!(uri, "file:///home/me/my%20project/%C3%A4%231.rs");
        assert_eq!(uri_to_path(&uri).unwrap(), path);
        assert!(uri_to_path("https://example.com").is_none());
    }

    #[test]
    fn test_message_framing_round_trip() {
        let mut buf = Vec::new();
        write_message(&mut buf, &json!({"id": 1, "result": "é"})).unwrap();
        write_message(&mut buf, &json!({"method": "x"})).unwrap();
        let mut input = std::io::Cursor::new(buf);
        assert_eq!(read_message(&mut input).unwrap().unwrap()["result"], "é");
        assert_eq!(read_message(&mut input).unwrap().unwrap()["method"], "x");
        assert!(read_message(&mut input).unwrap().is_none());
    }

    #[test]
    fn test_position_from_symbol_counts_utf16() {
        let text = "fn main() {\n    let é = foo(1);\n}\n";
        assert_eq!(
            position_in(text, 2, Some("foo"), None).unwrap(),
            json!({"line": 1, "character": 12})
        );
        assert_eq!(
            position_in(text, 1, None, Some(4)).unwrap(),
            json!({"line": 0, "character": 3})
        );
        let err = position_in(text, 2, Some("bar"), None).unwrap_err();
        assert!(err.contains("let é = foo(1);"), "{err}");
        assert!(position_in(text, 9, Some("foo"), None).is_err());
        assert!(position_in(text, 1, None, None).is_err());
    }

    #[test]
    fn test_format_locations_accepts_every_shape() {
        let root = Path::new("/nonexistent/root");
        let range =
            json!({"start": {"line": 4, "character": 2}, "end": {"line": 4, "character": 5}});
        let single = json!({"uri": "file:///nonexistent/root/src/a.rs", "range": range});
        assert_eq!(format_locations(&single, root), vec!["src/a.rs:5:3: "]);
        let links = json!([{
            "targetUri": "file:///elsewhere/b.rs",
            "targetRange": range,
            "targetSelectionRange": {"start": {"line": 0, "character": 0}},
        }]);
        assert_eq!(
            format_locations(&links, root),
            vec!["/elsewhere/b.rs:1:1: "]
        );
        assert!(format_locations(&Value::Null, root).is_empty());
    }

    #[test]
    fn test_format_document_symbols_nests_children() {
        let symbols = json!([{
            "name": "Foo", "kind": 23,
            "range": {"start": {"line": 0}, "end": {"line": 9}},
            "children": [{
                "name": "bar", "kind": 6, "detail": "fn(&self)",
                "range": {"start": {"line": 2}, "end": {"line": 4}},
            }],
        }]);
        assert_eq!(
            format_document_symbols(&symbols),
            vec![
                "struct Foo (lines 1-10)",
                "  method bar fn(&self) (lines 3-5)"
            ]
        );
    }

    #[test]
    fn test_format_hover_flattens_every_content_shape() {
        assert_eq!(
            format_hover(&json!({"contents": {"kind": "markdown", "value": "`x: i32`"}})),
            "`x: i32`"
        );
        assert_eq!(
            format_hover(&json!({"contents": ["a", {"language": "rust", "value": "b"}]})),
            "a\n\nb"
        );
    }

    #[test]
    fn test_format_diagnostics() {
        let d = json!([{
            "range": {"start": {"line": 2, "character": 4}},
            "severity": 1, "code": "E0308", "source": "rustc",
            "message": "mismatched types",
        }]);
        assert_eq!(
            format_diagnostics(d.as_array().unwrap(), "src/a.rs"),
            vec!["src/a.rs:3:5: error: mismatched types [E0308] (rustc)"]
        );
    }

    #[test]
    fn test_sandboxed_server_args_bind_only_the_root_writable() {
        let args = sandboxed_server_args("/proj", "/usr/bin/clangd", &["--log=error".to_string()]);
        let binds: Vec<_> = args
            .windows(3)
            .filter(|w| w[0] == "--bind")
            .map(|w| (w[1].clone(), w[2].clone()))
            .collect();
        assert_eq!(binds, vec![("/proj".to_string(), "/proj".to_string())]);
        assert!(args.contains(&"--unshare-net".to_string()));
        assert!(args.contains(&"--die-with-parent".to_string()));
        assert_eq!(
            &args[args.len() - 3..],
            ["--", "/usr/bin/clangd", "--log=error"]
        );
    }

    #[test]
    fn test_project_file_stays_inside_current_directory() {
        let cwd = Path::new("/proj");
        assert_eq!(
            project_file(cwd, "src/main.rs"),
            Ok(PathBuf::from("/proj/src/main.rs"))
        );
        assert!(project_file(cwd, "/etc/passwd").is_err());
        assert!(project_file(cwd, "../x.rs").is_err());
    }

    /// End to end against a real server, when clangd is installed (it's
    /// quick to start and needs no project setup). Runs unsandboxed so it
    /// doesn't also depend on bubblewrap.
    #[test]
    fn test_clangd_end_to_end() {
        if crate::latex_kitty::resolve_on_path("clangd").is_none() {
            eprintln!("clangd not installed, skipping");
            return;
        }
        let dir = "_test_lsp_c";
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(format!("{dir}/compile_flags.txt"), "-xc\n").unwrap();
        let source = "int add(int a, int b) { return a + b; }\n\
                      \n\
                      int main(void) {\n\
                      \x20   int x = add(1, 2);\n\
                      \x20   return add(x, 3);\n\
                      }\n";
        std::fs::write(format!("{dir}/main.c"), source).unwrap();
        let ctx = ToolContext::new(|_: &str| {});
        let call = |params: Value| run(&params.to_string(), &ctx, false).unwrap();
        let file = format!("{dir}/main.c");

        let out = call(json!({"action": "definition", "path": file, "line": 4, "symbol": "add"}));
        assert_eq!(
            out,
            format!("{dir}/main.c:1:5: int add(int a, int b) {{ return a + b; }}")
        );

        let out = call(json!({"action": "references", "path": file, "line": 1, "symbol": "add"}));
        assert_eq!(out.lines().count(), 3, "{out}");

        let out = call(json!({"action": "hover", "path": file, "line": 4, "symbol": "add"}));
        assert!(out.contains("int add(int a, int b)"), "{out}");

        let out = call(json!({"action": "symbols", "path": file}));
        assert!(out.contains("function add"), "{out}");
        assert!(out.contains("function main"), "{out}");

        let out = call(json!({"action": "diagnostics", "path": file}));
        assert_eq!(out, "No problems reported.");

        // An edit on disk is picked up by the next call.
        std::fs::write(&file, source.replace("add(x, 3)", "add(x)")).unwrap();
        let out = call(json!({"action": "diagnostics", "path": file}));
        assert!(out.contains("main.c:5:") && out.contains("error"), "{out}");

        std::fs::remove_dir_all(dir).unwrap();
    }
}
