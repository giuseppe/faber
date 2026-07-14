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

use log::{debug, info, warn};
use reqwest::blocking::Client as ReqwestClient;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, mpsc};

#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct JsonRpcNotification {
    jsonrpc: String,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    #[allow(dead_code)]
    id: Option<u64>,
    result: Option<serde_json::Value>,
    error: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct McpToolDef {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(rename = "inputSchema")]
    input_schema: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct McpToolsListResult {
    tools: Vec<McpToolDef>,
}

#[derive(Deserialize, Debug)]
struct McpToolCallContent {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize, Debug)]
struct McpToolCallResult {
    content: Vec<McpToolCallContent>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct McpServerConfig {
    pub command: Option<String>,
    #[serde(default)]
    pub args: Option<Vec<String>>,
    #[serde(default)]
    pub env: Option<HashMap<String, String>>,
    pub url: Option<String>,
    pub sse: Option<String>,
    #[serde(default)]
    pub headers: Option<HashMap<String, String>>,
}

trait McpTransport: Send + Sync {
    fn send_request(&self, request: &JsonRpcRequest) -> Result<JsonRpcResponse, Box<dyn Error>>;
    fn send_notification(&self, notification: &JsonRpcNotification) -> Result<(), Box<dyn Error>>;
    fn shutdown(&self);
}

struct StdioTransport {
    stdin: Mutex<ChildStdin>,
    stdout: Mutex<BufReader<ChildStdout>>,
    child: Mutex<Child>,
}

impl McpTransport for StdioTransport {
    fn send_request(&self, request: &JsonRpcRequest) -> Result<JsonRpcResponse, Box<dyn Error>> {
        let json = serde_json::to_string(request)?;
        debug!("MCP stdio send: {}", json);

        let mut stdin = self
            .stdin
            .lock()
            .map_err(|e| format!("stdin lock: {}", e))?;
        writeln!(stdin, "{}", json)?;
        stdin.flush()?;
        drop(stdin);

        let mut stdout = self
            .stdout
            .lock()
            .map_err(|e| format!("stdout lock: {}", e))?;
        let mut line = String::new();
        stdout.read_line(&mut line)?;
        debug!("MCP stdio recv: {}", line.trim());

        let response: JsonRpcResponse = serde_json::from_str(line.trim())?;
        Ok(response)
    }

    fn send_notification(&self, notification: &JsonRpcNotification) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string(notification)?;
        debug!("MCP stdio notification: {}", json);

        let mut stdin = self
            .stdin
            .lock()
            .map_err(|e| format!("stdin lock: {}", e))?;
        writeln!(stdin, "{}", json)?;
        stdin.flush()?;
        Ok(())
    }

    fn shutdown(&self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct HttpTransport {
    url: String,
    client: ReqwestClient,
    headers: HeaderMap,
}

impl McpTransport for HttpTransport {
    fn send_request(&self, request: &JsonRpcRequest) -> Result<JsonRpcResponse, Box<dyn Error>> {
        let json = serde_json::to_string(request)?;
        debug!("MCP HTTP POST to {}: {}", self.url, json);

        let resp = self
            .client
            .post(&self.url)
            .headers(self.headers.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(json)
            .send()?;

        let status = resp.status();
        let body = resp.text()?;
        debug!("MCP HTTP response ({}): {}", status, body);

        if !status.is_success() {
            return Err(format!("MCP HTTP error {}: {}", status, body).into());
        }

        let response: JsonRpcResponse = serde_json::from_str(&body)?;
        Ok(response)
    }

    fn send_notification(&self, notification: &JsonRpcNotification) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string(notification)?;
        debug!("MCP HTTP notification to {}: {}", self.url, json);

        let resp = self
            .client
            .post(&self.url)
            .headers(self.headers.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(json)
            .send()?;

        if !resp.status().is_success() {
            warn!(
                "MCP HTTP notification failed ({}): {}",
                resp.status(),
                resp.text().unwrap_or_default()
            );
        }
        Ok(())
    }

    fn shutdown(&self) {}
}

struct SseTransport {
    post_url: String,
    client: ReqwestClient,
    headers: HeaderMap,
    responses: Mutex<mpsc::Receiver<JsonRpcResponse>>,
    shutdown_flag: Arc<AtomicBool>,
}

impl SseTransport {
    fn connect(sse_url: &str, headers: HeaderMap) -> Result<Self, Box<dyn Error>> {
        let client = ReqwestClient::new();

        let resp = client
            .get(sse_url)
            .headers(headers.clone())
            .header("Accept", "text/event-stream")
            .send()?;

        if !resp.status().is_success() {
            return Err(format!("SSE connection failed ({})", resp.status()).into());
        }

        let base_url = {
            let parsed = reqwest::Url::parse(sse_url)?;
            format!("{}://{}", parsed.scheme(), parsed.authority())
        };

        let reader = BufReader::new(resp);

        let (endpoint_tx, endpoint_rx) = mpsc::channel::<String>();
        let (response_tx, response_rx) = mpsc::channel::<JsonRpcResponse>();
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown_flag.clone();
        let base_url_clone = base_url.clone();

        std::thread::spawn(move || {
            let mut event_type = String::new();
            let mut data_buf = String::new();
            let mut endpoint_sent = false;
            let mut lines = reader.lines();

            while !shutdown_clone.load(Ordering::Relaxed) {
                let line = match lines.next() {
                    Some(Ok(l)) => l,
                    _ => break,
                };

                if line.is_empty() {
                    if !event_type.is_empty() && !data_buf.is_empty() {
                        let data = data_buf.trim().to_string();
                        match event_type.as_str() {
                            "endpoint" => {
                                if !endpoint_sent {
                                    let url = if data.starts_with('/') {
                                        format!("{}{}", base_url_clone, data)
                                    } else {
                                        data
                                    };
                                    let _ = endpoint_tx.send(url);
                                    endpoint_sent = true;
                                }
                            }
                            "message" => {
                                if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(&data) {
                                    let _ = response_tx.send(resp);
                                }
                            }
                            _ => {
                                debug!("SSE unknown event: {}", event_type);
                            }
                        }
                    }
                    event_type.clear();
                    data_buf.clear();
                    continue;
                }

                if let Some(val) = line.strip_prefix("event: ") {
                    event_type = val.trim().to_string();
                } else if let Some(val) = line.strip_prefix("data: ") {
                    if !data_buf.is_empty() {
                        data_buf.push('\n');
                    }
                    data_buf.push_str(val);
                }
            }
        });

        let url = endpoint_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .map_err(|_| "SSE: timed out waiting for endpoint event")?;
        info!("SSE transport: POST endpoint is {}", url);

        Ok(SseTransport {
            post_url: url,
            client,
            headers,
            responses: Mutex::new(response_rx),
            shutdown_flag,
        })
    }
}

impl McpTransport for SseTransport {
    fn send_request(&self, request: &JsonRpcRequest) -> Result<JsonRpcResponse, Box<dyn Error>> {
        let json = serde_json::to_string(request)?;
        debug!("MCP SSE POST to {}: {}", self.post_url, json);

        let resp = self
            .client
            .post(&self.post_url)
            .headers(self.headers.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(json)
            .send()?;

        if !resp.status().is_success() {
            return Err(format!(
                "MCP SSE POST error {}: {}",
                resp.status(),
                resp.text().unwrap_or_default()
            )
            .into());
        }

        let rx = self
            .responses
            .lock()
            .map_err(|e| format!("response lock: {}", e))?;
        let response = rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .map_err(|_| "SSE: timed out waiting for response")?;
        Ok(response)
    }

    fn send_notification(&self, notification: &JsonRpcNotification) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string(notification)?;
        debug!("MCP SSE notification POST to {}: {}", self.post_url, json);

        let resp = self
            .client
            .post(&self.post_url)
            .headers(self.headers.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(json)
            .send()?;

        if !resp.status().is_success() {
            warn!(
                "MCP SSE notification failed ({}): {}",
                resp.status(),
                resp.text().unwrap_or_default()
            );
        }
        Ok(())
    }

    fn shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);
    }
}

struct McpServer {
    name: String,
    transport: Box<dyn McpTransport>,
    next_id: AtomicU64,
}

impl McpServer {
    fn launch(name: &str, config: &McpServerConfig) -> Result<Self, Box<dyn Error>> {
        let transport: Box<dyn McpTransport> = if let Some(ref command) = config.command {
            let args = config.args.as_deref().unwrap_or(&[]);
            info!("Launching MCP server '{}': {} {:?}", name, command, args);

            let mut cmd = Command::new(command);
            cmd.args(args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null());

            if let Some(ref env) = config.env {
                for (k, v) in env {
                    cmd.env(k, v);
                }
            }

            let mut child = cmd.spawn().map_err(|e| {
                format!(
                    "Failed to launch MCP server '{}' ({}): {}",
                    name, command, e
                )
            })?;

            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| format!("MCP server '{}': no stdin", name))?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| format!("MCP server '{}': no stdout", name))?;

            Box::new(StdioTransport {
                stdin: Mutex::new(stdin),
                stdout: Mutex::new(BufReader::new(stdout)),
                child: Mutex::new(child),
            })
        } else if let Some(ref url) = config.url {
            info!("Connecting to remote MCP server '{}': {}", name, url);

            let mut header_map = HeaderMap::new();
            if let Some(ref headers) = config.headers {
                for (k, v) in headers {
                    header_map.insert(
                        reqwest::header::HeaderName::from_bytes(k.as_bytes())?,
                        HeaderValue::from_str(v)?,
                    );
                }
            }

            Box::new(HttpTransport {
                url: url.clone(),
                client: ReqwestClient::new(),
                headers: header_map,
            })
        } else if let Some(ref sse_url) = config.sse {
            info!("Connecting to SSE MCP server '{}': {}", name, sse_url);

            let mut header_map = HeaderMap::new();
            if let Some(ref headers) = config.headers {
                for (k, v) in headers {
                    header_map.insert(
                        reqwest::header::HeaderName::from_bytes(k.as_bytes())?,
                        HeaderValue::from_str(v)?,
                    );
                }
            }

            Box::new(SseTransport::connect(sse_url, header_map)?)
        } else {
            return Err(format!(
                "MCP server '{}': must have 'command', 'url', or 'sse'",
                name
            )
            .into());
        };

        let server = McpServer {
            name: name.to_string(),
            transport,
            next_id: AtomicU64::new(1),
        };

        server.initialize()?;
        Ok(server)
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    fn send_request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, Box<dyn Error>> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: self.next_id(),
            method: method.to_string(),
            params,
        };

        let response = self.transport.send_request(&request)?;

        if let Some(error) = response.error {
            return Err(format!("MCP server '{}' error: {}", self.name, error).into());
        }

        response
            .result
            .ok_or_else(|| format!("MCP server '{}': no result in response", self.name).into())
    }

    fn initialize(&self) -> Result<(), Box<dyn Error>> {
        let params = serde_json::json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {
                "name": "swarmblabla",
                "version": "0.1.0"
            }
        });

        let result = self.send_request("initialize", Some(params))?;
        info!(
            "MCP server '{}' initialized: {}",
            self.name,
            result
                .get("serverInfo")
                .and_then(|s| s.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("unknown")
        );

        let notification = JsonRpcNotification {
            jsonrpc: "2.0".to_string(),
            method: "notifications/initialized".to_string(),
            params: None,
        };
        self.transport.send_notification(&notification)?;

        Ok(())
    }

    fn list_tools(&self) -> Result<Vec<McpToolDef>, Box<dyn Error>> {
        let result = self.send_request("tools/list", None)?;
        let tools_result: McpToolsListResult = serde_json::from_value(result)?;
        Ok(tools_result.tools)
    }

    fn call_tool(&self, name: &str, arguments: &str) -> Result<String, Box<dyn Error>> {
        let args: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();

        let params = serde_json::json!({
            "name": name,
            "arguments": args
        });

        let result = self.send_request("tools/call", Some(params))?;
        let call_result: McpToolCallResult = serde_json::from_value(result)?;

        let text: Vec<String> = call_result
            .content
            .into_iter()
            .filter(|c| c.content_type == "text")
            .filter_map(|c| c.text)
            .collect();

        Ok(text.join("\n"))
    }

    fn shutdown(&self) {
        self.transport.shutdown();
    }
}

struct McpToolInfo {
    server_index: usize,
    original_name: String,
    schema: serde_json::Value,
}

pub struct McpManager {
    servers: Vec<McpServer>,
    tools: RwLock<HashMap<String, McpToolInfo>>,
}

impl McpManager {
    pub fn new(configs: HashMap<String, McpServerConfig>) -> Result<Self, Box<dyn Error>> {
        let mut servers = Vec::new();

        for (name, config) in &configs {
            match McpServer::launch(name, config) {
                Ok(server) => servers.push(server),
                Err(e) => {
                    warn!("Failed to launch MCP server '{}': {}", name, e);
                    return Err(e);
                }
            }
        }

        let manager = McpManager {
            servers,
            tools: RwLock::new(HashMap::new()),
        };

        manager.refresh()?;
        Ok(manager)
    }

    pub fn refresh(&self) -> Result<usize, Box<dyn Error>> {
        let mut new_tools = HashMap::new();

        for (idx, server) in self.servers.iter().enumerate() {
            match server.list_tools() {
                Ok(tool_defs) => {
                    for tool_def in tool_defs {
                        let prefixed_name = format!("mcp_{}_{}", server.name, tool_def.name);

                        let schema = serde_json::json!({
                            "type": "function",
                            "function": {
                                "name": prefixed_name,
                                "description": tool_def.description.unwrap_or_default(),
                                "parameters": tool_def.input_schema.unwrap_or(serde_json::json!({"type": "object", "properties": {}}))
                            }
                        });

                        debug!(
                            "MCP tool registered: {} (from server '{}')",
                            prefixed_name, server.name
                        );

                        new_tools.insert(
                            prefixed_name,
                            McpToolInfo {
                                server_index: idx,
                                original_name: tool_def.name,
                                schema,
                            },
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to list tools from MCP server '{}': {}",
                        server.name, e
                    );
                }
            }
        }

        let count = new_tools.len();
        info!(
            "MCP refresh: {} tools across {} servers",
            count,
            self.servers.len()
        );

        let mut tools = self
            .tools
            .write()
            .map_err(|e| format!("tools lock: {}", e))?;
        *tools = new_tools;

        Ok(count)
    }

    pub fn has_tools(&self) -> bool {
        self.tools.read().map(|t| !t.is_empty()).unwrap_or(false)
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tools
            .read()
            .map(|t| t.contains_key(name))
            .unwrap_or(false)
    }

    pub fn call_tool(&self, name: &str, arguments: &str) -> Result<String, Box<dyn Error>> {
        let tools = self
            .tools
            .read()
            .map_err(|e| format!("tools lock: {}", e))?;
        let info = tools
            .get(name)
            .ok_or_else(|| format!("MCP tool '{}' not found", name))?;

        let server = &self.servers[info.server_index];
        server.call_tool(&info.original_name, arguments)
    }

    pub fn get_tool_schemas(&self) -> Vec<serde_json::Value> {
        self.tools
            .read()
            .map(|t| t.values().map(|info| info.schema.clone()).collect())
            .unwrap_or_default()
    }

    pub fn shutdown(&self) {
        for server in &self.servers {
            server.shutdown();
        }
    }
}
