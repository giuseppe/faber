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

//! A fan-out run end to end, by the faber binary itself, against a model
//! server that misbehaves the ways real ones do: going silent partway
//! through a response, rate limiting, failing, answering slowly. Every
//! item must still get its result, without more requests in flight than
//! allowed - and the whole run must end: a fan-out that stops making
//! progress fails the test, with what both sides were doing.
//!
//! A process of its own, so its request limit and stall timeout are its
//! own too.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const WORKER_MARK: &str = "You are a worker in a fan-out";

/// What the fake model server knows, and saw.
#[derive(Default)]
struct Server {
    /// Requests per worker item, so far.
    attempts: Mutex<HashMap<String, usize>>,
    /// Requests being answered now, and the most there were at once.
    answering: AtomicUsize,
    most: AtomicUsize,
    /// The fan-out's result, as the main agent got it.
    fan_out_result: Mutex<Option<String>>,
    log: Mutex<Vec<String>>,
}

impl Server {
    fn note(&self, line: String) {
        self.log.lock().unwrap().push(line);
    }
}

/// A streamed response of `chunks` (text deltas), then the end: each
/// chunk written as it is, so a test controls the timing.
fn sse_head() -> &'static str {
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
}

fn sse_delta(field: &str, text: &str) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({"choices": [{"delta": {field: text}, "finish_reason": null}]})
    )
}

fn sse_end(finish: &str) -> String {
    format!(
        "data: {}\n\ndata: [DONE]\n\n",
        serde_json::json!({"choices": [{"delta": {}, "finish_reason": finish}]})
    )
}

fn sse_tool_call(name: &str, arguments: serde_json::Value) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": format!("call_{}", name), "type": "function",
            "function": {"name": name, "arguments": arguments.to_string()}
        }]}, "finish_reason": null}]})
    )
}

fn plain_response(status: &str, extra_headers: &str) -> String {
    let body = r#"{"error":{"message":"try again"}}"#;
    format!(
        "HTTP/1.1 {}\r\n{}Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{}",
        status,
        extra_headers,
        body.len(),
        body
    )
}

/// The request's method and body.
fn read_request(stream: &mut TcpStream) -> Option<(String, String)> {
    let mut request = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = stream.read(&mut buf).ok()?;
        if n == 0 {
            return None;
        }
        request.extend_from_slice(&buf[..n]);
        let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
            continue;
        };
        let head = String::from_utf8_lossy(&request[..end]).to_string();
        let length = head
            .lines()
            .find_map(|l| {
                l.to_lowercase()
                    .strip_prefix("content-length:")
                    .map(String::from)
            })
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        while request.len() < end + 4 + length {
            let n = stream.read(&mut buf).ok()?;
            if n == 0 {
                return None;
            }
            request.extend_from_slice(&buf[..n]);
        }
        let method = head.split_whitespace().next().unwrap_or("").to_string();
        let body = String::from_utf8_lossy(&request[end + 4..end + 4 + length]).to_string();
        return Some((method, body));
    }
}

/// Answers one request, as the main agent's model or a worker's.
fn answer(server: &Server, mut stream: TcpStream) {
    let Some((method, body)) = read_request(&mut stream) else {
        return;
    };
    if method != "POST" {
        // e.g. the model listing faber looks up the context window in.
        let _ = stream.write_all(plain_response("404 Not Found", "").as_bytes());
        return;
    }
    let request: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
    let messages = request["messages"].as_array().cloned().unwrap_or_default();
    let is_worker = messages.iter().any(|m| {
        m["content"]
            .as_str()
            .is_some_and(|c| c.contains(WORKER_MARK))
    });
    let last = messages.last().cloned().unwrap_or_default();

    let now = server.answering.fetch_add(1, Ordering::SeqCst) + 1;
    server.most.fetch_max(now, Ordering::SeqCst);
    // Stops counting before the last of the answer is written: once it's
    // read, the client may let its request slot go.
    let done = || {
        server.answering.fetch_sub(1, Ordering::SeqCst);
    };

    if !is_worker {
        if last["role"] == "tool" {
            *server.fan_out_result.lock().unwrap() = last["content"].as_str().map(String::from);
            server.note("main: got the fan-out's result".to_string());
            done();
            let _ = stream.write_all(
                format!(
                    "{}{}{}",
                    sse_head(),
                    sse_delta("content", "all done"),
                    sse_end("stop")
                )
                .as_bytes(),
            );
        } else {
            server.note("main: asking for the fan-out".to_string());
            let items = [
                "plain-1",
                "stall-2",
                "ratelimited-3",
                "flaky-4",
                "slow-5",
                "tool-6",
                "plain-7",
                "stall-8",
            ];
            let call = sse_tool_call(
                "fan_out",
                serde_json::json!({"items": items, "prompt": "review {item}", "max_parallel": 4}),
            );
            done();
            let _ = stream
                .write_all(format!("{}{}{}", sse_head(), call, sse_end("tool_calls")).as_bytes());
        }
        return;
    }

    let item = messages
        .iter()
        .filter_map(|m| m["content"].as_str())
        .find_map(|c| c.strip_prefix("review "))
        .unwrap_or("?")
        .to_string();
    let attempt = {
        let mut attempts = server.attempts.lock().unwrap();
        let n = attempts.entry(item.clone()).or_insert(0);
        *n += 1;
        *n
    };
    let kind = item.split('-').next().unwrap_or("").to_string();
    server.note(format!("{}: request {}", item, attempt));
    let ok = || {
        format!(
            "{}{}{}{}",
            sse_head(),
            sse_delta("reasoning_content", "Looking at it."),
            sse_delta("content", &format!("reviewed {}", item)),
            sse_end("stop")
        )
    };
    match kind.as_str() {
        // Starts answering, then goes silent with the connection open.
        "stall" if attempt == 1 => {
            let _ = stream.write_all(sse_head().as_bytes());
            let _ = stream.write_all(sse_delta("reasoning_content", "Let me think").as_bytes());
            let _ = stream.flush();
            done();
            thread::sleep(Duration::from_secs(300));
        }
        "ratelimited" if attempt <= 2 => {
            done();
            let _ = stream.write_all(
                plain_response("429 Too Many Requests", "Retry-After: 0\r\n").as_bytes(),
            );
        }
        "flaky" if attempt == 1 => {
            done();
            let _ = stream.write_all(plain_response("500 Internal Server Error", "").as_bytes());
        }
        // Slow, but never silent for long: not a stall.
        "slow" => {
            let _ = stream.write_all(sse_head().as_bytes());
            for _ in 0..8 {
                let _ = stream.write_all(sse_delta("reasoning_content", "hmm ").as_bytes());
                let _ = stream.flush();
                thread::sleep(Duration::from_millis(300));
            }
            done();
            let _ = stream.write_all(
                format!(
                    "{}{}",
                    sse_delta("content", &format!("reviewed {}", item)),
                    sse_end("stop")
                )
                .as_bytes(),
            );
        }
        // A tool call first, then the answer.
        "tool" if last["role"] != "tool" => {
            done();
            let _ = stream.write_all(
                format!(
                    "{}{}{}",
                    sse_head(),
                    sse_tool_call("read_file", serde_json::json!({"path": "notes.txt"})),
                    sse_end("tool_calls")
                )
                .as_bytes(),
            );
        }
        _ => {
            done();
            let _ = stream.write_all(ok().as_bytes());
        }
    }
}

/// Runs the fake model server; its endpoint.
fn serve(server: Arc<Server>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let server = server.clone();
            thread::spawn(move || answer(&server, stream));
        }
    });
    endpoint
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// The faber process, killed however the test ends.
struct Faber(Child);

impl Drop for Faber {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn http(method: &str, url: &str, body: Option<serde_json::Value>) -> Option<serde_json::Value> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?;
    let request = match method {
        "POST" => client.post(url).json(&body.unwrap_or_default()),
        _ => client.get(url),
    };
    request.send().ok()?.json().ok()
}

#[test]
fn a_fan_out_against_a_misbehaving_model_server_gets_every_result_and_ends() {
    let base: PathBuf =
        std::env::temp_dir().join(format!("faber_fan_out_http_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let work = base.join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("notes.txt"), "nothing to see\n").unwrap();

    let server = Arc::new(Server::default());
    let endpoint = serve(server.clone());
    let port = free_port();
    let out = std::fs::File::create(base.join("faber.log")).unwrap();
    let _faber = Faber(
        Command::new(env!("CARGO_BIN_EXE_faber"))
            .current_dir(&work)
            // Nothing of the user's own: no config file, no keys.
            .env("HOME", &base)
            .env("XDG_CONFIG_HOME", base.join("config"))
            .env_remove("OPENAI_API_KEY")
            .args(["--endpoint", &endpoint, "--model", "test-model"])
            .args(["--db-path", base.join("faber.db").to_str().unwrap()])
            .args([
                "--max-parallel-requests",
                "2",
                "--stream-idle-timeout",
                "2s",
            ])
            .args([
                "serve",
                "--no-auth",
                "--bind",
                &format!("127.0.0.1:{}", port),
            ])
            .args(["--run-agent", "default"])
            .stdout(Stdio::from(out.try_clone().unwrap()))
            .stderr(Stdio::from(out))
            .spawn()
            .expect("faber starts"),
    );
    let api = format!("http://127.0.0.1:{}/api", port);
    let started = Instant::now();
    while http("GET", &format!("{}/tasks", api), None).is_none() {
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "faber serve didn't come up"
        );
        thread::sleep(Duration::from_millis(100));
    }
    http(
        "POST",
        &format!("{}/tasks", api),
        Some(serde_json::json!({"command": "review everything"})),
    )
    .expect("task added");

    let explain = |why: &str| -> String {
        format!(
            "{}\n--- the model server saw:\n{}\n--- faber said:\n{}",
            why,
            server.log.lock().unwrap().join("\n"),
            std::fs::read_to_string(base.join("faber.log")).unwrap_or_default()
        )
    };
    let limit = Duration::from_secs(90);
    let task = loop {
        let tasks = http("GET", &format!("{}/tasks", api), None).unwrap_or_default();
        let task = tasks
            .as_array()
            .and_then(|t| t.first())
            .cloned()
            .unwrap_or_default();
        if task["status"] == "done" {
            break task;
        }
        assert!(
            started.elapsed() < limit,
            "{}",
            explain(&format!("no end after {:?}: task {}", limit, task))
        );
        thread::sleep(Duration::from_millis(250));
    };

    let result = server
        .fan_out_result
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| {
            panic!(
                "{}",
                explain(&format!("the fan-out never returned: {}", task))
            )
        });
    for (i, item) in [
        "plain-1",
        "stall-2",
        "ratelimited-3",
        "flaky-4",
        "slow-5",
        "tool-6",
        "plain-7",
        "stall-8",
    ]
    .iter()
    .enumerate()
    {
        let expected = format!("## {}. {}\nreviewed {}", i + 1, item, item);
        assert!(
            result.contains(&expected),
            "{}",
            explain(&format!("{} missing from:\n{}", expected, result))
        );
    }
    let attempts = server.attempts.lock().unwrap().clone();
    for (item, at_least) in [
        ("stall-2", 2),
        ("stall-8", 2),
        ("ratelimited-3", 3),
        ("flaky-4", 2),
        ("tool-6", 2),
    ] {
        assert!(
            attempts.get(item).copied().unwrap_or(0) >= at_least,
            "{}",
            explain(&format!("{} wasn't retried: {:?}", item, attempts))
        );
    }
    let most = server.most.load(Ordering::SeqCst);
    assert!(
        most <= 2,
        "{}",
        explain(&format!("{} requests at once, with a limit of 2", most))
    );
    assert_eq!(
        task["last_outcome"],
        "succeeded",
        "{}",
        explain(&format!("the task didn't succeed: {}", task))
    );
    let _ = std::fs::remove_dir_all(&base);
}
