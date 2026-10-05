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

use log::{debug, info, trace, warn};
use reqwest::blocking::Client as ReqwestClient;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

/// Error type for when an operation is interrupted by the user (Ctrl-C)
#[derive(Debug)]
pub struct InterruptedError {
    pub message: String,
}

impl std::fmt::Display for InterruptedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for InterruptedError {}

impl InterruptedError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

/// Error returned when a request does not fit in the model's context window.
#[derive(Debug)]
pub struct ContextLengthError {
    pub message: String,
    /// The conversation as it stood when the request failed, including the
    /// tool calls already made during the current turn.
    pub history: Vec<Message>,
}

impl std::fmt::Display for ContextLengthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ContextLengthError {}

/// Phrases providers use to report that the prompt is larger than the
/// context window.  Rate-limit messages ("too many tokens per minute") are
/// deliberately not matched.
const CONTEXT_LENGTH_MARKERS: &[&str] = &[
    "context_length_exceeded",
    "maximum context length",
    "context length exceeded",
    "context window",
    "context size",
    "exceed_context_size",
    "prompt is too long",
    "input is too long",
    "reduce the length of the messages",
];

pub fn is_context_length_error(text: &str) -> bool {
    let text = text.to_lowercase();
    CONTEXT_LENGTH_MARKERS
        .iter()
        .any(|marker| text.contains(marker))
}

/// Turns `err` into a `ContextLengthError` when it reports a context
/// overflow, so callers can recover by shrinking `messages`.
fn classify_context_error(err: Box<dyn Error>, messages: &[Message]) -> Box<dyn Error> {
    let text = err.to_string();
    if is_context_length_error(&text) {
        Box::new(ContextLengthError {
            message: text,
            history: messages.to_vec(),
        })
    } else {
        err
    }
}

/// Normalize endpoint URL to ensure it ends with "/chat/completions"
pub fn normalize_endpoint(endpoint: &str) -> String {
    if endpoint.ends_with("/chat/completions") {
        endpoint.to_string()
    } else if endpoint.ends_with("/") {
        format!("{}chat/completions", endpoint)
    } else {
        format!("{}/chat/completions", endpoint)
    }
}

/// Check for Ctrl-C signal and return InterruptedError if found
/// The error a request loop stops with once it has used up its
/// `ToolContext::max_requests`.
pub fn request_budget_error(limit: usize) -> Box<dyn Error> {
    format!(
        "stopped after using up its budget of {} requests to the model",
        limit
    )
    .into()
}

/// Limits how many requests to the model are in flight at once. The
/// process has one (`request_limiter`), shared by the chat, sub-agents,
/// fan-out workers and task turns alike: a model server with few slots
/// (llama.cpp's --parallel) otherwise gets more concurrent requests than it
/// can serve, each splitting its context further.
pub struct RequestLimiter {
    /// (limit, in use); a limit of 0 means none.
    state: Mutex<(usize, usize)>,
    changed: std::sync::Condvar,
}

/// A request's place among the ones a `RequestLimiter` allows at once;
/// frees it when dropped.
pub struct RequestSlot<'a>(Option<&'a RequestLimiter>);

impl Drop for RequestSlot<'_> {
    fn drop(&mut self) {
        if let Some(limiter) = self.0 {
            limiter.state.lock().unwrap_or_else(|e| e.into_inner()).1 -= 1;
            limiter.changed.notify_one();
        }
    }
}

impl RequestLimiter {
    pub fn new(limit: usize) -> Self {
        Self {
            state: Mutex::new((limit, 0)),
            changed: std::sync::Condvar::new(),
        }
    }

    pub fn set_limit(&self, limit: usize) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).0 = limit;
        self.changed.notify_all();
    }

    /// Waits - interruptibly, telling `mode`'s progress handler once - for
    /// a free slot.
    fn acquire(
        &self,
        ctrl_c_rx: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
        mode: &ResponseMode,
        start_time: Instant,
    ) -> Result<RequestSlot<'_>, Box<dyn Error>> {
        let mut reported = false;
        loop {
            {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if state.0 == 0 {
                    return Ok(RequestSlot(None));
                }
                if state.1 < state.0 {
                    state.1 += 1;
                    return Ok(RequestSlot(Some(self)));
                }
                if reported {
                    let _ = self
                        .changed
                        .wait_timeout(state, Duration::from_millis(100))
                        .unwrap_or_else(|e| e.into_inner());
                }
            }
            if !reported {
                reported = true;
                if let ResponseMode::Streaming {
                    progress_handler, ..
                } = mode
                {
                    progress_handler(&ProgressInfo {
                        status: StatusUpdate::WaitingForSlot,
                        elapsed_ms: start_time.elapsed().as_millis() as u64,
                    })?;
                }
            }
            check_ctrl_c_signal(ctrl_c_rx)?;
        }
    }
}

/// The process's `RequestLimiter`.
fn request_limiter() -> &'static RequestLimiter {
    static LIMITER: std::sync::OnceLock<RequestLimiter> = std::sync::OnceLock::new();
    LIMITER.get_or_init(|| RequestLimiter::new(0))
}

/// Sets how many model requests may be in flight at once in this process
/// (0: unlimited).
/// The client every request to a model goes through. One for the whole
/// process: each client has its own runtime - a thread, an epoll and an
/// eventfd - and its own connections, so one per request had every agent
/// waiting for its turn (`request_limiter`) hold some, and a large fan-out
/// run out of file descriptors.
fn model_client() -> Result<ReqwestClient, Box<dyn Error>> {
    static CLIENT: std::sync::OnceLock<ReqwestClient> = std::sync::OnceLock::new();
    if let Some(client) = CLIENT.get() {
        return Ok(client.clone());
    }
    let client = ReqwestClient::builder()
        .timeout(Duration::from_secs(1000))
        .build()?;
    Ok(CLIENT.get_or_init(|| client).clone())
}

pub fn set_max_parallel_requests(limit: usize) {
    request_limiter().set_limit(limit);
}

/// How many model requests may be in flight at once (0: unlimited).
pub fn max_parallel_requests() -> usize {
    request_limiter()
        .state
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .0
}

/// Runs a blocking call - sending a request, reading a whole response -
/// on a helper thread, checking for Ctrl-C every 100ms meanwhile, so the
/// user isn't stuck waiting for a slow or hung server. If interrupted, the
/// call is abandoned: its thread finishes in the background and its result
/// is dropped. Without a `ctrl_c_rx`, it just runs the call.
fn interruptible<T: Send + 'static>(
    call: impl FnOnce() -> T + Send + 'static,
    ctrl_c_rx: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<T, Box<dyn Error>> {
    if ctrl_c_rx.is_none() {
        return Ok(call());
    }
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(call());
    });
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(result) => return Ok(result),
            Err(mpsc::RecvTimeoutError::Timeout) => check_ctrl_c_signal(ctrl_c_rx)?,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("the request thread died".into());
            }
        }
    }
}

fn check_ctrl_c_signal(
    ctrl_c_rx: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<(), Box<dyn Error>> {
    if let Some(rx) = ctrl_c_rx {
        if let Ok(receiver) = rx.lock() {
            if receiver.try_recv().is_ok() {
                debug!("Received Ctrl-C signal, exiting operation");
                return Err(
                    Box::new(InterruptedError::new("Operation interrupted by user"))
                        as Box<dyn Error>,
                );
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
pub struct Opts {
    pub max_tokens: Option<u32>,
    pub model: String,
    pub endpoint: String,
    pub tool_choice: Option<String>,
    pub api_key: Option<String>,
    pub max_retries: Option<usize>,
    pub retry_base_delay_secs: Option<u64>,
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
}

pub type ToolCallback = fn(&String, &crate::ToolContext) -> Result<String, Box<dyn Error>>;
pub type ToolsCollection = HashMap<String, ToolItem>;
#[derive(Clone)]
pub struct ToolItem {
    pub callback: ToolCallback,
    pub schema: String,
}

/// Reads the API key from the specified file path; a leading `~/` is the
/// home directory, as in the config file's example.
fn read_api_key(api_key_file: &String) -> Result<String, Box<dyn Error>> {
    let key_path = match (api_key_file.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => std::path::PathBuf::from(home).join(rest),
        _ => std::path::PathBuf::from(api_key_file),
    };

    let api_key = std::fs::read_to_string(&key_path)
        .map_err(|e| format!("can't read the API key file {}: {}", key_path.display(), e))?;

    let api_key = api_key.trim().to_string();
    if api_key.is_empty() {
        return Err(format!("API key file {} is empty", key_path.display()).into());
    }

    Ok(api_key)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u64>,
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

#[derive(Serialize, Debug)]
pub struct OpenAIRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Asks for `usage` in the final chunk of a streamed response: OpenAI
    /// and llama.cpp leave it out of streams otherwise, and without it
    /// there's no way to tell how full the context is getting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<serde_json::Value>,
    #[serde(flatten)]
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIErrorMetadata {
    pub raw: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIError {
    pub code: Option<u64>,
    pub message: String,
    pub metadata: Option<OpenAIErrorMetadata>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// Adds `usage` into `total`, treating a missing field on either side as
/// 0. `None` stays `None` only while nothing has reported usage at all, so
/// "no request reported usage" stays distinguishable from "zero tokens".
pub fn accumulate_usage(total: &mut Option<Usage>, usage: Option<&Usage>) {
    let Some(usage) = usage else {
        return;
    };
    let sum = |a: Option<u32>, b: Option<u32>| match (a, b) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0).saturating_add(b.unwrap_or(0))),
    };
    *total = Some(match total.take() {
        None => usage.clone(),
        Some(t) => Usage {
            prompt_tokens: sum(t.prompt_tokens, usage.prompt_tokens),
            completion_tokens: sum(t.completion_tokens, usage.completion_tokens),
            total_tokens: sum(t.total_tokens, usage.total_tokens),
        },
    });
}

#[derive(Deserialize, Debug)]
pub struct OpenAIResponse {
    pub error: Option<OpenAIError>,
    pub choices: Option<Vec<Choice>>,
    /// Usage of the final request only - its `prompt_tokens` reflects the
    /// current size of the conversation.
    pub usage: Option<Usage>,

    /// Usage summed over every request made to produce this response,
    /// including each intermediate tool-call round trip - what was
    /// actually billed for the turn.
    #[serde(skip_deserializing)]
    pub turn_usage: Option<Usage>,

    #[serde(skip_deserializing)]
    pub history: Vec<Message>,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
    pub native_finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct StreamingChoice {
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct StreamingToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    pub function: StreamingFunctionCall,
}

#[derive(Deserialize, Debug)]
pub struct StreamingFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct Delta {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<StreamingToolCall>>,
    // Add support for thinking content which might come in different fields
    pub thinking: Option<String>,
    pub reasoning_content: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct StreamingResponse {
    pub choices: Option<Vec<StreamingChoice>>,
    pub error: Option<OpenAIError>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone)]
pub enum StatusUpdate {
    Thinking,
    ToolAccumulating {
        name: String,
        arguments: String,
    },
    ToolStart {
        name: String,
        arguments: String,
    },
    ToolComplete {
        name: String,
        duration_ms: u64,
        /// The tool's result, as handed back to the model.
        output: String,
    },
    /// About to run several tool calls concurrently (see `run_tool_calls`).
    /// Their output is buffered and each is then reported through the
    /// usual `ToolStart`/`ToolComplete` pair, in order, once all finish.
    ToolBatchStart {
        names: Vec<String>,
    },
    /// Waiting for another request to finish first: no more may be in
    /// flight at once (see `set_max_parallel_requests`).
    WaitingForSlot,
    /// About to send a request and wait for the response - reported once
    /// per turn (the first request and every one that follows a tool call),
    /// with the size of the outgoing request body, so a long wait on a
    /// large prompt doesn't look indistinguishable from a hang.
    SendingRequest {
        bytes: usize,
    },
    StreamProcessing {
        bytes_read: usize,
        chunks_processed: u32,
    },
    Complete {
        usage: Option<Usage>,
    },
}

#[derive(Debug, Clone)]
pub struct ProgressInfo {
    pub status: StatusUpdate,
    pub elapsed_ms: u64,
}

pub enum ResponseMode {
    Complete,
    Streaming {
        stream_handler: Box<dyn Fn(&str) -> Result<(), Box<dyn Error>>>,
        /// Receives the model's reasoning ("thinking") tokens, which are shown
        /// but never stored in the conversation history.
        reasoning_handler: Box<dyn Fn(&str) -> Result<(), Box<dyn Error>>>,
        progress_handler: Box<dyn Fn(&ProgressInfo) -> Result<(), Box<dyn Error>>>,
    },
}

/// Represents a single model entry.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub hugging_face_id: Option<String>,
    pub name: Option<String>,
    pub created: Option<u64>,
    pub description: Option<String>,
    pub context_length: Option<u32>,
    pub architecture: Option<Architecture>,
    pub pricing: Option<Pricing>,
    pub top_provider: Option<TopProvider>,
    pub supported_parameters: Option<Vec<String>>,
    /// llama.cpp's server reports model details here instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<ModelMeta>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelMeta {
    /// The context size the server was started with (`--ctx-size`), which
    /// is what actually limits requests - not the model's training length.
    pub n_ctx: Option<u32>,
}

impl ModelInfo {
    /// The model's context window, from whichever field the endpoint
    /// reports it in.
    pub fn context_window(&self) -> Option<u32> {
        self.context_length
            .or_else(|| self.meta.as_ref().and_then(|m| m.n_ctx))
    }
}

/// Picks `model` out of an endpoint's model list. A server that lists a
/// single model (e.g. llama.cpp's) serves it whatever name the request
/// uses, so that one is taken when nothing matches by name.
pub fn find_model(models: Vec<ModelInfo>, model: &str) -> Option<ModelInfo> {
    if models.len() == 1 {
        return models.into_iter().next();
    }
    models.into_iter().find(|m| m.id == model)
}

/// Represents the architecture details of a model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Architecture {
    pub modality: String,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub tokenizer: String,
    pub instruct_type: Option<String>,
}

/// Represents the pricing details for a model.
/// All monetary values are stored as strings as they appear in the JSON.
/// These could be parsed into a decimal type if needed.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Pricing {
    pub prompt: String,
    pub completion: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal_reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cache_read: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cache_write: Option<String>,
}

/// Represents the top provider details for a model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TopProvider {
    pub context_length: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub is_moderated: bool,
}

#[derive(Deserialize, Debug)]
struct ModelsApiResponse {
    data: Vec<ModelInfo>,
}

/// Fetches the list of available models from the specified endpoint.
pub fn list_models_from_endpoint(
    endpoint: &str,
    api_key: Option<&String>,
) -> Result<Vec<ModelInfo>, Box<dyn Error>> {
    debug!("Fetching list of models from {}", endpoint);

    let client = ReqwestClient::builder().build()?;

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if let Some(key_file) = api_key {
        let api_key = read_api_key(key_file)?;
        let bearer_auth = format!("Bearer {}", &api_key);
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_auth)?);
    }

    let response = client.get(endpoint).headers(headers).send()?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .unwrap_or_else(|e| format!("Failed to read error body: {}", e));
        warn!(
            "Failed to fetch models. Status: {}. Body: {}",
            status, error_text
        );
        return Err(format!("Failed to fetch models: {} - {}", status, error_text).into());
    }

    let models_api_response: ModelsApiResponse = response.json()?;

    Ok(models_api_response.data)
}

/// Walks an error's cause chain and returns the deepest (most specific)
/// message available, falling back to the error's own message when it has
/// no cause chain. Some wrappers (notably pathrs, used for every safe file
/// tool) have an uninformative top-level message like "openat2 one-shot
/// open failed" while the actual reason ("No such file or directory") is a
/// couple of `source()` calls deeper; this surfaces that instead.
pub fn describe_error(e: &(dyn Error + 'static)) -> String {
    let mut deepest = e.to_string();
    let mut source = e.source();
    while let Some(s) = source {
        deepest = s.to_string();
        source = s.source();
    }
    deepest
}

/// Perform a tool call and return the message to send back.
/// Shortens a tool result to at most about `max_chars` characters, keeping
/// the beginning and the end (where a command's errors or a summary line
/// usually are) and saying in the middle what was left out and how to get
/// it. Returns `content` unchanged if it already fits.
pub fn truncate_tool_output(content: &str, max_chars: usize) -> String {
    let total = content.chars().count();
    if total <= max_chars {
        return content.to_string();
    }
    let head_chars = max_chars * 2 / 3;
    let tail_chars = max_chars - head_chars;
    let head_end = content
        .char_indices()
        .nth(head_chars)
        .map_or(content.len(), |(i, _)| i);
    let tail_start = content
        .char_indices()
        .nth(total - tail_chars)
        .map_or(content.len(), |(i, _)| i);
    format!(
        "{}\n\n[... {} characters omitted: this tool result was too large to show in full. \
         Ask for less (e.g. read_file with start_line/end_line, a more specific search \
         pattern, or a command with less output) to see the rest ...]\n\n{}",
        &content[..head_end],
        total - head_chars - tail_chars,
        &content[tail_start..]
    )
}

/// A tool result is never shrunk below this many characters by
/// `shrink_tool_results`, so it stays useful to the model.
const MIN_SHRUNK_TOOL_RESULT_CHARS: usize = 2000;

/// Reads the first run of digits right after `marker` in `text`.
fn number_after(text: &str, marker: &str) -> Option<u64> {
    let rest = &text[text.find(marker)? + marker.len()..];
    let rest = rest.trim_start_matches([' ', ':']);
    let digits: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == ',')
        .filter(|c| *c != ',')
        .collect();
    digits.parse().ok()
}

/// `(prompt_tokens, context_tokens)` from a context-overflow error, when
/// the provider reports them: llama.cpp's `n_prompt_tokens`/`n_ctx`
/// fields, OpenAI's "maximum context length is N tokens. However, your
/// messages resulted in M tokens", or Anthropic's "prompt is too long: M
/// tokens > N maximum".
pub fn context_overflow_sizes(error_text: &str) -> Option<(u64, u64)> {
    let pairs = [
        ("\"n_prompt_tokens\":", "\"n_ctx\":"),
        ("resulted in", "maximum context length is"),
        ("prompt is too long:", "tokens >"),
    ];
    pairs.iter().find_map(|(prompt_marker, context_marker)| {
        let prompt = number_after(error_text, prompt_marker)?;
        let context = number_after(error_text, context_marker)?;
        (prompt > context && context > 0).then_some((prompt, context))
    })
}

fn content_chars(msg: &Message) -> usize {
    msg.content.as_deref().map_or(0, |c| c.chars().count())
}

/// Cheaper recovery from a context overflow than summarizing: shortens the
/// largest tool results in `messages` (keeping the head and tail of each)
/// until the history should fit. Everything else, including what the model
/// did so far in the current turn, is kept, so the turn can simply carry on.
///
/// How much to cut comes from the sizes in `error_text` when the provider
/// reports them (aiming at 80% of the context, to leave room for the
/// reply), otherwise half. Returns `None` if shrinking tool results can't
/// get there, so the caller should summarize instead.
pub fn shrink_tool_results(messages: &[Message], error_text: &str) -> Option<Vec<Message>> {
    let ratio = match context_overflow_sizes(error_text) {
        Some((prompt, context)) => context as f64 * 0.8 / prompt as f64,
        None => 0.5,
    };
    shrink_tool_results_to(messages, ratio)
}

/// Shortens the largest tool results in `messages` until the history's
/// total size is about `ratio` of what it is now (see
/// `shrink_tool_results`). `None` if tool results alone can't get there.
pub fn shrink_tool_results_to(messages: &[Message], ratio: f64) -> Option<Vec<Message>> {
    let ratio = ratio.clamp(0.05, 0.9);
    let total: usize = messages.iter().map(content_chars).sum();
    let mut to_cut = total - (total as f64 * ratio) as usize;

    let mut tool_results: Vec<(usize, usize)> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == "tool")
        .map(|(i, m)| (i, content_chars(m)))
        .filter(|(_, len)| *len > MIN_SHRUNK_TOOL_RESULT_CHARS)
        .collect();
    let can_cut: usize = tool_results
        .iter()
        .map(|(_, len)| len - MIN_SHRUNK_TOOL_RESULT_CHARS)
        .sum();
    if can_cut < to_cut {
        return None;
    }

    tool_results.sort_by(|a, b| b.1.cmp(&a.1));
    let mut shrunk = messages.to_vec();
    for (i, len) in tool_results {
        if to_cut == 0 {
            break;
        }
        let keep = len.saturating_sub(to_cut).max(MIN_SHRUNK_TOOL_RESULT_CHARS);
        to_cut = to_cut.saturating_sub(len - keep);
        let content = shrunk[i].content.as_deref().unwrap_or("");
        shrunk[i].content = Some(truncate_tool_output(content, keep));
    }
    Some(shrunk)
}

/// Above this fraction of the context window, the tool loop shortens old
/// tool results before sending the next request.
const CONTEXT_TRIM_THRESHOLD: f64 = 0.75;

/// What the tool loop shortens the history to when it goes over
/// `CONTEXT_TRIM_THRESHOLD`, leaving room for several more tool results
/// and a long reply.
const CONTEXT_TRIM_TARGET: f64 = 0.5;

/// Proactive counterpart to `shrink_tool_results`, run in the tool loop
/// before each follow-up request: estimates the next request's size from
/// the last response's `usage` plus the `new_chars` of tool results added
/// since (or, if the server reported no usage, from the size of the whole
/// history), and if that's over `CONTEXT_TRIM_THRESHOLD` of `context_window`,
/// returns the history with old tool results shortened to about
/// `CONTEXT_TRIM_TARGET`. Without this the context can fill up with no
/// error at all - the server just cuts the reply short once the prompt
/// leaves no room for it.
fn trim_for_next_request(
    messages: &[Message],
    usage: Option<&Usage>,
    new_chars: usize,
    context_window: Option<u32>,
) -> Option<Vec<Message>> {
    let window = context_window? as f64;
    // ~4 characters per token for whatever the server hasn't counted.
    let estimated = match usage.and_then(|u| u.prompt_tokens) {
        Some(prompt) => {
            let completion = usage.and_then(|u| u.completion_tokens).unwrap_or(0);
            (prompt + completion) as f64 + new_chars as f64 / 4.0
        }
        None => messages.iter().map(content_chars).sum::<usize>() as f64 / 4.0,
    };
    if estimated <= window * CONTEXT_TRIM_THRESHOLD {
        return None;
    }
    shrink_tool_results_to(messages, window * CONTEXT_TRIM_TARGET / estimated)
}

pub fn tool_call(
    tools_collection: &ToolsCollection,
    req: &ToolCall,
    ctx: &crate::ToolContext,
) -> Result<Message, Box<dyn Error>> {
    let tool_name = &req.function.name;

    // Validate tool call has complete data
    if tool_name.is_empty() {
        return Err("Tool call missing name".into());
    }
    if req.function.arguments.is_empty() {
        return Err(format!("Tool call '{}' missing arguments", tool_name).into());
    }
    if req.id.is_empty() {
        return Err(format!("Tool call '{}' missing ID", tool_name).into());
    }

    info!("Requesting tool {}", tool_name);

    if let Some(ref mcp) = ctx.mcp {
        if mcp.has_tool(tool_name) {
            info!("Dispatching to MCP tool {}", tool_name);
            let content = match mcp.call_tool(tool_name, &req.function.arguments) {
                Ok(result) => result,
                Err(e) => {
                    let error_msg = format!(
                        "error: MCP tool '{}' failed: {}",
                        tool_name,
                        describe_error(e.as_ref())
                    );
                    warn!("{}", error_msg);
                    error_msg
                }
            };
            let content = truncate_tool_output(&content, ctx.max_tool_output_chars());
            let msg = Message {
                role: "tool".to_string(),
                content: Some(content),
                tool_call_id: Some(req.id.clone()),
                name: Some(tool_name.clone()),
                tool_calls: None,
            };
            return Ok(msg);
        }
    }

    let tool = tools_collection.get(tool_name);
    let content: String = match tool {
        None => {
            let error_msg = format!("error: invalid tool requested: '{}'", tool_name);
            warn!("{}", error_msg);
            error_msg
        }
        Some(t) => {
            info!("Executing tool {}", tool_name);
            debug!(
                "Passing arguments {:?} to tool {}",
                req.function.arguments, tool_name
            );
            // A tool that panics fails its call, not the whole turn - or,
            // on the chat's own thread, faber.
            let called = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (t.callback)(&req.function.arguments, ctx)
            }))
            .unwrap_or_else(|panic| {
                let what = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown".to_string());
                Err(format!("it crashed ({}) - this is a bug in faber", what).into())
            });
            match called {
                Ok(result) => result,
                // The user pressed Ctrl-C while the tool ran: stop the
                // whole turn, as anywhere else, rather than reporting it
                // to the model as a failed tool call.
                Err(e) if e.downcast_ref::<InterruptedError>().is_some() => return Err(e),
                Err(e) => {
                    let error_msg = format!(
                        "error: tool '{}' failed: {}",
                        tool_name,
                        describe_error(e.as_ref())
                    );
                    warn!("{}", error_msg);
                    error_msg
                }
            }
        }
    };

    trace!("Tool {} gave output {:?}", tool_name, content);
    let content = truncate_tool_output(&content, ctx.max_tool_output_chars());

    let msg = Message {
        role: "tool".to_string(),
        content: Some(content),
        tool_call_id: Some(req.id.clone()),
        name: Some(tool_name.clone()),
        tool_calls: None,
    };
    Ok(msg)
}

/// Create a Message from the specified role and content.
/// How a tool call touches shared state, used by `plan_tool_call_groups`
/// to decide which calls from one turn can safely run at the same time.
#[derive(Debug, Clone, PartialEq)]
enum ToolAccess {
    /// Touches nothing another call in the same turn could change (network
    /// fetches, GitHub API and DB reads).
    Independent,
    /// Reads one file.
    ReadPath(PathBuf),
    /// Reads across the whole working tree (`glob`, `grep`).
    ReadTree,
    /// Writes one file.
    WritePath(PathBuf),
    /// Anything else - arbitrary commands, deletions, DB/agent/task
    /// writes, sub-agents, MCP tools, unknown tools - always runs alone.
    Exclusive,
}

impl ToolAccess {
    fn conflicts_with(&self, other: &ToolAccess) -> bool {
        use ToolAccess::*;
        match (self, other) {
            (Exclusive, _) | (_, Exclusive) => true,
            (WritePath(a), WritePath(b) | ReadPath(b)) | (ReadPath(b), WritePath(a)) => a == b,
            (WritePath(_), ReadTree) | (ReadTree, WritePath(_)) => true,
            _ => false,
        }
    }
}

/// Resolves `path` against the current directory and folds away `.`/`..`
/// lexically, so different spellings of the same file compare equal.
/// Symlinks aren't resolved - the file may not exist yet.
fn normalize_tool_path(path: &str) -> PathBuf {
    let path = Path::new(path);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            c => normalized.push(c),
        }
    }
    normalized
}

/// Classifies a built-in tool call by name and arguments. A call whose
/// `path` can't be parsed is treated as `Exclusive` rather than guessed at.
fn tool_access(name: &str, arguments: &str) -> ToolAccess {
    let path = || {
        serde_json::from_str::<serde_json::Value>(arguments)
            .ok()
            .and_then(|v| v.get("path")?.as_str().map(normalize_tool_path))
    };
    match name {
        "read_file" => path().map_or(ToolAccess::Exclusive, ToolAccess::ReadPath),
        "write_file" | "patch_file" => path().map_or(ToolAccess::Exclusive, ToolAccess::WritePath),
        "glob" | "grep" => ToolAccess::ReadTree,
        "fetch_web_content"
        | "github_pull_request"
        | "github_issue"
        | "agent_list"
        | "agent_get"
        | "task_list"
        | "plan_get"
        | "kb_search"
        | "kb_read"
        | "kb_list" => ToolAccess::Independent,
        _ => ToolAccess::Exclusive,
    }
}

/// Splits a turn's tool calls (given by their `accesses`, in order) into
/// consecutive groups whose members don't conflict with each other. Groups
/// run one after another and the calls within a group run concurrently,
/// so every conflicting pair still runs in the order the model asked for.
fn plan_tool_call_groups(accesses: &[ToolAccess]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, access) in accesses.iter().enumerate() {
        match groups.last_mut() {
            Some(group) if !group.iter().any(|&j| accesses[j].conflicts_with(access)) => {
                group.push(i)
            }
            _ => groups.push(vec![i]),
        }
    }
    groups
}

fn report_progress(
    mode: &ResponseMode,
    status: StatusUpdate,
    start_time: Instant,
) -> Result<(), Box<dyn Error>> {
    if let ResponseMode::Streaming {
        progress_handler, ..
    } = mode
    {
        progress_handler(&ProgressInfo {
            status,
            elapsed_ms: start_time.elapsed().as_millis() as u64,
        })?;
    }
    Ok(())
}

/// Runs one tool call with its output going straight to `ctx.println`.
fn run_tool_call_live(
    tools_collection: &ToolsCollection,
    req: &ToolCall,
    ctx: &crate::ToolContext,
    mode: &ResponseMode,
    start_time: Instant,
) -> Result<Message, Box<dyn Error>> {
    report_progress(
        mode,
        StatusUpdate::ToolStart {
            name: req.function.name.clone(),
            arguments: req.function.arguments.clone(),
        },
        start_time,
    )?;
    debug!(
        "Executing tool call '{}' with complete arguments (length: {})",
        req.function.name,
        req.function.arguments.len()
    );
    let tool_start_time = Instant::now();
    // Mark the tool's own output as "boxed" for the duration of the call
    // only, so a println closure that wants to frame it can tell it apart
    // from unrelated output - reset unconditionally (even on error) so the
    // flag never gets stuck on for later, unrelated println calls.
    ctx.boxed.store(true, Ordering::Relaxed);
    let result = tool_call(tools_collection, req, ctx);
    ctx.boxed.store(false, Ordering::Relaxed);
    let msg = result?;
    report_progress(
        mode,
        StatusUpdate::ToolComplete {
            name: req.function.name.clone(),
            duration_ms: tool_start_time.elapsed().as_millis() as u64,
            output: msg.content.clone().unwrap_or_default(),
        },
        start_time,
    )?;
    Ok(msg)
}

/// Runs `calls` concurrently, one thread each. Each tool's output is
/// buffered and replayed through `ctx.println` afterwards, in call order,
/// framed by its own `ToolStart`/`ToolComplete`, so concurrent output never
/// interleaves.
fn run_tool_call_group(
    tools_collection: &ToolsCollection,
    calls: &[&ToolCall],
    ctx: &crate::ToolContext,
    mode: &ResponseMode,
    start_time: Instant,
) -> Result<Vec<Message>, Box<dyn Error>> {
    report_progress(
        mode,
        StatusUpdate::ToolBatchStart {
            names: calls.iter().map(|c| c.function.name.clone()).collect(),
        },
        start_time,
    )?;
    let outcomes: Vec<(Result<Message, String>, Vec<String>, Duration)> = thread::scope(|scope| {
        let handles: Vec<_> = calls
            .iter()
            .map(|&req| {
                scope.spawn(move || {
                    let output = Arc::new(Mutex::new(Vec::new()));
                    let sink = output.clone();
                    let mut buffered = crate::ToolContext::new(move |msg: &str| {
                        sink.lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(msg.to_string())
                    });
                    buffered.db = ctx.db.clone();
                    buffered.agent_name = ctx.agent_name.clone();
                    buffered.extra = ctx.extra.clone();
                    buffered.mcp = ctx.mcp.clone();
                    buffered.max_tool_output_chars = ctx.max_tool_output_chars;
                    buffered.context_window = ctx.context_window;
                    buffered.interrupt = ctx.interrupt.clone();
                    buffered.file_versions = ctx.file_versions.clone();
                    buffered.unsafe_tools = ctx.unsafe_tools;
                    buffered.cwd = ctx.cwd.clone();
                    buffered.task_id = ctx.task_id;
                    let tool_start_time = Instant::now();
                    let result =
                        tool_call(tools_collection, req, &buffered).map_err(|e| e.to_string());
                    let lines =
                        std::mem::take(&mut *output.lock().unwrap_or_else(|e| e.into_inner()));
                    (result, lines, tool_start_time.elapsed())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
            .collect()
    });

    let mut messages = Vec::with_capacity(calls.len());
    for (req, (result, lines, duration)) in calls.iter().zip(outcomes) {
        report_progress(
            mode,
            StatusUpdate::ToolStart {
                name: req.function.name.clone(),
                arguments: req.function.arguments.clone(),
            },
            start_time,
        )?;
        ctx.boxed.store(true, Ordering::Relaxed);
        for line in &lines {
            ctx.println(line);
        }
        ctx.boxed.store(false, Ordering::Relaxed);
        let msg = result.map_err(|e| -> Box<dyn Error> { e.into() })?;
        report_progress(
            mode,
            StatusUpdate::ToolComplete {
                name: req.function.name.clone(),
                duration_ms: duration.as_millis() as u64,
                output: msg.content.clone().unwrap_or_default(),
            },
            start_time,
        )?;
        messages.push(msg);
    }
    Ok(messages)
}

/// Runs a turn's tool calls, returning their result messages in call
/// order. Calls that can't interfere with each other (see `ToolAccess`)
/// run concurrently; everything else runs one at a time, in order.
pub fn run_tool_calls(
    tools_collection: &ToolsCollection,
    tool_calls: &[ToolCall],
    ctx: &crate::ToolContext,
    mode: &ResponseMode,
    start_time: Instant,
) -> Result<Vec<Message>, Box<dyn Error>> {
    let accesses: Vec<ToolAccess> = tool_calls
        .iter()
        .map(|tc| {
            // tool_call dispatches to MCP first, even for a name that
            // shadows a built-in, and MCP tools are opaque.
            let is_mcp = ctx
                .mcp
                .as_ref()
                .is_some_and(|m| m.has_tool(&tc.function.name));
            if is_mcp {
                ToolAccess::Exclusive
            } else {
                tool_access(&tc.function.name, &tc.function.arguments)
            }
        })
        .collect();

    let mut messages = Vec::with_capacity(tool_calls.len());
    for group in plan_tool_call_groups(&accesses) {
        if let [i] = group[..] {
            messages.push(run_tool_call_live(
                tools_collection,
                &tool_calls[i],
                ctx,
                mode,
                start_time,
            )?);
        } else {
            let calls: Vec<&ToolCall> = group.iter().map(|&i| &tool_calls[i]).collect();
            debug!("Running {} tool calls concurrently", calls.len());
            messages.extend(run_tool_call_group(
                tools_collection,
                &calls,
                ctx,
                mode,
                start_time,
            )?);
        }
    }
    Ok(messages)
}

pub fn make_message(role: &str, content: String) -> Message {
    Message {
        role: role.to_string(),
        content: Some(content),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

/// The tools as a request lists them: always in the same order, by name.
/// The tool definitions are near the start of the prompt the server makes,
/// so an order that changed between requests - a `HashMap`'s does, from one
/// collection to the next - would leave nothing of a cached prompt
/// reusable, for the model to process the whole prompt again every time.
fn request_tools(
    tools_collection: &ToolsCollection,
    ctx: &crate::ToolContext,
) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
    let mut names: Vec<&String> = tools_collection.keys().collect();
    names.sort();
    let mut tools = Vec::with_capacity(names.len());
    for name in names {
        tools.push(serde_json::from_str(&tools_collection[name].schema)?);
    }
    if let Some(ref mcp) = ctx.mcp {
        let mut mcp_tools = mcp.get_tool_schemas();
        mcp_tools.sort_by(|a, b| {
            let name =
                |v: &serde_json::Value| v["function"]["name"].as_str().unwrap_or("").to_string();
            name(a).cmp(&name(b))
        });
        tools.extend(mcp_tools);
    }
    Ok(tools)
}

/// Sends a POST request to the OpenAI API with the given messages and options.
pub fn post_request(
    messages: Vec<Message>,
    tools_collection: &ToolsCollection,
    opts: &Opts,
    ctx: &crate::ToolContext,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    post_request_with_mode(
        messages,
        tools_collection,
        opts,
        ResponseMode::Complete,
        ctx,
        None,
    )
}

pub fn post_request_with_mode(
    messages: Vec<Message>,
    tools_collection: &ToolsCollection,
    opts: &Opts,
    mode: ResponseMode,
    ctx: &crate::ToolContext,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    if crate::scripted_llm::is_scripted_model(&opts.model) {
        return crate::scripted_llm::post_request_scripted(
            messages,
            &opts.model,
            tools_collection,
            mode,
            ctx,
            ctrl_c_rx,
        );
    }
    if crate::dummy_llm::is_dummy_model(&opts.model) {
        return crate::dummy_llm::post_request_dummy(
            messages,
            tools_collection,
            mode,
            ctx,
            ctrl_c_rx,
        );
    }
    post_request_with_mode_and_recursion(messages, tools_collection, opts, mode, ctx, ctrl_c_rx)
}

/// Internal function with iterative tool call handling.
fn post_request_with_mode_and_recursion(
    messages: Vec<Message>,
    tools_collection: &ToolsCollection,
    opts: &Opts,
    mode: ResponseMode,
    ctx: &crate::ToolContext,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    let start_time = Instant::now();
    let mut messages = messages;
    let mut turn_usage: Option<Usage> = None;
    let mut requests = 0;

    loop {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(ref key_file) = opts.api_key {
            let api_key = read_api_key(key_file)?;
            let bearer_auth = format!("Bearer {}", &api_key);
            headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer_auth)?);
        }

        let tools = request_tools(tools_collection, ctx)?;

        let tool_choice = if tools.len() > 0 {
            // If tools are available, use user's choice or default to "auto"
            opts.tool_choice.clone()
        } else {
            None
        };

        // Use streaming based on the mode
        let use_streaming = match mode {
            ResponseMode::Streaming { .. } => true,
            ResponseMode::Complete => false,
        };

        let request_body = OpenAIRequest {
            model: opts.model.clone(),
            max_tokens: opts.max_tokens,
            messages: messages.clone(),
            tools: if tools.len() > 0 { Some(tools) } else { None },
            tool_choice: tool_choice,
            stream: if use_streaming { Some(true) } else { None },
            stream_options: use_streaming.then(|| serde_json::json!({"include_usage": true})),
            parameters: opts.parameters.clone(),
        };

        let request_json = serde_json::to_string(&request_body)?;
        trace!("Send request: {}", request_json);
        debug!("Request has {} custom parameters", opts.parameters.len());

        if let ResponseMode::Streaming {
            progress_handler, ..
        } = &mode
        {
            let progress_info = ProgressInfo {
                status: StatusUpdate::SendingRequest {
                    bytes: request_json.len(),
                },
                elapsed_ms: start_time.elapsed().as_millis() as u64,
            };
            progress_handler(&progress_info)?;
        }

        let client = model_client()?;

        requests += 1;
        if let Some(limit) = ctx.max_requests {
            if requests > limit {
                return Err(request_budget_error(limit));
            }
        }
        // Held until this round trip's response has been read in full.
        let _slot = request_limiter().acquire(&ctrl_c_rx, &mode, start_time)?;

        let max_retries = opts.max_retries.unwrap_or(5);
        let base_delay_secs = opts.retry_base_delay_secs.unwrap_or(1);
        let mut response = None;

        for attempt in 1..=max_retries {
            let request = client
                .post(&opts.endpoint)
                .headers(headers.clone())
                .json(&request_body);
            let response_result = interruptible(move || request.send(), &ctrl_c_rx)?;

            check_ctrl_c_signal(&ctrl_c_rx)?;

            match response_result {
                Ok(resp) => {
                    let status = resp.status();

                    if status.is_success() {
                        response = Some(resp);
                        break;
                    }

                    if (status.is_server_error() || status.as_u16() == 429) && attempt < max_retries
                    {
                        let delay_duration =
                            Duration::from_secs(base_delay_secs * 2_u64.pow(attempt as u32 - 1));
                        warn!(
                            "Server error {} (attempt {}/{}). Retrying after {} seconds for endpoint: {}",
                            status,
                            attempt,
                            max_retries,
                            delay_duration.as_secs(),
                            opts.endpoint
                        );
                        // Check for interruption during sleep in smaller intervals
                        let sleep_interval = Duration::from_millis(100);
                        let mut remaining = delay_duration;
                        while remaining > Duration::ZERO {
                            check_ctrl_c_signal(&ctrl_c_rx)?;
                            let sleep_time = std::cmp::min(sleep_interval, remaining);
                            thread::sleep(sleep_time);
                            remaining = remaining.saturating_sub(sleep_time);
                        }
                        continue;
                    }

                    // Non-retryable error or max retries reached
                    let error_text = resp
                        .text()
                        .unwrap_or_else(|_| "Unable to read response".to_string());
                    let err: Box<dyn Error> =
                        format!("got error code: {}: {}", status, error_text).into();
                    return Err(if matches!(status.as_u16(), 400 | 413 | 422) {
                        classify_context_error(err, &messages)
                    } else {
                        err
                    });
                }
                Err(e) => {
                    if attempt < max_retries {
                        let delay_duration =
                            Duration::from_secs(base_delay_secs * 2_u64.pow(attempt as u32 - 1));
                        warn!(
                            "Request failed (attempt {}/{}): {}. Retrying after {} seconds for endpoint: {}",
                            attempt,
                            max_retries,
                            e,
                            delay_duration.as_secs(),
                            opts.endpoint
                        );
                        // Check for interruption during sleep in smaller intervals
                        let sleep_interval = Duration::from_millis(100);
                        let mut remaining = delay_duration;
                        while remaining > Duration::ZERO {
                            check_ctrl_c_signal(&ctrl_c_rx)?;
                            let sleep_time = std::cmp::min(sleep_interval, remaining);
                            thread::sleep(sleep_time);
                            remaining = remaining.saturating_sub(sleep_time);
                        }
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }

        let response = response.ok_or("Max retries reached without successful response")?;

        let mut openai_response: OpenAIResponse = if use_streaming {
            handle_streaming_response(response, &mode, ctrl_c_rx.clone())
                .map_err(|e| classify_context_error(e, &messages))?
        } else {
            let response_text = interruptible(move || response.text(), &ctrl_c_rx)??;
            trace!("Got response {:?}", response_text);
            serde_json::from_str(&response_text)?
        };
        drop(_slot);
        accumulate_usage(&mut turn_usage, openai_response.usage.as_ref());

        if let Some(mut err) = openai_response.error {
            if err.metadata.is_some() {
                if let Some(raw) = err.metadata.unwrap().raw {
                    let raw_response: OpenAIResponse =
                        serde_json::from_str::<OpenAIResponse>(&raw)?;
                    if let Some(inner_err) = raw_response.error {
                        err = inner_err;
                    }
                }
            }
            let err: Box<dyn Error> = format!(
                "got API error code: {}: {}",
                err.code.unwrap_or_else(|| 400),
                err.message
            )
            .into();
            return Err(classify_context_error(err, &messages));
        }

        let mut finish: bool = true;

        if let Some(choices) = &openai_response.choices {
            trace!("Got {} choices", choices.len());
            if let Some(choice) = choices.first() {
                trace!("Got choice {:?}", choice);
                let finish_reason = choice
                    .finish_reason
                    .clone()
                    .unwrap_or_else(|| "".to_string());
                let has_tools = !tools_collection.is_empty()
                    || ctx.mcp.as_ref().map_or(false, |m| m.has_tools());
                finish = finish_reason != "" && (finish_reason != "tool_calls" || !has_tools);
                if finish_reason == "error" {
                    let native_finish_reason = choice
                        .native_finish_reason
                        .clone()
                        .unwrap_or_else(|| finish_reason);
                    return Err(format!("got API error: {}", native_finish_reason).into());
                }

                // Add the complete assistant message (with both content and tool_calls if present)
                let assistant_msg = choice.message.clone();
                if assistant_msg.content.is_some() || assistant_msg.tool_calls.is_some() {
                    debug!(
                        "Adding assistant message to conversation history - content: {:?}, tool_calls: {}",
                        assistant_msg.content.as_ref().map(|c| c.len()),
                        assistant_msg
                            .tool_calls
                            .as_ref()
                            .map(|tc| tc.len())
                            .unwrap_or(0)
                    );
                    messages.push(assistant_msg);
                }

                if choice.message.tool_calls.is_some() {
                    let tool_calls = choice
                        .message
                        .tool_calls
                        .as_ref()
                        .ok_or("Invalid response")?;

                    let results =
                        run_tool_calls(tools_collection, tool_calls, ctx, &mode, start_time)?;
                    let new_chars: usize = results.iter().map(content_chars).sum();
                    messages.extend(results);
                    if let Some(trimmed) = trim_for_next_request(
                        &messages,
                        openai_response.usage.as_ref(),
                        new_chars,
                        ctx.context_window,
                    ) {
                        ctx.println(
                            "(Shortened older tool results to stay within the context window.)",
                        );
                        messages = trimmed;
                    }
                }
            }
        }

        if !finish {
            debug!(
                "Continuing conversation: finish={}, current_messages={}",
                finish,
                messages.len()
            );

            // The next iteration reports SendingRequest once it has built
            // the next request body, so there's nothing to show here.
            continue;
        }

        // If we reach here, the conversation is finished
        // Show final completion status with usage information
        if let ResponseMode::Streaming {
            progress_handler, ..
        } = &mode
        {
            let progress_info = ProgressInfo {
                status: StatusUpdate::Complete {
                    usage: openai_response.usage.clone(),
                },
                elapsed_ms: start_time.elapsed().as_millis() as u64,
            };
            progress_handler(&progress_info)?;
        }

        debug!("Final response: messages in history = {}", messages.len());
        openai_response.history = messages;
        openai_response.turn_usage = turn_usage;
        return Ok(openai_response);
    }
}

/// Handle streaming response from the API
fn handle_streaming_response(
    response: reqwest::blocking::Response,
    mode: &ResponseMode,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    let (stream_handler, reasoning_handler, progress_handler) = match mode {
        ResponseMode::Streaming {
            stream_handler,
            reasoning_handler,
            progress_handler,
        } => (stream_handler, reasoning_handler, progress_handler),
        ResponseMode::Complete => return Err("Invalid mode for streaming response".into()),
    };

    let reader = BufReader::new(response);
    let mut accumulated_content = String::new();
    let mut accumulated_tool_calls: HashMap<usize, ToolCall> = HashMap::new();
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<Usage> = None;
    let mut bytes_read = 0usize;
    let mut chunks_processed = 0u32;
    let mut tool_accumulation_start: Option<std::time::Instant> = None;
    let mut thinking_reported = false;
    let mut reasoning_active = false;
    let start_time = std::time::Instant::now();

    // Create a channel for line reading
    let (line_tx, line_rx) = mpsc::channel::<Result<String, std::io::Error>>();

    let _reader_thread = std::thread::spawn(move || {
        for line in reader.lines() {
            match line {
                Ok(line_content) => {
                    if line_tx.send(Ok(line_content)).is_err() {
                        break; // Receiver dropped, exit thread
                    }
                }
                Err(e) => {
                    let _ = line_tx.send(Err(e));
                    break;
                }
            }
        }
    });

    loop {
        // Check for Ctrl-C signal first
        check_ctrl_c_signal(&ctrl_c_rx)?;

        match line_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Ok(line)) => {
                bytes_read += line.len();

                let data = if line.starts_with("data: ") {
                    &line[6..]
                } else if line.starts_with("data:") {
                    &line[5..]
                } else {
                    continue;
                };
                chunks_processed += 1;

                if data == "" {
                    continue;
                }

                if data == "[DONE]" {
                    break;
                }

                let streaming_response: StreamingResponse = match serde_json::from_str(data) {
                    Ok(response) => response,
                    Err(e) => {
                        debug!("Skipping invalid JSON chunk: {}, data: '{}'", e, data);
                        if data.len() > 10 {
                            warn!(
                                "Large chunk failed to parse, potential data loss: {} chars",
                                data.len()
                            );
                        }
                        continue;
                    }
                };

                if let Some(error) = streaming_response.error {
                    return Err(format!("Streaming API error: {}", error.message).into());
                }

                if let Some(choices) = streaming_response.choices {
                    if let Some(choice) = choices.first() {
                        // Report thinking status on first meaningful chunk
                        if !thinking_reported {
                            let progress_info = ProgressInfo {
                                status: StatusUpdate::Thinking,
                                elapsed_ms: start_time.elapsed().as_millis() as u64,
                            };
                            let _ = progress_handler(&progress_info);
                            thinking_reported = true;
                        }

                        // Report progress on a cadence regardless of what kind of
                        // payload this chunk carries: reasoning-heavy models can
                        // spend most (or all) of a turn generating reasoning
                        // tokens, and without this a long reasoning phase would
                        // otherwise leave the status frozen on "Thinking" with no
                        // sign that anything is happening.
                        if chunks_processed % 10 == 0 {
                            let progress_info = ProgressInfo {
                                status: StatusUpdate::StreamProcessing {
                                    bytes_read,
                                    chunks_processed,
                                },
                                elapsed_ms: 0,
                            };
                            let _ = progress_handler(&progress_info);
                        }

                        // Reasoning tokens arrive in a separate field, whose name depends on the server
                        for reasoning in [&choice.delta.thinking, &choice.delta.reasoning_content]
                            .into_iter()
                            .flatten()
                        {
                            if !reasoning.is_empty() {
                                reasoning_handler(reasoning)?;
                                reasoning_active = true;
                            }
                        }

                        // Handle regular content
                        if let Some(content) = &choice.delta.content {
                            if content != "" {
                                if reasoning_active {
                                    reasoning_handler("")?;
                                    reasoning_active = false;
                                }
                                accumulated_content.push_str(content);
                                stream_handler(content)?;
                            }
                        }

                        if let Some(tool_calls) = &choice.delta.tool_calls {
                            // Reasoning or content may have been streaming
                            // just before the model switched to emitting a
                            // tool call in the same response, with no
                            // newline to naturally close its partial line.
                            // Close it now (a no-op if there's nothing to
                            // close) so the status bar's spinner isn't left
                            // suppressed - thinking there's still an open
                            // partial line to protect - for however long
                            // tool-call argument accumulation takes.
                            if reasoning_active {
                                reasoning_handler("")?;
                                reasoning_active = false;
                            }
                            stream_handler("")?;

                            // Set accumulation start time if this is the first tool call chunk
                            if tool_accumulation_start.is_none() {
                                tool_accumulation_start = Some(std::time::Instant::now());
                            }

                            for streaming_tool_call in tool_calls {
                                let index =
                                    streaming_tool_call.index.unwrap_or(usize::MAX as u64) as usize;

                                let updated_tool_call = match accumulated_tool_calls.get_mut(&index)
                                {
                                    Some(existing_call) => {
                                        // Accumulate the arguments
                                        if let Some(args) = &streaming_tool_call.function.arguments
                                        {
                                            existing_call.function.arguments.push_str(args);
                                        }
                                        // Update other fields if they have values
                                        if let Some(name) = &streaming_tool_call.function.name {
                                            if !name.is_empty() {
                                                existing_call.function.name = name.clone();
                                            }
                                        }
                                        if let Some(id) = &streaming_tool_call.id {
                                            if !id.is_empty() {
                                                existing_call.id = id.clone();
                                            }
                                        }
                                        if let Some(tool_type) = &streaming_tool_call.tool_type {
                                            if !tool_type.is_empty() {
                                                existing_call.tool_type = tool_type.clone();
                                            }
                                        }
                                        existing_call.clone()
                                    }
                                    None => {
                                        // First chunk for this tool call - convert to regular ToolCall
                                        debug!(
                                            "Creating new tool call at index {}, streaming_tool_call.id: {:?}",
                                            index, streaming_tool_call.id
                                        );
                                        // Use empty ID initially, will be filled when available or at the end
                                        let tool_call = ToolCall {
                                            index: streaming_tool_call.index,
                                            id: streaming_tool_call.id.clone().unwrap_or_default(),
                                            tool_type: streaming_tool_call
                                                .tool_type
                                                .clone()
                                                .unwrap_or("function".to_string()),
                                            function: FunctionCall {
                                                name: streaming_tool_call
                                                    .function
                                                    .name
                                                    .clone()
                                                    .unwrap_or_default(),
                                                arguments: streaming_tool_call
                                                    .function
                                                    .arguments
                                                    .clone()
                                                    .unwrap_or_default(),
                                            },
                                        };
                                        debug!("Created tool call: {:?}", tool_call);
                                        accumulated_tool_calls.insert(index, tool_call.clone());
                                        tool_call
                                    }
                                };

                                // Show accumulating status if we have a progress handler
                                if let ResponseMode::Streaming {
                                    progress_handler, ..
                                } = mode
                                {
                                    let elapsed_ms = tool_accumulation_start
                                        .map(|start| start.elapsed().as_millis() as u64)
                                        .unwrap_or(0);

                                    let progress_info = ProgressInfo {
                                        status: StatusUpdate::ToolAccumulating {
                                            name: updated_tool_call.function.name.clone(),
                                            arguments: updated_tool_call.function.arguments.clone(),
                                        },
                                        elapsed_ms,
                                    };
                                    let _ = progress_handler(&progress_info);
                                }
                            }
                        }

                        if choice.finish_reason.is_some() {
                            finish_reason = choice.finish_reason.clone();
                        }
                    }
                }

                // Capture usage information if available in this chunk
                if let Some(chunk_usage) = streaming_response.usage {
                    usage = Some(chunk_usage);
                }
            }
            Ok(Err(e)) => {
                // Line reading error
                return Err(Box::new(e));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Timeout - continue checking for Ctrl-C signals
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Reader thread finished - break out of loop
                break;
            }
        }
    }

    // Ensure we call the handlers one final time to flush output
    reasoning_handler("")?;
    stream_handler("")?;

    // Build the complete response
    let tool_calls_vec: Option<Vec<ToolCall>> = if accumulated_tool_calls.is_empty() {
        None
    } else {
        let calls: Vec<ToolCall> = accumulated_tool_calls.into_values().collect();

        debug!("Accumulated calls {:?}", calls);

        // Use index as ID when actual ID is not available
        let calls_with_ids: Vec<ToolCall> = calls
            .into_iter()
            .map(|mut call| {
                if call.id.is_empty() {
                    call.id = call.index.unwrap_or(0).to_string();
                    debug!("Using index '{}' as ID for tool call", call.id);
                }
                call
            })
            .collect();

        // Validate tool calls are complete before including them
        let valid_calls: Vec<ToolCall> = calls_with_ids
            .into_iter()
            .filter(|call| {
                if call.function.name.is_empty() {
                    warn!("Dropping tool call with empty name");
                    false
                } else if call.function.arguments.is_empty() {
                    warn!(
                        "Dropping tool call '{}' with empty arguments",
                        call.function.name
                    );
                    false
                } else if call.id.is_empty() {
                    warn!("Dropping tool call '{}' with empty ID", call.function.name);
                    false
                } else {
                    true
                }
            })
            .collect();

        if valid_calls.is_empty() {
            warn!("All tool calls were invalid and dropped, clearing finish_reason");
            // If all tool calls were invalid, clear finish_reason to continue conversation
            // instead of causing "Invalid response" error or premature exit
            finish_reason = None;
            None
        } else {
            // Check for duplicate IDs and make them unique
            let mut seen_ids = std::collections::HashSet::new();
            let mut unique_calls: Vec<ToolCall> = valid_calls
                .into_iter()
                .map(|mut call| {
                    if seen_ids.contains(&call.id) {
                        // Make ID unique by appending the index
                        let new_id = format!("{}_{}", call.id, call.index.unwrap_or(0));
                        warn!(
                            "Duplicate tool call ID '{}' found, changing to '{}'",
                            call.id, new_id
                        );
                        call.id = new_id;
                    }
                    seen_ids.insert(call.id.clone());
                    call
                })
                .collect();

            unique_calls.sort_by_key(|call| call.index.unwrap_or(usize::MAX as u64));
            Some(unique_calls)
        }
    };

    let message = Message {
        role: "assistant".to_string(),
        content: if accumulated_content.is_empty() {
            None
        } else {
            Some(accumulated_content)
        },
        tool_calls: tool_calls_vec,
        tool_call_id: None,
        name: None,
    };

    let choice = Choice {
        message,
        finish_reason,
        native_finish_reason: None,
    };

    // Use captured usage information from streaming response if available

    Ok(OpenAIResponse {
        error: None,
        choices: Some(vec![choice]),
        usage,
        turn_usage: None,
        history: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_limiter_limits_concurrency_and_stays_interruptible() {
        // Its own limiter: the process-wide one is shared with every other
        // test making requests.
        let limiter: &'static RequestLimiter = Box::leak(Box::new(RequestLimiter::new(1)));
        let acquire = |ctrl_c: &Option<Arc<Mutex<mpsc::Receiver<()>>>>| {
            limiter.acquire(ctrl_c, &ResponseMode::Complete, Instant::now())
        };
        let first = acquire(&None).unwrap();
        // A second request waits for the first...
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let slot = limiter
                .acquire(&None, &ResponseMode::Complete, Instant::now())
                .unwrap();
            done_tx.send(()).unwrap();
            drop(slot);
        });
        assert!(
            done_rx.recv_timeout(Duration::from_millis(300)).is_err(),
            "should wait"
        );
        drop(first);
        done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("got the slot once freed");
        waiter.join().unwrap();

        // ...and Ctrl-C stops the wait.
        let held = acquire(&None).unwrap();
        let (tx, rx) = mpsc::channel();
        tx.send(()).unwrap();
        let err = acquire(&Some(Arc::new(Mutex::new(rx))))
            .err()
            .expect("interrupted");
        assert!(err.downcast_ref::<InterruptedError>().is_some());
        drop(held);

        // No limit: never waits.
        limiter.set_limit(0);
        let _a = acquire(&None).unwrap();
        let _b = acquire(&None).unwrap();
    }

    #[test]
    fn test_interruptible_returns_the_result_or_stops_on_ctrl_c() {
        let (tx, rx) = mpsc::channel();
        let ctrl_c = Some(Arc::new(Mutex::new(rx)));
        assert_eq!(interruptible(|| 42, &ctrl_c).unwrap(), 42);
        assert_eq!(interruptible(|| 7, &None).unwrap(), 7);

        let started = Instant::now();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(200));
            let _ = tx.send(());
        });
        let err = interruptible(|| thread::sleep(Duration::from_secs(30)), &ctrl_c).unwrap_err();
        assert!(err.downcast_ref::<InterruptedError>().is_some(), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn test_truncate_tool_output_keeps_head_and_tail() {
        let content: String = (0..1000).map(|i| format!("{:04}\n", i)).collect();
        let out = truncate_tool_output(&content, 600);
        assert!(out.starts_with("0000\n"));
        assert!(out.ends_with("0999\n"));
        assert!(out.contains("characters omitted"));
        assert!(out.chars().count() < 600 + 300);
        assert_eq!(truncate_tool_output("short", 600), "short");
    }

    #[test]
    fn test_truncate_tool_output_respects_char_boundaries() {
        let out = truncate_tool_output(&"é".repeat(100), 10);
        assert!(out.starts_with("éééééé"));
    }

    fn model(id: &str, json_extra: serde_json::Value) -> ModelInfo {
        let mut v = serde_json::json!({"id": id});
        v.as_object_mut()
            .unwrap()
            .extend(json_extra.as_object().unwrap().clone());
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn test_context_window_reads_llama_cpp_meta() {
        let m = model(
            "m",
            serde_json::json!({"meta": {"n_ctx": 66816, "n_vocab": 5}}),
        );
        assert_eq!(m.context_window(), Some(66816));
        let m = model("m", serde_json::json!({"context_length": 8192}));
        assert_eq!(m.context_window(), Some(8192));
    }

    #[test]
    fn test_find_model_takes_the_only_model_whatever_its_name() {
        let only = vec![model("/models/qwen.gguf", serde_json::json!({}))];
        assert_eq!(
            find_model(only, "google/gemini-2.5-pro").unwrap().id,
            "/models/qwen.gguf"
        );
        let many = vec![
            model("a", serde_json::json!({})),
            model("b", serde_json::json!({})),
        ];
        assert_eq!(find_model(many.clone(), "b").unwrap().id, "b");
        assert!(find_model(many, "c").is_none());
    }

    fn tool_result(len: usize) -> Message {
        let mut msg = make_message("tool", "z".repeat(len));
        msg.name = Some("read_file".to_string());
        msg.tool_call_id = Some("call_1".to_string());
        msg
    }

    #[test]
    fn test_overflow_sizes_are_read_from_known_error_formats() {
        let llama = r#"got error code: 400 Bad Request: {"error":{"code":400,"message":"request (96822 tokens) exceeds the available context size (66816 tokens)","type":"exceed_context_size_error","n_prompt_tokens":96822,"n_ctx":66816}}"#;
        assert_eq!(context_overflow_sizes(llama), Some((96822, 66816)));
        let openai = "This model's maximum context length is 128000 tokens. However, your messages resulted in 130,512 tokens.";
        assert_eq!(context_overflow_sizes(openai), Some((130512, 128000)));
        let anthropic = "prompt is too long: 210000 tokens > 200000 maximum";
        assert_eq!(context_overflow_sizes(anthropic), Some((210000, 200000)));
        assert_eq!(context_overflow_sizes("context length exceeded"), None);
    }

    #[test]
    fn test_shrink_cuts_the_largest_tool_result_and_keeps_the_turn() {
        let messages = vec![
            make_message("user", "read it".to_string()),
            Message {
                role: "assistant".to_string(),
                content: None,
                tool_call_id: None,
                name: None,
                tool_calls: Some(vec![make_call("call_1", "read_file", "{}")]),
            },
            tool_result(100_000),
        ];
        // Needs to get down to 80% of 50/100 of the current size.
        let error = r#"{"n_prompt_tokens":100000,"n_ctx":50000}"#;
        let shrunk = shrink_tool_results(&messages, error).unwrap();
        assert_eq!(shrunk.len(), 3);
        assert!(shrunk[1].tool_calls.is_some());
        assert_eq!(shrunk[2].tool_call_id.as_deref(), Some("call_1"));
        let total: usize = shrunk.iter().map(content_chars).sum();
        let before: usize = messages.iter().map(content_chars).sum();
        assert!(total <= before * 41 / 100, "{total} of {before}");
    }

    #[test]
    fn test_shrink_gives_up_when_tool_results_are_not_the_problem() {
        let messages = vec![
            make_message("user", "x".repeat(100_000)),
            tool_result(3_000),
        ];
        assert!(shrink_tool_results(&messages, "context length exceeded").is_none());
    }

    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            prompt_tokens: Some(prompt),
            completion_tokens: Some(completion),
            total_tokens: Some(prompt + completion),
        }
    }

    #[test]
    fn test_trim_for_next_request_only_near_the_limit() {
        let messages = vec![make_message("user", "go".to_string()), tool_result(200_000)];
        // 20k of 100k tokens used: nothing to do.
        assert!(
            trim_for_next_request(&messages, Some(&usage(20_000, 100)), 0, Some(100_000)).is_none()
        );
        // Unknown window: nothing to do.
        assert!(trim_for_next_request(&messages, Some(&usage(90_000, 100)), 0, None).is_none());
        // No usage reported: estimated from the history (~50k tokens here).
        assert!(trim_for_next_request(&messages, None, 0, Some(100_000)).is_none());
        assert!(trim_for_next_request(&messages, None, 0, Some(60_000)).is_some());
        // 60k used + 200k new chars (~50k tokens) is over 75%.
        let trimmed =
            trim_for_next_request(&messages, Some(&usage(60_000, 0)), 200_000, Some(100_000))
                .unwrap();
        assert!(content_chars(&trimmed[1]) < 200_000 / 2);
    }

    fn read(p: &str) -> ToolAccess {
        ToolAccess::ReadPath(PathBuf::from(p))
    }

    fn write(p: &str) -> ToolAccess {
        ToolAccess::WritePath(PathBuf::from(p))
    }

    #[test]
    fn test_plan_tool_call_groups_independent_reads_share_a_group() {
        let accesses = [
            read("/a"),
            read("/b"),
            ToolAccess::ReadTree,
            ToolAccess::Independent,
        ];
        assert_eq!(plan_tool_call_groups(&accesses), vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn test_plan_tool_call_groups_same_path_write_splits_in_order() {
        let accesses = [read("/a"), write("/a"), read("/a")];
        assert_eq!(
            plan_tool_call_groups(&accesses),
            vec![vec![0], vec![1], vec![2]]
        );
    }

    #[test]
    fn test_plan_tool_call_groups_writes_to_different_paths_share_a_group() {
        let accesses = [write("/a"), write("/b"), read("/c")];
        assert_eq!(plan_tool_call_groups(&accesses), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn test_plan_tool_call_groups_write_conflicts_with_tree_read() {
        let accesses = [ToolAccess::ReadTree, write("/a")];
        assert_eq!(plan_tool_call_groups(&accesses), vec![vec![0], vec![1]]);
    }

    #[test]
    fn test_plan_tool_call_groups_exclusive_runs_alone() {
        let accesses = [
            read("/a"),
            ToolAccess::Exclusive,
            ToolAccess::Independent,
            ToolAccess::Independent,
        ];
        assert_eq!(
            plan_tool_call_groups(&accesses),
            vec![vec![0], vec![1], vec![2, 3]]
        );
    }

    #[test]
    fn test_tool_access_normalizes_path_spellings() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            tool_access("read_file", r#"{"path": "./src/../x.txt"}"#),
            ToolAccess::ReadPath(cwd.join("x.txt"))
        );
        assert_eq!(
            tool_access("patch_file", r#"{"path": "x.txt", "edits": []}"#),
            ToolAccess::WritePath(cwd.join("x.txt"))
        );
    }

    #[test]
    fn test_tool_access_unknown_or_unparseable_is_exclusive() {
        assert_eq!(tool_access("run_command", "{}"), ToolAccess::Exclusive);
        assert_eq!(tool_access("some_new_tool", "{}"), ToolAccess::Exclusive);
        assert_eq!(tool_access("read_file", "not json"), ToolAccess::Exclusive);
    }

    fn sleepy_tool(args: &String, ctx: &crate::ToolContext) -> Result<String, Box<dyn Error>> {
        ctx.println(&format!("start {}", args));
        thread::sleep(Duration::from_millis(300));
        ctx.println(&format!("end {}", args));
        Ok(format!("done {}", args))
    }

    fn make_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            index: None,
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    #[test]
    fn test_request_tools_are_always_in_the_same_order() {
        let tool = |name: &str| ToolItem {
            callback: |_, _| Ok(String::new()),
            schema: serde_json::json!({"type": "function", "function": {"name": name}}).to_string(),
        };
        let names = ["write_file", "glob", "read_file", "agent_wait", "fan_out"];
        let ctx = crate::ToolContext::new(|_: &str| {});
        let mut seen = std::collections::HashSet::new();
        for _ in 0..5 {
            let mut tools = ToolsCollection::new();
            for name in names {
                tools.insert(name.to_string(), tool(name));
            }
            seen.insert(serde_json::to_string(&request_tools(&tools, &ctx).unwrap()).unwrap());
        }
        assert_eq!(seen.len(), 1, "{seen:?}");
        let first: Vec<serde_json::Value> =
            serde_json::from_str(seen.iter().next().unwrap()).unwrap();
        let listed: Vec<&str> = first
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            listed,
            ["agent_wait", "fan_out", "glob", "read_file", "write_file"]
        );
    }

    #[test]
    fn test_run_tool_calls_runs_independent_calls_concurrently_in_order() {
        let mut tools = ToolsCollection::new();
        // Registered under read-only built-in names so they classify as
        // parallel-safe.
        for name in ["github_issue", "fetch_web_content", "task_list"] {
            tools.insert(
                name.to_string(),
                ToolItem {
                    callback: sleepy_tool,
                    schema: String::new(),
                },
            );
        }
        let output = Arc::new(Mutex::new(Vec::new()));
        let sink = output.clone();
        let ctx = crate::ToolContext::new(move |msg: &str| {
            sink.lock().unwrap().push(msg.to_string());
        });
        let calls = [
            make_call("1", "github_issue", "a"),
            make_call("2", "fetch_web_content", "b"),
            make_call("3", "task_list", "c"),
        ];

        let started = Instant::now();
        let messages =
            run_tool_calls(&tools, &calls, &ctx, &ResponseMode::Complete, started).unwrap();
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_millis(800),
            "three 300ms tools took {:?}, so they didn't run concurrently",
            elapsed
        );
        let ids: Vec<_> = messages
            .iter()
            .map(|m| m.tool_call_id.clone().unwrap())
            .collect();
        assert_eq!(ids, vec!["1", "2", "3"]);
        assert_eq!(messages[1].content.as_deref(), Some("done b"));
        // Each tool's output is replayed as one uninterrupted block.
        assert_eq!(
            *output.lock().unwrap(),
            vec!["start a", "end a", "start b", "end b", "start c", "end c"]
        );
    }

    #[test]
    fn test_run_tool_calls_runs_exclusive_calls_one_at_a_time() {
        let mut tools = ToolsCollection::new();
        tools.insert(
            "run_command".to_string(),
            ToolItem {
                callback: sleepy_tool,
                schema: String::new(),
            },
        );
        let ctx = crate::ToolContext::new(|_: &str| {});
        let calls = [
            make_call("1", "run_command", "a"),
            make_call("2", "run_command", "b"),
        ];
        let started = Instant::now();
        run_tool_calls(&tools, &calls, &ctx, &ResponseMode::Complete, started).unwrap();
        assert!(started.elapsed() >= Duration::from_millis(600));
    }

    #[test]
    fn test_accumulate_usage_sums_across_round_trips() {
        let mut total = None;
        let round = |p, c, t| Usage {
            prompt_tokens: Some(p),
            completion_tokens: Some(c),
            total_tokens: Some(t),
        };
        accumulate_usage(&mut total, Some(&round(100, 10, 110)));
        accumulate_usage(&mut total, Some(&round(150, 20, 170)));
        let total = total.unwrap();
        assert_eq!(total.prompt_tokens, Some(250));
        assert_eq!(total.completion_tokens, Some(30));
        assert_eq!(total.total_tokens, Some(280));
    }

    #[test]
    fn test_accumulate_usage_none_stays_none_until_something_reports() {
        let mut total = None;
        accumulate_usage(&mut total, None);
        assert!(total.is_none());
        accumulate_usage(
            &mut total,
            Some(&Usage {
                prompt_tokens: Some(5),
                completion_tokens: None,
                total_tokens: None,
            }),
        );
        accumulate_usage(&mut total, None);
        let total = total.unwrap();
        assert_eq!(total.prompt_tokens, Some(5));
        assert_eq!(total.completion_tokens, None);
    }

    #[test]
    fn test_normalize_endpoint_already_complete() {
        assert_eq!(
            normalize_endpoint("http://localhost:8080/chat/completions"),
            "http://localhost:8080/chat/completions"
        );
    }

    #[test]
    fn test_normalize_endpoint_trailing_slash() {
        assert_eq!(
            normalize_endpoint("http://localhost:8080/"),
            "http://localhost:8080/chat/completions"
        );
    }

    #[test]
    fn test_normalize_endpoint_no_trailing_slash() {
        assert_eq!(
            normalize_endpoint("http://localhost:8080"),
            "http://localhost:8080/chat/completions"
        );
    }

    #[test]
    fn test_normalize_endpoint_with_v1() {
        assert_eq!(
            normalize_endpoint("http://localhost:8080/v1"),
            "http://localhost:8080/v1/chat/completions"
        );
    }

    #[test]
    fn test_make_message() {
        let msg = make_message("user", "hello".to_string());
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, Some("hello".to_string()));
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());
        assert!(msg.name.is_none());
    }

    #[test]
    fn test_make_message_system() {
        let msg = make_message("system", "you are helpful".to_string());
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, Some("you are helpful".to_string()));
    }

    #[test]
    fn test_tool_call_missing_name() {
        let tools = ToolsCollection::new();
        let req = ToolCall {
            index: None,
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let ctx = crate::ToolContext::new(|_: &str| {});
        let result = tool_call(&tools, &req, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_missing_arguments() {
        let tools = ToolsCollection::new();
        let req = ToolCall {
            index: None,
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: "".to_string(),
            },
        };
        let ctx = crate::ToolContext::new(|_: &str| {});
        let result = tool_call(&tools, &req, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_missing_id() {
        let tools = ToolsCollection::new();
        let req = ToolCall {
            index: None,
            id: "".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let ctx = crate::ToolContext::new(|_: &str| {});
        let result = tool_call(&tools, &req, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_call_unknown_tool() {
        let tools = ToolsCollection::new();
        let req = ToolCall {
            index: None,
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "nonexistent_tool".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let ctx = crate::ToolContext::new(|_: &str| {});
        let result = tool_call(&tools, &req, &ctx).unwrap();
        assert!(result.content.unwrap().contains("invalid tool"));
    }

    #[test]
    fn test_is_context_length_error() {
        assert!(is_context_length_error(
            "got error code: 400 Bad Request: {\"error\":{\"code\":\"context_length_exceeded\"}}"
        ));
        assert!(is_context_length_error(
            "This model's maximum context length is 8192 tokens. However, you requested 9000"
        ));
        assert!(is_context_length_error(
            "prompt is too long: 250000 tokens > 200000 maximum"
        ));
        assert!(is_context_length_error(
            "Input exceeds the CONTEXT WINDOW of the model"
        ));
        // llama.cpp server
        assert!(is_context_length_error(
            r#"got error code: 400 Bad Request: {"error":{"code":400,"message":"request (4115 tokens) exceeds the available context size (4096 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":4115,"n_ctx":4096}}"#
        ));
        assert!(!is_context_length_error(
            "Rate limit reached: too many tokens per minute"
        ));
        assert!(!is_context_length_error(
            "got error code: 401: invalid api key"
        ));
    }

    #[test]
    fn test_classify_context_error_keeps_history() {
        let history = vec![make_message("user", "hi".to_string())];
        let err = classify_context_error("prompt is too long".into(), &history);
        let overflow = err.downcast_ref::<ContextLengthError>().unwrap();
        assert_eq!(overflow.history.len(), 1);

        let err = classify_context_error("connection reset".into(), &history);
        assert!(err.downcast_ref::<ContextLengthError>().is_none());
    }

    /// Serves one canned server-sent-events response on a local port and
    /// returns the endpoint to use.
    fn serve_sse(events: Vec<serde_json::Value>) -> String {
        serve_sse_turns(vec![events])
    }

    /// Like `serve_sse`, but serves one set of events per request, in order
    /// (each round trip through a tool call is a separate HTTP request).
    fn serve_sse_turns(turns: Vec<Vec<serde_json::Value>>) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for events in turns {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = Vec::new();
                let mut buf = [0u8; 4096];
                let (header_end, content_length) = loop {
                    let n = stream.read(&mut buf).unwrap();
                    request.extend_from_slice(&buf[..n]);
                    if let Some(pos) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&request[..pos]).to_lowercase();
                        let length = headers
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        break (pos + 4, length);
                    }
                };
                while request.len() < header_end + content_length {
                    let n = stream.read(&mut buf).unwrap();
                    request.extend_from_slice(&buf[..n]);
                }
                let mut response = String::from(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                );
                for event in events {
                    response.push_str(&format!("data: {}\n\n", event));
                }
                response.push_str("data: [DONE]\n\n");
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        format!("http://{}/chat/completions", addr)
    }

    fn delta(field: &str, text: &str, finish: Option<&str>) -> serde_json::Value {
        serde_json::json!({"choices": [{"delta": {field: text}, "finish_reason": finish}]})
    }

    fn stream_to_vecs(events: Vec<serde_json::Value>) -> (OpenAIResponse, String, String) {
        let (response, answer, reasoning, _statuses, _events) =
            stream_to_vecs_with(serve_sse(events), &ToolsCollection::new());
        (response, answer, reasoning)
    }

    /// One thing observed while streaming, in the order it actually
    /// happened - used to check *ordering* between reasoning's flush and
    /// the tool-call statuses, which a plain count can't do (a reasoning
    /// flush also happens harmlessly once at the end of every turn, tool
    /// call or not, so counting alone can't tell a flush that happened
    /// too late, or not at the right place, from one that didn't).
    #[derive(Debug, Clone)]
    enum TestEvent {
        ReasoningFlush,
        Status(StatusUpdate),
    }

    /// Like `stream_to_vecs`, but against a caller-provided endpoint (so a
    /// multi-turn `serve_sse_turns` can be used) and tools collection.
    /// Returns every `StatusUpdate` reported via the progress handler, in
    /// order, and (interleaved with those) every reasoning-flush event, in
    /// the single combined order everything actually happened in.
    fn stream_to_vecs_with(
        endpoint: String,
        tools: &ToolsCollection,
    ) -> (
        OpenAIResponse,
        String,
        String,
        Vec<StatusUpdate>,
        Vec<TestEvent>,
    ) {
        use std::cell::RefCell;
        use std::rc::Rc;

        let answer = Rc::new(RefCell::new(String::new()));
        let reasoning = Rc::new(RefCell::new(String::new()));
        let statuses = Rc::new(RefCell::new(Vec::new()));
        let events = Rc::new(RefCell::new(Vec::new()));
        let (answer_out, reasoning_out, statuses_out, events_out) = (
            answer.clone(),
            reasoning.clone(),
            statuses.clone(),
            events.clone(),
        );
        let events_out2 = events.clone();
        let mode = ResponseMode::Streaming {
            stream_handler: Box::new(move |chunk| {
                answer_out.borrow_mut().push_str(chunk);
                Ok(())
            }),
            reasoning_handler: Box::new(move |chunk| {
                if chunk.is_empty() {
                    events_out.borrow_mut().push(TestEvent::ReasoningFlush);
                }
                reasoning_out.borrow_mut().push_str(chunk);
                Ok(())
            }),
            progress_handler: Box::new(move |info| {
                statuses_out.borrow_mut().push(info.status.clone());
                events_out2
                    .borrow_mut()
                    .push(TestEvent::Status(info.status.clone()));
                Ok(())
            }),
        };
        let opts = Opts {
            max_tokens: None,
            model: "test-model".to_string(),
            endpoint,
            tool_choice: None,
            api_key: None,
            max_retries: Some(1),
            retry_base_delay_secs: None,
            parameters: HashMap::new(),
        };
        let ctx = crate::ToolContext::new(|_: &str| {});
        let response = post_request_with_mode(
            vec![make_message("user", "hi".to_string())],
            tools,
            &opts,
            mode,
            &ctx,
            None,
        )
        .unwrap();
        let (answer, reasoning, statuses, events) = (
            answer.borrow().clone(),
            reasoning.borrow().clone(),
            statuses.borrow().clone(),
            events.borrow().clone(),
        );
        (response, answer, reasoning, statuses, events)
    }

    #[test]
    fn test_streaming_reasoning_is_separate_and_not_in_history() {
        for field in ["reasoning_content", "thinking"] {
            let (response, answer, reasoning) = stream_to_vecs(vec![
                delta(field, "The user", None),
                delta(field, " said hi", None),
                delta("content", "Hello", None),
                delta("content", "!", Some("stop")),
            ]);
            assert_eq!(reasoning, "The user said hi");
            assert_eq!(answer, "Hello!");
            let last = response.history.last().unwrap();
            assert_eq!(last.role, "assistant");
            assert_eq!(last.content.as_deref(), Some("Hello!"));
        }
    }

    #[test]
    fn test_streaming_truncated_response_reports_length() {
        let (response, answer, reasoning) = stream_to_vecs(vec![
            delta("reasoning_content", "The user said hello. This is", None),
            delta("reasoning_content", "", Some("length")),
        ]);
        assert_eq!(answer, "");
        assert_eq!(reasoning, "The user said hello. This is");
        let choice = &response.choices.as_ref().unwrap()[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("length"));
        // Reasoning alone must not end up in the history as an answer.
        assert_eq!(response.history.len(), 1);
    }

    #[test]
    fn test_streaming_reports_progress_during_pure_reasoning() {
        // A long reasoning-only stream (no answer content at all) used to
        // leave the status frozen on "Thinking" for its whole duration,
        // since progress was only ever reported from the content branch.
        let mut events: Vec<_> = (0..10)
            .map(|_| delta("reasoning_content", "word ", None))
            .collect();
        *events.last_mut().unwrap() = delta("reasoning_content", "word ", Some("stop"));
        let (_response, _answer, reasoning, statuses, _events) =
            stream_to_vecs_with(serve_sse(events), &ToolsCollection::new());
        assert_eq!(reasoning, "word ".repeat(10));
        assert!(
            statuses
                .iter()
                .any(|s| matches!(s, StatusUpdate::StreamProcessing { .. })),
            "expected at least one StreamProcessing update during reasoning, got {:?}",
            statuses
        );
    }

    #[test]
    fn test_streaming_progress_does_not_wait_for_content() {
        // Fewer than the reporting cadence: no StreamProcessing update is
        // expected yet, but SendingRequest (before the request goes out)
        // and the initial "Thinking" (on the first byte back) must still
        // fire immediately, in that order.
        let (_response, _answer, _reasoning, statuses, _events) = stream_to_vecs_with(
            serve_sse(vec![delta("reasoning_content", "hi", Some("stop"))]),
            &ToolsCollection::new(),
        );
        assert!(matches!(
            statuses.first(),
            Some(StatusUpdate::SendingRequest { .. })
        ));
        assert!(matches!(statuses.get(1), Some(StatusUpdate::Thinking)));
    }

    fn ok_tool(_args: &String, _ctx: &crate::ToolContext) -> Result<String, Box<dyn Error>> {
        Ok("tool result".to_string())
    }

    #[test]
    fn test_streaming_tool_call_reports_a_single_start_and_complete() {
        let mut tools = ToolsCollection::new();
        tools.insert(
            "lookup".to_string(),
            ToolItem {
                callback: ok_tool,
                schema: r#"{"type":"function","function":{"name":"lookup","parameters":{}}}"#
                    .to_string(),
            },
        );

        let call = serde_json::json!({"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}
        }]}, "finish_reason": "tool_calls"}]});
        let endpoint = serve_sse_turns(vec![
            vec![call],
            vec![delta("content", "done", Some("stop"))],
        ]);

        let (response, answer, _reasoning, statuses, _events) =
            stream_to_vecs_with(endpoint, &tools);
        assert_eq!(answer, "done");
        assert_eq!(
            response.choices.unwrap()[0].finish_reason.as_deref(),
            Some("stop")
        );

        // Exactly one "starting" update and one "complete" update per tool
        // call: no separate, redundant "executing" update in between (the
        // two used to be reported back to back with identical text and no
        // way to tell them apart).
        let starts = statuses
            .iter()
            .filter(|s| matches!(s, StatusUpdate::ToolStart { name, .. } if name == "lookup"))
            .count();
        let completes = statuses
            .iter()
            .filter(|s| matches!(s, StatusUpdate::ToolComplete { name, .. } if name == "lookup"))
            .count();
        assert_eq!(starts, 1);
        assert_eq!(completes, 1);

        // One SendingRequest per turn (the tool call's turn, and the
        // follow-up that reports the result), each with a plausible,
        // nonzero size for the request actually sent at that point.
        let sending: Vec<usize> = statuses
            .iter()
            .filter_map(|s| match s {
                StatusUpdate::SendingRequest { bytes } => Some(*bytes),
                _ => None,
            })
            .collect();
        assert_eq!(sending.len(), 2);
        assert!(sending.iter().all(|&b| b > 0));
        // The second request includes the tool result, so it's larger.
        assert!(sending[1] > sending[0]);
    }

    #[test]
    fn test_reasoning_directly_into_a_tool_call_flushes_reasoning_first() {
        // The bug this guards against: the model reasons about which tool
        // to use, then calls it directly with no answer content in
        // between (very common - there's no reason for a model to say
        // anything before invoking a tool). Reasoning's partial line must
        // be closed as soon as tool-call streaming starts, not left open
        // for the whole tool-accumulation phase (which is exactly when
        // the status bar has the most to show - "Preparing tool(...)").
        let mut tools = ToolsCollection::new();
        tools.insert(
            "lookup".to_string(),
            ToolItem {
                callback: ok_tool,
                schema: r#"{"type":"function","function":{"name":"lookup","parameters":{}}}"#
                    .to_string(),
            },
        );

        let reasoning_chunk = delta("reasoning_content", "I should look this up", None);
        let call = serde_json::json!({"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}
        }]}, "finish_reason": "tool_calls"}]});
        let endpoint = serve_sse_turns(vec![
            vec![reasoning_chunk, call],
            vec![delta("content", "done", Some("stop"))],
        ]);

        let (_response, _answer, reasoning, _statuses, events) =
            stream_to_vecs_with(endpoint, &tools);
        assert_eq!(reasoning, "I should look this up");

        // A reasoning flush also happens harmlessly once at the end of
        // every turn regardless of tool calls (existing behavior, not
        // this fix), so a plain count can't distinguish "flushed too
        // late" from "flushed at the right time" - only ordering can:
        // the *first* reasoning flush must come no later than the *first*
        // ToolStart, proving reasoning's partial line was closed before
        // (or, at worst, in the same processing pass as) tool-call
        // accumulation begins, not left open for however long that takes.
        let first_flush = events
            .iter()
            .position(|e| matches!(e, TestEvent::ReasoningFlush))
            .expect("reasoning's partial line was never closed - left dangling");
        // Check against ToolAccumulating specifically, not ToolStart:
        // ToolStart/ToolComplete fire from the *outer* function only after
        // the whole SSE stream (and its own trailing end-of-stream flush)
        // has already finished, so they'd always come after some flush
        // regardless of this fix. ToolAccumulating fires from within the
        // same delta-processing loop as the fix, while the stream is
        // still being read - exactly the window the bug left the spinner
        // suppressed for.
        let first_accumulating = events
            .iter()
            .position(|e| matches!(e, TestEvent::Status(StatusUpdate::ToolAccumulating { .. })))
            .expect("expected a ToolAccumulating status");
        assert!(
            first_flush <= first_accumulating,
            "reasoning flushed at position {} but ToolAccumulating was already reported at \
             position {} - the status bar's spinner would have been suppressed until then",
            first_flush,
            first_accumulating
        );
    }

    #[test]
    fn test_sending_request_reports_actual_request_size() {
        // A bigger prompt must be reflected in a bigger reported size, not
        // a placeholder - this is what lets the status line explain a long
        // wait instead of looking like a hang.
        let short = "hi".to_string();
        let long = "x".repeat(50_000);

        let bytes_for = |user_message: &str| -> usize {
            use std::cell::RefCell;
            use std::rc::Rc;
            let sizes = Rc::new(RefCell::new(Vec::new()));
            let sizes_out = sizes.clone();
            let mode = ResponseMode::Streaming {
                stream_handler: Box::new(|_| Ok(())),
                reasoning_handler: Box::new(|_| Ok(())),
                progress_handler: Box::new(move |info| {
                    if let StatusUpdate::SendingRequest { bytes } = info.status {
                        sizes_out.borrow_mut().push(bytes);
                    }
                    Ok(())
                }),
            };
            let opts = Opts {
                max_tokens: None,
                model: "test-model".to_string(),
                endpoint: serve_sse(vec![delta("content", "ok", Some("stop"))]),
                tool_choice: None,
                api_key: None,
                max_retries: Some(1),
                retry_base_delay_secs: None,
                parameters: HashMap::new(),
            };
            let ctx = crate::ToolContext::new(|_: &str| {});
            post_request_with_mode(
                vec![make_message("user", user_message.to_string())],
                &ToolsCollection::new(),
                &opts,
                mode,
                &ctx,
                None,
            )
            .unwrap();
            sizes.borrow()[0]
        };

        assert!(bytes_for(&long) > bytes_for(&short) + 40_000);
    }

    #[derive(Debug)]
    struct WrapperError {
        message: &'static str,
        source: Option<Box<dyn Error>>,
    }

    impl std::fmt::Display for WrapperError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.message)
        }
    }

    impl Error for WrapperError {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            self.source.as_deref()
        }
    }

    #[test]
    fn test_describe_error_no_source_returns_own_message() {
        let err = WrapperError {
            message: "top level failure",
            source: None,
        };
        assert_eq!(describe_error(&err), "top level failure");
    }

    #[test]
    fn test_describe_error_walks_to_deepest_cause() {
        // Mirrors the real case this was written for: pathrs wraps the
        // actual OS error behind an uninformative "openat2 ... failed"
        // message, several `source()` calls deep.
        let os_error = WrapperError {
            message: "No such file or directory (os error 2)",
            source: None,
        };
        let syscall = WrapperError {
            message: "openat2(...)",
            source: Some(Box::new(os_error)),
        };
        let top = WrapperError {
            message: "openat2 one-shot open failed",
            source: Some(Box::new(syscall)),
        };
        assert_eq!(
            describe_error(&top),
            "No such file or directory (os error 2)"
        );
    }

    #[test]
    fn test_interrupted_error_display() {
        let err = InterruptedError::new("user cancelled");
        assert_eq!(format!("{}", err), "user cancelled");
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message {
            role: "assistant".to_string(),
            content: Some("hello".to_string()),
            tool_call_id: None,
            tool_calls: None,
            name: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, "assistant");
        assert_eq!(parsed.content, Some("hello".to_string()));
    }

    #[test]
    fn test_message_skips_none_fields() {
        let msg = Message {
            role: "user".to_string(),
            content: Some("hi".to_string()),
            tool_call_id: None,
            tool_calls: None,
            name: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("tool_call_id"));
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("name"));
    }
}
