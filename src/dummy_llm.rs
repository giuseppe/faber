use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use crate::openai::{
    Choice, ContextLengthError, FunctionCall, InterruptedError, Message, OpenAIResponse,
    ProgressInfo, RequestLimiter, ResponseMode, StatusUpdate, ToolCall, Usage, accumulate_usage,
    request_limiter, run_tool_calls,
};

static TURN_COUNTER: AtomicUsize = AtomicUsize::new(0);

const MAX_LINES: usize = 10_000;

/// Requests carrying more text than this are rejected the way a real API
/// rejects a prompt that exceeds the context window.
const CONTEXT_LIMIT_CHARS: usize = 15_000;

struct DummyToolCall {
    name: &'static str,
    arguments: &'static str,
}

const TOOL_CYCLE: &[DummyToolCall] = &[
    DummyToolCall {
        name: "read_file",
        arguments: r#"{"path": "src/main.rs"}"#,
    },
    DummyToolCall {
        name: "write_file",
        arguments: r#"{"path": "dummy_test.tmp", "content": "test content line 1\ntest content line 2\n"}"#,
    },
    DummyToolCall {
        name: "write_file",
        arguments: r#"{"path": "dummy_test.tmp", "content": "replaced", "old_content": "test content line 1"}"#,
    },
    DummyToolCall {
        name: "run_command",
        arguments: r#"{"command": "echo hello from dummy"}"#,
    },
    DummyToolCall {
        name: "glob",
        arguments: r#"{"pattern": "*.rs"}"#,
    },
];

/// The read-only tools a `work` step calls, together, when available.
const WORK_TOOLS: &[DummyToolCall] = &[
    DummyToolCall {
        name: "glob",
        arguments: r#"{"pattern": "src/*.rs"}"#,
    },
    DummyToolCall {
        name: "read_file",
        arguments: r#"{"path": "Cargo.toml"}"#,
    },
];

enum DummyCommand {
    Long(usize),
    Slow(usize),
    /// `work [STEPS [DELAY_MS]]`: like an agent doing a job - see
    /// `work_response`.
    Work {
        steps: usize,
        delay_ms: u64,
    },
    /// `call TOOL ARGUMENTS`: calls TOOL with ARGUMENTS, JSON on the same
    /// line, then answers with what it returned.
    Call {
        name: String,
        arguments: String,
    },
    Normal,
}

impl DummyCommand {
    fn word_delay_ms(&self) -> u64 {
        match self {
            DummyCommand::Slow(_) => 200,
            DummyCommand::Work { .. } | DummyCommand::Call { .. } => 0,
            _ => 15,
        }
    }

    fn forces_text(&self) -> bool {
        !matches!(self, DummyCommand::Normal)
    }
}

fn check_interrupted(
    ctrl_c_rx: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<(), InterruptedError> {
    if let Some(rx) = ctrl_c_rx {
        if let Ok(receiver) = rx.lock() {
            if receiver.try_recv().is_ok() {
                return Err(InterruptedError {
                    message: "Interrupted by user".to_string(),
                });
            }
        }
    }
    Ok(())
}

fn last_user_message(messages: &[Message]) -> &str {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_deref())
        .unwrap_or("")
}

fn conversation_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| {
            m.content.as_ref().map_or(0, |c| c.len())
                + m.tool_calls
                    .iter()
                    .flatten()
                    .map(|tc| tc.function.arguments.len())
                    .sum::<usize>()
        })
        .sum()
}

fn make_long_text(num_lines: usize) -> String {
    let n = num_lines.min(MAX_LINES);
    let mut text = String::new();
    for i in 1..=n {
        use std::fmt::Write;
        let _ = writeln!(
            text,
            "Line {:>3}: The quick brown fox jumps over the lazy dog. \
             Pack my box with five dozen liquor jugs. \
             How vexingly quick daft zebras jump.",
            i
        );
    }
    text
}

fn parse_dummy_command(msg: &str) -> DummyCommand {
    let trimmed = msg.trim();
    // Only its first line: whoever delegates the job may add instructions
    // below it (e.g. to call report_result).
    let mut words = trimmed.lines().next().unwrap_or("").split_whitespace();
    let first = trimmed.lines().next().unwrap_or("");
    if let Some((name, arguments)) = first
        .strip_prefix("call ")
        .and_then(|rest| rest.trim().split_once(' '))
    {
        return DummyCommand::Call {
            name: name.to_string(),
            arguments: arguments.trim().to_string(),
        };
    }
    if words.next() == Some("work") {
        let steps = words.next().and_then(|w| w.parse().ok());
        let delay_ms = steps.and(words.next().and_then(|w| w.parse().ok()));
        return DummyCommand::Work {
            steps: steps.unwrap_or(2),
            delay_ms: delay_ms.unwrap_or(0),
        };
    }
    for (prefix, ctor, default) in [
        ("long", DummyCommand::Long as fn(usize) -> DummyCommand, 50),
        ("slow", DummyCommand::Slow as fn(usize) -> DummyCommand, 20),
    ] {
        if trimmed == prefix {
            return ctor(default);
        }
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            if let Some(rest) = rest.strip_prefix(' ') {
                if let Ok(n) = rest.trim().parse::<usize>() {
                    return ctor(n);
                }
            }
        }
    }
    DummyCommand::Normal
}

fn make_normal_text(turn: usize, message_count: usize) -> String {
    use std::fmt::Write;

    let mut text = String::new();
    let _ = writeln!(
        text,
        "Dummy response #{} (conversation has {} messages).\n",
        turn, message_count
    );

    let sections = [
        (
            "Analysis",
            &[
                "The current implementation follows a standard request-response pattern.",
                "Each component is loosely coupled through well-defined interfaces.",
                "Error handling propagates through the Result type consistently.",
                "The module structure separates concerns between I/O and business logic.",
                "Thread safety is maintained via Arc and Mutex where shared state is needed.",
            ][..],
        ),
        (
            "Observations",
            &[
                "Memory usage remains stable across long-running sessions.",
                "The streaming path handles backpressure through bounded channels.",
                "Configuration is loaded once at startup and shared immutably.",
                "Tool execution is sandboxed to the working directory.",
                "Signal handling cooperates with the main event loop via channels.",
            ][..],
        ),
        (
            "Recommendations",
            &[
                "Consider adding retry logic for transient network failures.",
                "The status bar refresh rate could be adaptive based on terminal speed.",
                "Batch database writes when processing multiple tool results.",
                "Add structured logging with span context for debugging agent chains.",
                "Profile the JSON serialization path if message history grows large.",
            ][..],
        ),
        (
            "Next steps",
            &[
                "Run the test suite to verify no regressions were introduced.",
                "Review the diff for any unintended changes to public interfaces.",
                "Update documentation if the configuration format changed.",
                "Check that the CI pipeline passes with the new dependencies.",
                "Tag a release candidate once all reviewers have approved.",
            ][..],
        ),
    ];

    let section_idx = turn % sections.len();

    for i in 0..sections.len() {
        let (title, points) = sections[(section_idx + i) % sections.len()];
        let _ = writeln!(text, "## {}\n", title);
        for point in points {
            let _ = writeln!(text, "- {}", point);
        }
        text.push('\n');
    }

    let _ = write!(
        text,
        "This was turn {} of the conversation. {} messages have been exchanged so far.",
        turn, message_count
    );
    text
}

fn make_text_response(turn: usize, messages: Vec<Message>) -> OpenAIResponse {
    let user_msg = last_user_message(&messages);
    let text = match parse_dummy_command(user_msg) {
        DummyCommand::Long(n) | DummyCommand::Slow(n) => make_long_text(n),
        DummyCommand::Work { .. } | DummyCommand::Call { .. } | DummyCommand::Normal => {
            make_normal_text(turn, messages.len())
        }
    };
    OpenAIResponse {
        error: None,
        choices: Some(vec![Choice {
            message: Message {
                role: "assistant".to_string(),
                content: Some(text),
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
            finish_reason: Some("stop".to_string()),
            native_finish_reason: None,
        }]),
        usage: Some(Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            total_tokens: Some(30),
        }),
        turn_usage: None,
        history: messages,
    }
}

fn make_tool_call_response(turn: usize, messages: Vec<Message>) -> OpenAIResponse {
    let idx = (turn / 2) % TOOL_CYCLE.len();
    let dummy = &TOOL_CYCLE[idx];
    let call_id = format!("call_dummy_{}", turn);

    OpenAIResponse {
        error: None,
        choices: Some(vec![Choice {
            message: Message {
                role: "assistant".to_string(),
                content: None,
                tool_call_id: None,
                name: None,
                tool_calls: Some(vec![ToolCall {
                    index: Some(0),
                    id: call_id,
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: dummy.name.to_string(),
                        arguments: dummy.arguments.to_string(),
                    },
                }]),
            },
            finish_reason: Some("tool_calls".to_string()),
            native_finish_reason: None,
        }]),
        usage: Some(Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
        }),
        turn_usage: None,
        history: messages,
    }
}

/// The latest user message that's a `work` command, with its position:
/// it governs the rest of the conversation, through any nudge that follows
/// (e.g. to report the result).
fn work_command(messages: &[Message]) -> Option<(usize, usize, u64)> {
    messages.iter().enumerate().rev().find_map(|(i, m)| {
        if m.role != "user" {
            return None;
        }
        match parse_dummy_command(m.content.as_deref()?) {
            DummyCommand::Work { steps, delay_ms } => Some((i, steps, delay_ms)),
            _ => None,
        }
    })
}

fn assistant_message(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> Message {
    Message {
        role: "assistant".to_string(),
        content,
        tool_call_id: None,
        name: None,
        tool_calls,
    }
}

/// The response to a `work` job given at `start` in `messages`, like an
/// agent's: `steps` rounds of read-only tool calls, made together; then,
/// when it was asked to and can, a call to report_result (with the job as
/// `data.job`); then a short answer naming the job. Each depends only on the conversation so far,
/// so agents working at once can't change each other's course.
fn work_response(
    start: usize,
    steps: usize,
    messages: Vec<Message>,
    tools_collection: &crate::openai::ToolsCollection,
) -> OpenAIResponse {
    let job = messages[start]
        .content
        .as_deref()
        .and_then(|c| c.lines().next())
        .unwrap_or("")
        .to_string();
    let since = &messages[start..];
    let calls = || {
        since
            .iter()
            .filter(|m| m.role == "assistant")
            .flat_map(|m| m.tool_calls.iter().flatten())
    };
    let rounds = since
        .iter()
        .filter(|m| {
            m.tool_calls
                .iter()
                .flatten()
                .any(|tc| tc.function.name != "report_result")
        })
        .count();
    let reported = calls().any(|tc| tc.function.name == "report_result");
    let asked_to_report = since
        .iter()
        .any(|m| m.role == "user" && m.content.as_deref().unwrap_or("").contains("report_result"));
    let call = |i: usize, name: &str, arguments: String| ToolCall {
        index: Some(i as u64),
        id: format!("call_work_{}_{}", messages.len(), i),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments,
        },
    };
    let step: Vec<ToolCall> = WORK_TOOLS
        .iter()
        .filter(|t| tools_collection.contains_key(t.name))
        .enumerate()
        .map(|(i, t)| call(i, t.name, t.arguments.to_string()))
        .collect();
    let summary = format!("Worked {} steps on: {}", steps, job);
    let (message, finish_reason) = if rounds < steps && !step.is_empty() {
        (assistant_message(None, Some(step)), "tool_calls")
    } else if asked_to_report && !reported && tools_collection.contains_key("report_result") {
        let arguments = serde_json::json!({
            "status": "succeeded",
            "summary": summary,
            "data": {"job": job},
        });
        (
            assistant_message(
                None,
                Some(vec![call(0, "report_result", arguments.to_string())]),
            ),
            "tool_calls",
        )
    } else {
        (assistant_message(Some(summary), None), "stop")
    };
    OpenAIResponse {
        error: None,
        choices: Some(vec![Choice {
            message,
            finish_reason: Some(finish_reason.to_string()),
            native_finish_reason: None,
        }]),
        usage: Some(Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
        }),
        turn_usage: None,
        history: messages,
    }
}

/// The response to `call NAME ARGUMENTS`, the latest user message: the
/// call, then once it's made, what it returned as the answer.
fn call_response(name: &str, arguments: &str, messages: Vec<Message>) -> OpenAIResponse {
    let asked = messages.iter().rposition(|m| m.role == "user").unwrap_or(0);
    let returned = messages[asked..]
        .iter()
        .rev()
        .find(|m| m.role == "tool")
        .map(|m| m.content.clone().unwrap_or_default());
    let (message, finish_reason) = match returned {
        Some(text) => (assistant_message(Some(text), None), "stop"),
        None => (
            assistant_message(
                None,
                Some(vec![ToolCall {
                    index: Some(0),
                    id: format!("call_dummy_{}", messages.len()),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments: arguments.to_string(),
                    },
                }]),
            ),
            "tool_calls",
        ),
    };
    OpenAIResponse {
        error: None,
        choices: Some(vec![Choice {
            message,
            finish_reason: Some(finish_reason.to_string()),
            native_finish_reason: None,
        }]),
        usage: Some(Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
        }),
        turn_usage: None,
        history: messages,
    }
}

/// The model names that select the dummy: `dummy` or `test`, optionally
/// followed by `:` and comma-separated options:
///
/// - `slots=N`: at most N of its requests in flight at once - like
///   --max-parallel-requests, which the dummy otherwise follows as a real
///   model's requests do, but a limit of its own, shared only by whoever
///   uses this very model name. A test can so have one without slowing
///   down, or being slowed down by, any other running alongside it.
/// - `id=NAME`: nothing but a different model name, and so a separate
///   `slots` limit.
pub fn is_dummy_model(model: &str) -> bool {
    let base = model.split(':').next().unwrap_or("");
    base == "dummy" || base == "test"
}

/// The limit on requests in flight `model` follows (see `is_dummy_model`).
fn limiter(model: &str) -> Result<&'static RequestLimiter, String> {
    let Some((_, options)) = model.split_once(':') else {
        return Ok(request_limiter());
    };
    let mut slots = None;
    for option in options.split(',') {
        match option.split_once('=') {
            Some(("slots", n)) => {
                slots = Some(
                    n.parse::<usize>()
                        .map_err(|_| format!("dummy model: invalid slots '{}'", n))?,
                )
            }
            Some(("id", _)) => {}
            _ => return Err(format!("dummy model: unknown option '{}'", option)),
        }
    }
    let Some(slots) = slots else {
        return Ok(request_limiter());
    };
    static LIMITERS: OnceLock<Mutex<HashMap<String, &'static RequestLimiter>>> = OnceLock::new();
    let mut limiters = LIMITERS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    Ok(*limiters
        .entry(model.to_string())
        .or_insert_with(|| Box::leak(Box::new(RequestLimiter::new(slots)))))
}

/// How many of `model`'s requests are in flight right now.
pub fn requests_in_flight(model: &str) -> usize {
    limiter(model).map_or(0, |l| l.in_use())
}

/// Waits `ms`, or until Ctrl-C.
fn pause(
    ms: u64,
    ctrl_c_rx: &Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<(), InterruptedError> {
    let deadline = Instant::now() + Duration::from_millis(ms);
    loop {
        check_interrupted(ctrl_c_rx)?;
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Ok(());
        }
        std::thread::sleep(left.min(Duration::from_millis(5)));
    }
}

pub fn post_request_dummy(
    messages: Vec<Message>,
    model: &str,
    tools_collection: &crate::openai::ToolsCollection,
    mode: ResponseMode,
    ctx: &crate::ToolContext,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<OpenAIResponse, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let mut messages = messages;
    let mut turn_usage: Option<Usage> = None;
    let limiter = limiter(model)?;

    loop {
        check_interrupted(&ctrl_c_rx)?;

        let chars = conversation_chars(&messages);
        if chars > CONTEXT_LIMIT_CHARS {
            return Err(Box::new(ContextLengthError {
                message: format!(
                    "got API error code: 400: This model's maximum context length is {} characters, \
                     however the request has {} characters (context_length_exceeded)",
                    CONTEXT_LIMIT_CHARS, chars
                ),
                history: messages,
            }));
        }

        // Like a real request, it waits for a free slot, and holds it
        // until the response has been read in full - not while its tools
        // run.
        let slot = limiter.acquire(&ctrl_c_rx, &mode, start_time)?;

        let work = work_command(&messages);
        match work {
            Some((_, _, delay_ms)) => pause(delay_ms, &ctrl_c_rx)?,
            None => std::thread::sleep(Duration::from_millis(50)),
        }

        let turn = TURN_COUNTER.fetch_add(1, Ordering::Relaxed);

        let has_tools = !tools_collection.is_empty();
        let is_tool_result = messages.last().map(|m| m.role == "tool").unwrap_or(false);
        let cmd = match work {
            Some((_, steps, delay_ms)) => DummyCommand::Work { steps, delay_ms },
            None => parse_dummy_command(last_user_message(&messages)),
        };
        let is_tool_turn = turn % 2 == 1 && !cmd.forces_text();

        let response = if let Some((start, steps, _)) = work {
            work_response(start, steps, messages.clone(), tools_collection)
        } else if let DummyCommand::Call { name, arguments } = &cmd {
            call_response(name, arguments, messages.clone())
        } else if is_tool_turn && has_tools && !is_tool_result {
            make_tool_call_response(turn, messages.clone())
        } else {
            make_text_response(turn, messages.clone())
        };
        accumulate_usage(&mut turn_usage, response.usage.as_ref());

        let choice = response
            .choices
            .as_ref()
            .and_then(|c| c.first())
            .ok_or("No choices in dummy response")?;

        if let ResponseMode::Streaming {
            ref stream_handler,
            ref progress_handler,
            ..
        } = mode
        {
            progress_handler(&ProgressInfo {
                status: StatusUpdate::Thinking,
                elapsed_ms: start_time.elapsed().as_millis() as u64,
            })?;

            if work.is_none() {
                std::thread::sleep(Duration::from_millis(100));
            }

            let word_delay = cmd.word_delay_ms();

            if let Some(ref content) = choice.message.content {
                for line in content.split('\n') {
                    if !line.is_empty() {
                        for word in line.split_whitespace() {
                            check_interrupted(&ctrl_c_rx).map_err(|e| {
                                let _ = stream_handler("");
                                e
                            })?;
                            stream_handler(&format!("{} ", word))?;
                            std::thread::sleep(Duration::from_millis(word_delay));
                        }
                    }
                    stream_handler("\n")?;
                }
                stream_handler("")?;
            }
        }

        drop(slot);
        let assistant_msg = choice.message.clone();
        let has_tool_calls = assistant_msg.tool_calls.is_some();

        if assistant_msg.content.is_some() || assistant_msg.tool_calls.is_some() {
            messages.push(assistant_msg.clone());
        }

        if has_tool_calls {
            let tool_calls = assistant_msg.tool_calls.as_ref().unwrap();

            messages.extend(run_tool_calls(
                tools_collection,
                tool_calls,
                ctx,
                &mode,
                start_time,
            )?);
            ctx.checkpoint(&messages);

            continue;
        }

        if let ResponseMode::Streaming {
            ref progress_handler,
            ..
        } = mode
        {
            progress_handler(&ProgressInfo {
                status: StatusUpdate::Complete {
                    usage: response.usage.clone(),
                },
                elapsed_ms: start_time.elapsed().as_millis() as u64,
            })?;
        }

        return Ok(OpenAIResponse {
            error: None,
            choices: response.choices,
            usage: response.usage,
            turn_usage,
            history: messages,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_and_call_commands() {
        assert!(matches!(
            parse_dummy_command("work 3 20 on a.rs\n\nThen call report_result"),
            DummyCommand::Work {
                steps: 3,
                delay_ms: 20
            }
        ));
        assert!(matches!(
            parse_dummy_command("work on a.rs"),
            DummyCommand::Work {
                steps: 2,
                delay_ms: 0
            }
        ));
        match parse_dummy_command(r#"call glob {"pattern": "*.rs"}"#) {
            DummyCommand::Call { name, arguments } => {
                assert_eq!(name, "glob");
                assert_eq!(arguments, r#"{"pattern": "*.rs"}"#);
            }
            _ => panic!("not a call"),
        }
        assert!(matches!(
            parse_dummy_command("workout"),
            DummyCommand::Normal
        ));
    }

    #[test]
    fn test_model_options_pick_the_request_limit() {
        assert!(is_dummy_model("dummy") && is_dummy_model("dummy:slots=2"));
        assert!(!is_dummy_model("dummyish"));
        assert!(std::ptr::eq(limiter("dummy").unwrap(), request_limiter()));
        let a = limiter("dummy:slots=2,id=options-a").unwrap();
        assert!(std::ptr::eq(
            a,
            limiter("dummy:slots=2,id=options-a").unwrap()
        ));
        assert!(!std::ptr::eq(
            a,
            limiter("dummy:slots=2,id=options-b").unwrap()
        ));
        assert!(limiter("dummy:slots=x").is_err());
        assert!(limiter("dummy:fast=1").is_err());
    }
}
