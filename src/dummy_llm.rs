use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use crate::openai::{
    Choice, ContextLengthError, FunctionCall, InterruptedError, Message, OpenAIResponse,
    ProgressInfo, ResponseMode, StatusUpdate, ToolCall, Usage, tool_call,
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

enum DummyCommand {
    Long(usize),
    Slow(usize),
    Normal,
}

impl DummyCommand {
    fn word_delay_ms(&self) -> u64 {
        match self {
            DummyCommand::Slow(_) => 200,
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
        DummyCommand::Normal => make_normal_text(turn, messages.len()),
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
        history: messages,
    }
}

pub fn is_dummy_model(model: &str) -> bool {
    model == "dummy" || model == "test"
}

pub fn post_request_dummy(
    messages: Vec<Message>,
    tools_collection: &crate::openai::ToolsCollection,
    mode: ResponseMode,
    ctx: &crate::ToolContext,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
) -> Result<OpenAIResponse, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let mut messages = messages;

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

        std::thread::sleep(Duration::from_millis(50));

        let turn = TURN_COUNTER.fetch_add(1, Ordering::Relaxed);

        let has_tools = !tools_collection.is_empty();
        let is_tool_result = messages.last().map(|m| m.role == "tool").unwrap_or(false);
        let cmd = parse_dummy_command(last_user_message(&messages));
        let is_tool_turn = turn % 2 == 1 && !cmd.forces_text();

        let response = if is_tool_turn && has_tools && !is_tool_result {
            make_tool_call_response(turn, messages.clone())
        } else {
            make_text_response(turn, messages.clone())
        };

        let choice = response
            .choices
            .as_ref()
            .and_then(|c| c.first())
            .ok_or("No choices in dummy response")?;

        if let ResponseMode::Streaming {
            ref stream_handler,
            ref progress_handler,
        } = mode
        {
            progress_handler(&ProgressInfo {
                status: StatusUpdate::Thinking,
                elapsed_ms: start_time.elapsed().as_millis() as u64,
            })?;

            std::thread::sleep(Duration::from_millis(100));

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

        let assistant_msg = choice.message.clone();
        let has_tool_calls = assistant_msg.tool_calls.is_some();

        if assistant_msg.content.is_some() || assistant_msg.tool_calls.is_some() {
            messages.push(assistant_msg.clone());
        }

        if has_tool_calls {
            let tool_calls = assistant_msg.tool_calls.as_ref().unwrap();

            for tc in tool_calls {
                if let ResponseMode::Streaming {
                    ref progress_handler,
                    ..
                } = mode
                {
                    progress_handler(&ProgressInfo {
                        status: StatusUpdate::ToolStart {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                        elapsed_ms: start_time.elapsed().as_millis() as u64,
                    })?;
                    progress_handler(&ProgressInfo {
                        status: StatusUpdate::ToolExecuting {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                        elapsed_ms: start_time.elapsed().as_millis() as u64,
                    })?;
                }

                let tool_start = start_time.elapsed();
                let msg = tool_call(tools_collection, tc, ctx)?;
                let tool_duration = start_time.elapsed() - tool_start;

                if let ResponseMode::Streaming {
                    ref progress_handler,
                    ..
                } = mode
                {
                    progress_handler(&ProgressInfo {
                        status: StatusUpdate::ToolComplete {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                            duration_ms: tool_duration.as_millis() as u64,
                        },
                        elapsed_ms: start_time.elapsed().as_millis() as u64,
                    })?;
                }

                messages.push(msg);
            }

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
            history: messages,
        });
    }
}
