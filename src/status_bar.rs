use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SPINNER_CHARS: &[&str] = &["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];
const TICK_INTERVAL_MS: u64 = 200;
const CYCLE_TICKS: u64 = 10; // cycle to next agent every 10 ticks (2s)

struct AgentEntry {
    message: String,
    start_time: Instant,
    color: u8,
}

/// Coordinates the status spinner with normal output - including output
/// streamed in as it arrives, with no trailing newline yet - on the same
/// terminal.
///
/// Whatever is on the current, not-yet-terminated line is tracked here.
/// Real output stays in the terminal's normal scrollback and the spinner
/// lives on the current line, redrawn in place each tick with
/// `\r\x1b[2K`; this makes no assumptions about terminal size, scroll
/// regions or cursor save/restore, none of which are implemented
/// consistently enough across terminals to rely on (an earlier version
/// reserved a fixed bottom row via a scroll region, which could fall out
/// of sync with the terminal - e.g. across a resize - and let real output
/// and the spinner land on the same row, corrupting both).
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum LineState {
    /// Nothing is on the current line.
    Empty,
    /// The ticker's own spinner frame is on the current line.
    Spinner,
    /// Streamed output with no trailing newline yet is on the current line
    /// (e.g. a model's answer, printed as it arrives rather than only once
    /// a full line has accumulated).
    Partial,
}

static LINE_STATE: Mutex<LineState> = Mutex::new(LineState::Empty);

fn raw_write(s: &str) {
    if s.is_empty() {
        return;
    }
    let bytes = s.as_bytes();
    unsafe {
        libc::write(
            libc::STDERR_FILENO,
            bytes.as_ptr() as *const libc::c_void,
            bytes.len(),
        );
    }
}

/// One thing that can happen to the current line.
enum LineOp<'a> {
    /// A complete, self-contained, already newline-terminated chunk of
    /// output (e.g. a tool-completion message or an injected notification).
    /// Never a continuation of an open partial line - that would be a
    /// different stream's or thread's content - so whatever is on the
    /// current line is closed out first.
    CompleteLine(&'a str),
    /// Appends `text` (no trailing newline) to the current line. A no-op
    /// for empty text.
    Partial(&'a str),
    /// An explicit newline byte from a stream's own output: always writes
    /// `\n` (erasing the spinner first if shown), whatever is on the
    /// current line - open partial text, or nothing (a deliberate blank
    /// line, which this preserves exactly as the stream produced it).
    Newline,
    /// "Nothing more is coming for now": closes the current line *only if*
    /// there's an open partial line to close, otherwise a true no-op - in
    /// particular, it never invents a newline that wasn't there, and it
    /// leaves the spinner alone rather than erasing it. Safe to call
    /// idempotently (e.g. once a stream ends, or before switching to
    /// reporting a different kind of progress) without it ever printing a
    /// spurious blank line for a partial line that was already closed, or
    /// never opened at all.
    Finish,
    /// Draws one spinner frame, erasing the previous one first if it's
    /// still shown.
    DrawSpinner(&'a str),
    /// Erases the spinner if that's what's currently shown.
    ClearSpinner,
}

/// Pure decision of what bytes to actually write to the terminal for `op`
/// given the current line state, and the resulting state - kept separate
/// from the actual write so the (bug-prone) state transitions can be tested
/// without touching the terminal.
fn apply(state: LineState, op: LineOp) -> (String, LineState) {
    let erase_spinner = if state == LineState::Spinner {
        "\r\x1b[2K"
    } else {
        ""
    };
    match op {
        LineOp::CompleteLine(s) => {
            let prefix = match state {
                LineState::Spinner => "\r\x1b[2K",
                LineState::Partial => "\n",
                LineState::Empty => "",
            };
            (format!("{}{}", prefix, s), LineState::Empty)
        }
        LineOp::Partial(s) => {
            if s.is_empty() {
                (String::new(), state)
            } else {
                (format!("{}{}", erase_spinner, s), LineState::Partial)
            }
        }
        LineOp::Newline => (format!("{}\n", erase_spinner), LineState::Empty),
        LineOp::Finish => {
            if state == LineState::Partial {
                ("\n".to_string(), LineState::Empty)
            } else {
                (String::new(), state)
            }
        }
        LineOp::DrawSpinner(text) => (format!("{}{}", erase_spinner, text), LineState::Spinner),
        LineOp::ClearSpinner => {
            if state == LineState::Spinner {
                ("\r\x1b[2K".to_string(), LineState::Empty)
            } else {
                (String::new(), state)
            }
        }
    }
}

/// Builds one spinner frame, truncating it (with a flat color instead of the
/// usual two-tone split) if it would be too long for `cols` columns.
///
/// A status line longer than the terminal is wrapped by the terminal itself
/// onto a second row, which breaks the single-line erase-and-redraw scheme
/// above: the next tick's `\r\x1b[2K` only clears the row the cursor is
/// actually on (the wrapped second row), permanently leaving the first
/// row's text behind as stray scrollback. Measured by character count, not
/// true display width, so a message full of wide (e.g. CJK) characters
/// could still wrap; that's an accepted approximation, consistent with the
/// rest of this codebase, rather than pulling in a unicode-width crate for
/// what a truncated ellipsis mostly papers over anyway.
fn build_status_line(
    color: u8,
    spinner: &str,
    agent: &str,
    message: &str,
    secs: f64,
    suffix: &str,
    cols: usize,
) -> String {
    let plain = format!(
        " {} {}: {} │ {:.1}s{}",
        spinner, agent, message, secs, suffix
    );
    let width = cols.saturating_sub(1).max(1);
    if plain.chars().count() <= width {
        format!(
            "\x1b[{}m {} {}: {} \x1b[0m\x1b[90m│ {:.1}s{}\x1b[0m",
            color, spinner, agent, message, secs, suffix
        )
    } else {
        let budget = width.saturating_sub(1); // room for the ellipsis
        let truncated: String = plain.chars().take(budget).collect();
        format!("\x1b[{}m{}…\x1b[0m", color, truncated)
    }
}

fn run(op: LineOp) {
    let mut state = LINE_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let (bytes, new_state) = apply(*state, op);
    raw_write(&bytes);
    *state = new_state;
}

/// Writes a complete, self-contained chunk of output (normally a line
/// ending in "\n"), closing out whatever is on the current line first.
pub(crate) fn write_stderr(s: &str) {
    run(LineOp::CompleteLine(s));
}

/// Appends `text` (no trailing newline) to the current line, so it's
/// visible immediately instead of waiting for a full line to accumulate.
pub(crate) fn write_partial(text: &str) {
    run(LineOp::Partial(text));
}

/// Writes an explicit newline byte from a stream's own output - always,
/// even to reproduce a deliberate blank line exactly as the stream wrote
/// it.
pub(crate) fn write_newline() {
    run(LineOp::Newline);
}

/// Closes the current line if (and only if) there's an open partial line
/// to close; a safe no-op otherwise. Call this whenever a stream is done
/// contributing to the current line for now (it ended, or it's handing off
/// to something else, like tool-call argument accumulation) without
/// knowing - or needing to know - whether anything was actually left open.
pub(crate) fn finish_partial_line() {
    run(LineOp::Finish);
}

pub struct StatusBar {
    enabled: bool,
    entries: Arc<Mutex<Vec<(String, AgentEntry)>>>,
    stop_ticker: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    default_color: Arc<AtomicU8>,
    ticker_handle: Option<std::thread::JoinHandle<()>>,
}

impl StatusBar {
    pub fn new() -> Self {
        if !console::Term::stderr().is_term() {
            return Self::disabled();
        }

        let entries: Arc<Mutex<Vec<(String, AgentEntry)>>> = Arc::new(Mutex::new(Vec::new()));
        let stop_ticker = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let default_color = Arc::new(AtomicU8::new(36));

        let ticker_handle = {
            let entries = entries.clone();
            let stop = stop_ticker.clone();
            let paused = paused.clone();

            std::thread::spawn(move || {
                let mut tick: u64 = 0;
                let mut was_active = false;
                while !stop.load(Ordering::Relaxed) {
                    let text = if paused.load(Ordering::Relaxed) {
                        None
                    } else if let Ok(guard) = entries.lock() {
                        if guard.is_empty() {
                            None
                        } else {
                            let idx = if guard.len() > 1 {
                                ((tick / CYCLE_TICKS) as usize) % guard.len()
                            } else {
                                0
                            };
                            let (agent, entry) = &guard[idx];
                            let spinner = SPINNER_CHARS[(tick as usize) % SPINNER_CHARS.len()];
                            let secs = entry.start_time.elapsed().as_secs_f64();
                            let count = guard.len();
                            let suffix = if count > 1 {
                                format!(" (+{} more)", count - 1)
                            } else {
                                String::new()
                            };
                            let (_, cols) = console::Term::stderr().size();
                            Some(build_status_line(
                                entry.color,
                                spinner,
                                agent,
                                &entry.message,
                                secs,
                                &suffix,
                                cols as usize,
                            ))
                        }
                    } else {
                        None
                    };

                    // While streamed output is on the current line, leave
                    // it alone: it isn't safe to erase-and-redraw a line
                    // that holds something other than the spinner's own
                    // last frame, and the streamed text itself is now the
                    // "it's alive" signal.
                    let mut state = LINE_STATE.lock().unwrap_or_else(|e| e.into_inner());
                    if *state != LineState::Partial {
                        let op = match &text {
                            Some(text) => Some(LineOp::DrawSpinner(text)),
                            None if was_active => Some(LineOp::ClearSpinner),
                            None => None,
                        };
                        if let Some(op) = op {
                            let (bytes, new_state) = apply(*state, op);
                            raw_write(&bytes);
                            *state = new_state;
                        }
                    }
                    drop(state);

                    was_active = text.is_some();
                    tick += 1;
                    std::thread::sleep(Duration::from_millis(TICK_INTERVAL_MS));
                }
                run(LineOp::ClearSpinner);
            })
        };

        Self {
            enabled: true,
            entries,
            stop_ticker,
            paused,
            default_color,
            ticker_handle: Some(ticker_handle),
        }
    }

    fn disabled() -> Self {
        Self {
            enabled: false,
            entries: Arc::new(Mutex::new(Vec::new())),
            stop_ticker: Arc::new(AtomicBool::new(true)),
            paused: Arc::new(AtomicBool::new(true)),
            default_color: Arc::new(AtomicU8::new(36)),
            ticker_handle: None,
        }
    }

    pub fn set_color(&self, ansi_code: u8) {
        self.default_color.store(ansi_code, Ordering::Relaxed);
    }

    pub fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
    }

    pub fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
    }

    pub fn set_agent_status(&self, agent: &str, msg: &str, reset_timer: bool) {
        if !self.enabled {
            return;
        }
        if let Ok(mut guard) = self.entries.lock() {
            if let Some(pos) = guard.iter().position(|(name, _)| name == agent) {
                guard[pos].1.message = msg.to_string();
                if reset_timer {
                    guard[pos].1.start_time = Instant::now();
                }
            } else {
                guard.push((
                    agent.to_string(),
                    AgentEntry {
                        message: msg.to_string(),
                        start_time: Instant::now(),
                        color: self.default_color.load(Ordering::Relaxed),
                    },
                ));
            }
        }
    }

    pub fn set_agent_color(&self, agent: &str, ansi_code: u8) {
        if let Ok(mut guard) = self.entries.lock() {
            if let Some(pos) = guard.iter().position(|(name, _)| name == agent) {
                guard[pos].1.color = ansi_code;
            }
        }
    }

    pub fn clear_agent_status(&self, agent: &str) {
        if !self.enabled {
            return;
        }
        if let Ok(mut guard) = self.entries.lock() {
            guard.retain(|(name, _)| name != agent);
        }
    }
}

impl Drop for StatusBar {
    fn drop(&mut self) {
        self.stop_ticker.store(true, Ordering::Relaxed);
        if let Some(handle) = self.ticker_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strips `\x1b[...m` SGR sequences, leaving only what would actually
    /// occupy columns on screen - used to measure the *visible* width of a
    /// built status line, which is the thing that must never exceed the
    /// terminal's column count.
    fn visible(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn test_apply_complete_line_from_empty_has_no_prefix() {
        let (bytes, state) = apply(LineState::Empty, LineOp::CompleteLine("done\n"));
        assert_eq!(bytes, "done\n");
        assert_eq!(state, LineState::Empty);
    }

    #[test]
    fn test_apply_complete_line_erases_spinner() {
        let (bytes, state) = apply(LineState::Spinner, LineOp::CompleteLine("done\n"));
        assert_eq!(bytes, "\r\x1b[2Kdone\n");
        assert_eq!(state, LineState::Empty);
    }

    #[test]
    fn test_apply_complete_line_closes_open_partial_with_newline() {
        // A complete line is never a continuation of an open partial line
        // (that's a different stream's or thread's content, e.g. a
        // sub-agent's message arriving while the main agent is mid-answer);
        // it must not be visually run onto the same line.
        let (bytes, state) = apply(LineState::Partial, LineOp::CompleteLine("done\n"));
        assert_eq!(bytes, "\ndone\n");
        assert_eq!(state, LineState::Empty);
    }

    #[test]
    fn test_apply_partial_empty_text_is_a_no_op() {
        for state in [LineState::Empty, LineState::Spinner, LineState::Partial] {
            let (bytes, new_state) = apply(state, LineOp::Partial(""));
            assert_eq!(bytes, "");
            assert_eq!(new_state, state);
        }
    }

    #[test]
    fn test_apply_partial_from_empty_just_writes_the_text() {
        let (bytes, state) = apply(LineState::Empty, LineOp::Partial("hi"));
        assert_eq!(bytes, "hi");
        assert_eq!(state, LineState::Partial);
    }

    #[test]
    fn test_apply_partial_erases_spinner_first() {
        let (bytes, state) = apply(LineState::Spinner, LineOp::Partial("hi"));
        assert_eq!(bytes, "\r\x1b[2Khi");
        assert_eq!(state, LineState::Partial);
    }

    #[test]
    fn test_apply_partial_continuing_an_open_line_just_appends() {
        // Continuing its own open line must never insert a newline or
        // erase anything - that would corrupt streamed text mid-word.
        let (bytes, state) = apply(LineState::Partial, LineOp::Partial("hi"));
        assert_eq!(bytes, "hi");
        assert_eq!(state, LineState::Partial);
    }

    #[test]
    fn test_apply_newline_from_empty_writes_a_bare_newline() {
        // A deliberate blank line (two consecutive newlines in the
        // model's own output) must still produce a newline.
        let (bytes, state) = apply(LineState::Empty, LineOp::Newline);
        assert_eq!(bytes, "\n");
        assert_eq!(state, LineState::Empty);
    }

    #[test]
    fn test_apply_newline_from_partial_appends_newline() {
        let (bytes, state) = apply(LineState::Partial, LineOp::Newline);
        assert_eq!(bytes, "\n");
        assert_eq!(state, LineState::Empty);
    }

    #[test]
    fn test_apply_newline_from_spinner_erases_then_writes_newline() {
        let (bytes, state) = apply(LineState::Spinner, LineOp::Newline);
        assert_eq!(bytes, "\r\x1b[2K\n");
        assert_eq!(state, LineState::Empty);
    }

    #[test]
    fn test_apply_finish_from_empty_is_a_no_op() {
        // Unlike Newline, Finish never invents a newline that wasn't
        // there - it's an idempotent "close if open" cleanup signal (end
        // of stream, or handing off to a different kind of progress
        // report), and must not print a spurious blank line when nothing
        // was actually open.
        let (bytes, state) = apply(LineState::Empty, LineOp::Finish);
        assert_eq!(bytes, "");
        assert_eq!(state, LineState::Empty);
    }

    #[test]
    fn test_apply_finish_from_partial_closes_it() {
        let (bytes, state) = apply(LineState::Partial, LineOp::Finish);
        assert_eq!(bytes, "\n");
        assert_eq!(state, LineState::Empty);
    }

    #[test]
    fn test_apply_finish_from_spinner_is_a_no_op() {
        // Finish isn't the right operation to be touching the spinner:
        // leave it alone rather than erasing it.
        let (bytes, state) = apply(LineState::Spinner, LineOp::Finish);
        assert_eq!(bytes, "");
        assert_eq!(state, LineState::Spinner);
    }

    #[test]
    fn test_apply_draw_spinner_from_empty() {
        let (bytes, state) = apply(LineState::Empty, LineOp::DrawSpinner("[spin]"));
        assert_eq!(bytes, "[spin]");
        assert_eq!(state, LineState::Spinner);
    }

    #[test]
    fn test_apply_draw_spinner_erases_previous_frame() {
        let (bytes, state) = apply(LineState::Spinner, LineOp::DrawSpinner("[spin2]"));
        assert_eq!(bytes, "\r\x1b[2K[spin2]");
        assert_eq!(state, LineState::Spinner);
    }

    #[test]
    fn test_apply_clear_spinner_erases_only_when_shown() {
        let (bytes, state) = apply(LineState::Spinner, LineOp::ClearSpinner);
        assert_eq!(bytes, "\r\x1b[2K");
        assert_eq!(state, LineState::Empty);

        for state_in in [LineState::Empty, LineState::Partial] {
            let (bytes, state_out) = apply(state_in, LineOp::ClearSpinner);
            assert_eq!(bytes, "");
            assert_eq!(state_out, state_in);
        }
    }

    #[test]
    fn test_build_status_line_fits_keeps_two_tone_style() {
        let line = build_status_line(36, "⣾", "default", "Thinking", 1.2, "", 80);
        assert!(line.contains("\x1b[36m"));
        assert!(line.contains("\x1b[90m"));
        assert_eq!(visible(&line), " ⣾ default: Thinking │ 1.2s");
    }

    #[test]
    fn test_build_status_line_includes_suffix_when_it_fits() {
        let line = build_status_line(36, "⣾", "default", "Thinking", 1.2, " (+2 more)", 80);
        assert_eq!(visible(&line), " ⣾ default: Thinking │ 1.2s (+2 more)");
    }

    #[test]
    fn test_build_status_line_truncates_instead_of_wrapping() {
        // The message alone is far longer than the terminal is wide - this
        // is exactly the case that used to wrap onto a second row and leave
        // stray text behind once the next tick redrew a shorter frame.
        let long_message = "Streaming (128829 bytes, 450 chunks)".repeat(3);
        let line = build_status_line(36, "⣾", "default", &long_message, 12.3, "", 40);
        let shown = visible(&line);
        assert!(
            shown.chars().count() <= 39,
            "visible line is {} chars, wider than the 40-column terminal: {:?}",
            shown.chars().count(),
            shown
        );
        assert!(shown.ends_with('…'));
        assert!(!shown.contains('\n'));
    }

    #[test]
    fn test_build_status_line_never_exceeds_terminal_width() {
        let long_message = "x".repeat(500);
        for cols in [0usize, 1, 2, 10, 20, 40, 80, 200] {
            let line = build_status_line(
                36,
                "⣾",
                "agent-name",
                &long_message,
                999.9,
                " (+9 more)",
                cols,
            );
            let width = visible(&line).chars().count();
            let budget = cols.saturating_sub(1).max(1);
            assert!(
                width <= budget,
                "cols={} allowed budget={} but produced a {}-char visible line: {:?}",
                cols,
                budget,
                width,
                line
            );
        }
    }

    #[test]
    fn test_build_status_line_short_terminal_does_not_panic() {
        // Degenerate widths must degrade gracefully, not panic (e.g. via
        // subtraction overflow or an out-of-bounds char take).
        for cols in [0, 1, 2, 3] {
            let _ = build_status_line(36, "⣾", "default", "Thinking", 0.0, "", cols);
        }
    }
}
