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

/// Coordinates the status spinner with normal output on the same terminal.
///
/// The spinner lives on the current, not-yet-terminated line: each tick
/// erases the previous frame with `\r\x1b[2K` and redraws it, and any real
/// output line erases it the same way before printing.  Real output stays in
/// the terminal's normal scrollback, so this makes no assumptions about
/// terminal size, scroll regions or cursor save/restore, none of which are
/// implemented consistently enough across terminals to rely on: an earlier
/// version reserved a fixed bottom row via a scroll region, which could fall
/// out of sync with the terminal (e.g. across a resize) and let real output
/// and the spinner land on the same row, corrupting both.
static SPINNER_SHOWN: Mutex<bool> = Mutex::new(false);

fn raw_write(s: &str) {
    let bytes = s.as_bytes();
    unsafe {
        libc::write(
            libc::STDERR_FILENO,
            bytes.as_ptr() as *const libc::c_void,
            bytes.len(),
        );
    }
}

/// Erases the spinner frame if one is currently shown.
fn clear_spinner_locked(shown: &mut bool) {
    if *shown {
        raw_write("\r\x1b[2K");
        *shown = false;
    }
}

/// Writes a complete, self-contained chunk of output (normally a line ending
/// in "\n"), erasing the spinner first if one is on the current line.
pub(crate) fn write_stderr(s: &str) {
    let mut shown = SPINNER_SHOWN.lock().unwrap_or_else(|e| e.into_inner());
    clear_spinner_locked(&mut shown);
    raw_write(s);
}

/// Draws one spinner frame (no trailing newline), erasing the previous frame
/// first if one is still on screen.
fn draw_spinner_locked(shown: &mut bool, text: &str) {
    if *shown {
        raw_write("\r\x1b[2K");
    }
    raw_write(text);
    *shown = true;
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
                                format!(" \x1b[90m(+{} more)\x1b[0m", count - 1)
                            } else {
                                String::new()
                            };
                            Some(format!(
                                "\x1b[{}m {} {}: {} \x1b[0m\x1b[90m│ {:.1}s\x1b[0m{}",
                                entry.color, spinner, agent, entry.message, secs, suffix
                            ))
                        }
                    } else {
                        None
                    };

                    let mut shown = SPINNER_SHOWN.lock().unwrap_or_else(|e| e.into_inner());
                    match &text {
                        Some(text) => draw_spinner_locked(&mut shown, text),
                        None if was_active => clear_spinner_locked(&mut shown),
                        None => {}
                    }
                    drop(shown);

                    was_active = text.is_some();
                    tick += 1;
                    std::thread::sleep(Duration::from_millis(TICK_INTERVAL_MS));
                }
                let mut shown = SPINNER_SHOWN.lock().unwrap_or_else(|e| e.into_inner());
                clear_spinner_locked(&mut shown);
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
