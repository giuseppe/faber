use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SPINNER_CHARS: &[&str] = &["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];
const TICK_INTERVAL_MS: u64 = 200;
const CYCLE_TICKS: u64 = 10; // cycle to next agent every 10 ticks (2s)

#[allow(dead_code)]
pub enum Position {
    Top,
    Bottom,
}

pub const STATUS_LINE_COUNT: u16 = 1;
pub const POSITION: Position = Position::Bottom;

struct AgentEntry {
    message: String,
    start_time: Instant,
    color: u8,
}

fn write_stderr(s: &str) {
    let bytes = s.as_bytes();
    unsafe {
        libc::write(
            libc::STDERR_FILENO,
            bytes.as_ptr() as *const libc::c_void,
            bytes.len(),
        );
    }
}

fn set_terminal_rows(rows: u16) {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 {
            ws.ws_row = rows;
            libc::ioctl(libc::STDERR_FILENO, libc::TIOCSWINSZ, &ws);
        }
    }
}

pub struct StatusBar {
    enabled: bool,
    original_rows: u16,
    entries: Arc<Mutex<Vec<(String, AgentEntry)>>>,
    stop_ticker: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    default_color: Arc<AtomicU8>,
    ticker_handle: Option<std::thread::JoinHandle<()>>,
}

impl StatusBar {
    pub fn new() -> Self {
        let term = console::Term::stderr();
        if !term.is_term() {
            return Self::disabled();
        }

        let (rows, _cols) = term.size();
        if rows < 3 {
            return Self::disabled();
        }

        let (scroll_top, scroll_bottom, status_line) = match POSITION {
            Position::Top => (STATUS_LINE_COUNT + 1, rows, 1),
            Position::Bottom => (1, rows - STATUS_LINE_COUNT, rows),
        };

        write_stderr(&format!(
            "\x1b[{};{}r\x1b[{};1H\x1b[2K\x1b[{};1H",
            scroll_top, scroll_bottom, status_line, scroll_bottom
        ));

        set_terminal_rows(rows - STATUS_LINE_COUNT);

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
                    let is_active;
                    if paused.load(Ordering::Relaxed) {
                        is_active = false;
                    } else if let Ok(guard) = entries.lock() {
                        is_active = !guard.is_empty();
                        if is_active {
                            let idx = if guard.len() > 1 {
                                ((tick / CYCLE_TICKS) as usize) % guard.len()
                            } else {
                                0
                            };
                            let (agent, entry) = &guard[idx];
                            let spinner = SPINNER_CHARS[(tick as usize) % SPINNER_CHARS.len()];
                            let secs = entry.start_time.elapsed().as_secs_f64();
                            let c = entry.color;
                            let count = guard.len();

                            let suffix = if count > 1 {
                                format!(" \x1b[90m(+{} more)\x1b[0m", count - 1)
                            } else {
                                String::new()
                            };

                            write_stderr(&format!(
                                "\x1b7\x1b[{};1H\x1b[2K\x1b[{}m {} {}: {} \x1b[0m\x1b[90m│ {:.1}s\x1b[0m{}\x1b8",
                                status_line, c, spinner, agent, entry.message, secs, suffix
                            ));
                        }
                    } else {
                        is_active = false;
                    }

                    if !is_active && was_active {
                        write_stderr(&format!(
                            "\x1b7\x1b[{};1H\x1b[2K\x1b8",
                            status_line
                        ));
                    }
                    was_active = is_active;
                    tick += 1;
                    std::thread::sleep(Duration::from_millis(TICK_INTERVAL_MS));
                }
                write_stderr(&format!(
                    "\x1b7\x1b[{};1H\x1b[2K\x1b8",
                    status_line
                ));
            })
        };

        Self {
            enabled: true,
            original_rows: rows,
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
            original_rows: 0,
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
        if self.enabled {
            set_terminal_rows(self.original_rows);
            write_stderr(&format!(
                "\x1b[r\x1b[{};1H\x1b[2K\x1b[{};1H",
                self.original_rows,
                self.original_rows - STATUS_LINE_COUNT
            ));
        }
    }
}
