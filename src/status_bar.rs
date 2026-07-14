use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SPINNER_CHARS: &[&str] = &["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];
const TICK_INTERVAL_MS: u64 = 200;

#[allow(dead_code)]
pub enum Position {
    Top,
    Bottom,
}

pub const STATUS_LINE_COUNT: u16 = 1;
pub const POSITION: Position = Position::Bottom;

pub struct StatusBar {
    enabled: bool,
    message: Arc<Mutex<String>>,
    start_time: Arc<Mutex<Instant>>,
    active: Arc<AtomicBool>,
    stop_ticker: Arc<AtomicBool>,
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

        eprint!(
            "\x1b[{};{}r\x1b[{};1H",
            scroll_top, scroll_bottom, scroll_bottom
        );
        let _ = std::io::stderr().flush();

        let message = Arc::new(Mutex::new(String::new()));
        let start_time = Arc::new(Mutex::new(Instant::now()));
        let active = Arc::new(AtomicBool::new(false));
        let stop_ticker = Arc::new(AtomicBool::new(false));

        let ticker_handle = {
            let msg = message.clone();
            let time = start_time.clone();
            let active = active.clone();
            let stop = stop_ticker.clone();

            std::thread::spawn(move || {
                let mut tick: u64 = 0;
                let mut was_active = false;
                while !stop.load(Ordering::Relaxed) {
                    let is_active = active.load(Ordering::Relaxed);
                    if is_active {
                        let text = msg.lock().map(|m| m.clone()).unwrap_or_default();
                        let elapsed = time.lock().map(|t| t.elapsed()).unwrap_or_default();
                        let spinner = SPINNER_CHARS[(tick as usize) % SPINNER_CHARS.len()];
                        let secs = elapsed.as_secs_f64();

                        eprint!(
                            "\x1b7\x1b[{};1H\x1b[2K {} {} \x1b[90m│ {:.1}s\x1b[0m\x1b8",
                            status_line, spinner, text, secs
                        );
                        let _ = std::io::stderr().flush();
                        tick += 1;
                    } else if was_active {
                        eprint!("\x1b7\x1b[{};1H\x1b[2K\x1b8", status_line);
                        let _ = std::io::stderr().flush();
                    }
                    was_active = is_active;
                    std::thread::sleep(Duration::from_millis(TICK_INTERVAL_MS));
                }
                eprint!("\x1b7\x1b[{};1H\x1b[2K\x1b8", status_line);
                let _ = std::io::stderr().flush();
            })
        };

        Self {
            enabled: true,
            message,
            start_time,
            active,
            stop_ticker,
            ticker_handle: Some(ticker_handle),
        }
    }

    fn disabled() -> Self {
        Self {
            enabled: false,
            message: Arc::new(Mutex::new(String::new())),
            start_time: Arc::new(Mutex::new(Instant::now())),
            active: Arc::new(AtomicBool::new(false)),
            stop_ticker: Arc::new(AtomicBool::new(true)),
            ticker_handle: None,
        }
    }

    pub fn set_status(&self, msg: &str) {
        if !self.enabled {
            return;
        }
        if let Ok(mut m) = self.message.lock() {
            *m = msg.to_string();
        }
        self.active.store(true, Ordering::Relaxed);
    }

    pub fn reset_timer(&self) {
        if let Ok(mut t) = self.start_time.lock() {
            *t = Instant::now();
        }
    }

    pub fn clear_status(&self) {
        if !self.enabled {
            return;
        }
        self.active.store(false, Ordering::Relaxed);
    }
}

impl Drop for StatusBar {
    fn drop(&mut self) {
        self.stop_ticker.store(true, Ordering::Relaxed);
        if let Some(handle) = self.ticker_handle.take() {
            let _ = handle.join();
        }
        if self.enabled {
            let term = console::Term::stderr();
            let (rows, _) = term.size();
            eprint!(
                "\x1b[r\x1b[{};1H\x1b[2K\x1b[{};1H",
                rows,
                rows - STATUS_LINE_COUNT
            );
            let _ = std::io::stderr().flush();
        }
    }
}
