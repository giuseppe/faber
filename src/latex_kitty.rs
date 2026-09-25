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

//! Renders LaTeX math blocks found in a response as images, displayed
//! inline via the Kitty terminal graphics protocol - opt-in via
//! `--display-graphics`, since it shells out to a local LaTeX toolchain for
//! every response and only does anything under a Kitty-compatible terminal.

use base64::Engine;
use pulldown_cmark::{Alignment, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use std::error::Error;
use std::path::Path;
use std::process::Command;

/// True if the terminal we're running in is (or presents itself as) Kitty,
/// the only terminal whose graphics protocol this module speaks. Same
/// detection Kitty's own `icat` kitten and other Kitty-aware tools use.
pub fn is_kitty_terminal() -> bool {
    is_kitty_from_env(
        std::env::var("TERM").ok().as_deref(),
        std::env::var("KITTY_WINDOW_ID").ok().as_deref(),
    )
}

fn is_kitty_from_env(term: Option<&str>, kitty_window_id: Option<&str>) -> bool {
    term.is_some_and(|t| t.contains("kitty")) || kitty_window_id.is_some()
}

/// Names of any required toolchain binaries (pdflatex, pdftocairo) that
/// aren't found on PATH - empty if both are available. Checked once
/// upfront (see `main.rs`'s startup diagnostic for --display-graphics)
/// rather than only discovering this silently after failing to render the
/// first LaTeX block a response happens to contain.
pub fn missing_toolchain_tools() -> Vec<&'static str> {
    ["pdflatex", "pdftocairo"]
        .into_iter()
        .filter(|tool| Command::new(tool).arg("--version").output().is_err())
        .collect()
}

/// The three LaTeX math delimiter pairs this module recognizes, in the
/// order checked - `\[...\]`, `\(...\)`, and `$$...$$`. Shared between
/// `split_latex_segments` and `LatexSplitter`, so the two can never drift
/// out of sync on what counts as a delimiter.
const DELIMITERS: &[(&str, &str)] = &[("\\[", "\\]"), ("\\(", "\\)"), ("$$", "$$")];

/// One piece of a response as `split_latex_segments` (or `LatexSplitter`)
/// breaks it up: either plain text to print as-is, or a complete, delimited
/// LaTeX block ready to render.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StreamSegment {
    Text(String),
    Latex(String),
}

/// Splits a whole, already-complete `text` into `StreamSegment`s covering
/// `\[...\]`, `\(...\)`, and `$$...$$` LaTeX blocks, in the order they
/// appear. Concatenating every segment's own text back together always
/// reproduces `text` exactly - nothing is ever dropped, only reclassified
/// as plain text vs. LaTeX.
///
/// Blocks don't nest, and an opening delimiter with no matching close
/// later in the text is treated as if it were never a delimiter at all
/// (its own characters end up as ordinary text) rather than misparsing the
/// rest of the text as part of it - but, importantly, it's *only* that one
/// occurrence that's given up on: the scan keeps going afterward, so a
/// single stray/unterminated delimiter (a lone `\(` the model never
/// closed, e.g. because it was discussing LaTeX syntax itself, or
/// truncated output) doesn't cause every well-formed block later in the
/// same text to be silently missed too.
///
/// For a response that's still streaming in rather than already fully
/// known, see `LatexSplitter` - `LatexSplitter::finish()` calls this same
/// function on whatever it has left buffered once the stream ends, so a
/// stray delimiter mid-stream gets exactly this same recovery once the
/// full remaining text is finally known, even though `LatexSplitter::push`
/// itself can't look ahead far enough to do that while more input might
/// still be coming.
pub fn split_latex_segments(text: &str) -> Vec<StreamSegment> {
    let mut events = Vec::new();
    let mut text_start = 0; // start of the plain-text run building up
    let mut pos = 0; // scan cursor
    while pos < text.len() {
        let next = DELIMITERS
            .iter()
            .filter_map(|(open, close)| {
                text[pos..].find(open).map(|idx| (pos + idx, *open, *close))
            })
            .min_by_key(|(idx, _, _)| *idx);

        let Some((start, open, close)) = next else {
            break;
        };

        let search_from = start + open.len();
        match text[search_from..].find(close) {
            Some(rel_end) => {
                let end = search_from + rel_end + close.len();
                if start > text_start {
                    events.push(StreamSegment::Text(text[text_start..start].to_string()));
                }
                events.push(StreamSegment::Latex(text[start..end].to_string()));
                pos = end;
                text_start = end;
            }
            // No close for *this* open delimiter anywhere in the rest of
            // the text - it wasn't a real delimiter after all. Leave its
            // own characters as part of the ongoing plain-text run (don't
            // advance text_start) and resume scanning from just past its
            // *first* character, not its full length: advancing past the
            // whole thing would silently drop the rest of it from the
            // final Text segment, since text_start would then be behind
            // pos with a gap this loop never covers.
            None => pos = start + 1,
        }
    }
    if text_start < text.len() {
        events.push(StreamSegment::Text(text[text_start..].to_string()));
    }
    events
}

/// Incrementally splits a stream of text into `StreamSegment`s as it
/// arrives, so a LaTeX block can be rendered and displayed as an image in
/// place of its raw text the moment it finishes streaming in, instead of
/// only after the whole response is done and already printed (which -
/// confirmed against a real report of blocks not appearing "converted" -
/// just means every image shows up in one pile at the end, disconnected
/// from the text it came from).
///
/// Text is never held back longer than necessary to stay unambiguous:
/// everything up to the earliest confirmed delimiter is emitted
/// immediately, and the only thing ever buffered waiting for more input is
/// a possible one-character delimiter *prefix* at the very end of what's
/// arrived so far - a lone `$` that might become `$$`, or a lone `\` that
/// might become `\[` or `\(`.
///
/// Blocks don't nest (matching `split_latex_segments`): once a delimiter
/// opens, only its own matching close ends it. Unlike that whole-text
/// scanner, though, an opened-but-never-closed block isn't specially
/// recovered from *mid-stream* - there's no "rest of the text" to look
/// ahead into yet, only what's arrived so far - so it just stays buffered
/// until `finish()`, which hands whatever's left over to
/// `split_latex_segments` for one final, fully-resilient pass (nothing
/// arrives after `finish()`, so at that point the "rest of the text" *is*
/// finally known).
pub struct LatexSplitter {
    buf: String,
    open: Option<(&'static str, &'static str)>,
}

impl LatexSplitter {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            open: None,
        }
    }

    /// Feeds a newly-arrived chunk, returning the segments it's now
    /// possible to emit. May return nothing at all, if `chunk` only extends
    /// an already-open block or an as-yet-unresolved possible delimiter
    /// prefix (see the type's own docs).
    pub fn push(&mut self, chunk: &str) -> Vec<StreamSegment> {
        self.buf.push_str(chunk);
        let mut events = Vec::new();
        loop {
            match self.open {
                None => {
                    let next = DELIMITERS
                        .iter()
                        .filter_map(|(open, close)| {
                            self.buf.find(open).map(|idx| (idx, *open, *close))
                        })
                        .min_by_key(|(idx, _, _)| *idx);
                    match next {
                        Some((idx, open, close)) => {
                            if idx > 0 {
                                events.push(StreamSegment::Text(self.buf[..idx].to_string()));
                            }
                            self.buf.drain(..idx);
                            self.open = Some((open, close));
                        }
                        None => {
                            // Nothing confirmed yet. A trailing lone "$" or
                            // "\" could still grow into a real delimiter
                            // once more input arrives - ASCII, so this is
                            // always a valid char-boundary split - so hold
                            // just that one byte back; everything else is
                            // unambiguous and safe to emit now.
                            let hold = if self.buf.ends_with('$') || self.buf.ends_with('\\') {
                                1
                            } else {
                                0
                            };
                            let emit_len = self.buf.len() - hold;
                            if emit_len > 0 {
                                events.push(StreamSegment::Text(self.buf[..emit_len].to_string()));
                                self.buf.drain(..emit_len);
                            }
                            break;
                        }
                    }
                }
                Some((open, close)) => {
                    // Search *after* the open delimiter itself, not from
                    // the start of buf - buf still contains it at position
                    // 0, and for "$$" (where open == close) searching from
                    // 0 would immediately "find" that same opening
                    // delimiter as if it were already the close.
                    let search_from = open.len();
                    match self.buf[search_from..].find(close) {
                        Some(rel_end) => {
                            let end = search_from + rel_end + close.len();
                            events.push(StreamSegment::Latex(self.buf[..end].to_string()));
                            self.buf.drain(..end);
                            self.open = None;
                        }
                        None => break,
                    }
                }
            }
        }
        events
    }

    /// Signals end of stream: nothing more is ever coming, so whatever's
    /// left buffered - plain text, an unresolved delimiter prefix, or an
    /// opened-but-never-closed block - is now, finally, the "whole rest of
    /// the text" `split_latex_segments` needs to fully resolve it,
    /// including recovering from a stray delimiter that never closed
    /// rather than losing whatever came after it.
    pub fn finish(&mut self) -> Vec<StreamSegment> {
        self.open = None;
        if self.buf.is_empty() {
            Vec::new()
        } else {
            split_latex_segments(&std::mem::take(&mut self.buf))
        }
    }
}

impl Default for LatexSplitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Splits base64-encoded `png` data into the escape sequences Kitty's
/// graphics protocol (https://sw.kovidgoyal.net/kitty/graphics-protocol/)
/// needs to transmit and display it, chunked to the protocol's 4096-byte
/// limit per chunk. Pure - doesn't write anywhere - so the chunking/framing
/// logic is testable without a real terminal.
pub fn kitty_image_escape_sequences(png: &[u8]) -> Vec<String> {
    const CHUNK_SIZE: usize = 4096;

    let encoded = base64::engine::general_purpose::STANDARD.encode(png);
    let bytes = encoded.as_bytes();
    if bytes.is_empty() {
        return Vec::new();
    }

    let mut sequences = Vec::new();
    let mut offset = 0;
    let mut first = true;
    while offset < bytes.len() {
        let end = (offset + CHUNK_SIZE).min(bytes.len());
        // base64 output is pure ASCII, so byte-slicing never splits a
        // multi-byte UTF-8 character.
        let chunk = std::str::from_utf8(&bytes[offset..end]).expect("base64 output is ASCII");
        let more = end < bytes.len();
        let control = if first {
            format!("a=T,f=100,m={}", more as u8)
        } else {
            format!("m={}", more as u8)
        };
        sequences.push(format!("\x1b_G{};{}\x1b\\", control, chunk));
        first = false;
        offset = end;
    }
    sequences
}

/// The last `n` non-empty lines of `s`, joined back with newlines - used to
/// keep a LaTeX toolchain's own (often long) log output down to a short,
/// relevant-looking snippet when reporting a failure.
fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

/// Renders `latex` (a full delimited block, e.g. `\[ \boxed{...} \]`) to a
/// transparent-background PNG via a local LaTeX toolchain: `pdflatex` to
/// produce a PDF, then `pdftocairo` to rasterize its (single) page. Neither
/// is bundled with faber - both need to already be on PATH - and any
/// failure (missing tool, or `latex` not actually compiling) is returned
/// as a short, readable error rather than a raw subprocess dump.
///
/// `color` is the caller's choice, not decided here, because the same
/// block can come from either the answer or reasoning stream - each with
/// its own live text color (see `answer_text_color`/`reasoning_text_color`)
/// - and this function has no way to tell which just from `latex` itself.
pub fn render_latex_to_png(latex: &str, color: (u8, u8, u8)) -> Result<Vec<u8>, Box<dyn Error>> {
    render_document_to_png(&latex_document(latex, color))
}

/// Renders `markdown_text` - the model's whole response or reasoning,
/// Markdown and all - to a single transparent-background PNG: everything,
/// not just its isolated `\[...\]`/`\(...\)`/`$$...$$` blocks, laid out as
/// one real (if minimal) LaTeX document via `full_latex_document`. This is
/// `--display-graphics=full`'s entry point, the whole-response counterpart
/// to `render_latex_to_png`'s one-block-at-a-time rendering.
///
/// See `render_latex_to_png` for why `color` is a parameter rather than
/// decided internally.
pub fn render_full_response_to_png(
    markdown_text: &str,
    color: (u8, u8, u8),
) -> Result<Vec<u8>, Box<dyn Error>> {
    render_document_to_png(&full_latex_document(markdown_text, color))
}

/// Writes `document` (a complete `.tex` source) to a fresh scratch
/// directory and runs it through the shared pdflatex+pdftocairo pipeline
/// (`render_document_to_png_in`), cleaning the directory up afterward
/// either way.
fn render_document_to_png(document: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let dir = std::env::temp_dir().join(format!(
        "faber_latex_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir)?;
    let result = render_document_to_png_in(document, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// The ANSI palette index `Style::new().cyan()` (`main.rs`'s
/// `create_response_mode`, used for the answer text) maps to - the
/// standard SGR 36 "cyan" slot.
const ANSWER_TEXT_ANSI_INDEX: u8 = 6;

/// Used only if the terminal doesn't answer `query_ansi_color` - a
/// reasonable approximation of "standard" cyan, since ANSI colors are
/// terminal-theme-dependent and there's no universally "correct" RGB for
/// one; this is just a fallback for when the real answer is unavailable.
const ANSWER_TEXT_COLOR_FALLBACK: (u8, u8, u8) = (0, 205, 205);

/// The color to render LaTeX text in for a block/document that came from
/// the answer stream: whatever the terminal actually renders
/// `Style::new().cyan()`'s ANSI palette slot as, so the rendered math
/// matches the answer text it came from pixel-for-pixel rather than some
/// other terminal's idea of "cyan" - ANSI colors are theme-dependent, so a
/// hardcoded RGB guess can (and, per user report, does) look noticeably
/// different from one terminal theme to the next.
///
/// See `reasoning_text_color` for the reasoning stream's own, deliberately
/// different color - rendering both the same way would lose the visual
/// distinction the live text itself has between them.
pub fn answer_text_color() -> (u8, u8, u8) {
    query_ansi_color(ANSWER_TEXT_ANSI_INDEX).unwrap_or(ANSWER_TEXT_COLOR_FALLBACK)
}

/// The 256-color palette index `Style::new().color256(244).italic()`
/// (`main.rs`'s `create_response_mode`, used for reasoning text) maps to.
const REASONING_TEXT_ANSI_INDEX: u8 = 244;

/// Used only if the terminal doesn't answer `query_ansi_color` -
/// approximately what xterm's own 256-color palette renders index 244 as
/// (a mid grey, from its greyscale ramp), since that's the palette slot
/// this is standing in for.
const REASONING_TEXT_COLOR_FALLBACK: (u8, u8, u8) = (128, 128, 128);

/// `reasoning_text_color`'s counterpart to `answer_text_color`, for a
/// block/document that came from the reasoning stream instead of the
/// answer - queries the terminal's actual rendering of
/// `REASONING_TEXT_ANSI_INDEX` (256-color, not the basic 16-color palette
/// `answer_text_color` queries, but `query_ansi_color`'s OSC 4 query works
/// the same way for any index) so reasoning renders in its own distinct
/// color, matching how it already looks live, rather than blending into
/// the answer's.
pub fn reasoning_text_color() -> (u8, u8, u8) {
    query_ansi_color(REASONING_TEXT_ANSI_INDEX).unwrap_or(REASONING_TEXT_COLOR_FALLBACK)
}

/// Queries the terminal for the RGB it actually renders ANSI palette color
/// `index` as, via an OSC 4 escape sequence (`ESC ] 4 ; index ; ? BEL`,
/// answered with `ESC ] 4 ; index ; rgb:RRRR/GGGG/BBBB` terminated by BEL
/// or ST). Returns `None` if the terminal doesn't answer within a short
/// timeout (not every terminal supports this query, and it isn't worth
/// delaying a response over it) or answers with something unparseable.
fn query_ansi_color(index: u8) -> Option<(u8, u8, u8)> {
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;
    use std::time::{Duration, Instant};

    let mut terminal = terminal_trx::terminal().ok()?;
    let mut lock = terminal.lock();
    let mut tty = lock.enable_raw_mode().ok()?;

    write!(tty, "\x1b]4;{index};?\x07").ok()?;
    tty.flush().ok()?;

    let fd = tty.as_raw_fd();
    let deadline = Instant::now() + Duration::from_millis(500);
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `pfd` is a valid, exclusively-owned pollfd for the
        // duration of this call, and `poll` only reads/writes through the
        // pointer we give it.
        let ready = unsafe { libc::poll(&mut pfd, 1, remaining.as_millis() as i32) };
        if ready <= 0 {
            break; // timed out, or a poll() error - either way, give up
        }
        match tty.read(&mut byte) {
            Ok(1) => {
                let terminated = byte[0] == 0x07 || (byte[0] == b'\\' && buf.last() == Some(&0x1b));
                buf.push(byte[0]);
                if terminated {
                    break;
                }
                if buf.len() > 64 {
                    break; // sanity cap against a malformed/unexpected reply
                }
            }
            _ => break,
        }
    }

    parse_osc4_reply(&buf, index)
}

/// Parses an OSC 4 reply (`ESC ] 4 ; index ; rgb:RRRR/GGGG/BBBB`, BEL- or
/// ST-terminated) into 8-bit-per-channel RGB. Pure, so it's testable
/// without a terminal.
fn parse_osc4_reply(buf: &[u8], index: u8) -> Option<(u8, u8, u8)> {
    let prefix = format!("\x1b]4;{index};").into_bytes();
    let body = buf.strip_prefix(prefix.as_slice())?;
    let body = body
        .strip_suffix(b"\x07")
        .or_else(|| body.strip_suffix(b"\x1b\\"))
        .unwrap_or(body);
    let color = xterm_color::Color::parse(body).ok()?;
    let scale = |c: u16| ((c as u32 * 255) / (u16::MAX as u32)) as u8;
    Some((scale(color.red), scale(color.green), scale(color.blue)))
}

/// Builds the standalone LaTeX document wrapping `latex`, with its text
/// color set to `color` (via the `xcolor` package) so it reads as part of
/// the answer it's rendered from.
fn latex_document(latex: &str, color: (u8, u8, u8)) -> String {
    let (r, g, b) = color;
    format!(
        "\\documentclass[preview,border=2pt]{{standalone}}\n\
         \\usepackage{{amsmath}}\n\
         \\usepackage{{amssymb}}\n\
         \\usepackage{{xcolor}}\n\
         \\begin{{document}}\n\
         \\color[RGB]{{{r},{g},{b}}}\n\
         {latex}\n\
         \\end{{document}}\n"
    )
}

/// A Unicode codepoint in the Supplementary Private Use Area-A
/// (U+F0000-U+FFFFD, ~65,000 codepoints - never assigned any real meaning,
/// and never produced by a real response) standing in for the `index`-th
/// math block `markdown_to_latex` pulled out of the response before handing
/// the rest to the Markdown parser. Markdown treats it as an ordinary
/// character wherever it landed - mid-sentence, inside emphasis, inside a
/// list item - preserving the real paragraph/inline structure around the
/// math, rather than parsing each surrounding stretch of text as its own
/// isolated fragment (which would otherwise force a paragraph break before
/// and after every inline formula).
fn math_placeholder(index: usize) -> char {
    char::from_u32(0xF0000 + index as u32).expect("within the Supplementary Private Use Area-A")
}

/// The inverse of `math_placeholder`: if `ch` is one of these placeholders,
/// the index of the math block it stands for.
fn math_placeholder_index(ch: char) -> Option<usize> {
    let c = ch as u32;
    // .then(||...), not .then_some(...): the latter evaluates its argument
    // eagerly regardless of the guard, so "c - 0xF0000" would underflow
    // (and panic, in debug builds) for every ordinary character below
    // 0xF0000 before the range check ever gets a chance to reject it.
    (0xF0000..=0xFFFFD)
        .contains(&c)
        .then(|| (c - 0xF0000) as usize)
}

/// Escapes a single character of ordinary (non-math, non-verbatim) text for
/// literal use in LaTeX - the standard set of characters that are
/// otherwise-active in LaTeX's own syntax (`\ { } $ & % # _ ^ ~`) - plus a
/// handful of common Unicode punctuation a model's prose actually uses
/// (non-breaking hyphen, en/em dash, curly quotes, ellipsis) mapped to
/// their standard LaTeX equivalents rather than relying on `inputenc`
/// alone to cover them.
fn push_escaped_char(out: &mut String, ch: char) {
    match ch {
        '\\' => out.push_str("\\textbackslash{}"),
        '{' => out.push_str("\\{"),
        '}' => out.push_str("\\}"),
        '$' => out.push_str("\\$"),
        '&' => out.push_str("\\&"),
        '%' => out.push_str("\\%"),
        '#' => out.push_str("\\#"),
        '_' => out.push_str("\\_"),
        '^' => out.push_str("\\textasciicircum{}"),
        '~' => out.push_str("\\textasciitilde{}"),
        '\u{2011}' => out.push('-'),       // non-breaking hyphen
        '\u{2013}' => out.push_str("--"),  // en dash
        '\u{2014}' => out.push_str("---"), // em dash
        '\u{2018}' | '\u{2019}' => out.push('\''),
        '\u{201c}' | '\u{201d}' => out.push('"'),
        '\u{2026}' => out.push_str("\\ldots{}"),
        _ => out.push(ch),
    }
}

/// Appends `text` to `out`, escaping every character for literal LaTeX use
/// (`push_escaped_char`) *except* a `math_placeholder`, which is replaced
/// with the real, verbatim math source it stands for (via `math_blocks`,
/// indexed the same way `math_placeholder` numbered them) - already valid
/// LaTeX, so it must not be escaped or otherwise altered.
fn push_escaped_with_math(out: &mut String, text: &str, math_blocks: &[&str]) {
    for ch in text.chars() {
        match math_placeholder_index(ch).and_then(|i| math_blocks.get(i)) {
            Some(math) => out.push_str(math),
            None => push_escaped_char(out, ch),
        }
    }
}

/// Like `push_escaped_with_math`, but for verbatim (code block) text: a
/// `math_placeholder` is still swapped back for its real source, but every
/// other character is pushed as-is, unescaped - LaTeX's `verbatim`
/// environment doesn't interpret its content as LaTeX at all, so escaping
/// it would show the literal escape sequences instead of the original
/// text.
fn push_verbatim_with_math(out: &mut String, text: &str, math_blocks: &[&str]) {
    for ch in text.chars() {
        match math_placeholder_index(ch).and_then(|i| math_blocks.get(i)) {
            Some(math) => out.push_str(math),
            None => out.push(ch),
        }
    }
}

/// Maps a table column's Markdown alignment to the LaTeX `tabular` column
/// specifier - `Alignment::None` (no `:---`/`---:` markers given) defaults
/// to left-aligned, the same as an ordinary paragraph.
fn tabular_column_spec(alignment: Alignment) -> char {
    match alignment {
        Alignment::Left | Alignment::None => 'l',
        Alignment::Center => 'c',
        Alignment::Right => 'r',
    }
}

/// Converts `text` - Markdown, as models actually write it, with faber's
/// own `\[...\]`/`\(...\)`/`$$...$$` math blocks mixed in - to a LaTeX
/// document body. Math blocks are pulled out and stand in as
/// `math_placeholder`s before Markdown parsing, then spliced back in
/// verbatim once it's done (see that function's docs for why: it keeps the
/// surrounding prose's real paragraph/inline structure intact instead of
/// parsing each fragment of text around a formula in isolation). Anything
/// else Markdown - headings, emphasis, lists, block quotes, tables, code
/// spans/blocks, links, images, horizontal rules, hard line breaks - is
/// translated to the closest plain-LaTeX equivalent (`article`-class
/// commands and environments, since that's what `full_latex_document`
/// bases its document on); literal text is escaped so it can't break
/// compilation or be misread as more LaTeX.
///
/// Deliberately not a complete Markdown-to-LaTeX renderer: links/images
/// render as their text/alt-text alone (a static image can't be clickable,
/// and downloading a linked image is out of scope), strikethrough needs
/// the `ulem` package (pulled in by `full_latex_document`), and tables get
/// a plain `tabular` with every column the same alignment style Markdown
/// gave it, no fancier styling.
pub fn markdown_to_latex(text: &str) -> String {
    let segments = split_latex_segments(text);
    let mut placeholder_text = String::with_capacity(text.len());
    let mut math_blocks: Vec<&str> = Vec::new();
    for segment in &segments {
        match segment {
            StreamSegment::Text(t) => placeholder_text.push_str(t),
            StreamSegment::Latex(block) => {
                placeholder_text.push(math_placeholder(math_blocks.len()));
                math_blocks.push(block.as_str());
            }
        }
    }

    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);

    let mut out = String::new();
    let mut table_column = 0usize;
    let mut in_code_block = false;
    for event in Parser::new_ext(&placeholder_text, options) {
        match event {
            Event::Start(tag) => match tag {
                Tag::Heading { level, .. } => out.push_str(match level {
                    HeadingLevel::H1 => "\\section*{",
                    HeadingLevel::H2 => "\\subsection*{",
                    _ => "\\subsubsection*{",
                }),
                Tag::BlockQuote(_) => out.push_str("\\begin{quote}\n"),
                Tag::CodeBlock(_) => {
                    in_code_block = true;
                    out.push_str("\\begin{verbatim}\n");
                }
                Tag::List(Some(_)) => out.push_str("\\begin{enumerate}\n"),
                Tag::List(None) => out.push_str("\\begin{itemize}\n"),
                Tag::Item => out.push_str("\\item "),
                Tag::Emphasis => out.push_str("\\textit{"),
                Tag::Strong => out.push_str("\\textbf{"),
                Tag::Strikethrough => out.push_str("\\sout{"),
                Tag::Table(alignments) => {
                    let spec: String = alignments.iter().map(|a| tabular_column_spec(*a)).collect();
                    out.push_str(&format!("\\begin{{tabular}}{{{spec}}}\n"));
                }
                Tag::TableCell => {
                    if table_column > 0 {
                        out.push_str(" & ");
                    }
                    table_column += 1;
                }
                Tag::Image { .. } => out.push_str("[image: "),
                Tag::Paragraph
                | Tag::TableHead
                | Tag::TableRow
                | Tag::Link { .. }
                | Tag::FootnoteDefinition(_)
                | Tag::HtmlBlock
                | Tag::MetadataBlock(_)
                | Tag::DefinitionList
                | Tag::DefinitionListTitle
                | Tag::DefinitionListDefinition
                | Tag::Superscript
                | Tag::Subscript => {}
            },
            Event::End(tag_end) => match tag_end {
                TagEnd::Paragraph => out.push_str("\n\n"),
                TagEnd::Heading(_) => out.push_str("}\n\n"),
                TagEnd::BlockQuote(_) => out.push_str("\\end{quote}\n\n"),
                TagEnd::CodeBlock => {
                    in_code_block = false;
                    out.push_str("\\end{verbatim}\n\n");
                }
                TagEnd::List(true) => out.push_str("\\end{enumerate}\n\n"),
                TagEnd::List(false) => out.push_str("\\end{itemize}\n\n"),
                TagEnd::Item => out.push('\n'),
                TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => out.push('}'),
                TagEnd::Image => out.push(']'),
                TagEnd::Table => out.push_str("\\end{tabular}\n\n"),
                TagEnd::TableHead => {
                    out.push_str("\\\\\n\\hline\n");
                    table_column = 0;
                }
                TagEnd::TableRow => {
                    out.push_str("\\\\\n");
                    table_column = 0;
                }
                TagEnd::TableCell
                | TagEnd::Link
                | TagEnd::FootnoteDefinition
                | TagEnd::HtmlBlock
                | TagEnd::MetadataBlock(_)
                | TagEnd::DefinitionList
                | TagEnd::DefinitionListTitle
                | TagEnd::DefinitionListDefinition
                | TagEnd::Superscript
                | TagEnd::Subscript => {}
            },
            Event::Text(t) => {
                if in_code_block {
                    // Verbatim: pushed as-is, no LaTeX escaping - but a
                    // math_placeholder is still swapped back for its
                    // original raw source (e.g. a model showing "\(x^2\)"
                    // as a literal syntax example inside a code fence),
                    // since that source text *is* the correct literal
                    // content here too, same as any other code-block text.
                    push_verbatim_with_math(&mut out, &t, &math_blocks);
                } else {
                    push_escaped_with_math(&mut out, &t, &math_blocks);
                }
            }
            Event::Code(t) => {
                out.push_str("\\texttt{");
                push_escaped_with_math(&mut out, &t, &math_blocks);
                out.push('}');
            }
            Event::SoftBreak => out.push(' '),
            Event::HardBreak => out.push_str("\\\\\n"),
            Event::Rule => out.push_str("\\par\\noindent\\hrulefill\\par\n\n"),
            Event::TaskListMarker(checked) => {
                out.push_str(if checked {
                    "$\\boxtimes$ "
                } else {
                    "$\\square$ "
                });
            }
            // Markdown's own "$...$"/"$$...$$" math syntax - never produced
            // here since ENABLE_MATH isn't turned on above (our own
            // \[...\]/\(...\)/$$...$$ blocks are already pulled out into
            // math_placeholders before this parser ever sees the text), but
            // matched anyway (as literal, escaped text) so this stays
            // exhaustive against future pulldown-cmark Event variants.
            Event::InlineMath(t) | Event::DisplayMath(t) => {
                push_escaped_with_math(&mut out, &t, &math_blocks)
            }
            Event::Html(_) | Event::InlineHtml(_) | Event::FootnoteReference(_) => {}
        }
    }
    out
}

/// The maximum width `full_latex_document` wraps its text to, in points
/// (≈6in / 432pt at 72pt/in - a comfortable, book-like reading width,
/// independent of the terminal's own current column count, which doesn't
/// otherwise correspond to any particular pixel width for a Kitty-displayed
/// image anyway).
const FULL_DOCUMENT_MAX_WIDTH_PT: u32 = 432;

/// Builds a complete standalone LaTeX document for `render_full_response_to_png`
/// (`--display-graphics=full`): the whole response, converted from Markdown
/// to LaTeX via `markdown_to_latex`, laid out as a real (if visually
/// minimal) document - proper paragraph flow, headings, lists, tables -
/// rather than one image per isolated math block. Based on the `article`
/// class (via `standalone`'s `class=article` option) so headings, lists
/// and quotes are actually available, but still cropped to its own content
/// like every other image this module renders, via `standalone`'s
/// `preview` mode; `varwidth` caps the width so long paragraphs actually
/// wrap instead of forming one endless line.
fn full_latex_document(markdown_text: &str, color: (u8, u8, u8)) -> String {
    let (r, g, b) = color;
    let body = markdown_to_latex(markdown_text);
    format!(
        "\\documentclass[preview,border=8pt,class=article,varwidth={width}pt]{{standalone}}\n\
         \\usepackage[utf8]{{inputenc}}\n\
         \\usepackage{{amsmath}}\n\
         \\usepackage{{amssymb}}\n\
         \\usepackage{{xcolor}}\n\
         \\usepackage[normalem]{{ulem}}\n\
         \\setlength{{\\parindent}}{{0pt}}\n\
         \\setlength{{\\parskip}}{{6pt}}\n\
         \\begin{{document}}\n\
         \\color[RGB]{{{r},{g},{b}}}\n\
         {body}\n\
         \\end{{document}}\n",
        width = FULL_DOCUMENT_MAX_WIDTH_PT,
    )
}

/// Finds `program` on `$PATH` (the *current*, unsandboxed process's PATH),
/// so callers can hand bwrap a resolved absolute path rather than a bare
/// name to look up itself - the same way `run_command`'s own schema
/// requires an absolute path from the model, for a consistent, predictable
/// executable regardless of what PATH looks like inside a given sandbox.
fn resolve_on_path(program: &str) -> Option<std::path::PathBuf> {
    let dirs = std::env::var_os("PATH")?;
    resolve_in_path_dirs(program, &dirs)
}

/// The actual search, taking the PATH value as a parameter rather than
/// reading the real environment variable, so it's testable without
/// mutating process-wide state (risky - Rust tests run in parallel within
/// one process).
fn resolve_in_path_dirs(program: &str, path_dirs: &std::ffi::OsStr) -> Option<std::path::PathBuf> {
    std::env::split_paths(path_dirs)
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file())
}

/// Builds the bwrap argument list for the LaTeX toolchain's sandbox: the
/// same isolation (`bwrap_isolation_flags`, in `main.rs`) as `run_command`'s
/// own sandbox, but a different filesystem strategy - the whole real root
/// read-only (`--ro-bind / /`) rather than a fresh empty root with a
/// curated list of read-only binds, with `cwd` then re-bound read-write on
/// top of it.
///
/// Two differences from `run_command`'s sandbox (`bwrap_args`, in
/// `main.rs`), both there to let pdflatex/pdftocairo actually find their
/// own files - the sandbox's other restrictions (no network, no
/// capabilities, no write access outside `cwd`) all still apply:
///
/// - The whole real filesystem is exposed read-only, instead of a curated
///   subset. Where LaTeX's own file-finding library (kpathsea) looks for
///   its files - most prominently pdflatex's own precompiled format file -
///   varies by distro and install (some under `/usr`, some under a
///   system-wide var tree like `/var/lib/texmf`, some per-user under
///   `$HOME`), with no reliable way to enumerate them all upfront; several
///   attempts at guessing the right subset of paths to bind (by kpathsea
///   variable name, then by locating the format file directly) all failed
///   against a real install. Exposing the whole (real, unwritable)
///   filesystem sidesteps that guessing game entirely. This is an
///   acceptable tradeoff specifically here because pdflatex/pdftocairo are
///   two fixed, known binaries being fed data, not an arbitrary command
///   chosen by the model - unlike `run_command`, which keeps the tighter,
///   curated sandbox since it has no such guarantee about what it's
///   running.
/// - No `--clearenv`: kpathsea also relies on `$HOME` (and variables
///   commonly derived from it) to locate its files, and a wiped
///   environment made that lookup fail outright even for files that ARE
///   now readable under the read-only root. Passing the real environment
///   through costs nothing here - there's no untrusted code running that
///   could read it back out, just the LaTeX toolchain locating its own
///   files.
fn whole_root_ro_bwrap_args(cwd: &str, command: &str, args: Option<&[String]>) -> Vec<String> {
    let mut a = crate::bwrap_isolation_flags(false);
    a.push("--ro-bind".to_string());
    a.push("/".to_string());
    a.push("/".to_string());
    a.push("--bind".to_string());
    a.push(cwd.to_string());
    a.push(cwd.to_string());
    a.extend(crate::bwrap_command_tail(command, args));
    a
}

/// Runs `program` (resolved via `resolve_on_path`) with `args`, sandboxed
/// with bwrap via `whole_root_ro_bwrap_args`: no network, no capabilities,
/// and only `dir` writable. pdflatex/pdftocairo process the model's own
/// LaTeX text, so - like any other untrusted input a tool might feed to a
/// subprocess - they get sandboxed, not unrestricted host access.
///
/// `not_found_hint` is appended only to a "not found on PATH" error (e.g.
/// "is a LaTeX distribution installed?") - a separate bwrap-spawn failure
/// already has its own, different hint ("is bubblewrap installed?"), so
/// this isn't tacked onto that one too.
fn run_sandboxed(
    program: &str,
    args: &[String],
    dir: &Path,
    not_found_hint: &str,
) -> Result<std::process::Output, Box<dyn Error>> {
    let resolved = resolve_on_path(program)
        .ok_or_else(|| format!("{} not found on PATH - {}", program, not_found_hint))?;
    let resolved = resolved
        .to_str()
        .ok_or("resolved program path is not valid UTF-8")?;
    let dir_str = dir
        .to_str()
        .ok_or("temp directory path is not valid UTF-8")?;

    let bwrap_flags = whole_root_ro_bwrap_args(dir_str, resolved, Some(args));

    let mut cmd = Command::new("bwrap");
    cmd.args(bwrap_flags);
    cmd.output().map_err(|e| {
        format!(
            "couldn't run the sandbox (bwrap): {} - is bubblewrap installed?",
            e
        )
        .into()
    })
}

/// Writes `document` (a complete `.tex` source) into `dir` and runs it
/// through pdflatex + pdftocairo, sandboxed, returning the resulting PNG.
/// Shared by every caller of `render_document_to_png` - `document`'s own
/// content is the only thing that differs between rendering one math block
/// (`latex_document`) and a whole response (`full_latex_document`).
fn render_document_to_png_in(document: &str, dir: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let tex_path = dir.join("input.tex");
    std::fs::write(&tex_path, document)?;

    let dir_str = dir
        .to_str()
        .ok_or("temp directory path is not valid UTF-8")?;
    let tex_path_str = tex_path
        .to_str()
        .ok_or("temp file path is not valid UTF-8")?;
    let output = run_sandboxed(
        "pdflatex",
        &[
            "-interaction=nonstopmode".to_string(),
            "-halt-on-error".to_string(),
            "-output-directory".to_string(),
            dir_str.to_string(),
            tex_path_str.to_string(),
        ],
        dir,
        "is a LaTeX distribution installed?",
    )?;
    if !output.status.success() {
        return Err(format!(
            "pdflatex failed to compile the LaTeX block:\n{}",
            tail_lines(&String::from_utf8_lossy(&output.stdout), 15)
        )
        .into());
    }

    let pdf_path = dir.join("input.pdf");
    let pdf_path_str = pdf_path.to_str().ok_or("PDF path is not valid UTF-8")?;
    let png_prefix = dir.join("page");
    let png_prefix_str = png_prefix
        .to_str()
        .ok_or("PNG prefix path is not valid UTF-8")?;
    // pdftocairo, not pdftoppm: pdftoppm rasterizes via poppler's Splash
    // backend into PPM, which has no alpha channel at all (its -png mode
    // is just PPM-then-converted, still opaque) - pdftocairo uses Cairo,
    // which does real ARGB rendering, so -transp (transparent background
    // instead of opaque white, so the math blends into the terminal
    // instead of showing up as a white box) actually does something.
    // -singlefile: write exactly "page.png", no "-1" page-number suffix,
    // since our document only ever has the one page.
    let output = run_sandboxed(
        "pdftocairo",
        &[
            "-png".to_string(),
            "-transp".to_string(),
            "-singlefile".to_string(),
            "-r".to_string(),
            "150".to_string(),
            pdf_path_str.to_string(),
            png_prefix_str.to_string(),
        ],
        dir,
        "is poppler-utils installed?",
    )?;
    if !output.status.success() {
        return Err(format!(
            "pdftocairo failed to rasterize the rendered PDF:\n{}",
            tail_lines(&String::from_utf8_lossy(&output.stderr), 15)
        )
        .into());
    }

    let png_path = dir.join("page.png");
    if !png_path.exists() {
        return Err("pdftocairo didn't produce a PNG file".into());
    }

    Ok(std::fs::read(png_path)?)
}

/// Writes the escape sequences to display `png` inline via Kitty's
/// graphics protocol, going straight to the terminal (bypassing the normal
/// line-buffered chat output, same as the status bar's own spinner) since
/// these are raw control sequences, not printable text.
pub fn display_png(png: &[u8]) {
    let sequences = kitty_image_escape_sequences(png);
    if sequences.is_empty() {
        return;
    }
    crate::status_bar::write_stderr(&sequences.concat());
    crate::status_bar::write_stderr("\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whole_root_ro_bwrap_args_binds_root_ro_and_cwd_rw() {
        let flags = whole_root_ro_bwrap_args("/tmp/scratch", "/usr/bin/pdflatex", None);
        assert!(!flags.contains(&"--clearenv".to_string()));

        let ro_pos = flags
            .windows(3)
            .position(|w| w == ["--ro-bind", "/", "/"])
            .expect("expected --ro-bind / /");
        let bind_pos = flags
            .windows(3)
            .position(|w| w == ["--bind", "/tmp/scratch", "/tmp/scratch"])
            .expect("expected --bind /tmp/scratch /tmp/scratch");
        // cwd must be re-bound read-write *after* the whole-root read-only
        // bind, so it actually ends up writable rather than shadowed back
        // to read-only by a later mount at the same path.
        assert!(bind_pos > ro_pos);

        assert_eq!(flags.last(), Some(&"/usr/bin/pdflatex".to_string()));
    }

    #[test]
    fn test_whole_root_ro_bwrap_args_with_args_appends_them_after_the_command() {
        let flags = whole_root_ro_bwrap_args(
            "/tmp/scratch",
            "/usr/bin/pdftocairo",
            Some(&["-png".to_string(), "in.pdf".to_string()]),
        );
        let tail = &flags[flags.len() - 3..];
        assert_eq!(
            tail,
            &[
                "/usr/bin/pdftocairo".to_string(),
                "-png".to_string(),
                "in.pdf".to_string()
            ]
        );
    }

    #[test]
    fn test_resolve_in_path_dirs_finds_existing_executable() {
        let dir = std::env::temp_dir().join(format!("faber_test_resolve_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("myprogram");
        std::fs::write(&exe, "").unwrap();

        let path_var = std::env::join_paths([dir.as_path()]).unwrap();
        let found = resolve_in_path_dirs("myprogram", &path_var);

        assert_eq!(found, Some(exe));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_resolve_in_path_dirs_not_found_returns_none() {
        let dir =
            std::env::temp_dir().join(format!("faber_test_resolve_missing_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path_var = std::env::join_paths([dir.as_path()]).unwrap();

        assert_eq!(
            resolve_in_path_dirs("does_not_exist_anywhere", &path_var),
            None
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_resolve_in_path_dirs_searches_multiple_dirs_in_order() {
        let base =
            std::env::temp_dir().join(format!("faber_test_resolve_multi_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        let exe_b = dir_b.join("only_in_b");
        std::fs::write(&exe_b, "").unwrap();

        let path_var = std::env::join_paths([dir_a.as_path(), dir_b.as_path()]).unwrap();
        assert_eq!(resolve_in_path_dirs("only_in_b", &path_var), Some(exe_b));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_is_kitty_from_env_term_contains_kitty() {
        assert!(is_kitty_from_env(Some("xterm-kitty"), None));
    }

    #[test]
    fn test_is_kitty_from_env_window_id_set() {
        assert!(is_kitty_from_env(Some("xterm-256color"), Some("1")));
    }

    #[test]
    fn test_is_kitty_from_env_neither() {
        assert!(!is_kitty_from_env(Some("xterm-256color"), None));
        assert!(!is_kitty_from_env(None, None));
    }

    fn segments_latex(events: &[StreamSegment]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                StreamSegment::Latex(t) => Some(t.as_str()),
                StreamSegment::Text(_) => None,
            })
            .collect()
    }

    /// Reassembles every segment's own text, in order - should always equal
    /// the original input for well-formed text (nothing is ever dropped,
    /// only reclassified as Text vs Latex).
    fn segments_concat(events: &[StreamSegment]) -> String {
        events
            .iter()
            .map(|e| match e {
                StreamSegment::Text(t) => t.as_str(),
                StreamSegment::Latex(t) => t.as_str(),
            })
            .collect()
    }

    #[test]
    fn test_split_latex_segments_plain_text_is_one_text_segment() {
        let events = split_latex_segments("just plain text, no math here");
        assert_eq!(
            events,
            vec![StreamSegment::Text(
                "just plain text, no math here".to_string()
            )]
        );
    }

    #[test]
    fn test_split_latex_segments_single_block() {
        let events = split_latex_segments("before \\[ x \\] after");
        assert_eq!(
            events,
            vec![
                StreamSegment::Text("before ".to_string()),
                StreamSegment::Latex("\\[ x \\]".to_string()),
                StreamSegment::Text(" after".to_string()),
            ]
        );
    }

    #[test]
    fn test_split_latex_segments_bracket_dollar_and_paren_delimiters() {
        let text = "The answer is \\[ \\boxed{\\frac{N(N+1)}{2}} \\] and $$ x^2 $$ and \\( y \\).";
        let events = split_latex_segments(text);
        assert_eq!(
            segments_latex(&events),
            vec![
                "\\[ \\boxed{\\frac{N(N+1)}{2}} \\]",
                "$$ x^2 $$",
                "\\( y \\)"
            ]
        );
        assert_eq!(segments_concat(&events), text);
    }

    #[test]
    fn test_split_latex_segments_no_math_is_one_text_segment() {
        let events = split_latex_segments("just plain text, no math here");
        assert_eq!(segments_latex(&events), Vec::<&str>::new());
        assert_eq!(segments_concat(&events), "just plain text, no math here");
    }

    #[test]
    fn test_split_latex_segments_unterminated_delimiter_is_skipped_not_fatal() {
        // A lone, never-closed delimiter must only cause *that one*
        // occurrence to be skipped, not abandon the rest of the scan -
        // otherwise a single stray "\(" (or "$") anywhere earlier in a long
        // response would silently drop every well-formed block after it,
        // with no error since this isn't treated as a failure. Its own
        // characters still end up as ordinary text, not lost.
        let text = "stray \\( never closed here. later: \\[ \\boxed{x} \\]";
        let events = split_latex_segments(text);
        assert_eq!(segments_latex(&events), vec!["\\[ \\boxed{x} \\]"]);
        assert_eq!(segments_concat(&events), text);
    }

    #[test]
    fn test_split_latex_segments_multiple_unterminated_delimiters_dont_compound() {
        let text = "\\( one. \\( two. valid: \\[ z \\]";
        let events = split_latex_segments(text);
        assert_eq!(segments_latex(&events), vec!["\\[ z \\]"]);
        assert_eq!(segments_concat(&events), text);
    }

    #[test]
    fn test_split_latex_segments_unterminated_block_flushed_as_text_at_finish() {
        let mut splitter = LatexSplitter::new();
        // The plain text before the open delimiter is safe to emit right
        // away - only the still-unresolved "\[ never closed" part is held.
        let mut events = splitter.push("leading \\[ never closed");
        assert_eq!(events, vec![StreamSegment::Text("leading ".to_string())]);
        events.extend(splitter.finish());
        assert_eq!(segments_latex(&events), Vec::<&str>::new());
        assert_eq!(segments_concat(&events), "leading \\[ never closed");
    }

    #[test]
    fn test_split_latex_segments_pending_dollar_and_backslash_resolve_later() {
        // A lone trailing "$" or "\" must not be emitted until it's clear
        // whether it's about to become a real delimiter.
        let mut splitter = LatexSplitter::new();
        assert_eq!(
            splitter.push("price is $"),
            vec![StreamSegment::Text("price is ".to_string())]
        );
        // Followed by a digit, not another "$" - it was never "$$" after all.
        assert_eq!(
            splitter.push("5 today"),
            vec![StreamSegment::Text("$5 today".to_string())]
        );

        let mut splitter2 = LatexSplitter::new();
        assert_eq!(
            splitter2.push("note the \\"),
            vec![StreamSegment::Text("note the ".to_string())]
        );
        // Followed by neither "[" nor "(" - just an ordinary backslash.
        assert_eq!(
            splitter2.push("nother thing"),
            vec![StreamSegment::Text("\\nother thing".to_string())]
        );
    }

    #[test]
    fn test_split_latex_segments_pending_dollar_resolves_into_a_real_block() {
        let mut splitter = LatexSplitter::new();
        assert_eq!(
            splitter.push("see: $"),
            vec![StreamSegment::Text("see: ".to_string())]
        );
        assert_eq!(
            splitter.push("$ x^2 $$ done"),
            vec![
                StreamSegment::Latex("$$ x^2 $$".to_string()),
                StreamSegment::Text(" done".to_string()),
            ]
        );
    }

    #[test]
    fn test_split_latex_segments_byte_by_byte_matches_one_shot() {
        let text = "The answer is \\[ \\boxed{\\frac{N(N+1)}{2}} \\] and $$ x^2 $$ and \\( y \\). Also stray \\( unmatched, then more text.";
        let one_shot = split_latex_segments(text);

        let mut splitter = LatexSplitter::new();
        let mut streamed = Vec::new();
        for ch in text.chars() {
            streamed.extend(splitter.push(&ch.to_string()));
        }
        streamed.extend(splitter.finish());

        // While the stream is still active, an unmatched delimiter (like
        // the stray "\(" near the end here) just makes push() buffer and
        // wait - it can't look ahead the way a whole-text scan can. But
        // finish() resolves whatever's left over via split_latex_segments
        // itself, so once the stream actually ends, byte-by-byte streaming
        // recovers exactly as well as feeding the whole text in one shot -
        // nothing is ever lost, and the same blocks are found. The exact
        // Text segment *boundaries* legitimately differ, though (streaming
        // one character at a time naturally emits many tiny Text segments
        // instead of one-shot's few large ones), so only the found Latex
        // blocks and the fully reassembled text are compared, not the
        // segment list itself.
        assert_eq!(segments_concat(&streamed), text);
        assert_eq!(segments_latex(&streamed), segments_latex(&one_shot));
    }

    #[test]
    fn test_split_latex_segments_dollar_then_end_of_stream_is_flushed_literally() {
        let mut splitter = LatexSplitter::new();
        splitter.push("trailing dollar $");
        let events = splitter.finish();
        assert_eq!(events, vec![StreamSegment::Text("$".to_string())]);
    }

    #[test]
    fn test_split_latex_segments_empty_input_is_empty() {
        assert!(split_latex_segments("").is_empty());
        let mut splitter = LatexSplitter::new();
        assert!(splitter.push("").is_empty());
        assert!(splitter.finish().is_empty());
    }

    #[test]
    fn test_split_latex_segments_block_split_across_many_small_chunks() {
        let mut splitter = LatexSplitter::new();
        let mut events = Vec::new();
        for piece in ["before ", "\\", "[", " x ", "\\", "] after"] {
            events.extend(splitter.push(piece));
        }
        events.extend(splitter.finish());
        assert_eq!(
            events,
            vec![
                StreamSegment::Text("before ".to_string()),
                StreamSegment::Latex("\\[ x \\]".to_string()),
                StreamSegment::Text(" after".to_string()),
            ]
        );
    }

    #[test]
    fn test_split_latex_segments_real_multi_block_response() {
        // A real step-by-step derivation with many inline \(...\) blocks
        // interleaved with display \[...\] ones, nested braces
        // (\frac{N}{2}), and non-ASCII prose (non-breaking hyphens) outside
        // the math - regression coverage for a real response where this
        // was suspected (wrongly - see git history) to be under-extracting.
        let text = "**Step\u{2011}by\u{2011}step reasoning**\n\n1. Let \\(S_N\\) be the sum of the first \\(N\\) positive integers:\n   \\[\n   S_N = 1+2+3+\\cdots+N.\n   \\]\n\n2. This is an arithmetic series with:\n   - first term \\(a_1 = 1\\),\n   - common difference \\(d = 1\\),\n   - number of terms \\(n = N\\),\n   - last term \\(a_N = N\\).\n\n3. For an arithmetic series, the sum can be found by pairing the first and last terms, the second and the second\u{2011}last terms, and so on:\n   \\[\n   S_N = \\frac{N}{2}\\,(a_1 + a_N).\n   \\]\n   This formula holds for both even and odd \\(N\\) and can be verified by direct pairing.\n\n4. Substitute \\(a_1 = 1\\) and \\(a_N = N\\):\n   \\[\n   S_N = \\frac{N}{2}\\,(1+N) = \\frac{N(N+1)}{2}.\n   \\]\n\n5. **Check by induction (optional):**\n   - Base case \\(N=1\\): \\(S_1 = 1 = \\frac{1\\cdot2}{2}\\).\n   - Inductive step: assuming \\(S_N = \\frac{N(N+1)}{2}\\), then\n     \\[\n     S_{N+1}=S_N+(N+1)=\\frac{N(N+1)}{2}+(N+1)=\\frac{(N+1)(N+2)}{2},\n     \\]\n     which matches the formula for \\(N+1\\). Thus the formula holds for all positive integers \\(N\\).\n\n\\[\n\\boxed{\\frac{N(N+1)}{2}}\n\\]\n";
        let events = split_latex_segments(text);
        let blocks = segments_latex(&events);
        assert_eq!(blocks.len(), 19, "found: {blocks:#?}");
        assert_eq!(blocks[0], "\\(S_N\\)");
        assert_eq!(blocks[2], "\\[\n   S_N = 1+2+3+\\cdots+N.\n   \\]");
        assert_eq!(blocks[18], "\\[\n\\boxed{\\frac{N(N+1)}{2}}\n\\]");
        assert_eq!(segments_concat(&events), text);
    }

    #[test]
    fn test_kitty_image_escape_sequences_empty_png_is_empty() {
        assert!(kitty_image_escape_sequences(&[]).is_empty());
    }

    #[test]
    fn test_kitty_image_escape_sequences_small_png_is_one_chunk() {
        let png = b"not a real png but small";
        let sequences = kitty_image_escape_sequences(png);
        assert_eq!(sequences.len(), 1);
        assert!(sequences[0].starts_with("\x1b_Ga=T,f=100,m=0;"));
        assert!(sequences[0].ends_with("\x1b\\"));

        let encoded = base64::engine::general_purpose::STANDARD.encode(png);
        assert!(sequences[0].contains(&encoded));
    }

    #[test]
    fn test_kitty_image_escape_sequences_large_png_is_chunked_with_correct_more_flags() {
        // Big enough that its base64 form spans more than one 4096-byte chunk.
        let png = vec![0u8; 10_000];
        let sequences = kitty_image_escape_sequences(&png);
        assert!(sequences.len() > 1, "expected multiple chunks, got 1");

        // Every chunk but the last says "more data is coming" (m=1); the
        // last says "this is the end" (m=0). Only the first chunk carries
        // the full control data (a=T,f=100,...).
        for (i, seq) in sequences.iter().enumerate() {
            let is_last = i == sequences.len() - 1;
            if i == 0 {
                assert!(seq.starts_with("\x1b_Ga=T,f=100,m="));
            } else {
                assert!(seq.starts_with("\x1b_Gm="));
            }
            assert_eq!(seq.contains("m=0;"), is_last, "chunk {i}: {seq:?}");
            assert!(seq.ends_with("\x1b\\"));
        }

        // Reassembling every chunk's payload must round-trip to the original.
        let reassembled: String = sequences
            .iter()
            .map(|seq| {
                let payload_start = seq.find(';').unwrap() + 1;
                &seq[payload_start..seq.len() - 2] // strip trailing ESC \
            })
            .collect();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(reassembled)
            .unwrap();
        assert_eq!(decoded, png);
    }

    #[test]
    fn test_tail_lines_keeps_only_the_last_n_nonempty_lines() {
        let text = "one\ntwo\n\nthree\nfour\nfive";
        assert_eq!(tail_lines(text, 2), "four\nfive");
        assert_eq!(tail_lines(text, 100), "one\ntwo\nthree\nfour\nfive");
    }

    #[test]
    fn test_latex_document_sets_the_given_color_before_the_content() {
        let doc = latex_document("\\[ x \\]", (255, 0, 128));
        assert!(doc.contains("\\usepackage{xcolor}"));
        let color_pos = doc.find("\\color[RGB]{255,0,128}").unwrap();
        let content_pos = doc.find("\\[ x \\]").unwrap();
        assert!(
            color_pos < content_pos,
            "color command must come before the math content"
        );
    }

    #[test]
    fn test_math_placeholder_roundtrips_through_its_index() {
        for i in [0, 1, 41, 6400, 65533] {
            assert_eq!(math_placeholder_index(math_placeholder(i)), Some(i));
        }
    }

    #[test]
    fn test_math_placeholder_index_rejects_ordinary_characters() {
        for ch in ['a', ' ', '\\', '$', '{', '\u{FFFF}'] {
            assert_eq!(math_placeholder_index(ch), None);
        }
    }

    #[test]
    fn test_push_escaped_char_escapes_every_latex_special_character() {
        let mut out = String::new();
        for ch in "\\{}$&%#_^~".chars() {
            push_escaped_char(&mut out, ch);
        }
        assert_eq!(
            out,
            "\\textbackslash{}\\{\\}\\$\\&\\%\\#\\_\\textasciicircum{}\\textasciitilde{}"
        );
    }

    #[test]
    fn test_push_escaped_char_maps_common_unicode_punctuation() {
        let mut out = String::new();
        for ch in ['\u{2011}', '\u{2013}', '\u{2014}', '\u{2026}'] {
            push_escaped_char(&mut out, ch);
        }
        // "-" (non-breaking hyphen) + "--" (en dash) + "---" (em dash) = 6 dashes.
        assert_eq!(out, "------\\ldots{}");
    }

    #[test]
    fn test_push_escaped_char_passes_ordinary_characters_through() {
        let mut out = String::new();
        for ch in "hello world 123".chars() {
            push_escaped_char(&mut out, ch);
        }
        assert_eq!(out, "hello world 123");
    }

    #[test]
    fn test_push_escaped_with_math_substitutes_placeholder_verbatim() {
        let math = ["\\(x\\)", "\\[y\\]"];
        let text = format!("a {} b {} c_1", math_placeholder(0), math_placeholder(1));
        let mut out = String::new();
        push_escaped_with_math(&mut out, &text, &math);
        assert_eq!(out, "a \\(x\\) b \\[y\\] c\\_1");
    }

    #[test]
    fn test_tabular_column_spec_maps_alignment() {
        assert_eq!(tabular_column_spec(Alignment::None), 'l');
        assert_eq!(tabular_column_spec(Alignment::Left), 'l');
        assert_eq!(tabular_column_spec(Alignment::Center), 'c');
        assert_eq!(tabular_column_spec(Alignment::Right), 'r');
    }

    #[test]
    fn test_markdown_to_latex_escapes_plain_text() {
        let out = markdown_to_latex("100% of $5 & change_1 #tag ~ok");
        assert_eq!(
            out.trim(),
            "100\\% of \\$5 \\& change\\_1 \\#tag \\textasciitilde{}ok"
        );
    }

    #[test]
    fn test_markdown_to_latex_bold_and_italic() {
        let out = markdown_to_latex("**bold** and *italic* and _also italic_");
        assert!(out.contains("\\textbf{bold}"));
        assert!(out.contains("\\textit{italic}"));
        assert!(out.contains("\\textit{also italic}"));
    }

    #[test]
    fn test_markdown_to_latex_inline_code_is_texttt_and_escaped() {
        let out = markdown_to_latex("call `foo_bar()` now");
        assert!(out.contains("\\texttt{foo\\_bar()}"), "got: {out:?}");
    }

    #[test]
    fn test_markdown_to_latex_headers() {
        let out = markdown_to_latex("# Title\n\n## Subtitle\n\n### Sub-subtitle\n");
        assert!(out.contains("\\section*{Title}"));
        assert!(out.contains("\\subsection*{Subtitle}"));
        assert!(out.contains("\\subsubsection*{Sub-subtitle}"));
    }

    #[test]
    fn test_markdown_to_latex_bullet_list() {
        let out = markdown_to_latex("- one\n- two\n- three\n");
        assert!(out.contains("\\begin{itemize}"));
        assert!(out.contains("\\end{itemize}"));
        assert!(out.contains("\\item one"));
        assert!(out.contains("\\item two"));
        assert!(out.contains("\\item three"));
    }

    #[test]
    fn test_markdown_to_latex_numbered_list() {
        let out = markdown_to_latex("1. first\n2. second\n3. third\n");
        assert!(out.contains("\\begin{enumerate}"));
        assert!(out.contains("\\end{enumerate}"));
        assert!(out.contains("\\item first"));
        assert!(out.contains("\\item second"));
        assert!(out.contains("\\item third"));
    }

    #[test]
    fn test_markdown_to_latex_nested_list() {
        let out = markdown_to_latex("- outer\n  - inner\n- outer2\n");
        // Two itemize environments (outer and nested inner), properly closed.
        assert_eq!(out.matches("\\begin{itemize}").count(), 2);
        assert_eq!(out.matches("\\end{itemize}").count(), 2);
        assert!(out.contains("\\item inner"));
    }

    #[test]
    fn test_markdown_to_latex_blockquote() {
        let out = markdown_to_latex("> quoted text\n");
        assert!(out.contains("\\begin{quote}"));
        assert!(out.contains("quoted text"));
        assert!(out.contains("\\end{quote}"));
    }

    #[test]
    fn test_markdown_to_latex_fenced_code_block_is_not_escaped() {
        let out = markdown_to_latex("```\nlet x = a_b & c;\n```\n");
        assert!(out.contains("\\begin{verbatim}"), "got: {out:?}");
        // Verbatim content must be untouched - no LaTeX escaping applied.
        assert!(out.contains("let x = a_b & c;"), "got: {out:?}");
        assert!(out.contains("\\end{verbatim}"), "got: {out:?}");
    }

    #[test]
    fn test_markdown_to_latex_horizontal_rule() {
        let out = markdown_to_latex("above\n\n---\n\nbelow\n");
        assert!(out.contains("\\hrulefill"));
    }

    #[test]
    fn test_markdown_to_latex_hard_break_is_double_backslash() {
        // Two trailing spaces before a newline is Markdown's hard line
        // break syntax - real model output uses this inside list items
        // (see test_markdown_to_latex_real_model_response).
        let out = markdown_to_latex("line one  \nline two\n");
        assert!(out.contains("line one\\\\\nline two"), "got: {out:?}");
    }

    #[test]
    fn test_markdown_to_latex_soft_break_becomes_a_space() {
        // A single newline with no trailing spaces is just a soft wrap -
        // rendered as a space, same as CommonMark's own default rendering,
        // not a forced line break.
        let out = markdown_to_latex("line one\nline two\n");
        assert!(out.contains("line one line two"), "got: {out:?}");
    }

    #[test]
    fn test_markdown_to_latex_strikethrough() {
        let out = markdown_to_latex("~~gone~~");
        assert!(out.contains("\\sout{gone}"));
    }

    #[test]
    fn test_markdown_to_latex_table() {
        let out = markdown_to_latex("| A | B |\n|:--|--:|\n| 1 | 2 |\n");
        assert!(out.contains("\\begin{tabular}{lr}"), "got: {out:?}");
        assert!(out.contains("A & B"));
        assert!(out.contains("\\hline"));
        assert!(out.contains("1 & 2"));
        assert!(out.contains("\\end{tabular}"));
    }

    #[test]
    fn test_markdown_to_latex_link_renders_text_only() {
        let out = markdown_to_latex("see [the docs](https://example.com/) here");
        assert!(out.contains("see the docs here"));
        assert!(!out.contains("example.com"));
    }

    #[test]
    fn test_markdown_to_latex_image_renders_alt_text_placeholder() {
        let out = markdown_to_latex("![a diagram](https://example.com/x.png)");
        assert!(out.contains("[image: a diagram]"), "got: {out:?}");
    }

    #[test]
    fn test_markdown_to_latex_task_list() {
        let out = markdown_to_latex("- [ ] todo\n- [x] done\n");
        assert!(out.contains("$\\square$"));
        assert!(out.contains("$\\boxtimes$"));
    }

    #[test]
    fn test_markdown_to_latex_preserves_inline_math_verbatim_mid_sentence() {
        let out = markdown_to_latex("the value \\(x^2\\) is positive");
        assert!(
            out.contains("the value \\(x^2\\) is positive"),
            "got: {out:?}"
        );
        // Underscores/carets inside the math weren't escaped.
        let out2 = markdown_to_latex("here \\(a_1 + a^2\\) done");
        assert!(out2.contains("\\(a_1 + a^2\\)"), "got: {out2:?}");
    }

    #[test]
    fn test_markdown_to_latex_preserves_display_math_block() {
        let out = markdown_to_latex("Result:\n\n\\[\n\\boxed{\\frac{N(N+1)}{2}}\n\\]\n");
        assert!(
            out.contains("\\[\n\\boxed{\\frac{N(N+1)}{2}}\n\\]"),
            "got: {out:?}"
        );
    }

    #[test]
    fn test_markdown_to_latex_math_survives_inside_bold() {
        // A model bolding a pseudo-heading that itself contains inline
        // math (e.g. "**Key term:** \(S(N)\)") - the math must still come
        // out verbatim, not escaped, even though it's inside a Strong span.
        let out = markdown_to_latex("**Key term:** \\(S(N)\\) is the sum.");
        assert!(out.contains("\\textbf{Key term:}"));
        assert!(out.contains("\\(S(N)\\)"), "got: {out:?}");
    }

    #[test]
    fn test_markdown_to_latex_real_model_response() {
        // Verbatim final answer captured from a real local model response
        // (see git history / session notes) to --display-graphics=full's
        // motivating prompt: bold pseudo-headers, numbered lists containing
        // bold + inline math + trailing parenthetical text, and Markdown
        // hard line breaks (trailing double space) inside list items.
        let text = "**Key term (bold):** The sum of the first N positive integers, denoted by \\(S(N)\\).\n\n**Step-by-step reasoning ( bullet list):**\n1. **Definition:** \\(S(N) = \\sum_{k=1}^{N} k\\).  (Inconcrete LaTeX formula)  \n2. **Arithmetic series formula:** \\(S(N) = \\frac{N}{2}(a_1 + a_N)\\).  (Intermediate LaTeX formula)  \n3. **Substitute endpoints:** With \\(a_1 = 1\\) and \\(a_N = N\\), this becomes \\(S(N) = \\frac{N}{2}(1 + N)\\).  (Intermediate LaTeX formula)  \n4. **Simplify:** \\(S(N) = \\frac{N(N+1)}{2}\\).  (Final LaTeX formula)\n\n**Numbered list of key equations:**\n1. \\(S(N) = \\sum_{k=1}^{N} k\\).  (Inconcrete LaTeX formula)  \n2. \\(S(N) = \\frac{N}{2}(1 + N)\\).  (Intermediate LaTeX formula)  \n3. \\(S(N) = \\frac{N(N+1)}{2}\\).  (Final LaTeX formula)\n\n**Display LaTeX formula (boxed final answer):**\n\\[\n\\boxed{S(N)=\\frac{N(N+1)}{2}}\n\\]\n";
        let out = markdown_to_latex(text);

        // Bold pseudo-headers survive as \textbf, not real sections (the
        // model used bold text, not "#" ATX headers, for these).
        assert!(out.contains("\\textbf{Key term (bold):}"));
        assert!(out.contains("\\textbf{Display LaTeX formula (boxed final answer):}"));

        // Both numbered lists became real enumerate environments.
        assert_eq!(out.matches("\\begin{enumerate}").count(), 2);
        assert_eq!(out.matches("\\end{enumerate}").count(), 2);
        assert!(out.contains("\\item \\textbf{Definition:}"));

        // All the inline math survived verbatim, including subscripts and
        // underscores that would otherwise have been escaped as plain text.
        assert!(out.contains("\\(S(N) = \\sum_{k=1}^{N} k\\)"));
        assert!(out.contains("\\(a_1 = 1\\)"));

        // The final boxed display formula survived verbatim too.
        assert!(out.contains("\\[\n\\boxed{S(N)=\\frac{N(N+1)}{2}}\n\\]"));

        // The trailing double space at the end of each item (right before
        // the next numbered item starts) is just the item's own paragraph
        // ending, not a Markdown hard break - a hard break only applies
        // *within* a paragraph, and there's no more content after it in
        // this one. It should end up as a plain item boundary, not a
        // dangling "\\" with nothing following it on the same line.
        assert!(
            out.contains("(Inconcrete LaTeX formula)\n\\item"),
            "got: {out:?}"
        );

        // Nothing was silently dropped: nested inline math delimiters like
        // "S(N) =" (containing parentheses, not backslash-escaped ones)
        // didn't confuse the scan.
        assert!(out.contains("With \\(a_1 = 1\\) and \\(a_N = N\\)"));
    }

    #[test]
    fn test_markdown_to_latex_second_real_model_response() {
        // A second, differently-structured verbatim real response (see
        // git history / session notes): a bullet list with inline math,
        // several sequential \[...\] display blocks, a bold pseudo-header,
        // and a parenthetical aside containing its own inline math.
        let text = "To find the sum of the first \\(N\\) positive integers we want to evaluate  \n\n\\[\nS_N = 1+2+3+\\cdots+N .\n\\]\n\n**Step 1: Use the arithmetic\u{2011}series formula.**  \nThe numbers \\(1,2,\\dots,N\\) form an arithmetic progression with:\n\n- first term \\(a_1 = 1\\),\n- last term \\(a_N = N\\),\n- common difference \\(d = 1\\),\n- number of terms \\(n = N\\).\n\nFor any arithmetic progression, the sum of its first \\(n\\) terms is  \n\n\\[\nS_n = \\frac{n}{2}\\,(a_1 + a_n).\n\\]\n\nApplying this here:\n\n\\[\nS_N = \\frac{N}{2}\\,(1 + N).\n\\]\n\n**Step 2: Simplify the expression.**  \n\n\\[\nS_N = \\frac{N(N+1)}{2}.\n\\]\n\nThis formula works for every positive integer \\(N\\). (It can also be verified by induction, but the arithmetic\u{2011}series argument is sufficient for the sum.)\n\n\\[\n\\boxed{\\frac{N(N+1)}{2}}\n\\]\n";
        let out = markdown_to_latex(text);

        // All four display blocks survived verbatim, in order.
        assert!(out.contains("\\[\nS_N = 1+2+3+\\cdots+N .\n\\]"));
        assert!(out.contains("\\[\nS_n = \\frac{n}{2}\\,(a_1 + a_n).\n\\]"));
        assert!(out.contains("\\[\nS_N = \\frac{N}{2}\\,(1 + N).\n\\]"));
        assert!(out.contains("\\[\n\\boxed{\\frac{N(N+1)}{2}}\n\\]"));

        // The bullet list of arithmetic-progression facts became a real
        // itemize with the inline math intact in each item.
        assert!(out.contains("\\begin{itemize}"));
        assert!(out.contains("\\item first term \\(a_1 = 1\\),"));
        assert!(out.contains("\\end{itemize}"));

        // The bold pseudo-headers survived, with the non-breaking hyphen
        // transliterated to a plain one.
        assert!(out.contains("\\textbf{Step 1: Use the arithmetic-series formula.}"));
        assert!(out.contains("\\textbf{Step 2: Simplify the expression.}"));

        // The parenthetical aside (plain text, containing its own
        // reference to \(N\)) survived as ordinary escaped text around the
        // preserved math, not swallowed or corrupted by it.
        assert!(out.contains(
            "This formula works for every positive integer \\(N\\). (It can also be verified by induction, but the arithmetic-series argument is sufficient for the sum.)"
        ), "got: {out:?}");
    }

    #[test]
    fn test_full_latex_document_uses_article_class_and_varwidth() {
        let doc = full_latex_document("# Hi\n", (1, 2, 3));
        assert!(doc.contains("class=article"));
        assert!(doc.contains(&format!("varwidth={}pt", FULL_DOCUMENT_MAX_WIDTH_PT)));
        assert!(doc.contains("\\usepackage[utf8]{inputenc}"));
        assert!(doc.contains("\\usepackage[normalem]{ulem}"));
        assert!(doc.contains("\\color[RGB]{1,2,3}"));
        assert!(doc.contains("\\section*{Hi}"));
    }

    #[test]
    fn test_answer_and_reasoning_text_color_ansi_indices_are_distinct() {
        // Regression test for a real report: the answer and reasoning
        // streams must render in their own distinct color, not both fall
        // back to the same one - which is exactly what would happen if
        // these two constants (or the functions querying them) were ever
        // accidentally merged into one.
        assert_ne!(ANSWER_TEXT_ANSI_INDEX, REASONING_TEXT_ANSI_INDEX);
    }

    #[test]
    fn test_answer_and_reasoning_text_color_fallbacks_are_distinct() {
        assert_ne!(ANSWER_TEXT_COLOR_FALLBACK, REASONING_TEXT_COLOR_FALLBACK);
    }

    #[test]
    fn test_parse_osc4_reply_bel_terminated() {
        let reply = b"\x1b]4;6;rgb:0000/ffff/ffff\x07";
        assert_eq!(parse_osc4_reply(reply, 6), Some((0, 255, 255)));
    }

    #[test]
    fn test_parse_osc4_reply_st_terminated() {
        let reply = b"\x1b]4;6;rgb:8000/4000/2000\x1b\\";
        // 0x8000*255/0xffff = 127 (integer division), and so on - not
        // quite 128/64/32 exactly, which is the point of asserting the
        // real computed values here rather than a hand-rounded guess.
        assert_eq!(parse_osc4_reply(reply, 6), Some((127, 63, 31)));
    }

    #[test]
    fn test_parse_osc4_reply_wrong_index_is_rejected() {
        // A reply for a different palette index than the one we asked
        // about must not be mistaken for a match.
        let reply = b"\x1b]4;3;rgb:ffff/ffff/ffff\x07";
        assert_eq!(parse_osc4_reply(reply, 6), None);
    }

    #[test]
    fn test_parse_osc4_reply_garbage_is_rejected() {
        assert_eq!(parse_osc4_reply(b"not an osc4 reply at all", 6), None);
        assert_eq!(parse_osc4_reply(b"", 6), None);
    }

    #[test]
    fn test_parse_osc4_reply_hash_format() {
        // Some terminals reply with "#rrggbb" instead of "rgb:RRRR/GGGG/BBBB".
        // "ff" is left-shifted to 0xff00 (not scaled to 0xffff), so it comes
        // back as 254, not 255 - again asserting the real value, not a
        // rounded guess.
        let reply = b"\x1b]4;6;#00ffff\x07";
        assert_eq!(parse_osc4_reply(reply, 6), Some((0, 254, 254)));
    }
}
