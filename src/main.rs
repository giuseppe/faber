/*
 * codehawk
 *
 * Copyright (C) 2025 Giuseppe Scrivano <giuseppe@scrivano.org>
 * codehawk is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * codehawk is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with codehawk.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

mod db;
mod github;
mod openai;

use clap::{Parser, Subcommand};
use console::Style;
use env_logger::Env;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use log::{debug, trace, warn};
use pathrs::{Root, flags::OpenFlags};
use prettytable::{Cell, Row, Table, format};
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::fs::Permissions;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use github::{
    Issues, PullRequests, get_github_issue, get_github_issue_comments, get_github_issues,
    get_github_pull_request, get_github_pull_request_patch, get_github_pull_requests,
};
use openai::{
    FunctionCall, InterruptedError, Message, OpenAIResponse, ProgressInfo, ResponseMode,
    StatusUpdate, ToolCall, ToolCallback, ToolItem, ToolsCollection, list_models_from_endpoint,
    make_message, post_request, post_request_with_mode, tool_call,
};
use std::collections::HashMap;

const DEFAULT_ENDPOINT: &str = "http://localhost:8080";
const DEFAULT_MODEL: &str = "google/gemini-2.5-pro";
const DEFAULT_DAYS: u64 = 7;

// Import ToolContext from the library crate
use codehawk::ToolContext;

/// Parse parameter strings in NAME=VALUE format into a HashMap
fn parse_parameters(
    param_strings: &[String],
) -> Result<HashMap<String, serde_json::Value>, Box<dyn Error>> {
    let mut parameters = HashMap::new();

    for param in param_strings {
        if let Some((key, value)) = param.split_once('=') {
            let key = key.trim().to_string();
            let value_str = value.trim();

            // Try to parse as different types
            let json_value = if let Ok(num) = value_str.parse::<f64>() {
                serde_json::Value::Number(
                    serde_json::Number::from_f64(num)
                        .unwrap_or_else(|| serde_json::Number::from(0)),
                )
            } else if let Ok(bool_val) = value_str.parse::<bool>() {
                serde_json::Value::Bool(bool_val)
            } else if value_str == "null" {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(value_str.to_string())
            };

            debug!("Parsed parameter: {} = {:?}", key, json_value);
            parameters.insert(key, json_value);
        } else {
            return Err(
                format!("Invalid parameter format: '{}'. Expected NAME=VALUE", param).into(),
            );
        }
    }

    if !parameters.is_empty() {
        debug!(
            "Using {} custom parameters: {:?}",
            parameters.len(),
            parameters
        );
    }

    Ok(parameters)
}

fn append_tool(tools: &mut ToolsCollection, name: String, callback: ToolCallback, schema: String) {
    let item = ToolItem {
        callback: callback,
        schema: schema,
    };
    debug!("Adding tool: {}", name);
    tools.insert(name, item);
}

/// entrypoint for the delete_path tool
fn tool_delete_path(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        path: String,
    }
    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Remove path: {}", params.path);
    let root = Root::open(".")?;

    let path = PathBuf::from(&params.path);

    root.remove_all(path)?;
    Ok("deleted".to_owned())
}

/// entrypoint for the read_file tool
fn tool_read_file(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        path: String,
    }

    #[derive(Serialize)]
    struct ReadFileResult {
        content: Option<String>,
        error: Option<String>,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Reading file: {}", params.path);

    let root = Root::open(".")?;
    let path = PathBuf::from(&params.path);
    let file = root.open_subpath(path, OpenFlags::O_RDONLY);

    let result = match file {
        Ok(mut file) => {
            let mut contents = String::new();
            match file.read_to_string(&mut contents) {
                Ok(_) => ReadFileResult {
                    content: Some(contents),
                    error: None,
                },
                Err(e) => ReadFileResult {
                    content: None,
                    error: Some(format!("Failed to read file: {}", e)),
                },
            }
        }
        Err(e) => ReadFileResult {
            content: None,
            error: Some(format!("File not found or cannot be opened: {}", e)),
        },
    };

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// Show diff between old and new content using system diff tool with file descriptors
fn show_diff(ctx: &ToolContext, old_content: &str, new_content: &str, file_path: &str) {
    use std::io::Write;
    use std::os::unix::io::{AsRawFd, FromRawFd};
    use std::process::Command;

    // Create temporary files using O_TMPFILE in current directory
    // Use rustix crate for system calls since it's available through pathrs
    let old_fd_result = rustix::fs::openat(
        rustix::fs::CWD,
        ".",
        rustix::fs::OFlags::TMPFILE | rustix::fs::OFlags::RDWR,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    );

    let new_fd_result = rustix::fs::openat(
        rustix::fs::CWD,
        ".",
        rustix::fs::OFlags::TMPFILE | rustix::fs::OFlags::RDWR,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    );

    let (old_fd, new_fd) = match (old_fd_result, new_fd_result) {
        (Ok(old), Ok(new)) => (old, new),
        _ => {
            debug!("Failed to create temporary file descriptors");
            return;
        }
    };

    // Write content to the file descriptors
    let write_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let old_raw_fd = old_fd.as_raw_fd();
        let new_raw_fd = new_fd.as_raw_fd();

        let mut old_file = unsafe { std::fs::File::from_raw_fd(old_raw_fd) };
        let mut new_file = unsafe { std::fs::File::from_raw_fd(new_raw_fd) };

        old_file.write_all(old_content.as_bytes())?;
        new_file.write_all(new_content.as_bytes())?;

        // Reset file position to beginning for reading
        use std::io::Seek;
        old_file.seek(std::io::SeekFrom::Start(0))?;
        new_file.seek(std::io::SeekFrom::Start(0))?;

        // Don't let File::drop close the fds, we'll manage them manually
        std::mem::forget(old_file);
        std::mem::forget(new_file);

        Ok(())
    })();

    if let Err(e) = write_result {
        debug!("Failed to write to temporary file descriptors: {}", e);
        return;
    }

    // Run diff command using /proc/self/fd/ paths
    let old_path = format!("/proc/self/fd/{}", old_fd.as_raw_fd());
    let new_path = format!("/proc/self/fd/{}", new_fd.as_raw_fd());

    let diff_result = Command::new("diff")
        .arg("--color=always")
        .arg("-Naur")
        .arg("--label")
        .arg(&format!("a/{}", file_path))
        .arg("--label")
        .arg(&format!("b/{}", file_path))
        .arg(&old_path)
        .arg(&new_path)
        .output();

    // File descriptors will be automatically closed when old_fd and new_fd go out of scope

    match diff_result {
        Ok(output) => {
            // diff returns 0 for no differences, 1 for differences, >1 for errors
            if output.status.code() == Some(0) {
                ctx.println("   No differences detected");
            } else if output.status.code() == Some(1) {
                ctx.println("   Changes:");
                let diff_output = String::from_utf8_lossy(&output.stdout);
                for line in diff_output.lines() {
                    ctx.println(&format!("   {}", line));
                }
            } else {
                debug!("diff command failed with status: {:?}", output.status);
            }
        }
        Err(e) => {
            debug!("Failed to run diff command: {}", e);
        }
    }
}

/// entrypoint for the write_file tool
fn tool_write_file(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        path: String,
        content: String,
        #[serde(default = "default_file_mode")]
        mode: String,
    }

    #[derive(Serialize)]
    struct WriteFileResult {
        path: String,
        bytes_written: usize,
        mode: String,
        created: bool,
        message: String,
    }

    fn default_file_mode() -> String {
        "0644".to_string()
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "write_file received params: path='{}', content_length={}, mode='{}'",
        params.path,
        params.content.len(),
        params.mode
    );

    let file_mode = if params.mode.starts_with("0o") {
        u32::from_str_radix(&params.mode[2..], 8)
    } else if params.mode.starts_with("0") && params.mode.len() > 1 {
        u32::from_str_radix(&params.mode[1..], 8)
    } else {
        if params.mode.starts_with("0x") {
            u32::from_str_radix(&params.mode[2..], 16)
        } else {
            params.mode.parse::<u32>()
        }
    }
    .map_err(|_| {
        format!(
            "Invalid file mode format: '{}'. Expected octal (0644, 0o644), hex (0x1a4), or decimal",
            params.mode
        )
    })?;

    debug!("Parsed file mode: {} (octal: {:o})", file_mode, file_mode);

    let root = Root::open(".")?;
    let path_buf = PathBuf::from(&params.path);

    // Try to read existing file content for diff display
    let existing_content = match root.open_subpath(&params.path, OpenFlags::O_RDONLY) {
        Ok(mut file) => {
            let mut content = Vec::new();
            match file.read_to_end(&mut content) {
                Ok(_) => Some(content),
                Err(_) => None,
            }
        }
        Err(_) => None,
    };

    let (created, mut file) =
        match root.open_subpath(&params.path, OpenFlags::O_WRONLY | OpenFlags::O_TRUNC) {
            Ok(file) => {
                debug!("File '{}' already exists, will overwrite", params.path);
                (false, file)
            }
            Err(e) => {
                debug!("File '{}' does not exist ({}), will create", params.path, e);

                // Create parent directories if needed
                if let Some(parent) = path_buf.parent() {
                    if !parent.as_os_str().is_empty() {
                        debug!("Creating parent directories for: {}", parent.display());
                        root.mkdir_all(parent, &Permissions::from_mode(0o755))?;
                        debug!("Parent directories created successfully");
                    }
                }

                let permissions = Permissions::from_mode(file_mode);
                debug!("Using file permissions: {:o}", permissions.mode());

                let new_file = root.create_file(
                    &params.path,
                    OpenFlags::O_WRONLY | OpenFlags::O_CREAT,
                    &permissions,
                )?;
                (true, new_file)
            }
        };

    let bytes_written = params.content.len();
    debug!("Writing {} bytes to file: {}", bytes_written, params.path);

    file.write_all(&params.content.as_bytes())?;

    let result = WriteFileResult {
        path: params.path.clone(),
        bytes_written,
        mode: format!("{:o}", file_mode),
        created: created,
        message: if created {
            format!("File '{}' created successfully", params.path)
        } else {
            format!("File '{}' overwritten successfully", params.path)
        },
    };

    debug!(
        "write_file completed successfully: {} bytes written to '{}' with mode {:o}",
        bytes_written, params.path, file_mode
    );

    // Display output using context callback
    ctx.println(&format!("📝 {}", result.message));
    ctx.println(&format!("   Path: {}", result.path));
    ctx.println(&format!("   Bytes written: {}", result.bytes_written));
    ctx.println(&format!("   File mode: {}", result.mode));
    ctx.println(&format!(
        "   Operation: {}",
        if result.created {
            "CREATE"
        } else {
            "OVERWRITE"
        }
    ));

    // Show diff if overwriting an existing textual file
    if !created {
        if let Some(old_content_bytes) = existing_content {
            if std::str::from_utf8(params.content.as_bytes()).is_ok() {
                let old_content = String::from_utf8(old_content_bytes);
                match old_content {
                    Ok(old_content) => {
                        show_diff(ctx, &old_content, &params.content, &params.path);
                    }
                    Err(_) => {}
                }
            }
        }
    }

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// entrypoint for the glob tool
fn tool_glob(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Params {
        pattern: String,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;
    let glob_pattern = &params.pattern;

    // Security check: reject patterns that try to escape current directory
    if glob_pattern.contains("../") || glob_pattern.starts_with('/') {
        return Err("Pattern cannot access parent directories or absolute paths".into());
    }

    trace!("Executing glob pattern: {}", glob_pattern);

    let mut files = Vec::new();
    for entry in glob::glob(glob_pattern)? {
        match entry {
            Ok(path) => {
                // Additional security check: ensure resolved path is within current directory
                let canonical_path = path.canonicalize().unwrap_or(path.clone());
                let current_dir = std::env::current_dir()?;

                if canonical_path.starts_with(&current_dir) {
                    files.push(path.to_string_lossy().to_string());
                } else {
                    debug!("Skipping file outside current directory: {:?}", path);
                }
            }
            Err(e) => {
                debug!("Error processing glob entry: {}", e);
            }
        }
    }

    files.sort();
    let result = files.join("\n");
    debug!(
        "Successfully matched {} files with pattern '{}'",
        files.len(),
        glob_pattern
    );
    Ok(result)
}

/// entrypoint for the fetch_web_content tool
fn tool_fetch_web_content(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        url: String,
        #[serde(default)]
        timeout_seconds: Option<u64>,
        #[serde(default)]
        follow_redirects: Option<bool>,
        #[serde(default)]
        max_content_length: Option<usize>,
    }

    #[derive(Serialize)]
    struct WebContentResult {
        url: String,
        status_code: u16,
        content_type: Option<String>,
        content: String,
        content_length: usize,
        final_url: Option<String>,
        headers: std::collections::HashMap<String, String>,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Fetching web content from URL: {}", params.url);

    // Validate URL
    if !params.url.starts_with("http://") && !params.url.starts_with("https://") {
        return Err("URL must start with http:// or https://".into());
    }

    let timeout_seconds = params.timeout_seconds.unwrap_or(30);
    let follow_redirects = params.follow_redirects.unwrap_or(true);
    let max_content_length = params.max_content_length.unwrap_or(10_000_000);

    ctx.println(&format!("🌐 Fetching content from: {}", params.url));
    ctx.println(&format!("   Timeout: {}s", timeout_seconds));
    ctx.println(&format!("   Follow redirects: {}", follow_redirects));
    ctx.println(&format!(
        "   Max content length: {} bytes",
        max_content_length
    ));

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_seconds))
        .redirect(if follow_redirects {
            reqwest::redirect::Policy::limited(10)
        } else {
            reqwest::redirect::Policy::none()
        })
        .user_agent("codehawk/0.1.0")
        .build()?;

    let response = client.get(&params.url).send()?;

    let status_code = response.status().as_u16();
    let final_url = if response.url().as_str() != params.url {
        Some(response.url().to_string())
    } else {
        None
    };

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Collect headers
    let mut headers = std::collections::HashMap::new();
    for (key, value) in response.headers() {
        if let Ok(value_str) = value.to_str() {
            headers.insert(key.to_string(), value_str.to_string());
        }
    }

    let content = if response.status().is_success() {
        let content_bytes = response.bytes()?;
        if content_bytes.len() > max_content_length {
            return Err(format!(
                "Content too large: {} bytes (max: {})",
                content_bytes.len(),
                max_content_length
            )
            .into());
        }

        // Try to decode as UTF-8, fallback to lossy conversion
        match std::str::from_utf8(&content_bytes) {
            Ok(text) => text.to_string(),
            Err(_) => String::from_utf8_lossy(&content_bytes).to_string(),
        }
    } else {
        format!("HTTP Error: {}", response.status())
    };

    let content_length = content.len();

    let result = WebContentResult {
        url: params.url.clone(),
        status_code,
        content_type,
        content: content.clone(),
        content_length,
        final_url,
        headers,
    };

    // Display result using context callback
    ctx.println(&format!("📄 Content fetched successfully"));
    ctx.println(&format!("   Status: {}", status_code));
    if let Some(ref ct) = result.content_type {
        ctx.println(&format!("   Content-Type: {}", ct));
    }
    if let Some(ref final_url) = result.final_url {
        ctx.println(&format!("   Final URL: {}", final_url));
    }
    ctx.println(&format!("   Content length: {} bytes", content_length));

    // Show a preview of the content
    let preview_lines: Vec<&str> = content.lines().take(5).collect();
    if !preview_lines.is_empty() {
        ctx.println("   Content preview:");
        for line in preview_lines {
            let truncated = if line.len() > 100 {
                format!("{}...", &line[..97])
            } else {
                line.to_string()
            };
            ctx.println(&format!("     {}", truncated));
        }
        if content.lines().count() > 5 {
            ctx.println("     ... (content truncated in preview)");
        }
    }

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// entrypoint for the run_command tool
fn tool_run_command(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    use serde::Serialize;

    #[derive(Deserialize)]
    struct Params {
        command: String,
        args: Option<Vec<String>>,
    }

    #[derive(Serialize)]
    struct CommandResult {
        stdout: String,
        stderr: String,
        exit_code: Option<i32>,
        success: bool,
    }

    debug!("run_command received params: {}", params_str);

    // Try normal parsing first, fallback to manual parsing if it fails
    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    let mut cmd = if params.command.contains(' ') {
        // If command contains spaces
        let mut c = Command::new("sh");
        c.arg("-c").arg(&params.command);
        c
    } else {
        // Normal command execution
        let mut c = Command::new(&params.command);
        if let Some(ref args) = params.args {
            c.args(args);
        }
        c
    };

    trace!("Executing command: {:?}", cmd);

    let result = match cmd.output() {
        Ok(output) => {
            let stdout = String::from_utf8(output.stdout)
                .unwrap_or_else(|_| "<non-utf8 output>".to_string());
            let stderr = String::from_utf8(output.stderr)
                .unwrap_or_else(|_| "<non-utf8 output>".to_string());
            let exit_code = output.status.code();
            let success = output.status.success();

            if success {
                debug!("Successfully run command {}", params.command);
            } else {
                debug!(
                    "Command {} failed with exit code {:?}",
                    params.command, exit_code
                );
            }

            // Display output directly using context callback
            if success {
                ctx.println("✅ Command executed successfully:");
            } else {
                ctx.println(&format!(
                    "❌ Command failed (exit code: {}):",
                    exit_code.unwrap_or(-1)
                ));
            }
            if !stdout.is_empty() {
                ctx.println("STDOUT:");
                for line in stdout.lines() {
                    ctx.println(&format!("  {}", line));
                }
            }
            if !stderr.is_empty() {
                ctx.println("STDERR:");
                for line in stderr.lines() {
                    ctx.println(&format!("  {}", line));
                }
            }
            if stdout.is_empty() && stderr.is_empty() {
                ctx.println("(no output)");
            }
            CommandResult {
                stdout,
                stderr,
                exit_code,
                success,
            }
        }
        Err(e) => {
            debug!("Failed to execute command {}: {}", params.command, e);
            ctx.println(&format!("ERROR: Failed to execute command: {}", e));
            CommandResult {
                stdout: String::new(),
                stderr: format!("Failed to execute command: {}", e),
                exit_code: None,
                success: false,
            }
        }
    };

    // Output is now displayed directly above during command execution

    let json_result = serde_json::to_string(&result)?;
    Ok(json_result)
}

/// entrypoint for the grep_in_current_directory tool
fn tool_grep_in_current_directory(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        pattern: String,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    let mut cmd = Command::new("grep");
    cmd.arg("-r");
    cmd.arg("-n");
    cmd.arg(&params.pattern);

    debug!(
        "Grepping for pattern '{}' in current directory",
        params.pattern
    );

    trace!("Executing grep command: {:?}", cmd);
    let output = cmd.output()?;
    // Grep returns 1 if no lines were selected, 0 if lines were selected, >1 for errors.
    // We consider no lines selected as a valid, empty result, not an error for the tool.
    if !output.status.success() && output.status.code() != Some(1) {
        let stderr = String::from_utf8(output.stderr)?;
        let err_msg = format!(
            "grep command failed with status {:?}. Stderr: {}",
            output.status, stderr
        );
        let err: Box<dyn Error> = err_msg.into();
        return Err(err);
    }

    let r = String::from_utf8(output.stdout)?;
    debug!(
        "Grep command successfully executed for pattern '{}'",
        params.pattern
    );

    // Display grep results directly using context callback
    ctx.println(&format!(
        "🔍 Grep results for pattern '{}':",
        params.pattern
    ));

    if r.is_empty() {
        ctx.println("(no matches found)");
    } else {
        let line_count = r.lines().count();
        ctx.println(&format!("Found {} matches:", line_count));
        for line in r.lines().take(10) {
            // Show first 10 matches
            ctx.println(&format!("  {}", line));
        }
        if line_count > 10 {
            ctx.println(&format!("  ... and {} more matches", line_count - 10));
        }
    }

    Ok(r)
}

/// entrypoint for the github_issue tool
fn tool_github_issue(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        issue: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!("Fetching GitHub issue: {}/{}", params.repo, params.issue);
    let issue = get_github_issue(&params.repo, params.issue)?;

    let s = serde_json::to_string(&issue)?;
    Ok(s)
}

/// entrypoint for the github_issue_comments tool
fn tool_github_issue_comments(
    params_str: &String,
    _ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        issue: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub issue comments: {}/{}",
        params.repo, params.issue
    );
    let comments = get_github_issue_comments(&params.repo, params.issue)?;
    let s = serde_json::to_string(&comments)?;
    Ok(s)
}

/// entrypoint for the github_issues tool
fn tool_github_issues(params_str: &String, _ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        days: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub issues from repository {} for the last {} days",
        params.repo, params.days
    );
    let issues = get_github_issues(&params.repo, params.days)?;
    let s = serde_json::to_string(&issues)?;
    Ok(s)
}

/// entrypoint for the github_pull_requests tool
fn tool_github_pull_requests(
    params_str: &String,
    _ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        days: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub pull requests from repository {} for the last {} days",
        params.repo, params.days
    );
    let pull_requests = get_github_pull_requests(&params.repo, params.days)?;
    let s = serde_json::to_string(&pull_requests)?;
    Ok(s)
}

/// entrypoint for the github_pull_request tool
fn tool_github_pull_request(
    params_str: &String,
    _ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        pull_request: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub PR: {}/{}",
        params.repo, params.pull_request
    );
    let pr = get_github_pull_request(&params.repo, params.pull_request)?;
    let s = serde_json::to_string(&pr)?;
    Ok(s)
}

/// entrypoint for the github_pull_request_patch tool
fn tool_github_pull_request_patch(
    params_str: &String,
    _ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        repo: String,
        pull_request: u64,
    }

    let params: Params = serde_json::from_str::<Params>(&params_str)?;

    debug!(
        "Fetching GitHub PR patch: {}/{}",
        params.repo, params.pull_request
    );
    let pr = get_github_pull_request_patch(&params.repo, params.pull_request)?;
    Ok(pr)
}

fn tool_agent_create(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
        #[serde(default)]
        description: Option<String>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    db::create_agent(
        &conn,
        &params.name,
        params.description.as_deref().unwrap_or(""),
    )?;
    ctx.println(&format!("Agent '{}' created successfully", params.name));
    let result = serde_json::json!({"status": "created", "name": params.name});
    Ok(result.to_string())
}

fn tool_agent_delete(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    let deleted = db::delete_agent(&conn, &params.name)?;
    if deleted {
        ctx.println(&format!("Agent '{}' deleted", params.name));
        let result = serde_json::json!({"status": "deleted", "name": params.name});
        Ok(result.to_string())
    } else {
        let result = serde_json::json!({"status": "not_found", "name": params.name});
        Ok(result.to_string())
    }
}

fn tool_agent_list(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    let _ = params_str;
    let conn = ctx.db_conn()?;
    let agents = db::list_agents(&conn)?;
    ctx.println(&format!("Found {} agent(s)", agents.len()));
    for a in &agents {
        ctx.println(&format!("  - {} ({})", a.name, a.description));
    }
    Ok(serde_json::to_string(&agents)?)
}

fn tool_agent_get(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    match db::get_agent(&conn, &params.name)? {
        Some(agent) => {
            ctx.println(&format!("Agent '{}': {}", agent.name, agent.description));
            Ok(serde_json::to_string(&agent)?)
        }
        None => {
            let result = serde_json::json!({"error": format!("Agent '{}' not found", params.name)});
            Ok(result.to_string())
        }
    }
}

fn tool_agent_data_set(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        agent: String,
        key: String,
        value: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    db::set_agent_data(&conn, &params.agent, &params.key, &params.value)?;
    ctx.println(&format!(
        "Set data for agent '{}': {} = {}",
        params.agent, params.key, params.value
    ));
    let result = serde_json::json!({"status": "ok", "agent": params.agent, "key": params.key});
    Ok(result.to_string())
}

fn tool_agent_data_get(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        agent: String,
        key: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    match db::get_agent_data(&conn, &params.agent, &params.key)? {
        Some(value) => {
            ctx.println(&format!(
                "Agent '{}' data: {} = {}",
                params.agent, params.key, value
            ));
            let result =
                serde_json::json!({"agent": params.agent, "key": params.key, "value": value});
            Ok(result.to_string())
        }
        None => {
            let result =
                serde_json::json!({"agent": params.agent, "key": params.key, "value": null});
            Ok(result.to_string())
        }
    }
}

fn tool_agent_data_delete(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        agent: String,
        key: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    let deleted = db::delete_agent_data(&conn, &params.agent, &params.key)?;
    if deleted {
        ctx.println(&format!(
            "Deleted key '{}' for agent '{}'",
            params.key, params.agent
        ));
    }
    let result = serde_json::json!({"deleted": deleted, "agent": params.agent, "key": params.key});
    Ok(result.to_string())
}

fn tool_agent_data_list(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        agent: String,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    let data = db::list_agent_data(&conn, &params.agent)?;
    ctx.println(&format!(
        "Agent '{}' has {} data entries",
        params.agent,
        data.len()
    ));
    for (k, v) in &data {
        ctx.println(&format!("  {} = {}", k, v));
    }
    let entries: Vec<serde_json::Value> = data
        .into_iter()
        .map(|(k, v)| serde_json::json!({"key": k, "value": v}))
        .collect();
    Ok(serde_json::to_string(&entries)?)
}

fn tool_task_create_cron(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
        cron_expression: String,
        #[serde(default)]
        command: Option<String>,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        agent_name: Option<String>,
        #[serde(default)]
        max_runs: Option<i64>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    let id = db::create_cron_task(
        &conn,
        &params.name,
        params.description.as_deref().unwrap_or(""),
        &params.cron_expression,
        params.command.as_deref().unwrap_or(""),
        params.agent_name.as_deref(),
        params.max_runs,
    )?;
    ctx.println(&format!(
        "Created cron task '{}' (id={}) with schedule '{}'",
        params.name, id, params.cron_expression
    ));
    let result = serde_json::json!({"status": "created", "id": id, "name": params.name});
    Ok(result.to_string())
}

fn tool_task_create_oneshot(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        name: String,
        #[serde(default)]
        run_at: Option<String>,
        #[serde(default)]
        delay_seconds: Option<u64>,
        #[serde(default)]
        command: Option<String>,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        agent_name: Option<String>,
    }
    let params: Params = serde_json::from_str(params_str)?;

    let run_at = match (params.run_at, params.delay_seconds) {
        (Some(t), _) => t,
        (None, Some(secs)) => {
            let when = chrono::Utc::now() + chrono::Duration::seconds(secs as i64);
            when.to_rfc3339()
        }
        (None, None) => {
            return Err("Either 'run_at' or 'delay_seconds' must be provided".into());
        }
    };

    let conn = ctx.db_conn()?;
    let id = db::create_oneshot_task(
        &conn,
        &params.name,
        params.description.as_deref().unwrap_or(""),
        &run_at,
        params.command.as_deref().unwrap_or(""),
        params.agent_name.as_deref(),
    )?;
    ctx.println(&format!(
        "Created one-shot task '{}' (id={}) scheduled at '{}'",
        params.name, id, run_at
    ));
    let result =
        serde_json::json!({"status": "created", "id": id, "name": params.name, "run_at": run_at});
    Ok(result.to_string())
}

fn tool_task_delete(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        id: i64,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    let deleted = db::delete_task(&conn, params.id)?;
    if deleted {
        ctx.println(&format!("Deleted task id={}", params.id));
    }
    let result = serde_json::json!({"deleted": deleted, "id": params.id});
    Ok(result.to_string())
}

fn tool_task_list(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        #[serde(default)]
        agent_name: Option<String>,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    let tasks = db::list_tasks(&conn, params.agent_name.as_deref())?;
    ctx.println(&format!("Found {} task(s)", tasks.len()));
    for t in &tasks {
        let status = if t.enabled { "enabled" } else { "disabled" };
        ctx.println(&format!(
            "  [{}] {} - {} ({})",
            t.id, t.name, t.task_type, status
        ));
    }
    Ok(serde_json::to_string(&tasks)?)
}

fn tool_task_get(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        id: i64,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    match db::get_task(&conn, params.id)? {
        Some(task) => {
            ctx.println(&format!(
                "Task [{}]: {} ({})",
                task.id, task.name, task.task_type
            ));
            Ok(serde_json::to_string(&task)?)
        }
        None => {
            let result = serde_json::json!({"error": format!("Task id={} not found", params.id)});
            Ok(result.to_string())
        }
    }
}

fn tool_task_set_enabled(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Params {
        id: i64,
        enabled: bool,
    }
    let params: Params = serde_json::from_str(params_str)?;
    let conn = ctx.db_conn()?;
    let updated = db::set_task_enabled(&conn, params.id, params.enabled)?;
    let status = if params.enabled {
        "enabled"
    } else {
        "disabled"
    };
    if updated {
        ctx.println(&format!("Task id={} {}", params.id, status));
    }
    let result =
        serde_json::json!({"updated": updated, "id": params.id, "enabled": params.enabled});
    Ok(result.to_string())
}

fn tool_task_pending(params_str: &String, ctx: &ToolContext) -> Result<String, Box<dyn Error>> {
    let _ = params_str;
    let conn = ctx.db_conn()?;
    let tasks = db::get_pending_tasks(&conn)?;
    ctx.println(&format!("Found {} pending task(s)", tasks.len()));
    for t in &tasks {
        ctx.println(&format!(
            "  [{}] {} - {} (next: {})",
            t.id,
            t.name,
            t.task_type,
            t.next_run_at.as_deref().unwrap_or("N/A")
        ));
    }
    Ok(serde_json::to_string(&tasks)?)
}

fn initialize_tools(unsafe_tools: bool) -> ToolsCollection {
    let mut tools: ToolsCollection = ToolsCollection::new();

    append_tool(
        &mut tools,
        "github_pull_request".to_string(),
        tool_github_pull_request,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_pull_request",
                "description": "Get information about a github pull request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. giuseppe/codehawk"
                        },
                        "pull_request": {
                            "type": "number",
                            "description": "number of the pull request"
                        }
                    },
                    "required": [
                        "repo",
                        "pull_request"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_pull_request_patch".to_string(),
        tool_github_pull_request_patch,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_pull_request_patch",
                "description": "Get the raw patch for a github pull request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. giuseppe/codehawk"
                        },
                        "pull_request": {
                            "type": "number",
                            "description": "number of the pull request"
                        }
                    },
                    "required": [
                        "repo",
                        "pull_request"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "read_file".to_string(),
        tool_read_file,
        r#"
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Get the content of a file stored in the repository. Returns JSON with 'content' field containing file content on success, or 'error' field with error message on failure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the repository, e.g. src/main.rs"
                        }
                    },
                    "required": [
                        "path"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "delete_path".to_string(),
        tool_delete_path,
        r#"
        {
            "type": "function",
            "function": {
                "name": "delete_path",
                "description": "Delete a file or a directory under the current directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the current directory"
                        }
                    },
                    "required": [
                        "path"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "write_file".to_string(),
        tool_write_file,
        r#"
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or replace the content of a file stored in the repository. Returns detailed information about the write operation including bytes written, file mode, and whether the file was created or overwritten.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "path of the file under the repository, e.g. src/main.rs"
                        },
                        "content": {
                            "type": "string",
                            "description": "the content of the new file"
                        },
                        "mode": {
                            "type": "string",
                            "description": "file permissions mode in octal format (e.g., '0644', '0755', '0600'). Defaults to '0644' for regular files. Use '0755' for executable files.",
                            "default": "0644"
                        }
                    },
                    "required": [
                        "path",
                        "content"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "glob".to_string(),
        tool_glob,
        r#"
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files matching a glob pattern in the current directory. Supports patterns like '*.rs', '**/*.txt', 'src/**/*.rs', etc. Cannot access parent directories or absolute paths for security.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The glob pattern to match files against (e.g., '*.rs', '**/*.txt', 'src/**/*.rs'). Cannot contain '../' or start with '/' for security."
                        }
                    },
                    "required": [
                        "pattern"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "grep_in_current_directory".to_string(),
        tool_grep_in_current_directory,
        r#"
        {
            "type": "function",
            "function": {
                "name": "grep_in_current_directory",
                "description": "Grep for a pattern in the current directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The pattern to search for."
                        }
                    },
                    "required": [
                        "pattern"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_issue".to_string(),
        tool_github_issue,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_issue",
                "description": "Get the github issue description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. giuseppe/codehawk"
                        },
                        "issue": {
                            "type": "number",
                            "description": "number of the github issue"
                        }
                    },
                    "required": [
                        "repo",
                        "issue"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_issue_comments".to_string(),
        tool_github_issue_comments,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_issue_comments",
                "description": "Get the comments associated with the github issue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. giuseppe/codehawk"
                        },
                        "issue": {
                            "type": "number",
                            "description": "number of the github issue"
                        }
                    },
                    "required": [
                        "repo",
                        "issue"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_issues".to_string(),
        tool_github_issues,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_issues",
                "description": "Get issues from a GitHub repository that have been updated within the last specified number of days.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. giuseppe/codehawk"
                        },
                        "days": {
                            "type": "number",
                            "description": "number of days to look back for updated issues"
                        }
                    },
                    "required": [
                        "repo",
                        "days"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "github_pull_requests".to_string(),
        tool_github_pull_requests,
        r#"
        {
            "type": "function",
            "function": {
                "name": "github_pull_requests",
                "description": "Get pull requests from a GitHub repository that have been updated within the last specified number of days.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "github repo name, e.g. giuseppe/codehawk"
                        },
                        "days": {
                            "type": "number",
                            "description": "number of days to look back for updated pull requests"
                        }
                    },
                    "required": [
                        "repo",
                        "days"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_create".to_string(),
        tool_agent_create,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_create",
                "description": "Create a new named agent in the database for organizing tasks and storing data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Unique name for the agent"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of the agent's purpose"
                        }
                    },
                    "required": [
                        "name"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_delete".to_string(),
        tool_agent_delete,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_delete",
                "description": "Delete an agent and all its associated data from the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the agent to delete"
                        }
                    },
                    "required": [
                        "name"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_list".to_string(),
        tool_agent_list,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_list",
                "description": "List all agents in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_get".to_string(),
        tool_agent_get,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_get",
                "description": "Get details of a specific agent by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the agent to retrieve"
                        }
                    },
                    "required": [
                        "name"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_data_set".to_string(),
        tool_agent_data_set,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_data_set",
                "description": "Store a key-value pair for an agent. Overwrites existing value if key exists.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent"
                        },
                        "key": {
                            "type": "string",
                            "description": "The key to store"
                        },
                        "value": {
                            "type": "string",
                            "description": "The value to associate with the key"
                        }
                    },
                    "required": [
                        "agent",
                        "key",
                        "value"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_data_get".to_string(),
        tool_agent_data_get,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_data_get",
                "description": "Retrieve the value for a key stored for an agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent"
                        },
                        "key": {
                            "type": "string",
                            "description": "The key to retrieve"
                        }
                    },
                    "required": [
                        "agent",
                        "key"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_data_delete".to_string(),
        tool_agent_data_delete,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_data_delete",
                "description": "Delete a key-value pair for an agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent"
                        },
                        "key": {
                            "type": "string",
                            "description": "The key to delete"
                        }
                    },
                    "required": [
                        "agent",
                        "key"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "agent_data_list".to_string(),
        tool_agent_data_list,
        r#"
        {
            "type": "function",
            "function": {
                "name": "agent_data_list",
                "description": "List all key-value pairs stored for an agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Name of the agent"
                        }
                    },
                    "required": [
                        "agent"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_create_cron".to_string(),
        tool_task_create_cron,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_create_cron",
                "description": "Create a recurring scheduled task using a cron expression. Uses 7-field cron format: 'sec min hour day_of_month month day_of_week year'. Example: '0 30 9 * * Mon-Fri *' means 9:30 AM every weekday. The command is sent to the AI when the task fires.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the task"
                        },
                        "cron_expression": {
                            "type": "string",
                            "description": "7-field cron expression: sec min hour day_of_month month day_of_week year"
                        },
                        "command": {
                            "type": "string",
                            "description": "JSON tool call to execute when the task fires, e.g. {\"tool\": \"run_command\", \"arguments\": {\"command\": \"echo hello\"}}"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of what the task does"
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "Optional agent to associate the task with"
                        },
                        "max_runs": {
                            "type": "integer",
                            "description": "Maximum number of times the task will run before auto-disabling. Omit for unlimited."
                        }
                    },
                    "required": [
                        "name",
                        "cron_expression",
                        "command"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_create_oneshot".to_string(),
        tool_task_create_oneshot,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_create_oneshot",
                "description": "Create a one-shot scheduled task that runs once. Use delay_seconds for relative timing (preferred) or run_at for absolute. The command is sent to the AI when the task fires.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the task"
                        },
                        "command": {
                            "type": "string",
                            "description": "JSON tool call to execute when the task fires, e.g. {\"tool\": \"run_command\", \"arguments\": {\"command\": \"echo hello\"}}"
                        },
                        "delay_seconds": {
                            "type": "number",
                            "description": "Number of seconds from now to fire the task (preferred over run_at)"
                        },
                        "run_at": {
                            "type": "string",
                            "description": "Absolute time in RFC 3339 format (e.g. '2025-12-31T23:59:59Z'). Use delay_seconds instead when possible."
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of what the task does"
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "Optional agent to associate the task with"
                        }
                    },
                    "required": [
                        "name",
                        "command"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_delete".to_string(),
        tool_task_delete,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_delete",
                "description": "Delete a scheduled task by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "number",
                            "description": "ID of the task to delete"
                        }
                    },
                    "required": [
                        "id"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_list".to_string(),
        tool_task_list,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_list",
                "description": "List all scheduled tasks, optionally filtered by agent name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Optional agent name to filter tasks by"
                        }
                    },
                    "required": [],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_get".to_string(),
        tool_task_get,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_get",
                "description": "Get details of a specific scheduled task by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "number",
                            "description": "ID of the task to retrieve"
                        }
                    },
                    "required": [
                        "id"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_set_enabled".to_string(),
        tool_task_set_enabled,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_set_enabled",
                "description": "Enable or disable a scheduled task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "number",
                            "description": "ID of the task"
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "true to enable, false to disable"
                        }
                    },
                    "required": [
                        "id",
                        "enabled"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "task_pending".to_string(),
        tool_task_pending,
        r#"
        {
            "type": "function",
            "function": {
                "name": "task_pending",
                "description": "List all enabled scheduled tasks whose next_run_at is in the past (i.e. tasks that are due to run).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    if !unsafe_tools {
        return tools;
    }

    append_tool(
        &mut tools,
        "fetch_web_content".to_string(),
        tool_fetch_web_content,
        r#"
        {
            "type": "function",
            "function": {
                "name": "fetch_web_content",
                "description": "Fetch content from a web URL. Returns JSON with URL, status code, content type, headers, and the actual content. Supports HTTP/HTTPS URLs with configurable timeout and redirect handling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The HTTP/HTTPS URL to fetch content from"
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "description": "Request timeout in seconds (default: 30)"
                        },
                        "follow_redirects": {
                            "type": "boolean",
                            "description": "Whether to follow HTTP redirects (default: true)"
                        },
                        "max_content_length": {
                            "type": "number",
                            "description": "Maximum content length in bytes (default: 10MB)"
                        }
                    },
                    "required": [
                        "url"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    append_tool(
        &mut tools,
        "run_command".to_string(),
        tool_run_command,
        r#"
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a command and return results in JSON format. Returns {\"stdout\": string, \"stderr\": string, \"exit_code\": number|null, \"success\": boolean}. Commands do not fail on non-zero exit codes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute, it must be the name of the executable only, e.g. /usr/bin/ls"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Additional arguments to pass to the command after the executable path, e.g. [\"-l\", \"-a\"]"
                        }
                    },
                    "required": [
                        "command"
                    ],
                    "additionalProperties": false
                }
            }
        }
"#
        .to_string(),
    );

    debug!("Tools initialization completed with {} tools", tools.len());
    tools
}

fn add_tools_prompt(messages: &mut Vec<Message>, use_tools: bool) {
    let prompts = if use_tools {
        vec![
            "Use the available tools as much as possible to find a solution.  Iterate until the problem is solved.  Terminate only when you are sure to have found the solution, if a tool fails, analyze the failure, fix the issue and call again the tool.  Never ask to run commands manually or ask for permissions, just run the tool.",
            "When you've completed the task, you must terminate immediately the execution, do not explain your choices multiple times.",
        ]
    } else {
        vec![
            "You have access to various tools for file operations and code analysis.  Only use these tools when the user explicitly asks for file operations, code analysis, or repository interactions.  For simple questions, conversations, or general requests, respond directly without using tools.",
        ]
    };

    for prompt in prompts {
        messages.push(make_message("system", prompt.to_string()));
    }
}

fn add_predefined_system_prompts(messages: &mut Vec<Message>) {
    let predefined_prompts = vec![
        "You are codehawk, an AI assistant that helps with software development and repository analysis.",
        "When working with code, maintain best practices and consider security implications.",
    ];

    for prompt in predefined_prompts {
        messages.push(make_message("system", prompt.to_string()));
    }
}

fn initialize_chat_messages(tools: &ToolsCollection, opts: &Opts) -> Vec<Message> {
    let mut messages: Vec<Message> = vec![];

    // Add predefined system prompts unless explicitly disabled
    if !opts.no_system_prompts {
        add_predefined_system_prompts(&mut messages);

        let use_tools = !tools.is_empty()
            && match opts.tool_choice {
                Some(ref v) => v != "none",
                None => true,
            };

        add_tools_prompt(&mut messages, use_tools);
    }

    debug!("Initialized chat with {} system messages", messages.len());
    messages
}

/// Sends a prompt to the OpenAI API and prints the AI's response to standard output.
fn post_request_and_print_output(
    prompt: &String,
    system_prompts: Option<Vec<String>>,
    opts: &Opts,
    db: Option<Arc<Mutex<rusqlite::Connection>>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Prompt: {}", prompt);

    let model = opts.model.clone().unwrap_or(DEFAULT_MODEL.to_string());
    debug!("Using model: {}", model);

    let parameters = parse_parameters(&opts.parameter)?;

    let endpoint = opts
        .endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    let normalized_endpoint = openai::normalize_endpoint(&endpoint);

    let openai_opts = openai::Opts {
        max_tokens: opts.max_tokens,
        model: model,
        endpoint: normalized_endpoint,
        tool_choice: opts.tool_choice.clone(),
        api_key: opts.api_key.clone(),
        max_retries: None,
        retry_base_delay_secs: None,
        parameters,
    };

    let tools = match opts.no_tools {
        true => {
            debug!("Tools are disabled");
            ToolsCollection::new()
        }
        false => {
            debug!("Initializing tools for AI request");
            initialize_tools(opts.unsafe_tools)
        }
    };

    let mut messages = initialize_chat_messages(&tools, opts);

    if let Some(ref sys_prompts) = system_prompts {
        debug!("Using {} system prompts", sys_prompts.len());
        for sp in sys_prompts {
            messages.push(make_message("system", sp.clone()));
        }
    }
    messages.push(make_message("user", prompt.clone()));

    let mut tool_context = ToolContext::new(|msg: &str| {
        println!("{}", msg);
    });
    tool_context.db = db;

    let response: OpenAIResponse = post_request(messages, &tools, &openai_opts, &tool_context)?;

    if let Some(choices) = response.choices {
        debug!("Received {} choices in response", choices.len());
        if let Some(choice) = choices.first() {
            if let Some(content) = &choice.message.content {
                println!("{}", content);
            }
        }
    } else {
        warn!("No choices received in the AI response");
    }
    Ok(())
}

/// Fetches a GitHub pull request and its patch, then sends them to the AI for review.
fn review_pull_request(
    repo: &String,
    pr_id: u64,
    opts: &Opts,
    db: Option<Arc<Mutex<rusqlite::Connection>>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Reviewing pull request {}/{}", repo, pr_id);

    let pr = get_github_pull_request(repo, pr_id)?;
    let patch = get_github_pull_request_patch(repo, pr_id)?;

    let pr_json = serde_json::to_string(&pr)?;

    let system_prompts: Vec<String> = vec![patch, pr_json];
    let prompt = "Review the following pull request and report any issue with it, pay attention to the code.  Report only what is wrong, don't highlight what is done correctly.".to_string();

    post_request_and_print_output(&prompt, Some(system_prompts), opts, db)
}

/// Fetches a GitHub issue and its comments, then sends them to the AI for triaging.
fn triage_issue(
    repo: &String,
    issue_id: u64,
    opts: &Opts,
    db: Option<Arc<Mutex<rusqlite::Connection>>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Triaging issue {}/{}", repo, issue_id);

    let issue = get_github_issue(repo, issue_id)?;
    let comments = get_github_issue_comments(repo, issue_id)?;

    let prompt = "Provide a triage for the specified issue, show a minimal reproducer for the issue reducing the dependencies needed to run it.".to_string();

    let issue_json = serde_json::to_string(&issue)?;
    let comments_json = serde_json::to_string(&comments)?;

    let system_prompts: Vec<String> = vec![issue_json, comments_json];

    post_request_and_print_output(&prompt, Some(system_prompts), opts, db)
}

/// Fetches recent issues and pull requests from specified repositories and sends them to the AI with a given command prompt.
fn prompt_issues_and_pull_requests(
    prompt: &str,
    repos: &Vec<String>,
    days: Option<u64>,
    opts: &Opts,
    db: Option<Arc<Mutex<rusqlite::Connection>>>,
) -> Result<(), Box<dyn Error>> {
    let days = days.unwrap_or_else(|| DEFAULT_DAYS);
    debug!(
        "Fetching issues and PRs from {} repositories for the past {} days",
        repos.len(),
        days
    );

    let mut issues: Issues = Issues::new();
    let mut pull_requests: PullRequests = PullRequests::new();

    for repo in repos {
        debug!("Processing repository: {}", repo);

        let mut repo_issues = get_github_issues(repo, days)?;
        issues.append(&mut repo_issues);

        let mut repo_pull_requests = get_github_pull_requests(repo, days)?;
        pull_requests.append(&mut repo_pull_requests);
    }

    debug!(
        "Total: {} issues and {} pull requests found",
        issues.len(),
        pull_requests.len()
    );

    let issues_json = serde_json::to_string(&issues)?;
    let prs_json = serde_json::to_string(&pull_requests)?;
    let system_prompts: Vec<String> = vec![issues_json, prs_json];
    let prompt_string = prompt.to_string();

    post_request_and_print_output(&prompt_string, Some(system_prompts), opts, db)
}

/// Analyzes recent issues and pull requests for the specified repositories.
fn analyze_repos(
    repos: &Vec<String>,
    days: Option<u64>,
    opts: &Opts,
    db: Option<Arc<Mutex<rusqlite::Connection>>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Analyzing repos: {:?} for past {:?} days", repos, days);
    prompt_issues_and_pull_requests(
        "Provide a summary of all the issues and pull requests listed, highlighting the most important ones\n",
        repos,
        days,
        opts,
        db,
    )
}

/// Prioritizes recent issues and pull requests for the specified repositories.
fn prioritize_repos(
    repos: &Vec<String>,
    days: Option<u64>,
    opts: &Opts,
    db: Option<Arc<Mutex<rusqlite::Connection>>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Prioritizing repos: {:?} for past {:?} days", repos, days);
    prompt_issues_and_pull_requests(
        "Given the issues and pull requests listed, order them by importance and highlight the ones I must address first and why\n",
        repos,
        days,
        opts,
        db,
    )
}

/// Sends the concatenated content of specified files as a prompt to the AI.
fn prompt_command(
    prompt: &String,
    files: &Vec<String>,
    opts: &Opts,
    db: Option<Arc<Mutex<rusqlite::Connection>>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Executing prompt command with {} files", files.len());
    let mut system_prompts: Vec<String> = vec![];

    for file in files {
        debug!("Reading file for prompt context: {}", file);
        let contents = fs::read_to_string(file)?;
        system_prompts.push(contents);
    }
    post_request_and_print_output(prompt, Some(system_prompts), opts, db)
}

enum ChatCommand {
    Quit,
    Clear,
    Show,
    Limit(usize),
    Backtrace(usize),
    System(String),
    Message(String),
    Empty,
    Invalid(String),
}

fn parse_chat_command(line: &str) -> ChatCommand {
    if line.is_empty() {
        return ChatCommand::Empty;
    }

    if line == "\\quit" {
        return ChatCommand::Quit;
    }
    if line == "\\clear" {
        return ChatCommand::Clear;
    }
    if line == "\\show" {
        return ChatCommand::Show;
    }
    if line.starts_with("\\limit ") {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            if let Ok(n) = parts[1].parse::<usize>() {
                return ChatCommand::Limit(n);
            }
        }
        return ChatCommand::Invalid("Usage: \\limit <number_of_messages>".to_string());
    }
    if line.starts_with("\\backtrace ") {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            if let Ok(n) = parts[1].parse::<usize>() {
                return ChatCommand::Backtrace(n);
            }
        }
        return ChatCommand::Invalid("Usage: \\backtrace <number_of_messages>".to_string());
    }
    if line.starts_with("\\system ") {
        let system_message = line.strip_prefix("\\system ").unwrap_or("").to_string();
        if !system_message.is_empty() {
            return ChatCommand::System(system_message);
        }
        return ChatCommand::Invalid("Usage: \\system <message>".to_string());
    }

    ChatCommand::Message(line.to_string())
}

fn handle_chat_command(
    command: ChatCommand,
    messages: &mut Vec<Message>,
    tools: &ToolsCollection,
    opts: &Opts,
    chat_pb: &ProgressBar,
) -> Result<bool, Box<dyn Error>> {
    match command {
        ChatCommand::Quit => Ok(false),
        ChatCommand::Clear => {
            *messages = initialize_chat_messages(tools, opts);
            chat_pb.println("Chat history cleared and system prompts restored.");
            Ok(true)
        }
        ChatCommand::Show => {
            if messages.is_empty() {
                chat_pb.println("Chat history is empty.");
            } else {
                chat_pb.println("Current chat history:");
                for (i, msg) in messages.iter().enumerate() {
                    chat_pb.println(&format!(
                        "{}: [{}] {}",
                        i + 1,
                        msg.role,
                        msg.content.as_ref().unwrap_or(&"<no content>".to_string())
                    ));
                    if let Some(tool_calls) = &msg.tool_calls {
                        if !tool_calls.is_empty() {
                            chat_pb.println("  Tool Calls:");
                            for (j, tool_call) in tool_calls.iter().enumerate() {
                                chat_pb.println(&format!(
                                    "    {}.{}: {} ({})",
                                    i + 1,
                                    j + 1,
                                    tool_call.function.name,
                                    tool_call.id
                                ));
                                chat_pb.println(&format!(
                                    "      Args: {}",
                                    tool_call.function.arguments
                                ));
                            }
                        }
                    }
                }
            }
            Ok(true)
        }
        ChatCommand::Limit(n) => {
            if n == 0 {
                chat_pb.println("Limit cannot be zero. Clearing history instead.");
                *messages = initialize_chat_messages(tools, opts);
            } else if messages.len() > n {
                *messages = messages.split_off(messages.len() - n);
                chat_pb.println(&format!("Chat history limited to the last {} messages.", n));
            } else {
                chat_pb.println(&format!(
                    "Chat history is already within the limit of {}.",
                    n
                ));
            }
            Ok(true)
        }
        ChatCommand::Backtrace(n) => {
            if n == 0 {
                chat_pb.println("Backtrace steps must be a positive number.");
            } else if n > messages.len() {
                chat_pb.println(&format!(
                    "Cannot go back {} steps, history has only {} messages. Clearing history.",
                    n,
                    messages.len()
                ));
                *messages = initialize_chat_messages(tools, opts);
            } else {
                messages.truncate(messages.len() - n);
                chat_pb.println(&format!("Went back {} steps in chat history.", n));
            }
            Ok(true)
        }
        ChatCommand::System(system_message) => {
            messages.push(make_message("system", system_message));
            chat_pb.println("System message added to conversation.");
            Ok(true)
        }
        ChatCommand::Message(_) => Ok(false),
        ChatCommand::Empty => Ok(true),
        ChatCommand::Invalid(error_msg) => {
            chat_pb.println(&error_msg);
            Ok(true)
        }
    }
}

fn format_tool_arguments(args_json: &str) -> String {
    let clean_args = args_json
        .replace('\n', " ")
        .replace('\r', " ")
        .replace('\t', " ");
    if clean_args.len() > 50 {
        format!("{}... ({})", &clean_args[..47], clean_args.len())
    } else {
        clean_args
    }
}

fn setup_progress_bars(
    global_multi_progress: &Arc<MultiProgress>,
) -> Result<(ProgressBar, ProgressBar), Box<dyn Error>> {
    let streaming_pb = global_multi_progress.add(ProgressBar::no_length());
    streaming_pb.set_style(ProgressStyle::with_template("{msg}")?);

    let status_pb = global_multi_progress.add(ProgressBar::new_spinner());
    status_pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan.bold} {msg:.white.bold} │ {elapsed}")?
            .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]),
    );

    Ok((streaming_pb, status_pb))
}

fn create_response_mode(streaming_pb: ProgressBar, status_pb: ProgressBar) -> ResponseMode {
    let tool_active = Arc::new(AtomicBool::new(false));
    let stream_buffer = Arc::new(std::sync::Mutex::new(String::new()));
    let previous_status = Arc::new(std::sync::Mutex::new(Option::<(String, String)>::None));

    let status_pb_progress_clone = status_pb.clone();
    let streaming_pb_progress_clone = streaming_pb.clone();
    let tool_active_stream_clone = tool_active.clone();
    let stream_buffer_clone = stream_buffer.clone();
    let streaming_pb_clone = streaming_pb.clone();
    let previous_status_clone = previous_status.clone();

    ResponseMode::Streaming {
        stream_handler: Box::new(move |chunk: &str| {
            let response_style = Style::new().cyan();
            if chunk.is_empty() {
                if let Ok(mut buffer) = stream_buffer_clone.lock() {
                    if !buffer.is_empty() {
                        let remaining_content = buffer.clone();
                        streaming_pb_clone
                            .println(response_style.apply_to(remaining_content).to_string());
                        buffer.clear();
                    }
                }
                streaming_pb_clone.finish_and_clear();
                return Ok(());
            }

            if tool_active_stream_clone.load(Ordering::Relaxed) {
                return Ok(());
            }

            if let Ok(mut buffer) = stream_buffer_clone.lock() {
                buffer.push_str(chunk);

                if buffer.contains('\n') {
                    let mut lines: Vec<&str> = buffer.split('\n').collect();
                    let remaining = lines.pop().unwrap_or("").to_string();

                    for line in lines {
                        streaming_pb_clone.println(response_style.apply_to(line).to_string());
                    }

                    *buffer = remaining.clone();
                    streaming_pb_clone.set_message(remaining);
                } else {
                    let current_buffer = buffer.clone();
                    streaming_pb_clone.set_message(current_buffer);
                }
            }

            Ok(())
        }),
        progress_handler: Box::new(move |progress_info: &ProgressInfo| {
            let elapsed_secs = progress_info.elapsed_ms as f64 / 1000.0;

            let is_tool_status = |status_name: &str| {
                matches!(
                    status_name,
                    "ToolAccumulating" | "ToolExecuting" | "ToolComplete"
                )
            };

            let print_previous_and_update = |status_name: &str, message: String| -> bool {
                if let Ok(mut prev_status) = previous_status_clone.lock() {
                    let status_changed =
                        if let Some((prev_name, _prev_message)) = prev_status.as_ref() {
                            prev_name != status_name
                        } else {
                            true
                        };
                    *prev_status = Some((status_name.to_string(), message));
                    status_changed
                } else {
                    true
                }
            };

            match &progress_info.status {
                StatusUpdate::Thinking => {
                    let message = "Thinking".to_string();
                    let status_changed = print_previous_and_update("Thinking", message);

                    if status_changed {
                        status_pb_progress_clone.reset_elapsed();
                    }

                    status_pb_progress_clone.set_style(
                        ProgressStyle::with_template(
                            "{spinner:.blue.bold} {msg:.blue} │ {elapsed}",
                        )?
                        .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]),
                    );
                    status_pb_progress_clone.set_message("Thinking");
                    status_pb_progress_clone.enable_steady_tick(Duration::from_millis(500));
                }
                StatusUpdate::ToolAccumulating { name, arguments } => {
                    let clean_args = arguments
                        .replace('\n', " ")
                        .replace('\r', " ")
                        .replace('\t', " ");

                    let formatted_args = if clean_args.len() > 50 {
                        format!("{} (length: {})", &clean_args[..47], clean_args.len())
                    } else {
                        clean_args
                    };

                    let message = format!("Preparing {}({})", name, formatted_args);
                    let status_changed = print_previous_and_update("ToolAccumulating", message);

                    if status_changed {
                        status_pb_progress_clone.reset_elapsed();
                    }

                    status_pb_progress_clone.set_style(
                        ProgressStyle::with_template(
                            "{spinner:.cyan.bold} {msg:.white.bold} │ {elapsed}",
                        )?
                        .tick_strings(&["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]),
                    );
                    status_pb_progress_clone
                        .set_message(format!("Preparing {}({})", name, formatted_args));
                    status_pb_progress_clone.enable_steady_tick(Duration::from_millis(500));

                    streaming_pb_progress_clone
                        .set_message(format!("Preparing {}({})", name, formatted_args));
                }
                StatusUpdate::ToolStart { name, arguments } => {
                    let formatted_args = format_tool_arguments(arguments);
                    let message = format!("Starting {}({})", name, formatted_args);
                    let status_changed = print_previous_and_update("ToolStart", message);

                    if status_changed {
                        status_pb_progress_clone.reset_elapsed();
                    }

                    status_pb_progress_clone.set_style(
                        ProgressStyle::with_template(
                            "{spinner:.yellow.bold} {msg:.white} │ {elapsed}",
                        )?
                        .tick_strings(&["🚀", "⚡", "✨", "🔧", "⚙️", "🛠️", "🎯", "🚀"]),
                    );
                    status_pb_progress_clone
                        .set_message(format!("Starting {}({})", name, formatted_args));
                    status_pb_progress_clone.enable_steady_tick(Duration::from_millis(500));
                }
                StatusUpdate::ToolExecuting { name, arguments } => {
                    let formatted_args = format_tool_arguments(arguments);
                    let message = format!("Processing {}({})", name, formatted_args);
                    let status_changed = print_previous_and_update("ToolExecuting", message);

                    if status_changed {
                        status_pb_progress_clone.reset_elapsed();
                    }

                    status_pb_progress_clone.set_style(
                        ProgressStyle::with_template(
                            "{spinner:.red.bold} {msg:.white} │ {elapsed}",
                        )?
                        .tick_strings(&["⣾⣿", "⣽⣿", "⣻⣿", "⢿⣿", "⡿⣿", "⣟⣿", "⣯⣿", "⣷⣿"]),
                    );
                    status_pb_progress_clone
                        .set_message(format!("Processing {}({})", name, formatted_args));
                    status_pb_progress_clone.enable_steady_tick(Duration::from_millis(500));
                }
                StatusUpdate::ToolComplete {
                    name,
                    arguments,
                    duration_ms,
                } => {
                    let duration_secs = *duration_ms as f64 / 1000.0;
                    tool_active.store(false, Ordering::Relaxed);

                    let formatted_args = format_tool_arguments(arguments);
                    let completion_msg = format!(
                        "✅ {}({}) │ completed in {:.1}s",
                        name, formatted_args, duration_secs
                    );
                    status_pb_progress_clone.println(&completion_msg);
                    streaming_pb_progress_clone.set_message("");
                }
                StatusUpdate::StreamProcessing {
                    bytes_read,
                    chunks_processed,
                    latest_content,
                } => {
                    let content_preview = if latest_content.len() > 30 {
                        format!("{}...", &latest_content[..27])
                    } else {
                        latest_content.clone()
                    };
                    let message = format!(
                        "Streaming {} bytes, {} chunks - \"{}\"",
                        bytes_read,
                        chunks_processed,
                        content_preview.trim()
                    );
                    let status_changed = print_previous_and_update("StreamProcessing", message);

                    if status_changed {
                        status_pb_progress_clone.reset_elapsed();
                    }

                    status_pb_progress_clone.set_style(
                        ProgressStyle::with_template(
                            "{spinner:.blue.bold} {msg:.blue} │ {elapsed}",
                        )?
                        .tick_strings(&[
                            "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂",
                        ]),
                    );
                    status_pb_progress_clone.set_message(format!(
                        "Streaming {} bytes, {} chunks - \"{}\"",
                        bytes_read,
                        chunks_processed,
                        content_preview.trim()
                    ));
                    status_pb_progress_clone.enable_steady_tick(Duration::from_millis(500));
                }
                StatusUpdate::Continuing => {
                    let message = "Continuing conversation".to_string();
                    let status_changed = print_previous_and_update("Continuing", message);

                    if status_changed {
                        status_pb_progress_clone.reset_elapsed();
                    }

                    status_pb_progress_clone.set_style(
                        ProgressStyle::with_template(
                            "{spinner:.magenta.bold} {msg:.white} │ {elapsed}",
                        )?
                        .tick_strings(&["💭", "🧠", "💡", "🔄", "🔀", "💫", "🌟", "💭"]),
                    );
                    status_pb_progress_clone.set_message("Continuing conversation");
                    status_pb_progress_clone.enable_steady_tick(Duration::from_millis(500));
                }
                StatusUpdate::Complete { usage } => {
                    if let Ok(mut prev_status) = previous_status_clone.lock() {
                        if let Some((prev_name, prev_message)) = prev_status.take() {
                            let indent = if is_tool_status(&prev_name) { "  " } else { "" };
                            status_pb_progress_clone
                                .println(&format!("{}{}", indent, prev_message));
                        }
                    }

                    if let Some(usage) = usage {
                        let mut parts = Vec::new();

                        if let Some(input_tokens) = usage.prompt_tokens {
                            parts.push(format!("Input: {} tokens", input_tokens));
                        }

                        if let Some(output_tokens) = usage.completion_tokens {
                            parts.push(format!("Output: {} tokens", output_tokens));
                        }

                        if let Some(total_tokens) = usage.total_tokens {
                            parts.push(format!("Total: {}", total_tokens));
                        }

                        if !parts.is_empty() {
                            let completion_msg = format!(
                                "🎯 Response complete │ {} │ {:.1}s",
                                parts.join(" → "),
                                elapsed_secs
                            );
                            status_pb_progress_clone.println(&completion_msg);
                        } else {
                            let completion_msg =
                                format!("🎯 Response complete │ {:.1}s", elapsed_secs);
                            status_pb_progress_clone.println(&completion_msg);
                        }
                    } else {
                        let completion_msg = format!("🎯 Response complete │ {:.1}s", elapsed_secs);
                        status_pb_progress_clone.println(&completion_msg);
                    }
                }
            }

            Ok(())
        }),
    }
}

fn execute_ai_request(
    messages: Vec<Message>,
    tools: &ToolsCollection,
    openai_opts: &openai::Opts,
    mode: ResponseMode,
    tool_context: &ToolContext,
    ctrl_c_rx: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
    signal_handler_active: &Arc<AtomicBool>,
    status_pb: &ProgressBar,
    streaming_pb: &ProgressBar,
    multi_progress: &Arc<MultiProgress>,
    chat_pb: &ProgressBar,
) -> Result<OpenAIResponse, Box<dyn Error>> {
    signal_handler_active.store(true, Ordering::Relaxed);

    let response = match post_request_with_mode(
        messages,
        tools,
        openai_opts,
        mode,
        tool_context,
        ctrl_c_rx.clone(),
    ) {
        Ok(response) => {
            signal_handler_active.store(false, Ordering::Relaxed);
            if let Some(ref ctrl_c_rx) = ctrl_c_rx {
                if let Ok(receiver) = ctrl_c_rx.lock() {
                    while receiver.try_recv().is_ok() {}
                }
            }
            response
        }
        Err(e) => {
            signal_handler_active.store(false, Ordering::Relaxed);
            if let Some(ref ctrl_c_rx) = ctrl_c_rx {
                if let Ok(receiver) = ctrl_c_rx.lock() {
                    while receiver.try_recv().is_ok() {}
                }
            }

            status_pb.finish_and_clear();
            streaming_pb.finish_and_clear();
            multi_progress.clear()?;

            if let Some(_interrupted) = e.downcast_ref::<InterruptedError>() {
                chat_pb.println("Operation interrupted. Type your next message or \\quit to exit.");
                return Err(e);
            } else {
                return Err(e);
            }
        }
    };

    status_pb.finish_and_clear();
    streaming_pb.finish_and_clear();
    multi_progress.clear()?;

    Ok(response)
}

fn execute_scheduled_command(
    command: &str,
    tools: &ToolsCollection,
    db: &Option<Arc<Mutex<rusqlite::Connection>>>,
) -> Option<(Message, Message)> {
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(command);
    let (tool_name, arguments) = match parsed {
        Ok(obj) => {
            let name = obj
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args = match obj.get("arguments") {
                Some(a) if a.is_string() => a.as_str().unwrap().to_string(),
                Some(a) => a.to_string(),
                None => "{}".to_string(),
            };
            (name, args)
        }
        Err(_) => return None,
    };

    if tool_name.is_empty() {
        return None;
    }

    let call_id = format!("scheduled-task-{}", chrono::Utc::now().timestamp_millis());

    let tc = ToolCall {
        index: None,
        id: call_id.clone(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: tool_name.clone(),
            arguments,
        },
    };

    let mut tool_context = ToolContext::new(|_: &str| {});
    tool_context.db = db.clone();

    let tool_msg = match tool_call(tools, &tc, &tool_context) {
        Ok(msg) => msg,
        Err(e) => Message {
            role: "tool".to_string(),
            content: Some(format!("error: {}", e)),
            tool_call_id: Some(call_id),
            name: Some(tool_name),
            tool_calls: None,
        },
    };

    let assistant_msg = Message {
        role: "assistant".to_string(),
        content: None,
        tool_call_id: None,
        name: None,
        tool_calls: Some(vec![tc]),
    };

    Some((assistant_msg, tool_msg))
}

fn scheduler_loop(
    db: Arc<Mutex<rusqlite::Connection>>,
    tools: Arc<ToolsCollection>,
    tx: mpsc::Sender<(String, Message, Message)>,
) {
    let db_opt = Some(db.clone());
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let conn = match db.lock() {
            Ok(c) => c,
            Err(_) => continue,
        };
        let tasks = match db::get_pending_tasks(&conn) {
            Ok(t) => t,
            Err(_) => continue,
        };
        drop(conn);

        for task in tasks {
            let command = if task.command.is_empty() {
                task.description.clone()
            } else {
                task.command.clone()
            };

            if let Some((assistant_msg, tool_msg)) =
                execute_scheduled_command(&command, &tools, &db_opt)
            {
                let _ = tx.send((command, assistant_msg, tool_msg));
            }

            if let Ok(conn) = db.lock() {
                let _ = db::mark_task_executed(
                    &conn,
                    task.id,
                    &task.task_type,
                    task.cron_expression.as_deref(),
                    task.max_runs,
                );
            }
        }
    }
}

/// Interactive session
fn chat_command(
    opts: &Opts,
    global_multi_progress: Arc<MultiProgress>,
    db: Option<Arc<Mutex<rusqlite::Connection>>>,
) -> Result<(), Box<dyn Error>> {
    debug!("Executing chat command");

    // Create a persistent streaming progress bar for chat messages
    let chat_pb = global_multi_progress.add(ProgressBar::no_length());
    chat_pb.set_style(ProgressStyle::with_template("{msg}")?);

    // Create rustyline editor with history
    let mut rl = DefaultEditor::new()?;

    let tools = match opts.no_tools {
        true => {
            debug!("Tools are disabled");
            ToolsCollection::new()
        }
        false => {
            debug!("Initializing tools for AI request");
            initialize_tools(opts.unsafe_tools)
        }
    };

    let mut messages = initialize_chat_messages(&tools, opts);

    let model = opts.model.clone().unwrap_or(DEFAULT_MODEL.to_string());
    debug!("Using model: {}", model);

    let parameters = parse_parameters(&opts.parameter)?;

    let endpoint = opts
        .endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    let normalized_endpoint = openai::normalize_endpoint(&endpoint);

    let openai_opts = openai::Opts {
        max_tokens: opts.max_tokens,
        model: model,
        endpoint: normalized_endpoint,
        tool_choice: opts.tool_choice.clone(),
        api_key: opts.api_key.clone(),
        max_retries: None,
        retry_base_delay_secs: None,
        parameters,
    };

    let (ctrl_c_tx, ctrl_c_rx) = mpsc::channel();
    let ctrl_c_rx = Arc::new(Mutex::new(ctrl_c_rx));

    let signal_handler_active = Arc::new(AtomicBool::new(false));

    let ctrl_c_tx_clone = ctrl_c_tx.clone();
    let signal_handler_active_clone = signal_handler_active.clone();
    ctrlc::set_handler(move || {
        if signal_handler_active_clone.load(Ordering::Relaxed) {
            debug!("Received SIGINT (Ctrl-C) - sending interrupt signal");
            let _ = ctrl_c_tx_clone.send(());
        }
    })
    .expect("Error setting up Ctrl-C handler");

    let (task_tx, task_rx) = mpsc::channel::<(String, Message, Message)>();
    if let Some(ref scheduler_db) = db {
        let scheduler_db = scheduler_db.clone();
        let scheduler_tools = Arc::new(tools.clone());
        std::thread::spawn(move || {
            scheduler_loop(scheduler_db, scheduler_tools, task_tx);
        });
    }

    let (input_tx, input_rx) = mpsc::channel::<Result<String, rustyline::error::ReadlineError>>();

    std::thread::spawn(move || {
        loop {
            let result = rl.readline("> ");
            let is_eof = matches!(result, Err(rustyline::error::ReadlineError::Eof));
            if let Ok(ref line) = result {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    let _ = rl.add_history_entry(trimmed);
                }
            }
            let _ = input_tx.send(result);
            if is_eof {
                break;
            }
        }
    });

    loop {
        global_multi_progress.clear()?;

        while let Ok((command, assistant_msg, tool_msg)) = task_rx.try_recv() {
            chat_pb.println(&format!("Scheduled task fired: {}", command));
            if let Some(ref output) = tool_msg.content {
                let display = if let Ok(obj) = serde_json::from_str::<serde_json::Value>(output) {
                    let mut parts = Vec::new();
                    if let Some(stdout) = obj.get("stdout").and_then(|v| v.as_str()) {
                        let s = stdout.trim();
                        if !s.is_empty() {
                            parts.push(s.to_string());
                        }
                    }
                    if let Some(stderr) = obj.get("stderr").and_then(|v| v.as_str()) {
                        let s = stderr.trim();
                        if !s.is_empty() {
                            parts.push(format!("stderr: {}", s));
                        }
                    }
                    if parts.is_empty() {
                        output.trim().to_string()
                    } else {
                        parts.join("\n")
                    }
                } else {
                    output.trim().to_string()
                };
                chat_pb.println(&display);
            }
            messages.push(assistant_msg);
            messages.push(tool_msg);
        }

        let line = match input_rx.recv_timeout(Duration::from_millis(200)) {
            Ok(Ok(line)) => line.trim().to_string(),
            Ok(Err(rustyline::error::ReadlineError::Interrupted)) => {
                continue;
            }
            Ok(Err(rustyline::error::ReadlineError::Eof)) => {
                return Ok(());
            }
            Ok(Err(err)) => {
                return Err(Box::new(err));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Ok(());
            }
        };

        debug!("User input: '{}' (length: {})", line, line.len());

        let command = parse_chat_command(&line);
        match handle_chat_command(command, &mut messages, &tools, opts, &chat_pb)? {
            true => continue, // Continue the loop for special commands
            false => {
                // Handle quit command or process user message
                if let ChatCommand::Quit = parse_chat_command(&line) {
                    return Ok(());
                }
                if let ChatCommand::Message(user_message) = parse_chat_command(&line) {
                    let message_count_before = messages.len();
                    messages.push(make_message("user", user_message));
                    debug!(
                        "Added user message. Message count: {} -> {}",
                        message_count_before,
                        messages.len()
                    );

                    let multi_progress = global_multi_progress.clone();
                    let (streaming_pb, status_pb) = setup_progress_bars(&multi_progress)?;
                    let mode = create_response_mode(streaming_pb.clone(), status_pb.clone());

                    status_pb.set_message("Connecting");
                    status_pb.enable_steady_tick(Duration::from_millis(500));

                    let streaming_pb_for_context = streaming_pb.clone();
                    let mut tool_context = ToolContext::new(move |msg: &str| {
                        streaming_pb_for_context.println(msg);
                    });
                    tool_context.db = db.clone();

                    match execute_ai_request(
                        messages.clone(),
                        &tools,
                        &openai_opts,
                        mode,
                        &tool_context,
                        Some(ctrl_c_rx.clone()),
                        &signal_handler_active,
                        &status_pb,
                        &streaming_pb,
                        &multi_progress,
                        &chat_pb,
                    ) {
                        Ok(response) => {
                            debug!(
                                "Response history contains: {} messages",
                                response.history.len()
                            );
                            messages = response.history;
                            debug!("After updating history: {} messages", messages.len());
                        }
                        Err(e) => {
                            if e.downcast_ref::<InterruptedError>().is_some() {
                                continue;
                            } else {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ModelInfo and ModelsApiResponse structs are removed from here.

/// Handles the listing of models by calling the openai module.
fn list_models_command(opts: &Opts) -> Result<(), Box<dyn Error>> {
    let base_endpoint = opts
        .endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());

    let models_endpoint = if base_endpoint.ends_with("/chat/completions") {
        base_endpoint.replace("/chat/completions", "/models")
    } else if base_endpoint.ends_with("/") {
        format!("{}models", base_endpoint)
    } else {
        format!("{}/models", base_endpoint)
    };

    match list_models_from_endpoint(&models_endpoint, opts.api_key.as_ref()) {
        Ok(models) => {
            if models.is_empty() {
                println!("No models found.");
            } else {
                let source = if opts.endpoint.is_some() {
                    "custom endpoint"
                } else {
                    "default endpoint"
                };
                println!("Available models from {}:", source);
                let mut table = Table::new();
                table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
                table.set_titles(Row::new(vec![
                    Cell::new("ID"),
                    Cell::new("Name"),
                    Cell::new("Context Length"),
                    Cell::new("Prompt_USD/1M"),
                    Cell::new("Compl_USD/1M"),
                    Cell::new("Supported Parameters"),
                ]));

                for model in models {
                    let prompt_price_str = model
                        .pricing
                        .as_ref()
                        .map(|p| p.prompt.as_str())
                        .unwrap_or("N/A");
                    let completion_price_str = model
                        .pricing
                        .as_ref()
                        .map(|p| p.completion.as_str())
                        .unwrap_or("N/A");

                    let prompt_price = match prompt_price_str.parse::<f64>() {
                        Ok(p) => format!("{:.6}", p),
                        Err(_) => prompt_price_str.to_string(),
                    };
                    let completion_price = match completion_price_str.parse::<f64>() {
                        Ok(c) => format!("{:.6}", c),
                        Err(_) => completion_price_str.to_string(),
                    };
                    let supported_parameters = model
                        .supported_parameters
                        .unwrap_or_else(|| vec![])
                        .join(",");

                    table.add_row(Row::new(vec![
                        Cell::new(&model.id),
                        Cell::new(&model.name.unwrap_or("".to_string())),
                        Cell::new(&model.context_length.unwrap_or(0).to_string()),
                        Cell::new(&prompt_price),
                        Cell::new(&completion_price),
                        Cell::new(&supported_parameters),
                    ]));
                }
                table.printstd();
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[derive(Parser, Debug, Serialize, Deserialize)]
#[clap(version = env!("CARGO_PKG_VERSION"))]
#[serde(default)]
struct Opts {
    #[clap(short = 'c', long = "config")]
    #[serde(skip)]
    /// Path to JSON configuration file
    ///
    /// If not specified, will automatically use 'config.json'
    /// from current directory if it exists.
    ///
    /// Example config file:
    /// {
    ///   "model": "granite",
    ///   "api_key": "~/.path/to/key",
    ///   "parameters": ["temperature=0.7"]
    /// }
    ///
    /// CLI arguments override config file values.
    config: Option<String>,
    #[clap(short, long)]
    /// Override the maximum number of tokens to generate
    max_tokens: Option<u32>,
    #[clap(long)]
    /// Specify the AI model to use
    model: Option<String>,
    #[clap(long)]
    /// Override the endpoint URL to use
    endpoint: Option<String>,
    #[clap(long)]
    /// Inhibit usage of any tool
    no_tools: bool,
    /// Enable unsafe tools
    #[clap(long)]
    unsafe_tools: bool,
    #[clap(long)]
    /// Control when tools are used: "auto" (default), "none", "required"
    tool_choice: Option<String>,
    #[clap(long)]
    /// Override the file path to read API key from
    api_key: Option<String>,
    #[clap(long)]
    /// Set model parameters in NAME=VALUE format (e.g., --parameter temperature=0.7 --parameter top_p=0.9)
    parameter: Vec<String>,
    #[clap(long)]
    /// Skip adding any system prompts
    no_system_prompts: bool,
    #[clap(long)]
    /// Path to the SQLite database file for persistent storage (agents, tasks, memory)
    db_path: Option<String>,

    #[clap(subcommand)]
    #[serde(skip)]
    command: CliCommand,

    #[clap()]
    #[serde(skip)]
    args: Vec<String>,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            config: None,
            max_tokens: None,
            model: None,
            endpoint: None,
            no_tools: false,
            unsafe_tools: false,
            tool_choice: None,
            api_key: None,
            parameter: Vec::new(),
            no_system_prompts: false,
            db_path: None,
            command: CliCommand::Chat {},
            args: Vec::new(),
        }
    }
}

impl Opts {
    /// Load configuration from a JSON file
    fn load_from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        debug!("Loading configuration from file: {}", path);

        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file '{}': {}", path, e))?;

        let config: Opts = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse config file '{}': {}", path, e))?;

        debug!("Successfully loaded configuration from {}", path);
        Ok(config)
    }

    /// Merge this config with CLI options, giving precedence to CLI options
    fn merge_with_config(&mut self, config: Opts) {
        debug!("Merging configuration file with CLI options");

        // Only use config values if CLI didn't provide them
        if self.max_tokens.is_none() {
            self.max_tokens = config.max_tokens;
        }

        if self.model.is_none() {
            self.model = config.model;
        }

        if self.endpoint.is_none() {
            self.endpoint = config.endpoint;
        }

        if !self.no_tools && config.no_tools {
            self.no_tools = true;
        }

        if !self.unsafe_tools && config.unsafe_tools {
            self.unsafe_tools = true;
        }

        if self.tool_choice.is_none() {
            self.tool_choice = config.tool_choice;
        }

        if self.api_key.is_none() {
            self.api_key = config.api_key;
        }

        // Merge parameters - config parameters are added first, then CLI parameters
        if !config.parameter.is_empty() {
            let mut merged_params = config.parameter;
            merged_params.extend(self.parameter.clone());
            self.parameter = merged_params;
        }

        if !self.no_system_prompts && config.no_system_prompts {
            self.no_system_prompts = true;
        }

        if self.db_path.is_none() {
            self.db_path = config.db_path;
        }

        debug!("Configuration merge completed");
    }
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Prioritize the issues and pull requests happened in the last DAYS
    Prioritize {
        /// Maximum age in days for the issue or pull request
        #[clap(long)]
        days: Option<u64>,
        /// Repository
        repo: Vec<String>,
    },
    /// Analyze the issues and pull requests happened in the last DAYS
    Analyze {
        /// Maximum age in days for the issue or pull request
        #[clap(long)]
        days: Option<u64>,
        /// Repository
        repo: Vec<String>,
    },
    /// Triage a specific issue
    Triage {
        /// Repository
        repo: String,
        /// Issue number
        issue: u64,
    },

    /// Review a pull request
    Review {
        /// Repository
        repo: String,
        /// PR number
        pr: u64,
    },

    /// Pass a request to the AI model and print its response
    Prompt {
        /// Prompt command to pass to the AI model
        prompt: String,
        /// List of files that are loaded and used as system context
        files: Vec<String>,
    },

    /// Interactive session
    Chat {},

    /// List available models from the configured endpoint
    Models {},
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize environment logger with custom configuration
    let env = Env::new()
        .filter_or("RUST_LOG", "warning")
        .write_style_or("LOG_STYLE", "always");

    // Set up indicatif log bridge to prevent logging interference with progress bars
    let logger = env_logger::Builder::from_env(env).build();
    let global_multi_progress = Arc::new(MultiProgress::new());
    LogWrapper::new((*global_multi_progress).clone(), logger).try_init()?;

    // Parse command line arguments
    let mut opts = Opts::parse();
    debug!("Command line options parsed");

    // Load and merge configuration file
    let config_path = match &opts.config {
        Some(path) => Some(path.clone()),
        None => {
            // Check for default config.json in current directory
            let default_config = "config.json";
            if std::path::Path::new(default_config).exists() {
                debug!("Found default config file: {}", default_config);
                Some(default_config.to_string())
            } else {
                None
            }
        }
    };

    if let Some(config_file) = config_path {
        match Opts::load_from_file(&config_file) {
            Ok(config) => {
                debug!("Loaded configuration file: {}", config_file);
                opts.merge_with_config(config);
            }
            Err(e) => {
                return Err(format!("Configuration file error: {}", e).into());
            }
        }
    }

    // Reset the model to use if an endpoint was provided
    if opts.model.is_none() && opts.endpoint.is_some() {
        opts.model = Some("".to_string());
    }

    let db_connection = if let Some(ref db_path) = opts.db_path {
        debug!("Opening SQLite database at: {}", db_path);
        let conn = rusqlite::Connection::open(db_path)?;
        db::initialize_db(&conn)?;
        Some(Arc::new(Mutex::new(conn)))
    } else {
        debug!("No db_path configured, database tools will be unavailable");
        None
    };

    // Execute the chosen command
    let result = match &opts.command {
        CliCommand::Analyze { days, repo } => {
            analyze_repos(&repo, *days, &opts, db_connection.clone())
        }
        CliCommand::Prioritize { days, repo } => {
            prioritize_repos(&repo, *days, &opts, db_connection.clone())
        }
        CliCommand::Triage { repo, issue } => {
            triage_issue(&repo, *issue, &opts, db_connection.clone())
        }
        CliCommand::Review { repo, pr } => {
            review_pull_request(&repo, *pr, &opts, db_connection.clone())
        }
        CliCommand::Prompt { prompt, files } => {
            prompt_command(&prompt, &files, &opts, db_connection.clone())
        }
        CliCommand::Chat {} => {
            chat_command(&opts, global_multi_progress.clone(), db_connection.clone())
        }
        CliCommand::Models {} => list_models_command(&opts),
    };

    result
}
