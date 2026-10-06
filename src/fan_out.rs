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

//! The `fan_out` tool: runs the same task for many items at once, each in
//! its own worker agent, and returns all their results together in one
//! tool result - unlike `spawn_agent`, whose results arrive later, one
//! message at a time.
//!
//! Workers only get read-only tools unless others are asked for by name:
//! workers writing to the same files concurrently could overwrite each
//! other's changes.

use serde::Deserialize;
use std::collections::VecDeque;
use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use crate::openai::{
    self, ContextLengthError, InterruptedError, ProgressInfo, ResponseMode, StatusUpdate,
    ToolsCollection, make_message, post_request_with_mode,
};
use crate::{ActivityRecorder, SubAgentContext};
use faber::ToolContext;
use faber::agent_io::{AgentEvent, AgentOutput, EventLog};
use faber::db::AgentConfig;

const MAX_ITEMS: usize = 1000;
/// Workers are cheap threads; how many requests actually reach the model
/// at once is up to `max_parallel_requests`, which applies to the whole
/// process.
const DEFAULT_MAX_PARALLEL: usize = 32;
const MAX_PARALLEL: usize = 256;

/// How many times a worker shortens its tool results to recover from a
/// context overflow before giving up.
const MAX_TOOL_RESULT_SHRINKS: usize = 3;

/// Smallest share of the output budget any one item's result gets.
const MIN_RESULT_CHARS: usize = 2000;

/// What workers get when `tools` isn't given: tools that only read.
const READ_ONLY_TOOLS: &[&str] = &[
    "read_file",
    "glob",
    "grep",
    "lsp",
    "fetch_web_content",
    "github_issue",
    "github_pull_request",
    "kb_search",
    "kb_read",
    "kb_list",
];

/// Tools a worker can never have: no nested agents, and no plan - workers
/// act as their caller, so they'd overwrite its plan.
const NEVER_FOR_WORKERS: &[&str] = &["spawn_agent", "fan_out", "plan_update", "plan_get"];

const WORKER_INSTRUCTIONS: &str = "You are a worker in a fan-out: the same task is being run for many items in parallel, \
and you handle exactly one of them. Work only on your item. Nobody will answer questions, so don't ask any. \
Finish with a concise, self-contained result - it's all the coordinating agent will see of your work.";

#[derive(Deserialize)]
struct Params {
    #[serde(deserialize_with = "item_texts")]
    items: Vec<String>,
    prompt: String,
    #[serde(default)]
    max_parallel: Option<usize>,
    #[serde(default)]
    tools: Option<Vec<String>>,
    #[serde(default)]
    result_schema: Option<serde_json::Value>,
    #[serde(default)]
    context: Option<crate::Handoff>,
    #[serde(default)]
    profile: Option<String>,
}

/// The items as text: a string as it is, anything else - an object
/// describing the item, a number - as its JSON, rather than refusing it.
fn item_texts<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    let items: Vec<serde_json::Value> = Deserialize::deserialize(d)?;
    Ok(items
        .into_iter()
        .map(|item| match item {
            serde_json::Value::String(s) => s,
            other => other.to_string(),
        })
        .collect())
}

/// The prompt for one item: `{item}` in `template` replaced by it, or the
/// item appended if there's no placeholder.
pub(crate) fn expand_prompt(template: &str, item: &str) -> String {
    if template.contains("{item}") {
        template.replace("{item}", item)
    } else {
        format!("{}\n\nItem: {}", template, item)
    }
}

/// The tools workers get: `requested` by name (each must exist and be
/// allowed), or else every available read-only tool.
pub(crate) fn worker_tools(
    available: &ToolsCollection,
    requested: Option<&[String]>,
) -> Result<ToolsCollection, String> {
    let names: Vec<String> = match requested {
        None => READ_ONLY_TOOLS
            .iter()
            .filter(|n| available.contains_key(**n))
            .map(|n| n.to_string())
            .collect(),
        Some(names) => {
            for name in names {
                if NEVER_FOR_WORKERS.contains(&name.as_str()) {
                    return Err(format!("workers can't use {}", name));
                }
                if !available.contains_key(name) {
                    return Err(format!("unknown tool '{}'", name));
                }
            }
            names.to_vec()
        }
    };
    Ok(names
        .into_iter()
        .map(|n| {
            let tool = available[&n].clone();
            (n, tool)
        })
        .collect())
}

/// All the results, in item order, each capped to its share of
/// `budget_chars`.
pub(crate) fn format_results(
    items: &[String],
    results: &[Option<Result<String, String>>],
    budget_chars: usize,
) -> String {
    let per_item = (budget_chars / items.len().max(1)).max(MIN_RESULT_CHARS);
    let mut out = String::new();
    for (i, (item, result)) in items.iter().zip(results).enumerate() {
        let body = match result {
            Some(Ok(text)) => openai::truncate_tool_output(text.trim(), per_item),
            Some(Err(e)) => format!("Error: {}", e),
            None => "Not run.".to_string(),
        };
        out.push_str(&format!("## {}. {}\n{}\n\n", i + 1, item, body));
    }
    out
}

/// The tool context a worker runs with: quiet, with the caller's database
/// and context window, and acting as the caller - e.g. seeing its private
/// knowledge base notes. No MCP tools, and no sub-agent machinery.
fn worker_context(ctx: &ToolContext) -> ToolContext {
    let mut worker_ctx = ToolContext::new(|_: &str| {});
    worker_ctx.db = ctx.db.clone();
    worker_ctx.agent_name = ctx.agent_name.clone();
    worker_ctx.context_window = ctx.context_window;
    // Its tools are the caller's own (see `worker_tools`), unsafe or not.
    worker_ctx.unsafe_tools = ctx.unsafe_tools;
    worker_ctx.cwd = ctx.cwd.clone();
    worker_ctx.task_id = ctx.task_id;
    worker_ctx
}

/// A worker's progress handler: the status bar, and its agent's activity.
fn worker_progress(
    sa_ctx: &SubAgentContext,
    status_key: &str,
    activity: Option<Arc<ActivityRecorder>>,
) -> Box<dyn Fn(&ProgressInfo) -> Result<(), Box<dyn Error>>> {
    let status_bar = sa_ctx.status_bar.clone();
    let key = status_key.to_string();
    Box::new(move |progress: &ProgressInfo| {
        if let Some(activity) = &activity {
            activity.follow(&progress.status);
        }
        let status = match &progress.status {
            StatusUpdate::Thinking => "Thinking".to_string(),
            StatusUpdate::ToolStart { name, .. } => format!("Running {}", name),
            StatusUpdate::SendingRequest { .. } => "Waiting for response".to_string(),
            StatusUpdate::WaitingForSlot => "Waiting for a free request slot".to_string(),
            StatusUpdate::RateLimited { until } => {
                format!(
                    "Rate limited, retrying at {}",
                    openai::format_retry_time(*until)
                )
            }
            _ => return Ok(()),
        };
        status_bar.set_agent_status(&key, &status, false);
        Ok(())
    })
}

/// Releases a worker's agent however its run ends.
struct ReleaseWhenDone(Option<(Arc<dyn faber::db_backend::DbBackend>, String, String)>);

impl Drop for ReleaseWhenDone {
    fn drop(&mut self) {
        if let Some((db, agent, session)) = &self.0 {
            let _ = db.release_agent(agent, session);
        }
    }
}

/// Runs one worker to completion. `cancel` gets a message on Ctrl-C.
/// With a database, it runs as `agent`, a sub-agent of the caller made
/// for it - so it can be seen, and followed, like any other.
fn run_worker(
    sa_ctx: &SubAgentContext,
    ctx: &ToolContext,
    tools: &ToolsCollection,
    config: &AgentConfig,
    status_key: &str,
    agent: &str,
    prompt: String,
    handoff: Option<&crate::Message>,
    schema: Option<&serde_json::Value>,
    cancel: mpsc::Receiver<()>,
) -> Result<String, Box<dyn Error>> {
    let mut worker_ctx = worker_context(ctx);
    worker_ctx.context_window = config.context_window.or(ctx.context_window);
    let db = match (&ctx.db, &ctx.agent_name) {
        (Some(db), Some(caller)) => {
            crate::make_worker_agent(db.as_ref(), agent, caller, ctx, &sa_ctx.session_id)?;
            worker_ctx.agent_name = Some(agent.to_string());
            Some(db.clone())
        }
        _ => None,
    };
    let _release = ReleaseWhenDone(
        db.clone()
            .map(|db| (db, agent.to_string(), sa_ctx.session_id.clone())),
    );
    let events = db
        .as_ref()
        .map(|db| EventLog::new(db.clone(), agent, ctx.task_id));
    let activity = db
        .as_ref()
        .map(|db| ActivityRecorder::new(db.clone(), agent));
    if let Some(events) = &events {
        events.emit(AgentEvent::Input {
            text: prompt.clone(),
        });
    }
    if let Some(activity) = &activity {
        activity.set("thinking");
    }
    let slot = Arc::new(Mutex::new(None));
    worker_ctx.result_slot = Some(slot.clone());
    worker_ctx.result_schema = schema.cloned();
    // Streamed, although nothing is shown, so Ctrl-C is noticed between
    // chunks rather than only once a whole response has arrived.
    let mode = || {
        crate::recorded_mode(
            ResponseMode::Streaming {
                stream_handler: Box::new(|_: &str| Ok(())),
                reasoning_handler: Box::new(|_: &str| Ok(())),
                progress_handler: worker_progress(sa_ctx, status_key, activity.clone()),
            },
            &events,
        )
    };
    let cancel = Some(Arc::new(Mutex::new(cancel)));
    // Its tools stop with it: a running command, say.
    worker_ctx.interrupt = cancel.clone();
    let mut messages = vec![
        make_message("system", WORKER_INSTRUCTIONS.to_string()),
        crate::working_directory_note(&worker_ctx.cwd()),
    ];
    if let Some(system_prompt) = &config.system_prompt {
        messages.push(make_message("system", system_prompt.clone()));
    }
    messages.extend(handoff.cloned());
    messages.push(make_message(
        "user",
        crate::delegated_prompt(&prompt, schema),
    ));
    // Like the chat, recover from a context overflow by shortening tool
    // results (keeping the worker's progress) and carrying on.
    let run_once = |mut messages: Vec<crate::Message>| {
        let mut shrinks = 0;
        loop {
            let result = post_request_with_mode(
                messages,
                tools,
                &sa_ctx.opts,
                mode(),
                &worker_ctx,
                cancel.clone(),
            );
            let Err(e) = &result else { return result };
            if e.downcast_ref::<InterruptedError>().is_some() {
                return Err(
                    Box::new(InterruptedError::new("Operation interrupted by user"))
                        as Box<dyn Error>,
                );
            }
            let shrunk = e
                .downcast_ref::<ContextLengthError>()
                .filter(|_| shrinks < MAX_TOOL_RESULT_SHRINKS)
                .and_then(|overflow| {
                    openai::shrink_tool_results(&overflow.history, &overflow.message)
                });
            match shrunk {
                Some(shrunk) => {
                    shrinks += 1;
                    messages = shrunk;
                }
                None => return result,
            }
        }
    };
    let result = crate::run_reporting(messages, &slot, schema, run_once);
    if let (Some(db), Ok(response)) = (&db, &result) {
        // Kept, with the agent, for `faber agents show` and the web UI.
        let values: Vec<serde_json::Value> = response
            .history
            .iter()
            .filter_map(|m| serde_json::to_value(m).ok())
            .collect();
        let _ = db.save_agent_messages(agent, &values);
    }
    if let Err(e) = &result {
        if e.downcast_ref::<InterruptedError>().is_some() {
            if let Some(activity) = &activity {
                activity.set("stopped");
            }
            crate::record_turn_end(
                &events,
                &faber::db::TaskOutcome {
                    succeeded: false,
                    exit_code: None,
                    result: "stopped".to_string(),
                },
            );
            return Err(Box::new(InterruptedError::new(
                "Operation interrupted by user",
            )));
        }
    }
    if let Some(usage) = result.as_ref().ok().and_then(|r| r.turn_usage.as_ref()) {
        sa_ctx
            .session_usage
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .record(usage);
    }
    let outcome = crate::work_outcome(result, &slot);
    let (succeeded, text) = match &outcome {
        Ok(outcome) => (outcome.succeeded, outcome.text()),
        Err(e) => (false, e.to_string()),
    };
    crate::record_turn_end(
        &events,
        &faber::db::TaskOutcome {
            succeeded,
            exit_code: None,
            result: text.clone(),
        },
    );
    if let Some(activity) = &activity {
        let first = crate::first_line(&text, 80);
        activity.set(&if succeeded {
            format!("finished: {}", first)
        } else {
            format!("failed: {}", first)
        });
    }
    match outcome {
        Ok(outcome) if outcome.succeeded => Ok(outcome.text()),
        Ok(outcome) => Err(format!("reported failure - {}", outcome.text()).into()),
        Err(e) => Err(e.into()),
    }
}

/// entrypoint for the fan_out tool
pub(crate) fn tool_fan_out(
    params_str: &String,
    ctx: &ToolContext,
) -> Result<String, Box<dyn Error>> {
    let params: Params = serde_json::from_str(params_str)?;
    let caller = ctx
        .extra
        .as_ref()
        .and_then(|e| e.downcast_ref::<SubAgentContext>())
        .ok_or("fan_out is only available to the main chat agent")?;
    if params.items.is_empty() || params.items.len() > MAX_ITEMS {
        return Err(format!("give between 1 and {} items", MAX_ITEMS).into());
    }
    // Workers run with the profile's settings and tools, if given, else
    // the caller's.
    let profile = crate::requested_profile(&caller.profiles, params.profile.clone());
    let config = match profile.as_deref() {
        Some(name) => crate::find_profile(&caller.profiles, name)?.agent_config(name),
        None => AgentConfig::default(),
    };
    let sa_ctx = &SubAgentContext {
        opts: crate::with_agent_config(&caller.opts, &config),
        tools: Arc::new(crate::effective_tools(&caller.tools, &config)),
        ..caller.clone()
    };
    let mut tools = worker_tools(&sa_ctx.tools, params.tools.as_deref())?;
    // Workers can always report their outcome.
    if let Some(report) = caller.tools.get("report_result") {
        tools.insert("report_result".to_string(), report.clone());
    }
    let handoff =
        crate::handoff_message(ctx, params.context.as_ref().unwrap_or(&Default::default()))?;
    // No more at once than may be sent to the model at once: the rest
    // would only queue, and taking turns, evict each other's prompts from
    // the server's cache (see --max-parallel-requests).
    let slots = match openai::max_parallel_requests() {
        0 => MAX_PARALLEL,
        n => n,
    };
    let parallel = params
        .max_parallel
        .unwrap_or(DEFAULT_MAX_PARALLEL)
        .clamp(1, MAX_PARALLEL)
        .min(slots)
        .min(params.items.len());
    let total = params.items.len();
    let worker_name = |index: usize| {
        format!(
            "{}-item-{}",
            ctx.agent_name.as_deref().unwrap_or("default"),
            index + 1
        )
    };
    // Every worker's agent, made up front: they all show, queued, even
    // those that have to wait their turn to start.
    if let (Some(db), Some(caller)) = (&ctx.db, &ctx.agent_name) {
        for index in 0..total {
            let name = worker_name(index);
            crate::make_worker_agent(db.as_ref(), &name, caller, ctx, &sa_ctx.session_id)?;
            let _ = db.set_agent_activity(&name, "queued");
        }
    }

    let queue: Mutex<VecDeque<usize>> = Mutex::new((0..total).collect());
    let results: Mutex<Vec<Option<Result<String, String>>>> = Mutex::new(vec![None; total]);
    let cancelled = AtomicBool::new(false);
    let finished = AtomicBool::new(false);
    let done_count = AtomicUsize::new(0);
    // Every running worker's Ctrl-C channel. The watcher sets `cancelled`
    // before sending to these under the lock, and a worker checks
    // `cancelled` after registering under the same lock, so none can miss
    // the signal.
    let cancel_senders: Mutex<Vec<mpsc::Sender<()>>> = Mutex::new(Vec::new());

    ctx.println(&format!(
        "Fanning out to {} items, {} at a time...",
        total, parallel
    ));
    // The caller's activity follows the workers' progress.
    let activity = match (&ctx.db, &ctx.agent_name) {
        (Some(db), Some(agent)) => Some(ActivityRecorder::new(db.clone(), agent)),
        _ => None,
    };
    if let Some(activity) = &activity {
        activity.set(&format!("fan_out: 0/{} done", total));
    }

    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !finished.load(Ordering::Relaxed) {
                let interrupted = ctx.interrupt.as_ref().is_some_and(|rx| {
                    rx.lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .try_recv()
                        .is_ok()
                });
                if interrupted {
                    cancelled.store(true, Ordering::Relaxed);
                    for tx in cancel_senders
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .iter()
                    {
                        let _ = tx.send(());
                    }
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });

        let workers: Vec<_> = (0..parallel)
            .map(|_| {
                scope.spawn(|| {
                    loop {
                        let Some(index) =
                            queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front()
                        else {
                            return;
                        };
                        let (tx, rx) = mpsc::channel();
                        cancel_senders
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(tx.clone());
                        if cancelled.load(Ordering::Relaxed) {
                            return;
                        }
                        let item = &params.items[index];
                        let key = format!("fan-out {}/{}", index + 1, total);
                        let agent = worker_name(index);
                        // Like a sub-agent: it can be stopped on its own
                        // (agent_cancel, /cancel, the web UI).
                        let stop_reason = Arc::new(Mutex::new(None));
                        sa_ctx
                            .running_subagents
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(
                                agent.clone(),
                                crate::RunningSubAgent {
                                    cancel: tx,
                                    stop_reason: stop_reason.clone(),
                                },
                            );
                        sa_ctx.status_bar.set_agent_status(&key, "Starting", true);
                        let result = run_worker(
                            sa_ctx,
                            ctx,
                            &tools,
                            &config,
                            &key,
                            &agent,
                            expand_prompt(&params.prompt, item),
                            handoff.as_ref(),
                            params.result_schema.as_ref(),
                            rx,
                        );
                        sa_ctx.status_bar.clear_agent_status(&key);
                        sa_ctx
                            .running_subagents
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .remove(&agent);
                        let result = match result {
                            // The whole fan-out stopped.
                            Err(e)
                                if e.downcast_ref::<InterruptedError>().is_some()
                                    && cancelled.load(Ordering::Relaxed) =>
                            {
                                return;
                            }
                            // Only this worker: its item failed, the rest
                            // carry on.
                            Err(e) if e.downcast_ref::<InterruptedError>().is_some() => {
                                Err(format!(
                                    "stopped: {}",
                                    stop_reason
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .clone()
                                        .unwrap_or_else(|| "cancelled".to_string())
                                ))
                            }
                            other => other.map_err(|e| e.to_string()),
                        };
                        let done = done_count.fetch_add(1, Ordering::Relaxed) + 1;
                        if let Some(activity) = &activity {
                            activity.set(&format!("fan_out: {}/{} done", done, total));
                        }
                        ctx.println(&format!(
                            "[{}/{}] {} {}",
                            done,
                            total,
                            if result.is_ok() { "done:" } else { "failed:" },
                            item
                        ));
                        results.lock().unwrap_or_else(|e| e.into_inner())[index] = Some(result);
                    }
                })
            })
            .collect();
        for worker in workers {
            let _ = worker.join();
        }
        finished.store(true, Ordering::Relaxed);
    });
    // Those that never started (stopped first) are let go too.
    if let Some(db) = &ctx.db {
        let results = results.lock().unwrap_or_else(|e| e.into_inner());
        for index in (0..total).filter(|i| results[*i].is_none()) {
            let name = worker_name(index);
            let queued = db
                .get_agent(&name)
                .ok()
                .flatten()
                .is_some_and(|a| a.activity.as_deref() == Some("queued"));
            if queued {
                let _ = db.set_agent_activity(&name, "not started: stopped");
            }
            let _ = db.release_agent(&name, &sa_ctx.session_id);
        }
    }
    // Workers finishing together can record their counts out of order:
    // the last word is the final count.
    if let Some(activity) = &activity {
        activity.set(&format!(
            "fan_out: {}/{} done",
            done_count.load(Ordering::Relaxed),
            total
        ));
    }

    if cancelled.load(Ordering::Relaxed) {
        return Err(Box::new(InterruptedError::new(
            "Operation interrupted by user",
        )));
    }
    let results = results.into_inner().unwrap_or_else(|e| e.into_inner());
    Ok(format_results(
        &params.items,
        &results,
        ctx.max_tool_output_chars(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ToolItem;
    use std::collections::HashMap;

    /// A chat-like context for fan_out against the built-in dummy model,
    /// with the Ctrl-C sender and the session usage it records into.
    fn dummy_chat_ctx() -> (
        ToolContext,
        mpsc::Sender<()>,
        Arc<Mutex<crate::SessionUsage>>,
    ) {
        dummy_chat_ctx_for("dummy")
    }

    fn dummy_chat_ctx_for(
        model: &str,
    ) -> (
        ToolContext,
        mpsc::Sender<()>,
        Arc<Mutex<crate::SessionUsage>>,
    ) {
        let usage = Arc::new(Mutex::new(crate::SessionUsage::default()));
        let (ctrl_c_tx, ctrl_c_rx) = mpsc::channel();
        let mut ctx = ToolContext::new(|_: &str| {});
        ctx.interrupt = Some(Arc::new(Mutex::new(ctrl_c_rx)));
        ctx.extra = Some(Arc::new(SubAgentContext {
            runs: Arc::new(crate::SubAgentRuns::default()),
            tools: Arc::new(crate::initialize_tools(false, None)),
            opts: openai::Opts {
                max_tokens: None,
                model: model.to_string(),
                endpoint: String::new(),
                tool_choice: None,
                api_key: None,
                max_retries: None,
                retry_base_delay_secs: None,
                parameters: Default::default(),
            },
            session_id: "test".to_string(),
            active_subagents: Arc::new(AtomicUsize::new(0)),
            running_subagents: Arc::new(Mutex::new(std::collections::HashMap::new())),
            status_bar: Arc::new(crate::status_bar::StatusBar::new()),
            session_usage: usage.clone(),
            profiles: Default::default(),
        }));
        (ctx, ctrl_c_tx, usage)
    }

    /// Like `dummy_chat_ctx_for`, but as the agent "boss" of a database, so
    /// workers are agents too - made, claimed and released, their events
    /// and activity recorded - as in a real session.
    fn dummy_agent_ctx(
        model: &str,
    ) -> (
        ToolContext,
        mpsc::Sender<()>,
        Arc<dyn faber::db_backend::DbBackend>,
    ) {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        faber::db::initialize_db(&conn).unwrap();
        let db: Arc<dyn faber::db_backend::DbBackend> =
            Arc::new(faber::local_db::LocalDb::new(Arc::new(Mutex::new(conn))));
        db.create_agent("boss", "test").unwrap();
        let (mut ctx, ctrl_c, _) = dummy_chat_ctx_for(model);
        ctx.db = Some(db.clone());
        ctx.agent_name = Some("boss".to_string());
        (ctx, ctrl_c, db)
    }

    fn running_workers(ctx: &ToolContext) -> Arc<Mutex<HashMap<String, crate::RunningSubAgent>>> {
        ctx.extra
            .as_ref()
            .and_then(|e| e.downcast_ref::<SubAgentContext>())
            .unwrap()
            .running_subagents
            .clone()
    }

    /// Runs fan_out on a thread of its own and fails the test, saying where
    /// everyone was, if it hasn't returned within `limit`: a fan-out that
    /// stops making progress fails here instead of hanging the test run.
    /// An error is its text, or "interrupted" for a stop.
    fn fan_out_within(
        ctx: Arc<ToolContext>,
        params: serde_json::Value,
        model: &str,
        limit: Duration,
    ) -> Result<String, String> {
        run_within(ctx, model, limit, move |ctx| {
            tool_fan_out(&params.to_string(), ctx)
        })
    }

    /// Runs `f` like `fan_out_within` runs fan_out.
    fn run_within(
        ctx: Arc<ToolContext>,
        model: &str,
        limit: Duration,
        f: impl FnOnce(&ToolContext) -> Result<String, Box<dyn Error>> + Send + 'static,
    ) -> Result<String, String> {
        let (tx, rx) = mpsc::channel();
        let fan_out_ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = f(&fan_out_ctx).map_err(|e| {
                if e.downcast_ref::<InterruptedError>().is_some() {
                    "interrupted".to_string()
                } else {
                    e.to_string()
                }
            });
            let _ = tx.send(result);
        });
        match rx.recv_timeout(limit) {
            Ok(result) => result,
            Err(_) => {
                let agents: Vec<String> = ctx
                    .db
                    .as_ref()
                    .and_then(|db| db.list_agents().ok())
                    .unwrap_or_default()
                    .into_iter()
                    .map(|a| format!("{}: {:?}", a.name, a.activity))
                    .collect();
                panic!(
                    "no progress for {:?}: {} requests in flight, {} workers \
                     running; agents: {:#?}",
                    limit,
                    faber::dummy_llm::requests_in_flight(model),
                    running_workers(&ctx).lock().unwrap().len(),
                    agents
                );
            }
        }
    }

    /// After a fan-out, however it ended: no request still holds a slot,
    /// and no worker is still running, still claimed, or still shows as
    /// queued or busy.
    fn assert_all_let_go(
        ctx: &ToolContext,
        db: &dyn faber::db_backend::DbBackend,
        model: &str,
        context: &str,
    ) {
        assert_eq!(
            faber::dummy_llm::requests_in_flight(model),
            0,
            "a request slot leaked ({context})"
        );
        assert!(
            running_workers(ctx).lock().unwrap().is_empty(),
            "a worker is still listed as running ({context})"
        );
        for agent in db.list_agents().unwrap() {
            if agent.parent.as_deref() != Some("boss") {
                continue;
            }
            assert_eq!(
                agent.session_id, None,
                "{} still claimed ({context})",
                agent.name
            );
            let activity = agent.activity.unwrap_or_default();
            assert!(
                ["finished: ", "failed: ", "stopped", "not started: stopped"]
                    .iter()
                    .any(|done| activity.starts_with(done)),
                "{} left as '{}' ({context})",
                agent.name,
                activity
            );
        }
    }

    /// A tiny deterministic generator, so a failing run can be replayed.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    #[test]
    fn test_fan_out_runs_every_item_and_returns_results_in_order() {
        let (ctx, _ctrl_c, usage) = dummy_chat_ctx();
        let params = serde_json::json!({
            "items": ["one", "two", "three"],
            "prompt": "Describe {item}.",
            "max_parallel": 2,
        });
        let out = tool_fan_out(&params.to_string(), &ctx).unwrap();
        let positions: Vec<usize> = ["## 1. one", "## 2. two", "## 3. three"]
            .iter()
            .map(|h| out.find(h).unwrap_or_else(|| panic!("{h} missing: {out}")))
            .collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]));
        assert!(out.contains("Dummy response"), "{out}");
        assert_eq!(
            usage.lock().unwrap().turns,
            3,
            "every worker's tokens count: {out}"
        );
    }

    #[test]
    fn test_fan_out_stops_every_worker_on_ctrl_c() {
        let (ctx, ctrl_c, _) = dummy_chat_ctx();
        // "slow N" makes the dummy stream N paragraphs a word every 200ms -
        // minutes, unless interrupted.
        let params = serde_json::json!({
            "items": ["slow 100", "slow 100", "slow 100", "slow 100"],
            "prompt": "{item}",
            "max_parallel": 2,
        });
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(500));
            let _ = ctrl_c.send(());
        });
        let started = std::time::Instant::now();
        let err = tool_fan_out(&params.to_string(), &ctx).unwrap_err();
        assert!(err.downcast_ref::<InterruptedError>().is_some(), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn test_fan_out_progresses_with_fewer_request_slots_than_workers() {
        // One request at a time among eight workers, each with several
        // rounds of tool calls: they must take turns rather than wait on
        // each other for good.
        let model = "dummy:slots=1,id=fan-out-scarce";
        let (ctx, _ctrl_c, db) = dummy_agent_ctx(model);
        let ctx = Arc::new(ctx);
        let items: Vec<String> = (1..=24).map(|i| format!("item-{i}")).collect();
        let params = serde_json::json!({
            "items": items,
            "prompt": "work 3 2 on {item}",
            "max_parallel": 8,
            "result_schema": {"job": "string"},
        });
        let out = fan_out_within(ctx.clone(), params, model, Duration::from_secs(60)).unwrap();
        for (i, item) in items.iter().enumerate() {
            let expected = format!(
                "## {}. {}\nWorked 3 steps on: work 3 2 on {}",
                i + 1,
                item,
                item
            );
            assert!(out.contains(&expected), "{expected} missing: {out}");
        }
        assert_all_let_go(&ctx, db.as_ref(), model, "after finishing");
    }

    #[test]
    fn test_a_fan_out_called_by_the_model_gets_the_request_slot_its_caller_had() {
        // The chat's own request to the model asks for the fan_out: it must
        // have let its one slot go by then, or no worker ever gets one.
        let model = "dummy:slots=1,id=fan-out-from-the-model";
        let (ctx, _ctrl_c, db) = dummy_agent_ctx(model);
        let ctx = Arc::new(ctx);
        let fan_out = serde_json::json!({
            "items": ["a", "b", "c"],
            "prompt": "work 2 1 on {item}",
        });
        let out = run_within(ctx.clone(), model, Duration::from_secs(30), move |ctx| {
            let caller = ctx
                .extra
                .as_ref()
                .and_then(|e| e.downcast_ref::<SubAgentContext>())
                .unwrap();
            let response = post_request_with_mode(
                vec![make_message("user", format!("call fan_out {}", fan_out))],
                &caller.tools,
                &caller.opts,
                ResponseMode::Complete,
                ctx,
                None,
            )?;
            Ok(crate::final_response_text(Ok(response))?)
        })
        .unwrap();
        assert_eq!(out.matches("Worked 2 steps").count(), 3, "{out}");
        assert_all_let_go(&ctx, db.as_ref(), model, "after the chat's turn");
    }

    #[test]
    fn test_fan_out_finishes_when_workers_are_stopped_one_by_one() {
        let model = "dummy:slots=2,id=fan-out-cancels";
        for seed in 0..6 {
            let (ctx, _ctrl_c, db) = dummy_agent_ctx(model);
            let ctx = Arc::new(ctx);
            let running = running_workers(&ctx);
            let done = Arc::new(AtomicBool::new(false));
            // Stops a random running worker now and then - whether it's
            // waiting for a slot, for a response or for its tools - as
            // agent_cancel would.
            let canceller = {
                let done = done.clone();
                std::thread::spawn(move || {
                    let mut rng = Lcg(seed);
                    while !done.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(5 + rng.below(30)));
                        let running = running.lock().unwrap();
                        let names: Vec<&String> = running.keys().collect();
                        if names.is_empty() || rng.below(3) != 0 {
                            continue;
                        }
                        let worker = &running[names[rng.below(names.len() as u64) as usize]];
                        *worker.stop_reason.lock().unwrap() = Some("test".to_string());
                        let _ = worker.cancel.send(());
                    }
                })
            };
            let items: Vec<String> = (1..=16).map(|i| format!("item-{i}")).collect();
            let params = serde_json::json!({
                "items": items,
                "prompt": "work 4 3 on {item}",
                "max_parallel": 6,
                "result_schema": {"job": "string"},
            });
            let result = fan_out_within(ctx.clone(), params, model, Duration::from_secs(60));
            done.store(true, Ordering::Relaxed);
            canceller.join().unwrap();
            let out = result.unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            // Every item got an outcome: its result, or why it stopped.
            for (i, item) in items.iter().enumerate() {
                let heading = format!("## {}. {}\n", i + 1, item);
                let at = out
                    .find(&heading)
                    .unwrap_or_else(|| panic!("seed {seed}: {item} missing: {out}"));
                let body = &out[at + heading.len()..];
                assert!(
                    body.starts_with("Worked 4 steps") || body.starts_with("Error: stopped: test"),
                    "seed {seed}: {item}: {body}"
                );
            }
            assert_all_let_go(&ctx, db.as_ref(), model, &format!("seed {seed}"));
        }
    }

    #[test]
    fn test_fan_out_stopped_at_any_point_returns_and_lets_everything_go() {
        // Ctrl-C at all sorts of moments: before any worker starts, while
        // workers queue for request slots, mid-response, mid-tool call,
        // while results are reported, after the last one is done.
        let model = "dummy:slots=2,id=fan-out-ctrl-c";
        for (run, stop_after_ms) in [0u64, 1, 5, 15, 30, 60, 100, 150, 250, 400, 2000]
            .into_iter()
            .enumerate()
        {
            let (ctx, ctrl_c, db) = dummy_agent_ctx(model);
            let ctx = Arc::new(ctx);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(stop_after_ms));
                let _ = ctrl_c.send(());
            });
            let params = serde_json::json!({
                "items": (1..=12).map(|i| format!("item-{i}")).collect::<Vec<_>>(),
                "prompt": "work 3 4 on {item}",
                "max_parallel": 6,
                "result_schema": {"job": "string"},
            });
            let context = format!("run {run}, stopped after {stop_after_ms}ms");
            match fan_out_within(ctx.clone(), params, model, Duration::from_secs(30)) {
                Ok(_) => {}
                Err(e) => assert_eq!(e, "interrupted", "{context}"),
            }
            assert_all_let_go(&ctx, db.as_ref(), model, &context);
        }
        // The slots all came back: a fan-out still gets every one done.
        let (ctx, _ctrl_c, _) = dummy_agent_ctx(model);
        let params = serde_json::json!({
            "items": ["a", "b", "c", "d"],
            "prompt": "work 2 1 on {item}",
        });
        let out = fan_out_within(Arc::new(ctx), params, model, Duration::from_secs(30)).unwrap();
        assert_eq!(out.matches("Worked 2 steps").count(), 4, "{out}");
    }

    #[test]
    fn test_worker_context_acts_as_the_caller() {
        let (mut ctx, _ctrl_c, _) = dummy_chat_ctx();
        ctx.agent_name = Some("boss".to_string());
        ctx.context_window = Some(32_000);
        let worker = worker_context(&ctx);
        assert_eq!(worker.agent_name.as_deref(), Some("boss"));
        assert_eq!(worker.context_window, Some(32_000));
        assert!(worker.extra.is_none(), "workers can't spawn agents");
        assert!(worker.mcp.is_none());
    }

    #[test]
    fn test_items_that_are_not_strings_are_their_json() {
        let params: Params =
            serde_json::from_str(r#"{"items": ["a", {"issue": 66}, 3], "prompt": "x"}"#).unwrap();
        assert_eq!(params.items, vec!["a", r#"{"issue":66}"#, "3"]);
    }

    #[test]
    fn test_fan_out_needs_the_main_chat_context() {
        let ctx = ToolContext::new(|_: &str| {});
        let params = serde_json::json!({"items": ["a"], "prompt": "x"});
        assert!(tool_fan_out(&params.to_string(), &ctx).is_err());
    }

    #[test]
    fn test_expand_prompt() {
        assert_eq!(
            expand_prompt("Review {item} now", "a.rs"),
            "Review a.rs now"
        );
        assert_eq!(
            expand_prompt("Review it", "a.rs"),
            "Review it\n\nItem: a.rs"
        );
    }

    fn noop(_: &String, _: &ToolContext) -> Result<String, Box<dyn Error>> {
        Ok(String::new())
    }

    fn available(names: &[&str]) -> ToolsCollection {
        names
            .iter()
            .map(|n| {
                (
                    n.to_string(),
                    ToolItem {
                        callback: noop,
                        schema: String::new(),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn test_worker_tools_default_to_the_available_read_only_ones() {
        let tools = available(&[
            "read_file",
            "write_file",
            "lsp",
            "run_command",
            "spawn_agent",
        ]);
        let mut names: Vec<_> = worker_tools(&tools, None).unwrap().into_keys().collect();
        names.sort();
        assert_eq!(names, vec!["lsp", "read_file"]);
    }

    #[test]
    fn test_worker_tools_explicit_list_is_checked() {
        let tools = available(&["read_file", "write_file", "spawn_agent"]);
        let picked = worker_tools(&tools, Some(&["write_file".to_string()])).unwrap();
        assert_eq!(picked.len(), 1);
        assert!(worker_tools(&tools, Some(&["spawn_agent".to_string()])).is_err());
        assert!(worker_tools(&tools, Some(&["nope".to_string()])).is_err());
    }

    #[test]
    fn test_format_results_in_order_with_errors_and_caps() {
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let results = vec![
            Some(Ok("x".repeat(10_000))),
            Some(Err("boom".to_string())),
            None,
        ];
        let out = format_results(&items, &results, 6000);
        let a = out.find("## 1. a").unwrap();
        let b = out.find("## 2. b\nError: boom").unwrap();
        let c = out.find("## 3. c\nNot run.").unwrap();
        assert!(a < b && b < c);
        assert!(out.contains("characters omitted"), "a's result is capped");
    }
}
