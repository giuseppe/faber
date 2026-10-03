// faber's web UI: the agents, the tasks, and what they're doing, live -
// all from `faber serve`'s /api (see src/web.rs). Polls; no build step,
// no dependencies.
"use strict";

const POLL_MS = 1500;
// Sub-agents listed under an agent before "N more".
const SHOWN_CHILDREN = 8;
// Most events kept for the all-agents feed.
const MAX_FEED_ROWS = 3000;
// Finished tasks shown per board column until "show all".
const DONE_SHOWN = 20;

const state = {
  agents: [],
  tasks: [],
  profiles: [],
  // What the feed follows: null (every agent), {kind: "agent", name} or
  // {kind: "task", id}.
  selected: null,
  // Every agent's events, as they come (for the all-agents feed and the
  // latest output lines).
  all: { rows: [], last: null },
  // The selected agent's or task's, fetched on their own: they can go
  // back further than `all` does.
  one: { rows: [], last: null },
  // The latest thing each agent, and each task's run, said or did.
  latestByAgent: new Map(),
  latestByTask: new Map(),
  // Tool calls that failed, by task: shown on its card and in its panel.
  errorsByTask: new Map(),
  // The server's own directory, a new task's default one.
  serverCwd: null,
  showAllDone: new Set(),
  // Agents whose every sub-agent is listed, not just the first few.
  expanded: new Set(),
  // Every tool an agent's list can name.
  tools: null,
};

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function toast(message) {
  const t = $("toast");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, 4000);
}

// --- API ------------------------------------------------------------------

function authKey() {
  try { return localStorage.getItem("faber-key"); } catch { return null; }
}

async function api(method, path, body) {
  const headers = {};
  const key = authKey();
  if (key) headers["Authorization"] = `Bearer ${key}`;
  // Required for anything but GET (see src/web.rs).
  if (method !== "GET") headers["Content-Type"] = "application/json";
  const response = await fetch(`/api/${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 401) {
    if (!$("auth").open) $("auth").showModal();
    throw new Error("unauthorized");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status}`);
  return data;
}

$("auth-form").addEventListener("submit", () => {
  const key = new FormData($("auth-form")).get("key");
  try { localStorage.setItem("faber-key", key); } catch { /* per-tab only */ }
  refresh();
});

// --- Text and time --------------------------------------------------------

function parseTime(text) {
  if (!text) return null;
  // The database's own "YYYY-MM-DD HH:MM:SS" is UTC.
  const t = /^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/.test(text) ? text.replace(" ", "T") + "Z" : text;
  const date = new Date(t);
  return isNaN(date) ? null : date;
}

function relative(text) {
  const date = parseTime(text);
  if (!date) return "-";
  const secs = Math.round((Date.now() - date) / 1000);
  const s = Math.abs(secs);
  const amount = s < 5 ? "now" : s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m`
    : s < 86400 ? `${Math.floor(s / 3600)}h` : `${Math.floor(s / 86400)}d`;
  if (amount === "now") return "just now";
  return secs < 0 ? `in ${amount}` : `${amount} ago`;
}

function firstLine(text, max = 80) {
  const line = (text || "").split("\n").find((l) => l.trim()) || "";
  return line.length > max ? line.slice(0, max - 1) + "…" : line.trim();
}

function lastLine(text, max = 100) {
  const lines = (text || "").split("\n").filter((l) => l.trim());
  const line = (lines[lines.length - 1] || "").trim();
  return line.length > max ? "…" + line.slice(line.length - max + 1) : line;
}

// --- Events: what agents said and did -------------------------------------

// A failed tool call's reason: "error: tool 'x' failed: why" -> "why", or
// the `error` of a JSON result.
function toolError(output) {
  const text = (output || "").trim();
  if (text.startsWith("{")) {
    try {
      const json = JSON.parse(text);
      const error = json.error || json.stderr;
      if (error) return typeof error === "string" ? firstLine(error, 300) : JSON.stringify(error);
    } catch { /* not JSON after all */ }
  }
  return text.replace(/^error: (tool '[^']*' failed: |MCP tool '[^']*' failed: )?/, "");
}

// Keeps a failed tool call of a task's run, for its card and panel.
function noteError(row) {
  const event = row.event;
  if (event.type !== "tool_end" || !event.failed || row.task_id == null) return;
  const errors = state.errorsByTask.get(row.task_id) || [];
  if (errors.some((e) => e.id === row.id)) return;
  errors.push({ id: row.id, agent: row.agent, tool: event.name, message: toolError(event.output) });
  errors.sort((a, b) => a.id - b.id);
  state.errorsByTask.set(row.task_id, errors);
}

// A one-line summary of the latest thing an agent did, from its events.
function noteLatest(row) {
  const event = row.event;
  const previous = state.latestByAgent.get(row.agent);
  let line;
  switch (event.type) {
    case "text": {
      // Streamed text continues the previous text event.
      const text = (previous?.type === "text" ? previous.text + event.text : event.text).slice(-2000);
      line = { type: "text", text, line: lastLine(text) };
      break;
    }
    case "tool_start":
      line = { type: "tool", line: `▸ ${event.name}(${firstLine(event.arguments, 60)})` };
      break;
    case "tool_end":
      line = { type: "tool", line: `✓ ${event.name}: ${firstLine(event.output, 70)}` };
      break;
    case "input": {
      // "[Scheduled task #2 "review"]: review the patch" -> "#2 review"
      const task = event.text.match(/^\[Scheduled task (#\d+) "(.*?)"\]/);
      line = { type: "input", line: `← ${task ? `${task[1]} ${task[2]}` : firstLine(event.text, 90)}` };
      break;
    }
    case "turn_end":
      line = { type: "end", line: `${event.succeeded ? "✓" : "✗"} ${firstLine(event.text, 90)}` };
      break;
    default:
      return;
  }
  line.at = row.at;
  line.agent = row.agent;
  state.latestByAgent.set(row.agent, line);
  if (row.task_id !== null && row.task_id !== undefined) state.latestByTask.set(row.task_id, line);
}

function feedLabel() {
  const s = state.selected;
  return !s || s.kind === "task";
}

// Adds one event to the feed: streamed text joins the block before it,
// and a tool's result completes its call.
function appendEvent(box, row, labelled) {
  const event = row.event;
  const last = box.lastElementChild;
  if ((event.type === "text" || event.type === "reasoning") && last
      && last.dataset.type === event.type && last.dataset.agent === row.agent) {
    last.querySelector(".body").append(event.text);
    return;
  }
  const who = labelled
    ? el("span", { class: "who", title: "Follow this agent",
      onclick: () => select({ kind: "agent", name: row.agent }) },
      `${row.agent}${row.task_id ? ` · task #${row.task_id}` : ""} · ${relative(row.at)}`)
    : null;
  let node;
  switch (event.type) {
    case "input":
      node = el("div", { class: "event input" },
        who || el("span", { class: "who" }, `input · ${relative(row.at)}`),
        el("span", { class: "body" }, event.text));
      break;
    case "text":
    case "reasoning":
      node = el("div", { class: `event ${event.type}` }, who, el("span", { class: "body" }, event.text));
      break;
    case "tool_start":
      node = el("details", { class: "event tool running" },
        el("summary", {}, `▸ ${labelled ? row.agent + ": " : ""}${event.name}(${firstLine(event.arguments, 120)})`),
        el("pre", { class: "arguments" }, event.arguments));
      node.dataset.name = event.name;
      break;
    case "tool_end": {
      const open = [...box.querySelectorAll("details.tool.running")]
        .reverse()
        .find((d) => d.dataset.agent === row.agent && d.dataset.name === event.name);
      const failed = !!event.failed;
      const summary = failed
        ? `✗ ${labelled ? row.agent + ": " : ""}${event.name} failed: ${firstLine(toolError(event.output), 140)}`
        : `✓ ${labelled ? row.agent + ": " : ""}${event.name} · ${(event.duration_ms / 1000).toFixed(1)}s · ${firstLine(event.output, 100)}`;
      if (open) {
        open.classList.remove("running");
        open.classList.toggle("error", failed);
        open.open = open.open || failed;
        open.querySelector("summary").textContent = summary;
        open.append(el("pre", {}, event.output));
        return;
      }
      node = el("details", { class: `event tool${failed ? " error" : ""}`, open: failed },
        el("summary", {}, summary), el("pre", {}, event.output));
      break;
    }
    case "turn_end":
      node = el("div", { class: `event end ${event.succeeded ? "ok" : "bad"}` },
        el("span", { class: "end-mark" }),
        `${event.succeeded ? "Finished" : "Failed"}${labelled ? `: ${row.agent}` : ""} · ${relative(row.at)}`,
        event.text ? el("div", { class: "end-text" }, firstLine(event.text, 300)) : null);
      break;
    default:
      return;
  }
  node.dataset.type = event.type;
  node.dataset.agent = row.agent;
  box.append(node);
}

function feedStream() {
  return state.selected ? state.one : state.all;
}

function renderFeed() {
  const box = $("events");
  const stream = feedStream();
  box.replaceChildren();
  for (const row of stream.rows) appendEvent(box, row, feedLabel());
  if (!stream.rows.length && stream.last !== null) {
    box.append(el("div", { class: "empty" }, state.selected
      ? "Nothing recorded for it yet."
      : "No agent has done anything yet. Agents record what they do as they work - start a chat, a `faber worker`, or `faber serve --run-agent NAME`."));
  }
  scrollFeed();
}

function scrollFeed() {
  if ($("follow").checked) $("detail-panel").scrollTop = $("detail-panel").scrollHeight;
}

// Fetches a stream's new events; returns them.
async function pollStream(stream, params) {
  if (stream.last !== null) params.set("after", stream.last);
  params.set("limit", stream.last === null ? "400" : "1000");
  const rows = await api("GET", `events?${params}`);
  if (rows.length) stream.last = rows[rows.length - 1].id;
  else if (stream.last === null) stream.last = 0;
  stream.rows.push(...rows);
  if (stream.rows.length > MAX_FEED_ROWS) stream.rows.splice(0, stream.rows.length - MAX_FEED_ROWS);
  return rows;
}

let polling = false;

async function pollEvents() {
  if (polling) return;
  polling = true;
  try {
    const firstAll = state.all.last === null;
    const rows = await pollStream(state.all, new URLSearchParams());
    rows.forEach(noteLatest);
    rows.forEach(noteError);
    const selected = state.selected;
    let shown = state.selected ? [] : rows;
    let changed = rows.length > 0;
    if (selected) {
      const stream = state.one;
      const firstOne = stream.last === null;
      const params = new URLSearchParams(selected.kind === "agent"
        ? { agent: selected.name } : { task: selected.id });
      const mine = await pollStream(stream, params);
      mine.forEach(noteError);
      if (state.selected !== selected) return; // switched meanwhile
      shown = mine;
      changed ||= mine.length > 0;
      if (firstOne) { renderFeed(); shown = []; }
    } else if (firstAll) {
      renderFeed();
      shown = [];
    }
    if (shown.length) {
      const box = $("events");
      box.querySelector(":scope > .empty")?.remove();
      for (const row of shown) appendEvent(box, row, feedLabel());
      scrollFeed();
    }
    if (changed) {
      renderAgents();
      renderTasks();
      renderDetail();
    }
  } catch { /* shown by the connection pill */ } finally {
    polling = false;
  }
}

// --- Agents ---------------------------------------------------------------

function busy(activity) {
  return !!activity && !/^(idle|stopped|finished|failed|queued|not started)/.test(activity);
}

// Where an agent stands, from its activity: what its dot and activity
// line show - a finished one green, a failed or stopped one red.
function agentState(agent) {
  const activity = agent.activity || "";
  if (/^finished/.test(activity)) return "finished";
  if (/^(failed|stopped|not started)/.test(activity)) return "failed";
  if (activity === "queued") return "queued";
  if (agent.live && busy(activity)) return "busy";
  if (agent.live) return "live";
  return "";
}

const STATE_TITLES = {
  finished: "finished",
  failed: "failed or stopped",
  queued: "queued: waiting for its turn",
  busy: "working",
  live: "live session, idle",
  "": "no live session",
};

function agentTree(agents) {
  const names = new Set(agents.map((a) => a.name));
  const children = new Map();
  const roots = [];
  for (const agent of agents) {
    if (agent.parent && names.has(agent.parent)) {
      if (!children.has(agent.parent)) children.set(agent.parent, []);
      children.get(agent.parent).push(agent);
    } else {
      roots.push(agent);
    }
  }
  // task-1-item-2 before task-1-item-10.
  for (const kids of children.values()) {
    kids.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  }
  const out = [];
  const walk = (agent, depth) => {
    out.push([agent, depth]);
    if (depth > 32) return;
    const kids = children.get(agent.name) || [];
    // A fan-out's hundreds of workers: the first few, until asked.
    const shown = state.expanded.has(agent.name) || kids.length <= SHOWN_CHILDREN
      ? kids : kids.slice(0, SHOWN_CHILDREN - 1);
    for (const kid of shown) walk(kid, depth + 1);
    if (shown.length < kids.length) out.push([{ more: kids.length - shown.length, parent: agent.name }, depth + 1]);
  };
  roots.forEach((r) => walk(r, 0));
  return out;
}

function renderAgents() {
  const list = $("agents");
  const live = state.agents.filter((a) => a.live).length;
  $("agents-count").textContent = `${live} live / ${state.agents.length}`;
  if (!state.agents.length) {
    list.replaceChildren(el("li", { class: "empty" }, "No agents yet."));
    return;
  }
  list.replaceChildren(...agentTree(state.agents).map(([agent, depth]) => {
    if (agent.more) {
      return el("li", {
        class: "more",
        style: `margin-left: ${depth * 14}px`,
        onclick: () => { state.expanded.add(agent.parent); renderAgents(); },
      }, `+ ${agent.more} more`);
    }
    const selected = state.selected?.kind === "agent" && state.selected.name === agent.name;
    const st = agentState(agent);
    const dot = `dot ${st}`;
    const latest = state.latestByAgent.get(agent.name);
    return el("li", {
      class: selected ? "selected" : "",
      style: `margin-left: ${depth * 14}px`,
      onclick: () => select({ kind: "agent", name: agent.name }),
    },
      el("div", { class: "agent-name" },
        el("span", { class: dot, title: STATE_TITLES[st] }),
        agent.name,
        agent.profile ? el("span", { class: "tag" }, agent.profile) : null,
        agent.unsafe_tools ? el("span", { class: "badge danger", title: "Has the unsafe tools: unsandboxed commands, web access" }, "unsafe") : null),
      el("div", { class: `agent-activity ${st}`, title: agent.activity || "" },
        agent.activity ? `${firstLine(agent.activity, 60)} · ${relative(agent.activity_at)}` : "-"),
      latest ? el("div", { class: "agent-output", title: latest.line }, latest.line) : null);
  }));
}

// --- Tasks ----------------------------------------------------------------

function taskState(task) {
  if (task.status === "running") return "running";
  if (task.status === "held") return "held";
  if (task.status === "disabled") return "disabled";
  if (task.status === "done") return task.last_outcome === "failed" ? "failed" : "succeeded";
  if (task.blocked_by?.length) return "blocked";
  return "waiting";
}

function taskTarget(task) {
  if (task.profile) return `profile ${task.profile}`;
  if (task.agent_name) return task.agent_name;
  return task.kind === "prompt" ? "any agent" : "scheduler";
}

function taskWhen(task, st) {
  if (st === "running") return `started ${relative(task.started_at)}`;
  if (st === "waiting") return `due ${relative(task.next_run_at)}`;
  if (task.last_run_at) return `ran ${relative(task.last_run_at)}`;
  return `created ${relative(task.created_at)}`;
}

function taskCard(task, draggable) {
  const st = taskState(task);
  const selected = state.selected?.kind === "task" && state.selected.id === task.id;
  const latest = state.latestByTask.get(task.id);
  const card = el("div", {
    class: `card ${st}${selected ? " selected" : ""}`,
    draggable: draggable ? "true" : null,
    onclick: () => select({ kind: "task", id: task.id }),
  },
    el("div", { class: "card-head" },
      el("div", { class: "card-title" }, `#${task.id} ${task.name}`),
      st !== "running" ? el("button", {
        class: "run",
        title: st === "succeeded" || st === "failed" ? "Run it again" : "Run it now",
        onclick: (e) => {
          e.stopPropagation();
          taskAction(task.id, "run");
        },
        "aria-label": "Run",
      }) : el("button", {
        class: "stop",
        title: task.stop_requested ? "Stopping…" : "Stop it",
        disabled: task.stop_requested,
        onclick: (e) => {
          e.stopPropagation();
          taskAction(task.id, "stop");
        },
        "aria-label": "Stop",
      })),
    el("div", { class: "card-meta" },
      st === "disabled" ? el("span", { class: "badge" }, "disabled") : null,
      errorCount(task) ? el("span", { class: "badge danger", title: "Tool calls that failed in its runs" },
        `${errorCount(task)} error${errorCount(task) === 1 ? "" : "s"}`) : null,
      `${taskTarget(task)} · ${taskWhen(task, st)}`,
      task.blocked_by?.length ? ` · after ${task.blocked_by.map((d) => "#" + d).join(", ")}` : "",
      task.cron_expression ? ` · cron ${task.cron_expression}` : ""),
    st === "running" ? runningLine(latest) : null,
    (st === "failed" || st === "succeeded") && task.last_result
      ? el("div", { class: "card-meta" }, firstLine(task.last_result, 90)) : null);
  if (draggable) {
    card.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", String(task.id));
      e.dataTransfer.effectAllowed = "move";
      card.classList.add("dragging");
      dragging = task;
    });
    card.addEventListener("dragend", () => {
      card.classList.remove("dragging");
      dragging = null;
      renderBoard();
    });
  }
  return card;
}

function errorCount(task) {
  return (state.errorsByTask.get(task.id) || []).length;
}

// What a running task's agent is up to: its latest output, or - until
// it has some - who took it and what they're doing.
function runningLine(latest) {
  if (!latest) return null;
  let line = latest.line;
  if (latest.type === "input") {
    const agent = state.agents.find((a) => a.name === latest.agent);
    line = `${latest.agent}: ${agent?.activity || "working"}`;
  }
  return el("div", { class: "card-output", title: line }, line);
}

const LIST_GROUPS = [
  ["running", "Running"],
  ["waiting", "Waiting"],
  ["blocked", "Blocked"],
  ["held", "Held"],
  ["failed", "Failed"],
  ["succeeded", "Done"],
  ["disabled", "Disabled"],
];

function renderTaskList() {
  const showFinished = $("show-finished").checked;
  const groups = new Map(LIST_GROUPS.map(([k]) => [k, []]));
  for (const task of state.tasks) groups.get(taskState(task))?.push(task);
  const shown = LIST_GROUPS.filter(([key]) => groups.get(key).length
    && (showFinished || !["succeeded", "failed", "disabled"].includes(key)));
  const list = $("task-list");
  if (!shown.length) {
    list.replaceChildren(el("div", { class: "empty" }, state.tasks.length ? "Nothing waiting." : "No tasks."));
    return;
  }
  list.replaceChildren(...shown.map(([key, title]) => {
    const tasks = groups.get(key).sort((a, b) => b.id - a.id);
    return el("div", { class: "group" },
      el("h3", {}, `${title} · ${tasks.length}`),
      tasks.map((t) => taskCard(t, false)));
  }));
}

// The scrum board's columns: which task states each holds, and which
// state a task dropped on it moves to.
const BOARD = [
  { key: "backlog", title: "Backlog", states: ["held", "disabled"], hint: "held or disabled until you start them", drop: "hold" },
  { key: "todo", title: "To do", states: ["waiting", "blocked"], hint: "picked up when due", drop: "release" },
  { key: "doing", title: "In progress", states: ["running"] },
  { key: "done", title: "Done", states: ["succeeded"], finished: true },
  { key: "failed", title: "Failed", states: ["failed"], finished: true },
];

let dragging = null;

function canDrop(column, task) {
  const st = taskState(task);
  if (column.drop === "hold") return ["waiting", "blocked"].includes(st);
  if (column.drop === "release") return st === "held" || st === "disabled";
  return false;
}

function renderBoard() {
  // Redrawn mid-drag, the card being dragged would be gone.
  if (dragging) return;
  const board = $("board");
  board.replaceChildren(...BOARD.map((column) => {
    let tasks = state.tasks.filter((t) => column.states.includes(taskState(t)));
    tasks.sort((a, b) => column.finished
      ? (parseTime(b.last_run_at) || 0) - (parseTime(a.last_run_at) || 0) || b.id - a.id
      : a.id - b.id);
    const total = tasks.length;
    const cut = column.finished && !state.showAllDone.has(column.key) && total > DONE_SHOWN;
    if (cut) tasks = tasks.slice(0, DONE_SHOWN);
    const node = el("div", { class: "column" },
      el("h3", {}, el("span", {}, column.title), el("span", {}, String(total))),
      column.hint ? el("p", { class: "hint" }, column.hint) : null,
      tasks.map((t) => taskCard(t, ["held", "disabled", "waiting", "blocked"].includes(taskState(t)))),
      cut ? el("button", { class: "ghost more", onclick: () => {
        state.showAllDone.add(column.key);
        renderBoard();
      } }, `show all ${total}`) : null);
    node.addEventListener("dragover", (e) => {
      if (dragging && canDrop(column, dragging)) {
        e.preventDefault();
        node.classList.add("drop-ok");
      }
    });
    node.addEventListener("dragleave", () => node.classList.remove("drop-ok"));
    node.addEventListener("drop", (e) => {
      e.preventDefault();
      node.classList.remove("drop-ok");
      const task = dragging;
      if (task && canDrop(column, task)) {
        taskAction(task.id, taskState(task) === "disabled" ? "enable" : column.drop);
      }
    });
    return node;
  }));
}

function renderTasks() {
  $("tasks-count").textContent = `${state.tasks.length}`;
  renderTaskList();
  renderBoard();
}

$("show-finished").addEventListener("change", renderTaskList);

async function taskAction(id, action, body) {
  try {
    if (action === "delete") {
      if (!confirm(`Delete task #${id}?`)) return;
      await api("DELETE", `tasks/${id}`);
      select(null);
    } else {
      await api("POST", `tasks/${id}/${action}`, body);
    }
  } catch (e) {
    toast(e.message);
  }
  refresh();
}

// --- The detail panel: everything, one agent, or one task -----------------

function select(what) {
  state.selected = what;
  state.one = { rows: [], last: null };
  document.body.classList.toggle("has-selection", !!what);
  renderAgents();
  renderTasks();
  renderDetail();
  renderFeed();
  pollEvents();
}

$("detail-back").addEventListener("click", () => select(null));

$("show-reasoning").addEventListener("change", (e) =>
  document.body.classList.toggle("show-reasoning", e.target.checked));

function field(label, value) {
  if (value === null || value === undefined || value === "") return [];
  return [el("dt", {}, label), el("dd", {}, value)];
}

// Under the feed: who of the agents it shows is working, on what.
function renderFeedStatus() {
  const s = state.selected;
  let names;
  if (!s) names = null;
  else if (s.kind === "agent") names = [s.name];
  else {
    const task = state.tasks.find((t) => t.id === s.id);
    names = task?.status === "running" ? [state.latestByTask.get(s.id)?.agent] : [];
  }
  const working = state.agents.filter((a) => a.live && busy(a.activity)
    && (!names || names.includes(a.name)));
  $("feed-status").textContent = working.map((a) => `${a.name}: ${a.activity}…`).join("  ·  ");
}

// At the top of a task's panel: why it failed, and the tool calls that
// failed along the way - not left to be found in the feed.
function renderProblems(task, st) {
  const box = $("problems");
  const errors = task ? state.errorsByTask.get(task.id) || [] : [];
  const failed = task && st === "failed";
  if (!failed && !errors.length) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.replaceChildren(...[
    failed ? el("div", { class: "problem-title" }, "Failed: ", firstLine(task.last_result, 300)) : null,
    errors.length ? el("div", { class: "problem-title" },
      `${errors.length} tool call${errors.length === 1 ? "" : "s"} failed:`) : null,
    errors.length ? el("ul", {}, errors.map((e) =>
      el("li", {}, el("code", {}, e.tool), ` (${e.agent}): ${e.message}`))) : null,
  ].filter(Boolean));
}

function renderDetail() {
  renderFeedStatus();
  if (state.selected?.kind !== "task") renderProblems(null);
  const selected = state.selected;
  const fields = $("detail-fields");
  const actions = $("detail-actions");
  actions.replaceChildren();
  $("detail-back").hidden = !selected;
  if (!selected) {
    const busyCount = state.agents.filter((a) => a.live && busy(a.activity)).length;
    $("detail-title").textContent = `All agents${busyCount ? ` · ${busyCount} working` : ""}`;
    fields.replaceChildren();
    return;
  }
  if (selected.kind === "agent") {
    const agent = state.agents.find((a) => a.name === selected.name);
    $("detail-title").textContent = `Agent ${selected.name}`;
    if (!agent) {
      fields.replaceChildren(...field("State", "gone"));
      return;
    }
    actions.append(el("button", { onclick: () => openNewTask(`agent:${agent.name}`) }, "Give it a task"));
    actions.append(el("button", { onclick: () => openAgentDialog(agent.name) }, "Edit…"));
    actions.append(el("button", { onclick: () => openAgentDialog(null, agent.name) }, "Clone…"));
    const kids = state.agents.filter((a) => a.parent === agent.name).map((a) => a.name);
    const queued = state.tasks.filter((t) => t.agent_name === agent.name && t.status !== "done");
    fields.replaceChildren(
      ...field("Session", agent.live ? "live" : "-"),
      ...field("Activity", agent.activity ? `${agent.activity} (${relative(agent.activity_at)})` : "-"),
      ...field("Description", agent.description),
      ...field("Model", agent.model),
      ...field("Profile", agent.profile),
      ...field("Works in", agent.cwd || "where it's run"),
      ...field("Unsafe tools", agent.unsafe_tools ? "yes" : null),
      ...field("Parent", agent.parent),
      ...field("Sub-agents", kids.join(", ")),
      ...field("Queued for it", queued.map((t) => `#${t.id}`).join(", ")));
    return;
  }
  const task = state.tasks.find((t) => t.id === selected.id);
  $("detail-title").textContent = `Task #${selected.id}${task ? ` ${task.name}` : ""}`;
  if (!task) {
    fields.replaceChildren(...field("State", "gone"));
    return;
  }
  const st = taskState(task);
  if (st !== "running") {
    actions.append(el("button", { class: "primary play", onclick: () => taskAction(task.id, "run") },
      st === "succeeded" || st === "failed" ? "Run again" : "Run now"));
  } else {
    actions.append(el("button", {
      class: "danger stop-button",
      disabled: task.stop_requested,
      onclick: () => taskAction(task.id, "stop"),
    }, task.stop_requested ? "Stopping…" : "Stop"));
  }
  if (["waiting", "blocked"].includes(st)) {
    actions.append(el("button", { onclick: () => taskAction(task.id, "hold") }, "Hold"));
  }
  if (st === "held") {
    actions.append(el("button", { onclick: () => taskAction(task.id, "release") }, "Release"));
  }
  if (st === "disabled") {
    actions.append(el("button", { onclick: () => taskAction(task.id, "enable") }, "Enable"));
  } else if (["waiting", "blocked", "held", "running"].includes(st)) {
    actions.append(el("button", { onclick: () => taskAction(task.id, "disable") }, "Disable"));
  }
  if (["waiting", "blocked", "held", "disabled"].includes(st) && task.kind === "prompt") {
    actions.append(el("button", { onclick: () => reassign(task) }, "Run on…"));
  }
  if (st !== "running") {
    actions.append(el("button", { class: "danger", onclick: () => taskAction(task.id, "delete") }, "Delete"));
  }
  renderProblems(task, st);
  fields.replaceChildren(
    ...field("State", st + (task.blocked_by?.length ? ` (after ${task.blocked_by.map((d) => "#" + d).join(", ")})` : "")),
    ...field("Runs on", taskTarget(task)),
    ...field("Runs in", task.cwd || "where its agent works"),
    ...field("Kind", task.kind),
    ...field("Schedule", task.cron_expression ? `cron ${task.cron_expression}` : `once, ${relative(task.run_at)}`),
    ...field("Next run", task.status === "scheduled" ? relative(task.next_run_at) : null),
    ...field("Runs", task.run_count ? `${task.run_count}${task.max_runs ? ` of ${task.max_runs}` : ""}` : null),
    ...field("Started", task.started_at ? relative(task.started_at) : null),
    ...field("Command", task.command),
    ...field("Last result", task.last_result));
}

// --- Agent settings: the user's to change, unsafe tools included --------

let editing = null; // the agent being edited, or null for a new one

// The agent a new one starts from: the last one made here, else "default".
function templateAgent() {
  let last = null;
  try { last = localStorage.getItem("faber-last-agent"); } catch { /* fine */ }
  for (const name of [last, "default"]) {
    if (name && state.agents.some((a) => a.name === name)) return name;
  }
  return null;
}

// Edits agent `name`, or - with none - makes a new one, starting from
// `from`'s settings (a clone), or else the template agent's.
async function openAgentDialog(name, from) {
  editing = name;
  const form = $("agent-form");
  form.reset();
  $("agent-error").hidden = true;
  $("agent-new-fields").hidden = !!name;
  $("agent-delete").hidden = !name;
  const source = name || from || templateAgent();
  $("agent-dialog-title").textContent = name ? `Agent ${name}` : from ? `Clone ${from}` : "New agent";
  $("agent-source").hidden = !!name || !source;
  $("agent-source").textContent = source ? `Starting from ${source}'s settings.` : "";
  try {
    if (!state.tools) state.tools = await api("GET", "tools");
    const detail = source ? await api("GET", `agents/${encodeURIComponent(source)}`) : null;
    const config = detail?.config || {};
    form.elements.description.value = (name || from) ? detail?.agent.description || "" : "";
    if (from) form.elements.name.value = uniqueName(`${from}-copy`);
    for (const field of ["model", "endpoint", "system_prompt", "max_tokens", "context_window", "cwd"]) {
      form.elements[field].value = config[field] ?? "";
    }
    form.elements.unsafe_tools.checked = !!config.unsafe_tools;
    form.elements.parameters.value = config.parameters ? JSON.stringify(config.parameters) : "";
    const chosen = new Set(config.tools || []);
    $("all-tools").checked = !config.tools;
    $("tool-list").replaceChildren(...state.tools.map((tool) => el("label", { class: "inline" },
      el("input", { type: "checkbox", name: "tool", value: tool, checked: chosen.has(tool) }), tool)));
    $("tool-list").classList.toggle("disabled", !config.tools);
  } catch (e) {
    toast(e.message);
    return;
  }
  $("agent-dialog").showModal();
}

$("all-tools").addEventListener("change", (e) =>
  $("tool-list").classList.toggle("disabled", e.target.checked));
$("new-agent-button").addEventListener("click", () => openAgentDialog(null));
$("agent-cancel").addEventListener("click", () => $("agent-dialog").close());

$("agent-delete").addEventListener("click", async () => {
  if (!editing || !confirm(`Delete agent ${editing}, with its conversation and the tasks only it would run?`)) return;
  try {
    await api("DELETE", `agents/${encodeURIComponent(editing)}`);
    $("agent-dialog").close();
    if (state.selected?.kind === "agent" && state.selected.name === editing) select(null);
    refresh();
  } catch (e) {
    showAgentError(e.message);
  }
});

$("agent-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const value = (field) => form.elements[field].value.trim() || null;
  const number = (field) => (value(field) === null ? null : Number(value(field)));
  let parameters = null;
  if (value("parameters")) {
    try {
      parameters = JSON.parse(value("parameters"));
    } catch {
      return showAgentError("Request parameters must be JSON, e.g. {\"temperature\": 0.2}");
    }
  }
  const config = {
    model: value("model"),
    endpoint: value("endpoint"),
    system_prompt: value("system_prompt"),
    max_tokens: number("max_tokens"),
    context_window: number("context_window"),
    cwd: value("cwd"),
    unsafe_tools: form.elements.unsafe_tools.checked,
    parameters,
    tools: $("all-tools").checked ? null
      : [...form.querySelectorAll("input[name=tool]:checked")].map((c) => c.value),
  };
  if (config.cwd && !config.cwd.startsWith("/")) return showAgentError("The working directory must be an absolute path.");
  try {
    if (editing) {
      await api("PATCH", `agents/${encodeURIComponent(editing)}/config`,
        { ...config, description: value("description") || "" });
    } else {
      const name = value("name");
      if (!name) return showAgentError("Give it a name.");
      await api("POST", "agents", { name, description: value("description") || "", config });
      try { localStorage.setItem("faber-last-agent", name); } catch { /* fine */ }
      select({ kind: "agent", name });
    }
    $("agent-dialog").close();
    refresh();
  } catch (err) {
    showAgentError(err.message);
  }
});

function uniqueName(base) {
  let name = base;
  for (let i = 2; state.agents.some((a) => a.name === name); i++) name = `${base}-${i}`;
  return name;
}

function showAgentError(message) {
  $("agent-error").textContent = message;
  $("agent-error").hidden = false;
}

function reassign(task) {
  const choices = ["(any agent)", ...state.agents.map((a) => a.name),
    ...state.profiles.map((p) => `profile:${p.name}`)];
  const answer = prompt(`Run task #${task.id} on which agent? One of:\n${choices.join("\n")}`,
    task.profile ? `profile:${task.profile}` : task.agent_name || "(any agent)");
  if (answer === null) return;
  const value = answer.trim();
  const body = value.startsWith("profile:") ? { profile: value.slice(8) }
    : value && value !== "(any agent)" ? { agent: value } : {};
  taskAction(task.id, "assign", body);
}

// --- Views ----------------------------------------------------------------

function setView(view) {
  document.body.dataset.view = view;
  try { localStorage.setItem("faber-view", view); } catch { /* fine */ }
}

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => setView(tab.dataset.view));
}

try {
  const view = localStorage.getItem("faber-view");
  if (view === "live" || view === "board") setView(view);
} catch { /* fine */ }

// --- New task -------------------------------------------------------------

function fillTargets(preset) {
  const select = $("target-select");
  const current = preset ?? select.value;
  const options = [el("option", { value: "" }, "Any agent (the first free one picks it up)")];
  if (state.profiles.length) {
    options.push(el("optgroup", { label: "New agent from a profile" },
      state.profiles.map((p) => el("option", { value: `profile:${p.name}` },
        p.name + (p.description ? ` — ${p.description}` : "")))));
  }
  if (state.agents.length) {
    options.push(el("optgroup", { label: "Agent" },
      state.agents.map((a) => el("option", { value: `agent:${a.name}` },
        a.name + (a.live ? "" : " (no live session)")))));
  }
  select.replaceChildren(...options);
  select.value = current;
  if (select.value !== current) select.value = "";
  updateTargetHint();
}

// The working directory a new task gets unless changed: the chosen
// agent's own, else the one last used, else the server's.
function defaultCwd() {
  const target = $("target-select").value;
  const agent = target.startsWith("agent:") && state.agents.find((a) => a.name === target.slice(6));
  if (agent?.cwd) return agent.cwd;
  try {
    const last = localStorage.getItem("faber-cwd");
    if (last) return last;
  } catch { /* fine */ }
  return state.serverCwd || "";
}

function updateTargetHint() {
  const input = $("cwd-input");
  if (!input.dataset.edited) input.value = defaultCwd();
  const value = $("target-select").value;
  $("target-hint").textContent = value.startsWith("profile:")
    ? "Run by a worker that has this profile, on a new agent made from it."
    : value.startsWith("agent:")
      ? "Run by this agent once its chat or worker is free."
      : "Run by whichever chat or worker is free first.";
}

$("target-select").addEventListener("change", updateTargetHint);

$("cwd-input").addEventListener("input", (e) => { e.target.dataset.edited = "1"; });

function openNewTask(preset) {
  delete $("cwd-input").dataset.edited;
  fillTargets(preset);
  $("new-task-error").hidden = true;
  $("new-task").showModal();
}

$("new-task-button").addEventListener("click", () => openNewTask());
$("new-task-cancel").addEventListener("click", () => $("new-task").close());

$("new-task-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const target = form.get("target") || "";
  const body = {
    command: form.get("command"),
    name: form.get("name") || null,
    tool: form.get("tool") === "on",
    hold: form.get("hold") === "on",
    agent: target.startsWith("agent:") ? target.slice(6) : null,
    profile: target.startsWith("profile:") ? target.slice(8) : null,
    depends_on: (form.get("depends_on") || "").split(/[\s,#]+/).filter(Boolean).map(Number),
    cwd: (form.get("cwd") || "").trim() || null,
  };
  if (body.cwd && !body.cwd.startsWith("/")) return showFormError("The working directory must be an absolute path.");
  const when = form.get("when");
  if (when === "at") {
    const at = new Date(form.get("at"));
    if (isNaN(at)) return showFormError("Pick a date and time.");
    body.at = at.toISOString();
  } else if (when === "cron") {
    body.cron = form.get("cron");
  }
  if (body.depends_on.some(isNaN)) return showFormError("Dependencies are task ids, e.g. 3, 5.");
  try {
    const { id } = await api("POST", "tasks", body);
    try { if (body.cwd) localStorage.setItem("faber-cwd", body.cwd); } catch { /* fine */ }
    $("new-task").close();
    e.target.reset();
    await refresh();
    select({ kind: "task", id });
  } catch (err) {
    showFormError(err.message);
  }
});

function showFormError(message) {
  $("new-task-error").textContent = message;
  $("new-task-error").hidden = false;
}

// --- Polling --------------------------------------------------------------

async function refresh() {
  const pill = $("connection");
  try {
    const [agents, tasks, profiles] = await Promise.all([
      api("GET", "agents"), api("GET", "tasks"), api("GET", "profiles"),
    ]);
    if (state.serverCwd === null) state.serverCwd = (await api("GET", "info")).cwd || "";
    state.agents = agents;
    state.tasks = tasks;
    state.profiles = profiles;
    pill.textContent = "connected";
    pill.className = "pill ok";
    renderAgents();
    renderTasks();
    renderDetail();
  } catch (e) {
    pill.textContent = e.message === "unauthorized" ? "auth needed" : "disconnected";
    pill.className = "pill bad";
  }
}

async function tick() {
  await refresh();
  await pollEvents();
  setTimeout(tick, POLL_MS);
}

tick();
