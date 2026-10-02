# faber

A multi-agent AI CLI tool for interactive software development. Supports
multiple concurrent agents, sub-agent spawning, inter-agent messaging,
scheduled tasks, MCP tool servers, and persistent state via SQLite.

## Setup

By default the tool uses `http://localhost:8080` as the API endpoint. Override
with `--endpoint`. If the endpoint requires authentication, point `--api-key`
at a file containing the key.

### Configuration file

Settings can be stored in a JSON file. By default `config.json` in the
current directory is loaded automatically. Use `-c`/`--config` to specify
a different path. CLI arguments override config file values.

```json
{
  "model": "google/gemini-2.5-pro",
  "endpoint": "http://localhost:8080",
  "api_key": "~/.keys/openai",
  "db_path": "state.db",
  "unsafe_tools": true,
  "parameter": ["temperature=0.7"],
  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

### GitHub token

A GitHub personal access token avoids rate limiting for GitHub tools.
Store it in `~/.github/token`:

```bash
mkdir -p ~/.github
echo "your-github-token" > ~/.github/token
```

## Usage

### Chat (interactive session)

```bash
faber chat
faber --db-path state.db chat          # with persistence
faber --db-path state.db --agent mybot chat  # start as a specific agent
```

### Chat commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/quit` | Exit the chat session |
| `/clear` | Clear chat history |
| `/show` | Display current chat history |
| `/limit N` | Keep only the last N messages (0 clears) |
| `/backtrace N` | Remove the last N messages |
| `/summarize` | Replace the chat history with a model-written summary (system prompts are kept) |
| `/system <msg>` | Inject a system message into the conversation |
| `/agents` | List all agents with message counts and status |
| `/create-agent <name>` | Create a new agent |
| `/select-agent <name>` | Switch to an existing agent |
| `/delete-agent <name>` | Delete an agent and its data |
| `/mcp-refresh` | Refresh tool definitions from MCP servers |
| `/tools` | List all available tools (built-in + MCP) |
| `/chdir <path>` | Change the current working directory (Tab-completes directory names) |
| `/pwd` | Show the current working directory |
| `/cost` | Show session token usage and estimated cost |
| `/plan` | Show the current agent's plan (see `plan_update`) |

Commands can also use `\` as the prefix (e.g. `\quit`).

`/chdir` is explicit and user-typed, so unlike `run_command` it isn't
sandboxed - it changes the real process directory. Every path-resolving
tool (`read_file`, `write_file`, `patch_file`, `glob`,
`grep_in_current_directory`, `run_command`'s sandbox bind, ...) re-resolves
the current directory on each call, so they immediately follow a `/chdir`
with no extra step.

Reasoning ("thinking") tokens from models that stream them separately are shown
in grey italics.  They are not kept in the conversation history, so they do not
use up the context window on later requests.  If the model stops because it hit
its token limit, the chat prints a warning instead of showing a silently
truncated answer.

The status bar reports `Waiting for response (N sent)` while a request is in
flight — showing the request's size, so a long wait on a large prompt (e.g.
after reading a large file) reads as "processing a lot of input", not a
hang — `Thinking` once the model starts responding, and `Streaming (N
bytes, M chunks)` while tokens keep arriving, including during a long
reasoning phase, so a model that "thinks out loud" for a while doesn't
leave the status looking frozen. Tool calls show `Preparing <tool>` while
their arguments are still streaming in, then `Running <tool>(<args>)` once
they're complete. The spinner and status text always render in the same
fixed color, separate from the agent's own color (shown on its name within
the status line and its prompt), so "something's in progress" is always
recognizable at a glance regardless of which agent is active.

A tool call's own output is framed between `── <tool>(<args>) ──` and
`── <tool> done in <N>s ──` markers, with each line in between prefixed by
`│ `, so it's clear where a tool's diagnostics start and end even when
several tool calls or a long model response follow one another.

Output is printed in real time as it streams in, character by character
rather than a line at a time, with the terminal's own wrapping handling
long lines. This means a response is never invisible while it accumulates,
however it's broken into lines and however long it runs - including a
model that degenerates into repeating itself with no line breaks at all -
and can always be interrupted with Ctrl-C as soon as something looks
wrong.

No single tool result can fill the context: each one is capped at about a
quarter of the model's context window (32,000 characters if the window is
unknown), keeping its beginning and end with a note in between saying what
was left out. `read_file` stops at a whole line instead, and its note gives
the `start_line` to continue from; it also accepts `start_line`/`end_line` to
read just a range.

Within a turn, once the conversation reaches about 75% of the context window
between tool calls, faber shortens the oldest large tool results before
sending the next request, so a long task (e.g. reading a big file chunk by
chunk) doesn't fill the context and get its reply cut off. This uses the
token counts the server reports (faber asks for them in streamed responses
with `stream_options.include_usage`), or an estimate if it reports none.

If a request still fails because the conversation doesn't fit, the chat
first shortens large tool results and retries - keeping everything else,
including what the model already did in the current turn - up to three
times. Only if that's not enough does it summarize the history and retry
the request. Work done by tool calls earlier in the failed turn is included
in the summary, and your last message is kept verbatim after it.

The chat also summarizes *proactively*, before that ever happens: once the
last request's reported prompt-token count reaches about 80% of the model's
context window, the next request summarizes the conversation first instead of
risking the same failure. This needs to know the context window size, which
faber tries to look up automatically (in the background, so it doesn't delay
your first message) from the endpoint's `/models` listing - OpenRouter's
`context_length` or llama.cpp's `meta.n_ctx`; an endpoint that lists a single
model is taken to serve that one, whatever the configured model name. If the
lookup fails, proactive summarization and tool-result trimming never trigger
- only the reactive fallback above still applies. Use `--context-window <N>`
to set it explicitly (in tokens) instead of relying on the automatic lookup.

`/cost` shows accumulated token usage for the whole session (every agent, not
just the current one, plus sub-agents and `fan_out` workers) - prompt/completion/total tokens across every request
so far, including each intermediate tool-call round trip - plus an estimated dollar cost, using the same background `/models`
lookup's pricing if the endpoint provides it; otherwise just the token
counts, with no guessed price shown.

### Prompt (one-shot)

Send a single prompt and print the response:

```bash
faber prompt "Explain this code" src/main.rs src/lib.rs
```

Files listed after the prompt are loaded as system context.

### List models

```bash
faber models
faber --endpoint https://api.openai.com/v1 models
```

### List tools

```bash
faber list-tools
```

### Garbage collect dormant agents

```bash
faber --db-path state.db gc
```

Removes agents with no active session (stale heartbeat or no session_id),
except the `default` agent.

## Agents

Each agent has its own conversation history, system prompt, model, and
endpoint. Agents are stored in the SQLite database.

### Per-agent configuration

Set agent-specific configuration via the `agent_data` key-value store:

```
config:model       - Override the model for this agent
config:endpoint    - Override the API endpoint
config:system_prompt - Custom system prompt
```

The model sets or clears them with the `agent_configure` tool, and
`agent_get` shows them.

### Sub-agents

The `spawn_agent` tool launches a sub-agent in a background thread.
Sub-agents run independently and deliver their result as a notification
when complete. Multiple sub-agents can run concurrently.

The terminal status bar cycles through active sub-agents, showing each
one's name, status, and elapsed time. A `(+N more)` suffix indicates
how many additional agents are running.

Sub-agents are automatically cleaned up when they finish: the agent
entry is deleted from the database and the status bar entry is removed.

### Inter-agent messaging

Agents can send messages to each other using the `send_message` tool.
Messages are delivered as notifications and injected into the receiving
agent's next conversation turn - but only once a live session is actually
running that agent (claimed it, e.g. via `--agent` or `/select-agent`) and
polling for it. A message to an agent nobody is currently running just
waits in the database and is delivered whenever a session next picks that
agent up, however much later that is.

### Session ownership

Agents use session-based claiming with heartbeats. Only one session
can own an agent at a time. Ownership expires after 10 seconds without
a heartbeat, allowing recovery from crashes.

## Scheduled tasks

Tasks can be created via the `task_create` tool, or from the command line
(`faber tasks add`, below).

- **Cron tasks**: recurring, using 7-field cron expressions
  (`sec min hour day_of_month month day_of_week year`)
- **One-shot tasks**: fire once at a specific time or after a delay

Tasks execute tool calls (JSON format) when they fire:

```json
{"tool": "run_command", "arguments": {"command": "echo", "args": ["hello"]}}
```

Tasks support `max_runs` to stop after N executions. Every `faber chat`
session runs a scheduler that checks for due tasks every second.

Each task has a status:

| Status | Meaning |
|---|---|
| `scheduled` | waiting for its next run |
| `held` | scheduled, but not picked up by anyone until released, even when due |
| `running` | claimed by one session, which is running it |
| `done` | a one-shot task that ran, or a cron task that reached `max_runs` |
| `disabled` | switched off with `task_set_enabled` (switching it back on makes it `scheduled`) |

A session claims a due task atomically before running it, so when several
sessions share a database (e.g. through `faber serve`) each run happens
exactly once. If the session running a task dies, its claim is abandoned
once it stops heartbeating (after ~30 seconds) and the task runs again.
How the last run went is recorded separately - `last_outcome`
(`succeeded`/`failed`), `last_exit_code` and the start of its output - so a
cron task whose last run failed is still `scheduled` for the next one. A
task whose command isn't a valid tool call fails with an explanation.

### Creating tasks from the command line

```bash
faber tasks add "Tell me a joke about SQLite" --in 30s       # a prompt for an agent
faber tasks add "Summarize open issues" --cron "0 0 9 * * * *" --agent triage
faber tasks add "Check the build" --at "2026-10-03 08:00"
faber tasks add --tool '{"tool":"glob","arguments":{"pattern":"*.md"}}' --in 1m
```

A task's `kind` says what its command is:

- `prompt` (the default for `tasks add`): an instruction for an agent. Once
  it's due, the first `faber chat` session **waiting at its prompt** claims
  it - with `--agent NAME`, only a session whose current agent is `NAME` -
  and runs it as a turn of that agent's conversation, shown in that chat
  (Ctrl-C interrupts it, like a turn you typed; it's then recorded as
  failed).
  The agent's final answer becomes the task's result. Until some agent is
  waiting, `faber tasks` shows it as "waiting for an agent".
- `tool` (`--tool`): a tool call run directly by any session's
  scheduler, with no LLM.

The model's `task_create` tool makes either
kind, depending on the command: a `{"tool": ...}` call is a `tool` task,
plain language ("tell the user a joke") is a `prompt` task for the agent
that created it, carried out once its chat is idle again.

With no `--in`, `--at` or `--cron`, a task is due right away. `--max-runs N`
limits a `--cron` task.

A task can be put on hold, so that nobody picks it up - e.g. to create
work ahead of time and start it on a signal:

```bash
faber tasks add "Deploy the release notes" --hold   # created held
faber tasks hold 7 8                                 # hold scheduled tasks
faber tasks release 7 8                              # picked up again - right away if already due
```

Only a `scheduled` task can be held (a running one is already assigned)
and only a held one released; `faber tasks` shows when a held task would
run, and the board marks it ✋.

### Listing tasks

```bash
faber --db-path state.db tasks              # every task, most recently active first
faber --db-path state.db tasks --last 10    # the 10 most recently active
faber --db-path state.db tasks --since 2h   # active in the last 2 hours (also 30m, 3d, 1w)
faber --db-path state.db tasks --since 2026-10-01 --agent mybot
faber --db-path state.db tasks --watch      # redraw every 2s until Ctrl-C (--watch=N for N seconds)
faber --db-path state.db tasks --board --watch   # as a board, one column per state
faber --db-path state.db tasks show 7       # everything about task #7: full command, times, last result
faber --db-path state.db tasks show 7 --json
```

`--board` shows the tasks as cards in four columns - **Scheduled**
(including disabled ones, marked ⏸), **Waiting** (prompt tasks that are
due but no agent has picked up yet), **Running** and **Done** (✓ or ✗ with
the start of the result) - sized to the terminal, at most 10 cards per
column. `--since`, `--last` and `--agent` work with it too.

```
 ID | Name           | Agent | Schedule        | Status    | Next run | Last run | Runs | Last result
----+----------------+-------+-----------------+-----------+----------+----------+------+--------------------------
 3  | check every 2s | -     | */2 * * * * * * | done      | -        | 6s ago   | 3/3  | ✓ src/main.rs:7966:fn main…
 2  | broken command | -     | once            | done      | -        | 10s ago  | 1    | ✗ the task's command isn't…
 4  | nightly report | -     | 0 0 3 * * * *   | scheduled | in 13h   | -        | 0    | -
```

`faber tasks` only reads: it never creates a database (a wrong path is an
error, not an empty table) and never upgrades or changes one - open a
database from an older faber once with `faber chat` first. A relative
`db_path` in a config file is relative to the current directory.

A task is "active" when it was created, started or finished a run.

### Cleaning up

Done tasks are kept, with their results, until deleted:

```bash
faber --db-path state.db tasks prune --older-than 7d --dry-run   # list what would go
faber --db-path state.db tasks prune --older-than 7d             # delete them
faber --db-path state.db tasks prune --done                      # delete every done task, however recent
```

`prune` only deletes `done` tasks (✓ and ✗ alike) whose last run is older
than `--older-than` - `7d` by default, or no limit with `--done` alone;
scheduled, running and disabled tasks are never touched. To do it automatically, set `"task_retention": "30d"` in the
config file (or pass `--task-retention 30d`): while a `faber chat` runs,
it prunes done tasks older than that every hour. Deleting an agent also
deletes its tasks, and the model can delete one with `task_delete`.
`--since` also takes a date or `"YYYY-MM-DD HH:MM"` in local time, or an
RFC 3339 timestamp. It works with `--server` too.

## Knowledge base

Agents keep a knowledge base of notes - project facts, decisions and why,
how-tos, conventions, gotchas, your preferences - in faber's SQLite
database, so it outlives conversations and is shared by every agent (and,
with `faber serve`, every client). The model uses it through the `kb_*`
tools:

| Tool | Description |
|---|---|
| `kb_search` | Ranked full-text search, with a snippet of each match. Common words like "how" or "the" are ignored, and words match their variants ("deploying" finds "deploy") |
| `kb_read` | A note in full, by id or title |
| `kb_list` | Notes by most recently updated, optionally by tag |
| `kb_write` | Saves a note (title, Markdown body, tags). Writing a title that already exists - ignoring case - replaces that note instead of adding a near-duplicate. Shared with all agents unless `private` |
| `kb_delete` | Deletes a note |

Search ranks matches in a note's title above its tags, and tags above its
body. There's no default system prompt telling the model about the
knowledge base; the tool descriptions tell it to search before asking you
or re-investigating something, and to save what's worth keeping.

You can look at and curate it from the command line - `faber kb` sees every
note, private ones included:

```bash
faber kb                                   # list notes, most recently updated first (--tag, --last N)
faber kb search "deploy staging"           # ranked search
faber kb show 3                            # a note in full (or by title: faber kb show "Release process")
faber kb add "Release process" "Cut every other Tuesday..." --tag release
echo "..." | faber kb add "Code style"     # the body from stdin
faber kb rm 3 4
```

A note is either **shared** - every agent sees it, and it outlives whoever
wrote it - or **private** to the agent that wrote it. A private note is
also seen by the sub-agents that agent starts, at any depth (`fan_out`
workers act as the agent that called them), but never by the agents above
it: a sub-agent can keep notes for itself and its own sub-agents that its
parent doesn't see. Each agent records which agent spawned it, and if a
title exists at several levels, the nearest note wins. Deleting an agent
deletes its private notes; its own sub-agents, if any are left, become
top-level agents. `faber kb add --agent NAME` makes a note private to
that agent.

## Tools

### Safe tools (always available)

| Tool | Description |
|---|---|
| `read_file` | Read file contents, optionally just a `start_line`..`end_line` range; reports `total_lines` |
| `write_file` | Create a file, or replace its whole content, with optional permissions. Anything else is refused and pointed at `patch_file` |
| `patch_file` | Change part of an existing file: a batch of edits, each replacing exact text (`old_content`) or a range of lines (`start_line`..`end_line`), applied in order, all-or-nothing, rewriting only the changed bytes; the result includes a numbered-context preview of where each edit landed, so a follow-up `read_file` usually isn't needed to confirm it |
| `delete_path` | Delete a file or directory |
| `glob` | Find files matching a glob pattern |
| `grep_in_current_directory` | Search file contents with a regex, using [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) if installed and `grep` otherwise. Skips `.gitignore`d, hidden and binary files (the `grep` fallback skips `.git`, `target` and `node_modules` instead); optional `path`, `glob`, `case_insensitive`, `fixed_strings`, `context_lines`, `files_only`, `include_ignored`; output sorted by path and cut off after `max_results` lines (default 200) with a note. Unless `--unsafe-tools` is set, the search runs in a bubblewrap sandbox like `run_command`'s, but with the current directory mounted read-only |
| `github_issue` | One GitHub issue by number (optionally with its comments), or the issues updated in the last few days |
| `github_pull_request` | One pull request by number (or its diff, with `patch`), or the pull requests updated in the last few days |
| `agent_create` | Create a new agent |
| `agent_delete` | Delete an agent |
| `agent_list` | List all agents |
| `agent_get` | Get an agent's details, including its configuration |
| `agent_configure` | Set or clear an agent's model, endpoint or system prompt |
| `lsp` | Ask a language server about code: `definition`, `references`, `hover`, `symbols` (a file's outline), `workspace_symbols` or `diagnostics`. The server is picked by file extension - rust-analyzer (`.rs`), clangd (C/C++), pyright-langserver or pylsp (`.py`), gopls (`.go`), typescript-language-server (JS/TS) - whichever is installed, started on first use and kept running. A symbol is given by `line` plus its text on that line (`symbol`). Files are re-synced on every call, so edits are picked up. Language servers can run project code (e.g. rust-analyzer builds `build.rs` and proc macros), so unless `--unsafe-tools` is set they run sandboxed with [bubblewrap](https://github.com/containers/bubblewrap): the whole filesystem read-only (so toolchains under `$HOME` still work), only the current directory writable, and no network |
| `plan_update` | Set the agent's plan for the current multi-step task (full list of items, each `pending`/`in_progress`/`completed`). Stored per agent in the DB under the `state:plan` key, shown in the status bar as progress, and cleared once every item is completed, on `/clear`, or when the agent is deleted (including a sub-agent when it finishes) |
| `plan_get` | Get the agent's current plan |
| `task_create` | Create a task: on a cron schedule, once after a delay or at a time, or right away |
| `task_delete` | Delete a scheduled task |
| `task_list` | List tasks (all, an agent's, or only the due ones), or get one by id |
| `task_set_enabled` | Enable or disable a task |
| `send_message` | Send a message to another agent |
| `spawn_agent` | Spawn a sub-agent for parallel work |
| `fan_out` | Run the same task for many items (e.g. files) at once, one worker agent each, at most `max_parallel` at a time (default 32, up to 256; up to 1000 items), and return all the results together, in item order, once every worker is done. `{item}` in the prompt is replaced by each worker's item. Workers only get read-only tools (`read_file`, `glob`, `grep_in_current_directory`, `lsp`, web/GitHub reads) unless `tools` names others - workers that write can overwrite each other's changes - and can't spawn agents. Ctrl-C stops every worker. Each result gets a share of the output cap |
| `run_command` | Execute a command, sandboxed with [bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`): no network access, no capabilities, a cleared environment, a read-only root with only the current directory writable, its own PID/IPC/UTS/cgroup namespaces (no visibility into other processes or the host's hostname), killed if faber itself dies, and detached from the controlling terminal. Requires `bwrap` to be installed; use `--unsafe-tools` for unrestricted execution instead |

### Unsafe tools (require `--unsafe-tools`)

| Tool | Description |
|---|---|
| `run_command` | Execute a command directly, with the same access as the faber process itself - no sandboxing |
| `fetch_web_content` | Fetch content from a URL |

### Tool filtering

```bash
faber --tools read_file,write_file,glob chat   # only these tools
faber --no-tools chat                           # disable all tools
faber --tool-choice required chat               # force tool usage
```

### Language servers

The `lsp` tool picks a server by file extension from a built-in table -
`rust` (rust-analyzer), `c` (clangd), `python` (pyright-langserver, else
pylsp), `go` (gopls), `typescript` (typescript-language-server). The config
file's `lsp_servers` changes it: an entry named like a built-in replaces
it, `null` removes it, and any other name adds a server:

```json
{
  "lsp_servers": {
    "zig": { "command": ["zls"], "extensions": ["zig"] },
    "python": { "command": ["pylsp"], "extensions": ["py", "pyi"] },
    "rust": {
      "command": ["rust-analyzer"],
      "extensions": ["rs"],
      "settings": { "cargo": { "features": "all" } }
    },
    "go": null
  }
}
```

`command` is the executable (looked up on `PATH`, or a path) and its
arguments. `language_id` sets the LSP language identifier, if the
extension isn't a well-known one - it defaults to the extension itself.
`settings` is passed to the server as its `initializationOptions` and as
the answer to its `workspace/configuration` requests. faber refuses to
start if two servers claim the same extension. Configured servers are
sandboxed exactly like the built-in ones.

### Parallel tool calls

When the model asks for several tool calls in one turn, calls that can't
interfere with each other run concurrently: file reads, `glob`/`grep`,
GitHub and DB reads, `fetch_web_content`, and `write_file`/`patch_file` on
different paths. A write conflicts with reads or writes of the same path and
with `glob`/`grep`. Everything else (`run_command`, `delete_path`,
sub-agents, DB/task writes, MCP tools) runs alone. Conflicting calls always
run in the order the model gave them. Output from calls that ran
concurrently is buffered and shown one tool at a time once they all finish.

## MCP (Model Context Protocol)

faber can connect to external tool servers using the MCP protocol.
Three transports are supported:

### Stdio (local process)

```json
{
  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": {"NODE_ENV": "production"}
    }
  }
}
```

### HTTP (remote server)

```json
{
  "mcp_servers": {
    "remote": {
      "url": "http://localhost:3000/mcp",
      "headers": {"Authorization": "Bearer token"}
    }
  }
}
```

### SSE (Server-Sent Events)

```json
{
  "mcp_servers": {
    "sse-server": {
      "sse": "http://localhost:3000/sse",
      "headers": {"Authorization": "Bearer token"}
    }
  }
}
```

### Adding a remote server from the command line

`--mcp-server` is a shortcut for adding an HTTP or SSE server without
editing the config file - useful for a one-off server or for overriding a
config file entry:

```bash
faber --mcp-server search=http://localhost:3000/mcp chat        # HTTP
faber --mcp-server search=sse:http://localhost:3000/sse chat    # SSE
```

The format is `NAME=URL` for HTTP, or `NAME=sse:URL` for SSE. Can be
repeated for multiple servers, and adds to (rather than replaces) any
`mcp_servers` from the config file; a name that matches a config file entry
overrides it. There's no CLI shortcut for stdio servers or per-server
headers - use the config file for those.

MCP tools are prefixed with `mcp_<server>_<tool>` to avoid name collisions.
Use `/mcp-refresh` in chat to reload tool definitions.

## Persistence

When `--db-path` is provided, the following state is persisted in SQLite:

- Agent definitions and configuration
- Agent key-value data store
- Conversation history per agent
- Scheduled tasks (cron and one-shot)
- Inter-agent notifications
- Readline history (Ctrl-R search works across sessions)

Without `--db-path`, readline history uses in-memory storage and
agent/task features are unavailable.

## Multi-node (server mode)

Start a server that exposes the database over TCP so remote instances
can share state:

```bash
faber --db-path state.db serve --bind 127.0.0.1:9090
faber --db-path state.db serve --bind 0.0.0.0:9090 --auth-key mysecret
faber --db-path state.db serve --auth-key-file /path/to/keyfile
```

Clients connect with `--server` instead of `--db-path`:

```bash
faber --server 127.0.0.1:9090 --server-key mysecret chat
```

The protocol is newline-delimited JSON over TCP. TLS is not built in --
use SSH tunneling, WireGuard, or a reverse proxy for encryption:

```bash
ssh -L 9090:localhost:9090 remote-host
faber --server 127.0.0.1:9090 --server-key mysecret chat
```

## Options

```
-c, --config <PATH>          Path to JSON configuration file
-m, --max-tokens <N>         Maximum tokens to generate
    --model <MODEL>          AI model to use
    --endpoint <URL>         API endpoint URL
    --no-tools               Disable all tools
    --unsafe-tools           Run commands unsandboxed and enable fetch_web_content
    --tools <LIST>           Comma-separated list of tools to enable
    --tool-choice <MODE>     Tool usage mode: auto, none, required
    --api-key <PATH>         File containing the API key
    --parameter <K=V>        Model parameter (can be repeated)
    --mcp-server <N=URL>     Add a remote MCP server (can be repeated); see MCP section
    --db-path <PATH>         SQLite database for persistent storage
    --agent <NAME>           Start chat as this agent instead of 'default'
    --display-graphics       Render LaTeX blocks as images on terminals that support it; see below
    --server <ADDR>          Connect to a remote faber server
    --server-key <KEY>       Pre-shared key for server authentication
    --server-key-file <PATH> Read server key from file
    --max-parallel-requests <N>  At most N model requests in flight at once (see below)
    --task-retention <DURATION>  Prune done tasks older than this while chatting
```

`--max-parallel-requests` (or `"max_parallel_requests"` in the config file)
caps how many requests to the model are in flight at once across
everything one faber process does - the chat, sub-agents, `fan_out`
workers and scheduled task turns. Set it to the number of requests your
server can serve at once, e.g. llama.cpp's `--parallel`: with more, they
queue there anyway, and each llama.cpp slot only gets its share of the
context. Requests over the limit wait their turn, shown in the status bar
("Waiting for other requests to the model to finish"); Ctrl-C still
interrupts them. Unlimited unless set.

### Displaying LaTeX as images

```bash
faber --display-graphics=partial chat
faber --display-graphics=full chat
```

Off by default; an explicit `partial` or `full` value is required (there's
no bare `--display-graphics` - since this flag has to come before the
subcommand, e.g. `chat`, a value-less form would make the CLI parser try
to consume the subcommand's own name as this flag's value instead, and
fail). Both modes need a supported terminal - currently Kitty's graphics
protocol, a no-op elsewhere - and a local LaTeX toolchain (see below).

**`partial`** watches each response - and, since it isn't otherwise kept
anywhere, any reasoning text the model streams separately - for `\[...\]`,
`\(...\)`, and `$$...$$` LaTeX blocks, and renders each one as an image.
The image replaces the block's raw text *in place*, the moment that block
finishes streaming in, rather than appearing separately after the whole
response is already printed - so it reads as part of the answer instead of
a disconnected pile of images at the end. If a particular block fails to
render (e.g. it doesn't actually compile), its raw text is printed instead,
right where the image would have gone, so nothing is ever silently
dropped. Status bar text reads `Rendering LaTeX` while a block is
compiling, since each one is a separate, non-instant render.

**`full`** prints nothing live while the response streams in (just the
usual `Streaming (N bytes)` status text, so it's clear something's still
happening) - there's no way to know in advance whether the response will
even end up rendering successfully once it's done, and printing it live
regardless would leave that raw text in the terminal - and copy-pasteable
from it - even after a successful render made it redundant. Once the
response (and separately, its reasoning, which isn't otherwise kept
anywhere - see below) is done, each is rendered as one properly typeset
document and shown: real Markdown (headings, bold/italic, lists, block
quotes, tables, code blocks/spans, links, images, task lists) converted to
LaTeX, with the math blocks preserved exactly as written and reading as
part of their surrounding sentence, rather than a small isolated image
dropped into plain text. Markdown conversion is deliberately not
exhaustive: links/images render as their text/alt-text alone (a static
image can't be clickable, and downloading a linked image is out of scope),
and tables get a plain left/center/right-aligned `tabular`, no fancier
styling. If a document fails to render, its raw text is printed instead -
this is the only case `full` ever shows raw text at all, so nothing the
model wrote is silently lost.

Rendering shells out to a local LaTeX toolchain (`pdflatex` and
`pdftocairo`, from a LaTeX distribution and poppler-utils respectively -
neither is bundled with faber). The LaTeX being rendered comes from the
model, so it's treated as untrusted input like any other and run under
bwrap: no network, no capabilities, only a scratch directory writable. Its
filesystem sandboxing differs from `run_command`'s own, though - rather
than a fresh empty root with a curated list of read-only binds, it's the
whole real filesystem bound read-only, with only the scratch directory
re-bound read-write on top, and the real environment passed through rather
than cleared. LaTeX's own file-finding library looks for its files
(most prominently pdflatex's own precompiled format file) in places that
vary by distro and install with no reliable way to enumerate them all
upfront, so rather than guessing which ones to bind, the whole read-only
filesystem is exposed instead - an acceptable tradeoff here since
pdflatex/pdftocairo are two fixed, known binaries being fed data, not an
arbitrary command chosen by the model. If the terminal isn't recognized as
supported or the toolchain isn't found, faber
says so once at chat startup rather than staying silent about why nothing
is being rendered; a per-block rendering failure is also logged (not just
shown as the raw-text fallback above), but quietly - by default nothing
shows even then (faber's own default log level is `error`), so set
`RUST_LOG=warn` to see the real error for a block that failed to render.

The rendered image has a transparent background, so it blends into the
terminal instead of standing out as its own rectangle, and its text color
is queried from the terminal itself (the same ANSI palette slot the answer
text renders in), so it matches that terminal's theme instead of a
hardcoded guess.

`pdflatex` needs these LaTeX packages installed - a "not found" compile
error for one of them (visible with `RUST_LOG=warn`) means it's missing
from the local install, not a bug in what faber generated:

- Both modes: `standalone` (and the `preview` package it builds on),
  `amsmath`, `amssymb`, `xcolor`.
- `full` only: `inputenc`, `ulem`, `varwidth` (used via `standalone`'s own
  `varwidth=` option), and the base `article` class (used via
  `standalone`'s `class=article` option - almost certainly already present
  in any real install, since it's one of LaTeX's own standard classes).

On TeX Live (e.g. Fedora's `texlive-*` packages), a `texlive-scheme-basic`
install typically needs at least `texlive-standalone`,
`texlive-preview`, `texlive-varwidth`, and `texlive-ulem` added
individually; `texlive-scheme-full` (or Debian/Ubuntu's
`texlive-latex-extra`) already includes all of the above.

### Model parameters

Use `--parameter` to tune model behavior:

```bash
faber --parameter temperature=0.7 chat
faber --parameter temperature=0.2 --parameter top_p=0.9 chat
```

Parameter types are auto-detected: numbers, booleans (`true`/`false`),
`null`, or strings.

## Scripted model, for testing workflows

`--model script:PATH` replaces the model with a script, to try agent
workflows - sub-agents, `fan_out`, scheduled tasks - deterministically and
without a model server. The script is a JSON list of rules; for each
request, the first rule whose `if` text appears in the latest message (a
user turn, an injected message, or a tool's result) gives the response, as
`reply` text or a `tool` call:

```json
[
  {"if": "triage", "tool": "fan_out",
   "arguments": {"items": ["101", "102"], "prompt": "check issue {item}"}},
  {"if": "check issue", "reply": "looks fine", "delay_ms": 200},
  {"if": "## 1.", "reply": "triaged both"}
]
```

Rules match on content rather than order, so agents running concurrently
stay deterministic. `once: true` makes a rule apply only the first time it
matches, and `delay_ms` makes a response slow (Ctrl-C still interrupts
it). A request no rule matches is answered with a note quoting the message.

## Building

```bash
cargo build --release
```

## License

faber is licensed under the GNU General Public License v2.0 or later.
