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

Reading a large file in full sends its entire content to the model as
context, which both grows every later request and can turn the next one
into the kind of slow prompt above; `read_file` accepts `start_line`/
`end_line` to read just a range, and reports `total_lines` plus a note
suggesting that range when a full read of a large file is requested.

When a request fails because the conversation no longer fits in the model's
context window, the chat summarizes the history automatically and retries the
request once.  Work done by tool calls earlier in the failed turn is included
in the summary, and your last message is kept verbatim after it.

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

These can be set through the `agent_data_set` tool or directly in the
database.

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

Tasks can be created via the `task_create_cron` and `task_create_oneshot`
tools.

- **Cron tasks**: recurring, using 7-field cron expressions
  (`sec min hour day_of_month month day_of_week year`)
- **One-shot tasks**: fire once at a specific time or after a delay

Tasks execute tool calls (JSON format) when they fire:

```json
{"tool": "run_command", "arguments": {"command": "echo", "args": ["hello"]}}
```

Tasks support `max_runs` to auto-disable after N executions. A background
scheduler thread checks for pending tasks every second.

## Tools

### Safe tools (always available)

| Tool | Description |
|---|---|
| `read_file` | Read file contents, optionally just a `start_line`..`end_line` range; reports `total_lines` |
| `write_file` | Create/overwrite a file with specified permissions, or partially edit an existing one via byte `offset`+`length`, `start_line`..`end_line`, or exact-text `old_content` search-and-replace (one mode at a time) |
| `patch_file` | Apply a batch of search-and-replace edits to a file, all-or-nothing, rewriting only the changed bytes; the result includes a numbered-context preview of where each edit landed, so a follow-up `read_file` usually isn't needed to confirm it |
| `delete_path` | Delete a file or directory |
| `glob` | Find files matching a glob pattern |
| `grep_in_current_directory` | Search for a pattern in the current directory |
| `github_issue` | Get a GitHub issue |
| `github_issue_comments` | Get comments on a GitHub issue |
| `github_issues` | List recent issues in a repository |
| `github_pull_request` | Get a GitHub pull request |
| `github_pull_request_patch` | Get the raw patch for a PR |
| `github_pull_requests` | List recent pull requests |
| `agent_create` | Create a new agent |
| `agent_delete` | Delete an agent |
| `agent_list` | List all agents |
| `agent_get` | Get agent details |
| `agent_data_set` | Store a key-value pair for an agent |
| `agent_data_get` | Retrieve a value for an agent |
| `agent_data_delete` | Delete a key-value pair |
| `agent_data_list` | List all key-value pairs for an agent |
| `task_create_cron` | Create a recurring scheduled task |
| `task_create_oneshot` | Create a one-shot scheduled task |
| `task_delete` | Delete a scheduled task |
| `task_list` | List scheduled tasks |
| `task_get` | Get task details |
| `task_set_enabled` | Enable or disable a task |
| `task_pending` | List tasks that are due to run |
| `send_message` | Send a message to another agent |
| `spawn_agent` | Spawn a sub-agent for parallel work |
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
```

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

## Building

```bash
cargo build --release
```

## License

faber is licensed under the GNU General Public License v2.0 or later.
