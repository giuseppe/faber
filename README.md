# swarmblabla

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
swarmblabla chat
swarmblabla --db-path state.db chat          # with persistence
swarmblabla --db-path state.db --agent mybot chat  # start as a specific agent
```

### Chat commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/quit` | Exit the chat session |
| `/clear` | Clear chat history and restore system prompts |
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

Commands can also use `\` as the prefix (e.g. `\quit`).

When a request fails because the conversation no longer fits in the model's
context window, the chat summarizes the history automatically and retries the
request once.  Work done by tool calls earlier in the failed turn is included
in the summary, and your last message is kept verbatim after it.

### Prompt (one-shot)

Send a single prompt and print the response:

```bash
swarmblabla prompt "Explain this code" src/main.rs src/lib.rs
```

Files listed after the prompt are loaded as system context.

### List models

```bash
swarmblabla models
swarmblabla --endpoint https://api.openai.com/v1 models
```

### List tools

```bash
swarmblabla list-tools
```

### Garbage collect dormant agents

```bash
swarmblabla --db-path state.db gc
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
agent's next conversation turn.

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
| `read_file` | Read file contents |
| `write_file` | Create or overwrite a file with specified permissions |
| `patch_file` | Apply a batch of search-and-replace edits to a file, all-or-nothing, rewriting only the changed bytes |
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

### Unsafe tools (require `--unsafe-tools`)

| Tool | Description |
|---|---|
| `run_command` | Execute a shell command |
| `fetch_web_content` | Fetch content from a URL |

### Tool filtering

```bash
swarmblabla --tools read_file,write_file,glob chat   # only these tools
swarmblabla --no-tools chat                           # disable all tools
swarmblabla --tool-choice required chat               # force tool usage
```

## MCP (Model Context Protocol)

swarmblabla can connect to external tool servers using the MCP protocol.
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
swarmblabla --db-path state.db serve --bind 127.0.0.1:9090
swarmblabla --db-path state.db serve --bind 0.0.0.0:9090 --auth-key mysecret
swarmblabla --db-path state.db serve --auth-key-file /path/to/keyfile
```

Clients connect with `--server` instead of `--db-path`:

```bash
swarmblabla --server 127.0.0.1:9090 --server-key mysecret chat
```

The protocol is newline-delimited JSON over TCP. TLS is not built in --
use SSH tunneling, WireGuard, or a reverse proxy for encryption:

```bash
ssh -L 9090:localhost:9090 remote-host
swarmblabla --server 127.0.0.1:9090 --server-key mysecret chat
```

## Options

```
-c, --config <PATH>          Path to JSON configuration file
-m, --max-tokens <N>         Maximum tokens to generate
    --model <MODEL>          AI model to use
    --endpoint <URL>         API endpoint URL
    --no-tools               Disable all tools
    --unsafe-tools           Enable unsafe tools (run_command, fetch_web_content)
    --tools <LIST>           Comma-separated list of tools to enable
    --tool-choice <MODE>     Tool usage mode: auto, none, required
    --api-key <PATH>         File containing the API key
    --parameter <K=V>        Model parameter (can be repeated)
    --db-path <PATH>         SQLite database for persistent storage
    --agent <NAME>           Start chat as this agent instead of 'default'
    --server <ADDR>          Connect to a remote swarmblabla server
    --server-key <KEY>       Pre-shared key for server authentication
    --server-key-file <PATH> Read server key from file
```

### Model parameters

Use `--parameter` to tune model behavior:

```bash
swarmblabla --parameter temperature=0.7 chat
swarmblabla --parameter temperature=0.2 --parameter top_p=0.9 chat
```

Parameter types are auto-detected: numbers, booleans (`true`/`false`),
`null`, or strings.

## Building

```bash
cargo build --release
```

## License

swarmblabla is licensed under the GNU General Public License v2.0 or later.
