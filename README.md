# faber

A multi-agent AI CLI tool for interactive software development. Supports
multiple concurrent agents, sub-agent spawning, inter-agent messaging,
scheduled tasks, MCP tool servers, and persistent state via SQLite.

## Setup

By default the tool uses `http://localhost:8080` as the API endpoint. Override
with `--endpoint`. If the endpoint requires authentication, point `--api-key`
at a file containing the key.

### Configuration file

Settings can be stored in a JSON file. By default
`~/.config/faber/config.json` (under `$XDG_CONFIG_HOME` when it's set) is
loaded if it exists - never one in the current directory, where agents
work and could write one. Use `-c`/`--config` to specify a different path.
CLI arguments override config file values.

Neither the config file nor the database may be inside a directory agents
work in: `faber chat`, `prompt`, `worker` and `serve` refuse to start
there, and an agent's or a task's working directory can't be set to one.
An agent without the unsafe tools can write where it works, and could
otherwise give itself the unsafe tools in the database, add a command to
run to the config file, or read every agent's conversation. Keep them
elsewhere, e.g. `"db_path": "~/.local/share/faber/faber.db"`.

```json
{
  "model": "google/gemini-2.5-pro",
  "endpoint": "http://localhost:8080",
  "api_key": "~/.keys/openai",
  "db_path": "~/.local/share/faber/faber.db",
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
faber --server host:9090 --agent coder:~/coder.json --agent fast:~/fast.json chat
```

`--agent NAME:CONFIG` runs the agent with the model, endpoint, `api_key`,
`max_tokens`, `context_window`, `parameter` and `tools` of that config
file, over whatever is stored for it - in this process only: they're
never written to the database, so a session on a shared server doesn't
change the agent for anyone else. Give `--agent` more than once to define
several, and switch between them with `/select-agent`; the first is the
chat's.

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
| `/cancel <name>` | Stop a running sub-agent |

Commands can also use `\` as the prefix (e.g. `\quit`).

`/chdir` is explicit and user-typed, so unlike `run_command` it isn't
sandboxed - it changes the real process directory. Every path-resolving
tool (`read_file`, `write_file`, `patch_file`, `glob`,
`grep`, `run_command`'s sandbox bind, ...) re-resolves
the current directory on each call, so they immediately follow a `/chdir`
with no extra step.

Reasoning ("thinking") tokens from models that stream them separately are shown
in grey italics.  They are not kept in the conversation history, so they do not
use up the context window on later requests - unless the request asks the model
to preserve its reasoning, as z.ai's GLM models do with
`"thinking": {"type": "enabled", "clear_thinking": false}` (their default on
z.ai's Coding Plan endpoint, but faber only knows it's on when it's set). Then
each response's reasoning is kept with it, exactly as it came, and sent back
with the conversation: the model carries on from its earlier thinking instead
of working its way there again on every step, and its server's prompt cache
covers more of each request. Set it as a parameter, e.g.
`--parameter 'thinking={"type":"enabled","clear_thinking":false}'`, or in a
profile's `parameters`. Through a proxy that only takes OpenAI's own
parameters, such as LiteLLM, put it in `extra_body` instead:
`--parameter 'extra_body={"thinking":{"type":"enabled","clear_thinking":false}}'`. If the model stops because it hit
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

Each agent can have its own settings, over the session's, kept in the
`agent_data` key-value store:

```
config:model          - the model
config:endpoint       - the API endpoint
config:system_prompt  - added to its system prompt
config:api_key        - file holding the API key
config:max_tokens     - maximum tokens to generate
config:context_window - the model's context window
config:parameters     - request parameters (JSON), over --parameter's
config:tools          - the only tools it may use (JSON list), of the session's
config:profile        - the profile it was made from, if any
config:unsafe_tools   - whether it has the unsafe tools (see Unsafe tools)
config:cwd            - the directory it works in
```

The model sets or clears the model, endpoint and system prompt with the
`agent_configure` tool, or replaces them all with a profile's; `agent_get`
and `faber agents show NAME` show them.

### Working directory

Each agent can work in its own directory instead of the current one of
whatever runs it: its file tools' relative paths are relative to it,
its commands run there, the sandbox lets it write only there, and the
language servers are started for it. Several agents in one chat, worker
or server can each work on a different project.

Setting it is the user's to do:

```bash
faber agents set reviewer --cwd ~/src/other-project
faber agents set reviewer --no-cwd       # back to wherever it's run
```

A task can have its own too - `faber tasks add --cwd DIR`, or the web
UI's New task form, which starts from the server's directory (or the
chosen agent's own) - and runs there whichever agent picks it up. An
agent is told in its instructions where it works, so it uses relative
paths instead of guessing absolute ones.

In a chat, `/cwd PATH` sets it for the current agent, `/cwd -` clears it,
and `/cwd` shows it (`/chdir` still changes the directory of the whole
process, for agents without their own). The web UI has it in each
agent's panel.

An agent an agent makes - a sub-agent, a worker's task agent - works
where its maker does. Only an agent with the unsafe tools can choose
another directory (`cwd` in `spawn_agent`, `agent_create`,
`agent_configure`, `task_create`); a safe one can't, nor reuse an agent the user set to
work elsewhere. A worker fails a task whose agent's directory doesn't
exist on its machine.

### Profiles

A profile is a named set of agent settings in the config file, to make
any number of agents with the same endpoint, model, effort and so on:

```json
{
  "profiles": {
    "fast": {
      "description": "quick, cheap checks",
      "endpoint": "http://gpu1:8080/v1",
      "model": "qwen3",
      "parameters": {"reasoning_effort": "low", "temperature": 0.2},
      "max_tokens": 4096,
      "tools": ["read_file", "glob", "grep", "report_result"]
    },
    "deep": {
      "model": "deepseek-r1",
      "api_key": "/home/me/.keys/deepseek",
      "parameters": {"reasoning_effort": "high"},
      "context_window": 128000,
      "system_prompt": "Think it through before answering."
    }
  }
}
```

Every field is optional: what a profile doesn't set comes from the
session (or, for a sub-agent, from the agent that started it). `tools`
lists built-in tools by name, and only narrows: a profile never gives an
agent a tool its session doesn't have, such as unsafe ones without
`--unsafe-tools`. `faber profiles` lists them.

A profile's `unsafe_tools` (`faber profiles set NAME --unsafe-tools`, or
`--safe-tools` to take it back; in the web UI's profile editor) is used
only for the agents of A2A contexts made from it; profile tasks,
`spawn_agent` and `fan_out` ignore it and follow their maker's tools. No
tool can set it.

An agent made from a profile gets a **copy** of its settings, so changing
the profile later doesn't change agents already made from it. Agents are
made from a profile by:

- `spawn_agent` with `profile` - a new sub-agent;
- `fan_out` with `profile` - every worker runs with its settings;
- `agent_create` with `profile`, and `agent_configure` with `profile` to
  re-copy one onto an existing agent;
- `faber --agent NAME --profile fast chat` - the chat's agent, made from it
  (or re-copied, if it exists);
- a task bound to the profile (see [Choosing where a task runs](#choosing-where-a-task-runs)).

The model is told which profiles there are, with their descriptions,
models and parameters, so it can pick one.

Profiles can also be kept in the database, made and changed in the web
UI (Profiles…): every chat and worker using the database has them, next
to its config file's - which win, if both have a name - and picks up
changes as they're made. An agent's Edit there can start it from any
profile.

### Sub-agents

The `spawn_agent` tool launches a sub-agent in a background thread.
Sub-agents run independently and deliver their result as a notification
when complete. Multiple sub-agents can run concurrently, and a sub-agent
can spawn its own, up to 8 levels deep.

A sub-agent can be given a budget: `timeout_seconds` stops it if it's still
working after that long, and `max_requests` once it has made that many
requests to the model. It can also be stopped on demand - by the agent
that started it (or one above it) with the `agent_cancel` tool, or by you
with `/cancel NAME` in the chat. A stopped sub-agent still reports back,
saying why: "Stopped: timed out after 60s", "Stopped: cancelled by the
user", and so on.

An agent can wait for its sub-agents with the `agent_wait` tool - all of
them, the ones it names, or (`any: true`) whichever finishes first, with a
timeout - and gets their results together, as one tool result, instead of
one message per sub-agent arriving later. With nothing to wait for (say,
after `fan_out`, which returns its results itself) it just says so.

Only a chat reads messages, so a sub-agent or a task whose turn ends with
sub-agents of its own still working doesn't end there: it waits for
their results and carries on with them, so they're never lost.

Delegated work - a sub-agent, a `fan_out` worker, or a scheduled task -
reports its outcome with the `report_result` tool: `succeeded` or
`failed`, a summary, and optional data. That's what the agent waiting on it
gets (a failure as "Failed: ..."), and what a task's record says, so a
failure counts as one even when the model words it politely. Two options
on `spawn_agent` and `fan_out` help delegation:

- `result_schema` asks for structured data, as field names and types -
  e.g. `{"duplicate": "boolean", "of": "integer"}`. A report that doesn't
  match is refused with what's wrong, so the agent corrects it; one that
  finishes without reporting is reminded once.
- `context` hands the new agent knowledge base notes (by title or id) and
  files up front, so it doesn't start cold:
  `{"notes": ["Deploying"], "files": ["src/main.rs"]}`.

`fan_out`'s workers are agents too, `<caller>-item-<n>`, under the agent
that fanned out: all of them show as soon as it starts - queued, until
their turn comes - and can be followed like sub-agents.

A finished sub-agent is kept, with its conversation, until `faber gc`
removes it (it then has no live session), so you can see what it did -
and its parent can run it again, by name. The agents faber makes for one
job - a fan-out's workers, a task's `task-<id>` or `<profile>-<id>` - go
by themselves: ten minutes after they're done (nothing runs them, nor
anything below them), with their conversations and events. Every chat
and worker removes them.

### Watching agents

```bash
faber agents                 # the agents as a tree by who spawned whom, with what each is doing
faber agents --watch         # redraw every 2s (--watch=N for N seconds)
faber agents show scout      # one agent in detail, with its conversation (--full for whole messages)
faber agents follow          # what every agent does, live: input, answers, tool calls, outcomes
faber agents follow scout    # only scout's (--reasoning to include its reasoning)
faber tasks follow 7         # everything done for task #7, by its agent, sub-agents and workers
faber agents message rev "look at src"     # a task for it: it gets it once it's free
faber agents set rev --profile deep --model qwen3-32b --tools read_file,glob   # its settings
faber agents cleanup --dry-run             # finished agents agents made (gc removes all idle ones)
faber profiles set deep --model big --system-prompt "Think."   # a profile in the database
faber profiles delete deep
```

What agents do is recorded in the database as it happens, wherever they
run - a chat, a `faber worker`, `faber serve --run-agent`, on this
machine or another sharing it through `faber serve` - so `follow` (and
the web UI) see every one of them, from any terminal: point it at the
same `--db-path`, or the same `--server`.

```
 Agent        | Session | Activity                        | Since | Model
--------------+---------+---------------------------------+-------+-------
 default      | live    | fan_out: 120/500 done           | 3s ago| -
 ├─ planner   | live    | running grep           | now | -
 │  └─ scout  | -       | finished: found the manifest    | 2m ago| -
 └─ checker   | -       | stopped: timed out after 60s    | 5m ago| -
```

Each agent's activity - thinking, running a tool, waiting for the model or
for a free request slot (see `--max-parallel-requests`), idle, or finished
/stopped/failed with the start of its result - is kept in the database,
so this works from any terminal, and with `--server` too.

The terminal status bar cycles through active sub-agents, showing each
one's name, status, and elapsed time. A `(+N more)` suffix indicates
how many additional agents are running.

A sub-agent's status bar entry is removed when it finishes; the agent
itself stays until `faber gc`.

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

A prompt task's run **carries on** where it stopped rather than starting
over, when it didn't finish: its agent's conversation is saved after every
round of tool calls, and the next run - by whichever chat or worker picks
it up - continues that conversation, told that it was cut short (the steps
after the last save are lost; what they did on disk isn't). That's when the
process running it is killed, or a worker is stopped with Ctrl-C, which
gives its tasks back instead of failing them. A run that does finish -
succeeded, failed, or stopped on purpose (`faber tasks stop`, Ctrl-C at a
chat) - is recorded as usual, and running the task again starts afresh.
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

A task can depend on others: it only becomes due once they're all done,
successfully, and fails as soon as one of them fails (or is deleted),
which in turn fails whatever depends on it. That's enough for a planner
agent to lay out a whole job as a chain or graph of tasks for agents to
work through:

```bash
faber tasks add "Write the release notes"                # 1
faber tasks add "Bump the version" --after 1             # 2
faber tasks add "Tag and publish" --after 1 --after 2    # 3
```

The model's `task_create` takes `depends_on` for the same. Until its
dependencies are done, `faber tasks` shows a task as "after #1, #2".

A prompt task is given what the tasks it runs after came to, their
results, along with its own text. And a task for an agent by name runs
as a turn of that agent's conversation, so it remembers what it did
before. Together, that's enough for work to go back and forth:

```bash
faber tasks add "Fix issue 42 and open a PR" --agent coder               # 1
faber tasks add "Review the PR for issue 42" --agent reviewer --after 1  # 2
faber tasks add "Address the review, push again" --agent coder --after 2 # 3
```

A task can also carry on where another left off: a task's conversation
is kept as it was when the task was done, and `--continue 1` runs, once
task #1 is done - however it went - on a new agent that starts from #1's:
a fork of the agent that ran it, with its settings, on whichever `faber
worker` is free (a chat doesn't pick these up). What that agent did
after #1 isn't part of it, and #1 can be carried on more than once, each
fork on its own. That's what it takes for an agent made for a task, from
a profile, which no session runs by name:

```bash
faber tasks add "Fix issue 42 and open a PR" --profile coder                # 1
faber tasks add "Review the PR for issue 42" --profile reviewer --after 1   # 2
faber tasks add "Address the review, push again" --continue 1 --after 2     # 3
```

The model's `task_create` takes `continue_task` for the same, and the web
UI has "Carry on…" on a task that's done. A task made by an agent
without the unsafe tools carries on one that had them without them.

#### Workflows with a conductor

Those pieces are enough for an agent to run a whole workflow: a
*conductor*, a profile whose system prompt is the playbook, delegating
every step to other profiles' agents with `task_create` and `task_wait`,
and deciding what comes next from what they report:

```json
{
  "profiles": {
    "fix-issue": {
      "description": "Runs the fix, review, fix again loop for an issue",
      "tools": ["task_create", "task_wait", "task_list", "plan_update", "report_result"],
      "system_prompt": "You run this playbook, delegating every step with task_create and never doing the work yourself.\n1. Have a coder (profile coder) fix the issue and open a PR.\n2. Have a reviewer (profile reviewer, depends_on the coder's task) review it; wait for it with task_wait.\n3. While the review asks for changes: have the same coder address them (continue_task: its task), then the same reviewer check its comments were addressed (continue_task: its task). At most 5 rounds.\n4. When approved, end your turn asking whether to merge. If told yes, merge."
    },
    "coder": {"description": "Writes the code"},
    "reviewer": {"description": "Reviews PRs; says approve, or which changes"}
  }
}
```

```bash
faber --unsafe-tools worker &          # runs the conductor and its steps
faber tasks add "Fix issue 42" --profile fix-issue          # 1
faber tasks                            # the conductor's tasks, as it makes them
faber tasks add "yes, merge" --continue 1                   # answering its question
```

- Each step is a task of its own, on the board and in the feed. The
  conductor's plan, if it keeps one with `plan_update`, shows in the web
  UI.
- `continue_task` lets the coder fix its own PR, and the reviewer check
  its own comments, remembering what they did.
- A reviewer that reports with `report_result`, e.g. `data: {"verdict":
  "changes", ...}`, gives the conductor something to branch on rather
  than prose.
- Its question is just how its task ends. Answering is carrying it on,
  from the terminal or "Carry on…" in the web UI, and the answer is
  added to everything it knew.
- From the web UI, the same is a new task for the `fix-issue` profile.

Its steps get the unsafe tools only if the conductor has them: started
by you on a worker with `--unsafe-tools`, it does, even with its `tools`
narrowed to the task tools as above. Nothing but the playbook limits how
many rounds it runs.

A task can be put on hold, so that nobody picks it up - e.g. to create
work ahead of time and start it on a signal:

```bash
faber tasks add "Deploy the release notes" --hold   # created held
faber tasks hold 7 8                                 # hold scheduled tasks
faber tasks release 7 8                              # picked up again - right away if already due
faber tasks run 7                                    # now: held, disabled, not due yet, or done (again)
faber tasks edit 7 "new text" --cron "0 0 9 * * * *" # change what's given, until it runs
faber tasks stop 7                                   # interrupt it while it runs, sub-agents and all
```

Only a `scheduled` task can be held (a running one is already assigned)
and only a held one released; `faber tasks` shows when a held task would
run, and the board marks it ✋.

### Choosing where a task runs

A prompt task runs on one of three:

- **an existing agent** (`--agent NAME`, `agent_name`): only a session
  running that agent - a chat, or `faber worker --agent NAME` - picks it up;
- **a new agent made from a profile** (`--profile NAME`, `profile`): a
  `faber worker` whose config file has the profile picks it up, and runs it
  on a new agent, `<profile>-<task id>`, made from the profile under the
  worker's agent (removed ten minutes after the task is done). Chats
  don't run these;
- **any agent** (neither): whichever chat or worker is free first. A
  worker runs it on a new agent of its own, `task-<task id>`, made with
  the worker's settings under the worker's agent.

```bash
faber tasks add "Review the diff in PR 12" --profile deep
faber tasks assign 7 8 --profile fast     # move tasks that haven't run yet
faber tasks assign 7 --agent triage
faber tasks assign 7 --any
```

The model does the same with `task_create`'s `profile` and the
`task_assign` tool. Which workers you start, and with which config files,
decides where each profile's tasks run - e.g. one machine's config points
`deep` at a big GPU, and `faber worker --profile deep` there takes only
those. Until a worker with the profile is running, `faber tasks` shows the
task as "waiting for a worker".

### Headless workers

`faber worker` works through tasks without a terminal, as an agent - like
a `faber chat` sitting idle, picking up prompt tasks for it (or for any
agent) and running tool tasks:

```bash
faber --db-path state.db worker --agent w1
faber --server host:9090 worker              # as worker-<pid>, against a shared server
```

A worker also takes on tasks bound to a profile in its config file - all
of them, or only those given with `--profile fast,deep` - each on a new
agent made from it (see [Choosing where a task runs](#choosing-where-a-task-runs)).

A worker runs any number of tasks at once, each on its own agent - a
`task-<id>` or `<profile>-<id>` one under the worker's, with its own
conversation and output - except tasks for the worker's agent by name,
which run as that agent, one at a time. What limits how many requests
reach the model at once is `--max-parallel-requests`; `--parallel N` caps
the tasks themselves, if you want that too.

Each task gets a fresh conversation (the agent's system prompt, then the
task), with every tool - including `spawn_agent`, `fan_out` and
`agent_wait` - and its outcome is recorded as usual. The worker logs a line
as each task starts and ends; `faber tasks` and `faber agents` show the
same from anywhere. Run several, next to a `faber serve`, and they share
the queue - each task is claimed by exactly one. Ctrl-C interrupts the
tasks still running and gives them back, to carry on from where they
stopped when next run (see above), releases the agent, and exits.

### Listing tasks

```bash
faber --db-path state.db tasks              # every task, most recently active first
faber --db-path state.db tasks --last 10    # the 10 most recently active
faber --db-path state.db tasks --since 2h   # active in the last 2 hours (also 30m, 3d, 1w)
faber --db-path state.db tasks --since 2026-10-01 --agent mybot
faber --db-path state.db tasks --profile fast   # tasks that run on the fast profile
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
 ID | Name           | Runs on | Schedule        | Status    | Next run | Last run | Runs | Last result
----+----------------+---------+-----------------+-----------+----------+----------+------+--------------------------
 3  | check every 2s | -       | */2 * * * * * * | done      | -        | 6s ago   | 3/3  | ✓ src/main.rs:7966:fn main…
 2  | broken command | -       | once            | done      | -        | 10s ago  | 1    | ✗ the task's command isn't…
 4  | nightly report | -       | 0 0 3 * * * *   | scheduled | in 13h   | -        | 0    | -
```

`faber tasks` only reads: it never creates or changes a database (a
wrong path is an error, not an empty table). A relative
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
| `write_file` | Create a file, or replace its whole content, with optional permissions. Replacing an existing file requires having read it, and is refused if it changed since (see below). Anything else is refused and pointed at `patch_file` |
| `patch_file` | Change part of an existing file: a batch of edits, each replacing exact text (`old_content`) or a range of lines (`start_line`..`end_line`), applied in order, all-or-nothing, rewriting only the changed bytes; the result includes a numbered-context preview of where each edit landed, so a follow-up `read_file` usually isn't needed to confirm it |
| `delete_path` | Delete a file or directory |
| `glob` | Find files matching a glob pattern |
| `grep` | Search file contents with a regex, using [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) if installed and `grep` otherwise, in the agent's working directory unless given `path` or `paths`. Skips `.gitignore`d, hidden and binary files (the `grep` fallback skips `.git`, `target` and `node_modules` instead); optional `glob`, `case_insensitive`, `fixed_strings`, `context_lines`, `files_only`, `count` (matching lines per file), `multiline` (ripgrep only), `include_ignored`; output sorted by path and cut off after `max_results` lines (default 200) with a note. For an agent without the unsafe tools it runs in a bubblewrap sandbox in which only the system's binaries and libraries and the working directory, read-only, exist: that alone keeps it in - a path outside the directory, or a symlink out of it, finds nothing - so any path may be given. |
| `github_issue` | One GitHub issue by number (optionally with its comments), or the issues updated in the last few days |
| `github_pull_request` | One pull request by number (or its diff, with `patch`), or the pull requests updated in the last few days |
| `agent_create` | Create a new agent |
| `agent_delete` | Delete an agent |
| `agent_list` | List all agents |
| `agent_get` | Get an agent's details, including its configuration |
| `agent_configure` | Set or clear an agent's model, endpoint (unsafe agents only) or system prompt |
| `lsp` | Ask a language server about code: `definition`, `references`, `hover`, `symbols` (a file's outline), `workspace_symbols` or `diagnostics`. The server is picked by file extension - rust-analyzer (`.rs`), clangd (C/C++), pyright-langserver or pylsp (`.py`), gopls (`.go`), typescript-language-server (JS/TS) - whichever is installed, started on first use and kept running. A symbol is given by `line` plus its text on that line (`symbol`). Files are re-synced on every call, so edits are picked up. Language servers can run project code (e.g. rust-analyzer builds `build.rs` and proc macros), so unless `--unsafe-tools` is set they run sandboxed with [bubblewrap](https://github.com/containers/bubblewrap): the whole filesystem read-only (so toolchains under `$HOME` still work), the environment cleared but for what toolchains need (`HOME`, `PATH`, locale, `CARGO_HOME`, `RUSTUP_HOME`, Go's, `JAVA_HOME`, ...), only the current directory writable, and no network. An agent without the unsafe tools never shares a server with one that has them, and the files it asks about are read inside its directory: a symlink out of it reads nothing |
| `plan_update` | Set the agent's plan for the current multi-step task (full list of items, each `pending`/`in_progress`/`completed`). Stored per agent in the DB under the `state:plan` key, shown in the status bar as progress, and cleared once every item is completed, on `/clear`, or when the agent is deleted |
| `plan_get` | Get the agent's current plan |
| `task_create` | Create a task: on a cron schedule, once after a delay or at a time, or right away |
| `task_delete` | Delete a scheduled task |
| `task_list` | List tasks (all, an agent's, or only the due ones), or get one by id |
| `task_set_enabled` | Enable or disable a task |
| `send_message` | Send a message to another agent |
| `spawn_agent` | Spawn a sub-agent for parallel work, optionally with a `timeout_seconds` or `max_requests` budget |
| `agent_cancel` | Stop a running sub-agent you started |
| `agent_wait` | Wait for your sub-agents (all, some, or the first to finish) and get their results together |
| `task_wait` | Wait for tasks (run by other agents) to finish, and get how each went |
| `report_result` | For a sub-agent, fan-out worker or task: report the outcome of its work - succeeded/failed, summary, data |
| `fan_out` | Run the same task for many items (e.g. files) at once, one worker agent each, at most `max_parallel` at a time (default 32, up to 256; up to 1000 items), and return all the results together, in item order, once every worker is done. `{item}` in the prompt is replaced by each worker's item. Workers only get read-only tools (`read_file`, `glob`, `grep`, `lsp`, web/GitHub reads) unless `tools` names others - workers that write can overwrite each other's changes - and can't spawn agents. Ctrl-C stops every worker. Each result gets a share of the output cap |
| `run_command` | Execute a command, sandboxed with [bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`): no network access, no capabilities, a cleared environment, a read-only root with only the current directory writable, its own PID/IPC/UTS/cgroup namespaces (no visibility into other processes or the host's hostname), killed if faber itself dies, and detached from the controlling terminal. Requires `bwrap` to be installed; use `--unsafe-tools` for unrestricted execution instead |

### Unsafe tools

| Tool | Description |
|---|---|
| `run_command` | Execute a command directly, with the same access as the faber process itself - no sandboxing |
| `fetch_web_content` | Fetch content from a URL |

(`grep` and `lsp` also run unsandboxed for an agent
with the unsafe tools.)

Whether an agent has them is up to each agent, not the whole session:

- `--unsafe-tools` gives them to the session's own agent: the chat's, a
  worker's, `faber serve --run-agent`'s.
- `faber agents set NAME --unsafe` gives them to an agent for good, in
  any session (`--safe` takes them away) - as does "unsafe tools" in the
  agent's Settings in the web UI. `faber agents` marks such agents
  `(unsafe)`, and so does the web UI.
- An agent an agent makes - `spawn_agent`, `agent_create` - is like its
  maker unless it asks otherwise (`unsafe_tools`). Only an agent that has
  the unsafe tools can give them, so a safe agent can only make safe
  ones; an unsafe one can make either. A safe agent can only change,
  delete or run (`agent_configure`, `agent_delete`, `spawn_agent` on an
  existing agent) the agents it made and theirs - not itself, the user's
  agents, other sessions', nor any agent with the unsafe tools - and can't
  set an endpoint at all: an agent pointed at a server of its choosing
  would send it its conversation and the API key, and run the tool calls
  it answers with. It sees only its own agents' endpoint, key file, system
  prompt and request parameters in `agent_get`. A profile never gives or
  takes the unsafe tools - except its `unsafe_tools` setting, which only
  the agents of A2A contexts made from it follow, and only the user can set.
- A task a safe agent creates (`task_create`) runs without them, whoever
  picks it up: a worker that has them runs it on a safe agent of its own,
  and a tool task gets the safe tools, in the task's working directory. A
  tool task only runs a tool its maker has, and a safe agent's tool task
  never writes into an agent's conversation: its result stays in the task.
  A safe agent can only enable, move or delete the tasks safe agents made,
  and sees only the name and state of the others - not their commands,
  nor their results.

- Like `run_command`, `send_message` comes in two versions: a safe
  agent's can only message agents without the unsafe tools - those that
  have them, for good or for the session running them, can't be asked
  to use them that way. As a message may wait for its agent, it's checked
  again when it's delivered: a chat with the unsafe tools drops safe
  agents' messages, other than from its own sub-agents.

- In the knowledge base, a safe agent doesn't see the private notes of
  the agents above it past the first one with the unsafe tools (a worker
  that has them runs its tasks on safe agents), and can't replace or
  delete a note the user or an unsafe agent wrote.

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

### Concurrent edits

Several agents - sub-agents, `fan_out` workers, task workers - and you can
change the same files. So that nobody's changes get silently overwritten,
each agent remembers what every file looked like when it last read or
wrote it, and `write_file` refuses to replace an existing file the agent
hasn't read, or one that has changed since: the agent has to read it
again and redo its change on top. `patch_file`'s text edits need no such
check - they only apply where `old_content` still matches - but its
line-range edits do, since line numbers mean nothing in a file that has
changed. A chat remembers across its turns, so this also protects your own
edits between them. Scheduled tool tasks, which run a fixed tool call
with no model involved, aren't checked.

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
headers - use the config file for those. `--mcp-server NAME`, with no URL,
names a server the config file defines.

MCP tools are prefixed with `mcp_<server>_<tool>`; servers whose tools
would end up with the same name (server `a_b`'s `c`, server `a`'s `b_c`)
are refused at startup. Use `/mcp-refresh` in chat to reload tool
definitions.

### Which agents get which servers

The config file and `--mcp-server` define the servers a faber process
runs; each agent gets the tools of some of them:

- The session's own agent - the chat's, a worker's, `faber prompt`'s - gets
  the servers its settings list, or else every one the session defines,
  plus those `--mcp-server` names. None with `--no-tools`.
- Every other agent - sub-agents, the agents tasks and fan-outs run on -
  only gets those its own settings list, from its profile or set by the
  user: none by default.

An agent's list is `"mcp_servers": ["github", ...]`, in a profile or an
agent's settings: `faber agents set NAME --mcp-servers github,jira`
(`none` for none, empty for the default), `faber profiles set`, or the
web UI. Agents can't change it - an MCP server runs with the user's
rights - so what a server may do is the user's choice, made with whom
they give it to. Servers run in the process that runs the agent: one
given on `faber chat`'s command line doesn't exist in a separate
`faber worker`, and an agent listing a server its process doesn't define
gets the others' tools only (with a warning in the log).

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
faber --db-path state.db serve --bind 0.0.0.0:9090 --auth-key-file /path/to/keyfile
```

Clients must give the server's key: whoever can, can do anything the
user can - give agents the unsafe tools, run commands. Without
`--auth-key-file`, `faber serve` makes a new key each start and writes
it, readable only by the user, to `$XDG_RUNTIME_DIR/faber/serve-<port>.key`
(else under `~/.local/state/faber/`). Clients on the same machine,
connecting to a loopback address, find it there by themselves; others
read it from a file. Keys are never given on the command line, where
`ps` would show them to every user. `--no-auth` serves without a key, on
a loopback address only: then any user or process on the machine is in
control.

Clients connect with `--server` instead of `--db-path`:

```bash
faber --server 127.0.0.1:9090 chat                         # this machine
faber --server faber-host:9090 --server-key-file ~/k chat  # another
```

The protocol is newline-delimited JSON over TCP. TLS is not built in --
use SSH tunneling, WireGuard, or a reverse proxy for encryption:

```bash
ssh -L 9090:localhost:9090 remote-host
faber --server 127.0.0.1:9090 --server-key-file ~/k chat
```

The server can also run agents itself, each working through tasks like
a `faber worker` would - any number at once:

```bash
faber --db-path state.db serve --run-agent default --run-agent reviewer
```

If a chat already has one of those agents, its worker waits until the
chat lets go of it. Ctrl-C stops the agents (interrupting their tasks),
then the server.

### Web UI

`faber serve` also serves a web UI, on the same address, at
`http://<bind>/ui/`, with two views:

- **Live**: every agent (as a tree, live or not, with what each is doing
  and the last thing it said), a feed of what all of them do as they do
  it - input, streamed answers and reasoning, each tool call with its
  result, how each turn ended - and the tasks. Pick an agent or a task to
  follow only that; a task's feed shows its runs, whichever agent ran
  them, and everything the sub-agents and fan-out workers they started
  did.
- A task's panel starts with what went wrong, if anything: why it
  failed, and every tool call that failed along the way. Failed tool
  calls are also red, and open, in the feed, and counted on the task's
  card.
- **Board**: the tasks as a scrum board - Backlog (held), To do, In
  progress (with the running agent's latest output), Done, Failed. Drag a
  card between Backlog and To do to hold or release it; disabled tasks
  are greyed out in the backlog. Every task not running has a run button
  (▶) - to start it now, whatever it was waiting for, or run it again -
  and every running one a stop button (■): whichever chat or worker runs
  it interrupts it, with its sub-agents and fan-out workers, and records
  it as failed ("stopped").

Agents are configured there too: "+ New agent" makes one - starting
from the settings of the last agent made there, or else `default`'s -
an agent's Clone makes a copy of it, and its Edit changes its
description, model, endpoint, system prompt, limits,
request parameters, tools, working directory and whether it has the
unsafe tools - the web UI is you, so it may do what agents can't do to
each other. Changes apply from the agent's next turn or task. A list of
an agent's sub-agents (a fan-out's workers) shows the first few until
asked for the rest.

A working agent's panel has Stop: whichever process runs it stops what
it's doing - the task it's running, or itself as a sub-agent or fan-out
worker (the rest of the fan-out carries on), or its turn in a chat. Its
Conversation view can clear the conversation of an agent nothing runs.
An agent's panel shows its plan, if it keeps one (`plan_update`), as a
checklist - and its row, how far along it is. An agent's panel has a
message box: what you send it reaches it however
it runs - a chat, a worker - as soon as it's free (it's a task for it),
and its answer shows in the feed. Its Conversation view shows the
messages it has kept. An agent a worker runs carries on its
conversation from one task for it to the next, as a chat does (cut to
fit the model's context window); agents made for a task start afresh.

New tasks can be added there too, choosing where they run: on any agent
(the first free chat or worker takes it - the default), on a given
agent, or on a new agent made from one of the profiles in `faber
serve`'s config file. Tasks can also be edited - what they say, where
and when they run, what they wait for, their directory - until they
run, and held, released, enabled, disabled, reassigned and deleted.
Run a tool… runs any tool - the knowledge base's `kb_*`, `glob`,
`run_command`... - from a form made from its parameters, and shows what
it returns: it's a tool task, run by whichever chat or worker is running,
with that session's tools, and kept like any task. Clean up… removes
finished tasks past an age, and the finished agents
agents made (sub-agents, task agents, fan-out workers) - never ones you
made, nor one with work still going on below it - showing what first.

What agents do is recorded in the database as it happens (see
`src/agent_io.rs`), by chats, workers and sub-agents alike, so the UI
sees agents in every process sharing the database - the last 2000
events per agent are kept, and go with the agent at `faber gc`.

The UI is built on a JSON API, for scripts too:

```bash
curl localhost:9090/api/agents
curl 'localhost:9090/api/events?agent=w1&after=120'        # or ?task=7
curl -H 'Content-Type: application/json' -d '{"command": "Fix the failing test", "profile": "fast"}' \
     localhost:9090/api/tasks
```

API requests need `Authorization: Bearer <key>`. The web UI gets the key
from the link `faber serve` prints - after `#`, which browsers don't send
to the server - or asks for it. With `--no-auth`, the API only answers
at `localhost` or an IP address, so a web page can't reach it through a
name it controls.

### A2A

`faber serve --a2a-keys-file FILE` also speaks the
[A2A protocol](https://a2a-protocol.org/) (v0.3, JSON-RPC) at `/a2a`, so
agents that aren't faber can use faber's. The line protocol isn't
touched; this is a translation layer over the same database.

`FILE` has one caller per line, `NAME KEY`, `#` for comments: `NAME` is
`[A-Za-z0-9_-]+` and unique, `KEY` at least 32 characters and not the
server's own key. It is read once, at start. A2A keys work on `/a2a` only
- never on `/api/` or the line protocol - and the server's key doesn't
work on `/a2a`: a caller can't make agents run commands by task, only
message the skills below. Without the option, `/a2a` and the card answer
404; there is no A2A without keys, even with `--no-auth`.

- **Discovery**: `GET /.well-known/agent-card.json`, without a key. One
  skill per profile **of the database** (`faber profiles set`, the web UI)
  - the profiles every worker sees, so any worker can run them. Profiles
  only in some process's config file aren't offered. The card's `url` is
  `http://` plus the request's `Host`: `faber serve` has no TLS, so put a
  reverse proxy in front of it for `https`, and mind that the keys travel
  in the clear without.
- **Methods**: `message/send`, `message/stream`, `tasks/get`,
  `tasks/cancel`, `tasks/resubscribe`. The bearer key goes in
  `Authorization`. `text` and `data` parts, in messages and artifacts.
- **Skills and contexts**: a message with no `contextId` starts a new
  context on the skill named in `message.metadata.skill` (or the only
  profile there is). A context is an agent made from the profile, with its
  own conversation, kept for the key that opened it: a follow-up message
  with the same `contextId` sees what came before, and messages to one
  context run one at a time, in order. Other callers can't see or use it -
  asking is answered as if it didn't exist. A context unused for 24 hours,
  with no task in progress, is removed with its agent and tasks. Only
  profiles are reachable: A2A can't message the agents you made, whose
  conversations are yours.
- **Tasks**: a message is a task, picked up by whichever `faber worker`
  or `serve --run-agent` is free, going `submitted`, `working`, then
  `completed`, `failed` or `canceled`. The answer is the task's `result`
  artifact: a `data` part if the agent's whole answer is a JSON object or
  array (alone or as one fenced `json` block), else a `text` part.
  `configuration.blocking` waits up to a minute. With
  `acceptedOutputModes` of `application/json` only, the agent is asked to
  answer with JSON only. An agent that needs more from the caller ends its
  turn with a question: the task is `completed`, and the answer goes in a
  new message with the same `contextId`.
- **Streaming**: `message/stream` and `tasks/resubscribe` answer with
  server-sent events: the task, `working` updates (one per tool call), the
  text as `result` artifact chunks, then a final status. The answer
  streams as text even when it is JSON; `tasks/get` then shows it as a
  `data` part. A stream counts as a connection, and closes after 30
  minutes without an event, to be resubscribed.
- **Safety**: a context's agent has the unsafe tools if, and only if, its
  profile says so (`faber profiles set NAME --unsafe-tools`, or in the web
  UI) - set when the context is made, whichever worker runs it. Such
  contexts' tasks are only taken by workers that have the unsafe tools
  (`--unsafe-tools`), and wait for one otherwise. Think twice before
  giving an agent that a stranger can talk to an unsandboxed shell.
- **Not supported**: push notifications (`-32003`), `file` parts
  (`-32005`), `auth-required` and `input-required`, the authenticated
  extended card, and the gRPC and REST bindings.

`tests/a2a_interop.py` runs the official Python SDK against a server, for
checking that it still understands faber:

```
faber profiles set echo --model script:/path/to/script.json
echo "tester $(head -c 24 /dev/urandom | base64)" > a2a-keys
faber serve --a2a-keys-file a2a-keys --run-agent w
python3 -m venv /tmp/a2a && /tmp/a2a/bin/pip install 'a2a-sdk<0.4' httpx
/tmp/a2a/bin/python tests/a2a_interop.py a2a-keys echo http://127.0.0.1:9090
```

## Options

```
-c, --config <PATH>          Path to JSON configuration file
-m, --max-tokens <N>         Maximum tokens to generate
    --model <MODEL>          AI model to use
    --endpoint <URL>         API endpoint URL
    --no-tools               Disable all tools
    --unsafe-tools           Give the session's own agent the unsafe tools (see Unsafe tools)
    --tools <LIST>           Comma-separated list of tools to enable
    --tool-choice <MODE>     Tool usage mode: auto, none, required
    --api-key <PATH>         File containing the API key
    --parameter <K=V>        Model parameter (can be repeated; a JSON object or array is taken as JSON)
    --mcp-server <N[=URL]>   Give the session's agent an MCP server, adding a remote one with =URL (can be repeated); see MCP section
    --db-path <PATH>         SQLite database for persistent storage
    --agent <NAME>           Start chat as this agent instead of 'default'
    --profile <NAME>         Make the chat's agent from this profile (see Profiles)
    --display-graphics       Render LaTeX blocks as images on terminals that support it; see below
    --server <ADDR>          Connect to a remote faber server
    --server-key-file <PATH> Read the server's key from this file (found by itself for one on this machine)
    --max-parallel-requests <N>  At most N model requests in flight at once (see below)
    --task-retention <DURATION>  Prune done tasks older than this while chatting
    --rate-limit-wait <DURATION> Keep retrying a rate-limited request this long (default 8h)
    --stream-idle-timeout <DURATION> Retry a response that sends nothing this long (default 3m)
```

`--max-parallel-requests` (or `"max_parallel_requests"` in the config file)
caps how many requests to the model are in flight at once across
everything one faber process does - the chat, sub-agents, `fan_out`
workers and scheduled task turns. Set it to the number of requests your
server can serve at once, e.g. llama.cpp's `--parallel`: with more, they
queue there anyway, and each llama.cpp slot only gets its share of the
context. Requests over the limit wait their turn, shown in the status bar
("Waiting for other requests to the model to finish"); Ctrl-C still
interrupts them. Unlimited unless set. `fan_out` runs no more workers at
once than this either: more would only take turns on the server's slots,
each evicting the others' prompts from its cache.

`--rate-limit-wait` (or `"rate_limit_wait"` in the config file) is how
long a request the model's server turns away as rate limited (429) keeps
being retried before it fails - 8h unless set, long enough for a usage
quota to reset, so a long-running session or task picks up again by
itself instead of failing. Each retry waits as long as the server's
`Retry-After` says, or else backs off up to 5 minutes between tries; if
`Retry-After` asks for longer than is left, the request fails right away.
Meanwhile the status bar says when the next try is ("Rate limited by the
model's server, retrying at 08:15"), and Ctrl-C still interrupts. Other
server errors and network failures are still retried only a few times,
for about 15 seconds.

`--stream-idle-timeout` (or `"stream_idle_timeout"` in the config file)
is how long a streamed response may send nothing before faber gives up on
it as stalled and sends the request again - 3m unless set. A server can
keep the connection open and stop sending; without it, the request would
wait for over a quarter of an hour, then fail. A stalled response counts
as one of the few retries other failures get. Whatever had been shown of
it stays on screen, and the answer starts again after it.

While a request waits to be retried, for whatever reason, it doesn't
count toward `--max-parallel-requests`: it takes its turn again to retry.

#### With a local server's prompt cache

A server like llama.cpp keeps each slot's last prompt and only processes
what's new in the next request - if it starts the same way. faber keeps
requests that way: the tools are always sent in the same order, and each
agent's conversation only grows (except when old tool results are
shortened to fit the context window). What still costs is more
conversations than slots: each switch on a slot means processing the
incoming conversation's prompt from where it differs from what was there.
So:

- Give the server as many slots as you want agents working at once
  (llama.cpp's `-np`), and set `--max-parallel-requests` to the same
  number. With one slot, a fan-out's workers run one after another - each
  then reuses the start it shares with the one before (instructions and
  tools), and only processes its own task.
- Each slot needs room for a whole conversation: in llama.cpp the context
  (`-c`) is split between the slots unless the KV cache is unified across
  them - see `llama-server --help` for the version you run, along with its
  options to keep and reuse more prompts than there are slots.

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
