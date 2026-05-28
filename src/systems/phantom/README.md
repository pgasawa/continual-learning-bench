# Phantom

Phantom is NinjaTech's agent. It uses Claude Code as the underlying model
runtime, and in production it talks to users through Slack. This folder
provides two CL-Bench systems built on Claude Code inside a Phantom Docker
image, so we can benchmark Phantom the same way we benchmark other systems.

## Two systems

### `phantom` — the simple version

A direct adapter. Each turn invokes `claude -p`, reads the assistant text from
stdout, and parses the JSON answer against the benchmark schema. There is no
Slack layer in the loop. This is useful as a clean baseline to compare against
production phantom and to debug the underlying Claude Code behaviour without
extra moving parts.

### `phantom_slack` — the production-faithful version (the one we mainly maintain)

This is the system we actively use and improve. It is built to simulate real
production Phantom as closely as possible.

In real Phantom:

1. A user posts a request in a Slack channel.
2. Phantom (Claude Code + a system prompt) reads the channel, thinks, and
   writes its reply back to Slack with the `slack_cli` tool.
3. The reply is the final answer that the user sees.

In `phantom_slack` we set up the exact same shape inside the benchmark:

1. The benchmark drops the task question into a **mock Slack inbox**
   (`inbox.jsonl`) instead of a real Slack channel.
2. Phantom is launched with the same prompt template, the same
   `slack_cli say` instruction, and the same Slack-aware identity that
   production Phantom sees.
3. Claude calls `python /opt/phantom/slack_cli say '<json>'` exactly like it
   would in production. Our mock library intercepts that call and writes the
   message into an **outbox** (`outbox.jsonl`) instead of going to the real
   Slack API.
4. The benchmark reads the outbox to get Phantom's answer for that turn.

The only differences from production are:

- The Slack backend is mocked (no network calls).
- The identity message references only `SLACK_INTERFACE.md`, since the other
  production docs are about browser automation and aren't relevant here.
- A short `phantom_memory.md` is wired up so memory can persist across the
  instances of a single rollout.

Everything else — the prompt wording, the section order, the `slack_cli`
invocation pattern, the default channel configuration, the timeout — matches
production byte-for-byte.

## Why two systems?

The simple `phantom` system is useful when you want to isolate the model and
remove the Slack layer (for example, to compare against vanilla Claude Code on
the same prompt). The `phantom_slack` system is what we use to measure how
real Phantom performs on CL-Bench tasks.

## Running it

```bash
clbench run <task> --system phantom_slack --schedule default
```

Works out of the box for the chat-style tasks (poker, cohort_studies,
database_exploration, blind_spectrum_monitoring).

## Key options

These are the options worth knowing about. The full list is in the
`PhantomSlackSystem.__init__` signature.

| Option | Default | What it does |
| --- | --- | --- |
| `model` | `claude-opus-4-7` | Which Claude model to invoke. |
| `slack_bridge` | `True` | Enables the mock Slack channel. Always `True` for `phantom_slack`. |
| `phantom_intro` | `True` | Inject the production Phantom identity message on the first turn of each session. |
| `single_conversation` | `True` | Keep one Claude session across instances within a rollout (so memory carries forward). |
| `baseline_mode` | `False` | Set `True` for baseline-only resumes. Suppresses the memory section to avoid an empty-memory prompt that can nudge the model into plain-text replies. |

## Memory

Phantom keeps a file called `phantom_memory.md` at the workspace root. Each
turn, its current content is injected into the prompt under a `## Your Memory`
section. The model can update the file with the `Write` or `Edit` tool, and
the new content is visible on the next turn. The file is wiped only at
rollout reset, so memory accumulates across the instances of a single rollout
and is fresh between rollouts.

## What's in this folder

- `system.py` — both `PhantomSystem` and `PhantomSlackSystem`.
- `clbench_mock.py` — patches the production `SlackClient` so messages go to
  the local inbox/outbox files instead of the network. Loaded automatically
  inside the container by a `.pth` file.
- `clbench_mock_autoload.pth` — the autoload trigger.
- `artifacts.py` — exports the per-instance Claude conversation JSONL files
  alongside the benchmark trace.
- The `phantom_slack` package next door (`src/systems/phantom_slack/`) is a
  thin alias that registers `PhantomSlackSystem` under the system name
  `phantom_slack`.

## Auth and container

The container expects two environment variables on the host: `ANTHROPIC_AUTH_TOKEN`
and `ANTHROPIC_BASE_URL` (typically the NinjaTech LiteLLM proxy). They are
passed through into the container, plus the standard Claude Code knobs
(`CLAUDE_CONFIG_DIR`, `IS_SANDBOX=1`, etc.). The Phantom Docker image
(`cl-bench/phantom`) is expected to be pre-built with the Phantom source tree,
Claude Code, and the agent docs already installed.
