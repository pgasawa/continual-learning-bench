# kaleidoscope_v2

[Kaleidoscope](https://memory.kleosresearch.xyz/) memory for Continual Learning
Bench. Kaleidoscope is a local memory store for AI agents
([npm](https://www.npmjs.com/package/@kleos-research/kaleidoscope)).

Per turn:

1. **Before the model answers,** Kaleidoscope's search puts relevant memories
   at the top of its prompt.
2. **While answering,** the model can call Kaleidoscope's `search` and
   `remember` tools itself. Their descriptions and instructions are exactly as
   `kscope` ships them, in `mcp_surface.json`. It then submits its answer with
   `submit_answer`.
3. **After the feedback,** a note-taker model call reads the current instance's
   conversation and the related memories. It creates, updates or deletes
   memories.
4. **Between instances,** only the memory carries over.

Every rollout starts with an empty, private vault. `src/systems/kaleidoscope/`
holds helpers this system imports, and an earlier version (`kaleidoscope`).

## Requirements

- **`kscope`**: `npm install -g @kleos-research/kaleidoscope`. The published
  runs used version 0.0.7, sha256
  `cc0edd01ef82fac3c779d7a731deec3c3fd9e3cdcc38b3ae1599d89ba3943902`.
- **A Kaleidoscope licence key**, as a file named `api-key` in a directory that
  you pass as `--system.kscope-entitlement-src`.
- **An OpenAI-compatible endpoint serving the model.** Set it in
  `OPENAI_BASE_URL` and `OPENAI_API_KEY`. The published runs used Azure
  OpenAI's `/openai/v1/` with GPT-5.6 Luna.

## Running

```bash
clbench run-all --name kaleidoscope-gpt-5.6-luna --system kaleidoscope_v2 \
  --system.model gpt-5.6-luna \
  --system.kscope-binary "$(which kscope)" \
  --system.kscope-sha256 "$(shasum -a 256 "$(which kscope)" | cut -d' ' -f1)" \
  --system.kscope-entitlement-src /path/to/licence-dir
```

## Reasoning effort: the start-up hook

The published runs used the start-up hook in `src/systems/kaleidoscope/hook/`.
Both this system's run and the `icl` entry on the same model used it. It:
- sets reasoning effort `high` on every model call
- registers GPT-5.6 Luna's price with litellm
- logs every call
- retries transient provider errors

To use it, put these in front of the command above:

```bash
PYTHONPATH=src/systems/kaleidoscope/hook CLBENCH_KS_HOOK=1 \
CLBENCH_KS_MODEL_ENTRY_SHA256="$(shasum -a 256 src/systems/kaleidoscope/hook/luna_model_entry.json | cut -d' ' -f1)" \
CLBENCH_KS_CALL_LOG_DIR=/path/to/call-logs REQUEST_TIMEOUT=600 \
clbench run-all ...
```

## Versions

This file is version 2.0.1.
- **Five of the six tasks** in the published run used 2.0.
- **2.0.1 differs only in how the note-taker counts tokens** on text that
  contains a single run longer than 20,000 characters. On such text, 2.0 could
  take hours.
- **2.0.1 was used for one `database_exploration` no-memory instance.** Its text
  never reached that length, so it ran 2.0's code path.
