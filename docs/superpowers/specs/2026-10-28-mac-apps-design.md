# Task 16 — Mac apps: Apple Notes and Mail drafts

*Written 2026-10-28 on branch `feat/task-16-mac-apps`. Status: draft for the owner's review.*

This replaces `plan.md` §6.7a and the Task 16 entry in `docs/superpowers/plans/2026-10-22-agent-bot.md` for this task. Those sections described a separate stdio MCP server that calls `osascript`, plus an approval gate for sending. This design keeps their safety rules but changes the mechanism and narrows the scope.

## 1. Intent

The agent should use Mac apps the way it uses its other tools: one call, a fast result, and no clicking through the UI. Two features come first:

1. **Apple Notes as the agent's notebook.** The agent writes new notes, saves changes to them, and reads them again in a later conversation. It must be able to tell notes apart, so it always works on the right one.
2. **Mail drafts.** The agent writes an email to a given address (it writes the subject and body itself from a short instruction like "email Bob that I'll be late") and opens it as a draft in Mail. The owner reviews it and presses Send.

The owner wants AppleScript, and tool calls fast enough to keep agent loops quick.

## 2. Goals, non-goals, anti-goals

**Goals**
- Notes tools that create, list, search, read, replace and append to notes, all inside one dedicated Notes folder.
- A Mail tool that opens a filled-in draft and never sends it.
- In-process AppleScript (PyObjC), registered in the host agent server, with no second server process.
- Unit tests that run without Notes or Mail, plus opt-in tests against the real apps.
- **Apple Notes replaces the JSON notes tools** (`add_note`, `list_notes`, `search_notes` in `agent_bot/tools.py`). They are deleted as the first step of this task (§5.0).

**Non-goals (this task)**
- iMessage, Reminders, Shortcuts, `open_app`, and the Approve/Decline gate. They come in a later task, once Notes and Mail work well.
- Deleting notes.
- Reading or searching the owner's other notes or their Mail inbox.
- Mail CC/BCC, attachments, or choosing the sending account (Mail's default account is used).

**Anti-goals: these count as failures even if everything else works**
- The agent reads, changes or lists a note outside its folder.
- An email is sent without the owner pressing Send.
- User or model text is spliced into AppleScript source, where it could break or inject script.
- The Linux sandbox container, or a non-Mac host, fails to start because of these tools.

## 3. Decisions made with the owner

- **Scope:** only Notes and Mail drafts.
- **Notes boundary:** the agent sees only notes it created, in its own folder (default name "Agent").
- **Mechanism:** AppleScript run in-process through PyObjC's `NSAppleScript` (the owner preferred AppleScript and asked for speed). The tools are plain host tools registered in `agent_bot/server.py`, not an MCP server.
- **Mail:** draft only. The agent writes the subject and body itself.
- **Replacing the JSON notes:** Apple Notes becomes the agent's only notebook. The JSON notes tools are deleted first, before the Apple Notes tools are built.

## 4. What the spike measured (2026-10-28, this Mac, Notes on macOS 26)

The throwaway script is `notes_spike.py` in the session scratchpad. It is not part of the repo.

| Finding | Detail |
|---|---|
| Scoped lookup fails for notes in other folders | `note id X of folder id F` errors with -1728 when X is not in F |
| Container check needs an explicit `get` | `set c to get container of n` then `id of c` works. `id of container of n` fails with -1728 |
| Ids are stable | A note keeps its id after its body is edited, and a folder keeps its id after being renamed |
| Note ids look like | `x-coredata://<store-uuid>/ICNote/p905`; folders `…/ICFolder/p903` |
| Readable properties | `name`, `plaintext`, `body` (HTML), `creation date`, `modification date`, `container` |
| `osascript` per call (median of 15) | list folder 151 ms, checked read 338 ms |
| In-process `NSAppleScript`, compiled once (median of 15) | list folder 56 ms, checked read 50 ms; compiling both scripts took 37 ms |
| Background thread | `NSAppleScript` worked from a non-main thread, which is where `function_tool` runs sync tools (`asyncio.to_thread`) |
| Passing arguments | User text arrives through a `run` Apple event (`aevt`/`oapp`) whose direct parameter is a list. The script reads it in `on run argv`, so text is never spliced into the source |
| Cold start | The first call took 17 s: Notes launching, plus the first "wants to control Notes" prompt |

Mail was not tested in the spike.

## 5. Design

### 5.0 First step: delete the JSON notes tools

Apple Notes replaces the JSON notes, so this cleanup happens before any Mac tool is built (owner's decision):
- `agent_bot/tools.py`: delete `_notes_path`, `_load_notes`, `_save_notes`, `add_note`, `list_notes` and `search_notes`, plus any imports only they used.
- `docker-compose.yml`: drop the `NOTES_FILE` environment line.
- `scripts/eval.py`: stop importing `_notes_path`, and use the `NOTES_FILE` constant for both modes. `reset_notes` and its test stay as they are. **The eval's two notes prompts (`dentist_recall`, `remember_wifi`) will fail from here on, because the tools they expect are gone.** Reworking the eval onto Apple Notes isn't part of this task, so the eval isn't a check for it.
- Tests: delete the JSON notes tests in `tests/test_tools.py` and update the expected tool-name list there. In `tests/test_tool_server.py:101`, check a different sandbox tool's `source`. `tests/test_parsing.py:76` only uses `list_notes` as a sample name and can stay. The full test suite passes after this step.
- Docs: update `docs/tool-calling-and-mcp.md` and `docs/agent-loop.md` if they list the JSON notes tools.
- **`data/notes.json` is left on disk, untouched.** It may hold the owner's saved notes, and migrating them isn't part of this task. The final report tells the owner the file is still there.
- `prompts/system.md`: until the Apple Notes tools exist, the "saved notes" wording refers to tools that aren't there. That's harmless, and §5.6 rewrites it.

### 5.1 Where the code lives

- **`agent_bot/mac_tools.py`** (new) holds the tools, the AppleScript sources, and the PyObjC runner. It is **not** `tools.py`, because `tools.py` is also loaded in the Linux sandbox container, which has no PyObjC.
- **`agent_bot/server.py`**: in `lifespan`, right after the `local_python_tools` block, if `sys.platform == "darwin"` and `cfg.mac_tools.enabled`, then `from agent_bot.mac_tools import register_mac_tools` and `register_mac_tools(state.registry, cfg.mac_tools)`. The import is lazy, so other platforms never import PyObjC. Each tool is registered with `function_tool(fn, source="mac")`. `agent_bot/cli.py` registers them the same way, if it builds its own registry.
- **`requirements.txt`**: add `pyobjc-framework-Cocoa; sys_platform == "darwin"`. `requirements-sandbox.txt` is unchanged.
- **`config.yaml` / `agent_bot/config.py`**: a new top-level section:

  ```yaml
  mac_tools:
    enabled: true          # host only, macOS only
    notes_folder: Agent    # name used when the folder is first created
    timeout_s: 60          # per-call limit for Mac tools (cold app launch + permission prompt)
  ```

  The `MacToolsConfig` defaults are `enabled: false`, `notes_folder: "Agent"` and `timeout_s: 60`. That way a config file without the section (tests, other machines) never touches Mac apps, and the repo's `config.yaml` turns them on.

### 5.2 The AppleScript runner

- There is one small class, as in the spike: it compiles the source once (`initWithSource_` + `compileAndReturnError_`). Then it sends a `run` Apple event whose direct parameter is a list of string descriptors built from the arguments, and returns `stringValue()` of the result.
- Every script is a **module-level constant** that starts with `on run argv` and reads all user and model text from `argv`. No script is ever built with string formatting. This is the injection rule from §2.
- Scripts are compiled lazily on first use, and the compiled objects are cached.
- **One `threading.Lock`** wraps every compile and every execute. Apple doesn't document `NSAppleScript` as thread-safe, and calls are short.
- **Errors.** The runner turns an `NSAppleScriptErrorNumber` into a `ToolError` that the model can act on:
  - `-1743` (automation not allowed): "macOS blocked control of Notes/Mail. Allow it in System Settings → Privacy & Security → Automation for the app that started the agent server, then try again."
  - `-1728` on a note lookup: "No note with id … in the Agent folder. Use list_apple_notes or search_apple_notes to find the right id."
  - A script's own `error "…"` gets its message passed through.
  - Anything else: "Notes/Mail error <number>: <message>".
- **Hangs and timeouts.** Today one timeout covers every tool: `tool_timeout(cfg)` in the server (25 s with the sandbox on). The cold start alone took 17 s in the spike, and the first call also resolves the folder, so a cold first call could time out. The tool would still finish in the background, and a retry by the model would create a **duplicate note or draft**. So:
  - `Tool` gains an optional `timeout_s: float | None = None`. `ToolRegistry.call` uses it in place of the caller's timeout when it is set. `register_mac_tools` sets it from `mac_tools.timeout_s` (default **60**), which covers a cold launch and the permission prompt. Other tools are unchanged. *(Builder's call, made during review.)*
  - An in-process call still can't be killed. If 60 s does run out, the worker thread waits for the app, the lock holds later Mac tool calls back until then, and the timeout message says: "Notes/Mail didn't answer in time; the action may still complete. Check with list_apple_notes before retrying." This is accepted.
  - No warm-up at server start. Starting the server never launches Notes or Mail.

### 5.3 The Agent folder: the security boundary

The boundary is enforced in code, never left to the model:

1. **No tool takes a folder parameter.** Every script starts from the pinned folder's id.
2. **The folder is pinned by id.** Its id is stored in `data/mac_tools.json` (`{"notes_folder_id": "x-coredata://…"}`). To resolve the folder:
   1. If the file has an id and `folder id <id>` exists, use it. A rename doesn't matter.
   2. Otherwise, **always create a new folder**. It is never adopted from existing folders, because an existing folder could hold the owner's own notes, and the owner asked that the agent see only notes it created. If no folder is called `notes_folder`, the new one gets that name. If one already exists, it is named `<notes_folder> (agent_bot)`. Then save its id. *(Builder's call, made during review.)*
   3. If a stored id no longer resolves (the owner deleted the folder), create a new folder as in step 2 and log a warning.

   The boundary is folder membership: the agent sees what's in its own folder. A note the owner moves into that folder becomes visible to the agent, which is the owner's choice. Tracking an allowlist of created ids was considered and rejected as extra state for no gain.

   Resolution happens on the first Notes tool call, not at server startup, so starting the server doesn't launch Notes. The resolved id is cached in memory for the life of the process.
3. **Every id-based script checks in the same script that acts.** It looks the note up with `note id (item 1 of argv) of folder id (item 2 of argv)`, and that scoped lookup is the check (spike: it fails for notes elsewhere). The check and the read or write run in one Apple event, so the note can't move in between and the check costs no extra round trip.
4. **What this protects against:** the model's mistakes, such as a wrong id, a hallucinated id, or a prompt-injected "edit my other notes". It does not protect against other code on the Mac. Once automation is allowed, the server process could reach all of Notes; it just never runs a script that does. The README must say this in one or two sentences.

A note the owner moves out of the folder becomes invisible to the agent. That is intended.

### 5.4 Notes tools

Names carry `apple_`. It helps the model link "my notes" and "the Notes app" to these tools. *(Builder's call: keep the prefix.)* Docstrings are Google-style with `Args:`, since `function_tool` builds the schema from them and the model reads them. Each docstring says the tool works only inside the agent's own folder in Apple Notes.

| Tool | Does | Returns |
|---|---|---|
| `create_apple_note(title, body)` | New note in the Agent folder | `Created note "<title>" (id: <id>).` |
| `list_apple_notes(max_results=20)` | The most recently modified notes in the folder | One line per note: `<id> \| <title> \| <modified, ISO local> \| <first 200 chars of the body text, newlines as spaces>`, or `The Agent folder has no notes yet.` |
| `search_apple_notes(query, max_results=10)` | Case-insensitive match on title or text | Same format as list, or `No notes in the Agent folder match '<query>'.` |
| `read_apple_note(note_id)` | Full text of one note | `Title: <title>\nModified: <iso>\n\n<plaintext without the title line>` |
| `update_apple_note(note_id, body)` | Replaces the body. The title stays | `Updated note "<title>".` |
| `append_to_apple_note(note_id, text)` | Adds text as new lines at the end | `Added to note "<title>".` |

Details:
- **Bodies are written as HTML** with a fixed builder in Python: `<h1>{escaped title}</h1>` followed by one `<div>{escaped line}</div>` per line, and `<div><br></div>` for empty lines. Escaping uses `html.escape`. This HTML is passed as an argv item, not spliced into the script. Notes derives the note's `name` from the first line, so the title stays the title.
- **`update_apple_note`** reads the current `name` and rebuilds `<h1>name</h1>` + the new body in the same script, so a body replacement never drops the title.
- **`append_to_apple_note`** sets `body` to `body & <divs for the new text>` in one script. It is the tool the model should prefer for "add X to my list". The docstring says so, because small models handle appending far better than rewriting a whole note.
- **List and search must fetch in bulk.** Use one property fetch per field (`id of notes of f`, `name of notes of f`, `modification date of notes of f`, `plaintext of notes of f`), not a per-note `repeat` loop. Each element access is an Apple event, and a loop slows down as the folder grows. Sorting by modification date and trimming to `max_results` can happen in Python. `search` filters with `whose name contains q or plaintext contains q`. AppleScript's `contains` ignores case by default; the builder confirms this.
- **Fields go back to Python** in a format that survives any note text: join with `ASCII character 31` (unit separator) between fields and `ASCII character 30` (record separator) between notes, and strip both from the text first. Plaintext can contain tabs and newlines, so tabs and newlines are not safe separators.
- **Limits:** `max_results` is clamped to 1–50. Preview text is capped at 200 chars, and the registry's existing `max_result_chars` caps everything.
- **Validation:** an empty title, body or query raises `ToolError` with a fix-it message. `note_id` must start with `x-coredata://` and contain `/ICNote/`. Otherwise the error is "That isn't a Notes id; use the id from list_apple_notes or search_apple_notes". This check is a helpful message, not the boundary; §5.3 is the boundary.

### 5.5 Mail tool

`compose_email_draft(to, subject, body)`:
- `to` is one or more comma-separated addresses. Each is stripped and must match `^[^@\s]+@[^@\s]+\.[^@\s]+$`. A failure raises `ToolError`: "'<x>' isn't an email address. Ask the user for the address."
- The script: `tell application "Mail"` → `make new outgoing message with properties {subject:…, content:…, visible:true}`, then a `make new to recipient at end of to recipients with properties {address:…}` for each address (addresses are argv items 3…n), then `activate`.
- **The Mail script never contains `send`.** A unit test asserts the word `send` doesn't appear in any Mail script source. Leaving a draft is safe by design, so no approval gate is needed.
- It returns: `Opened a draft to <addresses> in Mail with subject "<subject>". It has not been sent: the user reviews it and presses Send.`
- The docstring tells the model to write a complete, polished subject and body itself from the user's instruction, without asking the user for the wording. It should ask only when the recipient's address is unknown.
- `body` is plain text. Mail's `content` takes plain text, so it needs no HTML escaping.

### 5.6 System prompt

Add to `prompts/system.md`:
- Your notes live in Apple Notes, in your own folder. Use the Apple Notes tools for anything the user asks you to remember, save, write down or look up later. To change an existing note, find it with `search_apple_notes` or `list_apple_notes`, then use its id. Prefer `append_to_apple_note` for adding items.
- `compose_email_draft` only opens a draft. Write the subject and body yourself, then tell the user the draft is open for them to review and send. Never claim an email was sent.

The existing rule that mentions "the user's saved notes" stays and now refers to these tools.

### 5.7 Tests

**Unit tests (`tests/test_mac_tools.py`)** run on any OS with no PyObjC and no apps. They inject a fake runner in place of the PyObjC runner, which records `(script_name, argv)` and returns canned output:
- Hostile text in titles and bodies (`"`, `"; do shell script "rm -rf ~"`, newlines, `<b>`) shows up only in argv, the script constants are unchanged, and the HTML is escaped.
- `update_apple_note` and `append_to_apple_note` pass the pinned folder id as argv item 2.
- Fake errors map to the §5.2 messages: -1743 → Automation guidance, -1728 → "use list/search".
- Folder resolution: a stored id that exists, a stored id that's gone (falls back), a name that exists (adopted), and none (created). Each saves `data/mac_tools.json` (in a `tmp_path`).
- List and search parse the separator format, sort, clamp, and produce previews.
- Email validation passes and fails as expected, addresses go through argv, the return text says it wasn't sent, and no Mail script contains `send`.
- The server registers Mac tools only when `darwin` and `enabled` (with `sys.platform` monkeypatched).
- Importing `agent_bot.mac_tools` works without PyObjC. The PyObjC import happens inside the runner.

**Real-app tests (`tests/test_mac_tools_live.py`)** are skipped unless `RUN_MAC_TESTS=1` is set on macOS. They use a throwaway folder name and state file (`register_mac_tools` takes the state-file path as a parameter, defaulting to `data/mac_tools.json`), and in `finally` they delete everything they created:
- create → list → search → read → update → append round trip, with the ids staying stable.
- A note in a second throwaway folder can't be read, updated or appended through the tools.
- Median timing for list and read over 10 runs, printed so it can be recorded in `tasks.md`.
- Mail: open one draft to `test@example.com`, check that it's open with the right recipient and subject, and close it without saving. No send.

### 5.8 Docs and tracker

- `README.md`: a "Mac apps" section covering the Automation permission prompt and where to fix it, the Agent folder, draft-only Mail, and the §5.3 limit.
- `tasks.md`: rewrite the Task 16 row to this scope. Add a new row for the deferred iMessage, Reminders, Shortcuts and approval-gate work. Record the measured timings.
- `plan.md` §6.7a: add a one-line note pointing to this document.

## 6. Done when

1. In a GUI turn, "make a note called Packing list with passport and charger" creates the note in the Agent folder.
2. In a **new session**, "add sunscreen to my packing list" finds that note and appends to it, and Notes shows the change.
3. "Email bob@example.com that I'll be late" opens a Mail draft with a subject and body the model wrote, and it is not sent.
4. Trying to reach a note outside the folder by id gets a refusal from the tool (live test).
5. The unit tests pass on the host. The live tests pass with `RUN_MAC_TESTS=1`. Timings are in `tasks.md`.
6. The JSON notes tools are gone (§5.0) and the full test suite passes.

## 7. Left to the builder

- Exact docstring wording, within §5.4–5.6.
- Whether the runner is a class or a pair of functions, and how the fake runner is injected (module attribute or parameter).
- Retry behavior: none, unless the live tests show flakiness.
- Whether `cli.py` needs the registration (check how it builds its registry).

## 8. Settled during review

- **JSON notes vs Apple Notes (owner):** Apple Notes replaces the JSON notes tools, which are deleted as the first step (§5.0). The eval isn't updated in this task (owner).
- **Folder adoption (builder's call):** always create a new folder, never adopt an existing one (§5.3).
- **Timeouts (builder's call):** a per-tool `timeout_s` on `Tool`, with Mac tools at 60 s (§5.2).
- **Keeping `data/notes.json` (builder's call):** left on disk, not migrated (§5.0).
