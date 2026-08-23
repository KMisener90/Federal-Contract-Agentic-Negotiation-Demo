# AI Contract Negotiation — record & replay demo

Two-or-more AI agents autonomously negotiate a contract: one **Government** (awards
and signs) and one or more **Offerors**, each driven by *public* objectives everyone
can infer and *hidden* objectives that secretly steer their strategy. You **record**
one real negotiation offline, then **replay** it on stage — live logic trace on the
right, live redlining of the document on the left. The replay makes **no network
calls**, so it can't stall or wander mid-talk.

## Files
- `replay.html` — the presentation player. Self-contained; **double-click to open.**
  It boots with a bundled sample recording so you can see it immediately.
- `negotiate.py` — the recorder. Runs the real negotiation via the Anthropic API and
  writes a `transcript.json`.
- `scenario.json` — the parties, their public/hidden objectives, and the starting
  contract clauses. **Edit this to change the scenario.**
- `sample_transcript.json` — the pre-baked run shown on open (cloud AI platform,
  Government + two offerors, awarded to the challenger).

## Preview right now (no setup)
Open `replay.html`. Press **Play**. That's the whole demo loop.

## Record your own run
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...      # Windows: set ANTHROPIC_API_KEY=...
python negotiate.py                       # reads scenario.json -> transcript.json
```
Then in `replay.html` click **“Load recording…”** and pick your `transcript.json`.
Re-run until you get a version you like — each run differs. Review the trace before
you present; you own what goes on screen.

To swap the model, edit `MODEL` near the top of `negotiate.py`
(e.g. `claude-opus-4-8` for richer reasoning).

## Change the scenario
Edit `scenario.json`: rename the parties, rewrite their `public_objectives` and
`hidden_objectives`, and edit the `initial_clauses`. Keep clause `id`s stable
(`c1`, `c2`, …). Add a third offeror by adding another party with `"role":"offeror"`.
Then re-record.

## Presenting
- **Controls:** Play/Pause (also **Spacebar**), Step (also **← / →**), Restart,
  and a **Speed** slider. Everything is keyboard-drivable from the podium.
- **“Reveal hidden objectives”** checkbox — keep it off, let the audience watch the
  moves, then flip it on to show what each agent was secretly optimizing for. Their
  private reasoning in the trace panel references those hidden goals as it plays.
- Redlines: green underline = inserted, red strikethrough = removed. The acting
  party colors the changed clause's left border.
- Full-screen your browser (F11) and zoom to taste for the room.

## How it works (for the talk)
Each turn an agent returns structured output — private *reasoning*, a spoken
*message*, and a list of *edits* (clause id + old/new text + rationale) plus, for the
Government, an accept/counter/award *decision*. The recorder applies edits to a shared
contract; the player reconstructs the document at each turn and diffs old-vs-new to
render the redline. Structured edits (not free-form rewrites) are what make the
redlining reliable.
