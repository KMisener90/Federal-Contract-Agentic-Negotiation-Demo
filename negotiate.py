#!/usr/bin/env python3
"""
negotiate.py — record a multi-agent contract negotiation.

Runs each agent (one Government + one or more Offerors) against the Anthropic API.
Every turn, an agent returns its PRIVATE reasoning, a short spoken message, and a
list of STRUCTURED edits to the shared contract. The whole run is written to
transcript.json, which replay.html plays back on stage with no network needed.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python negotiate.py                 # uses scenario.json -> transcript.json
    python negotiate.py my_scenario.json my_transcript.json

Notes:
- This is the ONLY step that calls the API. Run it ahead of your talk, review the
  transcript, re-run until you get a version you like, then present replay.html.
- Swap the scenario by editing scenario.json (parties, objectives, clauses).
"""

import json
import os
import re
import sys
import copy

from anthropic import Anthropic

# You can switch to "claude-opus-4-8" for richer reasoning, or a faster model.
MODEL = "claude-sonnet-5"
MAX_TOKENS = 1500

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def clauses_to_text(clauses):
    return "\n".join(f'[{c["id"]}] {c["heading"]}: {c["text"]}' for c in clauses)


def public_log_text(turns):
    if not turns:
        return "(no offers yet)"
    lines = []
    for t in turns:
        who = t["actor_name"]
        lines.append(f'{who}: {t["message"]}')
    return "\n".join(lines)


def build_system_prompt(party, scenario):
    pub = "\n".join(f"- {o}" for o in party["public_objectives"])
    hid = "\n".join(f"- {o}" for o in party["hidden_objectives"])
    role_line = (
        "You are the GOVERNMENT contracting officer. You will ultimately award and "
        "sign the contract with the offeror that represents the best value."
        if party["role"] == "government"
        else "You are an OFFEROR competing to win this contract award."
    )
    return f"""You are an autonomous agent in a live contract negotiation demonstration.

{role_line}

Your name in this negotiation: {party['name']}

Context: {scenario['context']}

YOUR PUBLIC OBJECTIVES (the other parties can infer these):
{pub}

YOUR HIDDEN OBJECTIVES (never state these outright; they drive your strategy):
{hid}

Each turn, respond with ONLY a JSON object (no markdown, no code fences), shaped:
{{
  "reasoning": "2-4 sentences of PRIVATE strategy. Reference your hidden objectives explicitly here; this is shown only to the presenter's trace panel.",
  "message": "1-3 sentences you say ALOUD at the table. Persuasive, in character, never revealing hidden objectives.",
  "edits": [
    {{"clause_id": "c2", "op": "replace", "old": "<current text of that clause>", "new": "<your proposed new text>", "rationale": "one short line"}},
    {{"clause_id": "c7", "op": "insert_after", "heading": "8. New Clause Title", "new": "<text of the brand-new clause>", "rationale": "one short line"}}
  ],
  "decision": null
}}

Rules:
- "op" is one of: "replace", "insert_after", "delete".
- For "replace" and "delete", clause_id must be an EXISTING clause; copy its current text into "old" verbatim.
- For "insert_after", clause_id is the clause you want to place the new one after; include a "heading".
- Only include edits you actually want this turn. It is fine to have an empty "edits" list if you are only speaking.
- GOVERNMENT ONLY: set "decision" to one of:
    {{"type": "counter"}}  — you are pushing back and continuing,
    {{"type": "award", "awarded_to": "<offeror id>"}} — you are awarding now (ends the negotiation).
  Offerors must always use "decision": null.
- Award only when an offeror genuinely satisfies your objectives (including the hidden ones). Justify the award in your reasoning."""


def call_agent(party, scenario, clauses, turns):
    system = build_system_prompt(party, scenario)
    user = f"""CURRENT CONTRACT:
{clauses_to_text(clauses)}

WHAT HAS BEEN SAID SO FAR:
{public_log_text(turns)}

It is your turn, {party['name']}. Respond with the JSON object only."""

    for attempt in range(2):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == 0:
                user += "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object."
            else:
                raise RuntimeError(f"{party['name']} did not return valid JSON:\n{raw}")


def apply_edits(clauses, edits):
    """Apply a turn's edits to the running clause list (pure-ish: mutates a copy)."""
    clauses = copy.deepcopy(clauses)
    by_id = {c["id"]: c for c in clauses}
    new_counter = len(clauses)
    for e in edits:
        op = e.get("op")
        if op == "replace" and e["clause_id"] in by_id:
            by_id[e["clause_id"]]["text"] = e["new"]
        elif op == "delete" and e["clause_id"] in by_id:
            clauses = [c for c in clauses if c["id"] != e["clause_id"]]
            by_id.pop(e["clause_id"], None)
        elif op == "insert_after":
            new_counter += 1
            new_id = e.get("new_id") or f"n{new_counter}"
            e["new_id"] = new_id  # record the id we assigned so replay matches
            new_clause = {"id": new_id, "heading": e.get("heading", "New Clause"), "text": e["new"]}
            idx = next((i for i, c in enumerate(clauses) if c["id"] == e["clause_id"]), len(clauses) - 1)
            clauses.insert(idx + 1, new_clause)
            by_id[new_id] = new_clause
    return clauses


def main():
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else "scenario.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "transcript.json"

    with open(scenario_path) as f:
        scenario = json.load(f)

    parties = {p["id"]: p for p in scenario["parties"]}
    gov_id = next(p["id"] for p in scenario["parties"] if p["role"] == "government")
    offeror_ids = [p["id"] for p in scenario["parties"] if p["role"] == "offeror"]

    clauses = copy.deepcopy(scenario["initial_clauses"])
    turns = []
    outcome = None

    def run_turn(actor_id):
        nonlocal clauses
        party = parties[actor_id]
        print(f"  -> {party['name']} thinking...")
        result = call_agent(party, scenario, clauses, turns)
        edits = result.get("edits", []) or []
        clauses = apply_edits(clauses, edits)
        turn = {
            "index": len(turns),
            "actor": actor_id,
            "actor_name": party["name"],
            "role": party["role"],
            "reasoning": result.get("reasoning", ""),
            "message": result.get("message", ""),
            "edits": edits,
            "decision": result.get("decision"),
        }
        turns.append(turn)
        return turn

    max_rounds = scenario.get("max_rounds", 3)
    for rnd in range(max_rounds):
        print(f"Round {rnd + 1}")
        for oid in offeror_ids:
            run_turn(oid)
        gov_turn = run_turn(gov_id)
        decision = gov_turn.get("decision") or {}
        if decision.get("type") == "award":
            outcome = {"awarded_to": decision.get("awarded_to"), "round": rnd + 1}
            break

    if outcome is None:
        # Force a final award decision from the Government.
        print("Final award decision")
        party = parties[gov_id]
        result = call_agent(
            party, scenario, clauses,
            turns + [{"actor_name": "SYSTEM", "message": "This is the final round. You must award now."}],
        )
        decision = result.get("decision") or {}
        awarded_to = decision.get("awarded_to") or offeror_ids[0]
        turns.append({
            "index": len(turns), "actor": gov_id, "actor_name": party["name"],
            "role": "government", "reasoning": result.get("reasoning", ""),
            "message": result.get("message", ""), "edits": result.get("edits", []) or [],
            "decision": {"type": "award", "awarded_to": awarded_to},
        })
        clauses = apply_edits(clauses, turns[-1]["edits"])
        outcome = {"awarded_to": awarded_to, "round": max_rounds}

    winner = parties.get(outcome["awarded_to"], {}).get("name", outcome["awarded_to"])
    transcript = {
        "scenario": scenario,
        "turns": turns,
        "final_clauses": clauses,
        "outcome": {**outcome, "awarded_to_name": winner},
    }
    with open(out_path, "w") as f:
        json.dump(transcript, f, indent=2)

    print(f"\nDone. {len(turns)} turns. Awarded to: {winner}")
    print(f"Wrote {out_path} — open replay.html and load it.")


if __name__ == "__main__":
    main()
