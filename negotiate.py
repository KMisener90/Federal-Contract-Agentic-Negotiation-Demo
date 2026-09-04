#!/usr/bin/env python3
"""
negotiate.py — record a multi-phase federal source-selection & contract negotiation.

Runs a full negotiated-procurement lifecycle against the Anthropic API:
  1. Proposals        — each of 3 offerors submits Volumes I-IV against Section L.
  2. Discussions       — the Program Office privately advises the CO; the CO holds
                          separate discussions with each offeror; each offeror revises.
  3. Evaluation        — the Program Office scores factors 1-3 (strengths/weaknesses)
                          per offeror and recommends an awardee to the CO.
  4. Award decision     — the CO independently weighs the recommendation + price
                          (best-value tradeoff) and selects the awardee.
  5. Debrief & protest  — losing offerors are debriefed and each decides, on their
                          own reasoning, whether to protest.
  6. Contract negotiation — the CO and the awardee negotiate the awardee's commercial
                          terms of service down to something the government can sign.

The whole run is written to transcript.json, which replay.html plays back on stage
with no network needed.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python negotiate.py                 # uses scenario.json -> transcript.json
    python negotiate.py my_scenario.json my_transcript.json
    python negotiate.py --dry-run       # exercises the whole pipeline with canned
                                         # responses, no API calls (fast smoke test)

Notes:
- This is the ONLY step that calls the API. Run it ahead of your talk, review the
  transcript, re-run until you get a version you like, then present replay.html.
- Swap the scenario by editing scenario.json (parties, factors, clause templates).
"""

import json
import os
import re
import sys
import copy

MODEL = "claude-sonnet-5"
MAX_TOKENS = 5000

DRY_RUN = "--dry-run" in sys.argv
if not DRY_RUN:
    from anthropic import Anthropic
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment


# ----------------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------------

def clauses_to_text(clauses):
    return "\n".join(f'[{c["id"]}] {c["heading"]}: {c["text"]}' for c in clauses)


def apply_edits(clauses, edits):
    """Apply a turn's edits to a clause list (returns a new list)."""
    clauses = copy.deepcopy(clauses)
    by_id = {c["id"]: c for c in clauses}
    new_counter = len(clauses)
    for e in edits:
        op = e.get("op")
        if op == "replace" and e.get("clause_id") in by_id:
            by_id[e["clause_id"]]["text"] = e["new"]
        elif op == "delete" and e.get("clause_id") in by_id:
            clauses = [c for c in clauses if c["id"] != e["clause_id"]]
            by_id.pop(e["clause_id"], None)
        elif op == "insert_after":
            new_counter += 1
            new_id = e.get("new_id") or f"n{new_counter}"
            e["new_id"] = new_id
            new_clause = {"id": new_id, "heading": e.get("heading", "New Clause"), "text": e["new"]}
            idx = next((i for i, c in enumerate(clauses) if c["id"] == e["clause_id"]), len(clauses) - 1)
            clauses.insert(idx + 1, new_clause)
            by_id[new_id] = new_clause
    return clauses


_DRY_COUNTER = [0]


def call_llm(system, user, expect_keys=()):
    """Call the model and parse its JSON reply, with one retry on bad JSON.
    In --dry-run mode, returns a small canned object with the requested keys
    filled with placeholder content, so the whole pipeline can be exercised
    without hitting the network."""
    if DRY_RUN:
        _DRY_COUNTER[0] += 1
        n = _DRY_COUNTER[0]
        stub = {
            "reasoning": f"(dry-run reasoning #{n})",
            "message": f"(dry-run message #{n})",
        }
        for k in expect_keys:
            if k == "edits":
                stub["edits"] = []
            elif k == "price_numeric":
                stub["price_numeric"] = 3000000 + n * 1000
            elif k == "payload":
                stub["payload"] = {}
            elif k == "decision":
                stub["decision"] = None
        return stub

    tokens = MAX_TOKENS
    for attempt in range(3):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if resp.stop_reason == "max_tokens":
                # Truncated mid-JSON — retry with more room rather than nudging for valid JSON.
                tokens = min(tokens * 2, 8192)
                continue
            if attempt < 2:
                user += "\n\nYour previous reply was not valid JSON. Return ONLY the JSON object, and keep it concise."
            else:
                raise RuntimeError(f"Model did not return valid JSON:\n{raw}")


def fmt_price(n):
    try:
        return f"${int(n):,}/yr"
    except (TypeError, ValueError):
        return str(n)


# ----------------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------------

class Recorder:
    def __init__(self, scenario):
        self.scenario = scenario
        self.parties = {p["id"]: p for p in scenario["parties"]}
        self.co_id = "co"
        self.program_id = "program"
        self.offeror_ids = [p["id"] for p in scenario["parties"] if p["role"] == "offeror"]
        self.turns = []
        self.proposal_docs = {oid: copy.deepcopy(scenario["proposal_clause_template"]) for oid in self.offeror_ids}
        self.contract_doc = None
        self.price = {}  # oid -> latest numeric annual price

    # -- bookkeeping -----------------------------------------------------
    def add_turn(self, *, actor, phase, doc=None, reasoning="", message="",
                 edits=None, payload=None, decision=None, actor_name=None):
        party = self.parties.get(actor)
        turn = {
            "index": len(self.turns),
            "phase": phase,
            "actor": actor,
            "actor_name": actor_name or (party["name"] if party else "System"),
            "role": party["role"] if party else "system",
            "doc": doc,
            "reasoning": reasoning,
            "message": message,
            "edits": edits or [],
            "payload": payload,
            "decision": decision,
        }
        self.turns.append(turn)
        return turn

    def offeror_thread(self, oid):
        """What a given offeror has seen: only its own proposal doc's turns."""
        return [t for t in self.turns if t["doc"] == f"proposal_{oid}"]

    def public_log(self, turns):
        if not turns:
            return "(none yet)"
        return "\n".join(f'{t["actor_name"]}: {t["message"]}' for t in turns if t["message"])

    # -- phase 1: proposals ------------------------------------------------
    def run_proposals(self):
        sc = self.scenario
        for oid in self.offeror_ids:
            party = self.parties[oid]
            print(f"  -> {party['name']} preparing initial proposal...")
            system = self._offeror_system(party, phase_note=(
                "This is your INITIAL PROPOSAL submission. Write your Volume I-IV content "
                "into the clauses now via 'edits' (op: 'replace' on tech/mgmt/pp/price)."
            ))
            user = f"""SOLICITATION SECTION L (submission instructions):
{json.dumps(sc['solicitation']['section_L'], indent=2)}

YOUR CURRENT PROPOSAL DOCUMENT (placeholders — replace all four):
{clauses_to_text(self.proposal_docs[oid])}

Submit your full initial proposal now."""
            result = call_llm(system, user, expect_keys=("edits", "price_numeric"))
            edits = result.get("edits", []) or []
            self.proposal_docs[oid] = apply_edits(self.proposal_docs[oid], edits)
            price = result.get("price_numeric") or self.price.get(oid, 0)
            self.price[oid] = price
            self.add_turn(actor=oid, phase="proposal_submission", doc=f"proposal_{oid}",
                           reasoning=result.get("reasoning", ""), message=result.get("message", ""),
                           edits=edits, payload={"price_numeric": price})

    # -- phase 2: discussions -----------------------------------------------
    def run_discussions(self):
        rounds = self.scenario.get("discussion_rounds", 1)
        for rnd in range(rounds):
            print(f"  Discussion round {rnd + 1}")
            guidance = self._program_guidance()
            self.add_turn(actor=self.program_id, phase="discussion_guidance",
                           reasoning=guidance.get("reasoning", ""), message=guidance.get("message", ""),
                           payload={"guidance": guidance.get("guidance", [])})
            notes_by_offeror = {g.get("offeror"): g.get("note", "") for g in guidance.get("guidance", []) if g.get("offeror")}

            for oid in self.offeror_ids:
                party = self.parties[oid]
                print(f"    -> CO holds discussions with {party['name']}")
                co_result = self._co_discussion(oid, notes_by_offeror.get(oid, ""))
                self.add_turn(actor=self.co_id, phase="discussion", doc=f"proposal_{oid}",
                               reasoning=co_result.get("reasoning", ""), message=co_result.get("message", ""))

                print(f"    -> {party['name']} revises proposal")
                rev = self._offeror_revision(oid)
                edits = rev.get("edits", []) or []
                self.proposal_docs[oid] = apply_edits(self.proposal_docs[oid], edits)
                price = rev.get("price_numeric") or self.price.get(oid, 0)
                self.price[oid] = price
                self.add_turn(actor=oid, phase="revision", doc=f"proposal_{oid}",
                               reasoning=rev.get("reasoning", ""), message=rev.get("message", ""),
                               edits=edits, payload={"price_numeric": price})

    def _program_guidance(self):
        sc = self.scenario
        party = self.parties[self.program_id]
        system = self._role_system(party, extra=(
            "You are privately advising the Contracting Officer before discussions open with each "
            "offeror. This note is GOVERNMENT-INTERNAL and never shown to any offeror."
        ))
        proposals_text = "\n\n".join(
            f"=== {self.parties[oid]['name']} ===\n{clauses_to_text(self.proposal_docs[oid])}"
            for oid in self.offeror_ids
        )
        user = f"""CURRENT PROPOSALS FROM ALL THREE OFFERORS:
{proposals_text}

Respond with ONLY a JSON object:
{{
  "reasoning": "2-4 sentences of PRIVATE strategy/assessment — reference your hidden objectives explicitly here.",
  "message": "1-2 sentences summarizing your overall read for the record.",
  "guidance": [
    {{"offeror": "off_1", "note": "one or two sentences of technical guidance for the CO on what to press this offeror on"}},
    {{"offeror": "off_2", "note": "..."}},
    {{"offeror": "off_3", "note": "..."}}
  ]
}}"""
        return call_llm(system, user, expect_keys=("payload",))

    def _co_discussion(self, oid, guidance_note):
        party = self.parties[self.co_id]
        offeror = self.parties[oid]
        system = self._role_system(party, extra=(
            f"You are now holding DISCUSSIONS with {offeror['name']} specifically. Raise the issues, "
            "concerns, or clarifications you want addressed in their final proposal revision. Do not "
            "edit their document yourself — only offerors edit their own proposal clauses. Set "
            "\"edits\" to an empty list and \"decision\" to null; this is a discussion letter, not the "
            "award decision."
        ))
        thread = self.offeror_thread(oid)
        user = f"""{offeror['name']}'S PROPOSAL SO FAR:
{clauses_to_text(self.proposal_docs[oid])}

PRIOR EXCHANGE WITH THIS OFFEROR:
{self.public_log(thread)}

PROGRAM OFFICE GUIDANCE (internal — for your eyes only, never repeat verbatim to the offeror):
{guidance_note or '(none)'}

Respond with ONLY a JSON object: {{"reasoning": "...", "message": "..."}}
"reasoning" is your private strategy; "message" is what you actually say to {offeror['name']} at the table."""
        return call_llm(system, user)

    def _offeror_revision(self, oid):
        party = self.parties[oid]
        system = self._offeror_system(party, phase_note=(
            "The Contracting Officer has just raised discussion items with you. Revise your proposal "
            "via 'edits' to respond — this may be your Final Proposal Revision."
        ))
        thread = self.offeror_thread(oid)
        user = f"""YOUR CURRENT PROPOSAL:
{clauses_to_text(self.proposal_docs[oid])}

EXCHANGE SO FAR WITH THE GOVERNMENT:
{self.public_log(thread)}

Respond to the Contracting Officer's most recent discussion item with a revised proposal now."""
        return call_llm(system, user, expect_keys=("edits", "price_numeric"))

    # -- phase 3: evaluation --------------------------------------------
    def run_evaluation(self):
        evaluations = {}
        for oid in self.offeror_ids:
            party = self.parties[self.program_id]
            print(f"  -> Program Office evaluates {self.parties[oid]['name']}")
            system = self._role_system(party, extra=(
                f"You are writing the FORMAL, FOR-THE-RECORD technical evaluation of {self.parties[oid]['name']}'s "
                "final proposal against Factors 1-3 only (never evaluate price). List concrete strengths and "
                "weaknesses for each factor, grounded in what the proposal actually says."
            ))
            proposals_text = "\n\n".join(
                f"=== {self.parties[o]['name']} ({'THIS OFFEROR' if o == oid else 'for context only'}) ===\n"
                f"{clauses_to_text(self.proposal_docs[o])}"
                for o in self.offeror_ids
            )
            user = f"""ALL FINAL PROPOSALS (evaluate only {self.parties[oid]['name']} in this call):
{proposals_text}

Respond with ONLY a JSON object:
{{
  "reasoning": "2-4 sentences of PRIVATE assessment — be explicit here about any bias per your hidden objectives.",
  "message": "1-2 sentence summary suitable for the record.",
  "payload": {{
    "offeror": "{oid}",
    "factors": {{
      "technical": {{"strengths": ["..."], "weaknesses": ["..."]}},
      "management": {{"strengths": ["..."], "weaknesses": ["..."]}},
      "past_performance": {{"strengths": ["..."], "weaknesses": ["..."]}}
    }},
    "overall": "one-paragraph narrative assessment"
  }}
}}"""
            result = call_llm(system, user, expect_keys=("payload",))
            payload = result.get("payload") or {"offeror": oid, "factors": {}, "overall": ""}
            payload["offeror"] = oid
            evaluations[oid] = payload
            self.add_turn(actor=self.program_id, phase="evaluation",
                           reasoning=result.get("reasoning", ""), message=result.get("message", ""),
                           payload=payload)

        print("  -> Program Office recommends an awardee")
        party = self.parties[self.program_id]
        system = self._role_system(party, extra=(
            "You have evaluated all three offerors. Now rank them and recommend one to the Contracting "
            "Officer. You care about service quality far more than price; do not let price drive your rank."
        ))
        evals_text = json.dumps(evaluations, indent=2)
        user = f"""YOUR EVALUATIONS:
{evals_text}

Respond with ONLY a JSON object:
{{
  "reasoning": "2-4 sentences of PRIVATE reasoning behind the ranking.",
  "message": "1-2 sentences for the record.",
  "payload": {{"ranking": ["<offeror id best>", "<middle>", "<worst>"], "recommended": "<offeror id>", "rationale": "one paragraph"}}
}}"""
        rec = call_llm(system, user, expect_keys=("payload",))
        self.add_turn(actor=self.program_id, phase="recommendation",
                       reasoning=rec.get("reasoning", ""), message=rec.get("message", ""),
                       payload=rec.get("payload") or {})
        return evaluations

    # -- phase 4: award decision -----------------------------------------
    def run_award_decision(self, evaluations):
        print("  -> Contracting Officer makes the award decision")
        party = self.parties[self.co_id]
        system = self._role_system(party, extra=(
            "You will now independently decide the award. You have the Program Office's evaluations and "
            "recommendation, plus price, which is yours alone to weigh. Sanity-check the evaluations for "
            "unsupported bias before relying on them. Perform a best-value tradeoff and decide."
        ))
        rec_turn = next(t for t in reversed(self.turns) if t["phase"] == "recommendation")
        price_table = "\n".join(f"- {self.parties[oid]['name']}: {fmt_price(self.price.get(oid))}" for oid in self.offeror_ids)
        user = f"""PROGRAM OFFICE EVALUATIONS:
{json.dumps(evaluations, indent=2)}

PROGRAM OFFICE RECOMMENDATION:
{json.dumps(rec_turn['payload'], indent=2)}

FINAL PRICES:
{price_table}

YOUR BUDGET GUIDANCE (hidden — never state the number aloud): see your hidden objectives.

Respond with ONLY a JSON object:
{{
  "reasoning": "3-5 sentences of PRIVATE tradeoff reasoning — including whether you trust the Program Office's write-up as-is.",
  "message": "1-2 sentences suitable for the record announcing the outcome.",
  "payload": {{"tradeoff_rationale": "one paragraph explaining the best-value tradeoff", "price_comparison": {json.dumps({oid: self.price.get(oid) for oid in self.offeror_ids})}}},
  "decision": {{"type": "award", "awarded_to": "<offeror id>"}}
}}"""
        result = call_llm(system, user, expect_keys=("payload", "decision"))
        decision = result.get("decision") or {"type": "award", "awarded_to": self.offeror_ids[0]}
        payload = result.get("payload") or {}
        payload.setdefault("price_comparison", {oid: self.price.get(oid) for oid in self.offeror_ids})
        self.add_turn(actor=self.co_id, phase="award_decision",
                       reasoning=result.get("reasoning", ""), message=result.get("message", ""),
                       payload=payload, decision=decision)
        return decision.get("awarded_to") or self.offeror_ids[0]

    # -- phase 5: debrief & protest ---------------------------------------
    def run_debrief_and_protest(self, evaluations, awardee_id):
        protests = []
        for oid in self.offeror_ids:
            if oid == awardee_id:
                continue
            ev = evaluations.get(oid, {}).get("factors", {})
            strengths, weaknesses = [], []
            for factor_key, factor in ev.items():
                strengths.extend(factor.get("strengths", []))
                weaknesses.extend(factor.get("weaknesses", []))
            note = (f"{self.parties[oid]['name']} was not selected for award. Award went to "
                    f"{self.parties[awardee_id]['name']} at {fmt_price(self.price.get(awardee_id))} "
                    f"versus your {fmt_price(self.price.get(oid))}.")
            debrief_payload = {"offeror": oid, "strengths": strengths, "weaknesses": weaknesses, "note": note}
            self.add_turn(actor="system", phase="debrief", message=note, payload=debrief_payload,
                           actor_name="Source Selection Debrief")

            print(f"  -> {self.parties[oid]['name']} decides whether to protest")
            party = self.parties[oid]
            system = self._offeror_system(party, phase_note=(
                "You have just been debriefed after not receiving the award. Decide, based on your own "
                "hidden objectives and the substance of your debrief, whether to file a bid protest."
            ))
            user = f"""YOUR DEBRIEF:
Strengths noted: {json.dumps(strengths)}
Weaknesses noted: {json.dumps(weaknesses)}
{note}

Respond with ONLY a JSON object:
{{
  "reasoning": "2-4 sentences of PRIVATE reasoning about whether to protest.",
  "message": "1-2 sentences you would say publicly about your decision.",
  "payload": {{"protest": true, "grounds": "one paragraph, or null if not protesting"}}
}}"""
            result = call_llm(system, user, expect_keys=("payload",))
            payload = result.get("payload") or {"protest": False, "grounds": None}
            self.add_turn(actor=oid, phase="protest_decision",
                           reasoning=result.get("reasoning", ""), message=result.get("message", ""),
                           payload=payload)
            protests.append({"offeror": oid, "offeror_name": party["name"], "protest": bool(payload.get("protest")), "grounds": payload.get("grounds")})
        return protests

    # -- phase 6: contract negotiation -------------------------------------
    def run_contract_negotiation(self, awardee_id):
        sc = self.scenario
        awardee = self.parties[awardee_id]
        self.contract_doc = copy.deepcopy(sc["contract_clause_template"])

        fee_clause = next(c for c in self.contract_doc if c["id"] == "ct2")
        old_fee_text = fee_clause["text"]
        new_fee_text = (f"Customer shall pay Provider a firm-fixed annual fee of "
                         f"{fmt_price(self.price.get(awardee_id))} for the software subscription and "
                         f"Operations & Maintenance support, payable per the awarded CLIN structure.")
        insert_edit = {"clause_id": "ct2", "op": "replace", "old": old_fee_text, "new": new_fee_text,
                        "rationale": "Insert awarded price."}
        self.contract_doc = apply_edits(self.contract_doc, [insert_edit])
        self.add_turn(actor="system", phase="contract_negotiation", doc="contract",
                       message="Awarded price inserted into the fee clause.",
                       edits=[insert_edit], actor_name="Contract Administration")

        max_rounds = sc.get("contract_max_rounds", 3)
        executed = False
        for rnd in range(max_rounds):
            print(f"  Contract negotiation round {rnd + 1}")
            co_result = self._co_contract_turn(awardee_id, final=False)
            edits = co_result.get("edits", []) or []
            self.contract_doc = apply_edits(self.contract_doc, edits)
            decision = co_result.get("decision") or {"type": "counter"}
            self.add_turn(actor=self.co_id, phase="contract_negotiation", doc="contract",
                           reasoning=co_result.get("reasoning", ""), message=co_result.get("message", ""),
                           edits=edits, decision=decision)
            if decision.get("type") == "execute":
                executed = True
                break

            vendor_result = self._vendor_contract_turn(awardee_id)
            edits = vendor_result.get("edits", []) or []
            self.contract_doc = apply_edits(self.contract_doc, edits)
            self.add_turn(actor=awardee_id, phase="contract_negotiation", doc="contract",
                           reasoning=vendor_result.get("reasoning", ""), message=vendor_result.get("message", ""),
                           edits=edits)

        if not executed:
            print("  Final execution decision")
            result = self._co_contract_turn(awardee_id, final=True)
            edits = result.get("edits", []) or []
            self.contract_doc = apply_edits(self.contract_doc, edits)
            self.add_turn(actor=self.co_id, phase="contract_final", doc="contract",
                           reasoning=result.get("reasoning", ""), message=result.get("message", ""),
                           edits=edits, decision={"type": "execute"})
        else:
            # promote the last turn's phase for clarity in the trace
            self.turns[-1]["phase"] = "contract_final"

    def _co_contract_turn(self, awardee_id, final):
        party = self.parties[self.co_id]
        awardee = self.parties[awardee_id]
        finality = ("This is the FINAL round. You must set \"decision\" to "
                    '{"type": "execute"} and accept or make your last necessary edits now.'
                    if final else
                    'Set "decision" to {"type": "counter"} to keep negotiating, or '
                    '{"type": "execute"} if the contract is now acceptable to sign.')
        system = self._role_system(party, extra=(
            f"You are now negotiating {awardee['name']}'s commercial terms of service directly with them "
            "— the Program Office is not involved in this phase. Fix any term the government cannot lawfully "
            f"or practically accept per your hidden objectives. Keep proposed clause text focused (roughly "
            f"80-140 words). {finality}"
        ))
        contract_thread = [t for t in self.turns if t["doc"] == "contract"]
        user = f"""CURRENT CONTRACT:
{clauses_to_text(self.contract_doc)}

NEGOTIATION SO FAR:
{self.public_log(contract_thread)}

Respond with ONLY a JSON object:
{{
  "reasoning": "2-4 sentences of PRIVATE strategy referencing your hidden objectives.",
  "message": "1-3 sentences you say aloud.",
  "edits": [{{"clause_id": "ct7", "op": "replace", "old": "<current text>", "new": "<your proposed text>", "rationale": "one line"}}],
  "decision": {{"type": "counter"}}
}}"""
        return call_llm(system, user, expect_keys=("edits", "decision"))

    def _vendor_contract_turn(self, awardee_id):
        party = self.parties[awardee_id]
        system = self._role_system(party, extra=(
            "You are now negotiating your commercial terms of service directly with the Contracting Officer "
            "as the awardee. Respond to their latest edits/asks. You want maximum protection and rights under "
            "these terms per your hidden objectives, but you also want a signed contract. Keep proposed clause "
            "text focused (roughly 80-140 words)."
        ))
        contract_thread = [t for t in self.turns if t["doc"] == "contract"]
        user = f"""CURRENT CONTRACT:
{clauses_to_text(self.contract_doc)}

NEGOTIATION SO FAR:
{self.public_log(contract_thread)}

Respond with ONLY a JSON object:
{{
  "reasoning": "2-4 sentences of PRIVATE strategy referencing your hidden objectives.",
  "message": "1-3 sentences you say aloud.",
  "edits": [{{"clause_id": "ct7", "op": "replace", "old": "<current text>", "new": "<your proposed text>", "rationale": "one line"}}]
}}"""
        return call_llm(system, user, expect_keys=("edits",))

    # -- shared system-prompt builders -------------------------------------
    def _role_system(self, party, extra=""):
        sc = self.scenario
        pub = "\n".join(f"- {o}" for o in party["public_objectives"])
        hid = "\n".join(f"- {o}" for o in party["hidden_objectives"])
        return f"""You are an autonomous agent in a live, multi-agent federal source-selection and contract
negotiation demonstration. You are playing: {party['name']} ({party['role']}).

SOLICITATION: {sc['solicitation']['number']} — {sc['solicitation']['agency']}
REQUIREMENT: {sc['solicitation']['requirement_summary']}
BASIS OF AWARD: {sc['solicitation']['section_M']['basis_of_award']}

YOUR PUBLIC OBJECTIVES (other parties can infer these):
{pub}

YOUR HIDDEN OBJECTIVES (never state these outright; they drive your private reasoning and strategy):
{hid}

{extra}

Always respond with ONLY a JSON object (no markdown, no code fences) in the exact shape requested."""

    def _offeror_system(self, party, phase_note):
        sc = self.scenario
        return self._role_system(party, extra=(
            f"{phase_note}\n\n"
            "Keep each clause's text focused — roughly 80-140 words per clause is plenty for this demo; "
            "do not write a full proposal narrative. Only include edits for clauses you are actually "
            "changing this turn — never restate a clause you are leaving as-is. Do not insert new clauses "
            "(stick to editing tech/mgmt/pp/price) unless there is truly no other way to make your point, "
            "and even then add at most one. "
            "'op' is one of: 'replace', 'insert_after', 'delete'. For 'replace'/'delete', clause_id must be an "
            "EXISTING clause; copy its current text into 'old' verbatim. Also include a top-level "
            "\"price_numeric\": <integer, no symbols> with your current total annual price.\n\n"
            "Respond with ONLY a JSON object:\n"
            "{\n"
            '  "reasoning": "2-4 sentences of PRIVATE strategy, referencing your hidden objectives explicitly.",\n'
            '  "message": "1-3 sentences you say aloud at the table.",\n'
            '  "edits": [{"clause_id": "tech", "op": "replace", "old": "<current text>", "new": "<your text>", "rationale": "one line"}, ...],\n'
            '  "price_numeric": 3200000\n'
            "}"
        ))


def main():
    argv = [a for a in sys.argv[1:] if a != "--dry-run"]
    scenario_path = argv[0] if len(argv) > 0 else "scenario.json"
    out_path = argv[1] if len(argv) > 1 else "transcript.json"

    with open(scenario_path) as f:
        scenario = json.load(f)

    rec = Recorder(scenario)

    print("Phase 1/6: Proposals")
    rec.run_proposals()

    print("Phase 2/6: Discussions")
    rec.run_discussions()

    print("Phase 3/6: Evaluation")
    evaluations = rec.run_evaluation()

    print("Phase 4/6: Award decision")
    awardee_id = rec.run_award_decision(evaluations)

    print("Phase 5/6: Debrief & protest")
    protests = rec.run_debrief_and_protest(evaluations, awardee_id)

    print("Phase 6/6: Contract negotiation")
    rec.run_contract_negotiation(awardee_id)

    winner_name = rec.parties[awardee_id]["name"]
    transcript = {
        "scenario": scenario,
        "turns": rec.turns,
        "documents_final": {
            **{f"proposal_{oid}": rec.proposal_docs[oid] for oid in rec.offeror_ids},
            "contract": rec.contract_doc,
        },
        "outcome": {
            "awarded_to": awardee_id,
            "awarded_to_name": winner_name,
            "price_comparison": {oid: rec.price.get(oid) for oid in rec.offeror_ids},
            "protests": protests,
            "executed": True,
        },
    }
    with open(out_path, "w") as f:
        json.dump(transcript, f, indent=2)

    print(f"\nDone. {len(rec.turns)} turns. Awarded to: {winner_name}")
    protest_summary = ", ".join(f"{p['offeror_name']}: {'PROTEST' if p['protest'] else 'no protest'}" for p in protests) or "(no losing offerors)"
    print(f"Protest decisions — {protest_summary}")
    print(f"Wrote {out_path} — open replay.html and load it.")


if __name__ == "__main__":
    main()
