"""
LangGraph workflow (pipeline.py ke steps ko nodes bana kar)

START -> normalize -> classify -> extract -> llm_check -> policy -> template
      -> [template hai?] -> rephrase -> guardrail -> [guardrail fail?] -> fallback
      -> route -> record -> END
"""
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

import pipeline as P


class State(TypedDict, total=False):
    t: Any                 # Ticket
    template: str          # approved jawab (policy se)
    flags_before: list     # guardrail se pehle ke flags
    guard_failed: bool


def build_graph(llm_analyze, llm_rephrase, intents):
    # ---------- nodes ----------
    def normalize(s):
        return {"t": P.normalize(s["t"])}

    def classify(s):
        return {"t": P.classify(s["t"])}

    def extract(s):
        return {"t": P.extract(s["t"])}

    def llm_check(s):
        """LLM ki doosri raay + keyword se cross-check."""
        t = s["t"]
        kw_intent = t.intent
        out = llm_analyze(t.raw)
        llm_intent = out.get("intent") if out.get("intent") in intents else None

        if llm_intent is None:
            if kw_intent in ("question", "order"):
                t.confidence = 0.65                     # kam-risk intent: keyword par bharosa
            else:
                t.confidence = min(t.confidence, 0.5)   # refund/return/complaint: tasdeeq nahi -> insaan
        elif kw_intent == llm_intent:
            t.confidence = 0.9
        elif kw_intent == "unknown":
            t.intent, t.confidence = llm_intent, 0.65
        else:
            t.confidence = 0.4                          # ikhtilaf -> insaan

        t.angry = t.angry or (out.get("angry") is True)
        if t.order_no is None and isinstance(out.get("order_no"), (str, int)):
            t.order_no = str(out["order_no"])
        if t.days_since_delivery is None and isinstance(out.get("days_since_delivery"), int):
            t.days_since_delivery = out["days_since_delivery"]
        if t.item_unused is None and isinstance(out.get("item_unused"), bool):
            t.item_unused = out["item_unused"]
        return {"t": t}

    def policy(s):
        return {"t": P.policy_check(s["t"])}

    def template(s):
        t = P.draft_reply(s["t"])
        return {"t": t, "template": t.draft}

    def rephrase(s):
        t = s["t"]
        strip = lambda x: re.sub(r"^\s*(assalam[\s-]*o[\s-]*alaikum!?\s*)+", "", x, flags=re.I).strip()
        tmpl_body = strip(s["template"])
        body = strip(llm_rephrase(s["template"], t.angry))
        # bekaar LLM output (bohat lamba ya naya sawal) -> approved template
        if not body or len(body) > 3 * len(tmpl_body) or ("?" in body and "?" not in tmpl_body):
            body = tmpl_body
        t.draft = "Assalam o Alaikum! " + body     # salam sirf ek baar
        return {"t": t}

    def guardrail(s):
        t = s["t"]
        before = list(t.flags)
        t = P.guardrail(t)
        allowed = set(re.findall(r"\d+", s["template"])) | set(re.findall(r"\d+", t.order_no or ""))
        if set(re.findall(r"\d+", t.draft)) - allowed:
            t.flags.append("new_numbers")
        return {"t": t, "flags_before": before, "guard_failed": len(t.flags) > len(before)}

    def fallback(s):
        """LLM ka draft ghalat -> approved template par wapis."""
        t = s["t"]
        t.flags = list(s["flags_before"])
        t.draft = s["template"]
        return {"t": t}

    def route(s):
        return {"t": P.route(s["t"])}

    def record(s):
        return {"t": P.record(s["t"])}

    # ---------- conditional edges ----------
    def after_normalize(s):
        return "route" if "empty_message" in s["t"].flags else "classify"

    def after_template(s):
        return "rephrase" if s.get("template") else "route"

    def after_guardrail(s):
        return "fallback" if s.get("guard_failed") else "route"

    # ---------- graph ----------
    g = StateGraph(State)
    for name, fn in [("normalize", normalize), ("classify", classify), ("extract", extract),
                     ("llm_check", llm_check), ("policy", policy), ("template", template),
                     ("rephrase", rephrase), ("guardrail", guardrail), ("fallback", fallback),
                     ("route", route), ("record", record)]:
        g.add_node(name, fn)

    g.add_edge(START, "normalize")
    g.add_conditional_edges("normalize", after_normalize, ["classify", "route"])
    g.add_edge("classify", "extract")
    g.add_edge("extract", "llm_check")
    g.add_edge("llm_check", "policy")
    g.add_edge("policy", "template")
    g.add_conditional_edges("template", after_template, ["rephrase", "route"])
    g.add_edge("rephrase", "guardrail")
    g.add_conditional_edges("guardrail", after_guardrail, ["fallback", "route"])
    g.add_edge("fallback", "route")
    g.add_edge("route", "record")
    g.add_edge("record", END)
    return g.compile()