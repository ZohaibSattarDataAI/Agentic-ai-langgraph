"""
LangGraph E-Commerce Workflow

Flow:

START
  ↓
normalize
  ↓
classify
  ↓
extract
  ↓
llm_check
  ↓
policy
  ↓
      ┌── RAG question/info ──> rag ──> route
      │
      └── normal case ──> template ──> rephrase ──> guardrail
                                                   ↓
                                            fail? fallback
                                                   ↓
                                                 route
                                                   ↓
                                                record
                                                   ↓
                                                  END

RAG:
    Current rag.py
    ChromaDB
    products.csv
    policies.md
    nomic-embed-text

Important:
    Return/refund/complaint ke actual cases
    pipeline.py ki policy logic se handle honge.

    RAG sirf information/policy questions ke
    liye knowledge provide karega.
"""

import os
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

import pipeline as P


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RAG_ENABLED = (
    os.environ.get(
        "RAG_ENABLED",
        "true",
    )
    .strip()
    .lower()
    in ("1", "true", "yes", "on")
)

RAG_MIN_SCORE = float(
    os.environ.get(
        "RAG_MIN_SCORE",
        "0.45",
    )
)


# ============================================================
# QUESTION / PURCHASE DETECTION
# ============================================================

INFO_Q = re.compile(
    r"""
    policy
    |return
    |refund
    |exchange
    |delivery
    |shipping
    |charges
    |price
    |cost
    |kitne\s+din
    |kitne\s+ka
    |kitni\s+price
    |kaise
    |kya
    |kyun
    |kab
    |where
    |when
    |how
    |what
    |which
    |can\s+i
    |sakt
    |\?
    """,
    re.I | re.X,
)

PURCHASE = re.compile(
    r"""
    order
    |khareed
    |kharid
    |buy
    |purchase
    |lena
    |chahiye
    """,
    re.I | re.X,
)


# ============================================================
# RAG KNOWLEDGE BASE
# ============================================================

def _make_kb():
    """
    Current rag.py ke KnowledgeBase ko initialize karta hai.

    RAG configuration:
        products.csv
        policies.md
        chroma_db
        nomic-embed-text

    Agar RAG fail ho jaye to bot completely crash nahi karega.
    """

    if not RAG_ENABLED:

        print(
            "[kb] RAG disabled.",
            flush=True,
        )

        return None

    try:

        import rag

        print(
            "[kb] Loading current RAG knowledge base...",
            flush=True,
        )

        kb = rag.KnowledgeBase()

        kb.build()

        print(
            "[kb] RAG ready.",
            flush=True,
        )

        return kb

    except Exception as error:

        print(
            f"[kb] RAG unavailable: {error}",
            flush=True,
        )

        return None


# ============================================================
# LANGGRAPH STATE
# ============================================================

class State(TypedDict, total=False):

    # Ticket object created/updated by pipeline.py
    t: Any

    # Approved template response
    template: str

    # Flags before guardrail
    flags_before: list

    # Guardrail status
    guard_failed: bool

    # Whether RAG successfully answered
    rag_ok: bool

    # RAG score
    rag_score: float

    # RAG sources
    rag_sources: list


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(
    llm_analyze,
    llm_rephrase,
    intents,
    kb="auto",
):

    # --------------------------------------------------------
    # Load RAG
    # --------------------------------------------------------

    if kb == "auto":

        kb = _make_kb()


    # ========================================================
    # NODE 1 — NORMALIZE
    # ========================================================

    def normalize(s):

        ticket = P.normalize(
            s["t"]
        )

        return {
            "t": ticket
        }


    # ========================================================
    # NODE 2 — CLASSIFY
    # ========================================================

    def classify(s):

        ticket = P.classify(
            s["t"]
        )

        return {
            "t": ticket
        }


    # ========================================================
    # NODE 3 — EXTRACT
    # ========================================================

    def extract(s):

        ticket = P.extract(
            s["t"]
        )

        return {
            "t": ticket
        }


    # ========================================================
    # NODE 4 — LLM CHECK
    # ========================================================

    def llm_check(s):

        """
        LLM ki doosri opinion + keyword classification
        ka cross-check.
        """

        ticket = s["t"]

        keyword_intent = ticket.intent

        try:

            result = llm_analyze(
                ticket.raw
            )

        except Exception as error:

            print(
                f"[llm_check] Error: {error}",
                flush=True,
            )

            result = {}


        # ----------------------------------------------------
        # Extract LLM intent
        # ----------------------------------------------------

        llm_intent = result.get(
            "intent"
        )

        if llm_intent not in intents:

            llm_intent = None


        # ----------------------------------------------------
        # Compare keyword + LLM intent
        # ----------------------------------------------------

        if llm_intent is None:

            if keyword_intent in (
                "question",
                "order",
            ):

                ticket.confidence = 0.65

            else:

                ticket.confidence = min(
                    ticket.confidence,
                    0.5,
                )

        elif keyword_intent == llm_intent:

            ticket.confidence = 0.9

        elif keyword_intent == "unknown":

            ticket.intent = llm_intent
            ticket.confidence = 0.65

        else:

            # Intent disagreement
            ticket.confidence = 0.4


        # ----------------------------------------------------
        # Extract additional LLM information
        # ----------------------------------------------------

        if result.get("angry") is True:

            ticket.angry = True


        if (
            ticket.order_no is None
            and isinstance(
                result.get("order_no"),
                (str, int),
            )
        ):

            ticket.order_no = str(
                result["order_no"]
            )


        if (
            ticket.days_since_delivery is None
            and isinstance(
                result.get(
                    "days_since_delivery"
                ),
                int,
            )
        ):

            ticket.days_since_delivery = (
                result["days_since_delivery"]
            )


        if (
            ticket.item_unused is None
            and isinstance(
                result.get("item_unused"),
                bool,
            )
        ):

            ticket.item_unused = (
                result["item_unused"]
            )


        return {
            "t": ticket
        }


    # ========================================================
    # NODE 5 — POLICY
    # ========================================================

    def policy(s):

        ticket = P.policy_check(
            s["t"]
        )

        return {
            "t": ticket
        }


    # ========================================================
    # CHECK WHETHER RAG IS NEEDED
    # ========================================================

    def wants_rag(ticket):

        if kb is None:

            return False


        # General questions
        if ticket.intent in (
            "question",
            "order",
            "unknown",
        ):

            return True


        # ----------------------------------------------------
        # Return / refund / complaint
        #
        # Only use RAG when customer is asking about
        # the policy itself.
        #
        # Actual order case goes through policy.py.
        # ----------------------------------------------------

        if ticket.intent in (
            "return",
            "refund",
            "complaint",
        ):

            return (
                not ticket.angry
                and not ticket.order_no
                and bool(
                    INFO_Q.search(
                        ticket.text
                    )
                )
            )


        return False


    # ========================================================
    # NODE 6 — RAG
    # ========================================================

    def rag_node(s):

        ticket = s["t"]

        print(
            "\n[kb] Searching knowledge base...",
            flush=True,
        )

        try:

            result = kb.answer(
                ticket.raw
            )

        except Exception as error:

            print(
                f"[kb] Search error: {error}",
                flush=True,
            )

            return {
                "t": ticket,
                "rag_ok": False,
            }


        # ----------------------------------------------------
        # Read result safely
        # ----------------------------------------------------

        ok = bool(
            result.get(
                "ok",
                False,
            )
        )

        text = str(
            result.get(
                "text",
                "",
            )
            or ""
        ).strip()

        score = float(
            result.get(
                "score",
                0.0,
            )
            or 0.0
        )

        sources = result.get(
            "sources",
            [],
        )


        print(
            f"[kb] Score: {score:.2f}",
            flush=True,
        )

        print(
            f"[kb] Sources: {sources}",
            flush=True,
        )


        # ----------------------------------------------------
        # Reject weak / empty RAG results
        # ----------------------------------------------------

        if (
            not ok
            or not text
            or score < RAG_MIN_SCORE
        ):

            print(
                "[kb] No reliable answer found.",
                flush=True,
            )

            if ticket.intent in (
                "question",
                "unknown",
            ):

                if "kb_no_answer" not in ticket.flags:

                    ticket.flags.append(
                        "kb_no_answer"
                    )

            return {
                "t": ticket,
                "rag_ok": False,
                "rag_score": score,
                "rag_sources": sources,
            }


        # ----------------------------------------------------
        # Build customer response
        # ----------------------------------------------------

        body = text


        # ----------------------------------------------------
        # Purchase intent
        # ----------------------------------------------------

        if PURCHASE.search(
            ticket.text
        ):

            body += (
                "\n\nOrder karne ke liye "
                "product ka naam, size, rang "
                "aur delivery address bata dein."
            )


        # ----------------------------------------------------
        # Final RAG draft
        # ----------------------------------------------------

        ticket.draft = (
            "Assalam o Alaikum! "
            + body
        )


        # RAG result is considered high confidence
        ticket.confidence = max(
            ticket.confidence,
            0.85,
        )


        # Store source information
        source_text = "; ".join(
            str(source)
            for source in sources
        )

        ticket.policy = (
            "kb:" + source_text
        )[:120]


        print(
            "[kb] Answer found.",
            flush=True,
        )


        return {
            "t": ticket,
            "rag_ok": True,
            "rag_score": score,
            "rag_sources": sources,
        }


    # ========================================================
    # NODE 7 — TEMPLATE
    # ========================================================

    def template(s):

        ticket = P.draft_reply(
            s["t"]
        )

        return {
            "t": ticket,
            "template": ticket.draft,
        }


    # ========================================================
    # NODE 8 — REPHRASE
    # ========================================================

    def rephrase(s):

        ticket = s["t"]

        template_text = s.get(
            "template",
            "",
        )


        # ----------------------------------------------------
        # Remove duplicate Salam
        # ----------------------------------------------------

        def strip_salam(text):

            return re.sub(
                r"^\s*"
                r"(assalam[\s-]*o[\s-]*alaikum!?\s*)+",
                "",
                text or "",
                flags=re.I,
            ).strip()


        template_body = strip_salam(
            template_text
        )


        # ----------------------------------------------------
        # Ask LLM to naturally rephrase
        # ----------------------------------------------------

        try:

            llm_body = llm_rephrase(
                template_text,
                ticket.angry,
            )

        except Exception as error:

            print(
                f"[rephrase] Error: {error}",
                flush=True,
            )

            llm_body = ""


        body = strip_salam(
            llm_body
        )


        # ----------------------------------------------------
        # Validate LLM output
        # ----------------------------------------------------

        invalid_output = (
            not body
            or len(body) > 3 * max(
                len(template_body),
                1,
            )
            or (
                "?" in body
                and "?" not in template_body
            )
        )


        if invalid_output:

            body = template_body


        # ----------------------------------------------------
        # Add Salam exactly once
        # ----------------------------------------------------

        ticket.draft = (
            "Assalam o Alaikum! "
            + body
        )


        return {
            "t": ticket
        }


    # ========================================================
    # NODE 9 — GUARDRAIL
    # ========================================================

    def guardrail(s):

        ticket = s["t"]

        before = list(
            ticket.flags
        )


        # Run existing pipeline guardrails
        ticket = P.guardrail(
            ticket
        )


        # ----------------------------------------------------
        # Number protection
        #
        # LLM must not invent new numbers.
        # ----------------------------------------------------

        allowed_numbers = set(
            re.findall(
                r"\d+",
                s.get(
                    "template",
                    "",
                ),
            )
        )


        allowed_numbers.update(
            re.findall(
                r"\d+",
                ticket.order_no or "",
            )
        )


        draft_numbers = set(
            re.findall(
                r"\d+",
                ticket.draft or "",
            )
        )


        if draft_numbers - allowed_numbers:

            if "new_numbers" not in ticket.flags:

                ticket.flags.append(
                    "new_numbers"
                )


        guard_failed = (
            len(ticket.flags)
            > len(before)
        )


        return {
            "t": ticket,
            "flags_before": before,
            "guard_failed": guard_failed,
        }


    # ========================================================
    # NODE 10 — FALLBACK
    # ========================================================

    def fallback(s):

        """
        Agar LLM rephrase guardrail fail kare,
        approved template par wapas aao.
        """

        ticket = s["t"]

        ticket.flags = list(
            s.get(
                "flags_before",
                [],
            )
        )

        ticket.draft = s.get(
            "template",
            ticket.draft,
        )


        return {
            "t": ticket
        }


    # ========================================================
    # NODE 11 — ROUTE
    # ========================================================

    def route(s):

        ticket = P.route(
            s["t"]
        )

        return {
            "t": ticket
        }


    # ========================================================
    # NODE 12 — RECORD
    # ========================================================

    def record(s):

        ticket = P.record(
            s["t"]
        )

        return {
            "t": ticket
        }


    # ========================================================
    # CONDITIONAL ROUTING
    # ========================================================

    def after_normalize(s):

        ticket = s["t"]

        if "empty_message" in ticket.flags:

            return "route"

        return "classify"


    def after_policy(s):

        ticket = s["t"]

        if wants_rag(ticket):

            return "rag"

        return "template"


    def after_rag(s):

        ticket = s["t"]

        if s.get("rag_ok"):

            return "route"


        # ----------------------------------------------------
        # If RAG has no answer:
        #
        # Existing policy/template system handles
        # return/refund/complaint/order.
        #
        # General unknown/question goes to route.
        # ----------------------------------------------------

        if ticket.intent in (
            "return",
            "refund",
            "complaint",
            "order",
        ):

            return "template"

        return "route"


    def after_template(s):

        if s.get("template"):

            return "rephrase"

        return "route"


    def after_guardrail(s):

        if s.get("guard_failed"):

            return "fallback"

        return "route"


    # ========================================================
    # CREATE LANGGRAPH
    # ========================================================

    graph = StateGraph(
        State
    )


    # --------------------------------------------------------
    # Add nodes
    # --------------------------------------------------------

    graph.add_node(
        "normalize",
        normalize,
    )

    graph.add_node(
        "classify",
        classify,
    )

    graph.add_node(
        "extract",
        extract,
    )

    graph.add_node(
        "llm_check",
        llm_check,
    )

    graph.add_node(
        "policy",
        policy,
    )

    graph.add_node(
        "rag",
        rag_node,
    )

    graph.add_node(
        "template",
        template,
    )

    graph.add_node(
        "rephrase",
        rephrase,
    )

    graph.add_node(
        "guardrail",
        guardrail,
    )

    graph.add_node(
        "fallback",
        fallback,
    )

    graph.add_node(
        "route",
        route,
    )

    graph.add_node(
        "record",
        record,
    )


    # ========================================================
    # GRAPH EDGES
    # ========================================================

    graph.add_edge(
        START,
        "normalize",
    )


    graph.add_conditional_edges(
        "normalize",
        after_normalize,
        [
            "classify",
            "route",
        ],
    )


    graph.add_edge(
        "classify",
        "extract",
    )


    graph.add_edge(
        "extract",
        "llm_check",
    )


    graph.add_edge(
        "llm_check",
        "policy",
    )


    graph.add_conditional_edges(
        "policy",
        after_policy,
        [
            "rag",
            "template",
        ],
    )


    graph.add_conditional_edges(
        "rag",
        after_rag,
        [
            "route",
            "template",
        ],
    )


    graph.add_conditional_edges(
        "template",
        after_template,
        [
            "rephrase",
            "route",
        ],
    )


    graph.add_edge(
        "rephrase",
        "guardrail",
    )


    graph.add_conditional_edges(
        "guardrail",
        after_guardrail,
        [
            "fallback",
            "route",
        ],
    )


    graph.add_edge(
        "fallback",
        "route",
    )


    graph.add_edge(
        "route",
        "record",
    )


    graph.add_edge(
        "record",
        END,
    )


    # ========================================================
    # COMPILE
    # ========================================================

    return graph.compile()