"""
Ahmed ke store ka Sequential Workflow
Har step ka output agle step ka input hai:

 1 normalize -> 2 classify -> 3 extract -> 4 policy_check
   -> 5 draft_reply -> 6 guardrail -> 7 route -> 8 record
"""
import csv, os, re
from dataclasses import dataclass, field, asdict
from datetime import datetime

RETURN_WINDOW_DAYS = 7
CONF_THRESHOLD = 0.6


@dataclass
class Ticket:
    channel: str
    raw: str
    text: str = ""
    intent: str = "unknown"
    confidence: float = 0.0
    angry: bool = False
    order_no: str | None = None
    days_since_delivery: int | None = None
    item_unused: bool | None = None
    policy: str = "n/a"          # allowed / denied / need_info / n/a
    draft: str = ""
    flags: list = field(default_factory=list)
    route: str = ""              # auto_send / human_review


# ---------- Step 1: normalize ----------
def normalize(t: Ticket) -> Ticket:
    t.text = re.sub(r"\s+", " ", t.raw.strip().lower())
    if not t.text:
        t.flags.append("empty_message")
    return t


# ---------- Step 2: classify (intent + sentiment + confidence) ----------
INTENT_KEYWORDS = {
    "refund":    ["refund", "paisay wapis", "paise wapis", "money back"],
    "return":    ["return", "wapis", "exchange", "badalna", "size chota", "size bara"],
    "complaint": ["kharab", "damage", "phata", "galat", "late", "nahi aya", "nahi aaya", "complaint"],
    "order":     ["order karna", "order chahiye", "buy", "khareedna", "available hai", "price"],
    "question":  ["kab", "kitne", "delivery", "?", "kaise"],
}
ANGRY_WORDS = ["bakwas", "worst", "fraud", "dhoka", "bekar", "scam", "ghatiya", "pareshan", "!!"]


def classify(t: Ticket) -> Ticket:
    scores = {k: sum(w in t.text for w in ws) for k, ws in INTENT_KEYWORDS.items()}
    total = sum(scores.values())
    if total:
        best = max(scores, key=scores.get)
        t.intent = best
        t.confidence = round(scores[best] / total, 2) if scores[best] > 1 or total == 1 else 0.5
        # do intents barabar hon to confidence kam
        if sorted(scores.values())[-1] == sorted(scores.values())[-2]:
            t.confidence = 0.4
    t.angry = any(w in t.text for w in ANGRY_WORDS)
    return t


# ---------- Step 3: extract entities ----------
def extract(t: Ticket) -> Ticket:
    m = re.search(r"(?:ord[-\s]?|#)(\d{3,})", t.text)
    t.order_no = m.group(1) if m else None
    d = re.search(r"(\d+)\s*(?:din|days?)", t.text)
    t.days_since_delivery = int(d.group(1)) if d else None
    if re.search(r"use nahi|unused|pehna nahi|pehen(a|i) nahi|tags? (laga|on)", t.text):
        t.item_unused = True
    elif re.search(r"pehe?n\s*(liya|li|lia|chuk)|pehn(a|i) tha|use (kiya|kar liya|ho chuka)|istemal (kiya|ho)|dhula|wash", t.text):
        t.item_unused = False
    return t


# ---------- Step 4: policy check ----------
def policy_check(t: Ticket) -> Ticket:
    if t.intent not in ("return", "refund"):
        return t
    if t.days_since_delivery is None or (t.intent == "refund" and t.item_unused is None):
        t.policy = "need_info"
    elif t.days_since_delivery > RETURN_WINDOW_DAYS:
        t.policy = "denied"
    elif t.intent == "refund" and t.item_unused is False:
        t.policy = "denied"
    else:
        t.policy = "allowed"
    return t


# ---------- Step 5: draft reply ----------
SOFT = ("Aapko jo pareshani hui, uske liye hum dil se maazrat khwah hain. "
        "Aapki baat hamare liye bohat ahem hai. ")
POLICY_LINE = "Hamari policy: 7 din ke andar return, aur refund sirf unused items par."

TEMPLATES = {
    ("refund", "allowed"):   "Aapka refund request qabil-e-ghaur hai. Hum aapke order ki tasdeeq karke team ke zariye process shuru karte hain.",
    ("refund", "denied"):    "Afsos, hamari policy ke mutabiq yeh order refund ke qabil nahi. " + POLICY_LINE + " Hum koi aur madad kar sakte hain to zaroor batayen.",
    ("refund", "need_info"): "Refund check karne ke liye barah-e-karam order number, delivery ko kitne din hue, aur item unused hai ya nahi, yeh bata dein. " + POLICY_LINE,
    ("return", "allowed"):   "Aap return kar sakte hain. Item apni original halat mein rakhen, hum pickup/return ka tareeqa bhej dete hain.",
    ("return", "denied"):    "Afsos, return ki muddat (7 din) guzar chuki hai, is liye hum return nahi le sakte.",
    ("return", "need_info"): "Return ke liye barah-e-karam order number aur delivery ki tareekh bata dein. " + POLICY_LINE,
    ("complaint", "n/a"):    "Aapki shikayat note kar li gayi hai. Barah-e-karam order number aur masle ki tasveer bhej dein taake hum jaldi jaanch kar saken.",
    ("order", "n/a"):        "Shukriya! Order dene ke liye product ka naam, size aur address bata dein, hum confirm kar dete hain.",
    ("question", "n/a"):     "Shukriya poochne ka. Hamari team is baare mein aapko jaldi details bhej rahi hai.",
}


def draft_reply(t: Ticket) -> Ticket:
    body = TEMPLATES.get((t.intent, t.policy))
    if not body:
        t.draft = ""
        t.flags.append("no_template")
        return t
    t.draft = ("Assalam o Alaikum! " + (SOFT if t.angry else "") + body)
    return t


# ---------- Step 6: guardrail (jhoote wade pakarna) ----------
FORBIDDEN = [
    (r"\b(?:100\s?%|guaranteed|pakka)\b.*refund", "guaranteed_refund"),
    (r"\b(?:1[0-9]|[2-9][0-9])\s*din", "wrong_window"),          # 10+ din ka zikr
    (r"bina kisi shart|har haal mein", "unconditional_promise"),
    (r"refund (ho jaye ga|kar dein ge)", "refund_promise"),
]


def guardrail(t: Ticket) -> Ticket:
    for pattern, name in FORBIDDEN:
        if re.search(pattern, t.draft.lower()):
            t.flags.append(name)
    if t.policy == "denied" and re.search(r"process shuru|kar sakte hain\.", t.draft.lower()) and t.intent == "refund":
        t.flags.append("promise_vs_denied_policy")
    return t


# ---------- Step 7: route (auto ya insaan) ----------
def route(t: Ticket) -> Ticket:
    reasons = []
    if t.confidence < CONF_THRESHOLD:
        reasons.append("low_confidence")
    if t.flags:
        reasons.append("guardrail_or_template_flag")
    if t.angry and t.intent in ("refund", "return", "complaint"):
        reasons.append("angry_sensitive")
    if t.policy == "denied":
        reasons.append("policy_denial")   # inkar hamesha insaan dekhe
    t.route = "human_review" if reasons else "auto_send"
    if reasons:
        t.flags.append("route:" + "+".join(reasons))
    return t


# ---------- Step 8: record ----------
def record(t: Ticket, path="records.csv") -> Ticket:
    row = asdict(t)
    row["flags"] = "; ".join(t.flags)
    row["time"] = datetime.now().isoformat(timespec="seconds")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    return t


STEPS = [normalize, classify, extract, policy_check, draft_reply, guardrail, route, record]


def process(channel: str, message: str) -> Ticket:
    t = Ticket(channel=channel, raw=message)
    for step in STEPS:
        t = step(t)
    return t


if __name__ == "__main__":
    samples = [
        ("whatsapp", "Mujhe refund chahiye, order ORD-1042, 3 din pehle mila, use nahi kiya"),
        ("instagram", "Ye kya bakwas hai!! Dress phata hua aya, order #2231, refund do fraud ho tum"),
        ("email", "Order ORD-3310 ka return karna hai, 12 din ho gaye"),
        ("whatsapp", "Delivery kitne din mein hoti hai?"),
        ("instagram", "wapis karna hai shayad kharab bhi hai pata nahi"),
        ("whatsapp", "Ye black kurta available hai? price kya hai"),
    ]
    if os.path.exists("records.csv"):
        os.remove("records.csv")
    for ch, msg in samples:
        t = process(ch, msg)
        print(f"[{t.route.upper()}] intent={t.intent} conf={t.confidence} policy={t.policy} angry={t.angry}")
        print("  MSG  :", msg)
        print("  DRAFT:", t.draft)
        print("  FLAGS:", t.flags, "\n")