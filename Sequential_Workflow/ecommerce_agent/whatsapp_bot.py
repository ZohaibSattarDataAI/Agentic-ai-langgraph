"""
WhatsApp Web + Ollama + LangGraph bot

Files (same folder):
    pipeline.py
    graph_pipeline.py
    rag.py
    whatsapp_bot.py
    .env

Run:
    python whatsapp_bot.py

First time:
    Chrome khulega -> QR scan karein.
    Login "wa_profile" folder mein save hota hai.
"""

import json
import os
import re
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pipeline as P


# ============================================================
# WINDOWS CONSOLE
# ============================================================

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(
            encoding="utf-8",
            errors="replace"
        )
    except Exception:
        pass


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def log(*args):
    print(*args, flush=True)


# ============================================================
# ENV
# ============================================================

def load_env(path=None):
    """
    Simple .env loader.
    Supports:
    - UTF-8 BOM
    - comments
    - quoted values
    """

    path = path or os.path.join(
        BASE_DIR,
        ".env"
    )

    if not os.path.exists(path):
        return

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        for line in file:

            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or "=" not in line
            ):
                continue

            key, value = line.split(
                "=",
                1
            )

            key = key.strip()
            value = value.strip()

            if (
                len(value) >= 2
                and value[0] in ("'", '"')
                and value[-1] == value[0]
            ):
                value = value[1:-1]

            else:
                value = re.split(
                    r"\s+#",
                    value,
                    maxsplit=1
                )[0].strip()

            os.environ.setdefault(
                key,
                value
            )


def env_bool(
    key,
    default=False
):
    return os.environ.get(
        key,
        str(default)
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def env_int(
    key,
    default
):
    try:
        return int(
            os.environ.get(
                key,
                default
            )
        )
    except (
        TypeError,
        ValueError
    ):
        return default


load_env()


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://localhost:11434/api/chat"
)

MODEL = os.environ.get(
    "MODEL",
    "qwen2.5:1.5b"
)

PROFILE_DIR = os.path.join(
    BASE_DIR,
    "wa_profile"
)


# ------------------------------------------------------------
# Allowed chats
# ------------------------------------------------------------

ALLOWED_CHATS = [
    item.strip()
    for item in os.environ.get(
        "ALLOWED_CHATS",
        "TEST CONTACT NAME"
    ).split(",")
    if item.strip()
]

ALLOW_ALL = "*" in ALLOWED_CHATS


# ------------------------------------------------------------
# Bot behaviour
# ------------------------------------------------------------

DRY_RUN = env_bool(
    "DRY_RUN",
    True
)

HOLD_REPLY = os.environ.get(
    "HOLD_REPLY",
    ""
).strip()

REPHRASE = env_bool(
    "REPHRASE",
    False
)

TYPING_INDICATOR = env_bool(
    "TYPING_INDICATOR",
    True
)

MIN_TYPING_SECONDS = env_int(
    "MIN_TYPING_SECONDS",
    3
)

AUTO_OPEN_ALLOWED = env_bool(
    "AUTO_OPEN_ALLOWED",
    True
)

SCAN_EVERY = max(
    1,
    env_int(
        "SCAN_EVERY",
        4
    )
)

DEBUG = env_bool(
    "DEBUG",
    False
)

POLL_SECONDS = max(
    1,
    env_int(
        "POLL_SECONDS",
        3
    )
)

MAX_TRAILING = 5

SEND_VERIFY_SECONDS = 8


# ------------------------------------------------------------
# Closing messages
# ------------------------------------------------------------

CLOSING_RE = re.compile(
    r"^\s*("
    r"ok(ay)?"
    r"|thanks?( you)?"
    r"|thank u"
    r"|shukr(iya|ia)"
    r"|jazak\w*"
    r"|(theek|thik|acha|achha|accha)( hai| ha)?"
    r"|👍"
    r"|🙏"
    r"|❤️"
    r"|😊"
    r")[\s.!]*$",
    re.I
)


INTENTS = {
    "order",
    "return",
    "refund",
    "complaint",
    "question",
}


# ============================================================
# STRING NORMALIZATION
# ============================================================

def norm(text: str) -> str:
    """
    Comparison ke liye text normalize karta hai.
    Spaces, punctuation aur case ignore hota hai.
    """

    return re.sub(
        r"[^a-z0-9\u0600-\u06ff]",
        "",
        (text or "").lower()
    )


def compact_text(text: str) -> str:
    """
    WhatsApp verification ke liye text compact karta hai.
    """

    return re.sub(
        r"\s+",
        " ",
        (text or "").strip()
    )


# ============================================================
# OLLAMA
# ============================================================

def ollama(
    messages,
    json_mode=False,
    temperature=0.2,
    num_predict=200
):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_ctx": 2048,
            "num_predict": num_predict,
        },
    }

    if json_mode:
        payload["format"] = "json"

    request = urllib.request.Request(
        OLLAMA_URL,
        json.dumps(payload).encode("utf-8"),
        {
            "Content-Type": "application/json"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=180
    ) as response:

        data = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    return data["message"]["content"]


def ollama_check():

    base = OLLAMA_URL.split(
        "/api/",
        1
    )[0]

    try:

        with urllib.request.urlopen(
            base + "/api/tags",
            timeout=5
        ) as response:

            data = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

        names = [
            model.get("name", "")
            for model in data.get(
                "models",
                []
            )
        ]

        if MODEL in names:

            log(
                f"[ok] Ollama chal raha hai, "
                f"model '{MODEL}' maujood hai"
            )

            log(
                "     model memory mein load ho raha hai "
                "(pehli baar 30-60 second lag sakte hain)..."
            )

            started = time.time()

            try:

                ollama(
                    [
                        {
                            "role": "user",
                            "content": "hi"
                        }
                    ],
                    num_predict=1
                )

                log(
                    f"     model tayyar "
                    f"({time.time() - started:.0f}s)"
                )

            except Exception as error:

                log(
                    f"     [warm-up fail] {error}"
                )

        else:

            log(
                f"[WARNING] Ollama chal raha hai "
                f"lekin model '{MODEL}' nahi mila."
            )

            log(
                f"          Available models: {names}"
            )

            log(
                f"          Chalayen: "
                f"ollama pull {MODEL}"
            )

    except Exception as error:

        log(
            f"[WARNING] Ollama se rabta nahi: "
            f"{error}"
        )

        log(
            "          Naye CMD mein "
            "'ollama serve' chalayen."
        )


# ============================================================
# LLM CLASSIFIER
# ============================================================

def llm_analyze(
    text: str
) -> dict:

    system = (
        "Tum ek kapron ke online store ke "
        "customer message classifier ho. "

        "Message Roman Urdu, Urdu ya English "
        "mix ho sakta hai. "

        'Sirf is format mein JSON do: '
        '{"intent":"order|return|refund|complaint|'
        'question|other",'
        '"angry":true,'
        '"order_no":null,'
        '"days_since_delivery":null,'
        '"item_unused":null}. '

        "angry tab true jab customer ghussay "
        "ya badtameezi mein ho. "

        "Jo maloom na ho wo null rakho."
    )

    try:

        result = ollama(
            [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            json_mode=True,
            temperature=0,
            num_predict=120,
        )

        data = json.loads(
            result
        )

        return (
            data
            if isinstance(data, dict)
            else {}
        )

    except Exception as error:

        log(
            "  [LLM analyze fail]",
            error
        )

        return {}


# ============================================================
# LLM REPHRASE
# ============================================================

def llm_rephrase(
    template: str,
    angry: bool
) -> str:

    if not REPHRASE:
        return template

    system = (
        "Tum ek online kapron ke store ki "
        "customer support ho. "

        "Neeche diya gaya approved jawab "
        "Roman Urdu mein natural aur polite "
        "andaaz mein dobara likho. "

        "SAKHT QAWAID: "
        "matlab bilkul wohi rahe; "
        "koi naya wada, discount, refund, "
        "muddat ya number shamil NA karo. "

        "3 jumlon se zyada nahi. "
        "Sirf jawab likho."
    )

    if angry:

        system += (
            " Customer naraz hai, "
            "is liye lehja bohat naram "
            "aur maazrat wala rakho."
        )

    try:

        result = ollama(
            [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": template,
                },
            ],
            temperature=0.3,
            num_predict=160,
        ).strip()

        return result or template

    except Exception as error:

        log(
            "  [LLM rephrase fail]",
            error
        )

        return template


# ============================================================
# LANGGRAPH
# ============================================================

_GRAPH = None


def get_graph():

    global _GRAPH

    if _GRAPH is None:

        from graph_pipeline import build_graph

        _GRAPH = build_graph(
            llm_analyze,
            llm_rephrase,
            INTENTS
        )

    return _GRAPH


def handle_message(
    channel: str,
    text: str
) -> P.Ticket:

    graph = get_graph()

    result = graph.invoke(
        {
            "t": P.Ticket(
                channel=channel,
                raw=text
            )
        }
    )

    return result["t"]


# ============================================================
# CHAT ACCESS
# ============================================================

def is_allowed(
    name: str
) -> bool:

    if ALLOW_ALL:
        return bool(name)

    normalized = norm(name)

    if not normalized:
        return False

    return any(
        normalized == norm(allowed)
        for allowed in ALLOWED_CHATS
    )


# ============================================================
# WHATSAPP BOT
# ============================================================

class Bot:

    def __init__(
        self,
        page
    ):

        self.page = page

        # Processed incoming message IDs
        self.handled = set()

        # Already printed diagnostic messages
        self.noted = set()

        # Round robin allowed chats
        self.rr = 0

        # Last opened chat
        self.hint = ""

        # Reason for read failure
        self.last_reason = ""

        # Bot ke sent texts
        self.sent_norms = set()

        # Send safety limiter
        self.send_times = []


    # ========================================================
    # LOG ONCE
    # ========================================================

    def note(
        self,
        key,
        message
    ):

        if key in self.noted:
            return

        self.noted.add(key)

        log(message)


    # ========================================================
    # CHAT NAME
    # ========================================================

    def chat_name(self) -> str:

        selectors = (
            "#main header span[dir='auto']",
            "#main header span[title]",
            "#main header [title]",
        )

        for selector in selectors:

            try:

                locator = (
                    self.page
                    .locator(selector)
                    .first
                )

                if locator.count() == 0:
                    continue

                text = (
                    locator
                    .inner_text(timeout=1000)
                    or ""
                ).strip()

                if not text:

                    text = (
                        locator
                        .get_attribute(
                            "title",
                            timeout=1000
                        )
                        or ""
                    ).strip()

                if text:
                    return text

            except Exception:
                continue

        return self.hint


    # ========================================================
    # MESSAGE TEXT
    # ========================================================

    @staticmethod
    def message_text(
        element
    ) -> str:

        try:

            parts = (
                element
                .locator(
                    "span.selectable-text"
                )
                .all_inner_texts()
            )

            unique = []

            for part in (
                value.strip()
                for value in parts
            ):

                if (
                    part
                    and (
                        not unique
                        or unique[-1] != part
                    )
                ):
                    unique.append(part)

            if unique:

                return " ".join(
                    unique
                )

            raw = (
                element
                .inner_text(
                    timeout=1000
                )
                or ""
            ).strip()

            if re.fullmatch(
                r"[\d:\s.apmAPM]*",
                raw
            ):
                return ""

            raw = re.sub(
                r"\s*\d{1,2}:\d{2}\s*"
                r"([AaPp]\.?[Mm]\.?)?\s*$",
                "",
                raw
            ).strip()

            return raw

        except Exception:

            return ""


    # ========================================================
    # MESSAGE ROWS
    # ========================================================

    def message_rows(self):

        selectors = (
            "#main [data-id^='false_'], "
            "#main [data-id^='true_']",

            "#main [data-id]",

            "#main div.message-in, "
            "#main div.message-out",
        )

        for selector in selectors:

            try:

                locator = (
                    self.page
                    .locator(selector)
                )

                if locator.count() > 0:
                    return locator

            except Exception:
                continue

        return None


    # ========================================================
    # OUTGOING MESSAGE
    # ========================================================

    def is_outgoing(
        self,
        element,
        data_id: str
    ) -> bool:

        # Old WhatsApp IDs
        if data_id.startswith("true_"):
            return True

        if data_id.startswith("false_"):
            return False


        # message-out class
        try:

            if element.locator(
                ".message-out"
            ).count() > 0:
                return True

        except Exception:
            pass


        # outgoing status icons
        try:

            if element.locator(
                "[data-icon^='msg-'], "
                "[data-icon^='status-']"
            ).count() > 0:

                return True

        except Exception:
            pass


        # Bubble position fallback
        try:

            bubble = (
                element
                .locator(
                    "span.selectable-text"
                )
                .first
                .bounding_box(
                    timeout=1000
                )
            )

            container = (
                self.page
                .locator("#main")
                .first
                .bounding_box(
                    timeout=1000
                )
            )

            if bubble and container:

                left_gap = (
                    bubble["x"]
                    - container["x"]
                )

                right_gap = (
                    container["x"]
                    + container["width"]
                    - (
                        bubble["x"]
                        + bubble["width"]
                    )
                )

                if right_gap + 25 < left_gap:
                    return True

        except Exception:
            pass

        return False


    # ========================================================
    # READ LAST INCOMING
    # ========================================================

    def read_last_incoming(self):

        rows = self.message_rows()

        if rows is None:

            self.last_reason = "no_rows"

            return None

        try:

            total = rows.count()

        except Exception:

            self.last_reason = "no_rows"

            return None


        texts = []
        seen = set()
        last_id = None


        start = max(
            -1,
            total - 30
        )

        for index in range(
            total - 1,
            start,
            -1
        ):

            element = rows.nth(index)

            try:

                data_id = (
                    element
                    .get_attribute(
                        "data-id",
                        timeout=1000
                    )
                    or ""
                )

            except Exception:

                continue


            # Agar ID nahi hai to unique fallback
            if not data_id:

                try:

                    data_id = (
                        f"row-{index}-"
                        f"{hash(self.message_text(element))}"
                    )

                except Exception:

                    continue


            if data_id in seen:
                continue

            seen.add(data_id)


            # Text message hai?
            try:

                has_text = (
                    element
                    .locator(
                        "span.selectable-text"
                    )
                    .count()
                    > 0
                )

            except Exception:

                has_text = False


            if not has_text:

                # message-in/out class check
                try:

                    has_text = (
                        element
                        .locator(
                            ".copyable-text"
                        )
                        .count()
                        > 0
                    )

                except Exception:

                    pass


            if not has_text:
                continue


            text = self.message_text(
                element
            )


            # Hamara outgoing message
            if (
                self.is_outgoing(
                    element,
                    data_id
                )
                or (
                    text
                    and norm(text)
                    in self.sent_norms
                )
            ):
                break


            if last_id is None:

                last_id = data_id


            if text and text not in texts:

                texts.append(text)


            if len(texts) >= MAX_TRAILING:
                break


        if last_id is None:

            self.last_reason = "own_last"

            return None


        if not texts:

            self.last_reason = "no_text"

            return None


        texts.reverse()

        return (
            last_id,
            " ".join(texts)
        )


    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    def diagnose(
        self,
        name
    ):

        key = (
            "diag",
            name
        )

        if key in self.noted:
            return

        self.noted.add(key)

        page = self.page

        log(
            "  [diagnose] WhatsApp structure:"
        )

        selectors = (
            "#main",
            "#main header",
            "#main [data-id]",
            "#main [data-id^='false_']",
            "#main [data-id^='true_']",
            "div.message-in",
            "div.message-out",
            "span.selectable-text",
            "footer",
            "footer div[contenteditable='true']",
        )

        for selector in selectors:

            try:

                count = (
                    page
                    .locator(selector)
                    .count()
                )

            except Exception as error:

                count = (
                    f"error: "
                    f"{str(error)[:40]}"
                )

            log(
                f"     {selector}: {count}"
            )


        try:

            ids = page.locator(
                "#main [data-id]"
            ).evaluate_all(
                """
                els =>
                els.slice(-5)
                .map(e => e.getAttribute('data-id'))
                """
            )

            log(
                "     last data-id:",
                ids
            )

        except Exception:
            pass


        try:

            header = (
                page
                .locator(
                    "#main header"
                )
                .first
                .inner_text(
                    timeout=1000
                )
            )

            log(
                "     header:",
                repr(
                    (header or "")[:100]
                )
            )

        except Exception:
            pass


    # ========================================================
    # FIND MESSAGE BOX
    # ========================================================

    def find_box(self):

        selectors = (
            "#main footer div[contenteditable='true']",
            "#main footer [contenteditable='true']",
            "footer div[contenteditable='true'][role='textbox']",
            "footer div[contenteditable='true']",
            "#main div[contenteditable='true'][role='textbox']",
        )

        for selector in selectors:

            try:

                locator = (
                    self.page
                    .locator(selector)
                )

                if locator.count() > 0:

                    return locator.last

            except Exception:
                continue

        raise RuntimeError(
            "WhatsApp message box nahi mila."
        )


    # ========================================================
    # CLEAR MESSAGE BOX
    # ========================================================

    def clear_box(
        self,
        box
    ):

        try:

            box.click(
                timeout=3000
            )

        except Exception:

            pass

        try:

            box.press(
                "Control+A"
            )

            box.press(
                "Backspace"
            )

            return

        except Exception:

            pass

        try:

            self.page.keyboard.press(
                "Control+A"
            )

            self.page.keyboard.press(
                "Backspace"
            )

        except Exception:

            pass


    # ========================================================
    # BOX TEXT
    # ========================================================

    def typed_text(self) -> str:

        try:

            box = self.find_box()

            return (
                box
                .inner_text(
                    timeout=1000
                )
                or ""
            ).strip()

        except Exception:

            return ""


    # ========================================================
    # FIND SEND BUTTON
    # ========================================================

    def find_send_button(self):

        selectors = (
            "button[aria-label='Send']",
            "button[aria-label='send']",
            "[aria-label='Send']",
            "[aria-label='send']",
            "span[data-icon='send']",
            "div[role='button'][aria-label='Send']",
        )

        for selector in selectors:

            try:

                locator = (
                    self.page
                    .locator(selector)
                )

                if locator.count() > 0:

                    return locator.last

            except Exception:
                continue

        return None


    # ========================================================
    # CHECK SENT TEXT
    # ========================================================

    def sent_text_exists(
        self,
        message: str
    ) -> bool:

        target = compact_text(
            message
        )

        if not target:
            return False


        # 1. Exact text locator
        try:

            locator = self.page.locator(
                "#main span.selectable-text"
            )

            count = locator.count()

            for index in range(
                max(0, count - 20),
                count
            ):

                try:

                    current = compact_text(
                        locator
                        .nth(index)
                        .inner_text(
                            timeout=500
                        )
                    )

                    if current == target:
                        return True

                except Exception:
                    continue

        except Exception:
            pass


        # 2. Copyable text fallback
        try:

            locator = self.page.locator(
                "#main .copyable-text"
            )

            count = locator.count()

            for index in range(
                max(0, count - 20),
                count
            ):

                try:

                    current = compact_text(
                        locator
                        .nth(index)
                        .inner_text(
                            timeout=500
                        )
                    )

                    if current == target:
                        return True

                except Exception:
                    continue

        except Exception:
            pass


        return False


    # ========================================================
    # WAIT FOR SENT MESSAGE
    # ========================================================

    def wait_for_sent(
        self,
        message: str,
        seconds=SEND_VERIFY_SECONDS
    ) -> bool:

        end_time = (
            time.time()
            + seconds
        )

        while time.time() < end_time:

            if self.sent_text_exists(
                message
            ):
                return True

            try:
                self.page.wait_for_timeout(
                    500
                )
            except Exception:
                time.sleep(0.5)

        return False


    # ========================================================
    # SEND DIAGNOSIS
    # ========================================================

    def send_diagnose(self):

        try:

            info = self.page.locator(
                "div[contenteditable='true']"
            ).evaluate_all(
                """
                els => els.map(e => [
                    e.getAttribute('aria-label'),
                    e.getAttribute('data-tab'),
                    !!e.closest('footer'),
                    e.getAttribute('role'),
                    (e.innerText || '').slice(0, 50)
                ])
                """
            )

            log(
                "     [send-diagnose]:",
                info
            )

        except Exception as error:

            log(
                "     [send-diagnose fail]:",
                error
            )


    # ========================================================
    # SEND MESSAGE
    # ========================================================

    def send_text(
        self,
        message: str
    ):

        message = str(
            message or ""
        ).strip()

        if not message:

            raise ValueError(
                "Empty message send nahi kiya ja sakta."
            )


        # ----------------------------------------------------
        # Safety rate limiter
        # ----------------------------------------------------

        now = time.time()

        self.send_times = [
            timestamp
            for timestamp in self.send_times
            if now - timestamp < 60
        ]

        if len(self.send_times) >= 8:

            raise RuntimeError(
                "Safety stop: "
                "60 seconds mein 8 se zyada replies."
            )

        self.send_times.append(
            now
        )


        # ----------------------------------------------------
        # Remember own message
        # ----------------------------------------------------

        self.sent_norms.add(
            norm(message)
        )


        log(
            f"  [send] message length: "
            f"{len(message)}"
        )


        # ----------------------------------------------------
        # Attempt 1: locator.fill()
        # ----------------------------------------------------

        try:

            box = self.find_box()

            self.clear_box(
                box
            )

            try:

                box.fill(
                    message,
                    timeout=5000
                )

            except Exception:

                # contenteditable fill fallback
                box.click(
                    timeout=3000
                )

                self.page.keyboard.insert_text(
                    message
                )


            self.page.wait_for_timeout(
                500
            )


            typed = self.typed_text()

            if typed:

                log(
                    "  [send] text box mein aa gaya"
                )

                # Try Enter
                try:

                    box.press(
                        "Enter"
                    )

                except Exception:

                    self.page.keyboard.press(
                        "Enter"
                    )


                if self.wait_for_sent(
                    message
                ):

                    log(
                        "  [send] VERIFIED "
                        "(Enter)"
                    )

                    return


                # Try Send button
                button = self.find_send_button()

                if button:

                    try:

                        button.click(
                            timeout=3000
                        )

                        if self.wait_for_sent(
                            message
                        ):

                            log(
                                "  [send] VERIFIED "
                                "(Send button)"
                            )

                            return

                    except Exception as error:

                        log(
                            "  [send] Send button fail:",
                            error
                        )

            else:

                log(
                    "  [send] fill() ke baad "
                    "text box empty hai"
                )

        except Exception as error:

            log(
                "  [send] attempt 1 fail:",
                error
            )


        # ----------------------------------------------------
        # Attempt 2: keyboard typing
        # ----------------------------------------------------

        try:

            box = self.find_box()

            self.clear_box(
                box
            )

            box.click(
                timeout=3000
            )

            self.page.keyboard.type(
                message,
                delay=5
            )

            self.page.wait_for_timeout(
                700
            )


            if self.typed_text():

                log(
                    "  [send] keyboard typing successful"
                )

                try:

                    self.page.keyboard.press(
                        "Enter"
                    )

                except Exception:
                    pass


                if self.wait_for_sent(
                    message
                ):

                    log(
                        "  [send] VERIFIED "
                        "(keyboard + Enter)"
                    )

                    return


                button = self.find_send_button()

                if button:

                    try:

                        button.click(
                            timeout=3000
                        )

                        if self.wait_for_sent(
                            message
                        ):

                            log(
                                "  [send] VERIFIED "
                                "(keyboard + button)"
                            )

                            return

                    except Exception:
                        pass

            else:

                log(
                    "  [send] keyboard ke baad "
                    "text box empty hai"
                )

        except Exception as error:

            log(
                "  [send] attempt 2 fail:",
                error
            )


        # ----------------------------------------------------
        # Diagnostic
        # ----------------------------------------------------

        self.send_diagnose()

        raise RuntimeError(
            "WhatsApp message send verify nahi ho saka."
        )


    # ========================================================
    # PROCESS WITH TYPING
    # ========================================================

    def process_with_typing(
        self,
        text: str
    ) -> P.Ticket:

        log(
            "  [1/3] LLM se process ho raha hai..."
        )

        started = time.time()

        with ThreadPoolExecutor(
            max_workers=1
        ) as executor:

            future = executor.submit(
                handle_message,
                "whatsapp",
                text
            )

            typing_box = None


            # ------------------------------------------------
            # Typing indicator
            # ------------------------------------------------

            if (
                TYPING_INDICATOR
                and not DRY_RUN
            ):

                try:

                    typing_box = self.find_box()

                    typing_box.click(
                        timeout=3000
                    )

                except Exception as error:

                    log(
                        "  [typing indicator nahi chala]",
                        error
                    )

                    typing_box = None


            # ------------------------------------------------
            # Wait for graph
            # ------------------------------------------------

            while (
                not future.done()
                or (
                    typing_box
                    and (
                        time.time()
                        - started
                        < MIN_TYPING_SECONDS
                    )
                )
            ):

                if typing_box:

                    try:

                        self.page.keyboard.insert_text(
                            "."
                        )

                        self.page.wait_for_timeout(
                            250
                        )

                        self.page.keyboard.press(
                            "Backspace"
                        )

                    except Exception:

                        typing_box = None


                try:

                    self.page.wait_for_timeout(
                        1500
                    )

                except Exception:

                    time.sleep(
                        1.5
                    )


            # ------------------------------------------------
            # Clear typing box
            # ------------------------------------------------

            if typing_box:

                try:

                    self.clear_box(
                        typing_box
                    )

                except Exception:

                    pass


            ticket = future.result()


        log(
            f"  [2/3] process mukammal "
            f"({time.time() - started:.1f}s)"
        )

        return ticket


    # ========================================================
    # PROCESS CURRENT CHAT
    # ========================================================

    def check_open_chat(self):

        name = self.chat_name()


        if DEBUG:

            log(
                f"[debug] chat='{name}' "
                f"allowed={is_allowed(name)}"
            )


        # ----------------------------------------------------
        # No chat name
        # ----------------------------------------------------

        if not name:

            if (
                self.page
                .locator("#main")
                .count()
                > 0
            ):

                self.note(
                    ("noname",),
                    "[warn] open chat ka naam nahi mila"
                )

                self.diagnose(
                    "(no name)"
                )

            return


        # ----------------------------------------------------
        # Allowed chat check
        # ----------------------------------------------------

        if not is_allowed(name):

            self.note(
                ("skip", name),
                f"[skip] '{name}' allowed nahi hai. "
                f"ALLOWED_CHATS={ALLOWED_CHATS}"
            )

            return


        # ----------------------------------------------------
        # Read customer message
        # ----------------------------------------------------

        incoming = (
            self.read_last_incoming()
        )

        if incoming is None:

            if self.last_reason in (
                "no_rows",
                "no_text"
            ):

                self.note(
                    (
                        "reason",
                        name,
                        self.last_reason
                    ),
                    f"[warn] '{name}' mein "
                    f"message text nahi mila: "
                    f"{self.last_reason}"
                )

                self.diagnose(
                    name
                )

            elif DEBUG:

                log(
                    "[debug] "
                    "new incoming message nahi."
                )

            return


        message_id, text = incoming


        # ----------------------------------------------------
        # Duplicate protection
        # ----------------------------------------------------

        if (
            name,
            message_id
        ) in self.handled:

            return

        self.handled.add(
            (
                name,
                message_id
            )
        )


        log(
            f"\n[{name}] {text}"
        )


        # ----------------------------------------------------
        # Closing message
        # ----------------------------------------------------

        if CLOSING_RE.match(
            text
        ):

            log(
                "  [ignore] "
                "closing message."
            )

            return


        # ----------------------------------------------------
        # LangGraph
        # ----------------------------------------------------

        try:

            ticket = self.process_with_typing(
                text
            )


            log(
                f"  intent={ticket.intent} "
                f"conf={ticket.confidence} "
                f"policy={ticket.policy} "
                f"route={ticket.route}"
            )

            log(
                f"  draft: {ticket.draft}"
            )


            # ------------------------------------------------
            # Human review
            # ------------------------------------------------

            if ticket.route != "auto_send":

                log(
                    "  -> HUMAN REVIEW "
                    "(records.csv dekhein)"
                )

                if (
                    HOLD_REPLY
                    and not DRY_RUN
                ):

                    try:

                        self.send_text(
                            HOLD_REPLY
                        )

                        log(
                            "  -> HOLD reply bhej diya"
                        )

                    except Exception as error:

                        log(
                            "  [HOLD reply failed]",
                            error
                        )

                return


            # ------------------------------------------------
            # DRY RUN
            # ------------------------------------------------

            if DRY_RUN:

                log(
                    "  -> DRY_RUN=true "
                    "hai, message nahi bheja."
                )

                return


            # ------------------------------------------------
            # Empty draft
            # ------------------------------------------------

            if not ticket.draft:

                log(
                    "  -> draft empty hai."
                )

                return


            # ------------------------------------------------
            # SEND
            # ------------------------------------------------

            self.send_text(
                ticket.draft
            )

            log(
                "  [3/3] -> SENT"
            )


        except Exception:

            log(
                "  [ERROR] "
                "message process nahi ho saka:"
            )

            traceback.print_exc()


    # ========================================================
    # UNREAD CHAT ROWS
    # ========================================================

    def unread_rows(self):

        selectors = (
            "#pane-side "
            ":is([role='listitem'], "
            "[role='row'], "
            "[role='gridcell'])"
            ":has([aria-label*='unread' i])"
        )

        return self.page.locator(
            selectors
        )


    # ========================================================
    # OPEN NEXT UNREAD
    # ========================================================

    def open_next_unread(self) -> bool:

        try:

            rows = self.unread_rows()

            count = rows.count()

        except Exception:

            return False


        if DEBUG:

            log(
                f"[debug] unread chats: {count}"
            )


        for index in range(
            count
        ):

            row = rows.nth(
                index
            )

            try:

                title = (
                    row
                    .locator(
                        "span[title]"
                    )
                    .first
                    .get_attribute(
                        "title",
                        timeout=1000
                    )
                    or ""
                ).strip()

            except Exception:

                title = ""


            if (
                is_allowed(title)
                or (
                    ALLOW_ALL
                    and not title
                )
            ):

                log(
                    f"[unread] "
                    f"'{title or '?'}' "
                    f"chat khol raha hoon"
                )

                try:

                    row.click(
                        timeout=5000
                    )

                except Exception:

                    try:

                        row.locator(
                            "span[title]"
                        ).first.click(
                            timeout=3000
                        )

                    except Exception:

                        continue


                self.hint = title

                try:

                    self.page.wait_for_timeout(
                        1200
                    )

                except Exception:
                    pass

                return True


            if title:

                self.note(
                    (
                        "unread_skip",
                        title
                    ),
                    f"[skip] '{title}' allowed chat nahi."
                )


        return False


    # ========================================================
    # OPEN ALLOWED CHAT
    # ========================================================

    def open_allowed_chat(self):

        if (
            ALLOW_ALL
            or not AUTO_OPEN_ALLOWED
        ):
            return


        current = self.chat_name()

        if (
            is_allowed(current)
            and len(ALLOWED_CHATS) == 1
        ):
            return


        try:

            spans = self.page.locator(
                "#pane-side span[title]"
            )

            titles = spans.evaluate_all(
                """
                els =>
                els.map(
                    e =>
                    e.getAttribute('title') || ''
                )
                """
            )

        except Exception:

            return


        indexes = [
            index
            for index, title
            in enumerate(titles)
            if is_allowed(title)
        ]


        if not indexes:

            self.note(
                ("nochat",),
                f"[info] allowed chat "
                f"{ALLOWED_CHATS} nahi mili."
            )

            return


        index = indexes[
            self.rr % len(indexes)
        ]

        self.rr += 1


        try:

            spans.nth(
                index
            ).click(
                timeout=5000
            )

            self.hint = titles[index]

            self.page.wait_for_timeout(
                1200
            )

        except Exception as error:

            log(
                "[open chat error]",
                error
            )


    # ========================================================
    # MAIN LOOP
    # ========================================================

    def loop(self):

        counter = 0

        while True:

            try:

                if self.page.is_closed():

                    log(
                        "Browser window band ho gayi. "
                        "Bot band."
                    )

                    return


                # Current chat
                self.check_open_chat()


                # Unread allowed chat
                if self.open_next_unread():

                    self.check_open_chat()


                # Periodically open allowed chat
                elif (
                    counter % SCAN_EVERY == 0
                ):

                    self.open_allowed_chat()

                    self.check_open_chat()


            except KeyboardInterrupt:

                raise


            except Exception as error:

                if (
                    "closed"
                    in str(error).lower()
                ):

                    log(
                        "Browser band ho gaya."
                    )

                    return

                log(
                    "[loop error]",
                    error
                )

                traceback.print_exc()


            counter += 1


            try:

                self.page.wait_for_timeout(
                    POLL_SECONDS * 1000
                )

            except KeyboardInterrupt:

                raise

            except Exception:

                time.sleep(
                    POLL_SECONDS
                )


# ============================================================
# WAIT FOR WHATSAPP
# ============================================================

def wait_for_chat_list(
    page
):

    waited = 0

    while True:

        try:

            if (
                page
                .locator("#pane-side")
                .count()
                > 0
            ):

                return

        except Exception:
            pass


        if waited % 10 == 0:

            try:

                has_qr = (
                    page
                    .locator("canvas")
                    .count()
                    > 0
                )

            except Exception:

                has_qr = False


            log(
                f"  [{waited}s] "
                f"chat list abhi nahi mili | "
                f"QR dikh raha: {has_qr}"
            )


        try:

            page.wait_for_timeout(
                2000
            )

        except Exception:

            time.sleep(
                2
            )


        waited += 2


# ============================================================
# RUN
# ============================================================

def run():

    try:

        from playwright.sync_api import (
            sync_playwright
        )

    except ImportError:

        log(
            "playwright install nahi hai."
        )

        log(
            "Chalayen:"
        )

        log(
            "pip install playwright"
        )

        sys.exit(1)


    log(
        f"Model={MODEL} | "
        f"DRY_RUN={DRY_RUN} | "
        f"Chats={ALLOWED_CHATS}"
    )


    if ALLOW_ALL:

        log(
            "WARNING: "
            "ALLOWED_CHATS=* hai."
        )


    if (
        not ALLOW_ALL
        and any(
            norm(chat)
            == norm(
                "TEST CONTACT NAME"
            )
            for chat in ALLOWED_CHATS
        )
    ):

        log(
            "WARNING: "
            ".env mein ALLOWED_CHATS "
            "placeholder hai."
        )


    # --------------------------------------------------------
    # Load LangGraph
    # --------------------------------------------------------

    try:

        get_graph()

        log(
            "[ok] LangGraph loaded."
        )

    except Exception as error:

        log(
            f"LangGraph load nahi hua: "
            f"{error}"
        )

        traceback.print_exc()

        sys.exit(1)


    # --------------------------------------------------------
    # Ollama
    # --------------------------------------------------------

    ollama_check()


    # --------------------------------------------------------
    # Browser
    # --------------------------------------------------------

    with sync_playwright() as playwright:

        context = None

        try:

            # -----------------------------------------------
            # Try installed Chrome first
            # -----------------------------------------------

            try:

                context = (
                    playwright
                    .chromium
                    .launch_persistent_context(
                        PROFILE_DIR,
                        channel="chrome",
                        headless=False,
                        no_viewport=True,
                        args=[
                            "--disable-blink-features=AutomationControlled",
                        ],
                    )
                )

            except Exception as error:

                log(
                    "[Chrome launch fail]"
                )

                log(
                    str(error).splitlines()[0]
                )

                log(
                    "Playwright Chromium try ho raha hai..."
                )

                context = (
                    playwright
                    .chromium
                    .launch_persistent_context(
                        PROFILE_DIR,
                        headless=False,
                        no_viewport=True,
                    )
                )


            # -----------------------------------------------
            # Page
            # -----------------------------------------------

            if context.pages:

                page = context.pages[0]

            else:

                page = context.new_page()


            # -----------------------------------------------
            # WhatsApp
            # -----------------------------------------------

            try:

                page.goto(
                    "https://web.whatsapp.com",
                    wait_until="domcontentloaded",
                    timeout=120000
                )

            except Exception as error:

                log(
                    "[WhatsApp page load warning]",
                    error
                )


            log(
                "QR scan karein "
                "(agar pehli baar hai)... "
                "chat list ka intezar."
            )


            wait_for_chat_list(
                page
            )


            log(
                "WhatsApp ready. "
                "Bot chal raha hai "
                "(band karne ke liye Ctrl+C)."
            )


            # -----------------------------------------------
            # Start bot
            # -----------------------------------------------

            Bot(page).loop()


        except KeyboardInterrupt:

            log(
                "\nBot band kiya ja raha hai..."
            )


        except Exception:

            log(
                "\n[FATAL ERROR]"
            )

            traceback.print_exc()


        finally:

            if context:

                try:

                    context.close()

                except Exception:

                    pass


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run()