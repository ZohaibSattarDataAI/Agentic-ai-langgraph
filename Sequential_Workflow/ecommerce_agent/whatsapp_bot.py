"""
WhatsApp Web + Ollama + LangGraph bot

Files (same folder): pipeline.py, graph_pipeline.py, whatsapp_bot.py, .env
Run: setup.bat (ek baar) -> .env edit -> run.bat
Pehli baar Chrome khulega -> QR scan karein. Login "wa_profile" folder mein save hota hai.
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

# Windows console par emoji / Urdu naam print hon to crash na ho
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def log(*args):
    print(*args, flush=True)


# ================= CONFIG (.env se aata hai) =================
def load_env(path=None):
    """Chhota .env reader (BOM aur inline # comments handle karta hai)."""
    path = path or os.path.join(BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if v[:1] in ('"', "'") and v[-1:] == v[:1] and len(v) >= 2:
                v = v[1:-1]
            else:
                v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
            os.environ.setdefault(k.strip(), v)


def env_bool(key, default):
    return os.environ.get(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


def env_int(key, default):
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


load_env()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.environ.get("MODEL", "qwen2.5:1.5b")
PROFILE_DIR = os.path.join(BASE_DIR, "wa_profile")

# SAFETY: sirf in chats ko reply hoga (comma se alag). "*" = SAB chats (family/groups bhi!)
ALLOWED_CHATS = [c.strip() for c in os.environ.get("ALLOWED_CHATS", "TEST CONTACT NAME").split(",") if c.strip()]
ALLOW_ALL = "*" in ALLOWED_CHATS
DRY_RUN = env_bool("DRY_RUN", True)                  # true = reply bheja nahi jayega
HOLD_REPLY = os.environ.get("HOLD_REPLY", "").strip()  # human-review wale messages ko chhota jawab
REPHRASE = env_bool("REPHRASE", False)       # true = LLM jawab dobara likhe (chhota model galti karta hai)
TYPING_INDICATOR = env_bool("TYPING_INDICATOR", True)
MIN_TYPING_SECONDS = env_int("MIN_TYPING_SECONDS", 3)
AUTO_OPEN_ALLOWED = env_bool("AUTO_OPEN_ALLOWED", True)  # allowed chat khud kholna (badge ke baghair)
SCAN_EVERY = max(1, env_int("SCAN_EVERY", 4))            # kitne polls baad allowed chat dobara kholna
DEBUG = env_bool("DEBUG", False)
POLL_SECONDS = max(1, env_int("POLL_SECONDS", 3))
MAX_TRAILING = 5                                          # customer ke aakhri kitne messages jorne hain
# "ok", "thanks", "shukriya", 👍 jaise band karne wale messages par reply nahi jata
CLOSING_RE = re.compile(
    r"^\s*(ok(ay)?|thanks?( you)?|thank u|shukr(iya|ia)|jazak\w*|(theek|thik|acha|achha|accha)( hai| ha)?|"
    r"👍|🙏|❤️|😊)[\s.!]*$", re.I)
INTENTS = {"order", "return", "refund", "complaint", "question"}
# =============================================================


# ---------- Ollama ----------
def ollama(messages, json_mode=False, temperature=0.2, num_predict=200):
    payload = {"model": MODEL, "messages": messages, "stream": False, "keep_alive": "30m",
               "options": {"temperature": temperature, "num_ctx": 2048, "num_predict": num_predict}}
    if json_mode:
        payload["format"] = "json"
    req = urllib.request.Request(OLLAMA_URL, json.dumps(payload).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"]


def ollama_check():
    base = OLLAMA_URL.split("/api/")[0]
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=5) as r:
            names = [m.get("name", "") for m in json.loads(r.read().decode("utf-8")).get("models", [])]
        if MODEL in names:
            log(f"[ok] Ollama chal raha hai, model '{MODEL}' maujood hai")
            log("     model memory mein load ho raha hai (pehli baar 30-60 second lag sakte hain)...")
            t0 = time.time()
            try:
                ollama([{"role": "user", "content": "hi"}], num_predict=1)
                log(f"     model tayyar ({time.time() - t0:.0f}s)")
            except Exception as e:
                log(f"     [warm-up fail] {e}")
        else:
            log(f"[WARNING] Ollama chal raha hai lekin model '{MODEL}' nahi mila. Maujood: {names}")
            log(f"          Chalayen: ollama pull {MODEL}")
    except Exception as e:
        log(f"[WARNING] Ollama se rabta nahi ({e}). Naye CMD mein 'ollama serve' chalayen.")
        log("          Jab tak Ollama na chale, sab messages insaan review mein jayenge.")


def llm_analyze(text: str) -> dict:
    system = (
        "Tum ek kapdon ke online store ke customer message classifier ho. "
        "Message Roman Urdu, Urdu ya English mix ho sakta hai. "
        'Sirf is format mein JSON do: {"intent":"order|return|refund|complaint|question|other",'
        '"angry":true,"order_no":null,"days_since_delivery":null,"item_unused":null}. '
        "angry tab true jab customer ghussay/badtameezi mein ho. Jo maloom na ho wo null rakho."
    )
    try:
        out = json.loads(ollama([{"role": "system", "content": system},
                                 {"role": "user", "content": text}],
                                json_mode=True, temperature=0, num_predict=120))
        return out if isinstance(out, dict) else {}
    except Exception as e:
        log("  [LLM analyze fail]", e)
        return {}


def llm_rephrase(template: str, angry: bool) -> str:
    if not REPHRASE:
        return template                      # approved template seedha (tez aur mehfooz)
    system = (
        "Tum ek online kapdon ke store ki customer support ho. Neeche diya gaya approved jawab "
        "Roman Urdu mein thora natural aur dosti-bhare, polite andaaz mein dobara likho. "
        "SAKHT QAWAID: matlab bilkul wohi rahe; koi naya wada, discount, refund, muddat ya number "
        "shamil NA karo; 3 jumlon se zyada nahi; sirf jawab likho, koi wazahat nahi. "
        + ("Customer naraz hai, is liye lehja bohat naram aur maazrat wala rakho." if angry else "")
    )
    try:
        out = ollama([{"role": "system", "content": system},
                      {"role": "user", "content": template}], temperature=0.3, num_predict=160).strip()
        return out or template
    except Exception as e:
        log("  [LLM rephrase fail]", e)
        return template


# ---------- LangGraph workflow ----------
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        from graph_pipeline import build_graph
        _GRAPH = build_graph(llm_analyze, llm_rephrase, INTENTS)
    return _GRAPH


def handle_message(channel: str, text: str) -> P.Ticket:
    return get_graph().invoke({"t": P.Ticket(channel=channel, raw=text)})["t"]


# ---------- Chat naam matching ----------
def norm(s: str) -> str:
    """'ZohaibSattar Data. AI' == 'zohaibsattar data.ai' (space, dot, case ka farq nahi)."""
    return re.sub(r"[^a-z0-9\u0600-\u06ff]", "", (s or "").lower())


def is_allowed(name: str) -> bool:
    if ALLOW_ALL:
        return bool(name)
    n = norm(name)
    return bool(n) and any(n == norm(a) for a in ALLOWED_CHATS)


# ---------- WhatsApp Web ----------
class Bot:
    def __init__(self, page):
        self.page = page
        self.handled = set()
        self.noted = set()
        self.rr = 0
        self.hint = ""           # aakhri baar jo chat humne khud kholi uska naam
        self.last_reason = ""
        self.sent_norms = set()   # bot ke bheje hue texts (apne jawab ko dobara na parhe)
        self.send_times = []      # loop se bachao

    def note(self, key, msg):
        """Same message baar baar print na ho."""
        if key not in self.noted:
            self.noted.add(key)
            log(msg)

    # ----- padhna -----
    def chat_name(self) -> str:
        for sel in ("#main header span[dir='auto']", "#main header span[title]", "#main header [title]"):
            try:
                loc = self.page.locator(sel).first
                if loc.count() == 0:
                    continue
                txt = (loc.inner_text(timeout=1000) or "").strip()
                if not txt:
                    txt = (loc.get_attribute("title", timeout=1000) or "").strip()
                if txt:
                    return txt
            except Exception:
                continue
        return self.hint

    @staticmethod
    def message_text(el) -> str:
        try:
            parts = el.locator("span.selectable-text").all_inner_texts()
            uniq = []
            for x in (p.strip() for p in parts):
                if x and (not uniq or uniq[-1] != x):
                    uniq.append(x)
            if uniq:
                return " ".join(uniq)
            raw = (el.inner_text(timeout=1000) or "").strip()
            if re.fullmatch(r"[\d:\s.apmAPM]*", raw):   # sirf waqt (image/sticker wagera)
                return ""
            raw = re.sub(r"\s*\d{1,2}:\d{2}\s*([AaPp]\.?[Mm]\.?)?\s*$", "", raw).strip()
            return raw
        except Exception:
            return ""

    def message_rows(self):
        for sel in ("#main [data-id^='false_'], #main [data-id^='true_']",   # purana format
                    "#main [data-id]"):                                      # naya format (hash ids)
            loc = self.page.locator(sel)
            if loc.count() > 0:
                return loc
        return None

    def is_outgoing(self, el, did: str) -> bool:
        """Kya ye message hamara (bheja hua) hai?"""
        if did.startswith("true_"):
            return True
        if did.startswith("false_"):
            return False
        try:   # bheje hue messages par tick/clock icon hota hai
            if el.locator("[data-icon^='msg-'], [data-icon^='status-']").count() > 0:
                return True
        except Exception:
            pass
        try:   # bubble daayen taraf ho to outgoing
            box = el.locator("span.selectable-text").first.bounding_box(timeout=1000)
            cont = self.page.locator("#main").first.bounding_box(timeout=1000)
            if box and cont:
                left_gap = box["x"] - cont["x"]
                right_gap = (cont["x"] + cont["width"]) - (box["x"] + box["width"])
                if right_gap + 25 < left_gap:
                    return True
        except Exception:
            pass
        return False

    def read_last_incoming(self):
        """(key, text) ya None. Wajah self.last_reason mein: no_rows / own_last / no_text."""
        rows = self.message_rows()
        if rows is None:
            self.last_reason = "no_rows"
            return None

        n = rows.count()
        texts, seen, last_id = [], set(), None
        for i in range(n - 1, max(-1, n - 30), -1):
            el = rows.nth(i)
            try:
                did = el.get_attribute("data-id", timeout=1000) or ""
            except Exception:
                continue
            if not did or did in seen:
                continue
            seen.add(did)
            legacy = did.startswith(("false_", "true_"))
            try:
                has_text_span = el.locator("span.selectable-text").count() > 0
            except Exception:
                has_text_span = False
            if not legacy and not has_text_span:
                continue                                  # date/system/media row, message nahi
            tx = self.message_text(el)
            if self.is_outgoing(el, did) or (tx and norm(tx) in self.sent_norms):
                break                                     # yahan hamara reply hai, purane ho chuke
            if last_id is None:
                last_id = did
            if tx and tx not in texts:                    # bilkul same message dobara na jorey
                texts.append(tx)
            if len(texts) >= MAX_TRAILING:
                break
        if last_id is None:
            self.last_reason = "own_last"
            return None
        if not texts:
            self.last_reason = "no_text"
            return None
        texts.reverse()
        return last_id, " ".join(texts)

    def diagnose(self, name):
        """WhatsApp ka structure print karta hai (ek chat ke liye ek baar)."""
        if ("diag", name) in self.noted:
            return
        self.noted.add(("diag", name))
        pg = self.page

        def cnt(sel):
            try:
                return pg.locator(sel).count()
            except Exception as e:
                return f"err({str(e)[:40]})"
        log("  [diagnose] chat khuli hai lekin message parha nahi ja saka. WhatsApp ka structure:")
        for sel in ("#main", "#main header", "[data-id]", "[data-id^='false_']", "[data-id^='true_']",
                    "#main [data-icon^='msg-']",
                    "div.message-in", "div.message-out", "span.selectable-text",
                    "footer", "footer div[contenteditable='true']"):
            log(f"     {sel}: {cnt(sel)}")
        try:
            ids = pg.locator("[data-id]").evaluate_all("els => els.slice(-4).map(e => e.getAttribute('data-id'))")
            log("     aakhri data-id:", ids)
        except Exception:
            pass
        try:
            icons = pg.locator("#main [data-icon]").evaluate_all("els => els.slice(-8).map(e => e.getAttribute('data-icon'))")
            log("     aakhri data-icon:", icons)
        except Exception:
            pass
        try:
            log("     header text:", repr((pg.locator("#main header").first.inner_text(timeout=1000) or "")[:80]))
        except Exception:
            pass

    # ----- bhejna -----
    def find_box(self):
        for sel in ("#main footer div[contenteditable='true']",
                    "footer div[contenteditable='true']",
                    "#main div[contenteditable='true'][role='textbox']",
                    "#main div[contenteditable='true']"):
            loc = self.page.locator(sel)
            if loc.count() > 0:
                return loc.last
        raise RuntimeError("message likhne wala dabba nahi mila")

    def clear_box(self, box):
        box.click()
        self.page.keyboard.press("Control+A")
        self.page.keyboard.press("Backspace")

    def count_rows(self) -> int:
        rows = self.message_rows()
        return rows.count() if rows is not None else 0

    def typed_text(self) -> str:
        try:
            return (self.find_box().inner_text(timeout=1000) or "").strip()
        except Exception:
            return ""

    def wait_sent(self, before: int, seconds: float = 5) -> bool:
        """Chat mein naya message row aa jaye to bhej diya gaya."""
        end = time.time() + seconds
        while time.time() < end:
            if self.count_rows() > before:
                return True
            self.page.wait_for_timeout(500)
        return False

    def send_diagnose(self):
        try:
            info = self.page.locator("div[contenteditable='true']").evaluate_all(
                "els => els.map(e => [e.getAttribute('aria-label'), e.getAttribute('data-tab'),"
                " !!e.closest('footer'), (e.innerText || '').slice(0, 30)])")
            log("     [send-diagnose] contenteditable dabbe (aria-label, data-tab, footer mein?, text):", info)
        except Exception:
            pass

    def send_text(self, msg: str):
        """Bhejta hai aur VERIFY karta hai ke chat mein message asal mein aya."""
        now = time.time()
        self.send_times = [t for t in self.send_times if now - t < 60]
        if len(self.send_times) >= 8:
            raise RuntimeError("safety: 60 second mein 8 se zyada replies, bot ne khud ko roka (loop ka khatra)")
        self.send_times.append(now)
        self.sent_norms.add(norm(msg))

        before = self.count_rows()
        for attempt, method in enumerate(("insert_text", "type"), start=1):
            box = self.find_box()
            self.clear_box(box)
            if method == "insert_text":
                self.page.keyboard.insert_text(msg)
            else:
                self.page.keyboard.type(msg, delay=15)      # asli keystrokes
            self.page.wait_for_timeout(700)
            if not self.typed_text():
                log(f"  [send] koshish {attempt} ({method}): text dabbe mein gaya hi nahi (focus ka masla?)")
                continue
            self.page.keyboard.press("Enter")
            if self.wait_sent(before, 5):
                log(f"  [send] verified ({method}): message chat mein aa gaya")
                return
            btn = self.page.locator("button[aria-label='Send'], [aria-label='Send'], span[data-icon='send']")
            if btn.count() > 0:
                btn.last.click()
                if self.wait_sent(before, 5):
                    log(f"  [send] verified ({method} + Send button)")
                    return
            log(f"  [send] koshish {attempt} ({method}): Enter ke baad bhi message chat mein nahi aya")
        self.send_diagnose()
        raise RuntimeError("reply bheja nahi ja saka: chat mein naya message nahi aya (upar [send] lines dekhein)")

    def process_with_typing(self, text: str) -> P.Ticket:
        """LLM background thread mein; itni der customer ko 'typing...' dikhta hai.
        (Playwright sirf main thread se chalta hai; thread mein sirf LLM/graph.)"""
        log("  [1/3] LLM se process ho raha hai...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(handle_message, "whatsapp", text)
            box = None
            if TYPING_INDICATOR and not DRY_RUN:
                try:
                    box = self.find_box()
                    box.click()
                except Exception as e:
                    log("  [typing indicator nahi chala]", e)
                    box = None
            while not fut.done() or (box and time.time() - t0 < MIN_TYPING_SECONDS):
                if box:
                    try:
                        self.page.keyboard.insert_text(".")     # keystroke = 'typing...'
                        self.page.wait_for_timeout(300)
                        self.page.keyboard.press("Backspace")
                    except Exception:
                        box = None
                self.page.wait_for_timeout(2000)
            if box:
                try:
                    self.clear_box(box)
                except Exception:
                    pass
            ticket = fut.result()
        log(f"  [2/3] process mukammal ({time.time() - t0:.1f}s)")
        return ticket

    # ----- ek chat ka kaam -----
    def check_open_chat(self):
        name = self.chat_name()
        if DEBUG:
            log(f"[debug] khuli chat='{name}' allowed={is_allowed(name)}")
        if not name:
            if self.page.locator("#main").count() > 0:      # chat khuli hai magar naam nahi mila
                self.note(("noname",), "[warn] khuli chat ka naam header se nahi mila (selector ka masla)")
                self.diagnose("(no name)")
            return                                            # warna koi chat khuli hi nahi
        if not is_allowed(name):
            self.note(("skip", name),
                      f"[skip] khuli chat '{name}' ALLOWED_CHATS {ALLOWED_CHATS} mein nahi. "
                      f".env mein ye naam likhein agar is par reply chahiye.")
            return
        got = self.read_last_incoming()
        if got is None:
            if self.last_reason in ("no_rows", "no_text"):
                self.note(("reason", name, self.last_reason),
                          f"[warn] chat '{name}' mein message ka text nahi mila ({self.last_reason})")
                self.diagnose(name)
            elif DEBUG:
                log("[debug] aakhri message hamara apna hai, naya incoming nahi")
            return
        key, text = got
        if (name, key) in self.handled:
            return
        self.handled.add((name, key))
        log(f"\n[{name}] {text}")
        if CLOSING_RE.match(text):
            log("  [ignore] shukriya/ok jaisa band karne wala message, reply nahi diya")
            return

        try:
            t = self.process_with_typing(text)
            log(f"  intent={t.intent} conf={t.confidence} policy={t.policy} route={t.route}")
            log(f"  draft: {t.draft}")
            if t.route != "auto_send":
                log("  -> HUMAN REVIEW (records.csv dekhein)")
                if HOLD_REPLY and not DRY_RUN:
                    self.send_text(HOLD_REPLY)
                    log("  -> HOLD reply bheja")
                return
            if DRY_RUN:
                log("  -> DRY_RUN=true hai, bheja nahi. .env mein DRY_RUN=false karein")
                return
            if not t.draft:
                log("  -> draft khali hai, kuch nahi bheja")
                return
            self.send_text(t.draft)
            log("  [3/3] -> SENT")
        except Exception:
            log("  [ERROR] is message par kaam nahi ho saka:")
            traceback.print_exc()

    # ----- chats kholna -----
    def unread_rows(self):
        return self.page.locator(
            "#pane-side :is([role='listitem'], [role='row'], [role='gridcell']):has([aria-label*='unread' i])")

    def open_next_unread(self) -> bool:
        """Sirf ALLOWED chats kholta hai (baaqi chats 'read' nahi hoti)."""
        rows = self.unread_rows()
        n = rows.count()
        if DEBUG:
            log(f"[debug] unread chats: {n}")
        for i in range(n):
            row = rows.nth(i)
            try:
                title = (row.locator("span[title]").first.get_attribute("title", timeout=1000) or "").strip()
            except Exception:
                title = ""
            if is_allowed(title) or (ALLOW_ALL and not title):
                log(f"[unread] '{title or '?'}' chat khol raha hoon")
                row.click()
                self.hint = title
                self.page.wait_for_timeout(1500)
                return True
            if title:
                self.note(("unread_skip", title), f"[skip] '{title}' mein unread message hai, magar ye allowed chat nahi")
        return False

    def open_allowed_chat(self):
        """Badge ke baghair bhi: allowed chat chat-list se khud kholna."""
        if ALLOW_ALL or not AUTO_OPEN_ALLOWED:
            return
        current = self.chat_name()
        if is_allowed(current) and len(ALLOWED_CHATS) == 1:
            return
        spans = self.page.locator("#pane-side span[title]")
        try:
            titles = spans.evaluate_all("els => els.map(e => e.getAttribute('title') || '')")
        except Exception:
            return
        idx = [i for i, t in enumerate(titles) if is_allowed(t)]
        if not idx:
            self.note(("nochat",),
                      f"[info] chat list mein {ALLOWED_CHATS} wali chat nahi mili. "
                      f"List mein ye naam hain: {[t for t in titles if t][:10]}")
            return
        i = idx[self.rr % len(idx)]
        self.rr += 1
        spans.nth(i).click()
        self.hint = titles[i]
        self.page.wait_for_timeout(1500)

    # ----- main loop -----
    def loop(self):
        n = 0
        while True:
            try:
                if self.page.is_closed():
                    log("Browser window band ho gayi. Bot band.")
                    return
                self.check_open_chat()
                if self.open_next_unread():
                    self.check_open_chat()
                elif n % SCAN_EVERY == 0:
                    self.open_allowed_chat()
                    self.check_open_chat()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if "closed" in str(e).lower():
                    log("Browser band ho gaya. Bot band.")
                    return
                log("[loop error]")
                traceback.print_exc()
            n += 1
            try:
                self.page.wait_for_timeout(POLL_SECONDS * 1000)
            except KeyboardInterrupt:
                raise
            except Exception:
                time.sleep(POLL_SECONDS)


def wait_for_chat_list(page):
    waited = 0
    while page.locator("#pane-side").count() == 0:
        if waited % 10 == 0:
            has_qr = page.locator("canvas").count() > 0
            log(f"  [{waited}s] chat list abhi nahi mili | QR dikh raha: {has_qr}")
        page.wait_for_timeout(2000)
        waited += 2


def run():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log("playwright install nahi hai. Chalayen: pip install playwright")
        sys.exit(1)

    log(f"Model={MODEL} | DRY_RUN={DRY_RUN} | Chats={ALLOWED_CHATS}")
    if ALLOW_ALL:
        log("WARNING: ALLOWED_CHATS=* hai, sab chats ko reply hoga!")
    if not ALLOW_ALL and any(norm(a) == norm("TEST CONTACT NAME") for a in ALLOWED_CHATS):
        log("WARNING: .env mein ALLOWED_CHATS abhi placeholder hai, asli chat ka naam likhein.")

    try:
        get_graph()
    except Exception as e:
        log(f"LangGraph load nahi hua: {e}\nChalayen: pip install langgraph")
        sys.exit(1)
    ollama_check()

    with sync_playwright() as p:
        try:
            ctx = p.chromium.launch_persistent_context(
                PROFILE_DIR, channel="chrome", headless=False, no_viewport=True)
        except Exception as e:
            log(f"[Chrome nahi khula: {str(e).splitlines()[0]}]")
            log("Agar pehla bot abhi chal raha hai to usay band karein. Ab Playwright ka Chromium try kar raha hoon...")
            ctx = p.chromium.launch_persistent_context(PROFILE_DIR, headless=False, no_viewport=True)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://web.whatsapp.com", wait_until="domcontentloaded", timeout=120000)
        log("QR scan karein (agar pehli baar hai)... chat list ka intezar.")
        wait_for_chat_list(page)
        log("WhatsApp ready. Bot chal raha hai (band karne ke liye Ctrl+C).")
        try:
            Bot(page).loop()
        except KeyboardInterrupt:
            log("\nBot band kiya ja raha hai...")
        finally:
            try:
                ctx.close()
            except Exception:
                pass


if __name__ == "__main__":
    run()