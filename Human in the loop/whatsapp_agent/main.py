import re
import time
from typing import TypedDict, Optional

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# ============================================================
# CONFIGURATION
# ============================================================

WHATSAPP_URL = "https://web.whatsapp.com/"
SESSION_DIR = "whatsapp_session"

OLLAMA_MODEL = "qwen2.5:1.5b"

# How often WhatsApp is checked for new unread chats
POLL_INTERVAL = 0.5


# ============================================================
# LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):
    sender_name: str
    message: str

    english_reply: str
    roman_reply: str

    selected_reply: str
    language: str

    approved: bool


# ============================================================
# OLLAMA
# ============================================================

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.3,
)


# ============================================================
# AI REPLY GENERATION
# ============================================================

def generate_reply(state: AgentState):

    sender = state.get("sender_name", "Unknown")
    message = state.get("message", "").strip()

    prompt = f"""
You are a WhatsApp personal assistant.

Sender:
{sender}

Incoming WhatsApp message:
{message}

Generate TWO short and natural reply options.

Requirements:

1. ENGLISH:
- Natural WhatsApp English
- Friendly and human
- Short
- Directly respond to the incoming message

2. ROMAN:
- Roman Urdu / Roman English
- Use ONLY English alphabet
- Do NOT use Urdu or Arabic script
- Natural Pakistani WhatsApp style
- Short and friendly

Return EXACTLY this format:

ENGLISH:
<reply>

ROMAN:
<reply>
"""

    try:
        response = llm.invoke(prompt)
        text = (response.content or "").strip()
    except Exception as e:
        print("\nERROR calling Ollama:")
        print(e)
        text = ""

    english_reply = ""
    roman_reply = ""

    if text:

        english_match = re.search(
            r"ENGLISH:\s*(.*?)(?=\n\s*ROMAN:|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        roman_match = re.search(
            r"ROMAN:\s*(.*)",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if english_match:
            english_reply = english_match.group(1).strip()

        if roman_match:
            roman_reply = roman_match.group(1).strip()

    # Fallback if model format is imperfect or call failed
    if not english_reply:
        english_reply = text if text else "Okay, got it."

    if not roman_reply:
        roman_reply = english_reply

    print("\n" + "=" * 70)
    print("                    AI REPLY OPTIONS")
    print("=" * 70)

    print("\n[1] ENGLISH")
    print(english_reply)

    print("\n[2] ROMAN")
    print(roman_reply)

    print("=" * 70)

    return {
        "english_reply": english_reply,
        "roman_reply": roman_reply,
    }


# ============================================================
# HUMAN APPROVAL
# ============================================================

def human_approval(state: AgentState):

    english = state.get("english_reply", "")
    roman = state.get("roman_reply", "")

    print("\nChoose reply:")
    print("1 = English")
    print("2 = Roman")

    while True:
        choice = input("\nYour choice (1/2): ").strip()

        if choice == "1":
            selected_reply = english
            language = "English"
            break

        if choice == "2":
            selected_reply = roman
            language = "Roman"
            break

        print("Please enter only 1 or 2.")

    print("\n" + "-" * 70)
    print("SELECTED REPLY")
    print("-" * 70)
    print(selected_reply)
    print("-" * 70)

    while True:
        approval = input("\nSend this message? (YES/NO): ").strip().lower()

        if approval in ("yes", "y"):
            approved = True
            break

        if approval in ("no", "n"):
            approved = False
            break

        print("Please enter YES or NO.")

    return {
        "selected_reply": selected_reply,
        "language": language,
        "approved": approved,
    }


# ============================================================
# FINISH NODE
# ============================================================

def finish(state: AgentState):

    if state.get("approved"):
        print("\nApproval received.")
        print("Sending message to WhatsApp...")
    else:
        print("\nMessage cancelled by user.")

    return {}


# ============================================================
# LANGGRAPH
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("generate_reply", generate_reply)
workflow.add_node("human_approval", human_approval)
workflow.add_node("finish", finish)

workflow.add_edge(START, "generate_reply")
workflow.add_edge("generate_reply", "human_approval")
workflow.add_edge("human_approval", "finish")
workflow.add_edge("finish", END)

agent = workflow.compile()


# ============================================================
# PLAYWRIGHT HELPERS
# ============================================================

def visible_locator(page, selectors):

    for selector in selectors:

        try:
            locator = page.locator(selector)
            count = locator.count()

            for i in range(count):
                item = locator.nth(i)

                try:
                    if item.is_visible():
                        return item
                except Exception:
                    continue

        except Exception:
            continue

    return None


# ============================================================
# CHAT LIST CONTAINER
# ============================================================
# CRITICAL FIX:
# All chat-row queries MUST be scoped inside the actual chat-list
# pane. Without this, generic role selectors like div[role="listitem"]
# also match sidebar/nav elements (e.g. your own profile button),
# which can be misdetected as an "unread chat" with no real messages.

def get_chat_list_container(page):

    selectors = [
        'div[aria-label="Chat list"]',
        '#pane-side',
    ]

    for selector in selectors:

        try:
            loc = page.locator(selector)

            if loc.count() > 0:
                return loc.first

        except Exception:
            continue

    return None


# ============================================================
# GET CHAT ROWS (scoped to chat list only)
# ============================================================

def get_chat_rows(page):

    container = get_chat_list_container(page)

    if container is None:
        return None

    selectors = [
        'div[role="listitem"]',
        'div[role="row"]',
        '[data-testid="cell-frame-container"]',
    ]

    for selector in selectors:

        try:
            rows = container.locator(selector)

            if rows.count() > 0:
                return rows

        except Exception:
            continue

    return None


# ============================================================
# VALIDATE THAT A ROW IS AN ACTUAL CHAT
# ============================================================
# WhatsApp's chat-list container can also contain non-chat elements
# (headers, filter bars, your own profile row, etc.) that still match
# role="listitem". A REAL chat row always has a last-message preview
# and/or a timestamp next to the name. We require at least one of
# those before treating a row as a real chat, which is what prevents
# your own profile/name from being misdetected as an "unread chat".

def is_probable_chat_row(row):

    preview_selectors = [
        '[data-testid="cell-frame-secondary"]',
        '[data-testid="last-msg-status"]',
        'span[title][dir="ltr"]',
    ]

    for selector in preview_selectors:
        try:
            if row.locator(selector).count() > 0:
                return True
        except Exception:
            continue

    # Fallback: look for a timestamp-shaped string anywhere in the row
    # e.g. "10:42 PM", "Yesterday", "Mon", "12/09/2026"
    try:
        text = row.inner_text(timeout=1000)

        time_pattern = re.compile(
            r"\b\d{1,2}:\d{2}\s*(AM|PM)?\b"
            r"|Yesterday"
            r"|\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b"
            r"|\d{1,2}/\d{1,2}/\d{2,4}",
            re.IGNORECASE,
        )

        if time_pattern.search(text):
            return True

    except Exception:
        pass

    return False


# ============================================================
# CHECK UNREAD CHAT
# ============================================================

def has_unread(row):

    unread_selectors = [
        '[aria-label*="unread" i]',
        '[aria-label*="new message" i]',
        '[data-testid*="unread" i]',
        '[data-testid*="unread-count" i]',
        'span[aria-label$="unread messages"]',
        'span[aria-label$="unread message"]',
    ]

    for selector in unread_selectors:

        try:
            if row.locator(selector).count() > 0:
                return True
        except Exception:
            pass

    # WhatsApp sometimes displays the unread count as a small badge
    # containing only digits. This fallback is scoped to chat rows
    # only (via get_chat_rows), so it is much safer than checking
    # the whole page.
    try:

        text = row.inner_text(timeout=1000)

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for line in lines:

            if re.fullmatch(r"\d{1,3}", line):
                return True

    except Exception:
        pass

    return False


# ============================================================
# FIND UNREAD CHAT
# ============================================================

def find_unread_chat(page):

    rows = get_chat_rows(page)

    if rows is None:
        print("DEBUG: chat list container/rows not found at all.")
        return None

    try:
        count = rows.count()
        print(f"DEBUG: {count} candidate row(s) found in chat list.")

        for i in range(count):

            row = rows.nth(i)

            try:

                if not row.is_visible():
                    continue

                if not is_probable_chat_row(row):
                    # Skip non-chat elements (profile row, headers, etc.)
                    continue

                if has_unread(row):
                    return row

            except Exception:
                continue

    except Exception:
        pass

    return None


# ============================================================
# GET CHAT NAME FROM ROW
# ============================================================

def get_row_name(row):

    selectors = [
        '[data-testid="cell-frame-title"]',
        'span[title]',
        'span[dir="auto"]',
        '[title]',
    ]

    for selector in selectors:

        try:
            locator = row.locator(selector)
            count = locator.count()

            for i in range(count):

                item = locator.nth(i)

                try:

                    if not item.is_visible():
                        continue

                    text = (
                        item.inner_text(timeout=500).strip()
                        or item.get_attribute("title")
                        or ""
                    )

                    if text and len(text) < 200:
                        return text

                except Exception:
                    continue

        except Exception:
            continue

    try:
        text = row.inner_text(timeout=1000)

        for line in text.splitlines():
            line = line.strip()

            if line:
                return line

    except Exception:
        pass

    return "Unknown"


# ============================================================
# OPEN CHAT
# ============================================================

def open_chat(page, row):

    try:
        row.scroll_into_view_if_needed(timeout=3000)
    except Exception:
        pass

    try:
        row.click(timeout=3000)
        time.sleep(1)
        return True

    except Exception:

        try:
            row.click(force=True, timeout=3000)
            time.sleep(1)
            return True

        except Exception:
            return False


# ============================================================
# CURRENT CHAT NAME
# ============================================================

def get_current_chat_name(page):

    selectors = [
        '[data-testid="conversation-info-header-chat-title"]',
        'header span[title]',
        'header span[dir="auto"]',
    ]

    for selector in selectors:

        try:
            locator = page.locator(selector)
            count = locator.count()

            for i in range(count):

                item = locator.nth(i)

                try:

                    if item.is_visible():

                        text = (
                            item.inner_text(timeout=500).strip()
                            or item.get_attribute("title")
                            or ""
                        )

                        if text and len(text) < 200:
                            return text

                except Exception:
                    continue

        except Exception:
            continue

    return "Unknown"


# ============================================================
# MESSAGE CLEANER
# ============================================================

def clean_message(text):

    if not text:
        return ""

    text = text.replace("\u200e", "")
    text = text.replace("\u200f", "")
    text = text.replace("\u202a", "")
    text = text.replace("\u202b", "")
    text = text.replace("\u202c", "")
    text = text.strip()

    return text


# ============================================================
# STRIP TIMESTAMP / STATUS CLUTTER FROM MESSAGE TEXT
# ============================================================
# WhatsApp bubbles often render as a single text blob that includes
# the message plus a trailing time (and sometimes read-receipt glyphs)
# on their own line, e.g.:
#   "Hey, are we still on for tomorrow?\n10:42 PM"
# This trims that trailing metadata without touching the real message.

_TRAILING_META_PATTERN = re.compile(
    r"^\s*\d{1,2}:\d{2}\s*(AM|PM)?\s*[\u2713\u2714\u2705]*\s*$",
    re.IGNORECASE,
)


def strip_trailing_metadata(text):

    if not text:
        return text

    lines = [line for line in text.split("\n") if line.strip() != ""]

    while lines and _TRAILING_META_PATTERN.match(lines[-1]):
        lines.pop()

    return "\n".join(lines).strip()


# ============================================================
# EXTRACT TEXT FROM A SINGLE MESSAGE BUBBLE
# ============================================================

def extract_bubble_text(bubble):

    # Prefer the actual selectable text spans if present — this avoids
    # picking up icons/labels rendered as sibling elements.
    text_selectors = [
        'span.selectable-text',
        '[data-testid="selectable-text"]',
        'span.copyable-text',
        'span[dir="ltr"]',
        'span[dir="auto"]',
    ]

    for selector in text_selectors:

        try:
            loc = bubble.locator(selector)
            count = loc.count()

            collected = []

            for i in range(count):
                item = loc.nth(i)

                try:
                    if not item.is_visible():
                        continue

                    piece = item.inner_text(timeout=300).strip()

                    if piece:
                        collected.append(piece)

                except Exception:
                    continue

            if collected:
                return clean_message(" ".join(collected))

        except Exception:
            continue

    # Fallback: whole-bubble text, with timestamp/status trimmed off
    try:
        raw = bubble.inner_text(timeout=500)
        return clean_message(strip_trailing_metadata(raw))
    except Exception:
        return ""


# ============================================================
# GET LATEST INCOMING MESSAGE
# ============================================================
# We only ever call this right after WhatsApp flagged the chat as
# unread, which means the newest message in the conversation is,
# by definition, the one that just arrived from the other side.
# So rather than depending on exact class names for "incoming"
# bubbles (which WhatsApp changes periodically and breaks selectors),
# we simply take the LAST message row in the conversation.

def get_latest_incoming_message(page):

    main_panel = page.locator("#main")

    # --------------------------------------------------------
    # METHOD 1
    # Every message row WhatsApp renders carries a data-id attribute.
    # This has stayed stable across WhatsApp Web redesigns even when
    # CSS class names changed, so it's the most reliable anchor.
    # --------------------------------------------------------

    try:
        rows = main_panel.locator('div[data-id]')
        count = rows.count()

        for i in range(count - 1, -1, -1):

            row = rows.nth(i)

            try:
                if not row.is_visible():
                    continue

                text = extract_bubble_text(row)

                if text and len(text) <= 5000:
                    return text

            except Exception:
                continue

    except Exception:
        pass

    # --------------------------------------------------------
    # METHOD 2
    # Fallback: any element commonly used for message text/copy,
    # take the last visible one in the panel.
    # --------------------------------------------------------

    fallback_selectors = [
        'span.selectable-text',
        '[data-testid="selectable-text"]',
        'span.copyable-text',
    ]

    for selector in fallback_selectors:

        try:
            items = main_panel.locator(selector)
            count = items.count()

            for i in range(count - 1, -1, -1):

                item = items.nth(i)

                try:
                    if not item.is_visible():
                        continue

                    text = clean_message(item.inner_text(timeout=300))

                    if text and len(text) <= 5000:
                        return text

                except Exception:
                    continue

        except Exception:
            continue

    # --------------------------------------------------------
    # METHOD 3
    # Last resort: any element with a role="row" inside #main,
    # WhatsApp's most generic message-row marker.
    # --------------------------------------------------------

    try:
        rows = main_panel.locator('div[role="row"]')
        count = rows.count()

        for i in range(count - 1, -1, -1):

            row = rows.nth(i)

            try:
                if not row.is_visible():
                    continue

                text = extract_bubble_text(row)

                if text and len(text) <= 5000:
                    return text

            except Exception:
                continue

    except Exception:
        pass

    return None


# ============================================================
# MESSAGE BOX
# ============================================================

def get_message_box(page):

    selectors = [
        'footer div[contenteditable="true"]',
        'footer [contenteditable="true"]',
        'div[contenteditable="true"][data-tab="10"]',
        'div[contenteditable="true"][aria-label*="message" i]',
        'div[contenteditable="true"][title*="message" i]',
        'div[contenteditable="true"][data-placeholder*="message" i]',
    ]

    return visible_locator(page, selectors)


# ============================================================
# SEND WHATSAPP MESSAGE
# ============================================================

def send_message(page, message):

    box = get_message_box(page)

    if box is None:
        print("\nERROR: WhatsApp message box not found.")
        return False

    try:

        box.click()

        # Fill is not always supported correctly by WhatsApp's editor,
        # so use keyboard insertion.
        page.keyboard.insert_text(message)

        time.sleep(0.3)

        page.keyboard.press("Enter")

        time.sleep(1)

        print("\nMessage sent successfully.")
        return True

    except Exception as e:

        print("\nERROR while sending message:")
        print(e)

        return False


# ============================================================
# DEBUG SNAPSHOT
# ============================================================

# (Debug file/screenshot writing removed by request — detection is now
# handled directly, see get_latest_incoming_message below.)


# ============================================================
# PROCESS ONE MESSAGE
# ============================================================

def process_message(page, sender_name, incoming_message):

    print("\n" + "=" * 70)
    print("                    NEW WHATSAPP MESSAGE")
    print("=" * 70)

    print(f"\nSender: {sender_name}")
    print(f"Message: {incoming_message}")

    print("=" * 70)

    initial_state: AgentState = {
        "sender_name": sender_name,
        "message": incoming_message,
    }

    try:
        result = agent.invoke(initial_state)

    except Exception as e:

        print("\nERROR in LangGraph/Ollama:")
        print(e)

        return

    approved = result.get("approved", False)

    if not approved:
        print("\nMessage was NOT sent.")
        return

    selected_reply = result.get("selected_reply", "").strip()

    if not selected_reply:
        print("\nERROR: Selected reply is empty.")
        return

    print("\nSending selected reply...")

    send_message(page, selected_reply)


# ============================================================
# MAIN WHATSAPP AUTOMATION
# ============================================================

def start_whatsapp():

    with sync_playwright() as playwright:

        print("\nStarting WhatsApp Web...")

        context = playwright.chromium.launch_persistent_context(
            user_data_dir=SESSION_DIR,
            headless=False,
            viewport=None,
            args=[
                "--start-maximized",
            ],
        )

        # Use existing page if available
        if context.pages:
            page = context.pages[0]
        else:
            page = context.new_page()

        print("\nOpening WhatsApp Web...")

        try:
            page.goto(
                WHATSAPP_URL,
                wait_until="domcontentloaded",
                timeout=120000,
            )

        except PlaywrightTimeoutError:
            print("\nWhatsApp page took longer than expected to load.")

        except Exception as e:
            print("\nCould not open WhatsApp Web:")
            print(e)

        time.sleep(5)

        # ----------------------------------------------------
        # Manual readiness confirmation
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("                 WHATSAPP WEB")
        print("=" * 70)

        print("\nIf QR code appears, scan it using your phone.")
        print("\nWait until WhatsApp Web is completely loaded.")

        input("\nWhen WhatsApp is ready, press ENTER...")

        print("\nWhatsApp session ready!")

        print("\n")
        print("=" * 70)
        print("              AUTOMATIC MONITORING ACTIVE")
        print("=" * 70)

        print("\nYou do NOT need to open any chat manually.")
        print("Send a new WhatsApp message to this account.")
        print("The agent will detect the unread chat automatically.")

        print("\nPress CTRL+C to stop.")
        print("\n")

        # ----------------------------------------------------
        # Duplicate prevention
        # ----------------------------------------------------

        processed_messages = set()

        # ----------------------------------------------------
        # Monitoring loop
        # ----------------------------------------------------

        while True:

            try:
                unread_chat = find_unread_chat(page)

                if unread_chat is None:
                    time.sleep(POLL_INTERVAL)
                    continue

                sender_name = get_row_name(unread_chat)

                print(f"\nUnread chat detected: {sender_name}")

                # Open automatically
                if not open_chat(page, unread_chat):
                    print("Could not open unread chat.")
                    time.sleep(1)
                    continue

                time.sleep(1)

                # Get actual current chat name
                current_name = get_current_chat_name(page)

                if current_name != "Unknown":
                    sender_name = current_name

                # ------------------------------------------------
                # Get incoming message
                # ------------------------------------------------

                incoming_message: Optional[str] = None

                # Give WhatsApp a moment to render messages
                for _ in range(10):

                    incoming_message = get_latest_incoming_message(page)

                    if incoming_message:
                        break

                    time.sleep(0.3)

                if not incoming_message:
                    print(
                        "\nChat opened, but incoming "
                        "message text was not found."
                    )
                    time.sleep(1)
                    continue

                incoming_message = clean_message(incoming_message)

                print("\nIncoming message detected:")
                print(incoming_message)

                # ------------------------------------------------
                # Duplicate key
                # ------------------------------------------------

                message_key = f"{sender_name}|||{incoming_message}"

                if message_key in processed_messages:
                    time.sleep(POLL_INTERVAL)
                    continue

                processed_messages.add(message_key)

                # ------------------------------------------------
                # AI + HITL workflow
                # ------------------------------------------------

                process_message(page, sender_name, incoming_message)

                time.sleep(1)

            except KeyboardInterrupt:
                print("\n")
                print("=" * 70)
                print("Monitoring stopped by user.")
                print("=" * 70)

                break

            except Exception as e:
                print("\nMonitoring error:")
                print(e)

                time.sleep(2)

        # --------------------------------------------------------
        # Close browser
        # --------------------------------------------------------

        try:
            context.close()
        except Exception:
            pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    start_whatsapp()