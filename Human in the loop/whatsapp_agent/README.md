# 📱 WhatsApp AI Agent with Human-in-the-Loop (LangGraph + Playwright + Ollama)

> An intelligent WhatsApp automation agent that monitors incoming messages, generates AI-powered replies in **English** and **Roman Urdu**, and sends them **only after human approval** — built with **LangGraph**, **Playwright**, and **Ollama (qwen2.5:1.5b)**.

---

## 📌 Overview

This project demonstrates a **Human-in-the-Loop (HITL)** AI agent that:

- 🟢 Automatically opens **WhatsApp Web** using a persistent Playwright browser session
- 🔍 Continuously monitors for **unread chats / new messages**
- 🧠 Sends the sender name + message to a **LangGraph agent workflow**
- 🤖 Uses **Ollama (qwen2.5:1.5b)** to generate **two reply options**:
  - 🇬🇧 Natural English reply
  - 🇵🇰 Roman Urdu / Roman English reply
- 👤 Displays both replies to the human for **selection**
- ✅ Asks for **send approval** (YES / NO)
- 📤 Sends the message via WhatsApp Web only if approved
- 🔁 Returns to monitoring loop for the next message

---

## 🏗️ Architecture / Workflow

```
┌─────────────────────┐
│        START        │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────┐
│ Launch Python Application    │
│          main.py             │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Start Playwright             │
│ Persistent Browser Session   │
│     whatsapp_session/        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Open WhatsApp Web            │
│ web.whatsapp.com             │
└──────────────┬───────────────┘
               │
               ▼
        ◇ QR Code Available? ◇
           /              \
         YES               NO
          │                 │
          ▼                 │
 ┌────────────────┐         │
 │ Scan QR Code   │         │
 │ Using Phone    │         │
 └───────┬────────┘         │
         │                  │
         └────────┬─────────┘
                  │
                  ▼
  ┌──────────────────────────────┐
  │ WhatsApp Web Ready           │
  │ User Presses ENTER           │
  └──────────────┬───────────────┘
                 │
                 ▼
 ╔════════════════════════════════════════╗
 ║       AUTOMATIC MONITORING LOOP        ║
 ╚════════════════════╤═══════════════════╝
                      │
                      ▼
      ┌──────────────────────────────┐
      │ Check WhatsApp for           │
      │ Unread Chat / New Message    │
      └──────────────┬───────────────┘
                     │
                     ▼
             ◇ Unread Chat Found? ◇
                /             \
              NO              YES
               │                │
               │                ▼
               │     ┌─────────────────────┐
               │     │ Detect Chat         │
               │     └──────────┬──────────┘
               │                ▼
               │     ┌─────────────────────┐
               │     │ Get Sender Name     │
               │     └──────────┬──────────┘
               │                ▼
               │     ┌─────────────────────┐
               │     │ Open Unread Chat    │
               │     └──────────┬──────────┘
               │                ▼
               │     ┌─────────────────────┐
               │     │ Extract Latest      │
               │     │ Incoming Message    │
               │     └──────────┬──────────┘
               │                ▼
               │       ◇ Message Found? ◇
               │          /          \
               │        NO            YES
               │        │              │
               │        ▼              ▼
               │  ┌───────────┐  ┌──────────────────────┐
               │  │ Inspect / │  │ Sender + Message     │
               │  │ Debug DOM │  │ Successfully Captured│
               │  └─────┬─────┘  └──────────┬───────────┘
               │        │                   │
               │        └──► RETRY          │
               │                            ▼
               │              ┌─────────────────────────┐
               │              │       LANGGRAPH         │
               │              │      Agent Workflow     │
               │              └────────────┬────────────┘
               │                           ▼
               │              ┌─────────────────────────┐
               │              │ Send Sender + Message   │
               │              │ to AI Reply Generator   │
               │              └────────────┬────────────┘
               │                           ▼
               │              ┌─────────────────────────┐
               │              │        OLLAMA           │
               │              │      qwen2.5:1.5b       │
               │              └────────────┬────────────┘
               │                           ▼
               │              ┌─────────────────────────┐
               │              │ Generate TWO Replies    │
               │              └────────────┬────────────┘
               │                           │
               │             ┌─────────────┴─────────────┐
               │             ▼                           ▼
               │   ┌────────────────────┐     ┌────────────────────┐
               │   │   ENGLISH REPLY    │     │    ROMAN REPLY     │
               │   │ Natural English    │     │ Roman Urdu/English │
               │   └──────────┬─────────┘     └──────────┬─────────┘
               │              └─────────────┬────────────┘
               │                            ▼
               │              ┌─────────────────────────┐
               │              │ Show Both Replies to    │
               │              │       HUMAN             │
               │              └────────────┬────────────┘
               │                           ▼
               │              ◇ Human Selects Reply ◇
               │                     /        \
               │                    1          2
               │                    │          │
               │                    ▼          ▼
               │             ┌──────────┐ ┌──────────┐
               │             │ English  │ │  Roman   │
               │             └─────┬────┘ └────┬─────┘
               │                   └─────┬─────┘
               │                         ▼
               │              ┌─────────────────────────┐
               │              │ Display Selected Reply  │
               │              └────────────┬────────────┘
               │                           ▼
               │                 ◇ SEND APPROVAL? ◇
               │                    /           \
               │                  NO             YES
               │                  │               │
               │                  ▼               ▼
               │         ┌────────────────┐ ┌──────────────────┐
               │         │ CANCEL MESSAGE │ │ Send Selected    │
               │         │ Do NOT Send    │ │ Reply            │
               │         └───────┬────────┘ └────────┬─────────┘
               │                 │                   ▼
               │                 │         ┌─────────────────────┐
               │                 │         │ WhatsApp Web        │
               │                 │         │ Message Box         │
               │                 │         └──────────┬──────────┘
               │                 │                    ▼
               │                 │         ┌─────────────────────┐
               │                 │         │ Press ENTER         │
               │                 │         │ Send Message        │
               │                 │         └──────────┬──────────┘
               │                 │                    ▼
               │                 │         ┌─────────────────────┐
               │                 │         │ Message Sent ✅     │
               │                 │         └──────────┬──────────┘
               │                 │                    │
               └─────────────────┴────────────────────┘
                                 │
                                 ▼
                    ╔════════════════════════╗
                    ║ RETURN TO MONITORING   ║
                    ║     FOR NEW MESSAGE    ║
                    ╚════════════╤═══════════╝
                                 │
                                 └──────► LOOP
```

---

## 🧠 LangGraph Agent — State Flow

```
┌───────────────────────────────────────────────┐
│                  AgentState                   │
├───────────────────────────────────────────────┤
│ sender_name                                   │
│ message                                       │
│ english_reply                                 │
│ roman_reply                                   │
│ selected_reply                                │
│ language                                      │
│ approved                                      │
└───────────────────────────────────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │ Generate    │
              │ Reply       │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Human       │
              │ Selection   │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Approval    │
              │ YES / NO    │
              └──────┬──────┘
                     │
             ┌───────┴────────┐
             ▼                ▼
           YES                NO
             │                │
             ▼                ▼
        WhatsApp Send       Cancel
```

---

## 📂 Project Structure

```
Agentic-ai-langgraph/
│
├── main.py                     # Entry point — launches Playwright + agent
├── agent/
│   ├── graph.py                # LangGraph workflow definition
│   ├── state.py                # AgentState TypedDict
│   ├── nodes.py                # Reply generation, human selection, approval
│   └── llm.py                  # Ollama (qwen2.5:1.5b) wrapper
│
├── whatsapp/
│   ├── browser.py              # Playwright persistent session
│   ├── monitor.py              # Unread chat detection loop
│   ├── sender.py               # Message send logic
│   └── selectors.py            # WhatsApp Web DOM selectors
│
├── whatsapp_session/           # Persistent browser profile (gitignored)
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Requirements

- Python **3.10+**
- [Ollama](https://ollama.com/) installed and running
- WhatsApp account on phone (for QR scan)
- OS: Windows / macOS / Linux

### Install Ollama Model

```bash
ollama pull qwen2.5:1.5b
ollama serve
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

**`requirements.txt`**

```
langgraph
langchain
langchain-ollama
playwright
python-dotenv
```

---

## 🚀 Usage

1. **Start Ollama**

   ```bash
   ollama serve
   ```

2. **Run the agent**

   ```bash
   python main.py
   ```

3. **Scan the QR code** in the opened WhatsApp Web browser window (first run only).

4. **Press ENTER** in the terminal once WhatsApp Web is fully loaded.

5. The agent now **monitors incoming messages**. When a new message arrives:
   - It extracts the sender + message
   - Generates **English** and **Roman Urdu** replies
   - Shows both to you in the terminal
   - Waits for your **selection (1 or 2)**
   - Asks for **send approval (YES/NO)**
   - Sends via WhatsApp Web if approved

---

## 🧩 AgentState Schema

```python
from typing import TypedDict, Optional

class AgentState(TypedDict):
    sender_name: str
    message: str
    english_reply: str
    roman_reply: str
    selected_reply: Optional[str]
    language: Optional[str]     # "english" | "roman"
    approved: Optional[bool]
```

---

## 🖥️ Example Interaction

```
📩 New message from: Ali
💬 Message: "Kal meeting hai kya?"

🇬🇧 English Reply:
   "Yes, we have a meeting tomorrow. Please confirm the time."

🇵🇰 Roman Reply:
   "Haan, kal meeting hai. Time confirm kar lein."

Select reply [1=English, 2=Roman]: 2
Selected: Haan, kal meeting hai. Time confirm kar lein.
Send this message? (yes/no): yes

✅ Message sent successfully.
🔁 Returning to monitoring loop...
```

---

## 🔐 Privacy & Safety

- ✅ **Human-in-the-Loop** — no message is sent without approval
- ✅ Browser session is stored **locally** in `whatsapp_session/`
- ✅ No external API calls except local **Ollama**
- ⚠️ Do **not** commit `whatsapp_session/` or `.env` to Git

**`.gitignore`**

```
whatsapp_session/
.env
__pycache__/
*.pyc
.venv/
```

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| QR not showing | Delete `whatsapp_session/` and re-run |
| No unread chats detected | Update DOM selectors in `whatsapp/selectors.py` |
| Ollama not responding | Run `ollama serve` in a separate terminal |
| Message not sent | Ensure chat input box is focused before pressing ENTER |
| Playwright not found | Run `playwright install chromium` |

---

## 🗺️ Roadmap

- [ ] Support for **group chats**
- [ ] **Voice message** transcription
- [ ] Multi-language replies (Hindi, Arabic, etc.)
- [ ] **Web dashboard** for approvals instead of terminal
- [ ] Reply **memory / context** across messages
- [ ] Rate-limiting & anti-spam guard

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

```bash
git checkout -b feature/your-feature
git commit -m "Add: your feature description"
git push origin feature/your-feature
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Zohaib Sattar**
Agentic AI • LangGraph • Playwright • Ollama

> ⭐ If this project helped you, please give it a star!
