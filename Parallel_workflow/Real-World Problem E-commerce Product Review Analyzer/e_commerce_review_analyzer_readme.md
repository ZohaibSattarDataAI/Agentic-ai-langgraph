# 🛍️ E-Commerce Product Review Analyzer using LangGraph

An end-to-end parallel execution AI workflow built with **LangGraph** and **HuggingFace Transformers**. This project accepts unstructured customer reviews, processes them simultaneously across 3 worker nodes using fan-out/fan-in parallel architecture, and displays real-time results via an interactive dashboard.

---

## 📸 Workflow & Dashboard Overview

> **Note:** Place your project screenshot or architecture diagram inside an `assets` folder (e.g., `assets/preview.png`) to render it below.

![Project Overview & Dashboard Architecture](assets/preview.png)

---

## 🏗️ System Architecture

The core of this project is a **Parallel Directed Acyclic Graph (DAG)** built using `LangGraph`. Instead of analyzing reviews sequentially, the input is broadcasted simultaneously (Fan-out) to three specialized worker nodes before aggregating (Fan-in) into a single summary state using `operator.add`.

```
                  ┌────────────────────────┐
                  │        START           │
                  └───────────┬────────────┘
                              │
               ┌──────────────┼──────────────┐
               │ (Fan-out)    │              │
               ▼              ▼              ▼
     ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
     │Sentiment Analyzer│  │ Issues Extractor │  │Intent Classifier │
     │     (Node 1)     │  │     (Node 2)     │  │     (Node 3)     │
     └─────────┬────────┘  └─────────┬────────┘  └─────────┬────────┘
               │                     │                     │
               └──────────────┬──────┴─────────────────────┘
                              │ (Fan-in)
                              ▼
                  ┌────────────────────────┐
                  │    Aggregator Node     │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │          END           │
                  └───────────┬────────────┘
```

---

## ⚡ Key Features

- **Parallel Execution (Fan-out / Fan-in):** Cuts execution time by running Sentiment Analysis, Issue Extraction, and Intent Classification simultaneously.
- **State Management:** Utilizes Python `TypedDict` combined with `Annotated[list, operator.add]` to safely merge state updates across parallel nodes without overwriting.
- **API-Key-Free (Local Open-Source LLM):** Powered by HuggingFace's `TinyLlama/TinyLlama-1.1B-Chat-v1.0` pipeline for complete local or Google Colab execution.
- **Interactive UI Dashboard:** Comes with an integrated Gradio frontend for real-time review testing and interactive analysis.

---

## 📂 Project Structure

```text
.
├── assets/
│   └── preview.png          # Architecture diagram and dashboard screenshot
├── main.py                  # Complete LangGraph workflow & UI code
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Quickstart Guide

### 1. Install Dependencies
```bash
pip install -qU langgraph langchain-core langchain-community transformers accelerate gradio
```

### 2. Complete Application Code (`main.py` / Colab Notebook)

Copy and run the code below to execute the entire pipeline and open the web dashboard:

```python
import operator
from typing import TypedDict, Annotated
import gradio as gr
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langgraph.graph import StateGraph, START, END

# 1. Open-Source Model Setup
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.1
)

llm = HuggingFacePipeline(pipeline=pipe)

# 2. State Definition with List Reducer
class ReviewState(TypedDict):
    review: str
    sentiment: str
    issues: str
    intent: str
    summary_reports: Annotated[list, operator.add]

# 3. Parallel Worker Nodes & Aggregator Node
def node_sentiment(state: ReviewState):
    prompt = f"<|user|>\nAnalyze sentiment (Positive, Negative, or Neutral) of this product review: '{state['review']}'. Return only sentiment.\n<|assistant|>\n"
    res = llm.invoke(prompt).replace(prompt, "").strip()
    return {"sentiment": res, "summary_reports": [f"[SENTIMENT ANALYSIS]\n{res}"]}

def node_issues(state: ReviewState):
    prompt = f"<|user|>\nExtract main product issues or state 'None': '{state['review']}'\n<|assistant|>\n"
    res = llm.invoke(prompt).replace(prompt, "").strip()
    return {"issues": res, "summary_reports": [f"[KEY ISSUES]\n{res}"]}

def node_intent(state: ReviewState):
    prompt = f"<|user|>\nIdentify customer intent (e.g. Refund, Replacement, Feedback): '{state['review']}'\n<|assistant|>\n"
    res = llm.invoke(prompt).replace(prompt, "").strip()
    return {"intent": res, "summary_reports": [f"[CUSTOMER INTENT]\n{res}"]}

def aggregator_node(state: ReviewState):
    return state

# 4. Build Parallel Graph Architecture
builder = StateGraph(ReviewState)

builder.add_node("sentiment_analyzer", node_sentiment)
builder.add_node("issues_extractor", node_issues)
builder.add_node("intent_classifier", node_intent)
builder.add_node("aggregator", aggregator_node)

builder.add_edge(START, "sentiment_analyzer")
builder.add_edge(START, "issues_extractor")
builder.add_edge(START, "intent_classifier")

builder.add_edge("sentiment_analyzer", "aggregator")
builder.add_edge("issues_extractor", "aggregator")
builder.add_edge("intent_classifier", "aggregator")

builder.add_edge("aggregator", END)

review_graph = builder.compile()

# 5. Gradio Dashboard Application
def run_analyzer_ui(review_text):
    if not review_text.strip():
        return "Please enter a valid review.", "", ""
    
    state_input = {
        "review": review_text,
        "sentiment": "",
        "issues": "",
        "intent": "",
        "summary_reports": []
    }
    
    out = review_graph.invoke(state_input)
    return out.get("sentiment", "N/A"), out.get("issues", "N/A"), out.get("intent", "N/A")

theme = gr.themes.Soft(primary_hue="indigo")

with gr.Blocks(theme=theme, title="E-commerce Review Analyzer") as demo:
    gr.Markdown("# 🛍️ Parallel E-Commerce Review Analyzer")
    gr.Markdown("Powered by **LangGraph** & **HuggingFace (TinyLlama)** - Fan-out/Fan-in Architecture")
    
    with gr.Row():
        with gr.Column(scale=2):
            review_input = gr.Textbox(
                label="Customer Review Input",
                placeholder="Enter customer review here...",
                lines=4
            )
            submit_btn = gr.Button("🚀 Analyze Review (Parallel Run)", variant="primary")
            
            gr.Examples(
                examples=[
                    ["The battery life is terrible, dies in 2 hours! I want a refund."],
                    ["Absolutely love this laptop! The screen quality is fantastic and battery lasts all day."]
                ],
                inputs=review_input
            )
        
        with gr.Column(scale=2):
            out_sentiment = gr.Textbox(label="Detected Sentiment", interactive=False)
            out_issues = gr.Textbox(label="Extracted Product Issues", interactive=False)
            out_intent = gr.Textbox(label="Customer Intent", interactive=False)

    submit_btn.click(
        fn=run_analyzer_ui,
        inputs=[review_input],
        outputs=[out_sentiment, out_issues, out_intent]
    )

demo.launch(share=True)
```

---

## 📜 License

This project is open-source and licensed under the MIT License.
```