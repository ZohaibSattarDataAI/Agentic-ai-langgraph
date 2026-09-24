"""
E-Commerce RAG
==============

Knowledge sources:
    - products.csv
    - policies.md

Vector database:
    - ChromaDB

Embedding model:
    - Ollama nomic-embed-text

Flow:
    User question
        ↓
    ChromaDB semantic search
        ↓
    Relevant product/policy documents
        ↓
    Extractive answer
        ↓
    Optional LLM answer

Important:
    - RAG sirf available knowledge se answer deta hai.
    - Product price/stock ke liye products.csv ko directly verify karta hai.
    - Agar information available nahi ho to NOT_FOUND style response deta hai.
"""

from pathlib import Path
import re
from typing import Optional, Callable, List, Dict, Any

import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PRODUCTS_FILE = BASE_DIR / "products.csv"
POLICIES_FILE = BASE_DIR / "policies.md"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "ecommerce_rag"
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "ki", "ka", "ke", "ko", "se",
    "me", "mein", "main",
    "hai", "hain", "hay",
    "ho", "hoga", "hogi",
    "kya", "kia",
    "ye", "yeh",
    "wo", "woh",
    "aur", "ya",
    "mujhe", "mera", "meri",
    "aap", "aapka", "apka", "apki",
    "kitna", "kitni", "kitne",
    "chahiye",
    "please", "plz",
    "batao", "bata",
    "bataen",
    "do", "karo",
    "kar", "dein",
    "the",
    "is", "a", "an",
    "of", "for",
    "and", "to",
    "in", "what",
    "how", "much",
    "do", "you",
    "have", "are",
    "there",
    "can", "i",
    "my", "your",
    "it", "this",
    "that",
}


# ============================================================
# ALIASES
# ============================================================

ALIASES = {}


_ALIAS_GROUPS = {
    "price": """
        price prices qeemat keemat kimat rate rates
        cost daam dam rs rupay rupees rupee pkr
    """,

    "delivery": """
        delivery deliver shipping courier
        pohanch pahunch pohnch pohanchega
        delivered dispatch
    """,

    "return": """
        return returns wapsi waapis wapas
        exchange badalna badal
    """,

    "refund": """
        refund refunds paisay paise paisa money
    """,

    "size": """
        size sizes naap chest waist
    """,

    "color": """
        color colors colour colours rang
    """,

    "stock": """
        stock available availability
        mojood maujood
    """,

    "discount": """
        discount sale offer offers off
    """,

    "contact": """
        contact number phone whatsapp call
        rabta email
    """,

    "location": """
        address location shop store dukaan showroom kahan
    """,

    "payment": """
        payment cod cash easypaisa jazzcash
        bank card adaigi
    """,

    "track": """
        track tracking
    """,
}


for canonical, words in _ALIAS_GROUPS.items():
    for word in words.split():
        ALIASES[word] = canonical


GENERIC_WORDS = {
    "price",
    "delivery",
    "return",
    "refund",
    "size",
    "color",
    "stock",
    "discount",
    "contact",
    "location",
    "payment",
    "track",
}


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> List[str]:
    """
    Convert text into normalized search tokens.
    """

    tokens = []

    text = (text or "").lower()

    for token in re.findall(
        r"[a-z0-9\u0600-\u06ff]+",
        text,
    ):

        token = ALIASES.get(token, token)

        # Simple plural normalization
        if (
            len(token) > 3
            and token.endswith("s")
            and not token.endswith("ss")
        ):
            token = token[:-1]

        if token in STOPWORDS:
            continue

        tokens.append(token)

    return tokens


# ============================================================
# NUMBER HELPERS
# ============================================================

def _number(value: Any) -> Optional[int]:
    """
    Convert price/stock values into integer.
    """

    if value is None:
        return None

    value = str(value).strip()

    digits = re.sub(r"[^\d]", "", value)

    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def _money(value: Optional[int]) -> str:
    """
    Format Pakistani Rupees.
    """

    if value is None:
        return "N/A"

    return f"Rs {value:,}"


# ============================================================
# PRODUCT HELPERS
# ============================================================

def _split_values(value: Any) -> List[str]:
    """
    Convert semicolon-separated CSV values into a list.
    """

    if value is None:
        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


# ============================================================
# KNOWLEDGE BASE
# ============================================================

class KnowledgeBase:

    def __init__(
        self,
        embed_fn=None,
        kb_dir=None,
        cache_path=None,
        model=EMBEDDING_MODEL,
        min_score=0.35,
        answer_mode="extract",
        llm_fn: Optional[Callable] = None,
        log=print,
    ):

        self.model = model
        self.min_score = min_score
        self.answer_mode = answer_mode
        self.llm_fn = llm_fn
        self.log = log

        # New architecture
        self.kb_dir = Path(kb_dir) if kb_dir else BASE_DIR
        self.chroma_dir = self.kb_dir / "chroma_db"

        self.products_file = self.kb_dir / "products.csv"
        self.policies_file = self.kb_dir / "policies.md"

        self.embeddings = None
        self.vector_db = None

        self.products_df = pd.DataFrame()

        self.ready = False

        # Kept for compatibility with old code
        self.chunks = []
        self.vector_ok = False


    # ========================================================
    # BUILD
    # ========================================================

    def build(self):

        self.log("\n" + "=" * 60)
        self.log("BUILDING E-COMMERCE RAG")
        self.log("=" * 60)

        self.log(f"Knowledge directory : {self.kb_dir}")
        self.log(f"ChromaDB directory  : {self.chroma_dir}")


        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not self.products_file.exists():

            self.log(
                f"[rag] WARNING: products.csv not found:\n"
                f"{self.products_file}"
            )

        if not self.policies_file.exists():

            self.log(
                f"[rag] WARNING: policies.md not found:\n"
                f"{self.policies_file}"
            )


        # ----------------------------------------------------
        # Load products directly
        # ----------------------------------------------------

        if self.products_file.exists():

            try:

                self.products_df = pd.read_csv(
                    self.products_file
                ).fillna("")

                self.log(
                    f"[rag] Products loaded: "
                    f"{len(self.products_df)}"
                )

            except Exception as exc:

                self.log(
                    f"[rag] Product CSV error: {exc}"
                )


        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        try:

            self.embeddings = OllamaEmbeddings(
                model=self.model
            )

            self.log(
                f"[rag] Embedding model: {self.model}"
            )

        except Exception as exc:

            self.log(
                f"[rag] Embedding model error: {exc}"
            )

            return self


        # ----------------------------------------------------
        # Check ChromaDB
        # ----------------------------------------------------

        if not self.chroma_dir.exists():

            self.log(
                "[rag] WARNING: chroma_db does not exist."
            )

            self.log(
                "[rag] Run: python ingest.py"
            )

            return self


        # ----------------------------------------------------
        # Connect to ChromaDB
        # ----------------------------------------------------

        try:

            self.vector_db = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(
                    self.chroma_dir
                ),
            )

            count = self.vector_db._collection.count()

            self.log(
                f"[rag] ChromaDB documents: {count}"
            )

            if count == 0:

                self.log(
                    "[rag] WARNING: ChromaDB is empty."
                )

                return self


        except Exception as exc:

            self.log(
                f"[rag] ChromaDB connection error: {exc}"
            )

            return self


        # ----------------------------------------------------
        # Ready
        # ----------------------------------------------------

        self.ready = True
        self.vector_ok = True

        self.log(
            "[rag] RAG is ready."
        )

        self.log("=" * 60)

        return self


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Search ChromaDB for relevant documents.
        """

        if not self.ready or self.vector_db is None:

            self.log(
                "[rag] Search requested but RAG is not ready."
            )

            return []


        query = (query or "").strip()

        if not query:

            return []


        try:

            results = (
                self.vector_db
                .similarity_search_with_relevance_scores(
                    query,
                    k=k,
                )
            )

        except Exception as exc:

            self.log(
                f"[rag] Search error: {exc}"
            )

            # Compatibility fallback
            try:

                docs = self.vector_db.similarity_search(
                    query,
                    k=k,
                )

                results = [
                    (doc, 0.5)
                    for doc in docs
                ]

            except Exception as fallback_exc:

                self.log(
                    f"[rag] Search fallback error: "
                    f"{fallback_exc}"
                )

                return []


        hits = []

        for document, score in results:

            try:
                score = float(score)
            except Exception:
                score = 0.0

            hits.append(
                {
                    "score": score,
                    "cos": score,
                    "lex": 0.0,
                    "chunk": {
                        "text": document.page_content,
                        "metadata": document.metadata,
                        "type": document.metadata.get(
                            "type",
                            "unknown",
                        ),
                        "title": self._get_title(
                            document
                        ),
                    },
                    "document": document,
                }
            )


        # Chroma relevance scores normally range 0-1
        hits.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return hits


    # ========================================================
    # TITLE
    # ========================================================

    @staticmethod
    def _get_title(document) -> str:
        """
        Get a useful source title.
        """

        metadata = document.metadata or {}

        source_type = metadata.get(
            "type",
            "unknown",
        )

        if source_type == "product":

            text = document.page_content

            for line in text.splitlines():

                if line.lower().startswith("name:"):

                    return line.split(
                        ":", 1
                    )[1].strip()

            return "Product"

        if source_type == "policy":

            return "Store Policy"

        return metadata.get(
            "source",
            "Knowledge",
        )


    # ========================================================
    # FIND PRODUCT
    # ========================================================

    def _find_product(
        self,
        question: str,
    ) -> Optional[pd.Series]:
        """
        Try to identify a product directly from products.csv.

        This is important for exact:
            - price
            - sale price
            - stock
            - size
            - color
        """

        if self.products_df.empty:

            return None


        question_lower = question.lower()


        # ----------------------------------------------------
        # Exact product name match
        # ----------------------------------------------------

        if "name" in self.products_df.columns:

            for _, row in self.products_df.iterrows():

                name = str(
                    row.get("name", "")
                ).strip()

                if not name:
                    continue

                if name.lower() in question_lower:

                    return row


        # ----------------------------------------------------
        # SKU match
        # ----------------------------------------------------

        if "sku" in self.products_df.columns:

            for _, row in self.products_df.iterrows():

                sku = str(
                    row.get("sku", "")
                ).strip()

                if (
                    sku
                    and sku.lower()
                    in question_lower
                ):

                    return row


        return None


    # ========================================================
    # PRODUCT ANSWER
    # ========================================================

    def _product_answer(
        self,
        product: pd.Series,
        question: str,
    ) -> str:
        """
        Generate deterministic product answer.
        """

        name = str(
            product.get("name", "")
        ).strip()

        category = str(
            product.get("category", "")
        ).strip()

        price = _number(
            product.get("price")
        )

        sale_price = _number(
            product.get("sale_price")
        )

        stock = _number(
            product.get("stock")
        )

        sizes = _split_values(
            product.get("sizes")
        )

        colors = _split_values(
            product.get("colors")
        )

        fabric = str(
            product.get("fabric", "")
        ).strip()

        description = str(
            product.get("description", "")
        ).strip()


        # Effective price
        if (
            sale_price is not None
            and price is not None
            and sale_price < price
        ):

            price_text = (
                f"{_money(sale_price)} "
                f"(sale, asal qeemat {_money(price)})"
            )

        else:

            price_text = _money(price)


        # Stock
        if stock is None:

            stock_text = (
                "Stock ki exact quantity available "
                "data mein mention nahi hai."
            )

        elif stock > 0:

            stock_text = (
                f"Filhal stock mein hai "
                f"({stock} available)."
            )

        else:

            stock_text = (
                "Filhal ye stock mein nahi hai."
            )


        q_tokens = set(
            tokenize(question)
        )


        # ----------------------------------------------------
        # Price-only question
        # ----------------------------------------------------

        if (
            "price" in q_tokens
            and not (
                q_tokens
                & {
                    "size",
                    "color",
                    "stock",
                    "fabric",
                }
            )
        ):

            return (
                f"{name} ki qeemat "
                f"{price_text} hai."
            )


        # ----------------------------------------------------
        # Size question
        # ----------------------------------------------------

        if "size" in q_tokens:

            if sizes:

                return (
                    f"{name} ke available sizes: "
                    f"{', '.join(sizes)}."
                )

            return (
                f"{name} ke sizes ki information "
                f"available data mein nahi hai."
            )


        # ----------------------------------------------------
        # Color question
        # ----------------------------------------------------

        if "color" in q_tokens:

            if colors:

                return (
                    f"{name} ke available colors: "
                    f"{', '.join(colors)}."
                )

            return (
                f"{name} ke colors ki information "
                f"available data mein nahi hai."
            )


        # ----------------------------------------------------
        # Stock question
        # ----------------------------------------------------

        if "stock" in q_tokens:

            return (
                f"{name}: {stock_text}"
            )


        # ----------------------------------------------------
        # Full product information
        # ----------------------------------------------------

        answer_parts = [
            f"{name} ({category})",
            f"Qeemat: {price_text}",
        ]

        if sizes:

            answer_parts.append(
                f"Sizes: {', '.join(sizes)}"
            )

        if colors:

            answer_parts.append(
                f"Rang: {', '.join(colors)}"
            )

        if fabric:

            answer_parts.append(
                f"Fabric: {fabric}"
            )

        answer_parts.append(
            stock_text
        )

        if description:

            answer_parts.append(
                description
            )

        return "\n".join(answer_parts)


    # ========================================================
    # EXTRACT ANSWER
    # ========================================================

    def _extract(
        self,
        question: str,
        hits: List[Dict[str, Any]],
    ) -> str:
        """
        Extract answer from retrieved knowledge.
        """

        # ----------------------------------------------------
        # First: exact product from CSV
        # ----------------------------------------------------

        product = self._find_product(
            question
        )

        if product is not None:

            return self._product_answer(
                product,
                question,
            )


        # ----------------------------------------------------
        # No direct product
        # Use top RAG document
        # ----------------------------------------------------

        if not hits:

            return ""


        top = hits[0]["chunk"]

        source_type = top["type"]


        # ----------------------------------------------------
        # Policy / FAQ
        # ----------------------------------------------------

        if source_type == "policy":

            text = top["text"].strip()

            if len(text) > 700:

                text = (
                    text[:700]
                    .rsplit(" ", 1)[0]
                    + "..."
                )

            return text


        # ----------------------------------------------------
        # Product retrieved semantically
        # ----------------------------------------------------

        if source_type == "product":

            text = top["text"].strip()

            if text:

                return text[:700]


        return ""


    # ========================================================
    # LLM ANSWER
    # ========================================================

    def _llm(
        self,
        question: str,
        hits: List[Dict[str, Any]],
    ) -> str:
        """
        Optional LLM response.

        LLM is strictly restricted to retrieved context.
        """

        if not self.llm_fn:

            return ""


        context_parts = []

        for hit in hits[:3]:

            text = hit["chunk"]["text"]

            if text:

                context_parts.append(text)


        context = "\n---\n".join(
            context_parts
        )


        system_prompt = """
Tum ek e-commerce clothing store ke customer support agent ho.

IMPORTANT RULES:

1. Sirf provided context se jawab do.
2. Apni taraf se price invent mat karo.
3. Apni taraf se stock invent mat karo.
4. Apni taraf se delivery days invent mat karo.
5. Agar context mein answer nahi hai to exactly:
   NOT_FOUND
   likho.
6. Roman Urdu mein jawab do.
7. Maximum 3 short sentences.
8. Context ke bahar ki information mat do.
"""


        try:

            response = self.llm_fn(
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n"
                            f"{context}\n\n"
                            f"Sawal:\n"
                            f"{question}"
                        ),
                    },
                ],
                temperature=0,
                num_predict=200,
            )


            answer = (
                response or ""
            ).strip()


        except Exception as exc:

            self.log(
                f"[rag] LLM error: {exc}"
            )

            return ""


        if not answer:

            return ""


        if "NOT_FOUND" in answer:

            return "NOT_FOUND"


        # Prevent excessively long responses
        if len(answer) > 600:

            return ""


        return answer


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    def answer(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Main RAG answer function.

        Returns:

        {
            "ok": True/False,
            "text": "...",
            "score": 0.0,
            "sources": [...],
            "mode": "extract/llm/none"
        }
        """

        question = (
            question or ""
        ).strip()


        if not question:

            return {
                "ok": False,
                "text": "",
                "score": 0.0,
                "sources": [],
                "mode": "none",
            }


        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        hits = self.search(
            question,
            k=4,
        )


        if not hits:

            self.log(
                "[rag] No documents retrieved."
            )

            return {
                "ok": False,
                "text": "",
                "score": 0.0,
                "sources": [],
                "mode": "none",
            }


        top_score = hits[0]["score"]


        # ----------------------------------------------------
        # Score threshold
        # ----------------------------------------------------

        if top_score < self.min_score:

            self.log(
                f"[rag] Low relevance score: "
                f"{top_score:.2f}"
            )

            # Exact product can still be trusted
            product = self._find_product(
                question
            )

            if product is None:

                return {
                    "ok": False,
                    "text": "",
                    "score": top_score,
                    "sources": [],
                    "mode": "none",
                }


        # ----------------------------------------------------
        # Extractive answer
        # ----------------------------------------------------

        text = self._extract(
            question,
            hits,
        )

        mode = "extract"


        # ----------------------------------------------------
        # Optional LLM
        # ----------------------------------------------------

        if (
            self.answer_mode == "llm"
            and self.llm_fn
        ):

            llm_answer = self._llm(
                question,
                hits,
            )

            if llm_answer == "NOT_FOUND":

                self.log(
                    "[rag] LLM: answer not found."
                )

                return {
                    "ok": False,
                    "text": "",
                    "score": top_score,
                    "sources": [],
                    "mode": "llm_not_found",
                }


            if llm_answer:

                text = llm_answer
                mode = "llm"


        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        if not text:

            return {
                "ok": False,
                "text": "",
                "score": top_score,
                "sources": [],
                "mode": "none",
            }


        sources = []

        for hit in hits[:2]:

            title = hit["chunk"]["title"]

            if title and title not in sources:

                sources.append(title)


        self.log(
            f"[rag] top={top_score:.2f} "
            f"mode={mode} "
            f"sources={sources}"
        )


        return {
            "ok": True,
            "text": text,
            "score": top_score,
            "sources": sources,
            "mode": mode,
        }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("\nStarting RAG test...\n")

    rag = KnowledgeBase(
        kb_dir=BASE_DIR,
        model=EMBEDDING_MODEL,
        min_score=0.35,
        answer_mode="extract",
    )

    rag.build()


    if rag.ready:

        print("\n" + "=" * 60)
        print("RAG TEST")
        print("=" * 60)

        test_questions = [
            "What products do you have?",
            "delivery policy kya hai?",
            "return policy kya hai?",
        ]

        for question in test_questions:

            print(
                f"\nUser: {question}"
            )

            result = rag.answer(
                question
            )

            print(
                f"Answer: "
                f"{result['text']}"
            )

            print(
                f"Score: "
                f"{result['score']:.2f}"
            )

            print(
                f"Mode: "
                f"{result['mode']}"
            )

    else:

        print(
            "\nRAG is not ready."
        )

        print(
            "Pehle run karein:"
        )

        print(
            "python ingest.py"
        )