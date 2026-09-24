from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PRODUCTS_FILE = BASE_DIR / "products.csv"
POLICIES_FILE = BASE_DIR / "policies.md"

DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "ecommerce_rag"
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("E-COMMERCE RAG INGESTION")
print("=" * 60)

print(f"Base directory : {BASE_DIR}")
print(f"Products file  : {PRODUCTS_FILE}")
print(f"Policies file  : {POLICIES_FILE}")
print(f"ChromaDB       : {DB_DIR}")


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

if not PRODUCTS_FILE.is_file():
    raise FileNotFoundError(
        f"\nproducts.csv was not found.\n"
        f"Expected location:\n{PRODUCTS_FILE}"
    )

if not POLICIES_FILE.is_file():
    raise FileNotFoundError(
        f"\npolicies.md was not found.\n"
        f"Expected location:\n{POLICIES_FILE}"
    )

print("products.csv : FOUND")
print("policies.md  : FOUND")


# ============================================================
# LOAD PRODUCTS
# ============================================================

print("\n[1/4] Loading products.csv...")

df = pd.read_csv(PRODUCTS_FILE).fillna("")

print(f"Products loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# CREATE PRODUCT DOCUMENTS
# ============================================================

print("\nCreating product documents...")

documents = []
ids = []

for index, row in df.iterrows():

    product_lines = []

    for column, value in row.items():

        value = str(value).strip()

        if value:
            product_lines.append(
                f"{column}: {value}"
            )

    product_text = "\n".join(product_lines)

    if not product_text:
        continue

    document = Document(
        page_content=product_text,
        metadata={
            "source": "products.csv",
            "type": "product",
            "row": int(index),
        },
    )

    documents.append(document)
    ids.append(f"product-{index}")


print(
    f"Product documents created: "
    f"{len(documents)}"
)


# ============================================================
# LOAD POLICIES
# ============================================================

print("\n[2/4] Loading policies.md...")

policy_text = POLICIES_FILE.read_text(
    encoding="utf-8"
).strip()

if not policy_text:
    raise ValueError(
        "policies.md is empty."
    )

print(
    f"Policy characters: "
    f"{len(policy_text)}"
)


# ============================================================
# SPLIT POLICIES BY ## HEADINGS
# ============================================================

print("\nCreating policy documents...")

policy_documents = []

current_title = None
current_body = []


def save_current_policy():
    """
    Save the current policy as one
    complete RAG document.
    """

    if current_title is None:
        return

    body = "\n".join(
        current_body
    ).strip()

    if not body:
        return

    policy_content = (
        f"Policy: {current_title}\n\n"
        f"Answer: {body}"
    )

    document = Document(
        page_content=policy_content,
        metadata={
            "source": "policies.md",
            "type": "policy",
            "title": current_title,
        },
    )

    policy_documents.append(document)


# Read policies line by line
for raw_line in policy_text.splitlines():

    line = raw_line.strip()

    # Ignore empty lines
    if not line:
        continue

    # Ignore HTML comments
    if line.startswith("<!--"):
        continue

    # Ignore main Markdown title
    if line.startswith("# ") and not line.startswith("## "):
        continue

    # Detect a new policy
    if line.startswith("## "):

        # Save previous policy first
        save_current_policy()

        # Start new policy
        current_title = line[3:].strip()
        current_body = []

    else:

        # Add answer text to current policy
        current_body.append(line)


# Save final policy
save_current_policy()


print(
    f"Policy documents created: "
    f"{len(policy_documents)}"
)


# ============================================================
# ADD POLICY DOCUMENTS
# ============================================================

for index, document in enumerate(policy_documents):

    documents.append(document)

    ids.append(
        f"policy-{index}"
    )


print(
    f"Total documents created: "
    f"{len(documents)}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n[3/4] Loading embedding model...")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

print(
    f"Embedding model: "
    f"{EMBEDDING_MODEL}"
)


# ============================================================
# CREATE CHROMADB DIRECTORY
# ============================================================

print("\n[4/4] Creating ChromaDB...")

DB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)


# ============================================================
# ADD DOCUMENTS
# ============================================================

print("\nGenerating embeddings...")

if not documents:
    raise ValueError(
        "No documents were created."
    )

vector_db.add_documents(
    documents=documents,
    ids=ids,
)


# ============================================================
# VERIFY DATABASE
# ============================================================

try:

    collection_count = (
        vector_db._collection.count()
    )

except Exception:

    collection_count = len(documents)


# ============================================================
# SUCCESS
# ============================================================

print("\n" + "=" * 60)
print("RAG INGESTION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    f"Products loaded       : {len(df)}"
)

print(
    f"Product documents     : "
    f"{sum(1 for doc in documents if doc.metadata.get('type') == 'product')}"
)

print(
    f"Policy documents      : "
    f"{len(policy_documents)}"
)

print(
    f"Total documents       : "
    f"{len(documents)}"
)

print(
    f"ChromaDB documents    : "
    f"{collection_count}"
)

print(
    f"Embedding model       : "
    f"{EMBEDDING_MODEL}"
)

print(
    f"Collection            : "
    f"{COLLECTION_NAME}"
)

print(
    f"Database location     : "
    f"{DB_DIR}"
)

print("=" * 60)
print("Your RAG database is ready.")
print("=" * 60)