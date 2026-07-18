from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

from collections import Counter
import pickle
import glob
import os

print("=" * 70)
print("        INDEXATION HYBRID SEARCH (FAISS + BM25)")
print("=" * 70)

# ---------------------------------------------------
# Chargement des PDF
# ---------------------------------------------------

pdf_folder = "data"

pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

documents = []

print(f"\nNombre de CV trouvés : {len(pdf_files)}\n")

for pdf in pdf_files:

    print("Chargement :", os.path.basename(pdf))

    loader = PyPDFLoader(pdf)

    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = os.path.basename(pdf)

    documents.extend(docs)

print(f"\nNombre total de pages : {len(documents)}")

# ---------------------------------------------------
# Découpage
# ---------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300
)

chunks = text_splitter.split_documents(documents)

print(f"\nNombre total de chunks : {len(chunks)}")

counter = Counter()

for chunk in chunks:
    counter[chunk.metadata["source"]] += 1

print("\nRépartition des chunks :")

for cv, nb in counter.items():
    print(f"{cv} : {nb}")

# ---------------------------------------------------
# Création FAISS
# ---------------------------------------------------

print("\nCréation de FAISS...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

os.makedirs("db", exist_ok=True)

vectorstore.save_local("db")

print("FAISS créé.")

# ---------------------------------------------------
# Création BM25
# ---------------------------------------------------

print("\nCréation de BM25...")

tokenized_chunks = []

for chunk in chunks:
    tokenized_chunks.append(
        chunk.page_content.lower().split()
    )

bm25 = BM25Okapi(tokenized_chunks)

with open("db/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

with open("db/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("BM25 créé.")

print("=" * 70)
print("INDEXATION HYBRID TERMINÉE")
print("=" * 70)