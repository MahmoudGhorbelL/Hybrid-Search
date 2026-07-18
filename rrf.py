import pickle
import numpy as np
import re
from collections import defaultdict
from ollama import chat
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = FAISS.load_local(
    "db",
    embeddings,
    allow_dangerous_deserialization=True
)
with open("db/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("db/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


# =====================================================
# Fonction principale
# =====================================================

def ask(question):

    print("=" * 60)
    print("HYBRID SEARCH + RRF")
    print("=" * 60)

    # 1. FAISS

    semantic_docs = vectorstore.similarity_search(
        question,
        k=15
    )

    print("\nRésultats FAISS :")

    for i, doc in enumerate(semantic_docs):

        print(
            f"{i+1}. {doc.metadata['source']}"
        )

    # 2. BM25

    query = re.findall(r"\b\w+\b", question.lower())

    scores = bm25.get_scores(query)

    sorted_indices = np.argsort(scores)[::-1]

    lexical_docs = []

    print("\nRésultats BM25 :")

    for idx in sorted_indices:

        if scores[idx] <= 0:
            continue

        lexical_docs.append(chunks[idx])

        print(
            f"{len(lexical_docs)}. "
            f"{chunks[idx].metadata['source']} "
            f"(score={scores[idx]:.2f})"
        )

        if len(lexical_docs) == 15:
            break

    # 3. Reciprocal Rank Fusion

    k_rrf = 60

    rrf_scores = defaultdict(float)

    doc_map = {}

    # ---------- FAISS ----------

    for rank, doc in enumerate(semantic_docs, start=1):

        key =doc.metadata["source"]
        rrf_scores[key] += 1 / (k_rrf + rank)

        doc_map[key] = doc

    # ---------- BM25 ----------

    for rank, doc in enumerate(lexical_docs, start=1):

        key = doc.metadata["source"]

        rrf_scores[key] += 1 / (k_rrf + rank)

        doc_map[key] = doc

    # 4. Tri des scores

    ranked_docs = sorted(

        rrf_scores.items(),

        key=lambda x: x[1],

        reverse=True

    )

    print("\nClassement RRF :\n")

    final_docs = []

    for key, score in ranked_docs[:15]:

        doc = doc_map[key]

        final_docs.append(doc)

        print(
            f"{doc.metadata['source']} "
            f"RRF={score:.4f}"
        )


    context = ""

    for doc in final_docs:

        context += f"""

Source : {doc.metadata["source"]}

{doc.page_content}

------------------------------------------------

"""

    prompt = f"""
Tu es un assistant spécialisé dans l'analyse des CV.

Réponds uniquement à partir du contexte.

Consignes :

- Ne jamais inventer.

- Ne jamais mélanger deux candidats.

- Si plusieurs candidats correspondent,
liste-les tous.

- Si l'utilisateur demande les expériences,
retourne toutes les expériences.

- Si l'utilisateur demande les compétences,
retourne toute la section Skills.

- Si l'information est absente :

Information non trouvée dans les CV.

Contexte :

{context}

Question :

{question}

Réponse :
"""

    response = chat(

        model="gpt-oss:20b-cloud",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )

    return response.message.content