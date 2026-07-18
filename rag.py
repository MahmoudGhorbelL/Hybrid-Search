import pickle
import numpy as np
import re

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


# ======================================================
# Fonction principale
# ======================================================

def ask(question):

    print("=" * 60)
    print("HYBRID SEARCH")
    print("=" * 60)

    # ==================================================
    # 1. Recherche FAISS
    # ==================================================

    semantic_docs = vectorstore.similarity_search(
        question,
        k=15
    )

    print("\nRésultats FAISS :")

    for doc in semantic_docs:
        print(doc.metadata["source"])

    # ==================================================
    # 2. Recherche BM25
    # ==================================================

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
            f"{chunks[idx].metadata['source']} "
            f"(score={scores[idx]:.2f})"
        )

        if len(lexical_docs) == 10:
            break

    # ==================================================
    # 3. Fusion
    # ==================================================

    hybrid_docs = semantic_docs + lexical_docs

    # ==================================================
    # 4. Suppression des doublons
    # ==================================================

    unique_docs = []

    seen = set()

    for doc in hybrid_docs:

        key = (
            doc.metadata["source"],
            doc.page_content
        )

        if key not in seen:

            seen.add(key)

            unique_docs.append(doc)

    print("\nAprès fusion :")

    for doc in unique_docs:

        print(doc.metadata["source"])

    context = ""

    for doc in unique_docs:

        context += f"""

Source : {doc.metadata["source"]}

{doc.page_content}

------------------------------------------------

"""

    prompt = f"""
Tu es un assistant spécialisé dans l'analyse de CV.

Tu dois répondre uniquement à partir du contexte.

Consignes :

- Si plusieurs candidats correspondent, liste-les tous.

- Ne mélange jamais les informations entre plusieurs CV.

- Si l'utilisateur demande l'expérience,
retourne toute la section Experience.

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