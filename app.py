import streamlit as st
from rrf import ask

# Configuration de la page
st.set_page_config(
    page_title="CV RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 CV RAG Assistant")
st.caption("Posez vos questions sur les CV indexés.")

# Historique de la conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les anciens messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
question = st.chat_input("Écrivez votre question...")

if question:

    # Afficher le message utilisateur
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Générer la réponse
    with st.chat_message("assistant"):

        with st.spinner("Recherche dans les CV..."):

            answer = ask(question)

            st.markdown(answer)

    # Sauvegarder la réponse
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )