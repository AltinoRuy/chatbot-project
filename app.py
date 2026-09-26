import streamlit as st

from services.memory import Memory
from services.orchestrator import process_message

st.title("My AI Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []


if "memory" not in st.session_state:
    st.session_state.memory = Memory()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


prompt = st.chat_input("Type your message...")


if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    st.write("PROCESSANDO:", prompt)

    response = process_message(
        prompt,
        st.session_state.memory,
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    st.rerun()
