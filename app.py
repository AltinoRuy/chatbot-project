import html

import streamlit as st

from services.memory import Memory
from services.orchestrator import process_message

st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 80% 0%,
                rgba(255, 152, 0, 0.07),
                transparent 28%
            ),
            #0e1016;
        color: #f5f6fa;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11151e, #0d1017);
        border-right: 1px solid #262b38;
    }

    [data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2.5rem;
        padding-bottom: 7rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 15px;
        background: #ff9800;
        color: #14161c;
        font-size: 27px;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 750;
        color: #f5f6fa;
    }

    .brand-title span {
        color: #ff9800;
    }

    .brand-description {
        color: #9ca7bb;
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 25px;
    }

    .side-heading {
        color: #8290a7;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        margin-top: 24px;
        margin-bottom: 12px;
    }

    .feature {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid #262c3a;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.025);
        color: #dce2ed;
        font-size: 13px;
    }

    .feature-icon {
        font-size: 19px;
    }

    .side-footer {
        margin-top: 30px;
        padding-top: 16px;
        border-top: 1px solid #262b38;
        color: #8290a7;
        font-size: 12px;
        line-height: 1.7;
    }

    .hero {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 30px;
    }

    .hero-icon {
        color: #ff9800;
        font-size: 35px;
    }

    .hero-title {
        margin: 0;
        color: #f7f7fb;
        font-size: 34px;
        font-weight: 750;
        letter-spacing: -1px;
    }

    .hero-description {
        margin-top: 5px;
        color: #9ca7bb;
        font-size: 14px;
    }

    .welcome {
        padding: 25px;
        margin: 25px 0 18px;
        border: 1px solid #292f3d;
        border-radius: 18px;
        background: linear-gradient(
            125deg,
            rgba(255, 152, 0, 0.07),
            rgba(28, 33, 46, 0.6)
        );
    }

    .welcome-title {
        font-size: 22px;
        font-weight: 700;
        color: #f5f6fa;
        margin-bottom: 8px;
    }

    .welcome-description {
        color: #a5afc1;
        font-size: 14px;
        line-height: 1.7;
    }

    .stButton > button {
        min-height: 44px;
        border: 1px solid #343b4b;
        border-radius: 12px;
        background: #191d27;
        color: #e0e5ef;
        transition: border-color 0.2s, background 0.2s;
    }

    .stButton > button:hover {
        border-color: #ff9800;
        background: #242733;
        color: #ffffff;
    }

    .stButton > button:focus:not(:active) {
        border-color: #ff9800;
        box-shadow: 0 0 0 1px rgba(255, 152, 0, 0.2);
    }

    .chat-row {
        display: flex;
        align-items: flex-start;
        gap: 13px;
        padding: 17px;
        margin-bottom: 13px;
        border: 1px solid #292f3c;
        border-radius: 16px;
    }

    .chat-row--user {
        border-color: rgba(255, 65, 75, 0.25);
        background: linear-gradient(
            110deg,
            rgba(255, 65, 75, 0.09),
            rgba(27, 30, 40, 0.92)
        );
    }

    .chat-row--assistant {
        border-color: rgba(255, 152, 0, 0.18);
        background: linear-gradient(
            110deg,
            rgba(255, 152, 0, 0.055),
            rgba(27, 30, 40, 0.92)
        );
    }

    .chat-avatar {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        min-width: 42px;
        border-radius: 13px;
        color: #14161c;
        font-size: 22px;
    }

    .chat-row--user .chat-avatar {
        background: #ff414b;
    }

    .chat-row--assistant .chat-avatar {
        background: #ff9800;
    }

    .chat-body {
        flex: 1;
        min-width: 0;
        padding-top: 1px;
    }

    .chat-label {
        margin-bottom: 7px;
        color: #aab4c6;
        font-size: 12px;
        font-weight: 700;
    }

    .chat-row--user .chat-label {
        color: #ff777e;
    }

    .chat-row--assistant .chat-label {
        color: #ffb443;
    }

    .chat-content {
        color: #f3f4f8;
        font-size: 15px;
        line-height: 1.65;
        overflow-wrap: anywhere;
    }

    [data-testid="stChatInput"] {
        border: 1px solid #343b4c;
        border-radius: 17px;
        background: #191d27;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.18);
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #ff9800;
        box-shadow: 0 0 0 1px rgba(255, 152, 0, 0.18);
    }

    [data-testid="stChatInput"] textarea {
        color: #f5f6fa !important;
        background: transparent !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #8792a7 !important;
    }

    [data-testid="stChatInput"] button {
        border-radius: 11px;
        background: #ff9800;
        color: #14161c;
    }

    [data-testid="stChatInput"] button:hover {
        background: #ffad32;
        color: #14161c;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1.5rem;
        }

        .hero-title {
            font-size: 27px;
        }

        .chat-row {
            gap: 10px;
            padding: 12px;
        }

        .chat-avatar {
            width: 36px;
            height: 36px;
            min-width: 36px;
            font-size: 19px;
        }

        .welcome {
            padding: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🤖</div>
            <div class="brand-title">AI <span>Chatbot</span></div>
        </div>

        <div class="brand-description">
            A lightweight AI assistant for math and contextual
            follow-up questions.
        </div>

        <div class="side-heading">CAPABILITIES</div>

        <div class="feature">
            <span class="feature-icon">🧮</span>
            <span>Deterministic math tools</span>
        </div>

        <div class="feature">
            <span class="feature-icon">🧠</span>
            <span>Contextual calculations</span>
        </div>

        <div class="feature">
            <span class="feature-icon">💬</span>
            <span>Multiple LLM providers</span>
        </div>

        <div class="side-footer">
            Built with Python and Streamlit<br>
            Agent-based architecture
        </div>
        """,
        unsafe_allow_html=True,
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = Memory()


st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">✦</div>
        <div>
            <h1 class="hero-title">My AI Chatbot</h1>
            <div class="hero-description">
                Math, context, and helpful responses — all in one chat.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-title">What shall we solve today?</div>
            <div class="welcome-description">
                Choose an example to start, or type your own question below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        ("🧮  What is 5 + 4?", "What is 5 + 4?"),
        ("✖️  What is 12 × 3?", "What is 12 times 3?"),
        ("➗  What is 25 divided by 5?", "What is 25 divided by 5?"),
    ]

    columns = st.columns(3)

    for index, (label, message) in enumerate(suggestions):
        with columns[index]:
            if st.button(
                label,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state["suggested_prompt"] = message


for message in st.session_state.messages:
    is_user = message["role"] == "user"
    role_class = "user" if is_user else "assistant"
    label = "You" if is_user else "AI Assistant"
    avatar = "👤" if is_user else "🤖"

    content = html.escape(str(message["content"])).replace("\n", "<br>")

    st.markdown(
        f"""
        <div class="chat-row chat-row--{role_class}">
            <div class="chat-avatar">{avatar}</div>
            <div class="chat-body">
                <div class="chat-label">{label}</div>
                <div class="chat-content">{content}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


prompt = st.chat_input("Type your message...")

if prompt is None:
    prompt = st.session_state.pop("suggested_prompt", None)

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

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
