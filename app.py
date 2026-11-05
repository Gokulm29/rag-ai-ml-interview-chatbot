import re
import streamlit as st
from bson import ObjectId

from database import (
    connect_database,
    create_conversation,
    save_message,
    get_all_conversations,
    get_conversation_messages,
    delete_conversation,
    get_collections,
)

from rag_engine import RAGEngine, PDF_PATH

from llm import (
    generate_answer,
    rewrite_followup_question,
    generate_mock_question,
    evaluate_mock_answer,
    generate_mock_summary,
    repair_markdown_tables,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Interview AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "messages": [],
    "active_chat_id": None,
    "chat_initialized": False,
    "show_sources": True,
    "app_mode": "Chat Mode",
    "theme": "Light",
    "chat_search": "",
    "open_chat_menu_id": None,
    "mock_active": False,
    "mock_difficulty": "Beginner",
    "mock_target": 5,
    "mock_results": [],
    "mock_questions": [],
    "mock_current": None,
    "mock_last_evaluation": None,
    "mock_summary": None,
    "mock_show_summary": False,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# THEME COLORS
# =========================================================

DARK_MODE = st.session_state.theme == "Dark"

if DARK_MODE:
    C = {
        "bg": "#171326",
        "sidebar": "#211932",
        "surface": "#2A2140",
        "surface2": "#332849",
        "text": "#F5F1FF",
        "muted": "#B8AFCA",
        "border": "#463858",
        "purple": "#A77AFF",
        "cyan": "#38D6CB",
        "input": "#29213B",
    }
else:
    C = {
        "bg": "#F7F5FF",
        "sidebar": "#FFFFFF",
        "surface": "#FFFFFF",
        "surface2": "#F2EDFF",
        "text": "#28213D",
        "muted": "#77718B",
        "border": "#E6DEF2",
        "purple": "#7439D8",
        "cyan": "#21BDB6",
        "input": "#FFFFFF",
    }


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
    <style>

    :root {{
        --app-bg: {C["bg"]};
        --sidebar-bg: {C["sidebar"]};
        --surface: {C["surface"]};
        --surface2: {C["surface2"]};
        --app-text: {C["text"]};
        --app-muted: {C["muted"]};
        --app-border: {C["border"]};
        --app-purple: {C["purple"]};
        --app-cyan: {C["cyan"]};
        --app-input: {C["input"]};
    }}

    html, body, .stApp {{
        background: var(--app-bg) !important;
        color: var(--app-text) !important;
    }}

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{
        background: var(--app-bg) !important;
    }}

    .block-container {{
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
    }}

    /* SIDEBAR */

    [data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--app-border) !important;
    }}

    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] section {{
        background: var(--sidebar-bg) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {{
        color: var(--app-text) !important;
    }}

    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: var(--app-muted) !important;
    }}

    /* BRAND */

    .brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 8px 0 25px 0;
    }}

    .brand-icon {{
        width: 36px;
        height: 36px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white !important;
        font-size: 21px;
        background: linear-gradient(135deg, #8046E5, #20C6BD);
        flex-shrink: 0;
    }}

    .brand-title {{
        font-size: 17px;
        font-weight: 750;
        color: var(--app-text) !important;
    }}

    .section-label {{
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.25px;
        text-transform: uppercase;
        color: var(--app-muted) !important;
        margin-top: 20px;
        margin-bottom: 10px;
    }}

    /* BUTTONS */

    .stButton button {{
        border-radius: 11px !important;
        min-height: 39px;
        transition: all .18s ease;
    }}

    .stButton button:hover {{
        border-color: var(--app-purple) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] button[kind="secondary"] {{
        background: var(--surface2) !important;
        color: var(--app-text) !important;
        border: 1px solid var(--app-border) !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *,
    [data-testid="stSidebar"] button[kind="secondary"] * {{
        color: var(--app-text) !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] button svg {{
        color: var(--app-text) !important;
        stroke: var(--app-text) !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] button[kind="primary"] {{
        background: linear-gradient(
            105deg,
            #7541D8,
            #9654E8,
            #21BDB6
        ) !important;
        color: white !important;
        border: none !important;
        font-weight: 650 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] *,
    [data-testid="stSidebar"] button[kind="primary"] * {{
        color: white !important;
    }}

    /* SEARCH */

    [data-testid="stSidebar"] input {{
        background: var(--surface) !important;
        color: var(--app-text) !important;
        border-color: var(--app-border) !important;
        border-radius: 11px !important;
    }}

    [data-testid="stSidebar"] input::placeholder {{
        color: var(--app-muted) !important;
        opacity: 1 !important;
    }}

    /* CHAT HISTORY ALIGNMENT - FIXED */

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
        display: flex !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 6px !important;
        width: 100% !important;
    }}

    [data-testid="stSidebar"] [data-testid="column"] {{
        min-width: 0 !important;
        flex-shrink: 1 !important;
    }}

    /* Title column */

    [data-testid="stSidebar"] div[class*="st-key-chat_title_"] {{
        min-width: 0 !important;
        width: 100% !important;
    }}

    [data-testid="stSidebar"] div[class*="st-key-chat_title_"] button {{
        width: 100% !important;
        min-width: 0 !important;
        height: 42px !important;
        padding: 0 12px !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        text-align: left !important;
        justify-content: flex-start !important;
        background: var(--surface2) !important;
        color: var(--app-text) !important;
        border: 1px solid var(--app-border) !important;
    }}

    [data-testid="stSidebar"] div[class*="st-key-chat_title_"] button p {{
        color: var(--app-text) !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        text-align: left !important;
        margin: 0 !important;
        width: 100% !important;
    }}

    /* Pin and menu fixed-width columns */

    [data-testid="stSidebar"] div[class*="st-key-pin_"],
    [data-testid="stSidebar"] div[class*="st-key-menu_"] {{
        flex: 0 0 40px !important;
        width: 40px !important;
        min-width: 40px !important;
        max-width: 40px !important;
    }}

    [data-testid="stSidebar"] div[class*="st-key-pin_"] button,
    [data-testid="stSidebar"] div[class*="st-key-menu_"] button {{
        width: 40px !important;
        min-width: 40px !important;
        max-width: 40px !important;
        height: 42px !important;
        min-height: 42px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 10px !important;
        background: var(--surface2) !important;
        border: 1px solid var(--app-border) !important;
    }}

    [data-testid="stSidebar"] div[class*="st-key-pin_"] button svg,
    [data-testid="stSidebar"] div[class*="st-key-menu_"] button svg {{
        width: 17px !important;
        height: 17px !important;
        color: var(--app-text) !important;
        stroke: var(--app-text) !important;
    }}

    /* MENU OPTIONS */

    [data-testid="stSidebar"] div[class*="st-key-rename_"] button,
    [data-testid="stSidebar"] div[class*="st-key-delete_"] button {{
        font-size: 12px !important;
        min-height: 36px !important;
    }}

    /* WELCOME SCREEN */

    .welcome-icon {{
        width: 68px;
        height: 68px;
        border-radius: 22px;
        margin: 10px auto 20px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white !important;
        font-size: 32px;
        background: linear-gradient(135deg, #8046E5, #20C6BD);
    }}

    .welcome-title {{
        text-align: center;
        font-size: clamp(25px, 4vw, 36px);
        font-weight: 800;
        letter-spacing: -1.2px;
        color: var(--app-text) !important;
        margin-bottom: 10px;
    }}

    .gradient-text {{
        background: linear-gradient(100deg, #8950E7, #20BDB6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .welcome-subtitle {{
        text-align: center;
        color: var(--app-muted) !important;
        font-size: 14px;
        margin-bottom: 28px;
    }}

    /* SUGGESTION CARDS */

    .st-key-suggestion_0 button {{
        background: linear-gradient(100deg, rgba(151,108,245,.22), rgba(205,176,255,.12)) !important;
    }}

    .st-key-suggestion_1 button {{
        background: linear-gradient(100deg, rgba(213,123,225,.20), rgba(248,185,220,.12)) !important;
    }}

    .st-key-suggestion_2 button {{
        background: linear-gradient(100deg, rgba(76,199,190,.20), rgba(172,237,226,.12)) !important;
    }}

    .st-key-suggestion_3 button {{
        background: linear-gradient(100deg, rgba(244,194,111,.20), rgba(255,226,165,.12)) !important;
    }}

    /* CHAT MESSAGES */

    [data-testid="stChatMessage"] {{
        background: transparent !important;
        border: none !important;
        padding: 12px 5px;
    }}

    [data-testid="stChatMessageContent"] {{
        color: var(--app-text) !important;
        font-size: 14px;
        line-height: 1.8;
        overflow-wrap: anywhere;
    }}

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessageContent"] h3 {{
        color: var(--app-text) !important;
    }}

    /* CHAT INPUT */

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {{
        background: var(--app-bg) !important;
    }}

    [data-testid="stBottom"] {{
        border-top: 1px solid var(--app-border) !important;
    }}

    [data-testid="stBottomBlockContainer"] {{
        padding-top: 12px !important;
        padding-bottom: 18px !important;
    }}

    [data-testid="stChatInput"] {{
        background: var(--app-input) !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 18px !important;
        overflow: hidden;
        box-shadow: 0 5px 20px rgba(80,50,130,.07);
    }}

    [data-testid="stChatInput"] > div {{
        background: var(--app-input) !important;
    }}

    [data-testid="stChatInput"] textarea {{
        background: var(--app-input) !important;
        color: var(--app-text) !important;
        caret-color: var(--app-text) !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--app-muted) !important;
    }}

    [data-testid="stChatInput"] button {{
        background: var(--surface2) !important;
        color: var(--app-text) !important;
    }}

    [data-testid="stChatInput"] button svg {{
        color: var(--app-text) !important;
    }}

    /* HEADER */

    [data-testid="stHeader"] {{
        background: var(--app-bg) !important;
    }}

    /* EXPANDER */

    [data-testid="stExpander"] {{
        background: var(--surface) !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 12px !important;
    }}

    [data-testid="stExpander"] summary {{
        color: var(--app-text) !important;
    }}

    [data-testid="stMetric"] {{
        background: var(--surface) !important;
        padding: 15px;
        border: 1px solid var(--app-border);
        border-radius: 14px;
    }}

    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--app-text) !important;
    }}

    hr {{
        border-color: var(--app-border) !important;
    }}

    @media(max-width: 768px) {{

        .block-container {{
            padding: 1rem .8rem 5rem .8rem;
        }}

        [data-testid="stSidebar"] div[class*="st-key-pin_"],
        [data-testid="stSidebar"] div[class*="st-key-menu_"] {{
            flex: 0 0 34px !important;
            width: 34px !important;
            min-width: 34px !important;
            max-width: 34px !important;
        }}

        [data-testid="stSidebar"] div[class*="st-key-pin_"] button,
        [data-testid="stSidebar"] div[class*="st-key-menu_"] button {{
            width: 34px !important;
            min-width: 34px !important;
            max-width: 34px !important;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True,
)
# =========================================================
# DATABASE CONNECTION
# =========================================================

@st.cache_resource(show_spinner=False)
def initialize_database():
    return connect_database()


try:
    initialize_database()

except Exception as error:
    st.error(f"MongoDB connection failed: {error}")
    st.stop()


# =========================================================
# RAG ENGINE
# =========================================================

@st.cache_resource(show_spinner=False)
def load_rag_engine():
    engine = RAGEngine()
    engine.build_index(PDF_PATH)
    return engine


try:
    with st.spinner("Preparing your knowledge base..."):
        engine = load_rag_engine()

except Exception as error:
    st.error(f"Knowledge base initialization failed: {error}")
    st.stop()


# =========================================================
# CHAT HELPER FUNCTIONS
# =========================================================

def make_chat_title(question, max_words=6):

    title = re.sub(r"\s+", " ", str(question)).strip()
    title = re.sub(r"[?!.]+$", "", title)

    title = re.sub(
        r"^(please\s+)?(can you\s+|could you\s+)?",
        "",
        title,
        flags=re.IGNORECASE,
    )

    title = re.sub(
        r"^(explain|describe|tell me about|what is|what are|define)\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )

    words = title.split()

    if len(words) > max_words:
        title = " ".join(words[:max_words]) + "…"

    if title:
        title = title[0].upper() + title[1:]

    return title or "New Chat"


def update_chat_title(chat_id, title):

    conversations, _ = get_collections()

    conversations.update_one(
        {"_id": ObjectId(chat_id)},
        {"$set": {"title": title}},
    )


def update_pin_status(chat_id, pinned):

    conversations, _ = get_collections()

    conversations.update_one(
        {"_id": ObjectId(chat_id)},
        {"$set": {"pinned": pinned}},
    )


def load_chat(chat_id):

    stored_messages = get_conversation_messages(chat_id)

    st.session_state.active_chat_id = chat_id

    st.session_state.messages = [
        {
            "role": item.get("role"),
            "content": item.get("content", ""),
            "sources": item.get("sources", []),
        }
        for item in stored_messages
        if item.get("role") in {"user", "assistant"}
    ]

    st.session_state.open_chat_menu_id = None
    st.session_state.pop("pending_question", None)


def clean_answer(answer):

    if not isinstance(answer, str):
        return str(answer)

    answer = re.sub(
        r"<br\s*/?>",
        "\n",
        answer,
        flags=re.IGNORECASE,
    )

    answer = re.sub(
        r"</?(?:p|div)\b[^>]*>",
        "\n",
        answer,
        flags=re.IGNORECASE,
    )

    answer = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: "$$\n" + m.group(1).strip() + "\n$$",
        answer,
        flags=re.DOTALL,
    )

    answer = re.sub(
        r"\\\((.*?)\\\)",
        lambda m: "$" + m.group(1).strip() + "$",
        answer,
        flags=re.DOTALL,
    )

    answer = answer.replace("\r\n", "\n")
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    return answer.strip()


def render_answer(answer):
    st.markdown(clean_answer(answer))


def get_source_details(chunks):

    sources = []

    for chunk in chunks:

        if isinstance(chunk, dict):

            sources.append({
                "source": chunk.get("source", "Unknown source"),
                "page": chunk.get("page", "Unknown page"),
                "score": chunk.get("score"),
            })

        else:

            sources.append({
                "source": "Knowledge base",
                "page": "Unknown page",
                "score": None,
            })

    return sources


def display_sources(sources):

    if not sources:
        return

    with st.expander("📄 View retrieved sources"):

        for source in sources:

            st.markdown(
                f"**{source.get('source', 'Unknown source')}** · "
                f"Page {source.get('page', 'Unknown page')}"
            )

            score = source.get("score")

            if score is not None:
                st.caption(f"Similarity: {score:.4f}")


# =========================================================
# MOCK INTERVIEW RESET
# =========================================================

def reset_mock():

    st.session_state.mock_active = False
    st.session_state.mock_difficulty = "Beginner"
    st.session_state.mock_target = 5
    st.session_state.mock_results = []
    st.session_state.mock_questions = []
    st.session_state.mock_current = None
    st.session_state.mock_last_evaluation = None
    st.session_state.mock_summary = None
    st.session_state.mock_show_summary = False


# =========================================================
# RENAME DIALOG
# =========================================================

@st.dialog("Rename conversation")
def rename_chat_dialog():

    chat_id = st.session_state.rename_target_id
    current_title = st.session_state.rename_target_title

    new_title = st.text_input(
        "Conversation name",
        value=current_title,
        max_chars=60,
        key=f"rename_input_{chat_id}",
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Save name",
            type="primary",
            use_container_width=True,
        ):

            if not new_title.strip():

                st.warning("Chat name cannot be empty.")

            else:

                try:

                    update_chat_title(
                        chat_id,
                        new_title.strip(),
                    )

                    st.session_state.pop("rename_target_id", None)
                    st.session_state.pop("rename_target_title", None)

                    st.rerun()

                except Exception as error:

                    st.error(f"Rename failed: {error}")

    with col2:

        if st.button(
            "Cancel",
            use_container_width=True,
        ):

            st.session_state.pop("rename_target_id", None)
            st.session_state.pop("rename_target_title", None)

            st.rerun()


# =========================================================
# DELETE DIALOG
# =========================================================

@st.dialog("Delete conversation")
def delete_chat_dialog():

    chat_id = st.session_state.delete_target_id
    chat_title = st.session_state.delete_target_title

    st.write(f'Delete "{chat_title}"?')

    st.caption(
        "This permanently deletes the conversation and its messages."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Delete",
            type="primary",
            use_container_width=True,
        ):

            try:

                delete_conversation(chat_id)

                if st.session_state.active_chat_id == chat_id:

                    st.session_state.active_chat_id = None
                    st.session_state.messages = []

                st.session_state.pop("delete_target_id", None)
                st.session_state.pop("delete_target_title", None)

                st.rerun()

            except Exception as error:

                st.error(f"Delete failed: {error}")

    with col2:

        if st.button(
            "Cancel",
            use_container_width=True,
        ):

            st.session_state.pop("delete_target_id", None)
            st.session_state.pop("delete_target_title", None)

            st.rerun()


# =========================================================
# INITIAL CHAT LOAD
# =========================================================

if not st.session_state.chat_initialized:

    try:

        conversations = get_all_conversations()

        if conversations:
            load_chat(str(conversations[0]["_id"]))

        st.session_state.chat_initialized = True

    except Exception as error:

        st.error(f"Could not load previous chats: {error}")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✳</div>
            <div class="brand-title">Interview AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # NEW CHAT

    if st.button(
        "＋  New chat",
        key="new_chat_button",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.active_chat_id = None
        st.session_state.open_chat_menu_id = None

        st.session_state.pop("pending_question", None)

        st.rerun()

    # SEARCH CHATS

    st.markdown(
        '<div class="section-label">Search chats</div>',
        unsafe_allow_html=True,
    )

    search_text = st.text_input(
        "Search",
        placeholder="⌕  Search conversations...",
        label_visibility="collapsed",
        key="chat_search",
    )

    try:

        saved_conversations = get_all_conversations()

    except Exception as error:

        saved_conversations = []
        st.warning(f"History loading failed: {error}")

    if search_text:

        saved_conversations = [
            chat for chat in saved_conversations
            if search_text.lower() in chat.get("title", "").lower()
        ]

    pinned_chats = [
        chat for chat in saved_conversations
        if chat.get("pinned", False)
    ]

    recent_chats = [
        chat for chat in saved_conversations
        if not chat.get("pinned", False)
    ]

    # =====================================================
    # CHAT ITEM
    # =====================================================

    def render_chat_item(chat):

        chat_id = str(chat["_id"])

        chat_title = chat.get("title", "New Chat") or "New Chat"

        is_pinned = chat.get("pinned", False)

        is_active = (
            chat_id == st.session_state.active_chat_id
        )

        short_title = (
            chat_title
            if len(chat_title) <= 23
            else chat_title[:23].rstrip() + "…"
        )

        title_col, pin_col, menu_col = st.columns(
            [1, 0.15, 0.15],
            vertical_alignment="center",
            gap="small",
        )

        # CHAT TITLE

        with title_col:

            if st.button(
                ("● " if is_active else "") + short_title,
                key=f"chat_title_{chat_id}",
                use_container_width=True,
                type="secondary",
            ):

                try:

                    load_chat(chat_id)
                    st.rerun()

                except Exception as error:

                    st.error(f"Could not open chat: {error}")

        # PIN BUTTON

        with pin_col:

            pin_icon = (
                ":material/push_pin:"
                if is_pinned
                else ":material/keep:"
            )

            if st.button(
                " ",
                key=f"pin_{chat_id}",
                icon=pin_icon,
                help="Unpin chat" if is_pinned else "Pin chat",
                use_container_width=True,
                type="secondary",
            ):

                try:

                    update_pin_status(
                        chat_id,
                        not is_pinned,
                    )

                    st.rerun()

                except Exception as error:

                    st.error(f"Pin update failed: {error}")

        # THREE DOT BUTTON

        with menu_col:

            if st.button(
                " ",
                key=f"menu_{chat_id}",
                icon=":material/more_horiz:",
                use_container_width=True,
                type="secondary",
            ):

                if st.session_state.open_chat_menu_id == chat_id:

                    st.session_state.open_chat_menu_id = None

                else:

                    st.session_state.open_chat_menu_id = chat_id

                st.rerun()

        # RENAME / DELETE OPTIONS

        if st.session_state.open_chat_menu_id == chat_id:

            option_col1, option_col2 = st.columns(2)

            with option_col1:

                if st.button(
                    "Rename",
                    icon=":material/edit:",
                    key=f"rename_{chat_id}",
                    use_container_width=True,
                ):

                    st.session_state.rename_target_id = chat_id
                    st.session_state.rename_target_title = chat_title
                    st.session_state.open_chat_menu_id = None

                    st.rerun()

            with option_col2:

                if st.button(
                    "Delete",
                    icon=":material/delete:",
                    key=f"delete_{chat_id}",
                    use_container_width=True,
                ):

                    st.session_state.delete_target_id = chat_id
                    st.session_state.delete_target_title = chat_title
                    st.session_state.open_chat_menu_id = None

                    st.rerun()

    # =====================================================
    # PINNED CHATS
    # =====================================================

    st.markdown(
        '<div class="section-label">📌 Pinned</div>',
        unsafe_allow_html=True,
    )

    if pinned_chats:

        for chat in pinned_chats:
            render_chat_item(chat)

    else:

        st.caption("No pinned conversations")

    # =====================================================
    # RECENT CHATS
    # =====================================================

    st.markdown(
        '<div class="section-label">Recent</div>',
        unsafe_allow_html=True,
    )

    if recent_chats:

        for chat in recent_chats:
            render_chat_item(chat)

    else:

        st.caption("No recent conversations")

    st.divider()

    # =====================================================
    # PRACTICE MODE
    # =====================================================

    st.markdown(
        '<div class="section-label">Practice mode</div>',
        unsafe_allow_html=True,
    )

    selected_mode = st.radio(
        "Mode",
        ["Chat Mode", "Mock Interview Mode"],
        index=(
            0 if st.session_state.app_mode == "Chat Mode" else 1
        ),
        label_visibility="collapsed",
    )

    if selected_mode != st.session_state.app_mode:

        st.session_state.app_mode = selected_mode
        st.rerun()

    st.divider()

    # =====================================================
    # KNOWLEDGE BASE
    # =====================================================

    st.markdown(
        '<div class="section-label">Knowledge base</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"**Document:** `{PDF_PATH.name}`")

    st.caption(f"{len(engine.chunks)} text chunks indexed")

    st.session_state.show_sources = st.toggle(
        "Show retrieved sources",
        value=st.session_state.show_sources,
    )

    st.divider()

    # =====================================================
    # APPEARANCE
    # =====================================================

    st.markdown(
        '<div class="section-label">Appearance</div>',
        unsafe_allow_html=True,
    )

    selected_theme = st.radio(
        "Theme",
        ["Light", "Dark"],
        index=(
            0 if st.session_state.theme == "Light" else 1
        ),
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_theme != st.session_state.theme:

        st.session_state.theme = selected_theme
        st.rerun()

    st.divider()

    # =====================================================
    # SETTINGS
    # =====================================================

    with st.expander("Settings & controls"):

        if st.button(
            "Clear current chat",
            icon=":material/delete_sweep:",
            use_container_width=True,
        ):

            current_chat_id = st.session_state.active_chat_id

            if current_chat_id:

                try:

                    delete_conversation(current_chat_id)

                except Exception as error:

                    st.error(f"Could not clear chat: {error}")
                    st.stop()

            st.session_state.messages = []
            st.session_state.active_chat_id = None
            st.session_state.open_chat_menu_id = None

            st.rerun()

        if st.button(
            "Reload knowledge base",
            icon=":material/refresh:",
            use_container_width=True,
        ):

            load_rag_engine.clear()
            st.rerun()

        if st.session_state.mock_active:

            if st.button(
                "End mock interview",
                icon=":material/stop_circle:",
                use_container_width=True,
            ):

                reset_mock()
                st.rerun()

    st.caption("Python · Streamlit · FAISS · Groq")


# =========================================================
# DIALOGS
# =========================================================

if st.session_state.get("rename_target_id"):
    rename_chat_dialog()

if st.session_state.get("delete_target_id"):
    delete_chat_dialog()
    # =========================================================
# MAIN HEADER
# =========================================================

if st.session_state.app_mode == "Chat Mode":

    if st.session_state.messages:

        current_title = "Interview AI"

        if st.session_state.active_chat_id:

            try:

                conversations = get_all_conversations()

                for conversation in conversations:

                    if (
                        str(conversation["_id"])
                        == st.session_state.active_chat_id
                    ):

                        current_title = conversation.get(
                            "title",
                            "Interview AI",
                        )

                        break

            except Exception:
                pass

        st.markdown(
            f"""
            <div style="
                font-size:19px;
                font-weight:700;
                color:{C["text"]};
                margin-bottom:20px;
            ">
                {current_title}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="welcome-icon">✳</div>

            <div class="welcome-title">
                What shall we <span class="gradient-text">learn</span> today?
            </div>

            <div class="welcome-subtitle">
                Your AI/ML interview companion.
                Ask anything or start with a suggestion.
            </div>
            """,
            unsafe_allow_html=True,
        )

elif st.session_state.app_mode == "Mock Interview Mode":

    st.markdown(
        """
        <div class="welcome-title">
            Mock <span class="gradient-text">Interview</span>
        </div>

        <div class="welcome-subtitle">
            Practice technical questions and improve your interview skills.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# NORMAL CHAT MODE
# =========================================================

if st.session_state.app_mode == "Chat Mode":

    # -----------------------------------------------------
    # SUGGESTION CARDS
    # -----------------------------------------------------

    if not st.session_state.messages:

        suggestions = [
            (
                "Linear vs Logistic Regression",
                "What is the difference between linear and logistic regression?",
            ),
            (
                "Classification Metrics",
                "Explain precision, recall, F1-score, and accuracy.",
            ),
            (
                "Feature Scaling",
                "Why is feature scaling important in machine learning?",
            ),
            (
                "RAG Architecture",
                "Explain how a RAG chatbot works.",
            ),
        ]

        for index in range(0, 4, 2):

            col1, col2 = st.columns(2)

            for column, suggestion_index in zip(
                [col1, col2],
                [index, index + 1],
            ):

                title, prompt = suggestions[suggestion_index]

                with column:

                    if st.button(
                        title,
                        key=f"suggestion_{suggestion_index}",
                        use_container_width=True,
                    ):

                        st.session_state.pending_question = prompt
                        st.rerun()

    # -----------------------------------------------------
    # DISPLAY CHAT HISTORY
    # -----------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            if message["role"] == "assistant":

                render_answer(message["content"])

            else:

                st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and st.session_state.show_sources
                and message.get("sources")
            ):

                display_sources(message["sources"])

    # -----------------------------------------------------
    # CHAT INPUT
    # -----------------------------------------------------

    pending_question = st.session_state.pop(
        "pending_question",
        None,
    )

    question = st.chat_input("Message Interview AI...")

    if pending_question:
        question = pending_question

    # -----------------------------------------------------
    # PROCESS USER QUESTION
    # -----------------------------------------------------

    if question:

        question = question.strip()

        if question:

            conversation_history = [
                {
                    "role": item["role"],
                    "content": item["content"],
                }
                for item in st.session_state.messages[-6:]
                if item.get("role") in {"user", "assistant"}
            ]

            # CREATE NEW CONVERSATION

            if not st.session_state.active_chat_id:

                try:

                    title = make_chat_title(question)

                    st.session_state.active_chat_id = (
                        create_conversation(title)
                    )

                except Exception as error:

                    st.error(
                        f"Could not create conversation: {error}"
                    )

                    st.stop()

            chat_id = st.session_state.active_chat_id

            # SAVE USER MESSAGE

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            try:

                save_message(
                    chat_id,
                    "user",
                    question,
                )

            except Exception as error:

                st.error(
                    f"Could not save question: {error}"
                )

            with st.chat_message("user"):

                st.markdown(question)

            # GENERATE ASSISTANT RESPONSE

            with st.chat_message("assistant"):

                try:

                    with st.spinner(
                        "Understanding your question and searching your notes..."
                    ):

                        retrieval_question = rewrite_followup_question(
                            question,
                            conversation_history,
                        )

                        chunks = engine.retrieve(
                            retrieval_question
                        )

                        answer = clean_answer(
                            generate_answer(
                                question,
                                chunks,
                                conversation_history=conversation_history,
                            )
                        )

                        # Repair malformed comparison tables before display and saving.
                        answer = repair_markdown_tables(answer)

                    render_answer(answer)

                    sources = get_source_details(chunks)

                    if st.session_state.show_sources and sources:

                        display_sources(sources)

                    # SAVE ASSISTANT RESPONSE

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                    try:

                        save_message(
                            chat_id,
                            "assistant",
                            answer,
                            sources,
                        )

                    except Exception as error:

                        st.error(
                            f"Could not save assistant response: {error}"
                        )

                except Exception as error:

                    st.error(
                        f"Answer generation failed: "
                        f"{type(error).__name__}: {error}"
                    )


# =========================================================
# MOCK INTERVIEW MODE
# =========================================================

elif st.session_state.app_mode == "Mock Interview Mode":

    # -----------------------------------------------------
    # START INTERVIEW
    # -----------------------------------------------------

    if not st.session_state.mock_active:

        with st.container(border=True):

            st.subheader("Set up your practice session")

            st.caption(
                "Questions are generated from your indexed interview notes."
            )

            difficulty = st.selectbox(
                "Difficulty level",
                ["Beginner", "Intermediate", "Advanced"],
                index=0,
            )

            target = st.select_slider(
                "Number of questions",
                options=[3, 5, 7, 10],
                value=5,
            )

            if st.button(
                "Start mock interview",
                type="primary",
                use_container_width=True,
            ):

                st.session_state.mock_active = True
                st.session_state.mock_difficulty = difficulty
                st.session_state.mock_target = target

                st.session_state.mock_results = []
                st.session_state.mock_questions = []

                st.session_state.mock_current = None
                st.session_state.mock_last_evaluation = None

                st.session_state.mock_summary = None
                st.session_state.mock_show_summary = False

                st.rerun()

    # -----------------------------------------------------
    # ACTIVE INTERVIEW
    # -----------------------------------------------------

    if st.session_state.mock_active:

        results = st.session_state.mock_results

        completed = len(results)

        target = st.session_state.mock_target

        st.progress(
            min(completed / target, 1.0),
            text=f"Progress · {completed} of {target} questions",
        )

        # SCORE METRICS

        if completed:

            average = (
                sum(item["score"] for item in results)
                / len(results)
            )

            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Questions evaluated",
                completed,
            )

            metric2.metric(
                "Average score",
                f"{average:.1f}/10",
            )

        # -------------------------------------------------
        # GENERATE NEXT QUESTION
        # -------------------------------------------------

        if (
            completed < target
            and st.session_state.mock_current is None
            and st.session_state.mock_last_evaluation is None
            and not st.session_state.mock_show_summary
        ):

            topics = [
                "Python programming",
                "SQL and databases",
                "machine learning algorithms",
                "deep learning and neural networks",
                "NLP and generative AI",
                "RAG, embeddings, and vector databases",
                "AI agents and tool calling",
                "model evaluation and feature engineering",
            ]

            topic = topics[
                len(st.session_state.mock_questions)
                % len(topics)
            ]

            try:

                with st.spinner(
                    f"Preparing a question about {topic}..."
                ):

                    chunks = engine.retrieve(topic)

                    if not chunks:

                        chunks = engine.retrieve(
                            "AI ML technical interview questions"
                        )

                    new_question = generate_mock_question(
                        chunks,
                        difficulty=st.session_state.mock_difficulty,
                        previous_questions=st.session_state.mock_questions,
                    )

                    st.session_state.mock_current = {
                        "question": new_question,
                        "chunks": chunks,
                        "sources": get_source_details(chunks),
                        "topic": topic,
                    }

                    st.session_state.mock_questions.append(
                        new_question
                    )

            except Exception as error:

                st.error(
                    f"Could not generate a question: "
                    f"{type(error).__name__}: {error}"
                )

                st.stop()

        # -------------------------------------------------
        # DISPLAY EVALUATION
        # -------------------------------------------------

        if st.session_state.mock_last_evaluation is not None:

            evaluation = st.session_state.mock_last_evaluation

            st.markdown("### Your evaluation")

            st.markdown(
                f"**Question:** {evaluation['question']}"
            )

            score_col, verdict_col = st.columns(2)

            score_col.metric(
                "Your score",
                f"{evaluation['score']}/10",
            )

            verdict_col.metric(
                "Assessment",
                evaluation.get("verdict", "Evaluated"),
            )

            st.markdown("#### Feedback")

            st.write(
                evaluation.get("feedback", "")
            )

            with st.expander("Strengths", expanded=True):

                for item in evaluation.get("strengths", []):

                    st.markdown(f"- {item}")

            with st.expander("Areas to improve", expanded=True):

                for item in evaluation.get("improvements", []):

                    st.markdown(f"- {item}")

            with st.expander("Sample interview answer"):

                render_answer(
                    evaluation.get(
                        "sample_answer",
                        "No sample answer was provided.",
                    )
                )

            if st.session_state.show_sources:

                display_sources(
                    evaluation.get("sources", [])
                )

            button_label = (
                "Next question"
                if completed < target
                else "Finish interview and view summary"
            )

            if st.button(
                button_label,
                type="primary",
                use_container_width=True,
            ):

                st.session_state.mock_last_evaluation = None

                st.session_state.mock_current = None

                if completed >= target:

                    st.session_state.mock_show_summary = True

                st.rerun()

        # -------------------------------------------------
        # FINAL INTERVIEW SUMMARY
        # -------------------------------------------------

        elif st.session_state.mock_show_summary:

            st.subheader("Interview summary")

            results = st.session_state.mock_results

            scores = [
                item["score"]
                for item in results
            ]

            total_score = sum(scores)

            maximum_score = len(scores) * 10

            average = (
                total_score / len(scores)
                if scores
                else 0
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Questions",
                len(results),
            )

            c2.metric(
                "Total score",
                f"{total_score}/{maximum_score}",
            )

            c3.metric(
                "Average score",
                f"{average:.1f}/10",
            )

            st.markdown("### Question-wise results")

            for index, item in enumerate(
                results,
                start=1,
            ):

                with st.expander(
                    f"Q{index}: {item['question']} — {item['score']}/10"
                ):

                    st.markdown("**Your answer**")

                    st.write(
                        item["candidate_answer"]
                    )

                    st.markdown("**Feedback**")

                    st.write(
                        item.get("feedback", "")
                    )

            # GENERATE SUMMARY

            if st.session_state.mock_summary is None:

                try:

                    with st.spinner(
                        "Preparing your interview feedback..."
                    ):

                        st.session_state.mock_summary = (
                            generate_mock_summary(results)
                        )

                except Exception as error:

                    st.session_state.mock_summary = (
                        "The overall summary could not be generated. "
                        f"Error: {type(error).__name__}: {error}"
                    )

            st.markdown("### Interview coach feedback")

            render_answer(
                st.session_state.mock_summary
            )

            if st.button(
                "Start another interview",
                type="primary",
                use_container_width=True,
            ):

                reset_mock()

                st.rerun()

        # -------------------------------------------------
        # QUESTION AND ANSWER FORM
        # -------------------------------------------------

        elif st.session_state.mock_current is not None:

            current = st.session_state.mock_current

            st.caption(
                f"{st.session_state.mock_difficulty} · "
                f"Question {completed + 1} of {target}"
            )

            st.markdown("### Interview question")

            st.markdown(
                f"#### {current['question']}"
            )

            with st.form(
                key=f"mock_answer_form_{completed}",
                clear_on_submit=False,
            ):

                candidate_answer = st.text_area(
                    "Your answer",
                    height=200,
                    placeholder=(
                        "Explain the concept in your own words. "
                        "Include an example when possible."
                    ),
                    label_visibility="collapsed",
                )

                submitted = st.form_submit_button(
                    "Submit answer for evaluation",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:

                if not candidate_answer.strip():

                    st.warning(
                        "Please enter your answer before submitting."
                    )

                else:

                    try:

                        with st.spinner(
                            "Evaluating your answer..."
                        ):

                            evaluation = evaluate_mock_answer(
                                current["question"],
                                candidate_answer,
                                current["chunks"],
                            )

                        result = {
                            "question": current["question"],
                            "candidate_answer": candidate_answer.strip(),
                            "score": evaluation["score"],
                            "verdict": evaluation.get("verdict", ""),
                            "strengths": evaluation.get("strengths", []),
                            "improvements": evaluation.get("improvements", []),
                            "feedback": evaluation.get("feedback", ""),
                            "sample_answer": evaluation.get("sample_answer", ""),
                            "sources": current["sources"],
                        }

                        st.session_state.mock_results.append(
                            result
                        )

                        st.session_state.mock_last_evaluation = (
                            result
                        )

                        st.rerun()

                    except Exception as error:

                        st.error(
                            f"Evaluation failed: "
                            f"{type(error).__name__}: {error}"
<<<<<<< HEAD
                        )

st.markdown(
    """
    <style>
    .copyright-footer {
        text-align: center;
        color: #8b95a5;
        font-size: 12px;
        padding: 15px 0 10px 0;
        margin-top: 25px;
        border-top: 1px solid #30343b;
    }
    .copyright-footer a {
        color: #60a5fa;
        text-decoration: none;
    }
    .copyright-footer a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="copyright-footer">
        © 2026 <strong>Gokul M.</strong> All rights reserved.
        <br>
        Built with Python, Streamlit &amp; Generative AI.
        <br>
        <a href="https://github.com/Gokulm29/rag-ai-ml-interview-chatbot"
           target="_blank">View Source on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

=======
                        )
>>>>>>> affeed9 (Update chatbot features and fix testing issues)
