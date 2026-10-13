import re
import streamlit as st

from rag_engine import RAGEngine, PDF_PATH
from llm import (
    generate_answer,
    generate_mock_question,
    evaluate_mock_answer,
    generate_mock_summary,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI/ML Interview Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CHATGPT-INSPIRED UI
# ==================================================

st.markdown(
    """
    <style>
    :root {
        --chat-border: rgba(128, 128, 128, 0.22);
        --chat-muted: rgba(128, 128, 128, 0.82);
    }

    .block-container {
        max-width: 980px;
        padding-top: 1.4rem;
        padding-bottom: 6rem;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--background-color);
    }

    [data-testid="stSidebar"] {
        background: var(--secondary-background-color);
        border-right: 1px solid var(--chat-border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.5;
    }

    .app-brand {
        font-size: 1.12rem;
        font-weight: 700;
        letter-spacing: -0.35px;
        margin-bottom: 0.15rem;
    }

    .app-tagline {
        color: var(--chat-muted);
        font-size: 0.84rem;
        margin-bottom: 1rem;
    }

    .main-header {
        padding: 0.25rem 0 1.15rem 0;
        margin-bottom: 0.4rem;
    }

    .main-header h1 {
        font-size: clamp(1.65rem, 4vw, 2.15rem);
        letter-spacing: -0.8px;
        line-height: 1.25;
        margin-bottom: 0.45rem;
    }

    .main-header p {
        color: var(--chat-muted);
        font-size: 0.96rem;
        line-height: 1.55;
        margin-bottom: 0;
    }

    .welcome {
        text-align: center;
        padding: 2.6rem 0.8rem 1.5rem 0.8rem;
    }

    .welcome-icon {
        font-size: 2.4rem;
        margin-bottom: 0.55rem;
    }

    .welcome h2 {
        font-size: clamp(1.45rem, 4vw, 1.9rem);
        letter-spacing: -0.65px;
        margin-bottom: 0.45rem;
    }

    .welcome p {
        color: var(--chat-muted);
        font-size: 0.96rem;
    }

    .stButton > button {
        min-height: 42px;
        border-radius: 12px;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }

    .stButton > button:hover {
        border-color: rgba(128, 128, 128, 0.7);
    }

    [data-testid="stChatMessage"] {
        background: transparent;
        border: 0;
        padding: 0.85rem 0.2rem;
        gap: 0.8rem;
    }

    [data-testid="stChatMessageContent"] {
        line-height: 1.75;
        font-size: 0.98rem;
        overflow-wrap: anywhere;
    }

    [data-testid="stChatInput"] {
        border: 1px solid var(--chat-border);
        border-radius: 22px;
        overflow: hidden;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.045);
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(128, 128, 128, 0.65);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.07);
    }

    [data-testid="stChatInput"] textarea {
        font-size: 0.98rem;
        line-height: 1.5;
    }

    [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessageContent"] h3 {
        letter-spacing: -0.3px;
        line-height: 1.4;
        margin-top: 1.15rem;
        margin-bottom: 0.55rem;
    }

    [data-testid="stChatMessageContent"] table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
    }

    [data-testid="stChatMessageContent"] th,
    [data-testid="stChatMessageContent"] td {
        padding: 0.6rem;
        border: 1px solid var(--chat-border);
        text-align: left;
    }

    [data-testid="stChatMessageContent"] pre {
        border-radius: 12px;
        padding: 0.9rem;
        overflow-x: auto;
    }

    [data-testid="stExpander"] {
        border-radius: 12px;
        border-color: var(--chat-border);
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    .section-caption {
        color: var(--chat-muted);
        font-size: 0.9rem;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.8rem 5rem 0.8rem;
        }

        [data-testid="stChatMessage"] {
            padding: 0.6rem 0;
        }

        .welcome {
            padding-top: 1.7rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SESSION STATE
# ==================================================

DEFAULTS = {
    "messages": [],
    "show_sources": True,
    "app_mode": "Chat Mode",
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


def reset_mock():
    """Reset all mock interview state."""
    st.session_state.mock_active = False
    st.session_state.mock_difficulty = "Beginner"
    st.session_state.mock_target = 5
    st.session_state.mock_results = []
    st.session_state.mock_questions = []
    st.session_state.mock_current = None
    st.session_state.mock_last_evaluation = None
    st.session_state.mock_summary = None
    st.session_state.mock_show_summary = False


# ==================================================
# ANSWER FORMATTING
# ==================================================

def clean_answer(answer):
    """Normalize common HTML and LaTeX delimiters."""

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

    # Convert display math delimiters: \[...\] -> $$...$$
    answer = re.sub(
        r"\\\[(.*?)\\\]",
        lambda match: "$$\n" + match.group(1).strip() + "\n$$",
        answer,
        flags=re.DOTALL,
    )

    # Convert inline math delimiters: \(...\) -> $...$
    answer = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: "$" + match.group(1).strip() + "$",
        answer,
        flags=re.DOTALL,
    )

    answer = answer.replace("\r\n", "\n")
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def render_answer(answer):
    """Render Markdown and supported math."""
    st.markdown(clean_answer(answer))


# ==================================================
# RAG ENGINE
# ==================================================

@st.cache_resource(show_spinner=False)
def load_rag_engine():
    engine = RAGEngine()
    engine.build_index(PDF_PATH)
    return engine


try:
    with st.spinner("Preparing your interview knowledge base..."):
        engine = load_rag_engine()
except Exception as error:
    st.error(
        f"Knowledge base initialization failed: "
        f"{type(error).__name__}: {error}"
    )
    st.info("Check the PDF path, dependencies, and terminal logs.")
    st.stop()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown('<div class="app-brand">🤖 Interview AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-tagline">Your AI/ML interview companion</div>',
        unsafe_allow_html=True,
    )

    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_question", None)
        st.rerun()

    st.divider()

    st.caption("PRACTICE MODE")
    selected_mode = st.radio(
        "Choose a mode",
        ["Chat Mode", "Mock Interview Mode"],
        index=0 if st.session_state.app_mode == "Chat Mode" else 1,
        label_visibility="collapsed",
    )

    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.rerun()

    st.divider()
    st.caption("KNOWLEDGE BASE")
    st.write(f"**Document:** `{PDF_PATH.name}`")
    st.caption(f"{len(engine.chunks)} text chunks indexed")

    st.session_state.show_sources = st.toggle(
        "Show retrieved sources",
        value=st.session_state.show_sources,
    )

    st.divider()

    with st.expander("Settings & controls"):
        if st.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state.messages = []
            reset_mock()
            st.rerun()

        if st.button("🔄 Reload knowledge base", use_container_width=True):
            load_rag_engine.clear()
            st.rerun()

        if st.session_state.mock_active:
            if st.button("⏹️ End mock interview", use_container_width=True):
                reset_mock()
                st.rerun()

    st.divider()
    st.caption("Python · Streamlit · FAISS · Hugging Face · Groq")


# ==================================================
# HEADER
# ==================================================

if st.session_state.app_mode == "Chat Mode":
    st.markdown(
        """
        <div class="main-header">
            <h1>AI/ML Interview Assistant</h1>
            <p>Ask a question, explore a concept, or prepare for your next technical interview.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="main-header">
            <h1>🎯 Mock Interview</h1>
            <p>Practice one question at a time and get constructive feedback on your answers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# SOURCE HELPERS
# ==================================================

def display_sources(sources):
    if not sources:
        st.caption("No source passages were retrieved.")
        return

    with st.expander("📄 View sources"):
        for source in sources:
            st.markdown(
                f"**{source.get('source', 'Unknown source')}** · "
                f"Page {source.get('page', 'Unknown page')}"
            )
            score = source.get("score")
            if score is not None:
                st.caption(f"Similarity: {score:.4f}")


def get_source_details(chunks):
    return [
        {
            "source": chunk.get("source", "Unknown source"),
            "page": chunk.get("page", "Unknown page"),
            "score": chunk.get("score"),
        }
        for chunk in chunks
    ]


# ==================================================
# NORMAL CHAT MODE
# ==================================================

if st.session_state.app_mode == "Chat Mode":

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome">
                <div class="welcome-icon">✳️</div>
                <h2>What would you like to practice?</h2>
                <p>Choose a prompt below or ask your own AI/ML question.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "↔️  Linear vs Logistic Regression",
                use_container_width=True,
            ):
                st.session_state.pending_question = (
                    "What is the difference between linear and logistic regression?"
                )
                st.rerun()

            if st.button(
                "📊  Classification Metrics",
                use_container_width=True,
            ):
                st.session_state.pending_question = (
                    "Explain precision, recall, F1-score, and accuracy."
                )
                st.rerun()

        with col2:
            if st.button(
                "⚖️  Feature Scaling",
                use_container_width=True,
            ):
                st.session_state.pending_question = (
                    "Why is feature scaling important in machine learning?"
                )
                st.rerun()

            if st.button(
                "🧠  RAG Architecture",
                use_container_width=True,
            ):
                st.session_state.pending_question = (
                    "Explain how a RAG chatbot works."
                )
                st.rerun()

    # Display previous messages.
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

    pending_question = st.session_state.pop("pending_question", None)
    question = st.chat_input("Message Interview AI...")

    if pending_question:
        question = pending_question

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your notes and preparing an answer..."):
                    chunks = engine.retrieve(question)
                    answer = clean_answer(generate_answer(question, chunks))

                render_answer(answer)
                sources = get_source_details(chunks)

                if st.session_state.show_sources and sources:
                    display_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as error:
                st.error(
                    f"Answer generation failed: "
                    f"{type(error).__name__}: {error}"
                )


# ==================================================
# MOCK INTERVIEW MODE
# ==================================================

elif st.session_state.app_mode == "Mock Interview Mode":

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
                "🚀 Start mock interview",
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

    if st.session_state.mock_active:
        results = st.session_state.mock_results
        completed = len(results)
        target = st.session_state.mock_target

        st.progress(
            min(completed / target, 1.0),
            text=f"Progress · {completed} of {target} questions",
        )

        if completed:
            average = sum(item["score"] for item in results) / len(results)
            metric1, metric2 = st.columns(2)
            metric1.metric("Questions evaluated", completed)
            metric2.metric("Average score", f"{average:.1f}/10")

        # Generate next question when required.
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
                len(st.session_state.mock_questions) % len(topics)
            ]

            try:
                with st.spinner(f"Preparing a question about {topic}..."):
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
                    st.session_state.mock_questions.append(new_question)

            except Exception as error:
                st.error(
                    f"Could not generate a question: "
                    f"{type(error).__name__}: {error}"
                )
                st.stop()

        # Show evaluation.
        if st.session_state.mock_last_evaluation is not None:
            evaluation = st.session_state.mock_last_evaluation

            st.markdown("### Your evaluation")
            st.markdown(f"**Question:** {evaluation['question']}")

            score_col, verdict_col = st.columns(2)
            score_col.metric("Your score", f"{evaluation['score']}/10")
            verdict_col.metric(
                "Assessment",
                evaluation.get("verdict", "Evaluated"),
            )

            st.markdown("#### Feedback")
            st.write(evaluation.get("feedback", ""))

            with st.expander("✅ Strengths", expanded=True):
                strengths = evaluation.get("strengths", [])
                if strengths:
                    for item in strengths:
                        st.markdown(f"- {item}")
                else:
                    st.write("Keep practicing and refining your explanation.")

            with st.expander("📈 Areas to improve", expanded=True):
                improvements = evaluation.get("improvements", [])
                if improvements:
                    for item in improvements:
                        st.markdown(f"- {item}")
                else:
                    st.write("Continue refining your technical explanations.")

            with st.expander("💡 Sample interview answer"):
                render_answer(
                    evaluation.get(
                        "sample_answer",
                        "No sample answer was provided.",
                    )
                )

            if st.session_state.show_sources:
                display_sources(evaluation.get("sources", []))

            button_label = (
                "➡️ Next question"
                if completed < target
                else "🏁 Finish interview and view summary"
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

        # Final summary.
        elif st.session_state.mock_show_summary:
            st.subheader("🏆 Interview summary")

            results = st.session_state.mock_results
            scores = [item["score"] for item in results]
            total_score = sum(scores)
            maximum_score = len(scores) * 10
            average = total_score / len(scores) if scores else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Questions", len(results))
            c2.metric("Total score", f"{total_score}/{maximum_score}")
            c3.metric("Average score", f"{average:.1f}/10")

            st.markdown("### Question-wise results")
            for index, item in enumerate(results, start=1):
                with st.expander(
                    f"Q{index}: {item['question']} — {item['score']}/10"
                ):
                    st.markdown("**Your answer**")
                    st.write(item["candidate_answer"])
                    st.markdown("**Feedback**")
                    st.write(item.get("feedback", ""))

            if st.session_state.mock_summary is None:
                try:
                    with st.spinner("Preparing your interview feedback..."):
                        st.session_state.mock_summary = generate_mock_summary(
                            results
                        )
                except Exception as error:
                    st.session_state.mock_summary = (
                        "The overall summary could not be generated. "
                        f"Error: {type(error).__name__}: {error}"
                    )

            st.markdown("### 🎓 Interview coach feedback")
            render_answer(st.session_state.mock_summary)

            if st.button(
                "🔁 Start another interview",
                type="primary",
                use_container_width=True,
            ):
                reset_mock()
                st.rerun()

        # Question and answer form.
        elif st.session_state.mock_current is not None:
            current = st.session_state.mock_current

            st.caption(
                f"{st.session_state.mock_difficulty} · "
                f"Question {completed + 1} of {target}"
            )
            st.markdown("### 🎤 Interview question")
            st.markdown(f"#### {current['question']}")

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
                    st.warning("Please enter your answer before submitting.")
                else:
                    try:
                        with st.spinner("Evaluating your answer..."):
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

                        st.session_state.mock_results.append(result)
                        st.session_state.mock_last_evaluation = result
                        st.rerun()

                    except Exception as error:
                        st.error(
                            f"Evaluation failed: "
                            f"{type(error).__name__}: {error}"
                        )
