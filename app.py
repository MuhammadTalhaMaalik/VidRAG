import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_audio_chunks
from core.summerize import summarize_transcript, generate_title
from core.extractor import (
    extract_actionable_items,
    extract_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="VidRAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #0f1117;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .info-card {
        background: #171a23;
        border: 1px solid #272b36;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
    }

    .card-content {
        color: #d1d5db;
        line-height: 1.7;
    }

    /* Metric cards */
    .metric-card {
        background: #171a23;
        border: 1px solid #272b36;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    /* Chat bubbles */
    .user-message {
        background: #1d4ed8;
        color: white;
        padding: 12px 16px;
        border-radius: 14px;
        margin: 8px 0;
        margin-left: 15%;
    }

    .assistant-message {
        background: #171a23;
        border: 1px solid #272b36;
        color: #e5e7eb;
        padding: 14px 16px;
        border-radius: 14px;
        margin: 8px 15% 8px 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #11131a;
        border-right: 1px solid #272b36;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }

    /* Text area */
    textarea {
        border-radius: 10px !important;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background-color: #272b36;
        margin: 25px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------

if "processed" not in st.session_state:
    st.session_state.processed = False

if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

def run_pipeline(source: str):

    chunks = process_input(source)

    transcript = transcribe_audio_chunks(chunks)

    title = generate_title(transcript)

    summary = summarize_transcript(transcript)

    action_items = extract_actionable_items(transcript)

    decision = extract_decisions(transcript)

    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "Title": title,
        "Transcript": transcript,
        "Summary": summary,
        "Action_Items": action_items,
        "Key_Decision": decision,
        "Open_Question": questions,
        "rag_chain": rag_chain,
    }


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## 🎬 VidRAG")

    st.markdown(
        """
        <div style="color:#9ca3af; margin-bottom:20px;">
        AI-powered video understanding and RAG assistant.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📥 Video Source")

    source = st.text_input(
        "YouTube URL or local file path",
        placeholder="https://youtube.com/watch?v=...",
    )

    process_button = st.button(
        "🚀 Process Video",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")

    st.markdown("### Features")

    st.markdown(
        """
        🎙️ **Audio Transcription**

        🧠 **AI Summarization**

        ✅ **Action Item Extraction**

        💡 **Decision Detection**

        ❓ **Question Extraction**

        🔎 **RAG-powered Chat**
        """
    )

    st.markdown("---")

    st.caption("VidRAG • AI Video Assistant")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 VidRAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Turn long videos into searchable knowledge."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Process Video
# ---------------------------------------------------------

if process_button:

    if not source.strip():

        st.warning("Please enter a YouTube URL or local video path.")

    else:

        st.session_state.processed = False
        st.session_state.result = None
        st.session_state.chat_history = []

        progress = st.progress(0)

        status = st.empty()

        try:

            status.info("🎵 Processing audio...")

            progress.progress(15)

            chunks = process_input(source)

            status.info("🎙️ Transcribing audio with Whisper...")

            progress.progress(35)

            transcript = transcribe_audio_chunks(chunks)

            status.info("🧠 Generating AI analysis...")

            progress.progress(55)

            title = generate_title(transcript)

            summary = summarize_transcript(transcript)

            progress.progress(70)

            status.info("🔎 Extracting insights...")

            action_items = extract_actionable_items(transcript)

            decision = extract_decisions(transcript)

            questions = extract_questions(transcript)

            progress.progress(85)

            status.info("🔗 Building RAG knowledge base...")

            rag_chain = build_rag_chain(transcript)

            progress.progress(100)

            result = {
                "Title": title,
                "Transcript": transcript,
                "Summary": summary,
                "Action_Items": action_items,
                "Key_Decision": decision,
                "Open_Question": questions,
                "rag_chain": rag_chain,
            }

            st.session_state.result = result
            st.session_state.processed = True

            status.success("✅ Video processed successfully!")

        except Exception as e:

            st.error(f"❌ Processing failed: {str(e)}")

            st.exception(e)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

if st.session_state.processed:

    result = st.session_state.result

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">🎬 {result["Title"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    transcript = result["Transcript"]

    word_count = len(transcript.split())

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{word_count:,}</div>
                <div class="metric-label">Transcript Words</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        action_count = (
            len(result["Action_Items"])
            if isinstance(result["Action_Items"], list)
            else 1
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{action_count}</div>
                <div class="metric-label">Action Items</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        decision_count = (
            len(result["Key_Decision"])
            if isinstance(result["Key_Decision"], list)
            else 1
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{decision_count}</div>
                <div class="metric-label">Decisions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:

        question_count = (
            len(result["Open_Question"])
            if isinstance(result["Open_Question"], list)
            else 1
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{question_count}</div>
                <div class="metric-label">Questions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Tabs
    # -----------------------------------------------------

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📋 Overview",
            "📝 Transcript",
            "✅ Action Items",
            "💡 Decisions",
            "❓ Questions",
        ]
    )

    # -----------------------------------------------------
    # Overview
    # -----------------------------------------------------

    with tab1:

        st.markdown("### 🧠 Summary")

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-content">
                    {result["Summary"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # Transcript
    # -----------------------------------------------------

    with tab2:

        st.markdown("### 📝 Full Transcript")

        st.text_area(
            "Transcript",
            value=result["Transcript"],
            height=500,
            label_visibility="collapsed",
        )

    # -----------------------------------------------------
    # Action Items
    # -----------------------------------------------------

    with tab3:

        st.markdown("### ✅ Action Items")

        action_items = result["Action_Items"]

        if isinstance(action_items, list):

            for index, item in enumerate(action_items, start=1):

                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-content">
                        ☐ <strong>{index}.</strong> {item}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.write(action_items)

    # -----------------------------------------------------
    # Decisions
    # -----------------------------------------------------

    with tab4:

        st.markdown("### 💡 Key Decisions")

        decisions = result["Key_Decision"]

        if isinstance(decisions, list):

            for index, item in enumerate(decisions, start=1):

                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-content">
                        💡 <strong>{index}.</strong> {item}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.write(decisions)

    # -----------------------------------------------------
    # Questions
    # -----------------------------------------------------

    with tab5:

        st.markdown("### ❓ Open Questions")

        questions = result["Open_Question"]

        if isinstance(questions, list):

            for index, item in enumerate(questions, start=1):

                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-content">
                        ❓ <strong>{index}.</strong> {item}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.write(questions)

    # -----------------------------------------------------
    # RAG Chat
    # -----------------------------------------------------

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown("## 💬 Chat with your video")

    st.caption(
        "Ask questions about the video. "
        "The assistant will answer using the video's transcript."
    )

    # Display previous messages

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            st.markdown(
                f"""
                <div class="user-message">
                    <strong>You</strong><br>
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="assistant-message">
                    <strong>VidRAG</strong><br>
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Chat input

    question = st.chat_input(
        "Ask something about this video..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        try:

            with st.spinner("🤔 Thinking..."):

                answer = ask_question(
                    result["rag_chain"],
                    question,
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"❌ Failed to answer question: {str(e)}"
            )