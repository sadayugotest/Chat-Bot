"""
Web UI สำหรับระบบ Local RAG งานซ่อมบำรุงเครื่องจักร
- ใช้ Streamlit ทำหน้าตาแชทสไตล์โมเดิร์น
- เชื่อม backend จาก rag_chat_local.py
"""

from __future__ import annotations

import streamlit as st

from rag_chat_local import (
    OLLAMA_LLM_MODEL,
    answer_question,
    initialize_rag_components,
)


def apply_custom_style(theme_mode: str) -> None:
    """ใส่ธีมและสไตล์ให้หน้าเว็บดูคล้ายแอปแชทสมัยใหม่"""
    is_dark = theme_mode == "มืด"

    if is_dark:
        root_vars = """
            --bg-1: #0b1118;
            --bg-2: #0f1a24;
            --panel: rgba(18, 27, 38, 0.90);
            --panel-border: rgba(110, 153, 210, 0.24);
            --accent: #49d18e;
            --accent-soft: rgba(61, 166, 116, 0.24);
            --text-main: #edf3ff;
            --text-sub: #b8c7db;
            --bot-bubble: #131f2c;
            --user-bubble: linear-gradient(135deg, #1f7a4d 0%, #1c7288 100%);
            --user-text: #f4fffa;
            --sidebar-bg-1: #0a1017;
            --sidebar-bg-2: #101a26;
            --header-color: #f2f8ff;
            --input-bg: #0f1d2a;
            --input-text: #eff6ff;
            --input-border: rgba(148, 184, 236, 0.30);
            --surface-shadow: rgba(2, 5, 9, 0.45);
        """
    else:
        root_vars = """
            --bg-1: #f4f8f5;
            --bg-2: #e6efe8;
            --panel: rgba(255, 255, 255, 0.86);
            --panel-border: rgba(26, 71, 42, 0.14);
            --accent: #1c6e48;
            --accent-soft: #d8eee1;
            --text-main: #1b2620;
            --text-sub: #42544b;
            --bot-bubble: #ffffff;
            --user-bubble: linear-gradient(135deg, #1f7a4d 0%, #1f6b7f 100%);
            --user-text: #f8fffb;
            --sidebar-bg-1: #f8fcf9;
            --sidebar-bg-2: #edf6f1;
            --header-color: #133422;
            --input-bg: rgba(255, 255, 255, 0.92);
            --input-text: #173126;
            --input-border: rgba(23, 80, 50, 0.24);
            --surface-shadow: rgba(24, 56, 39, 0.06);
        """

    css_template = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            __ROOT_VARS__
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 14%, rgba(46, 160, 94, 0.10), transparent 38%),
                radial-gradient(circle at 84% 8%, rgba(26, 107, 127, 0.11), transparent 42%),
                linear-gradient(150deg, var(--bg-1), var(--bg-2));
            color: var(--text-main);
            font-family: 'Sarabun', sans-serif;
        }

        .main .block-container {
            max-width: 900px;
            padding-top: 1.6rem;
            padding-bottom: 1.2rem;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', 'Sarabun', sans-serif;
            letter-spacing: -0.02em;
            color: var(--header-color);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--sidebar-bg-1) 0%, var(--sidebar-bg-2) 100%);
            border-right: 1px solid var(--panel-border);
            color: var(--text-main);
        }

        [data-testid="stSidebar"] * {
            color: var(--text-main) !important;
        }

        [data-testid="stChatMessage"] {
            border-radius: 16px;
            border: 1px solid var(--panel-border);
            background: var(--panel);
            backdrop-filter: blur(4px);
            margin-bottom: 0.75rem;
            box-shadow: 0 12px 32px var(--surface-shadow);
            animation: fadeInUp 0.28s ease-out;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] pre,
        [data-testid="stChatMessage"] code {
            line-height: 1.55;
            font-size: 1rem;
            color: var(--text-main) !important;
            background: transparent !important;
            font-family: 'Sarabun', sans-serif !important;
        }

        [data-testid="stChatMessage"]:has([aria-label="user avatar"]) {
            background: var(--user-bubble);
            border: 1px solid rgba(14, 54, 35, 0.26);
        }

        [data-testid="stChatMessage"]:has([aria-label="user avatar"]) [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"]:has([aria-label="user avatar"]) [data-testid="stMarkdownContainer"] {
            color: var(--user-text);
            font-weight: 500;
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 14px !important;
            border: 1px solid var(--input-border) !important;
            background: var(--input-bg) !important;
            color: var(--input-text) !important;
            font-family: 'Sarabun', sans-serif !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--text-sub) !important;
            opacity: 0.9;
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid var(--panel-border);
            background: linear-gradient(135deg, rgba(32, 103, 69, 0.95), rgba(27, 96, 115, 0.95));
            color: #f4fffa;
            font-weight: 600;
        }

        .stButton > button:hover {
            filter: brightness(1.06);
        }

        .status-chip {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.4rem;
            background: var(--accent-soft);
            color: var(--text-main);
            border: 1px solid var(--panel-border);
        }

        .app-caption {
            color: var(--text-sub);
            margin-top: -0.2rem;
            margin-bottom: 0.8rem;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 768px) {
            .main .block-container {
                padding-top: 1rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
        }
        </style>
        """

    st.markdown(css_template.replace("__ROOT_VARS__", root_vars), unsafe_allow_html=True)


@st.cache_resource(show_spinner=True)
def load_rag_runtime() -> dict[str, object]:
    """โหลดองค์ประกอบ RAG (cache ไว้เพื่อลดเวลาเริ่มต้น)"""
    return initialize_rag_components()


def render_header(index_rebuilt: bool) -> None:
    """แสดงหัวข้อและสถานะระบบ"""
    st.title("Maintenance RAG Assistant")
    st.markdown(
        "<p class='app-caption'>ผู้ช่วยตอบคำถามงานซ่อมบำรุงเครื่องจักรจากคู่มือภายในองค์กร</p>",
        unsafe_allow_html=True,
    )

    index_status = "สร้าง/อัปเดตดัชนีใหม่" if index_rebuilt else "ใช้ดัชนีเดิม"
    st.markdown(
        (
            "<span class='status-chip'>Local Ollama</span>"
            f"<span class='status-chip'>Model: {OLLAMA_LLM_MODEL}</span>"
            f"<span class='status-chip'>Index: {index_status}</span>"
        ),
        unsafe_allow_html=True,
    )


def render_sidebar(runtime: dict[str, object]) -> None:
    """แสดงข้อมูลแหล่งไฟล์คู่มือและปุ่มควบคุม"""
    with st.sidebar:
        st.header("การตั้งค่า")
        st.caption("ระบบทำงานแบบ Local ทั้งหมด")

        st.radio(
            "ธีมหน้าจอ",
            options=["สว่าง", "มืด"],
            key="theme_mode",
            horizontal=True,
        )

        if st.button("Reload ดัชนี", use_container_width=True):
            # ล้าง cache เพื่อบังคับโหลดใหม่ในรอบถัดไป
            load_rag_runtime.clear()
            st.rerun()

        st.divider()
        st.subheader("ไฟล์คู่มือที่โหลด")
        for file_path in runtime["manual_files"]:
            st.write(f"- {file_path.name}")

        st.divider()
        st.subheader("ข้อกำหนดคำตอบ")
        st.caption("โมเดลจะตอบเป็นรูปแบบ MACHINE/อาการ/สาเหตุ/การแก้ไข เท่านั้น")


def ensure_chat_history() -> None:
    """เตรียม session state สำหรับเก็บข้อความในแชท"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "MACHINE: พร้อมใช้งาน\n"
                    "อาการ: รอรับคำถาม\n"
                    "สาเหตุ: -\n"
                    "การแก้ไข: พิมพ์คำถามเกี่ยวกับ Alarm หรืออาการเครื่องจักรได้ทันที"
                ),
            }
        ]


def render_chat_messages() -> None:
    """วาดข้อความแชทย้อนหลัง"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.text(msg["content"])


def process_user_prompt(runtime: dict[str, object], user_prompt: str) -> None:
    """ประมวลผลคำถามผู้ใช้และเพิ่มคำตอบลงประวัติแชท"""
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.text(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลที่ตรงที่สุด..."):
            try:
                answer = answer_question(
                    question=user_prompt,
                    retriever=runtime["retriever"],
                    llm=runtime["llm"],
                    prompt=runtime["prompt"],
                )
            except Exception as exc:
                answer = (
                    "MACHINE: ไม่พบข้อมูล\n"
                    "อาการ: ไม่พบข้อมูล\n"
                    "สาเหตุ: ไม่พบข้อมูล\n"
                    f"การแก้ไข: เกิดข้อผิดพลาดในการประมวลผล ({exc})"
                )

        st.text(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    st.set_page_config(
        page_title="Maintenance Local RAG Chat",
        page_icon="🛠️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "มืด"

    apply_custom_style(st.session_state.theme_mode)

    try:
        runtime = load_rag_runtime()
    except Exception as exc:
        st.error(f"เริ่มระบบไม่สำเร็จ: {exc}")
        st.stop()

    render_header(index_rebuilt=bool(runtime["index_needs_rebuild"]))
    render_sidebar(runtime)

    ensure_chat_history()
    render_chat_messages()

    user_prompt = st.chat_input("ถามเรื่อง Alarm หรืออาการเครื่องจักร...")
    if user_prompt:
        process_user_prompt(runtime, user_prompt)


if __name__ == "__main__":
    main()
