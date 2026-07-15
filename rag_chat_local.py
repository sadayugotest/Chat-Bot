"""
ระบบ Local RAG สำหรับงานซ่อมบำรุงเครื่องจักร
- ใช้ Ollama เป็น LLM/Embedding (รันบนเครื่อง)
- ใช้ ChromaDB เป็น Vector Database (persist ลงดิสก์)
- ใช้ LangChain สำหรับ pipeline Retrieval + Generation

วิธีใช้งาน (สรุป):
1) ติดตั้ง dependencies จาก requirements.txt
2) เปิด Ollama และ pull โมเดลที่ต้องใช้
3) รันไฟล์นี้ แล้วพิมพ์คำถามใน Terminal
4) พิมพ์ exit เพื่อจบโปรแกรม
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter


# =========================
# ส่วนตั้งค่า (Config)
# =========================
BASE_DIR = Path(__file__).resolve().parent
MANUALS_DIR = BASE_DIR / "manuals"
LEGACY_DATA_FILE = BASE_DIR / "maintenance_log.txt"
MANUAL_FILE_EXTENSIONS = {".txt"}
CHROMA_DIR = BASE_DIR / "chroma_db"
INDEX_MANIFEST_FILE = CHROMA_DIR / "index_manifest.json"
COLLECTION_NAME = "machine_maintenance_knowledge"
INDEX_SCHEMA_VERSION = "v2_strict_record_split"

# สามารถเปลี่ยนโมเดลผ่าน Environment Variable ได้
# ตัวอย่าง:
#   set OLLAMA_LLM_MODEL=llama3.1:8b
#   set OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# บังคับ re-index ทุกครั้งได้ผ่าน env (ปกติไม่ต้องเปิด)
# set FORCE_REBUILD_INDEX=true
FORCE_REBUILD_INDEX = os.getenv("FORCE_REBUILD_INDEX", "false").lower() == "true"


# =========================
# Utility Functions
# =========================
def discover_manual_files() -> list[Path]:
    """
    ค้นหาไฟล์คู่มือหลายไฟล์ในโฟลเดอร์เดียว
    พฤติกรรมการค้นหา:
    1) อ่านไฟล์ .txt ทุกไฟล์จากโฟลเดอร์ manuals (ถ้ามี)
    2) รวม maintenance_log.txt เดิมเข้าไปด้วย (ถ้ามี)
    3) คืนค่าไฟล์ที่ไม่ซ้ำกันและเรียงชื่อ
    """
    manual_files: list[Path] = []

    if MANUALS_DIR.exists() and MANUALS_DIR.is_dir():
        for path in sorted(MANUALS_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in MANUAL_FILE_EXTENSIONS:
                manual_files.append(path)

    if LEGACY_DATA_FILE.exists() and LEGACY_DATA_FILE.is_file():
        manual_files.append(LEGACY_DATA_FILE)

    if manual_files:
        # กันไฟล์ซ้ำกรณีชื่อเดียวกันหรือ path ซ้ำ
        uniq = sorted({file_path.resolve() for file_path in manual_files})
        return uniq

    raise FileNotFoundError(
        f"ไม่พบไฟล์คู่มือ: กรุณาสร้างโฟลเดอร์ {MANUALS_DIR.name} และวางไฟล์ .txt หรือสร้าง {LEGACY_DATA_FILE.name}"
    )


def load_manual_text(file_path: Path) -> str:
    """อ่านไฟล์คู่มือ/ประวัติซ่อมบำรุงแบบ text"""
    return file_path.read_text(encoding="utf-8")


def split_by_machine_record(raw_text: str) -> list[str]:
    """
    แบ่งข้อมูลโดยยึด separator = "=== END===" ตามที่กำหนด
    เพื่อให้ 1 เคสอาการ/สาเหตุ/วิธีแก้ = 1 chunk ชัดเจน

    หมายเหตุ:
    - ตั้ง chunk_size ใหญ่มาก เพื่อไม่ให้เกิดการซอยย่อยเพิ่ม
    - การแยกหลักจึงพึ่งพา separator เป็นหลัก
    """
    # สร้าง CharacterTextSplitter ตามข้อกำหนดของระบบ
    # (กำหนด separator เป็น === END===)
    separator = "=== END==="
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1_000_000,
        chunk_overlap=0,
        is_separator_regex=False,
    )

    # ใช้การ split แบบ strict ตาม delimiter โดยตรง
    # เพื่อบังคับให้ 1 record = 1 chunk แบบเด็ดขาดและไม่ตัดคำกลางบรรทัด
    strict_chunks: list[str] = []
    parts = [part.strip() for part in raw_text.split(separator) if part.strip()]
    for part in parts:
        strict_chunks.append(f"{part}\n{separator}")

    return strict_chunks


def build_documents(chunks: list[str], source_file: Path) -> list[Document]:
    """แปลงข้อความแต่ละ chunk เป็น LangChain Document พร้อม metadata"""
    docs: list[Document] = []
    for idx, text in enumerate(chunks, start=1):
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(source_file.name),
                    "chunk_id": idx,
                },
            )
        )
    return docs


def compute_files_signature(files: list[Path]) -> str:
    """
    สร้างลายนิ้วมือรวมของไฟล์คู่มือทั้งหมด
    ใช้สำหรับตรวจว่าไฟล์มีการเปลี่ยนแปลงหรือไม่
    """
    h = hashlib.sha256()
    # รวมเวอร์ชันโครงสร้างดัชนี เพื่อบังคับ rebuild เมื่อ logic split เปลี่ยน
    h.update(INDEX_SCHEMA_VERSION.encode("utf-8"))
    for file_path in sorted(files):
        content_bytes = file_path.read_bytes()
        h.update(file_path.name.encode("utf-8"))
        h.update(str(len(content_bytes)).encode("utf-8"))
        h.update(content_bytes)
    return h.hexdigest()


def load_manifest_signature() -> Optional[str]:
    """อ่าน signature เดิมจากไฟล์ manifest ถ้ามี"""
    if not INDEX_MANIFEST_FILE.exists():
        return None

    try:
        payload = json.loads(INDEX_MANIFEST_FILE.read_text(encoding="utf-8"))
        value = payload.get("files_signature")
        return value if isinstance(value, str) else None
    except Exception:
        return None


def save_manifest_signature(files_signature: str, files: list[Path], docs_count: int) -> None:
    """บันทึก signature ล่าสุดหลังสร้าง index สำเร็จ"""
    manifest = {
        "files_signature": files_signature,
        "files": [str(file_path.name) for file_path in files],
        "documents_count": docs_count,
    }
    INDEX_MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prepare_vectorstore(documents: list[Document], files_signature: str) -> Chroma:
    """
    สร้างหรือโหลด ChromaDB
    - re-index เมื่อไฟล์คู่มือเปลี่ยนเท่านั้น
    - หรือบังคับ re-index ด้วย FORCE_REBUILD_INDEX=true
    - persist ลงโฟลเดอร์ ./chroma_db
    """
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    previous_signature = load_manifest_signature()
    has_index = CHROMA_DIR.exists()
    index_changed = previous_signature != files_signature

    should_rebuild = FORCE_REBUILD_INDEX or (not has_index) or index_changed

    if should_rebuild:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_name=COLLECTION_NAME,
        )
        return vectorstore

    # กรณีไฟล์ไม่เปลี่ยน ให้โหลด index เดิม
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def parse_record_from_text(text: str) -> dict[str, str]:
    """
    พยายามดึงข้อมูลโครงสร้างจาก chunk สำหรับ fallback
    ใช้เมื่อโมเดลตอบไม่ตรง format ที่บังคับ
    """
    machine_match = re.search(r"===\s*MACHINE:\s*(.*?)\s*===", text, re.IGNORECASE)
    symptom_match = re.search(r"\[อาการ\]\s*(.*)", text)
    cause_match = re.search(r"\[สาเหตุ\]\s*(.*)", text)

    fix_match = re.search(
        r"\[การแก้ไข\]\s*(.*?)(?:\n\s*\[[^\]]+\]|\n\s*===\s*END===|$)",
        text,
        re.DOTALL,
    )

    return {
        "machine": machine_match.group(1).strip() if machine_match else "ไม่พบข้อมูล",
        "symptom": symptom_match.group(1).strip() if symptom_match else "ไม่พบข้อมูล",
        "cause": cause_match.group(1).strip() if cause_match else "ไม่พบข้อมูล",
        "fix": fix_match.group(1).strip() if fix_match else "ไม่พบข้อมูล",
    }


def normalize_output_or_fallback(raw_answer: str, top_doc_text: Optional[str]) -> str:
    """
    บังคับรูปแบบ output ให้เป็น 4 บรรทัดตาม requirement
    ถ้า LLM ตอบไม่ครบ จะ fallback จากข้อมูลที่ parse ได้จากเอกสาร top-1
    """
    pattern = re.compile(
        r"MACHINE\s*:\s*(.*?)\nอาการ\s*:\s*(.*?)\nสาเหตุ\s*:\s*(.*?)\nการแก้ไข\s*:\s*(.*)",
        re.DOTALL,
    )

    m = pattern.search(raw_answer.strip())
    if m:
        machine = m.group(1).strip() or "ไม่พบข้อมูล"
        symptom = m.group(2).strip() or "ไม่พบข้อมูล"
        cause = m.group(3).strip() or "ไม่พบข้อมูล"
        fix = m.group(4).strip() or "ไม่พบข้อมูล"
        return (
            f"MACHINE: {machine}\n"
            f"อาการ: {symptom}\n"
            f"สาเหตุ: {cause}\n"
            f"การแก้ไข: {fix}"
        )

    # fallback เมื่อ output หลุด format
    if top_doc_text:
        parsed = parse_record_from_text(top_doc_text)
        return (
            f"MACHINE: {parsed['machine']}\n"
            f"อาการ: {parsed['symptom']}\n"
            f"สาเหตุ: {parsed['cause']}\n"
            f"การแก้ไข: {parsed['fix']}"
        )

    return (
        "MACHINE: ไม่พบข้อมูล\n"
        "อาการ: ไม่พบข้อมูล\n"
        "สาเหตุ: ไม่พบข้อมูล\n"
        "การแก้ไข: ไม่พบข้อมูล"
    )


def format_answer_from_record(record: dict[str, str]) -> str:
    """จัดรูปแบบคำตอบมาตรฐานจากข้อมูล record ที่ parse แล้ว"""
    return (
        f"MACHINE: {record['machine'] or 'ไม่พบข้อมูล'}\n"
        f"อาการ: {record['symptom'] or 'ไม่พบข้อมูล'}\n"
        f"สาเหตุ: {record['cause'] or 'ไม่พบข้อมูล'}\n"
        f"การแก้ไข: {record['fix'] or 'ไม่พบข้อมูล'}"
    )


def extract_alarm_code(text: str) -> Optional[str]:
    """ดึงรหัส Alarm จากคำถามผู้ใช้ เช่น Alarm 5567 -> 5567"""
    m = re.search(r"alarm\s*([A-Za-z0-9\-_/]+)", text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip().lower()


def find_exact_record_from_docs(question: str, docs: list[Document]) -> Optional[dict[str, str]]:
    """
    พยายามหา record ที่ตรงกับ Alarm ในคำถามแบบตรงตัว
    เพื่อกันโมเดลสรุปขั้นตอนการแก้ไขจนข้อมูลหาย
    """
    alarm_code = extract_alarm_code(question)
    if not alarm_code:
        return None

    q_lower = question.lower()
    for doc in docs:
        record = parse_record_from_text(doc.page_content)
        symptom_lower = record["symptom"].lower()
        machine_lower = record["machine"].lower()

        # ต้อง match alarm ตรงก่อน
        if alarm_code not in symptom_lower:
            continue

        # ถ้าผู้ใช้ระบุชื่อเครื่องด้วย ให้พยายาม match เครื่องร่วมด้วย
        machine_tokens = [token for token in re.split(r"\s+", machine_lower) if token]
        if machine_tokens:
            overlap = sum(1 for token in machine_tokens if token in q_lower)
            if overlap == 0 and "machine" in q_lower:
                continue

        return record

    return None


def build_prompt() -> ChatPromptTemplate:
    """สร้าง System Prompt ที่บังคับรูปแบบคำตอบอย่างเคร่งครัด"""
    system_prompt = (
        "คุณคือผู้ช่วยช่างซ่อมบำรุงเครื่องจักร\n"
        "คุณต้องตอบเป็นภาษาไทยล้วนเท่านั้น ยกเว้นคำว่า MACHINE ที่ต้องคงตาม format\n"
        "ให้ใช้ข้อมูลจากบริบทที่ให้มาเท่านั้น ห้ามแต่งข้อมูลเพิ่มเอง\n"
        "ห้ามทักทาย ห้ามเกริ่นนำ ห้ามอธิบายเพิ่ม ห้ามใส่ markdown ห้ามใส่ code block\n"
        "กฎเหล็ก: ต้องตอบกลับเฉพาะ format นี้เท่านั้น และห้ามมีข้อความอื่นนอกเหนือจากนี้:\n"
        "MACHINE: [ชื่อรุ่นเครื่องจักร]\n"
        "อาการ: [ระบุอาการ/รหัส Alarm]\n"
        "สาเหตุ: [ระบุสาเหตุ]\n"
        "การแก้ไข: [ระบุขั้นตอนการแก้ไข]\n\n"
        "ถ้าไม่พบข้อมูลที่ตรง ให้ตอบคำว่า 'ไม่พบข้อมูล' ในแต่ละช่อง\n"
        "ถ้าพบหลายเคส ให้เลือกเคสที่ตรงที่สุดเพียง 1 เคส\n\n"
        "บริบทข้อมูล:\n{context}"
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "คำถามผู้ใช้: {question}"),
        ]
    )


def answer_question(
    question: str,
    retriever,
    llm: ChatOllama,
    prompt: ChatPromptTemplate,
) -> str:
    """ดึง context -> ให้ LLM ตอบ -> normalize ให้ตรง format บังคับ"""
    docs = retriever.invoke(question)

    # ถ้าจับคู่ Alarm ได้แบบตรงตัว ให้ตอบจากข้อมูลต้นฉบับทันที
    # เพื่อคงรายละเอียดการแก้ไขหลายบรรทัดไม่ให้ถูกย่อ
    exact_record = find_exact_record_from_docs(question=question, docs=docs)
    if exact_record:
        return format_answer_from_record(exact_record)

    context = "\n\n".join(doc.page_content for doc in docs)

    messages = prompt.format_messages(context=context, question=question)
    result = llm.invoke(messages)
    raw_answer = (result.content or "").strip()

    top_doc_text = docs[0].page_content if docs else None
    normalized = normalize_output_or_fallback(raw_answer=raw_answer, top_doc_text=top_doc_text)

    # ชั้นป้องกันเพิ่ม: ถ้าถาม Alarm ชัดเจนและ top doc match ให้ยึดข้อมูล top doc
    # เพื่อหลีกเลี่ยงกรณี LLM ตอบไม่ครบขั้นตอนการแก้ไข
    alarm_code = extract_alarm_code(question)
    if alarm_code and top_doc_text:
        top_record = parse_record_from_text(top_doc_text)
        if alarm_code in top_record["symptom"].lower():
            return format_answer_from_record(top_record)

    return normalized


def initialize_rag_components() -> dict[str, object]:
    """
    เตรียมองค์ประกอบ RAG กลางสำหรับทั้งโหมด Terminal และ Web UI
    คืนค่าทุกอย่างใน dict เพื่อเรียกใช้ต่อได้สะดวก
    """
    manual_files = discover_manual_files()
    all_docs: list[Document] = []

    for file_path in manual_files:
        raw_text = load_manual_text(file_path)
        chunks = split_by_machine_record(raw_text)
        all_docs.extend(build_documents(chunks, source_file=file_path))

    if not all_docs:
        raise RuntimeError("ไม่พบข้อมูลหลังการ split กรุณาตรวจสอบรูปแบบไฟล์คู่มือทั้งหมด")

    files_signature = compute_files_signature(manual_files)
    old_signature = load_manifest_signature()
    index_needs_rebuild = FORCE_REBUILD_INDEX or (old_signature != files_signature) or (not CHROMA_DIR.exists())

    vectorstore = prepare_vectorstore(all_docs, files_signature=files_signature)
    if index_needs_rebuild:
        save_manifest_signature(
            files_signature=files_signature,
            files=manual_files,
            docs_count=len(all_docs),
        )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOllama(
        model=OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    prompt = build_prompt()

    return {
        "retriever": retriever,
        "llm": llm,
        "prompt": prompt,
        "manual_files": manual_files,
        "index_needs_rebuild": index_needs_rebuild,
    }


def main() -> None:
    """Entry point ของโปรแกรมแชทบน Terminal"""
    print("กำลังเตรียมระบบ Local RAG...")
    components = initialize_rag_components()
    retriever = components["retriever"]
    llm = components["llm"]
    prompt = components["prompt"]
    manual_files = components["manual_files"]
    index_needs_rebuild = components["index_needs_rebuild"]

    print("ระบบพร้อมใช้งานแล้ว")
    print("ไฟล์คู่มือที่โหลด:")
    for file_path in manual_files:
        print(f"- {file_path}")
    if index_needs_rebuild:
        print("สถานะดัชนี: มีการสร้าง/อัปเดตดัชนีใหม่")
    else:
        print("สถานะดัชนี: ใช้ดัชนีเดิม (ไฟล์คู่มือไม่เปลี่ยน)")
    print("พิมพ์คำถามเกี่ยวกับ Alarm/อาการเครื่องจักรได้เลย (พิมพ์ 'exit' เพื่อออก)\n")

    while True:
        user_input = input("You> ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("จบการทำงาน")
            break

        try:
            answer = answer_question(
                question=user_input,
                retriever=retriever,
                llm=llm,
                prompt=prompt,
            )
            print(f"\n{answer}\n")
        except Exception as exc:
            # กรณีเกิดปัญหาระหว่างเรียกโมเดลหรือ retrieval
            print("\nMACHINE: ไม่พบข้อมูล")
            print("อาการ: ไม่พบข้อมูล")
            print("สาเหตุ: ไม่พบข้อมูล")
            print(f"การแก้ไข: เกิดข้อผิดพลาดในการประมวลผล ({exc})\n")


if __name__ == "__main__":
    main()
