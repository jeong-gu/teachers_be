# -*- coding: utf-8 -*-
import os
import re
import uuid
import json
import time
import shutil
import datetime as dt
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# LangChain & Clova
from openai import RateLimitError
from langchain_naver import ChatClovaX, ClovaXEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings

import requests

# =========================
# 0) ENV & PATHS
# =========================
load_dotenv(find_dotenv())

ROOT_DIR   = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.abspath(os.path.join(ROOT_DIR, "..", "data"))
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
AUDIO_DIR  = os.path.join(DATA_DIR, "audio")
VECTOR_DIR = os.getenv("VECTOR_DIR", os.path.join(DATA_DIR, "faiss"))
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join(DATA_DIR, "app.db"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# =========================
# 1) Persona & TTS Config
# =========================
PERSONAS: Dict[str, Dict[str, str]] = {
    "ngoeun": {
        "display": "고은", "sex": "여성", "speaker": "ngoeun",
        "persona_rules": (
            "당신은 친절하고 따뜻한 여자 선생님입니다. "
            "학생이 편안하게 느낄 수 있도록 차분하고 부드러운 말투를 사용하세요. "
            "어려운 개념은 기초부터 차근차근 설명하고, 이해가 잘 되도록 예시를 들어주세요. "
            "마지막에는 “너무 잘했어요!”, “조금씩 나아지고 있어요” 같은 격려 멘트를 반드시 포함하세요."
        ),
    },
    "nkyuwon": {
        "display": "규원", "sex": "남성", "speaker": "nkyuwon",
        "persona_rules": (
            "당신은 같은 또래 친구처럼 친근하고 편안한 톤으로 가르쳐주는 역할입니다. "
            "전문적인 용어를 쓰기보다는 일상적인 언어와 비유를 활용해 설명하세요. "
            "어려운 개념이 나오면 “이거는 마치 ○○ 같은 거야” 식으로 쉽게 풀어주세요. "
            "학생이 부담을 느끼지 않도록 가볍고 캐주얼한 분위기를 유지하세요."
        ),
    },
    "nminjeong": {
        "display": "민정", "sex": "여성", "speaker": "nminjeong",
        "persona_rules": (
            "당신은 카리스마 있고 임팩트 있는 일타강사 스타일의 강사입니다. "
            "빠른 템포와 강한 자신감으로 핵심만 콕 집어 설명하세요. "
            "불필요한 말은 줄이고, “이 부분은 무조건 외워야 한다”, “이거 시험에 100% 나온다” 같은 강조 멘트를 사용하세요. "
            "가끔은 학생을 긴장시키면서도 동기부여가 될 수 있도록 “이걸 모르면 큰일 납니다” 같은 멘트를 넣으세요."
        ),
    },
    "nheera": {
        "display": "희라", "sex": "여성", "speaker": "nheera",
        "persona_rules": (
            "당신은 차분하고 논리적인 분석형 튜터입니다. "
            "학생이 스스로 사고할 수 있도록 질문을 던지며 설명을 이끌어가세요. "
            "설명은 '핵심 요약 → 단계적 분석 → 관련 질문 → 정답 및 해설' 순서로 진행합니다. "
            "복잡한 개념은 도식적·단계적 언어로 분해해 주며, 마지막에는 “이제 당신 차례예요, 한번 풀어볼까요?” 같은 멘트를 덧붙이세요."
        ),
    },
    "ntaejin": {
        "display": "태진", "sex": "남성", "speaker": "ntaejin",
        "persona_rules": (
            "당신은 열정적인 남자 스포츠 코치입니다. "
            "학생에게 강의 내용을 마치 훈련처럼 느끼게 하며, 힘 있고 직설적인 말투를 사용하세요. "
            "설명은 “기본기 → 훈련 → 응용” 순서로 진행하고, 어려운 개념을 스포츠 훈련에 비유하세요. "
            "항상 동기부여를 주며 “좋아! 바로 그거야!”, “멈추지 마, 끝까지 가보자!” 같은 구호를 포함하세요."
        ),
    },
    "nmovie": {
        "display": "최무비", "sex": "남성", "speaker": "nmovie",
        "persona_rules": (
            "당신은 차갑고 날카로운 남자 비평가형 강사입니다. "
            "학생에게 단점을 숨기지 않고 솔직하게 지적하세요. "
            "설명은 핵심만 간결하게 전달하며, 부족한 부분이 있으면 “이건 전혀 충분하지 않다”, "
            "“이 부분을 반드시 보완해야 한다”처럼 단호하게 표현하세요. "
            "마지막에는 개선 방향을 제시하면서 “다시 해보라”는 식으로 도전 의식을 불러일으키세요."
        ),
    },
}
DEFAULT_PERSONA_KEY = os.getenv("DEFAULT_PERSONA", "ntaejin")
if DEFAULT_PERSONA_KEY not in PERSONAS:
    DEFAULT_PERSONA_KEY = "nminjeong"
P_DEFAULT = PERSONAS[DEFAULT_PERSONA_KEY]

# CLOVA Voice (Premium) endpoints
NCP_KEY_ID = os.getenv("NCP_CLOVA_KEY_ID")
NCP_KEY    = os.getenv("NCP_CLOVA_KEY")
TTS_PREMIUM_URL = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"

# =========================
# 2) DB
# =========================
Base = declarative_base()
engine = create_engine(f"sqlite:///{SQLITE_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), index=True)
    doc_id = Column(String(64), index=True, nullable=True)
    role = Column(String(16))      # 'user' or 'assistant'
    content = Column(Text)
    refs = Column(Text)            # JSON string
    audio_path = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

class DocumentMeta(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    doc_id = Column(String(64), unique=True, index=True)
    filename = Column(String(256))
    script = Column(Text)          # (옵션) 요약/스크립트 저장용
    created_at = Column(DateTime, default=dt.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def save_interaction(db, *, session_id: str, role: str, content: str,
                     doc_id: Optional[str] = None,
                     refs: Optional[Dict] = None,
                     audio_path: Optional[str] = None):
    it = Interaction(
        session_id=session_id,
        doc_id=doc_id,
        role=role,
        content=content,
        refs=json.dumps(refs, ensure_ascii=False) if refs else None,
        audio_path=audio_path
    )
    db.add(it)
    db.commit()

# =========================
# 3) FastAPI
# =========================
app = FastAPI(title="RAG TeachKit (FAISS + Persona + Clova Voice TTS Premium)")
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

# =========================
# 4) LLM & Embeddings
# =========================
api_key = os.getenv("CLOVASTUDIO_API_KEY")
if not api_key:
    raise RuntimeError("환경변수 CLOVASTUDIO_API_KEY 가 필요합니다.")
llm = ChatClovaX(model="HCX-007", api_key=api_key, temperature=0.2)
_base_embeddings = ClovaXEmbeddings(model="clir-emb-dolphin", api_key=api_key)

class RateLimitedEmbeddings(Embeddings):
    def __init__(self, base: Embeddings, throttle: float = 0.25,
                 base_delay: float = 0.6, max_retries: int = 6, batch_size: int = 16):
        self.base = base
        self.throttle = throttle
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.batch_size = batch_size
    def _sleep(self):
        if self.throttle > 0: time.sleep(self.throttle)
    def _with_retry(self, func, *args, **kwargs):
        delay = self.base_delay
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError:
                time.sleep(delay); delay *= 2
            except Exception as e:
                if "429" in str(e):
                    time.sleep(delay); delay *= 2
                else:
                    raise
        return func(*args, **kwargs)
    def embed_query(self, text: str) -> List[float]:
        self._sleep(); return self._with_retry(self.base.embed_query, text)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        n = len(texts)
        for s in range(0, n, self.batch_size):
            batch = texts[s:s+self.batch_size]
            self._sleep()
            vecs = self._with_retry(self.base.embed_documents, batch)
            out.extend(vecs)
        return out

embeddings = RateLimitedEmbeddings(_base_embeddings, throttle=0.25, base_delay=0.6, max_retries=6, batch_size=16)

# =========================
# 5) VectorStore (FAISS)
# =========================
def _vs_path(doc_id: str) -> str:
    return os.path.join(VECTOR_DIR, doc_id)

def persist_faiss(vs: FAISS, doc_id: str):
    target = _vs_path(doc_id)
    os.makedirs(target, exist_ok=True)
    vs.save_local(target)

def load_faiss(doc_id: str) -> FAISS:
    path = _vs_path(doc_id)
    if not os.path.isdir(path):
        raise FileNotFoundError("Vector store not found. 먼저 PDF를 업로드하세요.")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# =========================
# 6) Prompts (첫 번째 코드 스타일)
# =========================
def build_qa_prompt(P: Dict[str, str]) -> ChatPromptTemplate:
    qa_system_template = f"""
너는 {P['sex']} 화자의, **{P['display']}** 스타일 AI 튜터야.
아래의 페르소나 지침을 반드시 따른다:
{P['persona_rules']}

대화 전반 규칙:
- 자연스러운 대화체로 3~6문장, 문장은 짧게.
- 목록/번호/마크다운(#, ##, 1., - 등) 금지.
- 하나의 개념을 설명한 뒤 짧은 예시 1개를 덧붙여.
- 어려운 용어는 쉬운 말로 바로 풀어줘.
- 아래 문서 내용에 근거해서만 답해. 없으면 "자료엔 이 내용이 없어"라고 말하고, 대신 한 줄 제안.
- 마지막엔 아주 짧게 되묻기 1문장.

문서 내용:
{{context}}
"""
    return ChatPromptTemplate.from_messages([
        ("system", qa_system_template),
        ("human", "{question}"),
    ])

def build_lecture_prompt(P: Dict[str, str]) -> ChatPromptTemplate:
    lecture_system_template = f"""
너는 {P['sex']} 화자의, **{P['display']}** 스타일 AI 튜터야.
아래의 페르소나 지침을 반드시 따른다:
{P['persona_rules']}

출력 규칙(반드시 준수):
- 인사/수락/감탄/전환 멘트로 시작 금지(예: "좋아", "오케이", "알겠어", "자", "그럼", "설명해줄게").
- 곧바로 내용으로 시작. 첫 문장은 주제/핵심 개념으로.
- 지금은 p{{page_num}} 내용만 사용. 앞뒤 페이지 사전 언급 금지.
- 4~7문장: 핵심 요지 → 쉬운 예시 1개 → 마지막에 이해 확인 질문 1문장.
- 목록/번호/마크다운 금지. 어려운 용어는 바로 풀어서.
현재 페이지 텍스트:
{{context}}
"""
    return ChatPromptTemplate.from_messages([
        ("system", lecture_system_template),
        ("human", "강의 본문만 출력."),
    ])

# =========================
# 7) CSR & TTS (프리미엄, 분할)
# =========================
def recognize_csr(audio_path: str) -> str:
    CSR_URL = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"
    if not (NCP_KEY_ID and NCP_KEY):
        return ""
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NCP_KEY_ID,
        "X-NCP-APIGW-API-KEY": NCP_KEY,
        "Content-Type": "application/octet-stream",
    }
    with open(audio_path, "rb") as f:
        data = f.read()
    resp = requests.post(CSR_URL, headers=headers, data=data, timeout=60)
    if resp.status_code != 200:
        return ""
    try:
        return resp.json().get("text", "") or ""
    except Exception:
        return resp.text.strip()

def split_for_tts(text: str, max_len: int = 2800) -> List[str]:
    if len(text) <= max_len:
        return [text]
    parts, cur = [], []
    cur_len = 0
    for sent in re.split(r'(?<=[.?!…]|[가-힣]\))\s+', text):
        if cur_len + len(sent) + 1 > max_len and cur:
            parts.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(sent)
        cur_len += len(sent) + 1
    if cur:
        parts.append(" ".join(cur))
    return parts

def synthesize_clova_voice_premium_mp3(
    text: str,
    speaker: str,
    speed: int = 0,
    pitch: int = 0,
    volume: int = 0,
    audio_dir: str = AUDIO_DIR,
) -> List[str]:
    """
    CLOVA Voice Premium TTS (mp3) + 2800자 분할. 다중 파일 경로 반환.
    """
    if not (NCP_KEY_ID and NCP_KEY):
        return []
    os.makedirs(audio_dir, exist_ok=True)
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NCP_KEY_ID,
        "X-NCP-APIGW-API-KEY": NCP_KEY,
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
    }
    chunks = split_for_tts(text, max_len=2800)
    out_paths = []
    delay = 0.6
    for idx, chunk in enumerate(chunks, start=1):
        data = {
            "speaker": speaker,
            "text": chunk,
            "format": "mp3",
            "speed": str(speed),
            "pitch": str(pitch),
            "volume": str(volume),
        }
        for attempt in range(6):
            resp = requests.post(TTS_PREMIUM_URL, headers=headers, data=data, timeout=60)
            if resp.status_code == 200:
                fname = f"{uuid.uuid4().hex}"
                if len(chunks) == 1:
                    fname = f"{fname}.mp3"
                else:
                    fname = f"{fname}_part{idx}.mp3"
                path = os.path.join(audio_dir, fname)
                with open(path, "wb") as f:
                    f.write(resp.content)
                out_paths.append(path)
                break
            elif resp.status_code in (429, 503):
                time.sleep(delay); delay *= 2
            else:
                raise RuntimeError(f"TTS 실패 {resp.status_code}: {resp.text}")
    return out_paths

# =========================
# 8) Units (페이지 단원)
# =========================
UNITS_BY_DOC: Dict[str, List[Dict[str, Any]]] = {}  # doc_id -> units(list)
LECTURE_STATE: Dict[str, int] = {}                  # session_id -> cur index

def build_units_from_documents(documents) -> List[Dict[str, Any]]:
    """페이지별 텍스트를 모아 단원 생성 (첫 번째 코드 스타일, 4000자 제한)."""
    page_buckets: Dict[int, List[str]] = defaultdict(list)
    for d in documents:
        p = int(d.metadata.get("page", 0))
        page_buckets[p].append(d.page_content)
    units = []
    for p in sorted(page_buckets.keys()):
        text = "\n".join(page_buckets[p]).strip()
        # [FIX] 스캔 PDF 방지: 완전 빈 페이지는 스킵하지 말고 알림 문구로 대체
        if not text:
            text = "(이 페이지는 추출된 텍스트가 없습니다. 스캔 PDF일 수 있어요.)"
        if len(text) > 4000:
            text = text[:4000] + "\n(이하 생략)"
        units.append({"page0": p, "page": p + 1, "text": text})
    return units

def teach_unit_text(P: Dict[str, str], unit: Dict[str, Any]) -> str:
    prompt = build_lecture_prompt(P)
    msgs = prompt.format_messages(page_num=unit["page"], context=unit["text"])
    resp = llm.invoke(msgs)
    return getattr(resp, "content", str(resp))

# =========================
# 9) RetrievalQA Builder
# =========================
def build_qa_chain(P: Dict[str, str], vectorstore: FAISS) -> RetrievalQA:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt = build_qa_prompt(P)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",   # 지금 코드
            # ↓ 추가
            #"input_key": "query",                  # 질문 입력 필드 명시
            #"output_key": "result",                # 출력 필드 명시
        },
    )

# =========================
# 10) Schemas
# =========================
class ChatReq(BaseModel):
    session_id: str
    question: str
    doc_id: Optional[str] = "current_doc"
    want_tts: bool = False
    persona: Optional[str] = None
    tts_speed: Optional[int] = None
    tts_volume: Optional[int] = None
    tts_pitch: Optional[int] = None

class ChatResp(BaseModel):
    answer: str
    audio_url: Optional[str] = None
    audio_urls: Optional[List[str]] = None
    # [FIX] 가변 디폴트 제거
    used_refs: Optional[List[Dict[str, Any]]] = Field(default=None)

# =========================
# 11) Pages
# =========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# =========================
# 12) PDF Ingest → FAISS + Units
# =========================
@app.post("/ingest/pdf")
async def ingest_pdf_api(file: UploadFile = File(...)):
    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=80, separators=["\n\n", "\n", " ", ""]
    )
    chunks = [d for d in splitter.split_documents(documents) if len(d.page_content.strip()) >= 20]

    doc_id = "current_doc"
    shutil.rmtree(_vs_path(doc_id), ignore_errors=True)
    os.makedirs(_vs_path(doc_id), exist_ok=True)

    # === 임베딩 확인 부분 추가 ===
    sample_text = chunks[0].page_content[:200]  # 첫 청크 앞 200자만 확인
    sample_vector = embeddings.embed_query(sample_text)  # 1차원 list[float]

    vs = FAISS.from_documents(chunks, embedding=embeddings)
    persist_faiss(vs, doc_id)

    UNITS_BY_DOC[doc_id] = build_units_from_documents(documents)

    return {
        "doc_id": doc_id,
        "pages": len(documents),
        "chunks": len(chunks),
        "sample_text": sample_text,
        "sample_vector_dim": len(sample_vector),
        "sample_vector": sample_vector[:10],  # 앞 10차원만 확인
        "message": "PDF 업로드 및 벡터 저장 완료"
    }


# =========================
# 13) Chat (명령어/강의/RAG → TTS)
# =========================
def _pick_persona(req_persona: Optional[str]) -> Dict[str, str]:
    key = (req_persona or DEFAULT_PERSONA_KEY).strip()
    return PERSONAS.get(key, P_DEFAULT)

def _tts_paths_to_urls(paths: List[str]) -> Tuple[Optional[str], Optional[List[str]]]:
    if not paths:
        return None, None
    urls = [f"/audio/{os.path.basename(p)}" for p in paths]
    return urls[0], urls

@app.post("/chat", response_model=ChatResp)
async def chat_api(req: ChatReq):
    db = SessionLocal()
    doc_id = req.doc_id or "current_doc"
    P = _pick_persona(req.persona)

    # 벡터스토어 로드
    try:
        vectorstore = load_faiss(doc_id)
    except Exception:
        db.close()  # [FIX] 누수 방지
        return {"answer": "❗ 해당 문서를 찾을 수 없습니다. 먼저 PDF를 업로드하세요.", "used_refs": []}

    # user log
    save_interaction(db, session_id=req.session_id, role="user",
                     content=req.question, doc_id=doc_id)

    raw = req.question.strip()
    audio_url, audio_urls, audio_path = None, None, None

    # ===== 강의 명령어 처리 =====
    if raw in ("/teach", "/next", "/prev") or raw.startswith("/goto"):
        units = UNITS_BY_DOC.get(doc_id)
        if not units:
            try:
                metas = SessionLocal().query(DocumentMeta).filter_by(doc_id=doc_id).first()
                filename = metas.filename if metas else None
                if filename:
                    loader = PyPDFLoader(os.path.join(UPLOAD_DIR, filename))
                    documents = loader.load()
                    UNITS_BY_DOC[doc_id] = build_units_from_documents(documents)
                    units = UNITS_BY_DOC[doc_id]
            except Exception:
                pass
        if not units:
            db.close()
            return {"answer": "강의 단원이 준비되어 있지 않습니다. PDF를 다시 업로드해주세요.", "used_refs": []}

        cur = LECTURE_STATE.get(req.session_id, 0)

        if raw == "/prev":
            cur = max(0, cur - 1)
        elif raw == "/next":
            if cur < len(units) - 1:
                cur += 1
        elif raw.startswith("/goto"):
            m = re.search(r"/goto\s+(\d+)", raw)
            if m:
                page_target = int(m.group(1))
                idx = None
                for i, u in enumerate(units):
                    if u["page"] == page_target:
                        idx = i
                        break
                if idx is None:
                    if page_target <= units[0]["page"]:
                        idx = 0
                    elif page_target >= units[-1]["page"]:
                        idx = len(units) - 1
                    else:
                        idx = min(range(len(units)), key=lambda i: abs(units[i]["page"] - page_target))
                cur = idx

        LECTURE_STATE[req.session_id] = cur
        unit = units[cur]

        answer = teach_unit_text(P, unit)
        used_refs = [{"type": "page", "page": unit["page"]}]

        if req.want_tts:
            speed = req.tts_speed if req.tts_speed is not None else 0
            volume = req.tts_volume if req.tts_volume is not None else 0
            pitch = req.tts_pitch if req.tts_pitch is not None else 0
            paths = synthesize_clova_voice_premium_mp3(
                answer, speaker=P["speaker"], speed=speed, pitch=pitch, volume=volume
            )
            audio_url, audio_urls = _tts_paths_to_urls(paths)
            audio_path = paths[0] if paths else None
        else:
            audio_url, audio_urls, audio_path = None, None, None

        save_interaction(
            db,
            session_id=req.session_id,
            role="assistant",
            content=answer,
            doc_id=doc_id,
            refs={"mode": "lecture", "page": unit["page"]},
            audio_path=audio_path,
        )
        db.close()
        return ChatResp(answer=answer, audio_url=audio_url, audio_urls=audio_urls, used_refs=used_refs)

    # ===== Q&A (RetrievalQA) =====
    try:
        chain = build_qa_chain(P, vectorstore)
        res = chain({"query": raw})

        # [FIX] LangChain 버전 호환: result / output_text / answer 모두 대응
        answer = (
            (res.get("result") or res.get("output_text") or res.get("answer") or "")
        ).strip()
        if not answer:
            answer = "자료엔 이 내용이 없어. 대신 관련 키워드를 바꿔서 다시 물어보는 건 어때요?"

        srcs = res.get("source_documents", []) or []

        used_refs: List[Dict[str, Any]] = []
        for d in srcs:
            try:
                p0 = int(d.metadata.get("page", 0))
            except Exception:
                p0 = 0
            preview = (d.page_content or "").strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "…"
            used_refs.append({"type": "source", "page": p0 + 1, "preview": preview})

        # 중복 페이지 제거
        seen = set()
        deduped = []
        for r in used_refs:
            key = (r["type"], r["page"])
            if key not in seen:
                deduped.append(r); seen.add(key)
        used_refs = deduped[:5]

        if req.want_tts:
            speed = req.tts_speed if req.tts_speed is not None else 0
            volume = req.tts_volume if req.tts_volume is not None else 0
            pitch = req.tts_pitch if req.tts_pitch is not None else 0
            paths = synthesize_clova_voice_premium_mp3(
                answer, speaker=P["speaker"], speed=speed, pitch=pitch, volume=volume
            )
            audio_url, audio_urls = _tts_paths_to_urls(paths)
            audio_path = paths[0] if paths else None
        else:
            audio_url, audio_urls, audio_path = None, None, None

        save_interaction(
            db,
            session_id=req.session_id,
            role="assistant",
            content=answer,
            doc_id=doc_id,
            refs={"mode": "qa", "used_refs": used_refs},
            audio_path=audio_path,
        )
        db.close()
        return ChatResp(answer=answer, audio_url=audio_url, audio_urls=audio_urls, used_refs=used_refs)

    except Exception as e:
        err_msg = f"답변 중 오류가 발생했습니다: {e}"
        save_interaction(db, session_id=req.session_id, role="assistant", content=err_msg, doc_id=doc_id)
        db.close()
        return ChatResp(answer=err_msg, used_refs=[])

# =========================
# 14) Audio Static Serving
# =========================
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    from fastapi import HTTPException
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)

# =========================
# 15) Lecture Helpers
# =========================
@app.post("/lecture/reset")
async def lecture_reset(session_id: str):
    LECTURE_STATE.pop(session_id, None)
    return {"message": "ok", "session_id": session_id, "state": "reset"}

@app.get("/lecture/status")
async def lecture_status(session_id: str):
    cur = LECTURE_STATE.get(session_id, 0)
    return {"session_id": session_id, "current_index": cur}

# =========================
# 16) Persona Helpers
# =========================
@app.get("/personas")
async def list_personas():
    out = []
    for k, v in PERSONAS.items():
        out.append({"key": k, "display": v["display"], "sex": v["sex"], "speaker": v["speaker"]})
    return {"personas": out}

# =========================
# 17) Local run (optional)
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
