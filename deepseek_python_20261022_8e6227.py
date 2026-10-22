import io
import json
import os
import sqlite3
import tempfile
import time
import uuid
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# ==============================================================================
# CONSTANTES
# ==============================================================================
DB_PATH = "simulador.db"
GEMINI_MODEL = "gemini-3.8-flash"  # Modelo estável mais recente
SIMULADO_TARGET_MIN = 20
INLINE_PDF_LIMIT_MB = 18
MAX_OUTPUT_TOKENS = 65536

st.set_page_config(
    page_title="CGU 2026 — Simulador de TI & Governança",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# AVISO VISUAL DE DIAGNÓSTICO
# ==============================================================================
st.error(f"⚠️ DIAGNÓSTICO: O modelo configurado neste arquivo é `{GEMINI_MODEL}`. Se não for o 3.8, você editou o arquivo errado ou não salvou.")

# ==============================================================================
# TEMAS
# ==============================================================================
themes = {
    "Midnight Obsidian": {"bg": "#0b0f19", "panel": "#111827", "card": "#1f2937",
                          "text": "#FFFFFF", "accent": "#3b82f6", "border": "#374151"},
    "Deep Ocean": {"bg": "#0f172a", "panel": "#1e293b", "card": "#334155",
                   "text": "#F8FAFC", "accent": "#60a5fa", "border": "#475569"},
    "Clean Light": {"bg": "#f8fafc", "panel": "#ffffff", "card": "#f1f5f9",
                    "text": "#0f172a", "accent": "#2563eb", "border": "#cbd5e1"},
}
if "theme" not in st.session_state or st.session_state.theme not in themes:
    st.session_state.theme = list(themes.keys())[0]
t = themes[st.session_state.theme]

# ==============================================================================
# ESTILO
# ==============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .main {{ background-color: {t['bg']}; }}
    div[data-testid="stMetric"] {{
        background-color: {t['panel']} !important;
        padding: 18px; border-radius: 12px;
        border: 1px solid {t['border']} !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }}
    div[data-testid="stMetric"] label {{ color: {t['text']} !important; opacity: .85; font-weight: 500; }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {t['accent']} !important; font-weight: 700; }}
    .stButton>button {{ border-radius: 8px; font-weight: 600; transition: all .2s ease-in-out; }}
    div[data-testid="stSidebar"] {{ background-color: {t['panel']}; border-right: 1px solid {t['border']}; }}
    .badge {{
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: .75rem; font-weight: 600; margin-right: 6px;
    }}
    .badge-tf {{ background: #1e40af; color: #fff; }}
    .badge-mc {{ background: #7c3aed; color: #fff; }}
    .badge-topic {{ background: {t['border']}; color: {t['text']}; }}
    h1, h2, h3, h4, h5, p, label, span, div, .stRadio label, .stMarkdown {{ color: {t['text']} !important; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSÃO
# ==============================================================================
_defaults = {
    "current_index": 0, "gemini_api_key": "", "simulado_active": False,
    "simulado_answers": {}, "simulado_start_time": None, "simulado_run_id": "",
    "filter_sig": None, "flash": None,
}
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)

def flash(msg: str, level: str = "success") -> None:
    st.session_state.flash = (level, msg)

def _render_flash() -> None:
    if st.session_state.flash:
        level, msg = st.session_state.flash
        st.session_state.flash = None
        getattr(st, level, st.info)(msg)

# ==============================================================================
# BANCO DE DADOS
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db() -> None:
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, type TEXT, format TEXT,
            text TEXT, options TEXT, answer TEXT, exp TEXT, trap TEXT, imported INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS user_progress (
            question_id INTEGER PRIMARY KEY, answer TEXT, favorite INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_questions_topic ON questions(topic);
        CREATE INDEX IF NOT EXISTS idx_progress_fav ON user_progress(favorite);
    """)
    conn.commit()
    if conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 0:
        _seed_questions(conn)

def _seed_questions(conn: sqlite3.Connection) -> None:
    seed = [
        ("Alta disponibilidade", "CEBRASPE", "TF",
         "Em uma arquitetura de alta disponibilidade com duas zonas de disponibilidade, distribuir "
         "instâncias entre as zonas elimina a necessidade de testar os procedimentos de recuperação e failover.",
         "[]", "False", "Errado. Redundância reduz pontos únicos de falha, mas não comprova que a recuperação funcionará.",
         "Pegadinha: tratar redundância arquitetural como garantia de recuperação.", 0),
        ("Backup e recuperação", "CEBRASPE", "TF",
         "Um banco recebe backup completo no domingo e backups incrementais de segunda a sexta.",
         "[]", "True", "Certo. O incremental registra alterações desde o backup anterior.",
         "Pegadinha: restaurar somente o incremental de quinta não reconstrói a base.", 0),
        ("Governança COBIT", "FGV", "MC",
         "No contexto do COBIT 2019, o conceito que diferencia Governança de Gestão está relacionado a:",
         json.dumps([
             {"label": "A", "text": "A governança avalia, direciona e monitora; a gestão planeja, constrói, executa e monitora."},
             {"label": "B", "text": "A governança é de responsabilidade da diretoria executiva, enquanto a gestão cabe ao conselho."},
             {"label": "C", "text": "A gestão define a estratégia global e a governança lida com a operação diária."},
             {"label": "D", "text": "Não há diferença funcional entre governança e gestão no COBIT 2019."},
             {"label": "E", "text": "A governança substitui completamente os processos do ITIL 4."},
         ], ensure_ascii=False), "A", "O COBIT 2019 estabelece que Governança avalia (EDM) e Gestão planeja (PBRM).",
         "Pegadinha: inverter as atribuições.", 0),
    ]
    conn.executemany(
        "INSERT INTO questions (topic, type, format, text, options, answer, exp, trap, imported) VALUES (?,?,?,?,?,?,?,?,?)",
        seed,
    )
    conn.commit()

def _normalize_tf(raw: Any) -> bool:
    if isinstance(raw, bool): return raw
    return str(raw).strip().lower() in ("true", "c", "certo", "1", "v", "verdadeiro")

def _normalize_mc(raw: Any) -> str:
    return str(raw).strip().upper()[:1]

@st.cache_data(show_spinner=False)
def load_questions() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT id, topic, type, format, text, options, answer, exp, trap, imported FROM questions").fetchall()
    out = []
    for r in rows:
        fmt = r["format"]
        answer: Any = _normalize_tf(r["answer"]) if fmt == "TF" else _normalize_mc(r["answer"])
        out.append({
            "id": r["id"], "topic": r["topic"], "type": r["type"], "format": fmt, "text": r["text"],
            "options": json.loads(r["options"]) if r["options"] else [], "answer": answer, "exp": r["exp"], "trap": r["trap"], "imported": r["imported"],
        })
    return out

def load_progress() -> tuple[dict, set]:
    conn = get_conn()
    rows = conn.execute("SELECT question_id, answer, favorite FROM user_progress").fetchall()
    answers, favorites = {}, set()
    for r in rows:
        if r["answer"] is not None:
            raw = r["answer"]
            answers[r["question_id"]] = (raw == "True") if raw in ("True", "False") else raw
        if r["favorite"] == 1: favorites.add(r["question_id"])
    return answers, favorites

def save_answer(qid: int, ans: Any) -> None:
    if ans is None: return
    conn = get_conn()
    conn.execute("INSERT INTO user_progress (question_id, answer) VALUES (?, ?) ON CONFLICT(question_id) DO UPDATE SET answer = excluded.answer", (qid, str(ans)))
    conn.commit()

def toggle_favorite(qid: int) -> None:
    conn = get_conn()
    row = conn.execute("SELECT favorite FROM user_progress WHERE question_id = ?", (qid,)).fetchone()
    new_val = 0 if (row and row["favorite"] == 1) else 1
    conn.execute("INSERT INTO user_progress (question_id, favorite) VALUES (?, ?) ON CONFLICT(question_id) DO UPDATE SET favorite = excluded.favorite", (qid, new_val))
    conn.commit()

def reset_user_progress() -> None:
    conn = get_conn()
    conn.execute("DELETE FROM user_progress")
    conn.commit()
    st.session_state.current_index = 0

def delete_question(qid: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM questions WHERE id = ?", (qid,))
    conn.execute("DELETE FROM user_progress WHERE question_id = ?", (qid,))
    conn.commit()
    load_questions.clear()

# ==============================================================================
# IA — VISÃO NATIVA DE PDF
# ==============================================================================
def _build_extraction_prompt(default_topic: str) -> str:
    return f"""Você é um especialista em parsing e estruturação de provas de concursos públicos.
Analise o documento PDF anexo (que pode conter texto, imagens escaneadas, tabelas, gráficos e layouts complexos) e extraia TODAS as questões identificadas.

Assunto Padrão: {default_topic}
Regras:
1. Formato "TF" → questões de Certo/Errado (Cebraspe).
2. Formato "MC" → múltipla escolha A–E (FGV, FCC, Vunesp).
3. Responda ESTRITAMENTE com um array JSON válido, sem marcadores markdown, sem texto fora do JSON.
4. Se o documento não tiver questões, retorne [].

Estrutura esperada:
[
  {{
    "topic": "{default_topic}",
    "type": "FGV/CEBRASPE/...",
    "format": "TF" ou "MC",
    "text": "Enunciado completo da questão",
    "options": [{{"label": "A", "text": "Texto da alternativa A"}}, {{"label": "B", "text": "Texto da alternativa B"}}],
    "answer": true/false para TF, ou "A"/"B"/"C"/"D"/"E" para MC,
    "exp": "Fundamentação técnica da resposta",
    "trap": "A pegadinha clássica da banca presente no item"
  }}
]"""

def _is_valid_question(q: Any) -> bool:
    if not isinstance(q, dict) or not str(q.get("text", "")).strip(): return False
    fmt = str(q.get("format", "TF")).strip().upper()
    if fmt not in ("TF", "MC") or "answer" not in q: return False
    if fmt == "MC":
        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) < 2: return False
    return True

def _call_gemini_extraction(client: genai.Client, contents: list, prompt: str) -> list[dict]:
    st.info(f"🔄 Enviando requisição para o Google usando: **{GEMINI_MODEL}**")
    
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1, max_output_tokens=MAX_OUTPUT_TOKENS),
    )
    data = json.loads(response.text)
    if not isinstance(data, list): raise ValueError("A resposta da IA não é um array JSON.")
    return [q for q in data if _is_valid_question(q)]

def ai_extract_questions_from_pdf(pdf_bytes: bytes, api_key: str, default_topic: str, max_retries: int = 2) -> list[dict]:
    if not pdf_bytes: raise ValueError("PDF vazio.")
    client = genai.Client(api_key=api_key)
    prompt = _build_extraction_prompt(default_topic)
    use_files_api = (len(pdf_bytes) / (1024 * 1024)) > INLINE_PDF_LIMIT_MB

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        tmp_path, uploaded_name = None, None
        try:
            if not use_files_api:
                pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
                return _call_gemini_extraction(client, [pdf_part, prompt], prompt)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            uploaded = client.files.upload(file=tmp_path)
            uploaded_name = uploaded.name

            for _ in range(120):
                info = client.files.get(name=uploaded_name)
                state = getattr(info.state, "name", str(info.state))
                if state == "ACTIVE": break
                if state == "FAILED": raise RuntimeError("O upload do PDF falhou.")
                time.sleep(1)
            else: raise TimeoutError("O PDF não ficou pronto (ACTIVE).")

            uploaded = client.files.get(name=uploaded_name)
            return _call_gemini_extraction(client, [uploaded, prompt], prompt)
        except Exception as e:
            last_err = e
            if attempt < max_retries: time.sleep(1.5 * (attempt + 1))
        finally:
            if uploaded_name:
                try: client.files.delete(name=uploaded_name)
                except Exception: pass
            if tmp_path and os.path.exists(tmp_path):
                try: os.unlink(tmp_path)
                except Exception: pass
    raise RuntimeError(f"Falha ao extrair questões do PDF: {last_err}")

def ai_tutor(question_text: str, user_question: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    prompt = f"Você é um professor de TI para concursos.\n\nEnunciado:\n\"{question_text}\"\n\nDúvida:\n\"{user_question}\"\n\nResponda didaticamente."
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text

def pdf_page_count(pdf_bytes: bytes) -> Optional[int]:
    if not HAS_PYPDF: return None
    try: return len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception: return None

# ==============================================================================
# UI HELPERS
# ==============================================================================
def render_mc(options: list[dict], key: str, saved: Optional[str]) -> Optional[str]:
    labels = [f"({o['label']}) {o['text']}" for o in options]
    label_to_val = {f"({o['label']}) {o['text']}": o["label"] for o in options}
    idx = next((i for i, o in enumerate(options) if o["label"] == saved), None)
    chosen = st.radio("Escolha a alternativa:", labels, index=idx, key=key)
    return label_to_val.get(chosen) if chosen else None

def render_tf(key: str, saved: Optional[bool]) -> Optional[bool]:
    labels = ["Certo", "Errado"]
    idx = 0 if saved is True else (1 if saved is False else None)
    chosen = st.radio("Julgue o item:", labels, index=idx, key=key)
    return (chosen == "Certo") if chosen else None

def badge(q: dict) -> str:
    fmt = "TF" if q["format"] == "TF" else "MC"
    cls = "badge-tf" if fmt == "TF" else "badge-mc"
    return f'<span class="badge {cls}">{fmt}</span><span class="badge badge-topic">{q["topic"]}</span>'

# ==============================================================================
# INICIALIZAÇÃO
# ==============================================================================
init_db()

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🎯 CGU 2026")
    
    if st.button("🧹 Limpar Cache e Reiniciar"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
        
    st.divider()
    questions = load_questions()
    answers, favorites = load_progress()

    topics = ["Todos os assuntos"] + sorted({q["topic"] for q in questions})
    selected_topic = st.selectbox("📚 Filtrar por Assunto", topics)
    filter_opt = st.selectbox("🔍 Status da Questão", ["Todas as questões", "Não respondidas", "Errei", "Acertei", "Favoritas"])
    mode = st.radio("⚡ Modo de Estudo", ["Estudo Interativo (Feedback na hora)", "Simulado Cronometrado", "Estatísticas & Banco"])
    
    st.divider()
    st.header("🔑 Conexão IA")
    api_key_input = st.text_input("API Key do Google Gemini", value=st.session_state.gemini_api_key, type="password")
    if api_key_input: st.session_state.gemini_api_key = api_key_input

# ==============================================================================
# MÉTRICAS
# ==============================================================================
answer_by_id = {q["id"]: q["answer"] for q in questions}
correct_q = sum(1 for qid, ans in answers.items() if answer_by_id.get(qid) == ans)
answered_q = len(answers)
accuracy = (correct_q / answered_q * 100) if answered_q else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de Questões", len(questions))
c2.metric("Respondidas", answered_q)
c3.metric("Acertos", correct_q)
c4.metric("Aproveitamento", f"{accuracy:.1f}%")
st.divider()
_render_flash()

# Filtros
filtered = questions.copy()
if selected_topic != "Todos os assuntos": filtered = [q for q in filtered if q["topic"] == selected_topic]
if filter_opt == "Não respondidas": filtered = [q for q in filtered if q["id"] not in answers]
elif filter_opt == "Errei": filtered = [q for q in filtered if q["id"] in answers and answers[q["id"]] != q["answer"]]
elif filter_opt == "Acertei": filtered = [q for q in filtered if q["id"] in answers and answers[q["id"]] == q["answer"]]
elif filter_opt == "Favoritas": filtered = [q for q in filtered if q["id"] in favorites]

sig = (selected_topic, filter_opt)
if st.session_state.filter_sig != sig:
    st.session_state.filter_sig = sig
    st.session_state.current_index = 0
if filtered and st.session_state.current_index >= len(filtered): st.session_state.current_index = 0

# ==============================================================================
# MODO: SIMULADO
# ==============================================================================
if "Simulado" in mode:
    st.subheader("⏱️ Simulado Oficial Cronometrado")
    if not filtered: st.warning("Nenhuma questão atende ao filtro.")
    elif not st.session_state.simulado_active:
        st.info(f"O simulado será gerado com **{len(filtered)}** questões.")
        if st.button("🏁 Iniciar Prova Simulado", type="primary"):
            st.session_state.simulado_active = True
            st.session_state.simulado_answers = {}
            st.session_state.simulado_start_time = time.time()
            st.session_state.simulado_run_id = uuid.uuid4().hex[:8]
            st.rerun()
    else:
        elapsed = int(time.time() - st.session_state.simulado_start_time)
        mins, secs = divmod(elapsed, 60)
        st.warning(f"⏱️ Tempo Decorrido: **{mins:02d}:{secs:02d}**")

        run_id = st.session_state.simulado_run_id
        for idx, q in enumerate(filtered):
            st.markdown(f"**Item {idx+1} de {len(filtered)}** — *{q['topic']}*")
            st.markdown(q["text"])
            key = f"sim_{run_id}_{q['id']}"
            val = render_mc(q["options"], key, None) if q["format"] == "MC" else render_tf(key, None)
            if val is not None: st.session_state.simulado_answers[q["id"]] = val
            else: st.session_state.simulado_answers.pop(q["id"], None)
            st.write("---")

        if st.button("✅ Enviar e Finalizar Simulado", type="primary"):
            score = sum(1 for q in filtered if st.session_state.simulado_answers.get(q["id"]) == q["answer"])
            for q in filtered: save_answer(q["id"], st.session_state.simulado_answers.get(q["id"]))
            st.balloons()
            flash(f"Simulado concluído: **{score}/{len(filtered)}**")
            st.session_state.simulado_active = False
            st.session_state.simulado_answers = {}
            st.rerun()

# ==============================================================================
# MODO: ESTATÍSTICAS (COM GRÁFICOS)
# ==============================================================================
elif "Estatísticas" in mode:
    st.subheader("📊 Desempenho Detalhado")

    if not answers:
        st.info("Você ainda não respondeu nenhuma questão. Volte ao modo de estudo para começar a gerar estatísticas!")
    else:
        # Prepara os dados para o gráfico
        stats_list = []
        for q in questions:
            tp = q["topic"]
            s = next((item for item in stats_list if item["Assunto"] == tp), None)
            if not s:
                s = {"Assunto": tp, "Total": 0, "Respondidas": 0, "Acertos": 0}
                stats_list.append(s)
            s["Total"] += 1
            if q["id"] in answers:
                s["Respondidas"] += 1
                if answers[q["id"]] == q["answer"]:
                    s["Acertos"] += 1

        # Calcula aproveitamento
        for s in stats_list:
            s["Aproveitamento (%)"] = (s["Acertos"] / s["Respondidas"] * 100) if s["Respondidas"] > 0 else 0.0

        df = pd.DataFrame(stats_list)
        
        # 1. Gráfico de Barras (Aproveitamento por Assunto)
        st.write("**Aproveitamento por Assunto**")
        df_bar = df[df["Respondidas"] > 0].sort_values("Aproveitamento (%)", ascending=False)
        
        if not df_bar.empty:
            fig_bar = px.bar(
                df_bar, 
                x="Assunto", 
                y="Aproveitamento (%)", 
                text="Aproveitamento (%)",
                color="Aproveitamento (%)",
                color_continuous_scale="Blues",
                range_y=[0, 100]
            )
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=t['text']),
                margin=dict(t=30, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Responda questões para gerar o gráfico de barras.")

        st.divider()

        # 2. Gráfico de Pizza (Distribuição Geral)
        st.write("**Distribuição Geral das Questões**")
        total_respondidas = len(answers)
        total_erradas = total_respondidas - correct_q
        total_nao_respondidas = len(questions) - total_respondidas

        fig_pie = px.pie(
            values=[correct_q, total_erradas, total_nao_respondidas],
            names=["Acertos", "Erros", "Não Respondidas"],
            color=["Acertos", "Erros", "Não Respondidas"],
            color_discrete_map={
                "Acertos": "#22c55e",  # Verde
                "Erros": "#ef4444",    # Vermelho
                "Não Respondidas": "#64748b" # Cinza
            },
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=t['text']),
            margin=dict(t=30, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        # 3. Tabela Detalhada
        st.write("**Tabela Detalhada por Assunto**")
        st.dataframe(
            df.style.format({
                "Aproveitamento (%)": "{:.1f}%"
            }).background_gradient(cmap="Blues", subset=["Aproveitamento (%)"]),
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.subheader("🛠️ Gerenciar e Apagar Questões do Banco")
    for q in questions:
        col_a, col_b = st.columns([5, 1])
        with col_a:
            st.write(f"**[{q['id']}] {q['topic']}** — {q['text'][:120]}...")
        with col_b:
            if st.button("Excluir", key=f"del_{q['id']}"):
                delete_question(q["id"])
                flash(f"Questão #{q['id']} excluída.")
                st.rerun()

# ==============================================================================
# MODO: ESTUDO
# ==============================================================================
else:
    if not filtered: st.info("Nenhuma questão encontrada.")
    else:
        q = filtered[st.session_state.current_index]
        h1, h2 = st.columns([4, 1])
        with h1: st.subheader(f"Questão {st.session_state.current_index + 1} de {len(filtered)}")
        with h2: st.markdown(badge(q), unsafe_allow_html=True)
        
        st.markdown(q["text"])
        saved_ans = answers.get(q["id"])

        if q["format"] == "MC": user_val = render_mc(q["options"], f"ans_mc_{q['id']}", saved=saved_ans)
        else: user_val = render_tf(f"ans_tf_{q['id']}", saved=saved_ans)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Confirmar Resposta", type="primary", disabled=user_val is None):
                save_answer(q["id"], user_val)
                st.rerun()
        with b2:
            if st.button("★ Remover Favorita" if q["id"] in favorites else "☆ Favoritar"):
                toggle_favorite(q["id"])
                st.rerun()

        if saved_ans is not None:
            if saved_ans == q["answer"]: st.success(f"**Correta!** Gabarito: **{q['answer']}**")
            else: st.error(f"**Incorreta.** Gabarito: **{q['answer']}**")
            st.info(f"**Explicação:** {q.get('exp') or '—'}\n\n**⚠️ Pegadinha:** {q.get('trap') or '—'}")
            
            with st.expander("🤖 Perguntar ao Tutor IA (Google Gemini)"):
                doubt = st.text_input("Qual a sua dúvida técnica?", key=f"doubt_{q['id']}")
                if st.button("Enviar Pergunta ao Tutor", key=f"send_{q['id']}"):
                    if not st.session_state.gemini_api_key:
                        st.error("Insira sua API Key do Gemini na barra lateral.")
                    elif not doubt.strip():
                        st.warning("Digite sua dúvida antes de enviar.")
                    else:
                        with st.spinner("O Tutor está formulando a explicação..."):
                            try:
                                resp = ai_tutor(q["text"], doubt, st.session_state.gemini_api_key)
                                st.markdown(f"**Resposta do Mentor IA:**\n\n{resp}")
                            except Exception as e:
                                st.error(f"Erro ao conectar com a IA: {e}")

        st.divider()
        n1, n2 = st.columns(2)
        with n1:
            if st.button("← Anterior") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
        with n2:
            if st.button("Próxima →") and st.session_state.current_index < len(filtered) - 1:
                st.session_state.current_index += 1
                st.rerun()

# ==============================================================================
# IMPORTAÇÃO DE PDF
# ==============================================================================
st.divider()
with st.expander("📄 Importar Prova a partir de PDF (Visão Nativa do Gemini)"):
    uploaded_pdf = st.file_uploader("Selecione o arquivo PDF", type=["pdf"])
    default_topic = st.text_input("Assunto padrão", "Governança e Infraestrutura")

    if uploaded_pdf and st.button("🚀 Processar PDF com IA", type="primary"):
        if not st.session_state.gemini_api_key:
            st.error("Insira sua API Key do Gemini na barra lateral primeiro.")
        else:
            with st.spinner("Enviando PDF ao Gemini..."):
                try:
                    parsed = ai_extract_questions_from_pdf(uploaded_pdf.getvalue(), st.session_state.gemini_api_key, default_topic)
                    if not parsed:
                        st.warning("Nenhuma questão identificada no PDF.")
                    else:
                        conn = get_conn()
                        for item in parsed:
                            fmt = str(item.get("format", "TF")).upper()
                            ans = str(_normalize_tf(item["answer"])) if fmt == "TF" else _normalize_mc(item["answer"])
                            conn.execute("INSERT INTO questions (topic, type, format, text, options, answer, exp, trap, imported) VALUES (?,?,?,?,?,?,?,?,1)",
                                         (item.get("topic", default_topic), item.get("type", "Concurso"), fmt, item["text"], json.dumps(item.get("options", []), ensure_ascii=False), ans, item.get("exp", ""), item.get("trap", "")))
                        conn.commit()
                        load_questions.clear()
                        flash(f"✅ {len(parsed)} questões importadas com sucesso!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Erro ao processar PDF: {e}")