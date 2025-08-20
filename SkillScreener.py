"""
SkillScreener – Smart Resume Analyzer
A portfolio-ready AI Resume Screener with explainability, gap analysis, and modern UI.

How to run locally:
1) Create a virtual env (recommended)
   python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\\Scripts\\activate)
2) Install deps:
   pip install -U streamlit sentence-transformers scikit-learn PyPDF2 python-docx matplotlib
3) Start app:
   streamlit run app.py

Notes:
- Uses MiniLM embeddings for semantic similarity.
- Zero external API keys. Pure local.
- Works with PDF/DOCX/TXT resumes + pasted Job Description.
"""

import io
import re
import json
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# File readers
from PyPDF2 import PdfReader
from docx import Document

# Embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Config & Utilities
# -----------------------------
st.set_page_config(page_title="SkillScreener – Smart Resume Analyzer", layout="wide")

@st.cache_resource(show_spinner=False)
def load_embedder(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

EMBEDDER = load_embedder()

SKILL_TAXONOMY = [
    "python", "sql", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "keras", "xgboost", "nlp", "transformers", "bert", "cnn", "svm",
    "regression", "classification", "clustering", "time series", "statistics",
    "spark", "airflow", "mlflow", "docker", "kubernetes", "aws", "azure", "gcp"
]

@dataclass
class ScoreBreakdown:
    embed_score: float
    skill_match_score: float
    combined_score: float
    matched_skills: List[str]
    missing_skills: List[str]


def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().replace("\n", " "))


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def extract_text_from_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as bio:
        doc = Document(bio)
        return "\n".join([p.text for p in doc.paragraphs])


def detect_required_skills(jd_text: str, taxonomy: List[str]) -> List[str]:
    jd = normalize_text(jd_text)
    return sorted({s for s in taxonomy if re.search(rf"(?<!\\w){re.escape(s)}(?!\\w)", jd)})


def find_resume_skills(res_text: str, taxonomy: List[str]) -> List[str]:
    resume = normalize_text(res_text)
    return sorted({s for s in taxonomy if re.search(rf"(?<!\\w){re.escape(s)}(?!\\w)", resume)})


def compute_similarity(a: str, b: str) -> float:
    emb_a, emb_b = EMBEDDER.encode([a]), EMBEDDER.encode([b])
    return float(cosine_similarity(emb_a, emb_b)[0][0])


def tfidf_top_terms(text: str, n: int = 10) -> List[Tuple[str, float]]:
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform([text])
    tfidf = {t: X[0, i] for t, i in vec.vocabulary_.items()}
    return sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:n]


def score_resume(jd: str, resume: str, required_skills: List[str], w_embed=0.7) -> ScoreBreakdown:
    embed_score = compute_similarity(jd, resume)
    resume_skills = find_resume_skills(resume, SKILL_TAXONOMY)
    matched = sorted(set(required_skills).intersection(resume_skills))
    missing = sorted(set(required_skills) - set(matched))
    skill_match_score = len(matched) / len(required_skills) if required_skills else 0.0
    combined = w_embed * embed_score + (1 - w_embed) * skill_match_score
    return ScoreBreakdown(round(embed_score, 3), round(skill_match_score, 3), round(combined, 3), matched, missing)


def plot_skill_chart(matched: List[str], missing: List[str]):
    labels = ["Matched Skills", "Missing Skills"]
    values = [len(matched), len(missing)]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=["#2ECC71", "#E74C3C"])
    ax.set_title("Skills Coverage")
    st.pyplot(fig)


def plot_radar_chart(required_skills: List[str], matched: List[str]):
    if not required_skills:
        st.info("No required skills detected in JD for radar chart.")
        return

    categories = required_skills
    values = [1 if skill in matched else 0 for skill in categories]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#2E86C1", linewidth=2)
    ax.fill(angles, values, color="#5DADE2", alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("Skill Radar Chart", size=14, color="#2E86C1", y=1.1)
    st.pyplot(fig)


# -----------------------------
# UI
# -----------------------------
st.markdown("""<h1 style='text-align: center; color:#2E86C1;'>SkillScreener – Smart Resume Analyzer</h1>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Upload Inputs")
    resume_file = st.file_uploader("Upload Resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
    jd_text = st.text_area("Paste Job Description", height=200)
    role_title = st.text_input("Role Title", placeholder="e.g., Data Scientist")
    w_embed = st.slider("Weight: Semantic Similarity", 0.0, 1.0, 0.7, 0.05)

if st.button("Analyze Fit", type="primary"):
    if not resume_file or not jd_text.strip():
        st.error("Please upload a resume and paste a job description.")
        st.stop()

    ext = resume_file.name.split(".")[-1].lower()
    if ext == "pdf":
        resume_text = extract_text_from_pdf(resume_file.read())
    elif ext == "docx":
        resume_text = extract_text_from_docx(resume_file.read())
    else:
        resume_text = resume_file.read().decode("utf-8", errors="ignore")

    required_skills = detect_required_skills(jd_text, SKILL_TAXONOMY)
    breakdown = score_resume(jd_text, resume_text, required_skills, w_embed=w_embed)

    # Decision Logic
    decision = "Accepted ✅ Candidate is suitable for this job." if breakdown.combined_score >= 0.7 else "Rejected ❌ Candidate does not meet requirements."

    # Tabs for results
    tab1, tab2, tab3 = st.tabs(["Fit Score", "Skills Analysis", "Suggestions"])

    with tab1:
        st.subheader("Overall Fit Score")
        color = "#2ECC71" if breakdown.combined_score > 0.75 else ("#E67E22" if breakdown.combined_score > 0.5 else "#E74C3C")
        st.markdown(f"<h2 style='color:{color};'>{breakdown.combined_score:.2f}</h2>", unsafe_allow_html=True)
        st.write(f"Semantic Similarity: {breakdown.embed_score}")
        st.write(f"Skill Match: {breakdown.skill_match_score}")
        if "Accepted" in decision:
            st.success(decision)
        else:
            st.error(decision)
        plot_skill_chart(breakdown.matched_skills, breakdown.missing_skills)
        plot_radar_chart(required_skills, breakdown.matched_skills)

    with tab2:
        st.subheader("Matched Skills")
        st.success(", ".join(breakdown.matched_skills) if breakdown.matched_skills else "No skills matched.")
        st.subheader("Missing Skills")
        st.error(", ".join(breakdown.missing_skills) if breakdown.missing_skills else "None – Great Match!")
        st.subheader("Top JD Terms")
        st.write(", ".join([t for t, _ in tfidf_top_terms(jd_text)]))

    with tab3:
        st.subheader("Resume Improvement Suggestions")
        if breakdown.missing_skills:
            for s in breakdown.missing_skills:
                st.write(f"- Add a project or bullet point showing experience with **{s}**.")
        else:
            st.write("Your resume aligns very well with the JD!")

    report = {
        "combined_score": breakdown.combined_score,
        "embedding_score": breakdown.embed_score,
        "skill_match_score": breakdown.skill_match_score,
        "required_skills": required_skills,
        "matched_skills": breakdown.matched_skills,
        "missing_skills": breakdown.missing_skills,
        "decision": decision
    }
    st.download_button("Download JSON Report", data=json.dumps(report, indent=2).encode("utf-8"), file_name="skillscreener_report.json")

st.markdown("---")
st.info("This is skill screening for the candidate.")
