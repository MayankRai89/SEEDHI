import pandas as pd
from sentence_transformers import SentenceTransformer, util # type: ignore
from PyPDF2 import PdfReader
import docx
import re
from io import BytesIO
from html import unescape

# Load jobs dataset with normalized column names
JOBS_DF = pd.read_csv("app/jobs.csv")
JOBS_DF.columns = [col.strip().lower().replace(" ", "_") for col in JOBS_DF.columns]

# Load sentence transformer model
MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_bytes(file_bytes, file_type="pdf"):
    text = ""
    if file_type == "pdf":
        reader = PdfReader(BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file_type == "docx":
        doc = docx.Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif isinstance(file_bytes, str):
        text = file_bytes
    return text


def pre_process_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()


def extract_url_and_label(raw_html_or_text: str):
    if not isinstance(raw_html_or_text, str):
        return ("", "")
    s = raw_html_or_text.strip()
    m = re.search(r'href=[\'"]([^\'"]+)[\'"]', s, flags=re.IGNORECASE)
    if m:
        url = m.group(1).strip()
        label_match = re.search(r'>(.*?)<', s)
        label = label_match.group(1).strip() if label_match else ''
        return (unescape(url), unescape(label or "Apply"))
    m2 = re.search(r'(https?://[^\s"<>]+)', s)
    if m2:
        return (unescape(m2.group(1)), "Apply")
    return ("", "")


def tokenize_text_words(text: str):
    if not text:
        return set()
    cleaned = pre_process_text(text)
    tokens = re.findall(r'\w+', cleaned)
    return set(tokens)


def match_resume_with_jobs_ai(resume_text: str, min_score=0.35):
    resume_tokens = tokenize_text_words(resume_text)
    resume_embedding = MODEL.encode(resume_text, convert_to_tensor=True)

    recommendations = []

    for _, row in JOBS_DF.iterrows():
        company = str(row.get("company", "") or "")
        description = str(row.get("description", "") or "")
        role = str(row.get("role", "") or "")
        required_skills_raw = str(row.get("required_skills", "") or "")
        location = str(row.get("location", "") or "")
        date_posted = str(row.get("date_posted", "") or "")
        salary = str(row.get("salary", "") or "Not specified")  # <-- Added salary
        application_raw = row.get("application", "") or ""

        job_text = f"{role} at {company} in {location} requiring {required_skills_raw}"
        job_embedding = MODEL.encode(job_text, convert_to_tensor=True)
        score = util.cos_sim(resume_embedding, job_embedding).item()

        skill_tokens = tokenize_text_words(required_skills_raw)
        matched = sorted(list(resume_tokens.intersection(skill_tokens)))

        application_url, application_label = extract_url_and_label(str(application_raw))

        if score >= min_score or len(matched) > 0:
            recommendations.append({
                "company": company,
                "description": description,
                "role": role,
                "required_skills": required_skills_raw,
                "location": location,
                "date_posted": date_posted,
                "salary": salary,  # <-- Include salary in recommendation
                "application_url": application_url,
                "application_label": application_label,
                "similarity_score": round(float(score), 3),
                "keyword_matches": len(matched),
                "matched_keywords": matched
            })

    recommendations = sorted(
        recommendations,
        key=lambda x: (x["keyword_matches"], x["similarity_score"]),
        reverse=True
    )

    return recommendations
