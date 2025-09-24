import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import docx
import re
from io import BytesIO
from html import unescape

# Load jobs dataset with normalized column names
JOBS_DF = pd.read_csv("app/jobs.csv")
JOBS_DF.columns = [col.strip().lower().replace(" ", "_") for col in JOBS_DF.columns]

# Load sentence transformer model (keep your model)
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
    # Lowercase and remove extra punctuation (keep alphanumeric + spaces)
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()


def extract_url_and_label(raw_html_or_text: str):
    """
    Accepts either a raw URL, an HTML anchor string (<a href="...">Text</a>)
    or arbitrary text. Returns (url, label).
    """
    if not isinstance(raw_html_or_text, str):
        return ("", "")

    s = raw_html_or_text.strip()

    # 1) Try to find href inside an anchor tag
    m = re.search(r'href=[\'"]([^\'"]+)[\'"]', s, flags=re.IGNORECASE)
    if m:
        url = m.group(1).strip()
        # try to extract anchor text label
        label_match = re.search(r'>(.*?)<', s)
        label = label_match.group(1).strip() if label_match else ''
        return (unescape(url), unescape(label or "Apply"))

    # 2) Otherwise, try to find a bare URL in the string
    m2 = re.search(r'(https?://[^\s"<>]+)', s)
    if m2:
        return (unescape(m2.group(1)), "Apply")

    # 3) Nothing useful
    return ("", "")


def tokenize_text_words(text: str):
    """
    Return a set of normalized tokens from text. Keeps alphanumerics.
    Splits on all non-word characters.
    """
    if not text:
        return set()
    cleaned = pre_process_text(text)
    tokens = re.findall(r'\w+', cleaned)  # words and numbers
    return set(tokens)


def match_resume_with_jobs_ai(resume_text: str, min_score=0.35):
    """
    Returns recommendations list where each item contains:
      company, description, role, required_skills, location, date_posted,
      application_url, application_label, similarity_score,
      keyword_matches (int), matched_keywords (list)
    """
    # Prepare resume tokens
    resume_tokens = tokenize_text_words(resume_text)

    # Embedding once
    resume_embedding = MODEL.encode(resume_text, convert_to_tensor=True)

    recommendations = []

    for _, row in JOBS_DF.iterrows():
        # Safely read fields (use .get-like logic)
        company = str(row.get("company", "") or "")
        description = str(row.get("description", "") or "")
        role = str(row.get("role", "") or "")
        required_skills_raw = str(row.get("required_skills", "") or "")
        location = str(row.get("location", "") or "")
        date_posted = str(row.get("date_posted", "") or row.get("date_posted", "") or "")
        application_raw = row.get("application", "") or row.get("application", "")

        # Build job text for semantic matching
        job_text = f"{role} at {company} in {location} requiring {required_skills_raw}"

        # Embedding and similarity
        job_embedding = MODEL.encode(job_text, convert_to_tensor=True)
        score = util.cos_sim(resume_embedding, job_embedding).item()

        # Tokenize required skills properly (split on non-word characters)
        skill_tokens = tokenize_text_words(required_skills_raw)

        # Compute intersection (matched keywords)
        matched = sorted(list(resume_tokens.intersection(skill_tokens)))

        # Extract clean application URL & label
        application_url, application_label = extract_url_and_label(str(application_raw))

        # Only include jobs above a minimum semantic similarity (tunable)
        if score >= min_score or len(matched) > 0:
            recommendations.append({
                "company": company,
                "description": description,
                "role": role,
                "required_skills": required_skills_raw,
                "location": location,
                "date_posted": date_posted,
                "application_url": application_url,
                "application_label": application_label,
                "similarity_score": round(float(score), 3),
                "keyword_matches": len(matched),
                "matched_keywords": matched
            })

    # Sort recommended jobs: first by number of matched keywords, then similarity score
    recommendations = sorted(
        recommendations,
        key=lambda x: (x["keyword_matches"], x["similarity_score"]),
        reverse=True
    )

    return recommendations
