from fastapi import APIRouter, UploadFile, File
from app.service import extract_text_from_bytes, pre_process_text, match_resume_with_jobs_ai

router = APIRouter()

@router.post("/match-resume")
async def match_resume(file: UploadFile = File(...)):
    file_bytes = await file.read()

    # ✅ Safe extension handling
    ext = "pdf"  # default
    if file.filename and "." in file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()

    # ✅ Validate supported file types
    valid_exts = {"pdf", "docx", "txt"}
    if ext not in valid_exts:
        return {"error": f"Unsupported file type: {ext}"}

    text = extract_text_from_bytes(file_bytes, file_type=ext)
    processed_text = pre_process_text(text)
    recommendations = match_resume_with_jobs_ai(processed_text)
    
    return {"recommendations": recommendations}
