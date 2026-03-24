# =========================
# ATS RAG SYSTEM (ADVANCED)
# Structured Parsing + Skills + Job Listings
# =========================

import os
import json
import logging
import re
from typing import List
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from supabase import create_client, Client
from mistralai.client import MistralClient

import pdfplumber
import docx

# =========================
# CONFIG
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, MISTRAL_API_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

# =========================
# MISTRAL HELPERS
# =========================
def llm_complete(prompt: str) -> str:
    try:
        response = mistral_client.chat(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Mistral chat API error: {e}")
        raise

def get_embedding(text: str) -> List[float]:
    try:
        response = mistral_client.embeddings(
            model="mistral-embed",
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Mistral embedding API error: {e}")
        raise

# =========================
# JSON EXTRACTION
# =========================
def extract_json_from_text(text: str) -> str:
    # Extract from markdown code block
    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if code_match:
        candidate = code_match.group(1).strip()
        if candidate.startswith('{') and candidate.endswith('}'):
            return candidate
    # Fallback: find first { to last }
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    # For arrays (normalization returns array)
    array_match = re.search(r'(\[.*\])', text, re.DOTALL)
    if array_match:
        return array_match.group(1)
    return text

# =========================
# FILE PARSING
# =========================
def parse_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def parse_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(path: str) -> str:
    if path.endswith(".pdf"):
        return parse_pdf(path)
    elif path.endswith(".docx"):
        return parse_docx(path)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or DOCX.")

# =========================
# STRUCTURED PARSER
# =========================
def parse_resume_structured(text: str) -> dict:
    prompt = f"""
Extract structured data from this resume. Return ONLY valid JSON, with no additional text, explanations, or markdown formatting. The JSON must exactly match the schema below.

{{
  "name": string,
  "skills": [string],
  "experience": [
    {{
      "role": string,
      "years": number
    }}
  ],
  "education": [string]
}}

Resume:
{text}
"""
    response_text = llm_complete(prompt)
    logger.info(f"Raw LLM response: {response_text}")

    json_str = extract_json_from_text(response_text)
    try:
        parsed = json.loads(json_str)
        if not all(k in parsed for k in ("name", "skills", "experience", "education")):
            raise ValueError("Missing expected keys in parsed JSON")
        return parsed
    except Exception as e:
        logger.error(f"JSON parsing error: {e}")
        return {"error": "Parsing failed", "raw": response_text}

# =========================
# SKILL NORMALIZATION
# =========================
def normalize_skills(skills: List[str]) -> List[str]:
    if not skills:
        return []

    prompt = f"""
Normalize the following list of skills into standard industry terms.
Return a **JSON array of strings only** (e.g., ["JavaScript", "Python", "Microsoft Azure"]).
Do not include any additional text, explanations, or objects.

Examples:
Input: ["js", "py", "azure"]
Output: ["JavaScript", "Python", "Microsoft Azure"]

Now process this list:
{skills}
"""

    response_text = llm_complete(prompt)
    logger.info(f"Normalization raw response: {response_text}")

    json_str = extract_json_from_text(response_text)
    try:
        normalized = json.loads(json_str)
        if isinstance(normalized, list):
            if all(isinstance(item, str) for item in normalized):
                return normalized
            else:
                logger.warning("Normalization returned non-string elements, using original skills")
                return skills
        else:
            logger.warning(f"Normalization did not return a list: {normalized}")
            return skills
    except Exception as e:
        logger.warning(f"Failed to parse normalization response: {e}. Using original skills.")
        return skills

# =========================
# STORE DATA
# =========================
def store_resume(raw_text: str) -> dict:
    structured = parse_resume_structured(raw_text)
    if "error" in structured:
        raise ValueError(structured["raw"])

    skills = normalize_skills(structured.get("skills", []))
    embedding = get_embedding(raw_text)

    data = {
        "id": str(uuid4()),
        "content": raw_text,
        "type": "resume",
        "embedding": embedding,
        "structured": structured,
        "skills": skills
    }

    try:
        supabase.table("documents").insert(data).execute()
    except Exception as e:
        logger.error(f"Supabase insert error: {e}")
        if "Could not find the 'skills' column" in str(e):
            raise Exception("Supabase error: The 'documents' table is missing a 'skills' column. Please add it (type jsonb or text[]).")
        raise

    return structured

# =========================
# SEARCH & MATCHING
# =========================
def search(query: str):
    embedding = get_embedding(query)
    logger.info(f"Query embedding (first 5 values): {embedding[:5]}")
    result = supabase.rpc(
        "match_documents",
        {"query_embedding": embedding, "match_count": 5}
    ).execute()
    logger.info(f"Supabase RPC response: {result}")
    logger.info(f"Search returned {len(result.data)} documents")
    return result.data

def match_candidates(query: str):
    docs = search(query)

    if not docs:
        logger.info("No candidates found, returning empty list.")
        return []

    prompt = f"""
{query}

Candidates:
{json.dumps(docs)}

Evaluate and rank candidates.

Return JSON:
[
  {{
    "candidate_id": string,
    "match_score": number,
    "strengths": [],
    "gaps": [],
    "recommendation": ""
  }}
]
"""

    response_text = llm_complete(prompt)
    json_str = extract_json_from_text(response_text)
    try:
        return json.loads(json_str)
    except:
        logger.error(f"Invalid JSON in match_candidates: {response_text}")
        return []

# =========================
# FASTAPI SETUP
# =========================
app = FastAPI(title="ATS RAG with Job Listings")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PYDANTIC MODELS
# =========================
class QueryRequest(BaseModel):
    query: str

class JobListingCreate(BaseModel):
    title: str
    description: str

class JobListingUpdateTitle(BaseModel):
    title: str

class JobListingResponse(BaseModel):
    id: str
    title: str
    description: str
    created_at: str

# =========================
# EXISTING ENDPOINTS
# =========================
@app.post("/upload/resume")
async def upload_resume(file: UploadFile = File(...)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    temp_path = f"./temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        text = extract_text(temp_path)
        logger.info(f"Extracted text length: {len(text)}")

        structured = store_resume(text)

        return {
            "message": "Resume processed successfully",
            "structured": structured
        }
    except ValueError as e:
        logger.error(f"Resume parsing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Resume parsing failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ats/match")
async def match(request: QueryRequest):
    try:
        return match_candidates(request.query)
    except Exception as e:
        logger.error(f"Matching error: {e}")
        raise HTTPException(status_code=500, detail="Matching failed")

@app.get("/health")
async def health():
    return {"status": "ok"}

# =========================
# JOB LISTINGS ENDPOINTS
# =========================
@app.post("/job-listings", response_model=JobListingResponse)
async def create_job_listing(listing: JobListingCreate):
    """Create a new job listing."""
    new_id = str(uuid4())
    data = {
        "id": new_id,
        "title": listing.title,
        "description": listing.description,
        "created_at": "now()"  # will be set by database default
    }
    try:
        result = supabase.table("job_listings").insert(data).execute()
        inserted = result.data[0]
        return inserted
    except Exception as e:
        logger.error(f"Error creating job listing: {e}")
        raise HTTPException(status_code=500, detail="Failed to create job listing")

@app.get("/job-listings", response_model=List[JobListingResponse])
async def list_job_listings():
    """List all job listings."""
    try:
        result = supabase.table("job_listings").select("*").order("created_at", desc=True).execute()
        return result.data
    except Exception as e:
        logger.error(f"Error listing job listings: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job listings")

@app.get("/job-listings/{listing_id}", response_model=JobListingResponse)
async def get_job_listing(listing_id: str):
    """Get a single job listing by ID."""
    try:
        result = supabase.table("job_listings").select("*").eq("id", listing_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Job listing not found")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching job listing: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job listing")

@app.delete("/job-listings/{listing_id}")
async def delete_job_listing(listing_id: str):
    """Delete a job listing by ID."""
    try:
        result = supabase.table("job_listings").delete().eq("id", listing_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Job listing not found")
        return {"message": "Job listing deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job listing: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete job listing")

@app.patch("/job-listings/{listing_id}/rename")
async def rename_job_listing(listing_id: str, update: JobListingUpdateTitle):
    """Rename a job listing (update title only)."""
    try:
        result = supabase.table("job_listings").update({"title": update.title}).eq("id", listing_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Job listing not found")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming job listing: {e}")
        raise HTTPException(status_code=500, detail="Failed to rename job listing")

@app.post("/job-listings/{listing_id}/copy")
async def copy_job_listing(listing_id: str):
    """Create a copy of an existing job listing with '(copy)' appended to title."""
    try:
        # Fetch original
        orig = supabase.table("job_listings").select("*").eq("id", listing_id).execute()
        if not orig.data:
            raise HTTPException(status_code=404, detail="Job listing not found")
        original = orig.data[0]

        # Prepare new listing
        new_id = str(uuid4())
        new_title = f"{original['title']} (copy)"
        new_data = {
            "id": new_id,
            "title": new_title,
            "description": original["description"]
        }

        result = supabase.table("job_listings").insert(new_data).execute()
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error copying job listing: {e}")
        raise HTTPException(status_code=500, detail="Failed to copy job listing")