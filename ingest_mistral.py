import os
import json
import time
import glob
from sentence_transformers import SentenceTransformer
from supabase import create_client
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
RAW_DIR = "raw"

def chunk_text(text, source):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+CHUNK_SIZE])
        if len(chunk.strip()) > 30:
            chunks.append({"content": chunk.strip(), "source": source})
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def read_txt(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_chatgpt_json(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    text = ""
    for conversation in data:
        for node in conversation.get("mapping", {}).values():
            msg = node.get("message")
            if msg and msg.get("content") and msg["content"].get("parts"):
                role = msg.get("author", {}).get("role", "")
                parts = msg["content"]["parts"]
                for part in parts:
                    if isinstance(part, str) and part.strip():
                        text += f"{role.upper()}: {part.strip()}\n\n"
    return text

def collect_chunks():
    all_chunks = []
    for filepath in glob.glob(f"{RAW_DIR}/**/*.txt", recursive=True):
        print(f"Reading: {filepath}")
        all_chunks.extend(chunk_text(read_txt(filepath), filepath))
    for filepath in glob.glob(f"{RAW_DIR}/**/*.csv", recursive=True):
        print(f"Reading: {filepath}")
        all_chunks.extend(chunk_text(read_txt(filepath), filepath))
    for filepath in glob.glob(f"{RAW_DIR}/**/*.pdf", recursive=True):
        print(f"Reading PDF: {filepath}")
        all_chunks.extend(chunk_text(read_pdf(filepath), filepath))
    for filepath in glob.glob(f"{RAW_DIR}/**/*.json", recursive=True):
        print(f"Reading JSON: {filepath}")
        all_chunks.extend(chunk_text(read_chatgpt_json(filepath), filepath))
    for filepath in glob.glob(f"{RAW_DIR}/**/*.html", recursive=True):
        print(f"Reading HTML: {filepath}")
        all_chunks.extend(chunk_text(read_txt(filepath), filepath))
    return all_chunks

def embed_and_upload(chunks):
    print(f"\nTotal chunks: {len(chunks)}")
    batch_size = 32

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["content"] for c in batch]
        print(f"Batch {i//batch_size + 1} of {-(-len(chunks)//batch_size)}...")

        embeddings = model.encode(texts, show_progress_bar=False)

        rows = []
        for j, embedding in enumerate(embeddings):
            rows.append({
                "content": batch[j]["content"],
                "source": batch[j]["source"],
                "embedding": embedding.tolist()
            })

        supabase.table("memories").insert(rows).execute()
        print(f"✓ Uploaded {len(rows)} chunks")

    print("\nDone! All data in Supabase.")

if __name__ == "__main__":
    print("=== Samarth Brain Ingestion ===\n")
    chunks = collect_chunks()
    if not chunks:
        print("No files found in raw/ folder.")
    else:
        embed_and_upload(chunks)