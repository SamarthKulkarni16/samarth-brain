import os
from mistralai import Mistral
from supabase import create_client
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

SYSTEM_PROMPT = """You are Samarth. Not an assistant. You ARE Samarth.

STRICT RULES - NEVER BREAK THESE:
1. ONLY use information from the retrieved memories below
2. If memories don't cover the topic, say "I don't have my own thoughts on this yet"
3. NEVER use these words: "certainly", "absolutely", "great question", "here's the truth", "hard truth", "straightforward", "of course", "sure"
4. NEVER sound like ChatGPT or an AI assistant
5. NEVER give bullet point listicles unless Samarth naturally does that
6. Speak exactly how Samarth writes — short, direct, sometimes incomplete, real
7. Use Samarth's actual vocabulary and sentence patterns from the memories
8. If you are not sure, say so directly — don't fabricate
9. NEVER bring up people, names, or personal stories unless the user specifically asks

Samarth's natural communication style:
- Short punchy sentences
- Asks sharp questions back
- Direct, no fluff
- Thinks out loud
- Mixes casual and deep in same message

Retrieved memories from Samarth's own words:
{memories}

REMINDER: You are Samarth speaking. Not an AI describing Samarth."""

def search_memories(query, count=5):
    result = mistral.embeddings.create(
        model="mistral-embed",
        inputs=[query]
    )
    query_embedding = result.data[0].embedding
    response = supabase.rpc("search_memories", {
        "query_embedding": query_embedding,
        "match_count": count
    }).execute()
    return response.data

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "No message"}), 400

    memories = search_memories(user_message)
    memory_text = "\n\n---\n\n".join([
        f"[Source: {m['source']}]\n{m['content']}"
        for m in memories
    ]) if memories else "No specific memories found."

    system = SYSTEM_PROMPT.format(memories=memory_text)
    messages = [{"role": "system", "content": system}]
    for msg in history[-10:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    response = mistral.chat.complete(
        model="mistral-small-latest",
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )

    reply = response.choices[0].message.content

    try:
        supabase.table("conversations").insert({
            "user_message": user_message,
            "clone_reply": reply
        }).execute()
    except:
        pass

    return jsonify({"reply": reply, "memories_used": len(memories)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/")
def index():
    return send_file("brain.html")

@app.route("/save-conversation", methods=["POST"])
def save_conversation():
    data = request.json
    existing = supabase.table("conversations").select("id").eq("id", data["id"]).execute()
    if existing.data:
        supabase.table("conversations").update({
            "title": data["title"],
            "messages": data["messages"],
            "updated_at": "now()"
        }).eq("id", data["id"]).execute()
    else:
        supabase.table("conversations").insert({
            "id": data["id"],
            "title": data["title"],
            "messages": data["messages"]
        }).execute()
    return jsonify({"ok": True})

@app.route("/get-conversations", methods=["GET"])
def get_conversations():
    result = supabase.table("conversations").select("*").order("updated_at", desc=True).limit(50).execute()
    return jsonify({"conversations": result.data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))