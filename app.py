import faiss, openai, os
from openai import OpenAI
import tiktoken

client = OpenAI(api_key="YOUR_API_KEY")

# --- Load Documents ---

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "docs.txt")

with open(file_path, "r", encoding="utf-8") as f:
    docs = f.readlines()
# --- Create Embeddings ---
def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

embeddings = [embed(d) for d in docs]
dimension = len(embeddings[0])

# --- Create FAISS Vector DB ---
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype="float32"))

# --- Retrieve Top K ---
def retrieve(query, k=2):
    v = embed(query)
    D, I = index.search(np.array([v], dtype="float32"), k)
    return "\n".join(docs[i] for i in I[0])

# --- Generate Final Answer ---
def rag_answer(query):
    context = retrieve(query)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful RAG chatbot."},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

# --- Chat Loop ---
print("RAG Chatbot Started. Ask anything:")
while True:
    q = input("You: ")
    print("Bot:", rag_answer(q))
