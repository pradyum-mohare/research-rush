import os

from dotenv import load_dotenv
from google import genai

from vector_store import query_similar_chunks

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
GENERATION_MODEL = "gemini-2.5-flash"


def build_prompt(question, retrieved_chunks, chat_history):
    """
    Build a grounded prompt that includes:
    - Retrieved chunks as context (from Pinecone)
    - Full conversation history so Gemini understands follow-up questions
    - The new question

    The history is formatted as alternating Human/Assistant turns so
    Gemini can resolve references like "that", "it", "the second point"
    back to what was actually said earlier.
    """
    context = "\n\n---\n\n".join(
        chunk["metadata"]["text"] for chunk in retrieved_chunks
    )

    # Format chat history as a readable conversation transcript
    history_text = ""
    if chat_history:
        history_text = "\n\nConversation so far:\n"
        for turn in chat_history:
            history_text += f"Human: {turn['question']}\n"
            history_text += f"Assistant: {turn['answer']}\n\n"

    prompt = f"""You are a research assistant answering questions strictly based on the provided context from the user's uploaded documents.

Context from documents:
{context}
{history_text}
New Question: {question}

Instructions:
- Answer using ONLY the information in the context above.
- Use the conversation history to understand follow-up questions and references like "that", "it", or "the first point".
- If the context does not contain enough information to answer, say so clearly instead of guessing.
- Be concise and well-structured.
- Do not mention "the context" explicitly in your answer — just answer naturally.

Answer:"""

    return prompt


class RAGChain:
    """
    Stateful RAG chain that maintains conversation history across turns.

    Each instance holds its own chat_history list, so you can have
    multiple independent conversations (e.g. one per Streamlit session)
    without them interfering with each other.
    """

    def __init__(self, index):
        self.index = index
        self.chat_history = []  # list of {"question": ..., "answer": ...}

    def ask(self, question, top_k=5):
        """
        Ask a question with full conversational context.
        Retrieves relevant chunks, builds a memory-aware prompt,
        calls Gemini, stores the turn in history, and returns the answer.
        """
        retrieved_chunks = query_similar_chunks(self.index, question, top_k=top_k)

        if not retrieved_chunks:
            return "I couldn't find any relevant information in the uploaded documents."

        prompt = build_prompt(question, retrieved_chunks, self.chat_history)

        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

        answer = response.text

        # Store this turn so future questions can reference it
        self.chat_history.append({
            "question": question,
            "answer": answer
        })

        return answer

    def clear_history(self):
        """Reset the conversation — useful for starting a fresh topic."""
        self.chat_history = []
        print("[INFO] Conversation history cleared.")


# ---------------------------------------------------------------------------
# Keep this for backward compatibility with test_phase3.py
# ---------------------------------------------------------------------------
def ask_question(index, question, top_k=5):
    """Stateless single-turn question (no memory). Used by test_phase3.py."""
    from vector_store import query_similar_chunks
    retrieved_chunks = query_similar_chunks(index, question, top_k=top_k)

    if not retrieved_chunks:
        return "I couldn't find any relevant information in the uploaded documents."

    prompt = build_prompt(question, retrieved_chunks, chat_history=[])

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )
    return response.text