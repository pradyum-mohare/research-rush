from vector_store import create_index_if_not_exists
from rag_chain import RAGChain

print("=== Connecting to Pinecone index ===")
index = create_index_if_not_exists()

# Create a stateful RAG chain with memory
chain = RAGChain(index)

print("\n=== Testing Conversational Memory ===\n")

# This sequence is designed to test memory specifically:
# Q2 says "it" — needs Q1's answer to resolve what "it" refers to
# Q3 says "its advantages" — needs Q1+Q2 to understand the topic
# Q4 says "go back to" — tests ability to switch topics with history intact

conversation = [
    "What is a data warehouse?",
    "What are its characteristics?",                    # "its" refers to data warehouse from Q1
    "What are the advantages of using it?",             # "it" still refers to data warehouse
    "Now tell me about decision support systems.",      # topic switch
    "What phases are involved in developing one?",     # "one" refers to DSS from Q4
]

for i, question in enumerate(conversation):
    print(f"Q{i+1}: {question}")
    answer = chain.ask(question)
    print(f"A{i+1}: {answer}\n")
    print("-" * 80 + "\n")

print(f"Total turns in memory: {len(chain.chat_history)}")