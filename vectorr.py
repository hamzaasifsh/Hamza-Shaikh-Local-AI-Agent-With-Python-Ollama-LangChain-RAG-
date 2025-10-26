from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load dataset
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define vector store path
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Create Chroma vector store (will be persisted locally)
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings,
)

# Add documents if this is the first run
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        doc = Document(
            page_content=f"{row['Title']} {row['Review']}",
            metadata={"rating": row["Rating"], "date": row["Date"]},
        )
        documents.append(doc)
        ids.append(str(i))

    # Add to Chroma database
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"✅ Added {len(documents)} documents to the database.")

# Create retriever for later use
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
