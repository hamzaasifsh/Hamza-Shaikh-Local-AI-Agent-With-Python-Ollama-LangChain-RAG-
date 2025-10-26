from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorr import retriever  # renamed from retrieve -> retriever

# Initialize model
model = OllamaLLM(model="llama3.2")

# Define template
template = """
You are an expert in answering questions about pizza restaurants.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Combine prompt and model
chain = prompt | model

# Loop for user queries
while True:
    print("\n\n__________________________")
    question = input("Ask me any question (q to quit): ").strip()
    print("\n\n")
    if question.lower() == "q":
        break

    # Retrieve top reviews
    docs = retriever.invoke(question)
    reviews = "\n".join([d.page_content for d in docs])

    # Run the model
    result = chain.invoke({"reviews": reviews, "question": question})

    # Print result
    print("🧠 AI Answer:")
    print(result)
