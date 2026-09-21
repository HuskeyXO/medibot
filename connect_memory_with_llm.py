import os

from dotenv import load_dotenv

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS


# Load environment variables
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID = "openai/gpt-oss-120b:fastest"


# Load LLM

def load_llm(huggingface_repo_id):

    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )

    chat_model = ChatHuggingFace(llm=llm)

    return chat_model


# Custom Prompt

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.

If you don't know the answer, just say that you don't know.
Don't try to make up an answer.
Don't provide anything outside of the given context.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""


prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# Load FAISS database

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# Retriever

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# Load LLM

llm = load_llm(HUGGINGFACE_REPO_ID)


# Format documents

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# RAG Chain

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)


# User Query

user_query = input("Write your query: ")


# Retrieve source documents separately

source_documents = retriever.invoke(user_query)


# Generate answer

response = rag_chain.invoke(user_query)


# Print result

print("\nRESULT:\n")
print(response)


print("\nSOURCE DOCUMENTS:\n")

for i, doc in enumerate(source_documents, start=1):

    print(f"\n----- SOURCE {i} -----")

    print("Metadata:")
    print(doc.metadata)

    print("\nContent:")
    print(doc.page_content)