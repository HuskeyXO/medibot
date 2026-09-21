import os
import streamlit as st

from dotenv import load_dotenv

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID = "openai/gpt-oss-120b:fastest"

DB_FAISS_PATH = "vectorstore/db_faiss"


# Load Vector Store
@st.cache_resource
def get_vectorstore():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# Load LLM
@st.cache_resource
def load_llm():

    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )

    return ChatHuggingFace(llm=llm)


# Format Documents
def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


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


prompt_template = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def main():

    st.title("MediBot")

    # Message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(
            message["content"]
        )

    # User input
    user_query = st.chat_input("Pass your prompt here")

    if user_query:

        st.chat_message("user").markdown(user_query)

        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })

        try:

            # Load vector store
            vectorstore = get_vectorstore()

            # Create retriever
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

            # Load LLM
            llm = load_llm()

            # RAG chain
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": lambda x: x
                }
                | prompt_template
                | llm
                | StrOutputParser()
            )

            # Generate answer
            result = rag_chain.invoke(user_query)

            # Retrieve source documents
            source_documents = retriever.invoke(user_query)

            # Display answer
            st.chat_message("assistant").markdown(result)

            # Display sources
            with st.expander("Source Documents"):

                for i, doc in enumerate(source_documents, 1):

                    st.markdown(f"### Source {i}")

                    st.write(
                        "Metadata:",
                        doc.metadata
                    )

                    st.write(
                        doc.page_content
                    )

            # Store answer
            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })

        except Exception as e:

            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()