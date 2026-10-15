from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def _get_rag_pipeline():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.retrievers import BM25Retriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_groq import ChatGroq
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import (
        create_stuff_documents_chain,
    )
    from langchain_core.prompts import ChatPromptTemplate

    document_path = (
        Path(__file__).resolve().parents[1]
        / "documents"
        / "CAREER_counsellor.pdf"
    )
    raw_documents = PyPDFLoader(str(document_path)).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(raw_documents)

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant.

Use only the provided context.

If the answer is not available in the context, say:
'I don't know based on the provided document.'

Context:
{context}""",
            ),
            ("human", "{input}"),
        ]
    )

    retriever = BM25Retriever.from_documents(split_docs, k=3)
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, document_chain)

# -----------------------------------
# FUNCTION
# -----------------------------------

def ask_rag(question):
    response = _get_rag_pipeline().invoke({"input": question})
    return response["answer"]











