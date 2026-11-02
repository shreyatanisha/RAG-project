import hashlib
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def _get_rag_pipeline():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
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
        / "tvmto bbsr2.pdf"
    )
    document_hash = hashlib.sha256(document_path.read_bytes()).hexdigest()[:12]
    raw_documents = PyPDFLoader(str(document_path)).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(raw_documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        ids=[f"chunk-{index}" for index in range(len(split_docs))],
        collection_name=f"tvmto_bbsr2_{document_hash}",
        persist_directory=str(Path(__file__).resolve().parents[1] / "vectordb"),
    )

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a document question-answering assistant.

                    Rules:

                    1. Answer only from the provided context.

                    2. Do not use outside knowledge.

                    3. If the context does not contain
                       the answer, respond exactly:

                    'I don't know based on the provided document.'

                    4. Keep answers concise and factual.

                    Context:
                    {context}
                """,
            ),
            
                 ("human", "{input}"),
        ]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})    
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, document_chain)

# -----------------------------------
# FUNCTION
# -----------------------------------

def ask_rag(question):
    response = _get_rag_pipeline().invoke({"input": question})
    return {
    "answer": response["answer"],
    "sources": response["context"]
}










