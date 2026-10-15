from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

from langchain_groq import ChatGroq

from langchain_classic.chains import (
    create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

# -----------------------------------
# LOAD PDF
# -----------------------------------

document_path = (
    Path(__file__).resolve().parents[1]
    / "documents"
    / "CAREER_counsellor.pdf"
)
loader = PyPDFLoader(str(document_path))

raw_documents = loader.load()

# -----------------------------------
# SPLIT DOCUMENTS
# -----------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(
    raw_documents
)

# -----------------------------------
# EMBEDDINGS
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------
# VECTOR DB
# -----------------------------------

vector_database = Chroma.from_documents(
    split_docs,
    embeddings
)

# -----------------------------------
# LLM
# -----------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)

# -----------------------------------
# PROMPT
# -----------------------------------

system_instruction = """
You are a helpful assistant.

Use only the provided context.

If the answer is not available
in the context, say:

'I don't know based on the
provided document.'

Context:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_instruction),
        ("human", "{input}")
    ]
)

# -----------------------------------
# RETRIEVER
# -----------------------------------

retriever = vector_database.as_retriever(
    search_kwargs={"k": 3}
)

# -----------------------------------
# DOCUMENT CHAIN
# -----------------------------------

document_chain = create_stuff_documents_chain(
    llm,
    prompt_template
)

# -----------------------------------
# RAG CHAIN
# -----------------------------------

rag_pipeline = create_retrieval_chain(
    retriever,
    document_chain
)

# -----------------------------------
# FUNCTION
# -----------------------------------

def ask_rag(question):

    response = rag_pipeline.invoke(
        {
            "input": question
        }
    )

    return response["answer"]














