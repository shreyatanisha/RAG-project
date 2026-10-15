# from groq import Groq
# import os

# client = Groq(
#     api_key=os.getenv("GROQ_API_KEY")
# )

# conversation = [
#     {
#         "role": "system",
#         "content": """
#         You are a professional customer support chatbot.

#         Follow these rules:
#         1. Be polite and professional.
#         2. Keep answers simple.
#         3. Give useful responses.
#         4. If unsure, say you don't know.
#         """
#     }
# ]


# def ask_llm(question):

#     conversation.append(
#         {
#             "role": "user",
#             "content": question
#         }
#     )

#     response = client.chat.completions.create(
#         model="openai/gpt-oss-20b",
#         temperature=0.3,
#         messages=conversation
#     )

#     answer = response.choices[0].message.content

#     conversation.append(
#         {
#             "role": "assistant",
#             "content": answer
#         }
#     )

#     return answer

from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from memory.chat_memory import get_session_history


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a professional customer support chatbot.

    Question:{question}
    """
)

parser = StrOutputParser()

chain = prompt | llm | parser


def ask_llm(session_id, question):
    history = get_session_history(session_id)

    history.append(HumanMessage(content=question))

    answer = chain.invoke({
        "question": question,
        "history": history,
    })

    history.append(AIMessage(content=answer))
    return answer


















