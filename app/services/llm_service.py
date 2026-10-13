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

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


def ask_llm(question):

    return chain.invoke(
        {
            "question": question
        }
    )




















