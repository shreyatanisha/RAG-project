# from dotenv import load_dotenv
# from services.llm_service import ask_llm

# load_dotenv()

# print("================================")
# print("       MY AI CLIENT CHATBOT")
# print("================================")
# print("Type 'exit' to stop the chatbot.\n")

# while True:

#     user_question = input("Client: ")

#     if user_question.lower() == "exit":
#         print("Bot: Thank you! Goodbye.")
#         break

#     answer = ask_llm(user_question)

#     print("Bot:", answer)
#     print()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.chat_models import ChatRequest, ChatResponse
from services.llm_service import ask_llm
from services.rag_service import ask_rag

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "Welcome to the AI Chatbot API"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:

        answer = ask_llm(
            request.session_id,
            request.question,
        )

        return ChatResponse(
            answer=answer
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/chat/rag")
def chat_rag(request: ChatRequest):

    answer = ask_rag(
        request.question
    )

    return {
        "answer": answer
    }



































