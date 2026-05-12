import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

prompt_sugestion = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios."),
        ("placeholder", "{history}"),
        ("human", "{query}")
    ]
)

chain = prompt_sugestion | llm | StrOutputParser()

memory = {}
session = "pratica_de_langchain"

def history_session(session : str):
    if session not in memory:
        memory[session] = InMemoryChatMessageHistory()
    return memory[session]

questions = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=history_session,
    input_messages_key="query",
    history_messages_key="history"
)

# sem
for a_question in questions:
    response = chain_with_memory.invoke(
        {
            "query" : a_question
        },
        config={"session_id":session}
    )
    print("Usuário: ", a_question),
    print("IA: ", response, "\n")