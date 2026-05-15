import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# base de uma solução com LCEL e LangGraph

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key
)

prompt_consultor = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um consultor de viagens."),
        ("human", "{query}")
    ]
)

assistente = prompt_consultor | llm | StrOutputParser()

print(assistente.invoke({"query": "Quero férias em praias no Brasil."}))