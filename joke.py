from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)
prompt = ChatPromptTemplate.from_template("tell me a joke about {foo} in english and portuguese-br")
chain = prompt | model

resposta = chain.invoke({"foo": "bears"})
print(resposta.content)
