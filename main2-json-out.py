from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
# from langchain.globals import set_debug
import os

# set_debug(True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class Destino(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    motivo:str = Field("Motivo pelo qual é interessante visitar essa cidade")

parser = JsonOutputParser(pydantic_object=Destino)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parser.get_format_instructions()}
)

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

cadeia = prompt_cidade | modelo | parser

resposta = cadeia.invoke({ "interesse" : "praias" })
print(resposta)