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

class Restaurantes(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    restaurantes:str = Field("Restaurantes recomendados na cidade")

parser_destino = JsonOutputParser(pydantic_object=Destino)
parser_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parser_destino.get_format_instructions()}
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira restaurantes populares entre locais em {cidade}.
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parser_restaurantes.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template="Sugira atividades e locais culturais em {cidade}",
    partial_variables={"formato_de_saida": parser_restaurantes.get_format_instructions()}
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

cadeia_1 = prompt_cidade | llm | parser_destino
cadeia_2 = prompt_restaurantes | llm | parser_restaurantes
cadeia_3 = prompt_cultural | llm | StrOutputParser()

cadeia = cadeia_1 | cadeia_2 | cadeia_3

resposta = cadeia.invoke({ "interesse" : "praias" })
print(resposta)