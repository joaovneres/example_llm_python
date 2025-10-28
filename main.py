import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ==========================================================
# 1. Carrega variáveis de ambiente
# ==========================================================
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

if not OPENAI_KEY or not GOOGLE_KEY:
    raise RuntimeError("As chaves OPENAI_API_KEY e GOOGLE_API_KEY devem estar definidas no .env")

# ==========================================================
# 2. Exemplo: LLM com OpenAI
# ==========================================================
print("🔹 Exemplo com OpenAI (GPT-4o-mini)")

prompt = ChatPromptTemplate.from_template("Resuma em uma frase o seguinte texto: {texto}")
chain_openai = prompt | ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY)
resposta_openai = chain_openai.invoke({"texto": "A IA está transformando a forma como trabalhamos e aprendemos."})
print("Resposta OpenAI:", resposta_openai.content)

# ==========================================================
# 3. Exemplo: LLM com Gemini
# ==========================================================
print("\n🔹 Exemplo com Gemini (1.5-flash)")

chain_gemini = prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_KEY)
resposta_gemini = chain_gemini.invoke({"texto": "A IA está transformando a forma como trabalhamos e aprendemos."})
print("Resposta Gemini:", resposta_gemini.content)

# ==========================================================
# 4. Exemplo: Embeddings e similaridade
# ==========================================================
print("\n🔹 Exemplo com Embeddings (OpenAI)")

text_a = "Aprendizagem de máquina é uma área da inteligência artificial."
text_b = "Machine learning é um campo que pertence à inteligência artificial."

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_KEY)

vec_a = np.array(embeddings.embed_query(text_a))
vec_b = np.array(embeddings.embed_query(text_b))

similarity = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
percent = round(similarity * 100, 2)

print(f"Similaridade entre os textos: {percent}%")
