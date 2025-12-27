import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"🔑 Key found: {api_key[:5]}...*****")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    response = llm.invoke("Hello, MLOps")
    
    print("\n SUCCESS! Google Gemini replied:")
    print(response.content)
except Exception as e:
    print("\n ERROR. Something is wrong:")
    print(e)