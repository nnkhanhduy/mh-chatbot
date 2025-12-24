from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key =os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
    return llm
