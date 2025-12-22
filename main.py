import os
import gradio as gr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from model import initialize_llm
from vectordb import create_vector_db
from qa_chain import setup_qa_chain

def load_css(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

print("Initializing Chatbot...")

llm = initialize_llm()

db_path = "./chroma_db"

if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    print("Loaded existing ChromaDB")

qa_chain = setup_qa_chain(vector_db, llm)

print("Chatbot ready!")

def chatbot_response(message, history):
    response = qa_chain.invoke(message)
    return response.content

custom_theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="sky",
    font=[gr.themes.GoogleFont("Inter")]
)

custom_css = load_css("styles/chatbot.css")

app = gr.ChatInterface(
    fn=chatbot_response,
    title="Mental Health Chatbot",
    description="A calm space to talk, reflect, and heal",
    theme=custom_theme,
    css=custom_css
)

app.launch()
