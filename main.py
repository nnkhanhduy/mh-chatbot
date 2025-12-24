import os
import gradio as gr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.model import initialize_llm
from src.vectordb import create_vector_db
from src.qa_chain import setup_qa_chain
from src.voice import text_to_speech, speech_to_text

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

def voice_chat(audio_path):
    print("DEBUG audio_path:", audio_path)

    if audio_path is None:
        return "No audio", "No audio", None

    user_text = speech_to_text(audio_path)
    response = qa_chain.invoke(user_text)
    answer_text = response.content
    answer_audio = text_to_speech(answer_text)

    return user_text, answer_text, answer_audio

def chatbot_response(message, history):
    response = qa_chain.invoke(message)
    return response.content

custom_theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="sky",
    font=[gr.themes.GoogleFont("Inter")]
)

custom_css = load_css("styles/chatbot.css")

with gr.Blocks(css=custom_css) as app:
    gr.Markdown("## 🧠 Mental Health Chatbot")
    gr.Markdown("🌿 A calm space to talk, reflect, and heal")

    with gr.Tab("💬 Text Chat"):
        chatbot = gr.Chatbot(height=400)

        msg = gr.Textbox(
            placeholder="Type your thoughts here...",
            show_label=False
        )

        def respond(message, chat_history):
            response = qa_chain.invoke(message)
            chat_history.append((message, response.content))
            return "", chat_history

        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )

    with gr.Tab("🎧 Voice Chat"):
        mic = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak here",
        )

        user_text = gr.Textbox(label="You said")
        bot_text = gr.Textbox(label="Bot response")
        bot_audio = gr.Audio(label="Bot voice")

        mic.change(
            fn=voice_chat,
            inputs=mic,
            outputs=[user_text, bot_text, bot_audio]
        )

app.launch()


