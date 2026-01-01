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

def format_history(history):
    if not history:
        return "No previous conversation."
    formatted_str = ""
    for user_msg, bot_msg in history:
        formatted_str += f"User: {user_msg}\nChatbot: {bot_msg}\n"
    return formatted_str

def voice_chat(audio_path, history):
    if audio_path is None:
        return None, history, None

    user_text = speech_to_text(audio_path)
    
    # Format history for the chain
    history_str = format_history(history)
    
    response = qa_chain.invoke({
        "question": user_text,
        "chat_history": history_str
    })
    
    answer_text = response.content
    answer_audio = text_to_speech(answer_text)
    
    history.append((user_text, answer_text))

    # Return bot_audio first to trigger autoplay, then the updated history for the chatbot
    return answer_audio, history, user_text

def chatbot_response(message, history):
    history_str = format_history(history)
    response = qa_chain.invoke({
        "question": message,
        "chat_history": history_str
    })
    return response.content

custom_theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
)

custom_css = load_css("styles/chatbot.css")

with gr.Blocks(theme=custom_theme, css=custom_css, title="Mental Health AI") as app:
    # Shared history state for both tabs to stay in sync
    shared_history = gr.State([])

    with gr.Column(elem_id="header", variant="panel"):
        gr.Markdown("# 🌿 Mental Health Chatbot")
        gr.Markdown("A calm space to talk, reflect, and heal. *Your privacy is our priority.*")

    with gr.Tabs():
        with gr.Tab("💬 Text Chat"):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                bubble_full_width=False,
                elem_classes=["chatbot-container"]
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your thoughts here...",
                    show_label=False,
                    scale=9
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            def respond(message, chat_history):
                if not message.strip():
                    return "", chat_history
                
                history_str = format_history(chat_history)
                response = qa_chain.invoke({
                    "question": message,
                    "chat_history": history_str
                })
                
                chat_history.append((message, response.content))
                return "", chat_history

            msg.submit(
                respond,
                inputs=[msg, shared_history],
                outputs=[msg, chatbot]
            )
            submit_btn.click(
                respond,
                inputs=[msg, shared_history],
                outputs=[msg, chatbot]
            )

        with gr.Tab("🎙️ Voice Chat"):
            with gr.Row(elem_classes=["voice-row"]):
                with gr.Column(scale=1):
                    gr.Markdown("### 🎤 Voice Input")
                    mic = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Click to start talking",
                        elem_id="voice-mic"
                    )
                    
                    stt_output = gr.Textbox(label="You said", interactive=False)
                    # Invisible component to trigger autoplay
                    bot_audio_playback = gr.Audio(
                        label="Bot response", 
                        autoplay=True, 
                        # visible=False
                    )

                with gr.Column(scale=2):
                    voice_chatbot = gr.Chatbot(
                        height=500,
                        show_label=False,
                        bubble_full_width=False,
                        elem_classes=["chatbot-container"]
                    )

            mic.stop_recording(
                fn=voice_chat,
                inputs=[mic, shared_history],
                outputs=[bot_audio_playback, voice_chatbot, stt_output]
            )
            
            # Sync the chatbot on both tabs when history changes
            shared_history.change(
                fn=lambda x: x,
                inputs=[shared_history],
                outputs=[chatbot]
            )
            shared_history.change(
                fn=lambda x: x,
                inputs=[shared_history],
                outputs=[voice_chatbot]
            )

    gr.Markdown("---")
    gr.Markdown("⚠️ *Disclaimer: This chatbot is for emotional support only. If you are in crisis, please contact local emergency services.*")

app.launch()
