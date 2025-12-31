from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template="""You are a mental health support chatbot.
Your role is to provide emotional support, validation, and gentle guidance.
You are not a therapist or doctor.

Rules:
- Respond in a calm, warm, and empathetic tone.
- Keep answers short: maximum 2–4 sentences.
- Use simple, conversational language.
- Do not lecture or over-explain.
- Do not give medical or diagnostic advice.
- Avoid lists unless absolutely necessary.
- If the user is distressed, validate their feelings first before responding.
- Encourage gentle reflection, not solutions.

If the user asks something outside mental health support,
answer briefly and steer back to emotional well-being.

If you do not know something, say so gently.

Current Conversation:
{chat_history}

Relevant Context:
{context}

User: {question}
Chatbot:""",
        input_variables=["chat_history", "context", "question"]
    )

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
    )

    return chain
