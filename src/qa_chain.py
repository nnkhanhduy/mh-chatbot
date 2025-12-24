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

Context:
{context}

User: {question}
Chatbot:""",
        input_variables=["context", "question"]
    )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain