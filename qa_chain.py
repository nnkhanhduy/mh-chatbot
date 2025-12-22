from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template="""You are a compassionate mental health chatbot.
Use the context below to answer thoughtfully.

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