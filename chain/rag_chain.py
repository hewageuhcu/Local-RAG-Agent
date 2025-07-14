from langchain_core.output_parsers import StrOutputParser
from prompts.answer_prompt import answer_prompt
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3", temperature=0)
rag_chain = answer_prompt | llm | StrOutputParser()

def generate_answer(question, docs):
    return rag_chain.invoke({"context": docs, "question": question})
