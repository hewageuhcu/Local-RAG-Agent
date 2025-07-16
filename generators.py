from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from config import local_llm

# RAG Generation Setup
rag_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

llm = ChatOllama(model=local_llm, temperature=0)

# Post-processing function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = rag_prompt | llm | StrOutputParser()

# Test the generation
if __name__ == "__main__":
    from indexing import retriever
    
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})
    print(generation)