from langchain_core.prompts import PromptTemplate

retrieval_prompt = PromptTemplate(
    template="""<|begin_of_text|>...
    """,
    input_variables=["question", "document"],
)
