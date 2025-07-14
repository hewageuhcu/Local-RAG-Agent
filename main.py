from config.settings import question
from agents.graph_flow import workflow
workflow.compile().invoke({"question": question})
