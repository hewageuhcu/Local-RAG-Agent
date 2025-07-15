from langgraph.graph import END
from main import workflow, route_question, decide_to_generate, grade_generation_v_documents_and_question

# Build the complete graph workflow
def build_workflow():
    """
    Assembles the complete workflow with all edges and conditions
    """
    # Set conditional entry point
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )
    
    # Add edges
    workflow.add_edge("retrieve", "grade_documents")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    
    workflow.add_edge("websearch", "generate")
    
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )
    
    return workflow.compile()

# Usage example
if __name__ == "__main__":
    app = build_workflow()
    
    # Test the workflow
    inputs = {
        "question": "What is agent memory?",
        "generation": "",
        "web_search": "",
        "documents": [],
    }
    # for output in app.stream(inputs):
    #     for key, value in output.items():
    #         print(f"Finished running: {key}")
    #         print(f"Output: {value}")
    #         print("---")