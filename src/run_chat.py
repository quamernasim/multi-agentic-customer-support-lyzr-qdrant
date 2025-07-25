# run_chat.py
from agents import (
    huggingface_model,
    sentiment_agent,
    manager_agent,
    search_knowledge_base,
    get_history,
    save_message,
)
from lyzr_automata import Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from lyzr_automata.tasks.task_literals import InputType, OutputType
import uuid

def run_chat_session():
    """Runs an interactive chat session with the multi-agent system."""
    session_id = str(uuid.uuid4())
    print(f"Starting new chat session: {session_id}")
    print("Welcome to our customer support! How can I help you today? (type 'exit' to end)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for chatting with us. Goodbye!")
            break
        
        # Task 1: Analyze Sentiment
        sentiment_analysis_task = Task(
            name="Sentiment Analysis Task",
            agent=sentiment_agent,
            model=huggingface_model,
            instructions=f"Analyze the sentiment of this user query: '{user_input}'",
            output_type=OutputType.TEXT,
            input_type=InputType.TEXT,
        )

        # Prepare context for the main task
        history = get_history(session_id)
        kb_context = search_knowledge_base(user_input)
        save_message(session_id, "user", user_input)

        # Task 2: Generate Support Response (receives sentiment as input)
        support_task = Task(
            name="Customer Support Task",
            agent=manager_agent,
            model=huggingface_model,
            instructions=f"""
            A user has asked the following question: '{user_input}'

            First, an expert analyzed the user's sentiment. Their analysis is:

            Here is the conversation history for context:
            {history}

            Here is some relevant information from our knowledge base:
            {kb_context}

            Based on all of this information, especially the user's sentiment, please provide a clear and helpful response. If the sentiment is Negative, be extra empathetic.
            """,
            input_type=InputType.TEXT,
            output_type=OutputType.TEXT,
        )
        
        # Run the pipeline
        pipeline = LinearSyncPipeline(
            name="Support Pipeline",
            completion_message="Pipeline completed.",
            tasks=[sentiment_analysis_task, support_task],
        )
        
        result = pipeline.run()
        
        # The final output is from the last task in the pipeline
        agent_response = result[-1]['task_output']
        print(f"Agent: {agent_response}")
        
        # Save the agent's response to history
        save_message(session_id, "assistant", agent_response)

if __name__ == "__main__":
    run_chat_session()