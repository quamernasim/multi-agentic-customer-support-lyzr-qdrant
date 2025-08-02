# run_support.py

import json
import re
import uuid
from agents import (
    gemini_model, RouterAgent, SentimentAgent, KBAgent,
    ResponseAgent, EscalationAgent, retrieve_kb,
    get_history, save_message
)
from lyzr_automata import Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from lyzr_automata.tasks.task_literals import InputType, OutputType

def run_session():
    session_id = str(uuid.uuid4())
    print(f"Session {session_id} started.")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        save_message(session_id, "user", user_input)
        history = get_history(session_id)

        # 1. Route Issue
        route_task = Task(
            name="RouteIssue",
            agent=RouterAgent,
            model=gemini_model,
            instructions=user_input,
            input_type=InputType.TEXT,
            output_type=OutputType.TEXT
        )
        # 2. Analyze Sentiment
        senti_task = Task(
            name="AnalyzeSentiment",
            agent=SentimentAgent,
            model=gemini_model,
            instructions=user_input,
            input_type=InputType.TEXT,
            output_type=OutputType.TEXT
        )
        # 3. Retrieve KB
        kb_context = retrieve_kb(user_input)
        kb_task = Task(
            name="RetrieveKB",
            agent=KBAgent,
            model=gemini_model,
            instructions=kb_context,
            input_type=InputType.TEXT,
            output_type=OutputType.TEXT
        )
        # 4. Escalation Check
        escalation_task = Task(
            name="CheckEscalation",
            agent=EscalationAgent,
            model=gemini_model,
            instructions=user_input,
            input_type=InputType.TEXT,
            output_type=OutputType.TEXT,
            input_tasks = [
                route_task, 
                senti_task
            ]
        )
        # 5. Generate Response
        resp_instructions = (
            f"You have been provided with the Issue, "
            f"sentiment of the user, "
            f"Knowledge Base context, "
            f"History: {history}\n"
            f"and Escalation status\n\n"
            "Now craft the final support response based on all of the above."
        )
        response_task = Task(
            name="GenerateResponse",
            agent=ResponseAgent, model=gemini_model,
            instructions=resp_instructions,
            input_type=InputType.TEXT,
            output_type=OutputType.TEXT,
            input_tasks = [
                route_task, 
                senti_task, 
                kb_task, 
                escalation_task
            ]
        )

        pipeline = LinearSyncPipeline(
            name="MultiAgentSupport",
            completion_message="Done",
            tasks=[
                route_task,
                senti_task,
                kb_task,
                escalation_task,
                response_task
            ]
        )
        outputs = pipeline.run()

        response = outputs[-1]['task_output']
        response = re.sub(r"^```json\n|\n```$", "", response.strip())
        response = json.loads(response)
        response = response['response']['message']
        print(f"Agent: {response}")
        save_message(session_id, "assistant", response)

if __name__ == "__main__":
    run_session()
