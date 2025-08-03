import re
import json

# In-memory conversation storage (replace with a DB or Qdrant in prod)
conversation_history = {}

def save_message(session_id: str, role: str, content: str):
    """Saves a message to the conversation history for a given session."""
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append({"role": role, "content": content})
    return "Message saved."

def get_history(session_id: str):
    """Retrieves the conversation history for a given session."""
    return conversation_history.get(session_id, [])


def text_2_json(response):
    response = re.sub(r"^```json\n|\n```$", "", response.strip())
    response = json.loads(response)
    return response

def task_with_feedback_loop(
    user_input,
    task_name,
    tenant_id,
    task_response_keyword,
    session_id,
    agent_name,
    fallback_message,
    top_k=3,
    k_prefetch=10
):

    fail_count = 0
    need_decoding = True
    fail_feedback = None
    while need_decoding:
        try:
            task_response = task_name(user_input, tenant_id, top_k=top_k, k_prefetch=k_prefetch, fail_feedback=fail_feedback).execute()
            task_response = text_2_json(task_response)
            relevant_response = task_response['response'][task_response_keyword]
            # save_message(session_id, agent_name, task_response)
            need_decoding = False
        except Exception as e:
            fail_feedback = {
                "type": e,
                "instruction": "fix the above error by generating correct response",
                "previous_response": task_response
            }
            print(f'Retrying with feedback:\n\t\t{fail_feedback}')
            fail_count+=1
            if fail_count == 5:
                relevant_response = fallback_message
                need_decoding = False

    return relevant_response, task_response