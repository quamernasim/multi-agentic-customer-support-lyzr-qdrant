import uuid
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline

from llm import load_gemini_model
from utils import (
     get_history, 
     save_message,
     text_2_json,
     task_with_feedback_loop
)
from tasks import (
     get_tenant_identification_task, 
     get_customer_info_extraction_task,
     get_ticket_extraction_task,
     get_faq_extraction_task,
     get_handbook_extraction_task,
     get_policy_extraction_task,
     get_routing_task,
     get_sentiment_analysis_task,
     get_escalation_task,
     get_response_task
)
from caching import SemanticCache

gemini_model = load_gemini_model(model_name="gemini-2.0-flash")
cache = SemanticCache(threshold=0.2)

def run_session():
    session_id = str(uuid.uuid4())
    print(f"Session {session_id} started.")
    
    while True:
        user_input = input("User: ")
        # ask the user to provide with the user id for now
        # but when this becomes a product, it can be directly taken from the request body
        resolved_customer_id = 'CUST-010'
        if user_input.lower() == "exit":
            break

        save_message(session_id, "user", user_input)

        history = get_history(session_id)
        tenant_task_response = get_tenant_identification_task(user_input).execute()
        tenant_task_response = text_2_json(tenant_task_response)
        resolved_tenant_id = tenant_task_response['response']['tenant_type']
        # save_message(session_id, "TenantResolverAgent", tenant_task_response)

        cached_response = cache.check_cache(user_input, resolved_tenant_id, resolved_customer_id)
        if cached_response:
            final_message = cached_response
        else:
            customer_info_task_response = get_customer_info_extraction_task(user_input, resolved_tenant_id, resolved_customer_id).execute()
            customer_info_task_response = text_2_json(customer_info_task_response)
            final_customer_info = customer_info_task_response['response']['customer_info']
            # save_message(session_id, "CustomerInfoRetrieverAgent", customer_info_task_response)

            ticket_task_response = get_ticket_extraction_task(user_input, resolved_customer_id, resolved_tenant_id, top_k=3, k_prefetch=10).execute()
            ticket_task_response = text_2_json(ticket_task_response)
            relevant_ticket_info = ticket_task_response['response']['related_tickets']
            # save_message(session_id, "TicketInfoRetrieverAgent", ticket_task_response)

            # doesn't need to retrun anything if it doesn't find any related faq
            # can have multiple faqs, if need be
            # do not blindly rely on similary with query value, use your own brain
            relevant_faqs, faq_task_response = task_with_feedback_loop(
                user_input, get_faq_extraction_task, resolved_tenant_id, 'related_faqs', 
                session_id, 'FAQsRetrieverAgent',"Couldn't find any FAQs related to the user query", top_k=3, k_prefetch=10
            )

            relevant_policy, policy_task_response = task_with_feedback_loop(
                user_input, get_policy_extraction_task, resolved_tenant_id, 'related_policies', 
                session_id, 'PolicyRetrieverAgent',"Couldn't find any policy related to the user query", top_k=3, k_prefetch=10
            )

            relevant_handbook, handbook_task_response = task_with_feedback_loop(
                user_input, get_handbook_extraction_task, resolved_tenant_id, 'related_handbooks', 
                session_id, 'HandbookRetrieverAgent',"Couldn't find any handbook related to the user query", top_k=3, k_prefetch=10
            )
            
            full_context = f"""
Customer Info
-------------
{final_customer_info}

Related User's Issued Tickets
------------------------------
{relevant_ticket_info}

Relavant FAQs for the User Query
---------------------------------
{relevant_faqs}

Relavant Policies for the User Query
---------------------------------
{relevant_policy}

Relavant Handbooks for the User Query
---------------------------------
{relevant_handbook}
"""
            routing_task = get_routing_task(user_input)
            senti_task = get_sentiment_analysis_task(user_input)
            escalation_task = get_escalation_task(user_input, routing_task, senti_task)
            responding_task = get_response_task(full_context, history, routing_task, senti_task, escalation_task)


            pipeline = LinearSyncPipeline(
                name="MultiAgentSupport",
                completion_message="Done",
                tasks=[
                    routing_task,
                    senti_task,
                    escalation_task,
                    responding_task
                ]
            )
            outputs = pipeline.run()

            response = outputs[-1]['task_output']
            response = text_2_json(response)
            final_message = response['response']['message']
            cache.add_to_cache(user_input, final_message, resolved_tenant_id, resolved_customer_id)

        print(f"Agent: {final_message}")
        save_message(session_id, "assistant", final_message)

if __name__ == "__main__":
    run_session()
