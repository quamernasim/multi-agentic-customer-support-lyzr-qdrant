from qdrant_client import QdrantClient
from lyzr_automata import Task
from lyzr_automata.tasks.task_literals import InputType, OutputType

from llm import load_gemini_model
from agents_util.agents import (
    TenantResolverAgent, 
    CustomerInfoExtractorAgent,
    TicketExtractorAgent,
    FAQExtractorAgent,
    HandbookExtractorAgent,
    PolicyExtractorAgent,
    RouterAgent,
    SentimentAgent, 
    EscalationAgent, 
    ResponseAgent,
    ReturnValidationAgent,
    ProductQualityCheckAgent,
    OrderIDExtractorAgent,
    ImagePathExtractorAgent,
    OrderInfoExtractorAgent
)
from qdrant_util.qdrant_retriever import (
    retrieve_customer_info, 
    retrieve_customer_helpdesk_logs, 
    retrieve_related_knowledge_base,
    retrieve_order_info,
    retrieve_image_info
)



gemini_model = load_gemini_model(model_name="gemini-2.0-flash")
qdrant = QdrantClient(host="localhost", port=6333)

# resolved_faq_tags = ['payments']
# resolved_policy_tags = ['payments']
# resolved_handbook_tags = ['payments']

def get_image_path_extraction_task(user_input):
    task = Task(
        name="ImagePathExtraction",
        agent=ImagePathExtractorAgent,
        model=gemini_model,
        instructions=user_input,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task

def get_order_id_extraction_task(user_input):
    task = Task(
        name="OrderIDExtraction",
        agent=OrderIDExtractorAgent,
        model=gemini_model,
        instructions=user_input,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task

def get_return_product_validation_task(tenant_id, customer_id, order_id, image_path):

    order_info = retrieve_order_info(
        client = qdrant,
        tenant_id = tenant_id,
        customer_id = customer_id,
        order_id = order_id
    )
    retrived_image_info = retrieve_image_info(
        client = qdrant,
        image_path=image_path,
        tenant_id = tenant_id,
        customer_id = customer_id,
        top_k=1,
        k_prefetch=10
    )
    context = f"""
Original Product Info
----------------------
{order_info}

Retrived Image Info
--------------------
{retrived_image_info}
"""
    task = Task(
        name="ReturnValidation",
        agent=ReturnValidationAgent,
        model=gemini_model,
        instructions=context,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task


def get_order_info_task(tenant_id, customer_id, order_id):
    order_info = retrieve_order_info(
        client = qdrant,
        tenant_id = tenant_id,
        customer_id = customer_id,
        order_id = order_id
    )
    task = Task(
        name="OrderInfoExtraction",
        agent=OrderInfoExtractorAgent,
        model=gemini_model,
        instructions=order_info,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task

def get_product_quality_check_task(tenant_id, customer_id, order_id, image_path):
    order_info = retrieve_order_info(
        client = qdrant,
        tenant_id = tenant_id,
        customer_id = customer_id,
        order_id = order_id
    )
    retrived_image_info = retrieve_image_info(
        client = qdrant,
        image_path=image_path,
        tenant_id = tenant_id,
        customer_id = customer_id,
        top_k=1,
        k_prefetch=10
    )
    context = f"""
Original Product Info
----------------------
{order_info}

Retrived Image Info
--------------------
{retrived_image_info}
"""
    task = Task(
        name="ProductQualityChecker",
        agent=ProductQualityCheckAgent,
        model=gemini_model,
        instructions=context,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task
    

def get_tenant_identification_task(user_input):
    task = Task(
        name="TenantIdentification",
        agent=TenantResolverAgent,
        model=gemini_model,
        instructions=user_input,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task

def get_customer_info_extraction_task(user_input, tenant_id, customer_id):
    context = {
        "user_query": user_input,
        "customer_info": retrieve_customer_info(
            client = qdrant,
            tenant_id = tenant_id,
            customer_id = customer_id,
        )
    }
    task = Task(
        name="CustomerInfoExtraction",
        agent=CustomerInfoExtractorAgent,
        model=gemini_model,
        instructions=context,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task

def get_ticket_extraction_task(user_input, customer_id, tenant_id, top_k=3, k_prefetch=10):
    context = {
        "user_query": user_input,
        "helpdesk_logs": retrieve_customer_helpdesk_logs(
            client = qdrant,
            query =  user_input, 
            customer_id = customer_id,
            tenant_id = tenant_id,
            top_k = top_k,
            k_prefetch = k_prefetch
        )
    }
    task = Task(
        name="TicketExtraction",
        agent=TicketExtractorAgent,
        model=gemini_model,
        instructions=context,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task

def get_faq_extraction_task(user_input, tenant_id, top_k=3, k_prefetch=10, fail_feedback=None):
    context = {
        "user_query": user_input,
        "faqs": retrieve_related_knowledge_base(
            client = qdrant,
            query =  user_input, 
            source_type = "faqs",
            tenant_id = tenant_id,
            tags = None,
            top_k = top_k,
            k_prefetch = k_prefetch
        )
    }
    if fail_feedback:
        context['error_in_prev_generated_response'] = fail_feedback
    task = Task(
        name="FAQExtraction",
        agent=FAQExtractorAgent,
        model=gemini_model,
        instructions=context,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task

def get_handbook_extraction_task(user_input, tenant_id, top_k=3, k_prefetch=10, fail_feedback=None):
    context = {
        "user_query": user_input,
        "handbook": retrieve_related_knowledge_base(
            client = qdrant,
            query =  user_input, 
            source_type = "handbook",
            tenant_id = tenant_id,
            tags = None,
            top_k = top_k,
            k_prefetch = k_prefetch
        )
    }
    if fail_feedback:
        context['error_in_prev_generated_response'] = fail_feedback
    task = Task(
        name="HandbookExtraction",
        agent=HandbookExtractorAgent,
        model=gemini_model,
        instructions=context,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task

def get_policy_extraction_task(user_input, tenant_id, top_k=3, k_prefetch=10, fail_feedback=None):
    context = {
        "user_query": user_input,
        "policy": retrieve_related_knowledge_base(
            client = qdrant,
            query =  user_input, 
            source_type = "policy",
            tenant_id = tenant_id,
            tags = None,
            top_k = top_k,
            k_prefetch = k_prefetch
        )
    }
    if fail_feedback:
        context['error_in_prev_generated_response'] = fail_feedback
    task = Task(
        name="PolicyExtraction",
        agent=PolicyExtractorAgent,
        model=gemini_model,
        instructions=context,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return task



def get_routing_task(user_input):
    route_task = Task(
        name="RouteIssue",
        agent=RouterAgent,
        model=gemini_model,
        instructions=user_input,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return route_task


def get_sentiment_analysis_task(user_input):
    senti_task = Task(
        name="AnalyzeSentiment",
        agent=SentimentAgent,
        model=gemini_model,
        instructions=user_input,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT
    )
    return senti_task


def get_escalation_task(user_input, route_task, senti_task):
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
    return escalation_task

def get_response_task(full_context, history, route_task, senti_task, escalation_task):
    resp_instructions = (
        f"You have been provided with the Issue, "
        f"sentiment of the user, "
        f"Knowledge Base context:\n{full_context}\n,"
        f"History: {history}\n"
        f"and Escalation status\n\n"
        "Now craft the final support response based on all of the above."
    )
    response_task = Task(
        name="GenerateResponse",
        agent=ResponseAgent, 
        model=gemini_model,
        instructions=resp_instructions,
        input_type=InputType.TEXT,
        output_type=OutputType.TEXT,
        input_tasks = [
            route_task, 
            senti_task, 
            escalation_task
        ]
    )
    return response_task