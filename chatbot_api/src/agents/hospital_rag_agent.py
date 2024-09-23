import os
from chains.education_chain import education_chain
from chains.papers_chain import papers_chain
from chains.hospital_review_chain import reviews_vector_chain
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)

HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")

hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [
    Tool(
        name="Experiences",
        func=reviews_vector_chain.invoke,
        description="""Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """,
    ),

    Tool(
        name="EducationGraph",
        func=education_chain.invoke,  # Update the function name according to your chain
        description="""Useful for answering questions about Khayrul's education background, 
        including institutions, degrees, GPA, thesis titles, and courses taken. Use the entire 
        prompt as input to the tool. For instance, if the prompt is "What is Khayrul's GPA 
        for his Ph.D.?", the input should be "What is Khayrul's GPA for his Ph.D.?"
        """,
    ),
    Tool(
        name="PaperGraph",
        func=papers_chain.invoke,  # Ensure this points to your papers_chain
        description="""Useful for answering questions about Khayrul's published papers and the 
        skills he has utilized or gained through them. Use the entire prompt as input to the tool.
        For instance,If the prompt is "What softwares are utilized in Khayrul's latest paper?",
        the input should be "What softwares are utilized in Khayrul's latest paper?".
        """
    ),

    Tool(
        name="Waits",
        func=get_current_wait_times,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """,
    ),
    Tool(
        name="Availability",
        func=get_most_available_hospital,
        description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        """,
    ),
]

chat_model = ChatOpenAI(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0,
)

hospital_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_rag_agent_executor = AgentExecutor(
    agent=hospital_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
