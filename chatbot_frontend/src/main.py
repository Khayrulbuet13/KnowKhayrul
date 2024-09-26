import os

import requests
import streamlit as st

CHATBOT_URL = os.getenv(
    "CHATBOT_URL", "http://localhost:8000/chatbot-rag-agent"
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot is here to help you explore Khayrul's research and publication. 
        It provides insights into the key concepts behind his work in an interactive way.
        Powered by a [LangChain](https://python.langchain.com/docs/get_started/introduction) agent, 
        it uses retrieval-augmented generation (RAG) to pull information from structured data stored 
        in a Neo4j database and unstructured data managed by ChromaDB. The system is containerized 
        with Docker, ensuring flexibility and scalability.
        """
    )

    st.header("Example Questions")
    # st.markdown(
    #     """- Tell me about your educational background"""
    # )

    # st.markdown("- Can you describe the key methodologies and techniques you utilized \
    #             in your most recent publication on 'Multiplex Image Machine Learning'?")
    
    
    # st.markdown(
    #     """- Can you explain the overall idea of the paper titled 'Multiplex Image Machine Learning?"""
    # )
    # st.markdown(
    #     "- Tell me about your master thesis?"
    # )
    # st.markdown(
    #     """- Did you publish any first-author papers during PhD?"""
    # )

    # st.markdown(
    #     """- Did you have any course related to machine learning?"""
    # )

    # st.markdown(
    #     """- From where you got your master degree?"""
    # )

    st.markdown(
    """- Could you provide an overview of your educational background?"""
    )

    st.markdown(
        """- Can you elaborate on the methodologies and techniques you employed in your recent publication on 'Multiplex Image Machine Learning'?"""
    )

    st.markdown(
        """- Could you explain the main concept and significance of your paper titled 'Multiplex Image Machine Learning'?"""
    )

    st.markdown(
        """- Can you share insights into your master’s thesis and its key contributions?"""
    )

    st.markdown(
        """- Tell me about your research during your Ph.D.?"""
    )

    st.markdown(
        """- Did your coursework include any subjects related to machine learning?"""
    )

    st.markdown(
        """- Where did you complete your master's degree, and what was the focus of your studies?"""
    )





st.title("AskKhayrul!")
st.info(
    """
    Hello! I'm AskKhayrul, Khayrul's AI assistant. I can explain key concepts from 
    his papers and answer questions about his education and work experience as if 
    I’m Khayrul himself. Feel free to ask!
"""
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200:
            output_text = response.json()["output"]
            explanation = response.json()["intermediate_steps"]

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated?", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )
