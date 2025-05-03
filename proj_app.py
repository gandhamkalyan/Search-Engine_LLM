from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
import os
from langchain.agents import AgentType,initialize_agent
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()

arxiv_api_wrapper=ArxivAPIWrapper(doc_content_chars_max=300,top_k_results=1)
arxiv=ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

wiki_api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wikipedia=WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

search=DuckDuckGoSearchRun(name="Search") #Will use this to search from internet

st.title("Langchain - Chat with Search")

st.sidebar.title('Settings')
api_key=st.sidebar.text_input("Enter your Groq API Key:",type='password')

if 'messages' not in st.session_state: #Check if chat memory exists
    #Create it with a greeting message
    st.session_state['messages']=[{
        "role":"Assistant","content":"Hi,I'm a chatbot who can search the web.How can I help you? "
    }]

# If the user is already in session:
# Go through each message stored in memory (old messages)...
# and display it on the screen like a chat.”
# Example:
# Chatbot: "Hi..."
# You: "What is AI?"
# Chatbot: "AI means..."

# It keeps showing the full conversation history

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

#Display an input box in a chatbot interface (usually at the bottom of the screen).
if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    # Save the actual user message
    st.session_state.messages.append({"role":"user","content":prompt})
    # Display it on the screen like a chat bubble
    st.chat_message("user").write(prompt)

    # So, instead of waiting for the full answer to be completed and then receiving it in one go, 
    # with streaming, you'll get pieces of the answer immediately as the model works through it.
    llm=ChatGroq(model="llama3-8b-8192",groq_api_key=api_key,streaming=True)

    tools=[wikipedia,arxiv,search]

    # initialize_agent : This function is setting up the agent. It takes a list of tools, a language model (LLM), 
    # and additional configuration settings to define how the agent should behave.

    search_agents=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        #StreamlitCallbackHandler:It’s a LangChain helper that shows what the AI is doing step by step in the Streamlit app.
        # St.container():
        # Create a box in the UI where I want to print stuff step-by-step.”
        # This is where the assistant's steps like “Thinking...”, “Searching Wikipedia…” will appear.
        # So it's just where to show the output on the screen.

        # expand_new_thoughts:If it's True: each step (like "Tool selected", "Search query", "Observation") is fully expanded.
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agents.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)

