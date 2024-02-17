import streamlit as st
import os
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from trulens_eval import TruChain, Feedback, OpenAI, Huggingface, Tru

hugs = Huggingface()
openai = OpenAI()
tru = Tru()

# Load environment variables from .streamlit/secrets.toml
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]



# Define your page functions
def home_page():
    st.title("Home Page")
    st.write("Welcome to the home page!")

def Genarate_story():
    # Streamlit frontend
# Streamlit frontend
    st.title("Boost Your Comprehension")

    if "text" not in st.session_state:
        st.session_state.text = ""
    selected_values = st.multiselect("Select values", [
        "Honesty", "Courage", "Perseverance", "Humility", "Compassion", "Generosity",
        "Gratitude", "Forgiveness", "Loyalty", "Patience", "Respect", "Responsibility",
        "Tolerance", "Justice", "Fairness", "Caring", "Kindness", "Optimism", "Wisdom",
        "Trustworthiness"
    ])
    # Build LLM chain
    template = """Write a creative, engaging story that brings the scene to life. Describe the characters, setting, and actions in a way that would captivate a young audience the story must contains this """+' '.join(selected_values) +"""
            
            Human: {human_input}
            Chatbot:"""
    prompt = PromptTemplate(
        input_variables=[ "human_input"], template=template
    )
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

     # Build LLM chain
    template2 = """Generate quiz from this story:
            
            Human: {human_input}
            Chatbot:"""
    prompt2 = PromptTemplate(
        input_variables=["human_input"], template=template2
    )
    
    llm2 = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain2 = LLMChain(llm=llm2, prompt=prompt2, verbose=True)

    # Question/answer relevance between overall question and answer.
    f_relevance = Feedback(openai.relevance).on_input_output()

    # Moderation metrics on output
    f_hate = Feedback(openai.moderation_hate).on_output()
    f_violent = Feedback(openai.moderation_violence, higher_is_better=False).on_output()
    f_selfharm = Feedback(openai.moderation_selfharm, higher_is_better=False).on_output()
    f_maliciousness = Feedback(openai.maliciousness_with_cot_reasons, higher_is_better=False).on_output()

    # TruLens Eval chain recorder
    chain_recorder = TruChain(
        chain, app_id="GPT4-story", feedbacks=[f_relevance, f_hate, f_violent, f_selfharm, f_maliciousness]
    )
    chain_recorder2 = TruChain(
        chain2, app_id="GPT4-Quiz", feedbacks=[f_relevance, f_hate, f_violent, f_selfharm, f_maliciousness]
    )

    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.text_input("Something to add (optional)?")
    if st.button("GENERATE") :
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Record with TruLens
            with chain_recorder as recording:
                full_response = chain.run(prompt)
            message_placeholder = st.empty()
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.text =full_response
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    if st.button("test") :

        with st.chat_message("assistant"):
                with chain_recorder2 as recording:
                    prompt2=st.session_state.text
                    full_response2 = chain2.run(prompt2)
                message_placeholder = st.empty()
                message_placeholder.markdown(full_response2 + "▌")
                message_placeholder.markdown(full_response2)
        st.session_state.messages.append(
                {"role": "assistant", "content": full_response2})
        
    
        
def Quiz_page():
    st.title("Test Your Comprehension")
    
    
def contact_page():
    st.title("Contact Page")
    st.write("Feel free to contact us.")
    

# Create a dictionary to map page names to page functions
pages = {
    "Home": home_page,
    "Chat": Genarate_story,
    "Quiz": Quiz_page,
    "Contact": contact_page,
}

# Create a sidebar with menu items
selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

# Display the selected page
pages[selected_page]()


tru.run_dashboard()

