import streamlit as st
import os
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from trulens_eval import TruChain, Feedback, OpenAI, Huggingface, Tru
import streamlit as st
from helpers.openai_utils import get_quiz_data
from helpers.quiz_utils import string_to_list, get_randomized_options
import base64
from pathlib import Path
from openai import OpenAI as OP
import streamlit_antd_components as sac
from pinecone import Pinecone

pc = Pinecone(api_key="eee55b9a-6754-4a5b-8336-9122b7361457")
index = pc.Index("storyfy")
hugs = Huggingface()
openai = OpenAI()
vopenai = OP(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url="https://api.portkey.ai/v1", ## Point to Portkey's gateway URL
    default_headers= {
        "x-portkey-api-key": st.secrets["x-portkey-api-key"],
        "x-portkey-provider": "openai",
        "Content-Type": "application/json"
    }
)
tru = Tru()

# Load environment variables from .streamlit/secrets.toml
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def generate_speech(input_text):
    speech_file_path = Path("speech.mp3")
    response = vopenai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=input_text
    )
    response.stream_to_file(speech_file_path)


st.set_page_config(
        page_title="Mediterranean AI",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
def generate_image(prompt):
    try:
        print("\n" + "="*50 + "\nFinal Prompt Sent to DALL-E 3:\n" + "="*50)
        print(prompt)
        response = vopenai.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            response_format="b64_json",
            n=1,
        )
        if response:
            b64_data = response.data[0].b64_json
            # revised_prompt = response.data[0].revised_prompt
            st.session_state['b64_image'] = b64_data  # Store in session for display
            return b64_data
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

with open('style.css') as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
 # Display HTML content
st.markdown("""
    <div class="area" >
        <ul class="circles">
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
        </ul>
                
    </div >
    """, unsafe_allow_html=True)


# Define your page functions
def home_page():
    st.session_state['b64_image'] =""
    st.title("Home Page")
    st.write("Welcome to the home page!") 



def Genarate_story():

    if "text" not in st.session_state:
        st.session_state.text = ""

    if 'image_url' not in st.session_state:
            st.session_state.image_url = ""

    st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)
    with open("./frozen.jpg", "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode("utf-8")
    

   
        
    selected_values = st.multiselect("Select values", [
        "Honesty", "Courage", "Perseverance", "Humility", "Compassion", "Generosity",
        "Gratitude", "Forgiveness", "Loyalty", "Patience", "Respect", "Responsibility",
        "Tolerance", "Justice", "Fairness", "Caring", "Kindness", "Optimism", "Wisdom",
        "Trustworthiness"
    ])


    st.markdown('''<div class="pos_h1"><h1>Story Comprehension</h1></div>''',unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    prompt = st.text_input("Something to add (optional)?")
    prompt_image = prompt
    res = vopenai.embeddings.create(
    input=prompt,
    model="text-embedding-ada-002"
    )
    
    xq = res.data[0].embedding
    res = index.query(vector=xq, top_k=2, include_metadata=True)
    # get list of retrieved text
    contexts = [item['metadata']['text'] for item in res['matches']]
    augmented_query = prompt + "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"
    
    template = """Write a creative, engaging story that brings the scene to life. Describe the characters, setting, and actions in a way that would captivate a young audience the story must not exceed 200 words and contain these values 
            """ + ' '.join(selected_values) + """" and you have these references     : """ + ''.join(augmented_query) +   """"
            Human: {human_input} 
            Chatbot:"""
    prompt = PromptTemplate(
    input_variables=["human_input"], template=template
    )
    
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    #Question/answer relevance between overall question and answer.
    f_relevance = Feedback(openai.relevance).on_input_output()
    # Moderation metrics on output
    f_hate = Feedback(openai.moderation_hate).on_output()
    f_violent = Feedback(openai.moderation_violence, higher_is_better=False).on_output()
    f_selfharm = Feedback(openai.moderation_selfharm, higher_is_better=False).on_output()
    f_maliciousness = Feedback(openai.maliciousness_with_cot_reasons, higher_is_better=False).on_output()
    chain_recorder = TruChain(
        chain, app_id="First_chain", feedbacks=[f_relevance, f_hate, f_violent, f_selfharm, f_maliciousness]
    )
    # TruLens Eval chain recorder
    chain_recorder = TruChain(
        chain, app_id="GPT4-story", feedbacks=[f_relevance, f_hate, f_violent, f_selfharm, f_maliciousness]
    )
    
    
    #st.markdown('<button class="btn-donate">Try The Magic</button>', unsafe_allow_html=True)
    if st.button("Try The Magic"):
        
        st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": prompt})
        b64_data = generate_image(prompt_image)
        with st.chat_message("assistant"):
            # Record with TruLens
            with chain_recorder as recording:
                full_response = chain.run(prompt)
            st.markdown(
        f"""
        
        <div class="styled-div">
            <div class="styled-title">
                <h1>The Story</h1>  
            </div>
            <img class="styled-img" src="data:image/jpeg;base64,{b64_data}" alt="Frozen Image">
            <div class="styled-p">
                {full_response}
            </div> 
        </div>
           
           
        """,
        unsafe_allow_html=True
    )
            st.session_state.text =full_response
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})   
        ##################################################################################
        # generate_speech(st.session_state.text)
        # st.success("Speech generated successfully!")
        
        #  # Display the audio file to the user
        # st.audio("speech.mp3", format="audio/mp3") 
        ####################################################################################
def Quiz_page():
    if st.session_state.text != "":
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    
        
        with st.form("user_input"):
            st.markdown('''<div class="pos_h1"><h1>Let's craft the Quiz</h1></div>''',unsafe_allow_html=True)
            if 'text' not in st.session_state:
                st.session_state.text = ""
            
            story_text =st.session_state.text
            st.markdown(
            f"""
            
            <div class="styled-div">
                <div class="styled-title">
                    <h1>The Story</h1>  
                </div>
                <img class="styled-img" src="data:image/jpeg;base64,{st.session_state['b64_image']}" alt="Frozen Image">
                <div class="styled-p">
                    {story_text}
                </div> 
            </div>
            
            
            """,
            unsafe_allow_html=True
        )
            submitted = st.form_submit_button("Craft my quiz!")

        if submitted or ('quiz_data_list' in st.session_state):

            if not OPENAI_API_KEY:
                st.info("Please fill out the OpenAI API Key to proceed. If you don't have one, you can obtain it [here](https://platform.openai.com/account/api-keys).")
                st.stop()
                
            with st.spinner("Crafting your quiz...🤓"):
                if submitted:

                    quiz_data_str = get_quiz_data(story_text, OPENAI_API_KEY)
                    st.session_state.quiz_data_list = string_to_list(quiz_data_str)

                    if 'user_answers' not in st.session_state:
                        st.session_state.user_answers = [None for _ in st.session_state.quiz_data_list]
                    if 'correct_answers' not in st.session_state:
                        st.session_state.correct_answers = []
                    if 'randomized_options' not in st.session_state:
                        st.session_state.randomized_options = []

                    for q in st.session_state.quiz_data_list:
                        options, correct_answer = get_randomized_options(q[1:])
                        st.session_state.randomized_options.append(options)
                        st.session_state.correct_answers.append(correct_answer)

                with st.form(key='quiz_form'):
                    st.subheader("🧠 Quiz Time: Test Your Knowledge!", anchor=False)
                    for i, q in enumerate(st.session_state.quiz_data_list):
                        options = st.session_state.randomized_options[i]
                        default_index = st.session_state.user_answers[i] if st.session_state.user_answers[i] is not None else 0
                        response = st.radio(q[0], options, index=default_index)
                        user_choice_index = options.index(response)
                        st.session_state.user_answers[i] = user_choice_index  # Update the stored answer right after fetching it


                    results_submitted = st.form_submit_button(label='Unveil My Score!')

                    if results_submitted:
                        score = sum([ua == st.session_state.randomized_options[i].index(ca) for i, (ua, ca) in enumerate(zip(st.session_state.user_answers, st.session_state.correct_answers))])
                        st.success(f"Your score: {score}/{len(st.session_state.quiz_data_list)}")

                        if score == len(st.session_state.quiz_data_list):  # Check if all answers are correct
                            st.balloons()
                        else:
                            incorrect_count = len(st.session_state.quiz_data_list) - score
                            if incorrect_count == 1:
                                st.warning(f"Almost perfect! You got 1 question wrong. Let's review it:")
                            else:
                                st.warning(f"Almost there! You got {incorrect_count} questions wrong. Let's review them:")

                        for i, (ua, ca, q, ro) in enumerate(zip(st.session_state.user_answers, st.session_state.correct_answers, st.session_state.quiz_data_list, st.session_state.randomized_options)):
                            with st.expander(f"Question {i + 1}", expanded=False):
                                if ro[ua] != ca:
                                    st.info(f"Question: {q[0]}")
                                    st.error(f"Your answer: {ro[ua]}")
                                    st.success(f"Correct answer: {ca}")
        
        
def contact_page():
        st.title("Contact Page")
        st.write("Feel free to contact us.")
    

# # Create a dictionary to map page names to page functions
# pages = {
#     "Home": home_page,
#     "Chat": Genarate_story,
#     "Quiz": Quiz_page,
#     "Contact": contact_page,
# }

# # Create a sidebar with menu items
# selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

# # Display the selected page
# pages[selected_page]()
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Home'

# st.markdown("""
# <style>
#     [data-testid=stSidebar] {
#         background-color: #FFFFFF;
#     }
# </style>
# """, unsafe_allow_html=True)

with st.sidebar:

    selected_tab =sac.menu([
    sac.MenuItem('Home', icon='house-fill'),
    sac.MenuItem('Storyfy',icon='book-half'),
    sac.MenuItem('Quiz',icon='patch-question-fill'),
    #
    sac.MenuItem('products', icon='box-fill', children=[
        sac.MenuItem('apple', icon='apple'),
        sac.MenuItem('other', icon='pencil-fill', description='other items', children=[
            sac.MenuItem('google', icon='google', description='item description'),
            sac.MenuItem('gitlab', icon='gitlab'),
            sac.MenuItem('wechat', icon='wechat'),
        ]),
    ]),
    sac.MenuItem('disabled', disabled=True),
    sac.MenuItem(type='divider'),
    sac.MenuItem('link', type='group', children=[
        sac.MenuItem('antd-menu', icon='heart-fill', href='https://ant.design/components/menu#menu'),
        sac.MenuItem('bootstrap-icon', icon='bootstrap-fill', href='https://icons.getbootstrap.com/'),
    ]),
    
], color='cyan', size='lg', open_all=True)
if selected_tab != st.session_state.current_tab:
        st.session_state.current_tab = selected_tab

 
if st.session_state.current_tab == 'Home':
    home_page()
elif st.session_state.current_tab == 'Storyfy':
    Genarate_story()  
elif st.session_state.current_tab == 'Quiz':
    Quiz_page()
    



tru.run_dashboard()