import streamlit as st
import os
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from trulens_eval import TruChain, Feedback, OpenAI, Huggingface, Tru
from helpers.openai_utils import get_quiz_data
from helpers.quiz_utils import string_to_list, get_randomized_options
import base64
from pathlib import Path
from openai import OpenAI as OP
import glob
from PIL import Image
import requests
import json
import re
import time
import uuid
from io import BytesIO
import numpy as np
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path
import io


hugs = Huggingface()

tru = Tru()

# Load environment variables from .streamlit/secrets.toml
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["x-portkey-api-key"] = st.secrets["x-portkey-api-key"]
vopenai = OP(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url="https://api.portkey.ai/v1", ## Point to Portkey's gateway URL
    default_headers= {
        "x-portkey-api-key": st.secrets["x-portkey-api-key"],
        "x-portkey-provider": "openai",
        "Content-Type": "application/json"
    }
)
openai = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url="https://api.portkey.ai/v1", ## Point to Portkey's gateway URL
    default_headers= {
        "x-portkey-api-key": st.secrets["x-portkey-api-key"],
        "x-portkey-provider": "openai",
        "Content-Type": "application/json"
    }
)

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
    response = vopenai.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    st.session_state.image_url = response.data[0].url
    return st.session_state.image_url

# Define your page functions
def home_page():
    st.title("Home Page")
    st.write("Welcome to the home page!")

with open('style.css') as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def Genarate_story():
    # Streamlit frontend
# Streamlit frontend

    st.title("Boost Your Comprehension")

    if "text" not in st.session_state:
        st.session_state.text = ""

    if 'image_url' not in st.session_state:
            st.session_state.image_url = ""

    # Include external CSS file
    st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)
    with open("./frozen.jpg", "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode("utf-8")
    

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

        
    selected_values = st.multiselect("Select values", [
        "Honesty", "Courage", "Perseverance", "Humility", "Compassion", "Generosity",
        "Gratitude", "Forgiveness", "Loyalty", "Patience", "Respect", "Responsibility",
        "Tolerance", "Justice", "Fairness", "Caring", "Kindness", "Optimism", "Wisdom",
        "Trustworthiness"
    ])

    # Build LLM chain
    template = """Write a creative, engaging story that brings the scene to life. Describe the characters, setting, and actions in a way that would captivate a young audience the story must not exceed 200 words and contain these values """+' '.join(selected_values) +"""
            
            Human: {human_input}
            Chatbot:"""
    prompt = PromptTemplate(
        input_variables=[ "human_input"], template=template
    )
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo" ,base_url="https://api.portkey.ai/v1", ## Point to Portkey's gateway URL
    default_headers= {
        "x-portkey-api-key": st.secrets["x-portkey-api-key"],
        "x-portkey-provider": "openai",
        "Content-Type": "application/json"})
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)



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


    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    prompt = st.text_input("Something to add (optional)?")


    if st.button("Generate", type="primary"):
         
        st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.image_url = generate_image(prompt)
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
            <img class="styled-img" src="{st.session_state.image_url}" alt="Frozen Image">
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
        generate_speech(st.session_state.text)
        st.success("Speech generated successfully!")
        
         # Display the audio file to the user
        st.audio("speech.mp3", format="audio/mp3") 
def Quiz_page():
    st.title("Test Your Comprehension")
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    # Include external CSS file
    st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)

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
    with st.form("user_input"):
        if 'text' not in st.session_state:
            st.session_state.text = ""
        story_text =st.session_state.text
        st.write("You entered:", st.session_state.text)
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
def Creativity():
    api_key=st.secrets["OPENAI_API_KEY"]
    # Sidebar for API key input
    # api_key = st.sidebar.text_input("Enter your API key", type="password")
    # if not api_key:
    #     api_key = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"

    # Initialize the OpenAI client with the API Key provided by the user or from the environment
    # client = OpenAI(api_key=api_key)
    client = OP(api_key=os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY")
    st.title('AI Image Analyzer and Generator')
    if 'button_id' not in st.session_state:
            st.session_state.button_id = ""
    button_id = st.session_state["button_id"]
    def png_export():
        try:
            Path("tmp/").mkdir()
        except FileExistsError:
            pass

        # Regular deletion of tmp files
        # Hopefully callback makes this better
        now = time.time()
        N_HOURS_BEFORE_DELETION = 1
        for f in Path("tmp/").glob("*.png"):
            # st.write(f, os.stat(f).st_mtime, now)
            if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
                Path.unlink(f)

        if st.session_state["button_id"] == "":
            st.session_state["button_id"] = re.sub(
                "\d+", "", str(uuid.uuid4()).replace("-", "")
            )

        
        file_path = f"tmp/{button_id}.png"

        custom_css = f""" 
            <style>
                #{button_id} {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    background-color: rgb(255, 255, 255);
                    color: rgb(38, 39, 48);
                    padding: .25rem .75rem;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: rgb(230, 234, 241);
                    border-image: initial;
                }} 
                #{button_id}:hover {{
                    border-color: rgb(246, 51, 102);
                    color: rgb(246, 51, 102);
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: rgb(246, 51, 102);
                    color: white;
                    }}
            </style> """

        data = st_canvas(update_streamlit=True, key="png_export")
        if data is not None and data.image_data is not None:
            img_data = data.image_data

            # Create a new image with white background
            white_background = Image.new("RGBA", (data.image_data.shape[1], data.image_data.shape[0]), (255, 255, 255, 255))
            white_background.paste(Image.fromarray(img_data.astype("uint8"), mode="RGBA"), (0, 0), mask=Image.fromarray(img_data.astype("uint8"), mode="RGBA"))

            # Save the image with white background
            white_background.save(file_path, "PNG")

            buffered = BytesIO()
            white_background.save(buffered, format="PNG")
            img_data = buffered.getvalue()
            try:
                # some strings <-> bytes conversions necessary here
                b64 = base64.b64encode(img_data).decode()
            except AttributeError:
                b64 = base64.b64encode(img_data).decode()

            # dl_link = (
            #     custom_css
            #     + f'<a download="{file_path}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
            # )
            # st.markdown(dl_link, unsafe_allow_html=True)

    def save_image(image_bytes, filename, output_dir="original_image"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'wb') as image_file:
            image_file.write(image_bytes)
        st.write(f"Image saved to {file_path}")

    def download_and_encode_image(image_input):
        if image_input.startswith(('http://', 'https://')):
            response = requests.get(image_input, timeout=10)
            image_bytes = response.content
            filename = image_input.split('/')[-1]
        else:
            with open(image_input, "rb") as image_file:
                image_bytes = image_file.read()
                filename = os.path.basename(image_input)

        save_image(image_bytes, filename)  # Save the image to the local directory

        # Use io.BytesIO to wrap the image data before passing it to PIL
        image_io = io.BytesIO(image_bytes)
        pil_image = Image.open(image_io)

        # Convert the PIL image back to bytes
        pil_image_bytes = io.BytesIO()
        pil_image.save(pil_image_bytes, format='PNG')
        pil_image_bytes = pil_image_bytes.getvalue()

        return base64.b64encode(pil_image_bytes).decode('utf-8')


    def get_image_analysis_streamlit(base64_image):
        """Displays image analysis/description using GPT-4 Vision in Streamlit."""
        try:
            response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this image as a prompt for a text to generate a title of a story for kids according to this image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=250,
            )
            
            responses = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    responses += str(chunk.choices[0].delta.content)
            print("\n" + "="*50 + "\nVision Response:\n" + "="*50)
            print(responses.rstrip())
            return responses.rstrip()
        except Exception as e:
            print(f"An error occurred during image analysis: {e}")
            return None

    def modify_description_streamlit(original_description):
        """Allows users to modify the image description using a Streamlit text input."""
        st.write(f"Original Description: {original_description}")
        modification = st.text_input("How would you like to modify the image description? (e.g., 'add a hat, change the background, etc.'): ").strip()
        if modification:
            new_description = f"{original_description}, modified to include {modification}"
        else:
            new_description = original_description
        return new_description


    def generate_image_with_dalle_streamlit(prompt):
        try:
            print("\n" + "="*50 + "\nFinal Prompt Sent to DALL-E 3:\n" + "="*50)
            print(prompt)
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1792x1024",
                quality="standard",
                response_format="b64_json",
                n=1,
            )
            if response:
                b64_data = response.data[0].b64_json
                revised_prompt = response.data[0].revised_prompt
                st.session_state['b64_image'] = b64_data  # Store in session for display
                return b64_data, revised_prompt
        except Exception as e:
            st.error(f"Error generating image: {e}")
            return None, None


    def save_base64_image_streamlit(b64_data, original_name, output_dir="generated_images"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        base_filename = os.path.splitext(os.path.basename(original_name))[0]
        # Logic to create a unique filename
        existing_files = glob.glob(os.path.join(output_dir, f"{base_filename}_generated_*.png"))
        highest_num = 0
        for f in existing_files:
            try:
                num = int(f.rsplit('_', 1)[-1].split('.')[0])
                highest_num = max(highest_num, num)
            except ValueError:
                continue

        new_num = highest_num + 1
        new_filename = f"{base_filename}_generated_{new_num:02d}.png"
        file_path = os.path.join(output_dir, new_filename)

        img_data = base64.b64decode(b64_data)
        with open(file_path, 'wb') as f:
            f.write(img_data)
        
        st.success(f"Generated image saved as {new_filename}")


    def show_image_gallery(directory="generated_images"):
        if os.path.exists(directory):
            images = os.listdir(directory)
            for image in images:
                # Use a combination of directory and image name to ensure the key is unique
                unique_key = f"{directory}_{image}"
                col1, col2 = st.columns([20, 1])  # Adjust the ratio as needed
                with col1:  # Image display column
                    image_path = os.path.join(directory, image)
                    st.image(image_path, caption=image, use_column_width=True)
                with col2:  # Deletion button column
                    # Use the unique key for the button
                    if st.button("X", key=unique_key):
                        os.remove(image_path)  # Delete the image file
                        st.rerun()  
        

    def clear_inputs():
        # Explicitly clear the relevant session state keys
        keys_to_clear = ['image_input', 'modification', 'last_image_input', 'original_description', 'process_image']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()  # Force a rerun of the app to reset state
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    png_export()
    # Initialize or reset session state keys
    if 'image_input' not in st.session_state:
        st.session_state['image_input'] = ""
    if 'modification' not in st.session_state:
        st.session_state['modification'] = ""
    
    st.session_state['image_input'] = st.text_input("Enter the path to a local image or an image URL or Autofill:", value=st.session_state['image_input'] )

    #Input for image URL or path with session state management
    if st.button("Autofill"):
            st.session_state['image_input'] = f"tmp/{button_id}.png"
    # if know this would work: st.session_state['text'] = autofill_value
    else:
        autofill_value = ""

    # Process the image input if it's provided
    if st.session_state['image_input']:
        # Check if it's a new image input or the same as before
        if 'last_image_input' not in st.session_state or st.session_state['last_image_input'] != st.session_state['image_input']:
            base64_image = download_and_encode_image(st.session_state['image_input'])
            original_description = get_image_analysis_streamlit(base64_image)
            st.session_state['original_description'] = original_description
            st.session_state['last_image_input'] = st.session_state['image_input']
            st.session_state['modification'] = ""  # Reset modification input for new image

        # Show original description to the user
        if 'original_description' in st.session_state:
            st.write(f"Your Vision : {st.session_state['original_description']}")
        
        # Modification text input
        st.session_state['modification'] = st.text_input("Enter your description to add for Dalle here:", value=st.session_state['modification'])
        if 'modification' in st.session_state and st.session_state['modification']:
            # This ensures we only print when there's an actual modification provided.
            print("\n" + "="*50 + "\nUser's Modification Input:\n" + "="*50)
            print(st.session_state['modification'])

        
        # Generate modified image on button click
        if st.button("Generate Modified Image"):
            modified_description = f"{st.session_state['original_description']} {st.session_state['modification']}" if st.session_state['modification'] else st.session_state['original_description']
            b64_data, revised_prompt = generate_image_with_dalle_streamlit(modified_description)
            if b64_data:
                img_data = base64.b64decode(b64_data)
                st.image(img_data, caption="Generated Image", use_column_width=True)
                save_base64_image_streamlit(b64_data, st.session_state['image_input'])
            else:
                st.error("Unable to generate modified image due to an error.")

    # Clear button to reset inputs and state
    if st.button("Clear & Start Over"):
        clear_inputs()

    st.title("Image Gallery")


    # Expanders for viewing images in directories
    with st.expander("View Original Images"):
        show_image_gallery(directory="original_image")
    # Display the image gallery with deletion options
    with st.expander("View Generated Images"):
        show_image_gallery(directory="generated_images")
    

# Create a dictionary to map page names to page functions
pages = {
    "Home": home_page,
    "Chat": Genarate_story,
    "Quiz": Quiz_page,
    "Test your Creativity" :Creativity,
    "Contact": contact_page,
}

# Create a sidebar with menu items
selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

# Display the selected page
pages[selected_page]()


tru.run_dashboard() #port=8500
