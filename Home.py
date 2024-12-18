import os
import random

import pymysql.cursors
import streamlit as st

from datetime import datetime
from streamlit_extras.switch_page_button import switch_page


def login():
    # skip customize user name for debug mode

    with st.form("user_login"):
        # st.write('## Enter Your Name to Start the Session')
        st.write(
            '### Getting obsessed with tons of different text-to-image generation models available online? Want to find the most suitable one for your taste?')
        st.write('**GEMRec** is here to help you! Enter your name to try it out👇!')
        user_id = st.text_input(
            "Enter your name 👇",
            label_visibility='collapsed',
            disabled=False,
            placeholder='You can leave it blank to be anonymous'
        )
        # st.write('You can leave it blank to be anonymous.')

        # st.session_state.show_NSFW = st.toggle(':orange[show potentially mature content]', help='Inevitably, a few images might be NSFW, even if we tried to elimiate NFSW content in our prompts. We calculate a NSFW score to filter them out. Please check only if you are 18+ and want to take a look at the whole GEMRec-18k dataset', value=False, key='mature_content')
        st.session_state.show_NSFW = False # set to falso by default temporarily

        # Every form must have a submit button.
        submitted = st.form_submit_button("Start")
        if submitted:
            save_user_id(user_id)
            switch_page("gallery")


def save_user_id(user_id):
    user_id = user_id[:60]
    print(user_id)
    if not user_id:
        user_id = 'anonymous' + str(random.randint(0, 100000))
    st.session_state.user_id = [user_id, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")]
    st.session_state.assigned_rank_mode = random.choice(['Drag and Sort', 'Battle'])
    st.session_state.epoch = {'gallery': 0, 'ranking': {}, 'summary': {'overall': 0}}


def logout():
    st.session_state.pop('user_id', None)
    st.session_state.pop('selected_dict', None)
    st.session_state.pop('epoch', None)
    st.session_state.pop('score_weights', None)
    st.session_state.pop('gallery_state', None)
    st.session_state.pop('edit_state', None)
    st.session_state.pop('progress', None)
    st.session_state.pop('pointer', None)
    st.session_state.pop('counter', None)
    st.session_state.pop('gallery_focus', None)
    st.session_state.pop('assigned_rank_mode', None)
    st.session_state.pop('show_NSFW', None)
    st.session_state.pop('modelVersion_standings', None)


def project_info():
    with st.sidebar:
        st.write('## About')
        st.write(
            "This is a web application **for individual users to quickly dig out the most preferable text-to-image models from [civitai](https://civitai.com) for different prompts**. Our research aims to understand personal preference towards generative models  and you can contribute by playing with this tool and giving us your feedback! "
        )

        st.write(
            "After picking images you liked from Gallery and a Ranking Contest, a summary dashboard will be presented **indicating your preferred models with download links ready to be deployed in [Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)** !"
        )


def connect_to_db():
    conn = pymysql.connect(
        host=os.environ.get('RANKING_DB_HOST'),
        port=int(os.environ.get('RANKING_DB_PORT')),
        database='GEMRec_test',
        user=os.environ.get('RANKING_DB_USER'),
        password=os.environ.get('RANKING_DB_PASSWORD'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    return conn


if __name__ == '__main__':
    # print(st.source_util.get_pages('Home.py'))
    st.set_page_config(page_title="Login", page_icon="🏠", layout="wide")
    project_info()
    st.write('A Research by [MAPS Lab](https://whongyi.github.io/MAPS-research), [NYU Shanghai](https://shanghai.nyu.edu)')
    st.title("🙌 Welcome to GEMRec Gallery!")

    if 'user_id' not in st.session_state:
        login()
    else:
        st.write(f"You have already logged in as `{st.session_state.user_id[0]}`")
        st.write(f"Assigned ranking mode: `{st.session_state.assigned_rank_mode}`")
        st.button('Log out', on_click=logout)

    st.write('---')
    st.write('### FAQ')
    with st.expander(label='**🤔 How to use this webapp?**'):
        st.write('### Check out the demo video below')
        # st.info('Interface shown in this video demo is a bit different from the current webapp because it\'s outdated, but the basic idea is the same.')
        st.video('https://youtu.be/iSVM_yyIwlg')

    with st.expander(label='**ℹ️ What is GEMRec project?**'):
        st.write('### About GEMRec')
        st.write("**GE**nerative **M**odel **Rec**ommendation (**GEMRec**) is a research project by [MAPS Lab](https://github.com/MAPS-research), NYU Shanghai.")
        st.write('### Our Task')
        st.write('Navigate hundreds of text-to-image models through various categories of pre-defined prompts and a graph-based interface. Given a user’s preference and interaction data, we aim to recommend the most preferred generative model for the user.')
        st.write('### Our Approach')
        st.write('We propose a two-stage framework, which contains prompt-model retrieval and generative model  ranking. :red[Your participation in this web application will help us to improve our framework and to further our research on personalization.]')
        # st.write('### Key Contributions')
        # st.write('1. We propose a two-stage framework to approach the Generative Model Recommendation problem. Our framework allows end-users to effectively explore a diverse set of generative models to understand their expressiveness. It also allows system developers to elicit user preferences for items generated from personalized prompts.')
        # st.write('2. We release GEMRec-18K, a dense prompt-model interaction dataset that consists of 18K images generated by pairing 200 generative models with 90 prompts collected from real-world usages, accompanied by detailed metadata and generation configurations. This dataset builds the cornerstone for exploring Generative Recommendation and can be useful for other tasks related to understanding generative models')
        # st.write('3. We take the first step in examining evaluation metrics for personalized image generations and identify several limitations in existing metrics. We propose a weighted metric that is more suitable for the task and opens up directions for future improvements in model training and evaluations.')

    with st.expander(label='**📑 Where can I find the paper and dataset?**'):
        st.write('### Paper')
        st.write('Arxiv: [Towards Personalized Prompt-Model Retrieval for Generative Recommendation](https://arxiv.org/abs/2308.02205)')
        st.write('### GEMRec-18K Dataset')
        st.write('Image dataset: https://huggingface.co/datasets/MAPS-research/GEMRec-PromptBook  \n \
        Model dataset: https://huggingface.co/datasets/MAPS-research/GEMRec-Roster')
        st.write('### Code')
        st.write('Github: https://github.com/maps-research/gemrec')