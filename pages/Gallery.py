import itertools
import json
import os
import requests

import altair as alt
import extra_streamlit_components as stx
import random
import numpy as np
import pandas as pd
import streamlit as st

from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset, load_from_disk
from datetime import datetime
from huggingface_hub import login
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.tags import tagger_component
from streamlit_extras.no_default_selectbox import selectbox
from sklearn.svm import LinearSVC

from Home import connect_to_db

class GalleryApp:
    def __init__(self, promptBook, images_ds):
        self.promptBook = promptBook
        self.images_ds = images_ds

        # init gallery state
        if 'gallery_state' not in st.session_state:
            st.session_state.gallery_state = 'graph'

        # initialize selected_dict
        if 'selected_dict' not in st.session_state:
            st.session_state['selected_dict'] = {}

        # clear up empty entries in seleted_dict
        for prompt_id in list(st.session_state.selected_dict.keys()):
            if len(st.session_state.selected_dict[prompt_id]) == 0:
                st.session_state.selected_dict.pop(prompt_id)

        if 'gallery_focus' not in st.session_state:
            st.session_state.gallery_focus = {'tag': None, 'prompt': None}

    def gallery_standard(self, items, col_num, info, show_checkbox=True):
        rows = len(items) // col_num + 1
        containers = [st.container() for _ in range(rows)]
        for idx in range(0, len(items), col_num):
            row_idx = idx // col_num
            with containers[row_idx]:
                cols = st.columns(col_num)
                for j in range(col_num):
                    if idx + j < len(items):
                        with cols[j]:
                            # show image
                            # image = self.images_ds[items.iloc[idx + j]['row_idx'].item()]['image']
                            image = f"https://modelcofferbucket.s3-accelerate.amazonaws.com/{items.iloc[idx + j]['image_id']}.png"
                            st.image(image, use_column_width=True)

                            # handel checkbox information
                            prompt_id = items.iloc[idx + j]['prompt_id']
                            modelVersion_id = items.iloc[idx + j]['modelVersion_id']

                            check_init = True if modelVersion_id in st.session_state.selected_dict.get(prompt_id, []) else False

                            # st.write("Position: ", idx + j)

                            if show_checkbox:
                                # show checkbox
                                st.checkbox('Select', key=f'select_{prompt_id}_{modelVersion_id}', value=check_init)

                            # show selected info
                            for key in info:
                                st.write(f"**{key}**: {items.iloc[idx + j][key]}")

    def gallery_graph(self, items):
        items = load_tsne_coordinates(items)

        # sort items to be popularity from low to high, so that most popular ones will be on the top
        items = items.sort_values(by=['model_download_count'], ascending=True).reset_index(drop=True)

        scale = 50
        items.loc[:, 'x'] = items['x'] * scale
        items.loc[:, 'y'] = items['y'] * scale

        nodes = []
        edges = []

        for idx in items.index:
            nodes.append(Node(id=items.loc[idx, 'image_id'],
                              # label=str(items.loc[idx, 'model_name']),
                              # title=f"model name: {items.loc[idx, 'model_name']}\nmodelVersion name: {items.loc[idx, 'modelVersion_name']}\nclip score: {items.loc[idx, 'clip_score']}\nmcos score: {items.loc[idx, 'mcos_score']}\npopularity: {items.loc[idx, 'model_download_count']}",
                              title=f"model name: {items.loc[idx, 'model_name']}",
                              size=20,
                              shape='image',
                              image=f"https://modelcofferbucket.s3-accelerate.amazonaws.com/{items.loc[idx, 'image_id']}.png",
                              x=items.loc[idx, 'x'].item(),
                              y=items.loc[idx, 'y'].item(),
                              # fixed=True,
                              color={'background': '#E0E0E1', 'border': '#ffffff', 'highlight': {'border': '#F04542'}},
                              # opacity=opacity,
                              shadow={'enabled': True, 'color': 'rgba(0,0,0,0.4)', 'size': 10, 'x': 1, 'y': 1},
                              borderWidth=3,
                              borderWidthSelected=3,
                              shapeProperties={'useBorderWithImage': True},
                              )
                         )

        config = Config(width='100%',
                        height='600',
                        directed=True,
                        physics=False,
                        hierarchical=False,
                        interaction={'navigationButtons': True, 'dragNodes': False, 'multiselect': False, 'hover': True},
                        # **kwargs
                        )

        return agraph(nodes=nodes,
                      edges=edges,
                      config=config,
                      )

    def sidebar(self, items, prompt_id, note):
        with st.sidebar:
            # show source
            if isinstance(note, str):
                if note.isdigit():
                    st.caption(f"`Source: civitai`")
                else:
                    st.caption(f"`Source: {note}`")
            else:
                st.caption("`Source: Parti-prompts`")

            # show image metadata
            image_metadatas = ['prompt', 'negativePrompt', 'sampler', 'cfgScale', 'size', 'seed']
            for key in image_metadatas:
                label = ' '.join(key.split('_')).capitalize()
                st.write(f"**{label}**")
                if items[key][0] == ' ':
                    st.write('`None`')
                else:
                    st.caption(f"{items[key][0]}")

            # for note as civitai image id, add civitai reference
            if isinstance(note, str) and note.isdigit():
                try:
                    st.write(f'**[Civitai Reference](https://civitai.com/images/{note})**')
                    res = requests.get(f'https://civitai.com/images/{note}')
                    # st.write(res.text)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    image_section = soup.find('div', {'class': 'mantine-12rlksp'})
                    image_url = image_section.find('img')['src']
                    st.image(image_url, use_column_width=True)
                except:
                    pass

        # return prompt_tags, tag, prompt_id, items

    def text_coloring_add(self, tobe_colored:list, total_items, color_name='orange'):
        if color_name in ['orange', 'red', 'green', 'blue', 'violet', 'yellow']:
            colored = [f':{color_name}[{item}]' if item in tobe_colored else item for item in total_items]
        else:
            colored = [f'[{color_name}] {item}' if item in tobe_colored else item for item in total_items]
        return colored

    def text_coloring_remove(self, tobe_removed):
        if isinstance(tobe_removed, str):
            if tobe_removed.startswith(':'):
                tobe_removed = tobe_removed.split('[')[-1][:-1]

            elif tobe_removed.startswith('['):
                tobe_removed = tobe_removed.split(']')[-1][1:]
        return tobe_removed


    def app(self):
        # print(st.session_state.gallery_focus)
        st.write('### Prompt-Model Retrieval')
        with st.sidebar:
            tagger_component('**Gallery State:**', [st.session_state.gallery_state.title()], color_name=['orange'])
        # st.write('This is a gallery of images generated by the models')

        # build the tabular view
        prompt_tags = self.promptBook['tag'].unique()
        # sort tags by alphabetical order
        prompt_tags = np.sort(prompt_tags)[::1].tolist()

        # set focus tag and prompt index if exists
        if st.session_state.gallery_focus['tag'] is None:
             tag_focus_idx = 0
        else:
            tag_focus_idx = prompt_tags.index(st.session_state.gallery_focus['tag'])

        # add coloring to tag based on selection
        tags_tobe_colored = self.promptBook[self.promptBook['prompt_id'].isin(st.session_state.selected_dict.keys())]['tag'].unique()
        # colored_prompt_tags = [f':orange[{tag}]' if tag in tags_tobe_colored else tag for tag in prompt_tags]
        colored_prompt_tags = self.text_coloring_add(tags_tobe_colored, prompt_tags, color_name='orange')

        # save tag to session state on change
        tag = st.radio('Select a tag', colored_prompt_tags, index=tag_focus_idx, horizontal=True, key='tag', label_visibility='collapsed')

        # remove coloring from tag
        tag = self.text_coloring_remove(tag)
        # print('tag: ', tag)

        # print('current state: ', st.session_state.gallery_state)

        if st.session_state.gallery_state == 'graph':

            items = self.promptBook[self.promptBook['tag'] == tag].reset_index(drop=True)

            prompts = np.sort(items['prompt'].unique())[::1].tolist()

            # print('prompts: ', prompts, 'tags: ', prompt_tags)

            # selt focus prompt index if exists
            if st.session_state.gallery_focus['prompt'] is None or tag != st.session_state.gallery_focus['tag']:
                prompt_focus_idx = 0
            else:
                prompt_focus_idx = 1 + prompts.index(st.session_state.gallery_focus['prompt'])

            # st.caption('Select a prompt')
            subset_selector = st.columns([3, 1])
            with subset_selector[0]:
                selector_bar = st.columns([1, 15])
                with selector_bar[0]:
                    shuffle = st.button('🎲', key='prompt_shuffle', on_click=self.random_gallery_focus, args=(prompt_tags,), use_container_width=True)

                with selector_bar[-1]:
                    # add coloring to prompt based on selection
                    prompts_tobe_colored = self.promptBook[self.promptBook['prompt_id'].isin(st.session_state.selected_dict.keys())]['prompt'].unique()
                    colored_prompts = self.text_coloring_add(prompts_tobe_colored, prompts, color_name='✅')

                    selected_prompt = selectbox('Select prompt', colored_prompts, key=f'prompt_{tag}', no_selection_label='---', label_visibility='collapsed', index=prompt_focus_idx)

                    # remove coloring from prompt
                    selected_prompt = self.text_coloring_remove(selected_prompt)
                    # print('selected_prompt: ', selected_prompt)
                    st.session_state.prompt_idx_last_time = prompts.index(selected_prompt) if selected_prompt else 0

            if selected_prompt is None:
                # st.markdown(':orange[Please select a prompt above👆]')
                st.caption('Feel free to **navigate among tags and pages**! Your selection will be saved within one log-in session.')

                with subset_selector[-1]:
                    st.button(':orange[👈 **Please select a prompt**]', disabled=True, use_container_width=True)

            else:
                items = items[items['prompt'] == selected_prompt].reset_index(drop=True)
                prompt_id = items['prompt_id'].unique()[0]
                note = items['note'].unique()[0]

                # add safety check for some prompts
                safety_check = True

                # load unsafe prompts
                unsafe_prompts = json.load(open('./data/unsafe_prompts.json', 'r'))
                for prompt_tag in prompt_tags:
                    if prompt_tag not in unsafe_prompts:
                        unsafe_prompts[prompt_tag] = []
                # # manually add unsafe prompts
                # unsafe_prompts['world knowledge'] = [83]
                # unsafe_prompts['abstract'] = [1, 3]

                if int(prompt_id.item()) in unsafe_prompts[tag]:
                    st.warning('This prompt may contain unsafe content. They might be offensive, depressing, or sexual.')
                    safety_check = st.checkbox('I understand that this prompt may contain unsafe content. Show these images anyway.', key=f'safety_{prompt_id}')

            # print('current state: ', st.session_state.gallery_state)
            #
            # if st.session_state.gallery_state == 'graph':
                if safety_check:
                    self.graph_mode(prompt_id, items)
                with subset_selector[-1]:
                    has_selection = False
                    try:
                        if len(st.session_state.selected_dict.get(prompt_id, [])) > 0:
                            has_selection = True
                    except:
                        pass

                    if has_selection:
                        checkout = st.button('Check out selections ➡️', use_container_width=True, type='primary', on_click=self.switch_to_checkout, args=(tag, selected_prompt))
                    else:
                        st.button(':orange[👇 **Select images below**]', disabled=True, use_container_width=True)
                try:
                    self.sidebar(items, prompt_id, note)
                except:
                    pass

        elif st.session_state.gallery_state == 'check out':
            # select items under the current tag, while model_id in selected_dict keys with corresponding modelVersion_ids
            items = self.promptBook[self.promptBook['tag'] == tag].reset_index(drop=True)
            temp_items = pd.DataFrame()
            for prompt_id, selected_models in st.session_state.selected_dict.items():
                temp_items = pd.concat([temp_items, items[items['modelVersion_id'].isin(selected_models) & (items['prompt_id'] == prompt_id)]], axis=0)
            items = temp_items.reset_index(drop=True)

            self.checkout_mode(tag, items)

    def switch_to_checkout(self, tag, selected_prompt):
        # add focus to session state
        st.session_state.gallery_focus['tag'] = tag
        st.session_state.gallery_focus['prompt'] = selected_prompt

        st.session_state.gallery_state = 'check out'

    def random_gallery_focus(self, tags):
        st.session_state.gallery_focus['tag'] = random.choice(tags)
        # st.session_state.gallery_focus['prompt'] = random.choice(prompts)
        prompts = self.promptBook[self.promptBook['tag'] == st.session_state.gallery_focus['tag']]['prompt'].unique()
        st.session_state.gallery_focus['prompt'] = random.choice(prompts)

    def graph_mode(self, prompt_id, items):
        graph_cols = st.columns([3, 1])

        with graph_cols[0]:
            st.caption(
                'Please **:red[click on and select]** as many images as you like! You will be able to compare them later in ranking stage.')
            graph_space = st.empty()

            with graph_space.container():
                return_value = self.gallery_graph(items)

        with graph_cols[1]:
            if return_value:
                with st.form(key=f'{prompt_id}'):
                    image_url = f"https://modelcofferbucket.s3-accelerate.amazonaws.com/{return_value}.png"

                    st.image(image_url)

                    item = items[items['image_id'] == return_value].reset_index(drop=True).iloc[0]
                    modelVersion_id = item['modelVersion_id']

                    # handle selection
                    # get the latest record in database
                    cursor = GALLERY_CONN.cursor()
                    query = "SELECT * FROM gallery_clicks WHERE username = '{}' AND timestamp = '{}' AND prompt_id = '{}' AND modelVersion_id = {} ORDER BY clicktime DESC LIMIT 1".format(
                        st.session_state.user_id[0], st.session_state.user_id[1], prompt_id, modelVersion_id)
                    cursor.execute(query)
                    record = cursor.fetchone()
                    try:
                        image_status = record['status']
                    except:
                        image_status = None

                    print('image_status: ', image_status)

                    if 'selected_dict' in st.session_state:
                        if item['prompt_id'] not in st.session_state.selected_dict:
                            st.session_state.selected_dict[item['prompt_id']] = []

                        # if 'last_clicked' not in st.session_state or item['image_id'] != st.session_state.last_clicked:
                        #     print('last_clicked not in session state')
                        #     self.image_selection_control(item['tag'], item['prompt'], item['prompt_id'], modelVersion_id, 'select')
                        #     st.toast('Image selected.', icon='👍')
                        #
                        # st.session_state.last_clicked = item['image_id']

                        # if modelVersion_id in st.session_state.selected_dict[item['prompt_id']]:
                        #     checked = True
                        # else:
                        #     checked = False

                    if image_status == 'report':
                        st.warning('You have reported this image')
                        unreport = st.form_submit_button('Withdraw report', use_container_width=True, type='secondary', on_click=self.image_selection_control, args=(item['tag'], item['prompt'], item['prompt_id'], item['modelVersion_id'], 'deselect'))

                    else:
                        if image_status is None:
                            self.image_selection_control(item['tag'], item['prompt'], item['prompt_id'],
                                                         modelVersion_id,
                                                         'select')

                        if image_status == 'select' or image_status == 'reselect' or image_status is None:
                            # deselect = st.button('Deselect', key=f'select_{item["prompt_id"]}_{item["modelVersion_id"]}', use_container_width=True)
                            deselect = st.form_submit_button('Deselect', use_container_width=True, on_click=self.image_selection_control, args=(item['tag'], item['prompt'], item['prompt_id'], item['modelVersion_id'], 'deselect'))


                        elif image_status =='deselect':
                            # select = st.button('Select', key=f'select_{item["prompt_id"]}_{item["modelVersion_id"]}', use_container_width=True, type='primary')
                            reselect = st.form_submit_button('Reselect', use_container_width=True, type='primary', on_click=self.image_selection_control, args=(item['tag'], item['prompt'], item['prompt_id'], item['modelVersion_id'], 'reselect'))

                        report = st.form_submit_button('⚠️Report', use_container_width=True, type='secondary',
                                                       on_click=self.image_selection_control, args=(
                            item['tag'], item['prompt'], item['prompt_id'], item['modelVersion_id'], 'report'),
                                                       help='Report this image if it contains offensive, depressing, or sexual content.')

                        if image_status == 'select' or image_status == 'reselect' or image_status is None:
                            st.info(
                                "Image selected. **Click 'Check out selections ➡️' on top to see all selected images**.")

                    # st.write(item)
                    # infos = ['model_name', 'modelVersion_name', 'model_download_count', 'clip_score', 'mcos_score',
                    #          'nsfw_score']
                    #
                    # infos_df = item[infos]
                    # # rename columns
                    # infos_df = infos_df.rename(index={'model_name': 'Model', 'modelVersion_name': 'Version', 'model_download_count': 'Downloads', 'clip_score': 'Clip Score', 'mcos_score': 'mcos Score', 'nsfw_score': 'NSFW Score'})
                    # st.table(infos_df)

            else:
                st.info('You can click on and select an image.')

    def image_selection_control(self, tag, prompt, prompt_id, modelVersion_id, operation:['select', 'reselect', 'deselect','report']):
        # self.remove_ranking_states(prompt_id)

        if operation == 'select' or operation == 'reselect':
            st.session_state.selected_dict[prompt_id].append(modelVersion_id)
            # add focus to session state
            st.session_state.gallery_focus['tag'] = tag
            st.session_state.gallery_focus['prompt'] = prompt

        elif operation == 'deselect':
            if modelVersion_id in st.session_state.selected_dict[prompt_id]:
                st.session_state.selected_dict[prompt_id].remove(modelVersion_id)
        elif operation == 'report':
            pass

        cursor = GALLERY_CONN.cursor()
        clicktime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        query = "INSERT INTO gallery_clicks (username, timestamp, tag, prompt_id, modelVersion_id, clicktime, status) VALUES ('{}', '{}', '{}', '{}', {}, '{}', '{}')".format(
            st.session_state.user_id[0], st.session_state.user_id[1], tag, prompt_id, modelVersion_id, clicktime,
            operation)

        cursor.execute(query)
        GALLERY_CONN.commit()
        cursor.close()

    def checkout_mode(self, tag, items):
        # st.write(items)
        if len(items) > 0:
            prompt_ids = items['prompt_id'].unique()
            for i in range(len(prompt_ids)):
                prompt_id = prompt_ids[i]
                prompt = items[items['prompt_id'] == prompt_id]['prompt'].unique()[0]
                # default_expand = True if st.session_state.gallery_focus['prompt'] == prompt else False
                if tag == st.session_state.gallery_focus['tag'] and prompt == st.session_state.gallery_focus['prompt']:
                    default_expand = True
                elif tag != st.session_state.gallery_focus['tag'] and i==0:
                    default_expand = True
                else:
                    default_expand = False

                with st.expander(f'**{prompt}**', expanded=default_expand):
                    # st.caption('select info to show')
                    checkout_panel = st.columns([5, 3])
                    with checkout_panel[0]:
                        info = st.multiselect('Show Info',
                                              ['model_name', 'model_id', 'modelVersion_name', 'modelVersion_id',
                                               'total_score', 'model_download_count', 'clip_score', 'mcos_score',
                                               'norm_nsfw'],
                                              label_visibility='collapsed', key=f'info_{prompt_id}', placeholder='Select what info to show')

                    with checkout_panel[-1]:
                        checkout_buttons = st.columns([1, 1, 1])
                        with checkout_buttons[0]:
                            back = st.button('Back to 🖼️', key=f'checkout_back_{prompt_id}', use_container_width=True)
                            if back:
                                st.session_state.gallery_focus['tag'] = tag
                                st.session_state.gallery_focus['prompt'] = prompt
                                print(st.session_state.gallery_focus)
                                st.session_state.gallery_state = 'graph'
                                st.rerun()

                        with checkout_buttons[1]:
                            # init edit state
                            if 'edit_state' not in st.session_state:
                                st.session_state.edit_state = False

                            if not st.session_state.edit_state:
                                edit = st.button('Edit', key=f'checkout_edit_{prompt_id}', use_container_width=True)
                                if edit:
                                    st.session_state.edit_state = True
                                    st.rerun()
                            else:
                                done = st.button('Done', key=f'checkout_done_{prompt_id}', use_container_width=True)
                                if done:
                                    st.session_state.selected_dict[prompt_id] = []
                                    for key in st.session_state:

                                        # update selected_dict with edited selection
                                        keys = key.split('_')
                                        if keys[0] == 'select' and keys[1] == str(prompt_id):
                                            if st.session_state[key]:
                                                st.session_state.selected_dict[prompt_id].append(int(keys[2]))
                                                self.image_selection_control(tag, prompt, prompt_id, int(keys[2]), 'select')    # update database
                                    st.session_state.edit_state = False
                                    st.rerun()

                        with checkout_buttons[-1]:
                            proceed = st.button('Proceed ➡️', key=f'checkout_proceed_{prompt_id}', use_container_width=True,
                                                type='primary', disabled=st.session_state.edit_state)
                            if proceed:
                                self.remove_ranking_states(prompt_id)
                                st.session_state.gallery_focus['tag'] = tag
                                st.session_state.gallery_focus['prompt'] = prompt
                                st.session_state.gallery_state = 'graph'

                                print('selected_dict: ', st.session_state.selected_dict)

                                # # save the user selection to database
                                # cursor = GALLERY_CONN.cursor()
                                # st.session_state.epoch['gallery'] += 1
                                # checkouttime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                                # # for modelVersion_id in st.session_state.selected_dict[prompt_id]:
                                # for key, values in st.session_state.selected_dict.items():
                                #     # print('key: ', key, 'values: ', values)
                                #     key_tag = self.promptBook[self.promptBook['prompt_id'] == key]['tag'].unique()[0]
                                #     for value in values:
                                #         query = "INSERT INTO gallery_selections (username, timestamp, tag, prompt_id, modelVersion_id, checkouttime, epoch) VALUES ('{}', '{}', '{}', '{}', {}, '{}', {})".format(st.session_state.user_id[0], st.session_state.user_id[1], key_tag, key, value, checkouttime, st.session_state.epoch['gallery'])
                                #         print(query)
                                #         cursor.execute(query)
                                # GALLERY_CONN.commit()
                                # cursor.close()

                                # get the largest epoch number of this user and prompt
                                cursor = GALLERY_CONN.cursor()
                                db_table = 'battle_results' if st.session_state.assigned_rank_mode=='Battle' else 'sort_results'
                                query = "SELECT MAX(epoch) FROM {} WHERE username = '{}' AND timestamp = '{}' AND prompt_id = {}".format(db_table, st.session_state.user_id[0], st.session_state.user_id[1], prompt_id)
                                cursor.execute(query)
                                max_epoch = cursor.fetchone()['MAX(epoch)'],
                                # print('max epoch: ', max_epoch, type(max_epoch))
                                cursor.close()

                                try:
                                    st.session_state.epoch['ranking'][prompt_id] = max_epoch[0] + 1
                                except TypeError:
                                    st.session_state.epoch['ranking'][prompt_id] = 1
                                # st.session_state.epoch['summary'][tag] = st.session_state.epoch['summary'].get(tag, 0) + 1
                                # st.session_state.epoch['summary']['overall'] += 1
                                print('epoch: ', st.session_state.epoch)
                                switch_page('ranking')

                    self.gallery_standard(items[items['prompt_id'] == prompt_id].reset_index(drop=True), 4, info, show_checkbox=st.session_state.edit_state)
        else:
            # with st.form(key=f'checkout_{tag}'):
            st.info('No selection under this tag')
            back = st.button('🖼️ Back to gallery and select something you like', key=f'checkout_{tag}', type='primary')
            if back:
                st.session_state.gallery_focus['tag'] = tag
                st.session_state.gallery_focus['prompt'] = None
                st.session_state.gallery_state = 'graph'
                st.rerun()

    def remove_ranking_states(self, prompt_id):
        # for drag sort
        try:
            st.session_state.counter[prompt_id] = 0
            st.session_state.ranking[prompt_id] = {}
            print('remove ranking states')
        except:
            print('no sort ranking states to remove')

        # for battles
        try:
            st.session_state.pointer[prompt_id] = {'left': 0, 'right': 1}
            print('remove battles states')
        except:
            print('no battles states to remove')

        # for page progress
        try:
            st.session_state.progress[prompt_id] = 'ranking'
            print('reset page progress states')
        except:
            print('no page progress states to be reset')

@st.cache_data
def load_hf_dataset(show_NSFW=False):
    # login to huggingface
    login(token=os.environ.get("HF_TOKEN"))

    # load from huggingface
    roster = pd.DataFrame(load_dataset('MAPS-research/GEMRec-Roster', split='train'))
    promptBook = pd.DataFrame(load_dataset('MAPS-research/GEMRec-Metadata', split='train'))
    # images_ds = load_from_disk(os.path.join(os.getcwd(), 'data', 'promptbook'))
    images_ds = None  # set to None for now since we use s3 bucket to store images

    # # process dataset
    # roster = roster[['model_id', 'model_name', 'modelVersion_id', 'modelVersion_name',
    #                                                    'model_download_count']].drop_duplicates().reset_index(drop=True)

    # add 'custom_score_weights' column to promptBook if not exist
    if 'weighted_score_sum' not in promptBook.columns:
        promptBook.loc[:, 'weighted_score_sum'] = 0

    # merge roster and promptbook
    promptBook = promptBook.merge(roster[['model_id', 'model_name', 'modelVersion_id', 'modelVersion_name', 'model_download_count']],
                                                                    on=['model_id', 'modelVersion_id'], how='left')

    # add column to record current row index
    promptBook.loc[:, 'row_idx'] = promptBook.index

    # apply curation filter
    prompt_to_hide = json.load(open('./data/curation.json', 'r'))
    prompt_to_hide = list(itertools.chain.from_iterable(prompt_to_hide.values()))
    print('prompt to hide: ', prompt_to_hide)
    promptBook = promptBook[~promptBook['prompt_id'].isin(prompt_to_hide)].reset_index(drop=True)

    # apply a nsfw filter
    if not show_NSFW:
        promptBook = promptBook[promptBook['norm_nsfw'] <= 0.8].reset_index(drop=True)
        print('nsfw filter applied', len(promptBook))

    # add a column that adds up 'norm_clip', 'norm_mcos', and 'norm_pop'
    score_weights = [1.0, 0.8, 0.2]
    promptBook.loc[:, 'total_score'] = round(promptBook['norm_clip'] * score_weights[0] + promptBook['norm_mcos'] * score_weights[1] + promptBook['norm_pop'] * score_weights[2], 4)

    return roster, promptBook, images_ds

@st.cache_data
def load_tsne_coordinates(items):
    # load tsne coordinates
    tsne_df = pd.read_parquet('./data/feats_tsne.parquet')

    items = items.merge(tsne_df, on=['modelVersion_id', 'prompt_id'], how='left')
    return items


if __name__ == "__main__":
    st.set_page_config(page_title="Model Coffer Gallery", page_icon="🖼️", layout="wide")

    if 'user_id' not in st.session_state:
        st.warning('Please log in first.')
        home_btn = st.button('Go to Home Page')
        if home_btn:
            switch_page("home")
    else:
        GALLERY_CONN = connect_to_db()
        roster, promptBook, images_ds = load_hf_dataset(st.session_state.show_NSFW)

        app = GalleryApp(promptBook=promptBook, images_ds=images_ds)
        app.app()

    with open('./css/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

