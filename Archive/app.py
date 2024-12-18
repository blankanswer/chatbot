import streamlit as st
import numpy as np
import random
import pandas as pd
import glob
from PIL import Image
import datasets
from datasets import load_dataset, Dataset, load_from_disk
from huggingface_hub import login
import os
import requests
from bs4 import BeautifulSoup
import re

import altair as alt
from streamlit_vega_lite import vega_lite_component, altair_component, _component_func

SCORE_NAME_MAPPING = {'clip': 'clip_score', 'rank': 'avg_rank', 'pop': 'model_download_count'}


# hist_data = pd.DataFrame(np.random.normal(42, 10, (200, 1)), columns=["x"])
@st.cache_resource
def altair_histogram(hist_data, sort_by, mini, maxi):
    brushed = alt.selection_interval(encodings=['x'], name="brushed")

    chart = (
        alt.Chart(hist_data)
        .mark_bar(opacity=0.7, cornerRadius=2)
        .encode(alt.X(f"{sort_by}:Q", bin=alt.Bin(maxbins=25)), y="count()")
        # .add_selection(brushed)
        # .properties(width=800, height=300)
    )

    # Create a transparent rectangle for highlighting the range
    highlight = (
        alt.Chart(pd.DataFrame({'x1': [mini], 'x2': [maxi]}))
        .mark_rect(opacity=0.3)
        .encode(x='x1', x2='x2')
        # .properties(width=800, height=300)
    )

    # Layer the chart and the highlight rectangle
    layered_chart = alt.layer(chart, highlight)

    return layered_chart

    # return (
    #     alt.Chart(hist_data)
    #     .mark_bar()
    #     .encode(alt.X(f"{sort_by}:Q", bin=alt.Bin(maxbins=20)), y="count()")
    #     .add_selection(brushed)
    #     .properties(width=600, height=300)
    # )

class GalleryApp:
    def __init__(self, promptBook, images_ds):
        self.promptBook = promptBook
        self.images_ds = images_ds

    def gallery_masonry(self, items, col_num, info):
        cols = st.columns(col_num)
        # # sort items by brisque score
        # items = items.sort_values(by=['brisque'], ascending=True).reset_index(drop=True)
        for idx in range(len(items)):
            with cols[idx % col_num]:
                image = self.images_ds[items.iloc[idx]['row_idx'].item()]['image']
                st.image(image,
                         use_column_width=True,
                )
                # with st.expander('Similarity Info'):
                #     tab1, tab2 = st.tabs(['Most Similar', 'Least Similar'])
                #     with tab1:
                #         st.image(image, use_column_width=True)
                #     with tab2:
                #         st.image(image, use_column_width=True)

                # show checkbox
                self.promptBook.loc[items.iloc[idx]['row_idx'].item(), 'checked'] = st.checkbox(
                    'Select', value=self.promptBook.loc[items.iloc[idx]['row_idx'].item(), 'checked'],
                    key=f'select_{idx}')

                for key in info:
                    st.write(f"**{key}**: {items.iloc[idx][key]}")

    def gallery_standard(self, items, col_num, info):
        rows = len(items) // col_num + 1
        # containers = [st.container() for _ in range(rows * 2)]
        containers = [st.container() for _ in range(rows)]
        for idx in range(0, len(items), col_num):
            # assign one container for each row
            # row_idx = (idx // col_num) * 2
            row_idx = idx // col_num
            with containers[row_idx]:
                cols = st.columns(col_num)
                for j in range(col_num):
                    if idx + j < len(items):
                        with cols[j]:
                            # show image
                            image = self.images_ds[items.iloc[idx + j]['row_idx'].item()]['image']

                            st.image(image, use_column_width=True)

                            # show checkbox
                            self.promptBook.loc[items.iloc[idx + j]['row_idx'].item(), 'checked'] = st.checkbox(
                                'Select', value=self.promptBook.loc[items.iloc[idx + j]['row_idx'].item(), 'checked'],
                                key=f'select_{idx + j}')

                            # st.write(idx+j)
                            # show selected info
                            for key in info:
                                st.write(f"**{key}**: {items.iloc[idx + j][key]}")

                            # st.write(row_idx/2, idx+j, rows)
                            # extra_info = st.checkbox('Extra Info', key=f'extra_info_{idx+j}')
                            # if extra_info:
                            #     with containers[row_idx+1]:
                            #         st.image(image, use_column_width=True)

    def selection_panel(self, items):
        selecters = st.columns([4, 1, 1])

        with selecters[0]:
            types = st.columns([1, 3])
            with types[0]:
                sort_type = st.selectbox('Sort by', ['IDs and Names', 'Scores'])
            with types[1]:
                if sort_type == 'IDs and Names':
                    sort_by = st.selectbox('Sort by',
                                           ['model_name', 'model_id', 'modelVersion_name', 'modelVersion_id'],
                                           label_visibility='hidden')
                elif sort_type == 'Scores':
                    sort_by = st.multiselect('Sort by', ['clip_score', 'avg_rank', 'popularity'],
                                             label_visibility='hidden',
                                             default=['clip_score', 'avg_rank', 'popularity'])
                    # process sort_by to map to the column name

                    if len(sort_by) == 3:
                        sort_by = 'clip+rank+pop'
                    elif len(sort_by) == 2:
                        if 'clip_score' in sort_by and 'avg_rank' in sort_by:
                            sort_by = 'clip+rank'
                        elif 'clip_score' in sort_by and 'popularity' in sort_by:
                            sort_by = 'clip+pop'
                        elif 'avg_rank' in sort_by and 'popularity' in sort_by:
                            sort_by = 'rank+pop'
                    elif len(sort_by) == 1:
                        if 'popularity' in sort_by:
                            sort_by = 'model_download_count'
                        else:
                            sort_by = sort_by[0]
                    print(sort_by)

        with selecters[1]:
            order = st.selectbox('Order', ['Ascending', 'Descending'], index=1 if sort_type == 'Scores' else 0)
            if order == 'Ascending':
                order = True
            else:
                order = False

        items = items.sort_values(by=[sort_by], ascending=order).reset_index(drop=True)

        with selecters[2]:
            filter = st.selectbox('Filter', ['Safe', 'All', 'Unsafe'])
            print('filter', filter)
            # initialize unsafe_modelVersion_ids
            if filter == 'Safe':
                # return checked items
                items = items[items['checked'] == False].reset_index(drop=True)

            elif filter == 'Unsafe':
                # return unchecked items
                items = items[items['checked'] == True].reset_index(drop=True)
                print(items)

        info = st.multiselect('Show Info',
                              ['model_download_count', 'clip_score', 'avg_rank', 'model_name', 'model_id',
                               'modelVersion_name', 'modelVersion_id', 'clip+rank', 'clip+pop', 'rank+pop',
                               'clip+rank+pop'],
                              default=sort_by)

        # add one annotation
        mentioned_scores = []
        for i in info:
            if '+' in i:
                mentioned = i.split('+')
                for m in mentioned:
                    if SCORE_NAME_MAPPING[m] not in mentioned_scores:
                        mentioned_scores.append(SCORE_NAME_MAPPING[m])
        if len(mentioned_scores) > 0:
            st.info(
                f"**Note:** The scores {mentioned_scores} are normalized to [0, 1] for each score type, and then added together. The higher the score, the better the model.")

        col_num = st.slider('Number of columns', min_value=1, max_value=9, value=4, step=1, key='col_num')

        return items, info, col_num

    def selection_panel_2(self, items):

        selecters = st.columns([1, 4])

        # select sort type
        with selecters[0]:
            sort_type = st.selectbox('Sort by', ['Scores', 'IDs and Names'])
            if sort_type == 'Scores':
                sort_by = 'weighted_score_sum'

        # select other options
        with selecters[1]:
            if sort_type == 'IDs and Names':
                sub_selecters = st.columns([3, 1])
                # select sort by
                with sub_selecters[0]:
                    sort_by = st.selectbox('Sort by',
                                           ['model_name', 'model_id', 'modelVersion_name', 'modelVersion_id'],
                                           label_visibility='hidden')

                continue_idx = 1

            else:
                # add custom weights
                sub_selecters = st.columns([1, 1, 1, 1])

                if 'score_weights' not in st.session_state:
                    st.session_state.score_weights = [1.0, 0.8, 0.2]

                with sub_selecters[0]:
                    clip_weight = st.number_input('Clip Score Weight', min_value=-100.0, max_value=100.0, value=st.session_state.score_weights[0], step=0.1, help='the weight for normalized clip score')
                with sub_selecters[1]:
                    rank_weight = st.number_input('Distinctiveness Weight', min_value=-100.0, max_value=100.0, value=st.session_state.score_weights[1], step=0.1, help='the weight for average rank')
                with sub_selecters[2]:
                    pop_weight = st.number_input('Popularity Weight', min_value=-100.0, max_value=100.0, value=st.session_state.score_weights[2], step=0.1, help='the weight for normalized popularity score')

                st.session_state.score_weights = [clip_weight, rank_weight, pop_weight]

                items.loc[:, 'weighted_score_sum'] = round(items['norm_clip'] * clip_weight + items['avg_rank'] * rank_weight + items[
                    'norm_pop'] * pop_weight, 4)

                continue_idx = 3

            # select threshold
            with sub_selecters[continue_idx]:
                dist_threshold = st.number_input('Distinctiveness Threshold', min_value=0.0, max_value=1.0, value=0.84, step=0.01, help='Only show models with distinctiveness score lower than this threshold, set 1.0 to show all images')
                items = items[items['avg_rank'] < dist_threshold].reset_index(drop=True)

        # draw a distribution histogram
        if sort_type == 'Scores':
            try:
                with st.expander('Show score distribution histogram and select score range'):
                    st.write('**Score distribution histogram**')
                    chart_space = st.container()
                    # st.write('Select the range of scores to show')
                    hist_data = pd.DataFrame(items[sort_by])
                    mini = hist_data[sort_by].min().item()
                    mini = mini//0.1 * 0.1
                    maxi = hist_data[sort_by].max().item()
                    maxi = maxi//0.1 * 0.1 + 0.1
                    st.write('**Select the range of scores to show**')
                    r = st.slider('Select the range of scores to show', min_value=mini, max_value=maxi, value=(mini, maxi), step=0.05, label_visibility='collapsed')
                    with chart_space:
                        st.altair_chart(altair_histogram(hist_data, sort_by, r[0], r[1]), use_container_width=True)
                    # event_dict = altair_component(altair_chart=altair_histogram(hist_data, sort_by))
                    # r = event_dict.get(sort_by)
                    if r:
                        items = items[(items[sort_by] >= r[0]) & (items[sort_by] <= r[1])].reset_index(drop=True)
                        # st.write(r)
            except:
                pass

        display_options = st.columns([1, 4])

        with display_options[0]:
            # select order
            order = st.selectbox('Order', ['Ascending', 'Descending'], index=1 if sort_type == 'Scores' else 0)
            if order == 'Ascending':
                order = True
            else:
                order = False

        with display_options[1]:

            # select info to show
            info = st.multiselect('Show Info',
                                  ['model_download_count', 'clip_score', 'avg_rank', 'model_name', 'model_id',
                                   'modelVersion_name', 'modelVersion_id', 'clip+rank', 'clip+pop', 'rank+pop',
                                   'clip+rank+pop', 'weighted_score_sum'],
                                  default=sort_by)

        # apply sorting to dataframe
        items = items.sort_values(by=[sort_by], ascending=order).reset_index(drop=True)

        # select number of columns
        col_num = st.slider('Number of columns', min_value=1, max_value=9, value=4, step=1, key='col_num')

        return items, info, col_num

    def app(self):
        st.title('Model Visualization and Retrieval')
        st.write('This is a gallery of images generated by the models')

        with st.sidebar:
            prompt_tags = self.promptBook['tag'].unique()
            # sort tags by alphabetical order
            prompt_tags = np.sort(prompt_tags)[::-1]

            tag = st.selectbox('Select a tag', prompt_tags)

            items = self.promptBook[self.promptBook['tag'] == tag].reset_index(drop=True)

            original_prompts = np.sort(items['prompt'].unique())[::-1]

            # remove the first four items in the prompt, which are mostly the same
            if tag != 'abstract':
                prompts = [', '.join(x.split(', ')[4:]) for x in original_prompts]
                prompt = st.selectbox('Select prompt', prompts)

                idx = prompts.index(prompt)
                prompt_full = ', '.join(original_prompts[idx].split(', ')[:4]) + ', ' + prompt
            else:
                prompt_full = st.selectbox('Select prompt', original_prompts)

            prompt_id = items[items['prompt'] == prompt_full]['prompt_id'].unique()[0]
            items = items[items['prompt_id'] == prompt_id].reset_index(drop=True)

            # show image metadata
            image_metadatas = ['prompt_id', 'prompt', 'negativePrompt', 'sampler', 'cfgScale', 'size', 'seed']
            for key in image_metadatas:
                label = ' '.join(key.split('_')).capitalize()
                st.write(f"**{label}**")
                if items[key][0] == ' ':
                    st.write('`None`')
                else:
                    st.caption(f"{items[key][0]}")

            # for tag as civitai, add civitai reference
            if tag == 'civitai':
                try:
                    st.write('**Civitai Reference**')
                    res = requests.get(f'https://civitai.com/images/{prompt_id.item()}')
                    # st.write(res.text)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    image_section = soup.find('div', {'class': 'mantine-12rlksp'})
                    image_url = image_section.find('img')['src']
                    st.image(image_url, use_column_width=True)
                except:
                    pass

        # add safety check for some prompts
        safety_check = True
        unsafe_prompts = {}
        # initialize unsafe prompts
        for prompt_tag in prompt_tags:
            unsafe_prompts[prompt_tag] = []
        # manually add unsafe prompts
        unsafe_prompts['civitai'] = [375790, 366222, 295008, 256477]
        unsafe_prompts['people'] = [53]
        unsafe_prompts['art'] = [23]
        unsafe_prompts['abstract'] = [10, 12]
        unsafe_prompts['food'] = [34]

        if int(prompt_id.item()) in unsafe_prompts[tag]:
            st.warning('This prompt may contain unsafe content. They might be offensive, depressing, or sexual.')
            safety_check = st.checkbox('I understand that this prompt may contain unsafe content. Show these images anyway.', key=f'{prompt_id}')

        if safety_check:
            items, info, col_num = self.selection_panel_2(items)
            # self.gallery_standard(items, col_num, info)

            with st.form(key=f'{prompt_id}', clear_on_submit=True):
                # buttons = st.columns([1, 1, 1])
                buttons_space = st.columns([1, 1, 1, 1])
                gallery_space = st.empty()
                # with buttons[0]:
                #     submit = st.form_submit_button('Save selections', on_click=self.save_checked, use_container_width=True, type='primary')
                # with buttons[1]:
                #     submit = st.form_submit_button('Reset current prompt', on_click=self.reset_current_prompt, kwargs={'prompt_id': prompt_id} , use_container_width=True)
                # with buttons[2]:
                #     submit = st.form_submit_button('Reset all selections', on_click=self.reset_all, use_container_width=True)

                with gallery_space.container():
                    self.gallery_standard(items, col_num, info)

                with buttons_space[0]:
                    st.form_submit_button('Confirm and Continue', use_container_width=True, type='primary')

                with buttons_space[1]:
                    st.form_submit_button('Select All', use_container_width=True)

                with buttons_space[2]:
                    st.form_submit_button('Deselect All', use_container_width=True)

                with buttons_space[3]:
                    st.form_submit_button('Refresh', on_click=gallery_space.empty, use_container_width=True)


    def reset_current_prompt(self, prompt_id):
        # reset current prompt
        self.promptBook.loc[self.promptBook['prompt_id'] == prompt_id, 'checked'] = False
        self.save_checked()

    def reset_all(self):
        # reset all
        self.promptBook.loc[:, 'checked'] = False
        self.save_checked()

    def save_checked(self):
        # save checked images to huggingface dataset
        dataset = load_dataset('NYUSHPRP/ModelCofferMetadata', split='train')
        # get checked images
        checked_info = self.promptBook['checked']

        if 'checked' in dataset.column_names:
            dataset = dataset.remove_columns('checked')
        dataset = dataset.add_column('checked', checked_info)

        # print('metadata dataset: ', dataset)
        st.cache_data.clear()
        dataset.push_to_hub('NYUSHPRP/ModelCofferMetadata', split='train')


@st.cache_data
def load_hf_dataset():
    # login to huggingface
    login(token=os.environ.get("HF_TOKEN"))

    # load from huggingface
    roster = pd.DataFrame(load_dataset('NYUSHPRP/ModelCofferRoster', split='train'))
    promptBook = pd.DataFrame(load_dataset('NYUSHPRP/ModelCofferMetadata', split='train'))
    images_ds = load_from_disk(os.path.join(os.getcwd(), '../data', 'promptbook'))

    # process dataset
    roster = roster[['model_id', 'model_name', 'modelVersion_id', 'modelVersion_name',
                                                       'model_download_count']].drop_duplicates().reset_index(drop=True)

    # add 'checked' column to promptBook if not exist
    if 'checked' not in promptBook.columns:
        promptBook.loc[:, 'checked'] = False

    # add 'custom_score_weights' column to promptBook if not exist
    if 'weighted_score_sum' not in promptBook.columns:
        promptBook.loc[:, 'weighted_score_sum'] = 0

    # merge roster and promptbook
    promptBook = promptBook.merge(roster[['model_id', 'model_name', 'modelVersion_id', 'modelVersion_name', 'model_download_count']],
                                                                    on=['model_id', 'modelVersion_id'], how='left')

    # add column to record current row index
    promptBook.loc[:, 'row_idx'] = promptBook.index

    return roster, promptBook, images_ds


if __name__ == '__main__':
    st.set_page_config(layout="wide")

    roster, promptBook, images_ds = load_hf_dataset()

    app = GalleryApp(promptBook=promptBook, images_ds=images_ds)
    app.app()
