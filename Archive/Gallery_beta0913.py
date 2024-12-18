import os
import requests

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset, load_from_disk
from huggingface_hub import login
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_extras.switch_page_button import switch_page
from sklearn.svm import LinearSVC

SCORE_NAME_MAPPING = {'clip': 'clip_score', 'rank': 'msq_score', 'pop': 'model_download_count'}


class GalleryApp:
    def __init__(self, promptBook, images_ds):
        self.promptBook = promptBook
        self.images_ds = images_ds

    def gallery_standard(self, items, col_num, info):
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
            # if items.loc[idx, 'modelVersion_id'] in st.session_state.selected_dict.get(items.loc[idx, 'prompt_id'], 0):
            #     opacity = 0.2
            # else:
            #     opacity = 1.0

            nodes.append(Node(id=items.loc[idx, 'image_id'],
                              # label=str(items.loc[idx, 'model_name']),
                              title=f"model name: {items.loc[idx, 'model_name']}\nmodelVersion name: {items.loc[idx, 'modelVersion_name']}\nclip score: {items.loc[idx, 'clip_score']}\nmcos score: {items.loc[idx, 'mcos_score']}\npopularity: {items.loc[idx, 'model_download_count']}",
                              size=20,
                              shape='image',
                              image=f"https://modelcofferbucket.s3-accelerate.amazonaws.com/{items.loc[idx, 'image_id']}.png",
                              x=items.loc[idx, 'x'].item(),
                              y=items.loc[idx, 'y'].item(),
                              # fixed=True,
                              color={'background': '#E0E0E1', 'border': '#ffffff', 'highlight': {'border': '#F04542'}},
                              # opacity=opacity,
                              shadow={'enabled': True, 'color': 'rgba(0,0,0,0.4)', 'size': 10, 'x': 1, 'y': 1},
                              borderWidth=2,
                              shapeProperties={'useBorderWithImage': True},
                              )
                         )

        config = Config(width='100%',
                        height='600',
                        directed=True,
                        physics=False,
                        hierarchical=False,
                        interaction={'navigationButtons': True, 'dragNodes': False, 'multiselect': False},
                        # **kwargs
                        )

        return agraph(nodes=nodes,
                      edges=edges,
                      config=config,
                      )

    def selection_panel(self, items):
        # temperal function

        selecters = st.columns([1, 4])

        if 'score_weights' not in st.session_state:
            st.session_state.score_weights = [1.0, 0.8, 0.2, 0.8]

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
                                           ['model_name', 'model_id', 'modelVersion_name', 'modelVersion_id', 'norm_nsfw'],
                                           label_visibility='hidden')

                continue_idx = 1

            else:
                # add custom weights
                sub_selecters = st.columns([1, 1, 1, 1])

                with sub_selecters[0]:
                    clip_weight = st.number_input('Clip Score Weight', min_value=-100.0, max_value=100.0, value=1.0, step=0.1, help='the weight for normalized clip score')
                with sub_selecters[1]:
                    mcos_weight = st.number_input('Dissimilarity Weight', min_value=-100.0, max_value=100.0, value=0.8, step=0.1, help='the weight for m(eam) s(imilarity) q(antile) score for measuring distinctiveness')
                with sub_selecters[2]:
                    pop_weight = st.number_input('Popularity Weight', min_value=-100.0, max_value=100.0, value=0.2, step=0.1, help='the weight for normalized popularity score')

                items.loc[:, 'weighted_score_sum'] = round(items[f'norm_clip'] * clip_weight + items[f'norm_mcos'] * mcos_weight + items[
                    'norm_pop'] * pop_weight, 4)

                continue_idx = 3

                # save latest weights
                st.session_state.score_weights[0] = round(clip_weight, 2)
                st.session_state.score_weights[1] = round(mcos_weight, 2)
                st.session_state.score_weights[2] = round(pop_weight, 2)

            # select threshold
            with sub_selecters[continue_idx]:
                nsfw_threshold = st.number_input('NSFW Score Threshold', min_value=0.0, max_value=1.0, value=0.8, step=0.01, help='Only show models with nsfw score lower than this threshold, set 1.0 to show all images')
                items = items[items['norm_nsfw'] <= nsfw_threshold].reset_index(drop=True)

            # save latest threshold
            st.session_state.score_weights[3] = nsfw_threshold

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
                                  ['model_name', 'model_id', 'modelVersion_name', 'modelVersion_id',
                                   'weighted_score_sum', 'model_download_count', 'clip_score', 'mcos_score',
                                   'nsfw_score', 'norm_nsfw'],
                                  default=sort_by)

        # apply sorting to dataframe
        items = items.sort_values(by=[sort_by], ascending=order).reset_index(drop=True)

        # select number of columns
        col_num = st.slider('Number of columns', min_value=1, max_value=9, value=4, step=1, key='col_num')

        return items, info, col_num

    def sidebar(self):
        with st.sidebar:
            prompt_tags = self.promptBook['tag'].unique()
            # sort tags by alphabetical order
            prompt_tags = np.sort(prompt_tags)[::1]

            tag = st.selectbox('Select a tag', prompt_tags, index=5)

            items = self.promptBook[self.promptBook['tag'] == tag].reset_index(drop=True)

            prompts = np.sort(items['prompt'].unique())[::1]

            selected_prompt = st.selectbox('Select prompt', prompts, index=3)

            mode = st.radio('Select a mode', ['Gallery', 'Graph'], horizontal=True, index=1)

            items = items[items['prompt'] == selected_prompt].reset_index(drop=True)
            prompt_id = items['prompt_id'].unique()[0]
            note = items['note'].unique()[0]

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

        return prompt_tags, tag, prompt_id, items, mode

    def app(self):
        st.title('Model Visualization and Retrieval')
        st.write('This is a gallery of images generated by the models')

        prompt_tags, tag, prompt_id, items, mode = self.sidebar()
        # items, info, col_num = self.selection_panel(items)

        # subset = st.radio('Select a subset', ['All', 'Selected Only'], index=0, horizontal=True)
        # try:
        #     if subset == 'Selected Only':
        #         items = items[items['modelVersion_id'].isin(st.session_state.selected_dict[prompt_id])].reset_index(drop=True)
        # except:
        #     pass

        # add safety check for some prompts
        safety_check = True
        unsafe_prompts = {}
        # initialize unsafe prompts
        for prompt_tag in prompt_tags:
            unsafe_prompts[prompt_tag] = []
        # manually add unsafe prompts
        unsafe_prompts['world knowledge'] = [83]
        unsafe_prompts['abstract'] = [1, 3]

        if int(prompt_id.item()) in unsafe_prompts[tag]:
            st.warning('This prompt may contain unsafe content. They might be offensive, depressing, or sexual.')
            safety_check = st.checkbox('I understand that this prompt may contain unsafe content. Show these images anyway.', key=f'safety_{prompt_id}')

        if safety_check:
            if mode == 'Gallery':
                self.gallery_mode(prompt_id, items)
            elif mode == 'Graph':
                self.graph_mode(prompt_id, items)


    def graph_mode(self, prompt_id, items):
        graph_cols = st.columns([3, 1])
        prompt = st.chat_input(f"Selected model version ids: {str(st.session_state.selected_dict.get(prompt_id, []))}",
                               disabled=False, key=f'{prompt_id}')
        if prompt:
            switch_page("ranking")

        with graph_cols[0]:
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
                    if 'selected_dict' in st.session_state:
                        if item['prompt_id'] not in st.session_state.selected_dict:
                            st.session_state.selected_dict[item['prompt_id']] = []

                        if modelVersion_id in st.session_state.selected_dict[item['prompt_id']]:
                            checked = True
                        else:
                            checked = False

                    if checked:
                        # deselect = st.button('Deselect', key=f'select_{item["prompt_id"]}_{item["modelVersion_id"]}', use_container_width=True)
                        deselect = st.form_submit_button('Deselect', use_container_width=True)
                        if deselect:
                            st.session_state.selected_dict[item['prompt_id']].remove(item['modelVersion_id'])
                            self.remove_ranking_states(item['prompt_id'])
                            st.experimental_rerun()

                    else:
                        # select = st.button('Select', key=f'select_{item["prompt_id"]}_{item["modelVersion_id"]}', use_container_width=True, type='primary')
                        select = st.form_submit_button('Select', use_container_width=True, type='primary')
                        if select:
                            st.session_state.selected_dict[item['prompt_id']].append(item['modelVersion_id'])
                            self.remove_ranking_states(item['prompt_id'])
                            st.experimental_rerun()

                    # st.write(item)
                    infos = ['model_name', 'modelVersion_name', 'model_download_count', 'clip_score', 'mcos_score',
                             'nsfw_score']

                    infos_df = item[infos]
                    # rename columns
                    infos_df = infos_df.rename(index={'model_name': 'Model', 'modelVersion_name': 'Version', 'model_download_count': 'Downloads', 'clip_score': 'Clip Score', 'mcos_score': 'mcos Score', 'nsfw_score': 'NSFW Score'})
                    st.table(infos_df)

                    # for info in infos:
                    #     st.write(f"**{info}**:")
                    #     st.write(item[info])

            else:
                st.info('Please click on an image to show')


    def gallery_mode(self, prompt_id, items):
        items, info, col_num = self.selection_panel(items)

        if 'selected_dict' in st.session_state:
            # st.write('checked: ', str(st.session_state.selected_dict.get(prompt_id, [])))
            dynamic_weight_options = ['Grid Search', 'SVM', 'Greedy']
            dynamic_weight_panel = st.columns(len(dynamic_weight_options))

            if len(st.session_state.selected_dict.get(prompt_id, [])) > 0:
                btn_disable = False
            else:
                btn_disable = True

            for i in range(len(dynamic_weight_options)):
                method = dynamic_weight_options[i]
                with dynamic_weight_panel[i]:
                    btn = st.button(method, use_container_width=True, disabled=btn_disable, on_click=self.dynamic_weight, args=(prompt_id, items, method))

        prompt = st.chat_input(f"Selected model version ids: {str(st.session_state.selected_dict.get(prompt_id, []))}", disabled=False, key=f'{prompt_id}')
        if prompt:
            switch_page("ranking")

        with st.form(key=f'{prompt_id}'):
            # buttons = st.columns([1, 1, 1])
            buttons_space = st.columns([1, 1, 1, 1])
            gallery_space = st.empty()

            with buttons_space[0]:
                continue_btn = st.form_submit_button('Confirm Selection', use_container_width=True, type='primary')
                if continue_btn:
                    self.submit_actions('Continue', prompt_id)

            with buttons_space[1]:
                select_btn = st.form_submit_button('Select All', use_container_width=True)
                if select_btn:
                    self.submit_actions('Select', prompt_id)

            with buttons_space[2]:
                deselect_btn = st.form_submit_button('Deselect All', use_container_width=True)
                if deselect_btn:
                    self.submit_actions('Deselect', prompt_id)

            with buttons_space[3]:
                refresh_btn = st.form_submit_button('Refresh', on_click=gallery_space.empty, use_container_width=True)

            with gallery_space.container():
                with st.spinner('Loading images...'):
                    self.gallery_standard(items, col_num, info)

                st.info("Don't forget to scroll back to top and click the 'Confirm Selection' button to save your selection!!!")



    def submit_actions(self, status, prompt_id):
        # remove counter from session state
        # st.session_state.pop('counter', None)
        self.remove_ranking_states('prompt_id')
        if status == 'Select':
            modelVersions = self.promptBook[self.promptBook['prompt_id'] == prompt_id]['modelVersion_id'].unique()
            st.session_state.selected_dict[prompt_id] = modelVersions.tolist()
            print(st.session_state.selected_dict, 'select')
            st.experimental_rerun()
        elif status == 'Deselect':
            st.session_state.selected_dict[prompt_id] = []
            print(st.session_state.selected_dict, 'deselect')
            st.experimental_rerun()
            # self.promptBook.loc[self.promptBook['prompt_id'] == prompt_id, 'checked'] = False
        elif status == 'Continue':
            st.session_state.selected_dict[prompt_id] = []
            for key in st.session_state:
                keys = key.split('_')
                if keys[0] == 'select' and keys[1] == str(prompt_id):
                    if st.session_state[key]:
                        st.session_state.selected_dict[prompt_id].append(int(keys[2]))
            # switch_page("ranking")
            print(st.session_state.selected_dict, 'continue')
            st.experimental_rerun()

    def dynamic_weight(self, prompt_id, items, method='Grid Search'):
        selected = items[
            items['modelVersion_id'].isin(st.session_state.selected_dict[prompt_id])].reset_index(drop=True)
        optimal_weight = [0, 0, 0]

        if method == 'Grid Search':
            # grid search method
            top_ranking = len(items) * len(selected)

            for clip_weight in np.arange(-1, 1, 0.1):
                for mcos_weight in np.arange(-1, 1, 0.1):
                    for pop_weight in np.arange(-1, 1, 0.1):

                        weight_all = clip_weight*items[f'norm_clip'] + mcos_weight*items[f'norm_mcos'] + pop_weight*items['norm_pop']
                        weight_all_sorted = weight_all.sort_values(ascending=False).reset_index(drop=True)
                        # print('weight_all_sorted:', weight_all_sorted)
                        weight_selected = clip_weight*selected[f'norm_clip'] + mcos_weight*selected[f'norm_mcos'] + pop_weight*selected['norm_pop']

                        # get the index of values of weight_selected in weight_all_sorted
                        rankings = []
                        for weight in weight_selected:
                            rankings.append(weight_all_sorted.index[weight_all_sorted == weight].tolist()[0])
                        if sum(rankings) <= top_ranking:
                            top_ranking = sum(rankings)
                            print('current top ranking:', top_ranking, rankings)
                            optimal_weight = [clip_weight, mcos_weight, pop_weight]
            print('optimal weight:', optimal_weight)

        elif method == 'SVM':
            # svm method
            print('start svm method')
            # get residual dataframe that contains models not selected
            residual = items[~items['modelVersion_id'].isin(selected['modelVersion_id'])].reset_index(drop=True)
            residual = residual[['norm_clip_crop', 'norm_mcos_crop', 'norm_pop']]
            residual = residual.to_numpy()
            selected = selected[['norm_clip_crop', 'norm_mcos_crop', 'norm_pop']]
            selected = selected.to_numpy()

            y = np.concatenate((np.full((len(selected), 1), -1), np.full((len(residual), 1), 1)), axis=0).ravel()
            X = np.concatenate((selected, residual), axis=0)

            # fit svm model, and get parameters for the hyperplane
            clf = LinearSVC(random_state=0, C=1.0, fit_intercept=False, dual='auto')
            clf.fit(X, y)
            optimal_weight = clf.coef_[0].tolist()
            print('optimal weight:', optimal_weight)
            pass

        elif method == 'Greedy':
            for idx in selected.index:
                # find which score is the highest, clip, mcos, or pop
                clip_score = selected.loc[idx, 'norm_clip_crop']
                mcos_score = selected.loc[idx, 'norm_mcos_crop']
                pop_score = selected.loc[idx, 'norm_pop']
                if clip_score >= mcos_score and clip_score >= pop_score:
                    optimal_weight[0] += 1
                elif mcos_score >= clip_score and mcos_score >= pop_score:
                    optimal_weight[1] += 1
                elif pop_score >= clip_score and pop_score >= mcos_score:
                    optimal_weight[2] += 1

            # normalize optimal_weight
            optimal_weight = [round(weight/len(selected), 2) for weight in optimal_weight]
            print('optimal weight:', optimal_weight)
            print('optimal weight:', optimal_weight)

        st.session_state.score_weights[0: 3] = optimal_weight


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


@st.cache_data
def load_hf_dataset():
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

    # apply a nsfw filter
    promptBook = promptBook[promptBook['nsfw_score'] <= 0.84].reset_index(drop=True)

    # add a column that adds up 'norm_clip', 'norm_mcos', and 'norm_pop'
    score_weights = [1.0, 0.8, 0.2]
    promptBook.loc[:, 'total_score'] = round(promptBook['norm_clip'] * score_weights[0] + promptBook['norm_mcos'] * score_weights[1] + promptBook['norm_pop'] * score_weights[2], 4)

    return roster, promptBook, images_ds

@st.cache_data
def load_tsne_coordinates(items):
    # load tsne coordinates
    tsne_df = pd.read_parquet('./data/feats_tsne.parquet')

    # print(tsne_df['modelVersion_id'].dtype)

    print('before merge:', items)
    items = items.merge(tsne_df, on=['modelVersion_id', 'prompt_id'], how='left')
    print('after merge:', items)
    return items


if __name__ == "__main__":
    st.set_page_config(page_title="Model Coffer Gallery", page_icon="🖼️", layout="wide")

    if 'user_id' not in st.session_state:
        st.warning('Please log in first.')
        home_btn = st.button('Go to Home Page')
        if home_btn:
            switch_page("home")
    else:
        # st.write('You have already logged in as ' + st.session_state.user_id[0])
        roster, promptBook, images_ds = load_hf_dataset()
        # print(promptBook.columns)

        # initialize selected_dict
        if 'selected_dict' not in st.session_state:
            st.session_state['selected_dict'] = {}

        app = GalleryApp(promptBook=promptBook, images_ds=images_ds)
        app.app()

        # components.html(
        #     """
        #     <script>
        #       var iframe = window.parent.document.querySelector('[title="streamlit_agraph.agraph"]');
        #       console.log(iframe);
        #       var targetElement = iframe.contentDocument.querySelector('div.vis-network div.vis-navigation div.vis-button.vis-zoomExtends');
        #       console.log(targetElement);
        #       targetElement.style.background-image = "url(https://www.flaticon.com/free-icon-font/menu-burger_3917215?related_id=3917215#)";
        #     </script>
        #     """,
        #     # unsafe_allow_html=True,
        # )
