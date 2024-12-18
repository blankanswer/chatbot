import json
import os

import datasets
import numpy as np
import pandas as pd
import pymysql.cursors
import streamlit as st

from datetime import datetime
from streamlit_elements import elements, mui, html, dashboard, nivo
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from st_clickable_images import clickable_images

from pages.Gallery import load_hf_dataset
from Home import connect_to_db


class DashboardApp:
    def __init__(self, roster, promptBook, session_finished):
        self.roster = roster
        self.promptBook = promptBook
        self.session_finished = session_finished

        # init modelVersion_standings
        if 'modelVersion_standings' not in st.session_state:
            st.session_state.modelVersion_standings = {}

    def sidebar(self, tags, mode):
        with st.sidebar:
            # tag = st.selectbox('Select a tag', tags, key='tag')
            # st.write('---')
            with st.form('summary_sidebar_form'):
                st.write('## Want a more comprehensive summary?')
                st.write('Jump back to gallery and select more images to rank!')
                back_to_gallery = st.form_submit_button('🖼️ Go to Gallery')
                if back_to_gallery:
                    switch_page('gallery')
                back_to_ranking = st.form_submit_button('🎖️ Go to Ranking')
                if back_to_ranking:
                    switch_page('ranking')

            with st.form('overall_feedback'):
                comment = st.text_area('Please leave your comments here.', key='comment')
                submit_feedback = st.form_submit_button('Submit Feedback')
                if submit_feedback:
                    commenttime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    curser = RANKING_CONN.cursor()
                    # parse the comment to at most 300 to avoid SQL injection
                    for i in range(0, len(comment), 300):
                        curser.execute(f"INSERT INTO comments (username, timestamp, comment, commenttime) VALUES ('{st.session_state.user_id[0]}', '{st.session_state.user_id[1]}', '{comment[i:i+300]}', '{commenttime}')")
                    RANKING_CONN.commit()
                    curser.close()

                    st.sidebar.info('🙏 **Thanks for your feedback! We will take it into consideration in our future work.**')

    def leaderboard(self, tag, db_table):
        tag = '%' if tag == 'overview' else tag

        # print('tag', tag)

        # get the ranking results of the current user with the lastest epoch
        curser = RANKING_CONN.cursor()
        # curser.execute(f"SELECT * FROM {db_table} WHERE username = '{st.session_state.user_id[0]}' AND timestamp = '{st.session_state.user_id[1]}' AND tag LIKE '{tag}'")
        # curser.execute(f"SELECT * FROM {db_table} WHERE username = '{st.session_state.user_id[0]}' AND timestamp = '{st.session_state.user_id[1]}' AND tag LIKE '{tag}' ORDER BY epoch DESC LIMIT 1")
        curser.execute(
            f"SELECT * FROM {db_table}\
            WHERE username = '{st.session_state.user_id[0]}'\
            AND timestamp = '{st.session_state.user_id[1]}'\
            AND tag LIKE '{tag}'\
            AND epoch =\
                (SELECT MAX(epoch) FROM {db_table}\
                WHERE username = '{st.session_state.user_id[0]}'\
                AND timestamp = '{st.session_state.user_id[1]}'\
                AND tag LIKE '{tag}')")


        results = curser.fetchall()
        curser.close()

        # print('results', results, len(results))

        if tag not in st.session_state.modelVersion_standings:
            st.session_state.modelVersion_standings[tag] = self.score_calculator(results, db_table)

            # sort the modelVersion_standings by value into a list of tuples in descending order
            st.session_state.modelVersion_standings[tag] = sorted(st.session_state.modelVersion_standings[tag].items(), key=lambda x: x[1], reverse=True)
        print(st.session_state.modelVersion_standings[tag])
        example_prompts = []
        # get example images
        for key, value in st.session_state.selected_dict.items():
            for model in st.session_state.modelVersion_standings[tag]:
                if model[0] in value:
                    example_prompts.append(key)

        self.podium_expander(tag, n=len(st.session_state.modelVersion_standings[tag]), summary_mode='display', example_prompts=example_prompts)

        st.write('---')
        st.write('**Detailed information of all selected models**')
        detailed_info = pd.merge(pd.DataFrame(st.session_state.modelVersion_standings[tag], columns=['modelVersion_id', 'ranking_score']), self.roster, on='modelVersion_id')

        detailed_info = detailed_info[['model_name', 'modelVersion_name', 'model_download_count', 'tag', 'baseModel']]

        st.data_editor(detailed_info, hide_index=False, disabled=True)
        st.caption('You can click the header to sort the table by that column.')

    def podium_expander(self, tag, example_prompts, n=3, summary_mode: ['display', 'edit'] = 'display'):
        self.save_summary(tag)

        for i in range(n):
            modelVersion_id = st.session_state.modelVersion_standings[tag][i][0]
            winning_times = st.session_state.modelVersion_standings[tag][i][1]

            model_id, model_name, modelVersion_name, url = self.roster[self.roster['modelVersion_id'] == modelVersion_id][['model_id', 'model_name', 'modelVersion_name', 'modelVersion_url']].values[0]

            icon = '🥇'if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else '🎈'
            podium_display = st.columns([1, 14], gap='medium')
            with podium_display[0]:
                # st.title(f'{icon}')
                st.write(f'# {icon}')
                # if summary_mode == 'display':
                #     st.title(f'{icon}')
                # elif summary_mode == 'edit':
                settop = st.button('🔝', key=f'settop_{modelVersion_id}', help='Set this model to the top', disabled=i == 0, on_click=self.switch_order, args=(tag, i, 0))
                moveup = st.button('⬆', key=f'moveup_{modelVersion_id}', help='Move this model up', disabled=i == 0, on_click=self.switch_order, args=(tag, i, i - 1))
                movedown = st.button('⬇', key=f'movedown_{modelVersion_id}', help='Move this model down', disabled=i == n - 1, on_click=self.switch_order, args=(tag, i, i + 1))
            with podium_display[1]:
                title_display = st.columns([4, 1, 1])
                with title_display[0]:
                    st.write(f'##### {model_name}, {modelVersion_name}')
                    # st.write(f'Ranking Score: {winning_times}')
                with title_display[1]:
                    st.link_button('Download', url, use_container_width=True)
                with title_display[2]:
                    st.link_button('Civitai', f'https://civitai.com/models/{model_id}?modelVersionId={modelVersion_id}', use_container_width=True, type='primary')
                # st.write(f'[Civitai Page](https://civitai.com/models/{model_id}?modelVersionId={modelVersion_id}), [Model Download Link]({url}), Ranking Score: {winning_times}')
                # with st.expander(f'**{icon} {model_name}, [{modelVersion_name}](https://civitai.com/models/{model_id}?modelVersionId={modelVersion_id})**, Ranking Score: {winning_times}'):

                image_display = st.toggle('Show all images', key=f'image_display_{modelVersion_id}')
                if not image_display:
                    example_images = self.promptBook[self.promptBook['prompt_id'].isin(example_prompts) & (self.promptBook['modelVersion_id']==modelVersion_id)]['image_id'].values
                    example_images = [f"https://modelcofferbucket.s3-accelerate.amazonaws.com/{image}.png" for image in example_images]
                    clickable_images(
                        example_images,
                        img_style={"margin": "5px", "height": "120px"},
                    )

                else:
                # with st.expander(f'Show Images'):
                    images = self.promptBook[self.promptBook['modelVersion_id'] == modelVersion_id]['image_id'].values

                    # safety_check = st.toggle('Include potentially unsafe or offensive images', value=False, key=modelVersion_id)
                    # unsafe_prompts = json.load(open('data/unsafe_prompts.json', 'r'))
                    # # merge dict values into one list
                    # unsafe_prompts = [item for sublist in unsafe_prompts.values() for item in sublist]
                    # unsafe_images = self.promptBook[self.promptBook['prompt_id'].isin(unsafe_prompts)]['image_id'].values
                    #
                    # if not safety_check:
                    #     # exclude unsafe prompts from images
                    #     images = [image for image in images if image not in unsafe_images]

                    images = [f"https://modelcofferbucket.s3-accelerate.amazonaws.com/{image}.png" for image in images]
                    clickable_images(
                        images,
                        img_style={"margin": "5px", "height": "120px"}
                    )
                    st.write('🐌 It may take a while to load all images. Please be patient, and **NEVER USE THE REFRESH BUTTON ON YOUR BROWSER**.')

                    # # st.write(f'### Images generated with {icon} {model_name}, {modelVersion_name}')
                    # col_num = 4
                    # image_cols = st.columns(col_num)
                    #
                    # for j in range(len(images)):
                    #     with image_cols[j % col_num]:
                    #         image = f"https://modelcofferbucket.s3-accelerate.amazonaws.com/{images[j]}.png"
                    #         st.image(image, use_column_width=True)
                    #
            if i != n - 1:
                st.write('---')

    def save_summary(self, tag):
        # get the lastest summary_results epoch of the current user
        tag_name = 'overview' if tag == '%' else tag
        curser = RANKING_CONN.cursor()
        curser.execute(f"SELECT epoch FROM summary_results WHERE username = '{st.session_state.user_id[0]}' AND timestamp = '{st.session_state.user_id[1]}' AND tag = '{tag_name}' ORDER BY epoch DESC LIMIT 1")
        latest_epoch = curser.fetchone()
        curser.close()
        # print('latest_epoch',latest_epoch)
        if latest_epoch is None or latest_epoch['epoch'] < st.session_state.epoch['summary'][tag_name]:
            # save the current ranking results to the database
            summarytime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            curser = RANKING_CONN.cursor()
            for i in range(len(st.session_state.modelVersion_standings[tag])):
                curser.execute(f"INSERT INTO summary_results (username, timestamp, tag, modelVersion_id, position, ranking_score, summarytime, epoch, customized) VALUES ('{st.session_state.user_id[0]}', '{st.session_state.user_id[1]}', '{tag_name}', '{st.session_state.modelVersion_standings[tag][i][0]}', {i+1}, {st.session_state.modelVersion_standings[tag][i][1]}, '{summarytime}', {st.session_state.epoch['summary'][tag_name]}, 0)")
            RANKING_CONN.commit()
            curser.close()

    def switch_order(self, tag, current, target):
        # insert the current before the target
        st.session_state.modelVersion_standings[tag].insert(target, st.session_state.modelVersion_standings[tag].pop(current))
        tag_name = 'overview' if tag == '%' else tag
        summarytime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        curser = RANKING_CONN.cursor()
        # clear the current user's ranking results
        curser.execute(f"DELETE FROM summary_results WHERE username = '{st.session_state.user_id[0]}' AND timestamp = '{st.session_state.user_id[1]}' AND tag = '{tag_name}' AND epoch = {st.session_state.epoch['summary'][tag_name]}")
        for i in range(len(st.session_state.modelVersion_standings[tag])):
            curser.execute(f"INSERT INTO summary_results (username, timestamp, tag, modelVersion_id, position, ranking_score, summarytime, epoch, customized) VALUES ('{st.session_state.user_id[0]}', '{st.session_state.user_id[1]}', '{tag_name}', '{st.session_state.modelVersion_standings[tag][i][0]}', {i+1}, {st.session_state.modelVersion_standings[tag][i][1]}, '{summarytime}', {st.session_state.epoch['summary'][tag_name]}, 1)")
        RANKING_CONN.commit()
        curser.close()

    def score_calculator(self, results, db_table):
        modelVersion_standings = {}
        if db_table == 'battle_results':
            # sort results by battle time
            results = sorted(results, key=lambda x: x['battletime'])

            for record in results:
                modelVersion_standings[record['winner']] = modelVersion_standings.get(record['winner'], 0) + 1
                # add the loser who never wins
                if record['loser'] not in modelVersion_standings:
                    modelVersion_standings[record['loser']] = 0

                # add the winning time of the loser to the winner
                modelVersion_standings[record['winner']] += modelVersion_standings[record['loser']]

        elif db_table == 'sort_results':
            pts_map = {'position1': 5, 'position2': 3, 'position3': 1, 'position4': 0}
            for record in results:
                for i in range(1, 5):
                    modelVersion_standings[record[f'position{i}']] = modelVersion_standings.get(record[f'position{i}'], 0) + pts_map[f'position{i}']

        return modelVersion_standings

    def app(self):
        st.write('### Your Preferred Models')

        # mode = st.sidebar.radio('Ranking mode', ['Drag and Sort', 'Battle'], horizontal=True, index=1)
        mode = st.session_state.assigned_rank_mode
        # get tags from database of the current user
        db_table = 'sort_results' if mode == 'Drag and Sort' else 'battle_results'

        tags = []
        curser = RANKING_CONN.cursor()
        curser.execute(
            f"SELECT DISTINCT tag FROM {db_table} WHERE username = '{st.session_state.user_id[0]}' AND timestamp = '{st.session_state.user_id[1]}'")
        for row in curser.fetchall():
            tags.append(row['tag'])
        curser.close()

        if len(tags) == 0:
            st.info(f'No rankings are finished with {mode} mode yet.')

        else:
            # tags = tags[1:2] if len(tags) == 2 else tags
            tag = ['overview'] + tags if len(tags) > 1 else tags
            tag = st.radio('Select a tag', tags, index=0, horizontal=True, label_visibility='collapsed')
            self.sidebar(tags, mode)
            self.leaderboard(tag, db_table)


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    if 'user_id' not in st.session_state:
        st.warning('Please log in first.')
        home_btn = st.button('Go to Home Page')
        if home_btn:
            switch_page("home")

    elif 'progress' not in st.session_state:
        st.info('You have not checked any image yet. Please go back to the gallery page and check some images.')
        gallery_btn = st.button('🖼️ Go to Gallery')
        if gallery_btn:
            switch_page('gallery')

    else:
        session_finished = []

        for key, value in st.session_state.progress.items():
            if value == 'finished':
                session_finished.append(key)

        if len(session_finished) == 0:
            st.info('A dashboard showing your preferred models will appear after you finish any ranking session.')
            ranking_btn = st.button('🎖️ Go to Ranking')
            if ranking_btn:
                switch_page('ranking')
            gallery_btn = st.button('🖼️ Go to Gallery')
            if gallery_btn:
                switch_page('gallery')

        else:
            roster, promptBook, images_ds = load_hf_dataset(st.session_state.show_NSFW)
            RANKING_CONN = connect_to_db()
            app = DashboardApp(roster, promptBook, session_finished)
            app.app()

    with open('./css/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


