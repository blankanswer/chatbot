import os

import datasets
import numpy as np
import pandas as pd
import pymysql.cursors
import streamlit as st

from datetime import datetime
from streamlit_elements import elements, mui, html, dashboard, nivo
from streamlit_extras.switch_page_button import switch_page

from pages.Gallery import load_hf_dataset
from Home import connect_to_db


class RankingApp:
    def __init__(self, promptBook, images_endpoint, batch_size=4):
        self.promptBook = promptBook
        self.images_endpoint = images_endpoint
        self.batch_size = batch_size
        # self.batch_num = len(self.promptBook) // self.batch_size
        # self.batch_num += 1 if len(self.promptBook) % self.batch_size != 0 else 0

        if 'counter' not in st.session_state:
            st.session_state.counter = {}

    def sidebar(self, selected_prompt, items):
        with st.sidebar:
            # st.title('Personal Image Ranking')
            # st.write('Here you can test out your selected images with any prompt you like. ')
            # prompt_tags = self.promptBook['tag'].unique()
            # prompt_tags = np.sort(prompt_tags).tolist()
            #
            # print(st.session_state.gallery_focus)
            # tag_idx = prompt_tags.index(st.session_state.gallery_focus['tag']) if st.session_state.gallery_focus['tag'] in prompt_tags else 0
            # print(tag_idx)
            #
            # tag = st.selectbox('Select a prompt tag', prompt_tags, index=tag_idx)
            # items = self.promptBook[self.promptBook['tag'] == tag].reset_index(drop=True)
            # prompts = np.sort(items['prompt'].unique())[::-1].tolist()
            #
            # prompt_idx = prompts.index(st.session_state.gallery_focus['prompt']) if st.session_state.gallery_focus['prompt'] in prompts else 0
            # print(prompt_idx)
            #
            # selected_prompt = st.selectbox('Select a prompt', prompts, index=prompt_idx)
            #
            # # mode = st.radio('Select a mode', ['Drag and Sort', 'Battle'], index=1)
            # mode = st.session_state.assigned_rank_mode
            #
            # items = items[items['prompt'] == selected_prompt].reset_index(drop=True)
            # prompt_id = items['prompt_id'].unique()[0]

            with st.form(key='prompt_form'):
                # input image metadata
                prompt = st.text_area('Prompt', selected_prompt, height=150, key='prompt', disabled=True)
                negative_prompt = st.text_area('Negative Prompt', items['negativePrompt'].unique()[0], height=150, key='negative_prompt', disabled=True)
                st.form_submit_button('Generate Images [Coming Soon]', type='primary', use_container_width=True, disabled=True)

        return None

    def draggable_images(self, items, prompt_id, layout='portrait'):
        # init ranking by the order of items

        if 'ranking' not in st.session_state:
            st.session_state.ranking = {}

        if prompt_id not in st.session_state.ranking:
            st.session_state.ranking[prompt_id] = {}

        if st.session_state.counter[prompt_id] not in st.session_state.ranking[prompt_id]:
            st.session_state.ranking[prompt_id][st.session_state.counter[prompt_id]] = {}
            for i in range(len(items)):
                st.session_state.ranking[prompt_id][st.session_state.counter[prompt_id]][str(items['image_id'][i])] = i
        else:
            # set the index of items to the corresponding ranking value of the image_id
            items.index = items['image_id'].apply(lambda x: st.session_state.ranking[prompt_id][st.session_state.counter[prompt_id]][str(x)])

        with elements('dashboard'):
            if layout == 'portrait':
                col_num = 4
                layout = [dashboard.Item(str(items['image_id'][i]), i % col_num, i//col_num, 1, 2, isResizable=False) for i in range(len(items))]

            elif layout == 'landscape':
                col_num = 2
                layout = [
                    dashboard.Item(str(items['image_id'][i]), i % col_num * 2, i // col_num, 2, 1.6, isResizable=False) for
                    i in range(len(items))
                ]

            with dashboard.Grid(layout, cols={'lg': 4, 'md': 4, 'sm': 4, 'xs': 4, 'xxs': 2}, onLayoutChange=self.handle_layout_change, margin=[18, 18], containerPadding=[0, 0]):
                for i in range(len(layout)):
                    with mui.Card(key=str(items['image_id'][i]), variant="outlined"):
                        prompt_id = st.session_state.prompt_id_tmp
                        batch_idx = st.session_state.counter[prompt_id]

                        rank = st.session_state.ranking[prompt_id][batch_idx][str(items['image_id'][i])] + 1

                        mui.Chip(label=rank,
                                 # variant="outlined" if rank!=1 else "default",
                                 color="primary" if rank == 1 else "warning" if rank == 2 else "info",
                                 size="small",
                                 sx={"position": "absolute", "left": "-0.3rem", "top": "-0.3rem"})

                        img_url = self.images_endpoint + str(items['image_id'][i]) + '.png'

                        mui.CardMedia(
                            component="img",
                            # image={"data:image/png;base64", img_str},
                            image=img_url,
                            alt="There should be an image",
                            sx={"height": "100%", "object-fit": "contain", 'bgcolor': 'black'},
                        )

    def handle_layout_change(self, updated_layout):
        # print(updated_layout)
        sorted_list = sorted(updated_layout, key=lambda x: (x['y'], x['x']))
        sorted_list = [str(item['i']) for item in sorted_list]

        prompt_id = st.session_state.prompt_id_tmp
        batch_idx = st.session_state.counter[prompt_id]

        for k in st.session_state.ranking[prompt_id][batch_idx].keys():
            st.session_state.ranking[prompt_id][batch_idx][k] = sorted_list.index(k)

    def dragsort_mode(self, tag, items, prompt_id, batch_num):
        st.session_state.counter[prompt_id] = 0 if prompt_id not in st.session_state.counter else \
        st.session_state.counter[prompt_id]

        sorting, control = st.columns((11, 1), gap='large')
        with sorting:
            # st.write('## Sorting')
            # st.write('Please drag the images to sort them.')
            st.progress((st.session_state.counter[prompt_id] + 1) / batch_num,
                        text=f"Batch {st.session_state.counter[prompt_id] + 1} / {batch_num}")
            # st.write(items.iloc[self.batch_size*st.session_state.counter[prompt_id]: self.batch_size*(st.session_state.counter[prompt_id]+1)])

            width, height = items.loc[0, 'size'].split('x')
            if int(height) >= int(width):
                self.draggable_images(items.iloc[
                                      self.batch_size * st.session_state.counter[prompt_id]: self.batch_size * (
                                                  st.session_state.counter[prompt_id] + 1)].reset_index(drop=True),
                                      prompt_id=prompt_id, layout='portrait')
            else:
                self.draggable_images(items.iloc[
                                      self.batch_size * st.session_state.counter[prompt_id]: self.batch_size * (
                                                  st.session_state.counter[prompt_id] + 1)].reset_index(drop=True),
                                      prompt_id=prompt_id, layout='landscape')
            # st.write(str(st.session_state.ranking))
        with control:
            if st.session_state.counter[prompt_id] < batch_num - 1:
                st.button(":arrow_right:", key='next', on_click=self.next_batch, help='Next Batch',
                          kwargs={'tag': tag, 'prompt_id': prompt_id}, use_container_width=True, type='primary')
            else:
                st.button(":heavy_check_mark:", key='finished', on_click=self.next_batch, help='Finished',
                          kwargs={'tag': tag, 'prompt_id': prompt_id, 'progress': 'finished'}, use_container_width=True, type='primary')

            if st.session_state.counter[prompt_id] > 0:
                st.button(":arrow_left:", key='prev', on_click=self.prev_batch, help='Previous Batch',
                          kwargs={'prompt_id': prompt_id}, use_container_width=True)

    def next_batch(self, tag, prompt_id, progress=None):

        curser = RANKING_CONN.cursor()

        # a not so elegant way to get the modelVersion_id of each image, but it works
        position_version_dict = {}
        for image_id, position in st.session_state.ranking[prompt_id][st.session_state.counter[prompt_id]].items():
            modelVersion_id = self.promptBook[self.promptBook['image_id'] == image_id]['modelVersion_id'].values[0]
            position_version_dict[position] = modelVersion_id

        # # get all records of this user and prompt
        # query = "SELECT * FROM sort_results WHERE username = %s AND timestamp = %s AND prompt_id = %s"
        # curser.execute(query, (st.session_state.user_id[0], st.session_state.user_id[1], prompt_id))
        # results = curser.fetchall()
        # print(results)
        #
        # # remove the old ranking with the same modelVersion_id if exists
        # for result in results:
        #     prev_ids = [result['position1'], result['position2'], result['position3'], result['position4']]
        #     curr_ids = [position_version_dict[0], position_version_dict[1], position_version_dict[2], position_version_dict[3]]
        #     if len(set(prev_ids).intersection(set(curr_ids))) == 4:
        #         query = "DELETE FROM sort_results WHERE username = %s AND timestamp = %s AND prompt_id = %s AND position1 = %s AND position2 = %s AND position3 = %s AND position4 = %s"
        #         curser.execute(query, (st.session_state.user_id[0], st.session_state.user_id[1], prompt_id, result['position1'], result['position2'], result['position3'], result['position4']))

        sorttime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        # handle the case where user press the 'prev' button
        query = "DELETE FROM sort_results WHERE username = %s AND timestamp = %s AND prompt_id = %s AND epoch = %s"
        curser.execute(query, (st.session_state.user_id[0], st.session_state.user_id[1], prompt_id, st.session_state.epoch['ranking'][prompt_id]))
        query = "INSERT INTO sort_results (username, timestamp, tag, prompt_id, position1, position2, position3, position4, sorttime, epoch) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        curser.execute(query, (st.session_state.user_id[0], st.session_state.user_id[1], self.promptBook[self.promptBook['prompt_id'] == prompt_id]['tag'].values[0], prompt_id, position_version_dict[0], position_version_dict[1], position_version_dict[2], position_version_dict[3], sorttime, st.session_state.epoch['ranking'][prompt_id]))

        curser.close()
        RANKING_CONN.commit()

        if progress == 'finished':
            st.session_state.epoch['ranking'][prompt_id] += 1

            st.session_state.epoch['summary'][tag] = st.session_state.epoch['summary'].get(tag, 0) + 1
            st.session_state.epoch['summary']['overall'] += 1

            st.session_state.progress[prompt_id] = 'finished'
            # drop 'modelVersion_standings' from session state if exists
            st.session_state.pop('modelVersion_standings', None)
        else:
            st.session_state.counter[prompt_id] += 1

    def prev_batch(self, prompt_id):
        st.session_state.counter[prompt_id] -= 1

    def battle_images(self, tag, items, prompt_id):
        if 'pointer' not in st.session_state:
            st.session_state.pointer = {}

        if prompt_id not in st.session_state.pointer:
            st.session_state.pointer[prompt_id] = {'left': 0, 'right': 1}

        curr_position = max(st.session_state.pointer[prompt_id]['left'], st.session_state.pointer[prompt_id]['right'])
        progress = st.progress(curr_position / (len(items)-1), text=f"Progress {curr_position} / {len(items)-1}")

        # if curr_position == len(items) - 1:
        #     st.session_state.progress[prompt_id] = 'finished'
        #
        # else:
        left, right = st.columns(2)
        with left:
            image_id = items['image_id'][st.session_state.pointer[prompt_id]['left']]
            img_url = self.images_endpoint + str(image_id) + '.png'

            # # write the total score of this image
            # total_score = items['total_score'][st.session_state.pointer[prompt_id]['left']]
            # st.write(f'Total Score: {total_score}')

            btn_left = st.button('Left is better', key='left', on_click=self.next_battle, kwargs={'tag': tag, 'prompt_id': prompt_id, 'image_ids': items['image_id'], 'winner': 'left', 'curr_position': curr_position, 'total_num': len(items)}, use_container_width=True)
            st.image(img_url, use_column_width=True)

        with right:
            image_id = items['image_id'][st.session_state.pointer[prompt_id]['right']]
            img_url = self.images_endpoint + str(image_id) + '.png'

            # # write the total score of this image
            # total_score = items['total_score'][st.session_state.pointer[prompt_id]['right']]
            # st.write(f'Total Score: {total_score}')

            btn_right = st.button('Right is better', key='right', on_click=self.next_battle, kwargs={'tag': tag, 'prompt_id': prompt_id, 'image_ids': items['image_id'], 'winner': 'right', 'curr_position': curr_position, 'total_num': len(items)}, use_container_width=True)
            st.image(img_url, use_column_width=True)

    def next_battle(self, tag, prompt_id, image_ids, winner, curr_position, total_num):
        loser = 'left' if winner == 'right' else 'right'
        battletime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        curser = RANKING_CONN.cursor()

        winner_modelVersion_id = self.promptBook[self.promptBook['image_id'] == image_ids[st.session_state.pointer[prompt_id][winner]]]['modelVersion_id'].values[0]
        loser_modelVersion_id = self.promptBook[self.promptBook['image_id'] == image_ids[st.session_state.pointer[prompt_id][loser]]]['modelVersion_id'].values[0]

        # # remove the old battle result if exists
        # query = "DELETE FROM battle_results WHERE username = %s AND timestamp = %s AND prompt_id = %s AND winner = %s AND loser = %s"
        # curser.execute(query, (st.session_state.user_id[0], st.session_state.user_id[1], prompt_id, winner_modelVersion_id, loser_modelVersion_id))
        # curser.execute(query, (st.session_state.user_id[0], st.session_state.user_id[1], prompt_id, loser_modelVersion_id, winner_modelVersion_id))

        # insert the battle result into the database
        query = "INSERT INTO battle_results (username, timestamp, tag, prompt_id, winner, loser, battletime, epoch) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        curser.execute(query, (st.session_state.user_id[0], st.session_state.user_id[1], self.promptBook[self.promptBook['prompt_id'] == prompt_id]['tag'].values[0], prompt_id, winner_modelVersion_id, loser_modelVersion_id, battletime, st.session_state.epoch['ranking'][prompt_id]))

        curser.close()
        RANKING_CONN.commit()

        if curr_position == total_num - 1:
            st.session_state.epoch['ranking'][prompt_id] += 1

            st.session_state.epoch['summary'][tag] = st.session_state.epoch['summary'].get(tag, 0) + 1
            st.session_state.epoch['summary']['overall'] += 1

            st.session_state.progress[prompt_id] = 'finished'

            # drop 'modelVersion_standings' from session state if exists
            st.session_state.pop('modelVersion_standings', None)

            # st.experimental_rerun()
        else:
            st.session_state.pointer[prompt_id][loser] = curr_position + 1

    def battle_mode(self, tag, items, prompt_id):
        self.battle_images(tag, items, prompt_id)

    def app(self):
        st.write('### Generative Model Ranking')
        # st.write('Here you can test out your selected images with any prompt you like. ')
        # st.write(self.promptBook)

        # save the current progress to session state
        if 'progress' not in st.session_state:
            st.session_state.progress = {}
        print('current progress: ', st.session_state.progress)

        # select tag and prompt
        prompt_tags = self.promptBook['tag'].unique()
        prompt_tags = np.sort(prompt_tags).tolist()

        print(st.session_state.gallery_focus)
        tag_idx = prompt_tags.index(st.session_state.gallery_focus['tag']) if st.session_state.gallery_focus[
                                                                                  'tag'] in prompt_tags else 0
        print(tag_idx)

        # color the finished tags
        finished_tags = []
        for tag in prompt_tags:
            append_tag = True
            for prompt_id in self.promptBook[self.promptBook['tag'] == tag]['prompt_id'].unique():
                if prompt_id not in st.session_state.progress or st.session_state.progress[prompt_id] != 'finished':
                    append_tag = False
                    break
            if append_tag:
                finished_tags.append(tag)
        tag_tobe_colored = self.promptBook[self.promptBook['tag'].isin(finished_tags)]['tag'].unique().tolist()
        colored_tags = self.text_coloring_add(tag_tobe_colored, prompt_tags, color_name='orange')

        tag = st.radio('Select a tag', colored_tags, index=tag_idx, horizontal=True, label_visibility='collapsed')
        tag = self.text_coloring_remove(tag)
        print(tag)
        items = self.promptBook[self.promptBook['tag'] == tag].reset_index(drop=True)
        # pick out prompts such that st.session_state.progress[prompt_id] == 'finished'

        prompts = np.sort(items['prompt'].unique())[::-1].tolist()

        prompt_idx = prompts.index(st.session_state.gallery_focus['prompt']) if st.session_state.gallery_focus[
                                                                                    'prompt'] in prompts else 0
        print(prompt_idx)

        # color the finished prompts
        finished_prompts = []
        for prompt_id in items['prompt_id'].unique():
            if prompt_id in st.session_state.progress and st.session_state.progress[prompt_id] == 'finished':
                finished_prompts.append(prompt_id)
        prompt_tobe_colored = items[items['prompt_id'].isin(finished_prompts)]['prompt'].unique().tolist()
        colored_prompts = self.text_coloring_add(prompt_tobe_colored, prompts, color_name='✅')

        selected_prompt = st.selectbox('Select a prompt', colored_prompts, index=prompt_idx, label_visibility='collapsed')
        selected_prompt = self.text_coloring_remove(selected_prompt)

        st.session_state.gallery_focus = {'tag': tag, 'prompt': selected_prompt}

        # mode = st.radio('Select a mode', ['Drag and Sort', 'Battle'], index=1)
        mode = st.session_state.assigned_rank_mode

        items = items[items['prompt'] == selected_prompt].reset_index(drop=True)
        prompt_id = items['prompt_id'].unique()[0]

        self.sidebar(selected_prompt, items)
        batch_num = len(items) // self.batch_size
        batch_num += 1 if len(items) % self.batch_size != 0 else 0

        # st.session_state.counter[prompt_id] = 0 if prompt_id not in st.session_state.counter else st.session_state.counter[prompt_id]

        # save prompt_id in session state
        st.session_state.prompt_id_tmp = prompt_id

        if prompt_id not in st.session_state.progress:
            st.session_state.progress[prompt_id] = 'ranking'

        if st.session_state.progress[prompt_id] == 'ranking':
            st.session_state.epoch['ranking'][prompt_id] = st.session_state.epoch['ranking'].get(prompt_id, 1)
            st.caption("We might pair some other images that you haven't selected based on our evaluation matrix.")
            if mode == 'Drag and Sort':
                self.dragsort_mode(tag, items, prompt_id, batch_num)
            elif mode == 'Battle':
                self.battle_mode(tag, items, prompt_id)

        elif st.session_state.progress[prompt_id] == 'finished':
            print(st.session_state.gallery_focus)
            # st.toast('**Summary is available now!**')
            # st.write('---')
            with st.form(key='ranking_finished'):
                st.info('**🎉 You have ranked all models for this prompt!**')
                st.write('Feel free to do the following things:')

                # st.write('* Back to the gallery page to see more images.')
                # st.write('* Rank again for this tag and prompt.')
                options_panel = st.columns(4)

                with options_panel[0]:
                    summary_btn = st.form_submit_button('📊 See Summary', use_container_width=True, type='primary')
                    if summary_btn:
                        switch_page('summary')

                with options_panel[1]:
                    gallery_btn = st.form_submit_button('🖼️ Back to Gallery', use_container_width=True)
                    if gallery_btn:
                        switch_page('gallery')

                with options_panel[2]:
                    st.form_submit_button('👆 Rank other prompts', use_container_width=True)

                with options_panel[3]:
                    restart_btn = st.form_submit_button('🎖️ Re-rank this prompt', use_container_width=True, on_click=self.rerank, kwargs={'prompt_id': prompt_id})

        # with st.sidebar:
        #     st.write('epoch: ', st.session_state.epoch['ranking'][prompt_id])
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

    def rerank(self, prompt_id):
        st.session_state.progress[prompt_id] = 'ranking'
        if st.session_state.assigned_rank_mode == 'Drag and Sort':
            st.session_state.counter[prompt_id] = 0
        elif st.session_state.assigned_rank_mode == 'Battle':
            st.session_state.pointer[prompt_id] = {'left': 0, 'right': 1}


if __name__ == "__main__":
    st.set_page_config(page_title="Personal Image Ranking", page_icon="🎖️️", layout="wide")

    if 'user_id' not in st.session_state:
        st.warning('Please log in first.')
        home_btn = st.button('Go to Home Page')
        if home_btn:
            switch_page("home")

    else:
        has_selection = False

        for key, value in st.session_state.selected_dict.items():
            for v in value:
                if v:
                    has_selection = True
                    break

        if not has_selection:
            st.info('You have not checked any image yet. Please go back to the gallery page and check some images.')
            gallery_btn = st.button('🖼️ Go to Gallery')
            if gallery_btn:
                switch_page('gallery')
        else:
            # st.write('You have checked ' + str(len(selected_modelVersions)) + ' images.')
            roster, promptBook, images_ds = load_hf_dataset(st.session_state.show_NSFW)
            print(st.session_state.selected_dict)

            # st.write("# Full function is coming soon.")
            RANKING_CONN = connect_to_db()

            # only select the part of the promptbook where tag is the same as st.session_state.selected_dict.keys(), while model version ids are the same as corresponding values to each key
            promptBook_selected = pd.DataFrame()
            for key, value in st.session_state.selected_dict.items():
                # promptBook_selected = promptBook_selected.append(promptBook[(promptBook['prompt_id'] == key) & (promptBook['modelVersion_id'].isin(value))])
                # replace append with pd.concat
                user_selections = promptBook[(promptBook['prompt_id'] == key) & (promptBook['modelVersion_id'].isin(value))]

                # auto complete the selection with random images
                residual = len(user_selections) % 4
                if residual != 0:
                    # select 4-residual items from the promptbook outside the user_selections
                    npc = promptBook[(promptBook['prompt_id'] == key) & (~promptBook['modelVersion_id'].isin(value))].sort_values(by=['total_score'], ascending=False).reset_index(drop=True).iloc[:4-residual]
                    user_selections = pd.concat([user_selections, npc])

                promptBook_selected = pd.concat([promptBook_selected, user_selections])
            promptBook_selected = promptBook_selected.reset_index(drop=True)
            # sort promptBook by total_score
            promptBook_selected = promptBook_selected.sort_values(by=['total_score'], ascending=True).reset_index(drop=True)

            # st.write(promptBook_selected)
            images_endpoint = "https://modelcofferbucket.s3-accelerate.amazonaws.com/"

            app = RankingApp(promptBook_selected, images_endpoint, batch_size=4)
            app.app()

    with open('./css/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
