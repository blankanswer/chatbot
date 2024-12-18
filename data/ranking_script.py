from datasets import Dataset
from datetime import datetime


def init_ranking_data():
    ds = Dataset.from_dict({'image_id': [], 'modelVersion_id': [], 'ranking': [], "user_name": [], "timestamp": []})\

    # add example data
    # note that image_id is a string, other ids are int
    ds = ds.add_item({'image_id': '0', 'modelVersion_id': 0, 'ranking': 0, "user_name": "example_data", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    ds.push_to_hub("MAPS-research/GEMRec-Ranking", split='train')


if __name__ == '__main__':
    init_ranking_data()

