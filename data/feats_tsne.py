import os

import numpy as np
import pandas as pd
import torch

from sklearn.manifold import TSNE
from tqdm import tqdm


def load_feats(path='./feats'):
    print('==> loading feats')
    feats = {}
    for pt in os.listdir(path):
        if pt.split('.')[-1] == 'pt' and pt.split('.')[0].isdigit():
            feats[int(pt.split('.')[0])] = torch.load(os.path.join('../data/feats', pt))
    return feats


def calc_tsne(feat):
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)
    res = tsne.fit_transform(feat['all'].numpy())
    return res


def test_open(fp='./feats_tsne.parquet'):
    df = pd.read_parquet(fp)
    print(df.head())


if __name__ == '__main__':
    feats = load_feats()
    df = pd.DataFrame(columns=['x', 'y', 'prompt_id', 'modelVersion_id'])

    print('==> applying t-SNE')
    for k, v in tqdm(feats.items()):
        modelVersion_ids = []
        for id in v.keys():
            if id != 'all' and id != 'tsne':
                modelVersion_ids.append(int(id.item()))

        res = calc_tsne(v)

        tmp = pd.DataFrame(res, columns=['x', 'y'])
        tmp['prompt_id'] = k
        tmp['modelVersion_id'] = modelVersion_ids

        df = pd.concat([df, tmp], ignore_index=True)

    df.to_parquet('./feats_tsne.parquet')

    # test_open()