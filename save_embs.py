import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from interface import get_raw_emb

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint


import pandas as pd
import numpy as np
import time
import os

def load_model(ckpt_pth, device, doc_maxlen, nbits, nranks, kmeans_niters):
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=kmeans_niters)
    checkpoint = Checkpoint(ckpt_pth, colbert_config=config)
    return checkpoint.to(device)

if __name__=='__main__':

    pth_offers = "/home/sondors/Documents/price/offers_clusterization/data/14_categories_balanced.csv"

    dst_fld = "/home/sondors/Documents/price/14_categories_balanced_embs_colbert-4446"
    
    ckpt_pth = "/home/sondors/triples_X1_13_categories_filtered_shuffle_use_ib_negatives_lr04/colbert-4446-finish"
    doc_maxlen = 300
    nbits = 2   # bits определяет количество битов у каждого измерения в семантическом пространстве во время индексации
    nranks = 1  # nranks определяет количество GPU для использования, если они доступны
    kmeans_niters = 4 # kmeans_niters указывает количество итераций k-means кластеризации; 4 — хороший и быстрый вариант по умолчанию.  
    device = "cuda"

    checkpoint = load_model(ckpt_pth, device, doc_maxlen, nbits, nranks, kmeans_niters)
    
    df_offers = pd.read_csv(pth_offers, sep=";")
    offers = list(df_offers["name"])
    start_time = time.time()
    
    embs = get_raw_emb(offers[:2], checkpoint, 500)

    os.makedirs(dst_fld, exist_ok=True)

    for i in range(len(embs)):
        np.save(f'{dst_fld}/{i}.npy', embs[i])

    time_spent = time.time() - start_time
    print(f"time_spent = {time_spent}")