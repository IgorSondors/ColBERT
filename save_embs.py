import os
from interface import get_query_emb_batch
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

import pandas as pd
import numpy as np
import time

def load_model(ckpt_pth, device, doc_maxlen, nbits, nranks, kmeans_niters):
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=kmeans_niters)
    checkpoint = Checkpoint(ckpt_pth, colbert_config=config)
    return checkpoint.to(device)

if __name__ == '__main__':
    pth_offers = "/home/sondors/Documents/price/BERT_data/data/13_categories/triplets_13_categories_filtered_train.csv"
    dst_fld = "/home/sondors/Documents/price/triplets_13_categories_filtered_train_embs/bert-base-multilingual-colbert-2998"
    os.makedirs(dst_fld, exist_ok=True)
    ckpt_fld = "/home/sondors/Documents/ColBERT_weights/bert-base-multilingual-cased_dim_768_bsize_230_lr04_use_ib_negatives/none/2024-01/27/16.55.29/checkpoints/colbert-2998-finish"
    doc_maxlen = 300
    nbits = 2
    nranks = 1
    kmeans_niters = 4
    device = "cuda"

    checkpoint = load_model(ckpt_fld, device, doc_maxlen, nbits, nranks, kmeans_niters)
    
    df_offers = pd.read_csv(pth_offers, sep=";")
    offers = list(df_offers["name"])
    start_time = time.time()
    
    # Определение размера части и количества частей
    part_size = 30000
    # num_parts = len(offers) // part_size + (1 if len(offers) % part_size > 0 else 0)
    num_parts = (len(offers) + part_size - 1) // part_size
    # Индекс для имени файла
    file_index = 0
    
    for part in range(num_parts):
        start_idx = part * part_size
        end_idx = min((part + 1) * part_size, len(offers))
        part_offers = offers[start_idx:end_idx]
        # Получаем подсписок для текущей части
        # part_offers = offers[part * part_size : (part + 1) * part_size]
        
        # Получаем эмбеддинги для текущей части
        part_embs = get_query_emb_batch(part_offers, checkpoint, batch_size=100, batch_size2=5000)
        # Сохраняем эмбеддинги текущей части
        for emb in part_embs:
            np.save(f'{dst_fld}/{file_index}.npy', emb)
            file_index += 1
    
    time_spent = time.time() - start_time
    print(f"time_spent = {time_spent}")