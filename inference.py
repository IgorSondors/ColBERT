import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from interface import prepare_tsv, save_index, top_n_similar, Collection
import pandas as pd
import time


def filter_by_category_id(df, id_category):
    try:
        df = df.drop(columns=['true_match', 'false_match'])
    except:
        print('true_match, false_match - колонки отсутствуют')

    valid_category_ids = set(id_category.keys())
    filtered_df = df[df['category_id'].isin(valid_category_ids)]
    return filtered_df.reset_index(drop=True)

def offers_to_dict(df, cat_id):
    df = df[df.category_id == cat_id]#.reset_index(drop=True)
    return df['name'].to_dict(), tuple(df.index)

def top_n_to_df(df, indices, top_n, n):
    for idx, insert_dict in zip(indices, top_n):
        for i in range(n):
            col_model_id = f'model_id_pred_{i+1}'
            col_similarity = f'similarity_{i+1}'
            df.loc[idx, col_model_id] = insert_dict['model_ids'][i]
            df.loc[idx, col_similarity] = round(float(insert_dict['similarity'][i]), 2)
    return df

def convert_columns_to_int(df, num_columns):
    for i in range(1, num_columns + 1):
        col_name = f'model_id_pred_{i}'
        df[col_name] = df[col_name].astype(int)
    return df

if __name__=='__main__':
    # pth_offers = "/home/sondors/Documents/price/BERT_data/data/18_categories/triplets_test_18_categories.csv"
    pth_offers = "/home/sondors/Documents/price/BERT_data/data/27-03-2024_Timofey/740101_offers_active_model_id_0_unique.csv"
    # pth_models = "/home/sondors/Documents/price/ColBERT_data/18_categories/test/offers_and_models_18_categories.csv"
    pth_models = "/home/sondors/Documents/price/BERT_data/data/27-03-2024_Timofey/740101_models.csv"
    # dst_fld = "/home/sondors/Documents/price/ColBERT/accuracies/offers_and_models"
    dst_fld = "/home/sondors/Documents/price/ColBERT_data/10_categories/740101/test"

    ckpt_pth = "/home/sondors/Documents/ColBERT_weights/740101_lr04_bsize230/colbert-489-finish"
    experiment = "740101_lr04_bsize230"

    doc_maxlen = 300
    nbits = 2   # bits определяет количество битов у каждого измерения в семантическом пространстве во время индексации
    nranks = 1  # nranks определяет количество GPU для использования, если они доступны
    kmeans_niters = 4 # kmeans_niters указывает количество итераций k-means кластеризации; 4 — хороший и быстрый вариант по умолчанию.  
    n = 5 # top_n_similar
    n_to_df = 5 # top_n to df

    # id_category = {
    #     3902: 'диктофоны, портативные рекордеры',
    #     510402: 'электронные книги',
    #     4302: 'автомобильные телевизоры, мониторы',
    #     2815: 'смарт-часы и браслеты',
    #     3901: 'портативные медиаплееры',
    #     3904: 'портативная акустика',
    #     2801: 'мобильные телефоны',
    #     3908: 'VR-гарнитуры (VR-очки, шлемы, очки виртуальной реальности, FPV очки для квадрокоптеров)',
    #     510401: 'планшетные компьютеры и мини-планшеты',
    #     2102: 'наушники, гарнитуры, наушники c микрофоном',
    #     3903: 'радиоприемники, радиобудильники, радиочасы',
    #     3907: 'магнитолы',
    #     280801: 'GPS-навигаторы'
    #     }

    id_category = {
        # 921201: 'корм для кошек',
        # 963401: 'колбасы',
        # 710502: 'стиральные машины',
        # 977135: 'ботинки, полуботинки',
        # 911906: 'парфюмерия',
        # 976132: 'туфли и лоферы',
        740101: 'кухонные мойки',
        # 7106: 'микроволновые печи',
        # 921401: 'корм для собак',
        # 963302: 'пицца'
        }
    
    start_time = time.time()

    df_offers = pd.read_csv(pth_offers, sep=";")
    df_offers = filter_by_category_id(df_offers, id_category)
    df_models = pd.read_csv(pth_models, sep=";")

    for cat_id in id_category.keys():

        category_models = df_models[df_models.category_id == cat_id].reset_index(drop=True)
        prepare_tsv(category_models, dst_fld, cat_id)

        models = Collection(path=os.path.join(dst_fld, "tsv", f"{cat_id}_models.tsv"))
        index_name = f'models_{cat_id}_{nbits}bits'
        indexer = save_index(ckpt_pth, doc_maxlen, nbits, kmeans_niters, nranks, dst_fld, experiment, models, index_name)

        models_id = Collection(path=os.path.join(dst_fld, "tsv", f"{cat_id}_models_id.tsv"))
        offers, indices = offers_to_dict(df_offers, cat_id)
        top_n = top_n_similar(offers, dst_fld, nranks, experiment, index_name, models_id, n)

        df_offers = top_n_to_df(df_offers, indices, top_n, n_to_df)

    df_offers = convert_columns_to_int(df_offers, n_to_df)
    df_offers.to_csv(os.path.join(dst_fld, f"{experiment}_offers_top_n_model_id_0.csv"), sep=';',index=False)

    time_spent = time.time() - start_time
    print(f"time_spent = {time_spent}")