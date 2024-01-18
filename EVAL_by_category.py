import os
import sys
sys.path.insert(0, '../')
import time
import json
import pandas as pd

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

def read_df(pth_models, pth_offers):
    df_models = pd.read_csv(pth_models, sep=";")
    df_models = df_models.drop(columns=['average_price', 'comment'])
    df_offers = pd.read_csv(pth_offers, sep=";")
    df_offers = df_offers.drop(columns=['true_match', 'false_match'])

    return df_models, df_offers

def search(checkpoint, offers, models, nbits, doc_maxlen, tmp_fld):
    index_name = f'models.18_categories.{nbits}bits'

    offers = Queries(path=offers)
    models = Collection(path=models)
    with open(f"{tmp_fld}/logs.txt", "a") as txt:
        txt.write(f"\nLoaded {len(offers)} queries and {len(models):,} passages\n")

    start_time = time.time()
    with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=models, overwrite=True)
    indexer.get_index() # You can get the absolute path of the index, if needed.

    with Run().context(RunConfig(experiment='notebook')):
        searcher = Searcher(index=index_name)
    with open(f"{tmp_fld}/logs.txt", "a") as txt:
        txt.write(f"Подготовка моделей категории: time_spent = {time.time() - start_time}\n")
    print(f"Подготовка моделей категории: time_spent = {time.time() - start_time}\n")

    start_time = time.time()
    rankings = searcher.search_all(offers, k=5).todict()
    with open(f"{tmp_fld}/logs.txt", "a") as txt:
        txt.write(f"Инференс на всех офферах категории: time_spent = {time.time() - start_time}\n")
    print(f"Инференс на всех офферах категории: time_spent = {time.time() - start_time}\n\n\n")

    return rankings

def ranking_index(rankings, category_rankings, df, index_of_first):
    """
    сделать сдвиг passage_id на величину index_of_first для приведения к формату в котором поиск оффера ведется среди моделей всех категорий 
    упорядочить (passage_id, rank, score) в rankings согласно изначальным индексам в df_offers
    """
    assert len(category_rankings) == len(df)

    for i in category_rankings:
        for j in range(len(category_rankings[i])):
            category_rankings[i][j] = (category_rankings[i][j][0] + index_of_first, category_rankings[i][j][1], category_rankings[i][j][2])

        for k in range(5 - len(category_rankings[i])): # все предикты top-k делаем длиной 5
            category_rankings[i].append((0, 0, 0))#((index_of_first, 0, 0))#(category_rankings[i][- 1])
        assert len(category_rankings[i]) == 5

    i = -1
    for index, row in df.iterrows():
        i += 1
        rankings[index] = category_rankings[i]

    return rankings

def df_split(df, col="name"):
    df = df.reset_index(drop=True)

    df1 = pd.DataFrame()
    df1["id"], df1[col] = [i for i in range(len(df))], df[col]
    
    df2 = pd.DataFrame()
    df2["id"], df2["model_id"] = [i for i in range(len(df))], df['model_id']

    return df1, df2

def prepare_tsv(category_offers, category_models, pth_offers, pth_models):
    query, query_id = df_split(category_offers, col="name")
    query.to_csv(pth_offers, sep='\t', header=False, index=False)

    document, document_id = df_split(category_models, col="full_name")
    document.to_csv(pth_models, sep='\t', header=False, index=False)
    
def wrt_json(categories, pth_models, pth_offers, ckpt_pth, tmp_fld, pth_dst_json):
    with open(f"{tmp_fld}/logs.txt", "a") as txt:
        txt.write(f"ckpt_pth = {ckpt_pth}:\n")
    df_models, df_offers = read_df(pth_models, pth_offers)

    doc_maxlen = 300
    nbits = 2   # encode each dimension with 2 bits
    rankings = {}
    for category in categories:
        with open(f"{tmp_fld}/logs.txt", "a") as txt:
            txt.write(f"\n{category}:\n")
        print(category)
        index_of_first = df_models.index[df_models['category_name'] == category].tolist()[0]

        category_models = df_models[df_models['category_name'] == category]
        category_offers = df_offers[df_offers['category_name'] == category]

        pth_models = f"{tmp_fld}/models.tsv"
        pth_offers = f"{tmp_fld}/offers.tsv"
        prepare_tsv(category_offers, category_models, pth_offers, pth_models)

        category_rankings = search(ckpt_pth, pth_offers, pth_models, nbits, doc_maxlen, tmp_fld)
        rankings = ranking_index(rankings, category_rankings, category_offers, index_of_first)

    with open(pth_dst_json+f"_{ckpt_pth.split('/')[-1]}.json", 'w') as fp:
        json.dump(rankings, fp)
    with open(f"{tmp_fld}/logs.txt", "a") as txt:
        txt.write(f"Save pth: {pth_dst_json}_{ckpt_pth.split('/')[-1]}.json\n")

if __name__=='__main__':
    pth_models = "/mnt/vdb1/Datasets/ColBERT/18_categories/test/models_18_categories.csv"
    pth_offers = "/mnt/vdb1/Datasets/ColBERT/18_categories/test/triplets_test_18_categories.csv"
    tmp_fld = "/mnt/vdb1/ColBERT/tmp"
    pth_dst_json = "/mnt/vdb1/ColBERT/tmp/triples_accum_12"

    categories = [
        "диктофоны, портативные рекордеры",
        "электронные книги",
        "автомобильные телевизоры, мониторы",
        "смарт-часы и браслеты",
        "портативные медиаплееры",
        "чехлы, обложки для гаджетов (телефонов, планшетов etc)",
        "портативная акустика",
        "мобильные телефоны",
        "VR-гарнитуры (VR-очки, шлемы, очки виртуальной реальности, FPV очки для квадрокоптеров)",
        "планшетные компьютеры и мини-планшеты",
        "наушники, гарнитуры, наушники c микрофоном",
        "радиоприемники, радиобудильники, радиочасы",
        "магнитолы",
        "GPS-навигаторы"
        ]
    
    # ckpts_pth = "/home/sondors/HYPERPARAM/none/2024-01/09/22.18.23/checkpoints"
    # for checkpoint in os.listdir(ckpts_pth):
    #     ckpt_pth = os.path.join(ckpts_pth, checkpoint)
    ckpt_pth = "/mnt/vdb1/ColBERT/experiments/HYPERPARAM_accum_12/none/2024-01/16/13.07.51/checkpoints/colbert-6000-finish"
    all_categories_time = time.time()
    wrt_json(categories, pth_models, pth_offers, ckpt_pth, tmp_fld, pth_dst_json)
    with open(f"{tmp_fld}/logs.txt", "a") as txt:
        txt.write(f"all_categories_time = {time.time() - all_categories_time}\n\n")
        txt.write("-"*100)

