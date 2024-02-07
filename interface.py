import os
import sys
sys.path.insert(0, '../')
import pandas as pd

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

def prepare_tsv(df, dst_fld, category_id):
    """
    Делим модели по full_name и соответствующие им model_id на два tsv файла 

    {category_id}_models.tsv:

    0   model0
    1   model1
    2   model2

    {category_id}_models_id.tsv:

    0   model_id0
    1   model_id1
    2   model_id2
    
    """

    def df_split(df):
        df1 = pd.DataFrame()
        df1["id"], df1["full_name"] = [i for i in range(len(df))], df["full_name"]
        
        df2 = pd.DataFrame()
        df2["id"], df2["model_id"] = [i for i in range(len(df))], df["model_id"]

        return df1, df2
    
    os.makedirs(os.path.join(dst_fld, "tsv"), exist_ok=True)
    models, models_id = df_split(df)
    models.to_csv(os.path.join(dst_fld, "tsv", f"{category_id}_models.tsv"), sep='\t', header=False, index=False)
    models_id.to_csv(os.path.join(dst_fld, "tsv", f"{category_id}_models_id.tsv"), sep='\t', header=False, index=False)

def save_index(ckpt_pth, doc_maxlen, nbits, nranks, dst_fld, experiment, collection, index_name):
    with Run().context(RunConfig(nranks=nranks, root=dst_fld, experiment=experiment)):
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)
        indexer = Indexer(checkpoint=ckpt_pth, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
    return indexer

def top_n_similar(offers, src_fld, nranks, experiment, index_name, model_ids, n):
    with Run().context(RunConfig(nranks=nranks, root=src_fld, experiment=experiment)):
        searcher = Searcher(index=index_name, collection=model_ids)
        offers = Queries(data=offers)
        rankings = searcher.search_all(offers, k=n)
        top_n = rankings_to_dict(rankings, searcher)
    return top_n

def rankings_to_dict(rankings, searcher):
    result = []
    for key, value in rankings.todict().items():
        model_ids = [int(searcher.collection[item[0]]) for item in value]
        similarity = [item[2] for item in value]
        result.append({'model_ids': model_ids, 'similarity': similarity})
    return result