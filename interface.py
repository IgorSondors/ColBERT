from typing import Tuple, List, Dict, Union, Any
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

import pandas as pd
import os

def prepare_tsv(df: pd.DataFrame, dst_fld: str, category_id: str) -> None:
    """
    Prepare and save two TSV files containing model information. Below is default format for input of models to ColBERT to index them.

    {category_id}_models.tsv:

    0   model0
    1   model1
    2   model2

    {category_id}_models_id.tsv:

    0   model_id0
    1   model_id1
    2   model_id2

    Args:
        df (pd.DataFrame): DataFrame containing models information.
        dst_fld (str): Destination folder where TSV files will be saved.
        category_id (str): Identifier for the category of models.

    Returns:
        None
    """
    os.makedirs(os.path.join(dst_fld, "tsv"), exist_ok=True)
    models, models_id = df_split(df)
    models.to_csv(os.path.join(dst_fld, "tsv", f"{category_id}_models.tsv"), sep='\t', header=False, index=False)
    models_id.to_csv(os.path.join(dst_fld, "tsv", f"{category_id}_models_id.tsv"), sep='\t', header=False, index=False)

def df_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into two separate DataFrames.

    Args:
        df (pd.DataFrame): Input DataFrame containing models information.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame containing model full names and DataFrame containing model IDs.
    """
    df1 = pd.DataFrame({'id': range(len(df)), 'full_name': df["full_name"]})
    df2 = pd.DataFrame({'id': range(len(df)), 'model_id': df["model_id"]})
    return df1, df2

def save_index(ckpt_fld: str, doc_maxlen: int, nbits: int, kmeans_niters: int, nranks: int, dst_fld: str, experiment: str, collection: Collection, index_name: str) -> Indexer:
    """
    Index the collection of documents and save the index for fast search of model that matches to given offer.

    Args:
        ckpt_fld (str): Path to the checkpoint folder.
        doc_maxlen (int): Maximum length of the document.
        nbits (int): Number of bits for the embedding.
        kmeans_niters (int): Number of iterations of k-means clustering; 4 is a good and fast default.
        nranks (int): Number of GPUs to use, if they are available.
        dst_fld (str): Destination folder where the index will be saved.
        experiment (str): Experiment name.
        collection (Collection): Collection of documents to index.
        index_name (str): Name of the index.

    Returns:
        Indexer: Object representing the index.
    """
    with Run().context(RunConfig(nranks=nranks, root=dst_fld, experiment=experiment)):
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=kmeans_niters)
        indexer = Indexer(checkpoint=ckpt_fld, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
    return indexer

def rankings_to_dict(rankings: List[Tuple[str, List[Tuple[int, float]]]], searcher: Searcher) -> List[Dict[str, Any]]:
    """
    Convert rankings object to a list of dictionaries.

    Args:
        rankings (List[Tuple[str, List[Tuple[int, float]]]]): Rankings object containing search results.
        searcher (Searcher): Searcher object used for searching.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing model IDs and their similarities.
    """
    result = []
    for key, value in rankings.todict().items():
        model_ids = [int(searcher.collection[item[0]]) for item in value]
        similarity = [item[2] for item in value]
        result.append({'model_ids': model_ids, 'similarity': similarity})
    return result

def top_n_similar(offers: Dict[int, str], src_fld: str, nranks: int, experiment: str, index_name: str, model_ids: List[str], n: int) -> List[Dict[str, Union[List[int], List[float]]]]:
    """
    Retrieve top N similar models for given offers.

    Args:
        offers (Dict[int, str]): Dictionary where keys are integer IDs and values are offer strings.
        src_fld (str): Source folder where the index is located.
        nranks (int): Number of GPUs to use, if they are available.
        experiment (str): Experiment name.
        index_name (str): Name of the index.
        model_ids (List[str]): List of model IDs.
        n (int): Number of similar models to retrieve.

    Returns:
        List[Dict[str, Union[List[int], List[float]]]]: List of dictionaries containing model IDs and their similarities.
    """
    with Run().context(RunConfig(nranks=nranks, root=src_fld, experiment=experiment)):
        searcher = Searcher(index=index_name, collection=model_ids)
        offers = Queries(data=offers)
        rankings = searcher.search_all(offers, k=n)
        top_n = rankings_to_dict(rankings, searcher)
    return top_n