import os
import sys
sys.path.insert(0, '../')

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
import json
import time


def search(checkpoint, offers, models, nbits, doc_maxlen):
    index_name = f'models.18_categories.{nbits}bits'

    with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=models, overwrite=True)
    indexer.get_index() # You can get the absolute path of the index, if needed.

    with Run().context(RunConfig(experiment='notebook')):
        searcher = Searcher(index=index_name)

    start_time = time.time()
    rankings = searcher.search_all(offers, k=5).todict()
    print(f"time_spent = {time.time() - start_time}\n")
    return rankings

if __name__=='__main__':
    models = '/mnt/vdb1/Datasets/ColBERT/18_categories/test/models.tsv'
    offers = '/mnt/vdb1/Datasets/ColBERT/18_categories/test/offers.tsv'
    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300

    offers = Queries(path=offers)
    models = Collection(path=models)
    f'Loaded {len(offers)} queries and {len(models):,} passages'

    ckpts_pth = "/mnt/vdb1/ColBERT/experiments/HYPERPARAM_accum/none/2024-01/10/10.44.49/checkpoints"
    for checkpoint in os.listdir(ckpts_pth):
        ckpt_pth = os.path.join(ckpts_pth, checkpoint)
        print(ckpt_pth)
        
        rankings = search(ckpt_pth, offers, models, nbits, doc_maxlen)
        with open(f'/mnt/vdb1/Datasets/ColBERT/18_categories/metrics_data/EVAL/triplets_accum/triples_accum_{checkpoint}.json', 'w') as fp:
            json.dump(rankings, fp)
