"""
The file documents_train.tsv contains the training documents and has the following format:
0 text0
1 text1
2 text2
...
The file queries.tsv contains the queries and has the following format:
0 query0
1 query1
2 query2
...
The file triples.json contains the training triples and has the following format:
[0, 12692, 48199]
[0, 12693, 17664]
[1, 23783, 21169]
[1, 3796, 17104]
[1, 47897, 30084]
[2, 52241, 54085]
...
The first component is the id of the query, the second one the id of a relevant 
document and the third one the id of an irrelevant document.

"""

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer


"""
---------------------------------------------------------------------
TrainingSettings:

similarity: str = DefaultVal('cosine')
bsize: int = DefaultVal(32)
accumsteps: int = DefaultVal(1) - аккумуляция градиентов
lr: float = DefaultVal(3e-06)
maxsteps: int = DefaultVal(500_000) - число шагов обучения, линейного падения lr
save_every: int = DefaultVal(None)
resume: bool = DefaultVal(False)

## NEW:
warmup: int = DefaultVal(None) - через сколько шагов сделать warmup до изначального lr
warmup_bert: int = DefaultVal(None)
relu: bool = DefaultVal(False)
nway: int = DefaultVal(2) - 
use_ib_negatives: bool = DefaultVal(False)
reranker: bool = DefaultVal(False)
distillation_alpha: float = DefaultVal(1.0)
ignore_scores: bool = DefaultVal(False)
model_name: str = DefaultVal(None) # DefaultVal('bert-base-uncased')
---------------------------------------------------------------------

DocSettings:

dim: int = DefaultVal(128)
doc_maxlen: int = DefaultVal(220)
---------------------------------------------------------------------

nranks - число видеокарт
"""

checkpoint = 'colbert-ir/colbertv2.0'
if __name__=='__main__':
    triples="/home/sondors/Documents/price/BERT_data/data/18_categories/ColBERT_dataset/triples_10000.json"
    queries="/home/sondors/Documents/price/BERT_data/data/18_categories/ColBERT_dataset/queries_train.tsv"
    collection="/home/sondors/Documents/price/BERT_data/data/18_categories/ColBERT_dataset/documents_train.tsv"
    
    bsize=2
    lr=1e-06
    warmup=0
    doc_maxlen=180
    dim=128
    nway=2
    accumsteps=1
    use_ib_negatives=False
    save_every = None
    root="/home/sondors/Documents/1234567"   # не работает
    
    n_epochs = 2
    dataset_len = 10000                      # количество строк в triples.json
    steps_per_epoch = int(dataset_len/bsize) # количество батчей в эпохе
    maxsteps = n_epochs * steps_per_epoch    # количество батчей в обучении

    with Run().context(RunConfig(nranks=1, experiment="debug")):
        config = ColBERTConfig(bsize=bsize, 
                                lr=lr, 
                                warmup=warmup, 
                                doc_maxlen=doc_maxlen, 
                                dim=dim, 
                                nway=nway, 
                                accumsteps=accumsteps, 
                                use_ib_negatives=use_ib_negatives,
                                save_every=save_every,
                                root=root,
                                maxsteps=maxsteps)
        trainer = Trainer(
                                triples=triples,
                                queries=queries,
                                collection=collection,
                                config=config,)

        checkpoint_path = trainer.train(checkpoint=checkpoint)
        print(f"Saved checkpoint to {checkpoint_path}...")
