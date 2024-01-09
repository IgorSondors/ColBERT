from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

"""
---------------------------------------------------------------------
TrainingSettings: Что они значат???

similarity: str = DefaultVal('cosine')
save_every: int = DefaultVal(None)
resume: bool = DefaultVal(False)

## NEW:
relu: bool = DefaultVal(False)
use_ib_negatives: bool = DefaultVal(False)
reranker: bool = DefaultVal(False)
distillation_alpha: float = DefaultVal(1.0)
ignore_scores: bool = DefaultVal(False)
model_name: str = DefaultVal(None) # DefaultVal('bert-base-uncased')
"""

checkpoint = 'colbert-ir/colbertv2.0'
if __name__=='__main__':
    triples="/home/sondors/Documents/price/BERT_data/data/18_categories/ColBERT_dataset/triples_20_shufle.json"
    queries="/home/sondors/Documents/price/BERT_data/data/18_categories/ColBERT_dataset/queries_train.tsv"
    collection="/home/sondors/Documents/price/BERT_data/data/18_categories/ColBERT_dataset/documents_train.tsv"
    
    # DocSettings:
    doc_maxlen=180
    dim=128
    
    # TrainingSettings:
    use_ib_negatives=False
    save_every = None
    root="/home/sondors/Documents/1234567"      # не работает

    nway=2                                      # https://github.com/stanford-futuredata/ColBERT/issues/245
    lr=1e-05
    warmup=0                                    # через сколько шагов сделать warmup до изначального lr
    bsize=1
    accumsteps=bsize                            # на сколько элементов из батча аккумулировать лосс
    n_triplets = sum(1 for _ in open(triples))  # количество строк в triples.json
    steps_per_epoch = int(n_triplets/bsize)     # количество батчей в эпохе. ColBERT обучается по всем строкам файла один раз без эпох

    with Run().context(RunConfig(nranks=1, experiment="debug")): # nranks - число видеокарт
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
                                maxsteps=steps_per_epoch)
        trainer = Trainer(
                                triples=triples,
                                queries=queries,
                                collection=collection,
                                config=config,)

        checkpoint_path = trainer.train(checkpoint=checkpoint)
        print(f"Saved checkpoint to {checkpoint_path}...")
