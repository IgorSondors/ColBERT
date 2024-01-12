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
    triples="/mnt/vdb1/Datasets/ColBERT/18_categories/train/triples_shuffle.json"
    queries="/mnt/vdb1/Datasets/ColBERT/18_categories/train/queries_train.tsv"
    collection="/mnt/vdb1/Datasets/ColBERT/18_categories/train/documents_train.tsv"
    # DocSettings:
    doc_maxlen=180
    dim=128
    
    # TrainingSettings:
    use_ib_negatives=False
    save_every = None
    root="/home/sondors/Documents/1234567"      # не работает

    nway=2                                      # https://github.com/stanford-futuredata/ColBERT/issues/245
    lr=1e-05
    bsize=128
    accumsteps=1                                # на сколько элементов из батча аккумулировать лосс
    n_triplets = sum(1 for _ in open(triples))  # количество строк в triples.json
    steps_per_epoch = int(n_triplets/bsize)     # количество батчей в эпохе. ColBERT обучается по всем строкам файла один раз без эпох
    warmup=steps_per_epoch//10                  # через сколько шагов сделать warmup до изначального lr

    with Run().context(RunConfig(nranks=1, experiment="HYPERPARAM_shuffle_warmup")): # nranks - число видеокарт
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
