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

checkpoint = 'bert-base-multilingual-cased'
# checkpoint = 'colbert-ir/colbertv2.0'
if __name__=='__main__':
    # triples="/mnt/vdb1/Datasets/ColBERT_data/13_categories/train_aug/triples_X1_13_categories_aug_shuffle.json"
    # queries="/mnt/vdb1/Datasets/ColBERT_data/13_categories/train_aug/queries_train_13_categories_aug.tsv"
    # collection="/mnt/vdb1/Datasets/ColBERT_data/13_categories/train_aug/documents_train_13_categories_aug.tsv"

    # triples="/mnt/vdb1/Datasets/ColBERT_data/13_categories/train_nway_6/triples_X1_13_categories_aug_nway_6_shuffle.json"
    # queries="/mnt/vdb1/Datasets/ColBERT_data/13_categories/train_nway_6/queries_train_13_categories_aug_nway_6.tsv"
    # collection="/mnt/vdb1/Datasets/ColBERT_data/13_categories/train_nway_6/documents_train_13_categories_aug_nway_6.tsv"

    triples="/mnt/vdb1/Datasets/ColBERT_data/13_categories/train/triples_X1_13_categories_shuffle.json"
    queries="/mnt/vdb1/Datasets/ColBERT_data/13_categories/train/queries_train_13_categories.tsv"
    collection="/mnt/vdb1/Datasets/ColBERT_data/13_categories/train/documents_train_13_categories.tsv"

    # DocSettings:
    doc_maxlen=180
    dim=768#128
    
    # TrainingSettings:
    use_ib_negatives=True
    save_every = None
    root="/home/sondors/Documents/1234567"      # не работает

    nway=2#2                                    # https://github.com/stanford-futuredata/ColBERT/issues/245
    lr=1e-04
    bsize=230#128#40
    accumsteps=1                                # на сколько элементов из батча аккумулировать лосс
    n_triplets = sum(1 for _ in open(triples))  # количество строк в triples.json
    steps_per_epoch = int(n_triplets/bsize)     # количество батчей в эпохе. ColBERT обучается по всем строкам файла один раз без эпох
    warmup=0                                    # через сколько шагов сделать warmup до изначального lr

    experiment = "bert-base-multilingual-cased_dim_768_bsize_230_lr04_use_ib_negatives"
    with Run().context(RunConfig(nranks=1, experiment=experiment)): # nranks - число видеокарт
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
