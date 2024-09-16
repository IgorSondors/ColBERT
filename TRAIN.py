from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

"""
---------------------------------------------------------------------
RunSettings:

overwrite: Флаг, указывающий, можно ли перезаписывать результаты эксперимента.
root: Корневая директория, где будут храниться результаты экспериментов.
experiment: Имя текущего эксперимента.
index_root: Директория для хранения индексов.
name: Уникальное имя эксперимента, обычно содержащее временную метку.
rank, nranks: Информация о ранге и общем количестве доступных графических процессоров (GPU).
amp: Использование автоматического масштабирования для обучения на GPU.
---------------------------------------------------------------------
TokenizerSettings:

Настройки токенизатора, такие как идентификаторы и токены для запросов и документов.
ResourceSettings:

checkpoint, triples, collection, queries: Пути к различным ресурсам, таким как веса модели, тройки данных, коллекция документов и запросы.
index_name: Название индекса.
---------------------------------------------------------------------
DocSettings:

dim: Размерность векторного представления документа.
doc_maxlen: Максимальная длина документа в токенах.
mask_punctuation: Флаг, указывающий, следует ли маскировать пунктуацию.
---------------------------------------------------------------------
QuerySettings:

query_maxlen: Максимальная длина запроса в токенах.
attend_to_mask_tokens: Флаг, указывающий, следует ли учитывать маскированные токены в запросе.
interaction: Тип взаимодействия (например, "colbert").
---------------------------------------------------------------------
TrainingSettings:

Различные параметры, связанные с процессом обучения, такие как выбор функции сходства, размер пакета, шаги аккумуляции градиента и тд.
model_name: Название модели (например, "bert-base-uncased").

use_ib_negatives - использование отрицательных примеров (negatives) в процессе обучения ColBERT
---------------------------------------------------------------------
IndexingSettings:

index_path: Путь к индексу.
nbits: Количество битов для индексации.
kmeans_niters: Количество итераций для k-means при индексации.
resume: Флаг, указывающий, следует ли возобновлять индексацию.
---------------------------------------------------------------------
SearchSettings:

ncells: Количество ячеек при поиске.
centroid_score_threshold: Порог оценки для центроидов.
ndocs: Количество документов для возвращения при поиске.
load_index_with_mmap: Флаг, указывающий, следует ли загружать индекс с использованием mmap.
---------------------------------------------------------------------
"""

# checkpoint = 'bert-base-multilingual-cased'
checkpoint = 'colbert-ir/colbertv2.0'

if __name__=='__main__':

    triples="/home/sondors.igor/documents/ColBERT/DATASET/train_triplets/15mln/triples_X1_v0_shuffle.json"
    queries="/home/sondors.igor/documents/ColBERT/DATASET/train_triplets/15mln/queries_train_v0.tsv"
    collection="/home/sondors.igor/documents/ColBERT/DATASET/train_triplets/15mln/documents_train_v0.tsv"

    # DocSettings:
    doc_maxlen=300#180
    # dim=768#128
    dim= 128
    # TrainingSettings:
    use_ib_negatives=True
    save_every = None
    root="/home/sondors.igor/documents/ColBERT"      # не работает
    nway=2                                    # https://github.com/stanford-futuredata/ColBERT/issues/245
    lr=1e-04
    bsize=50#128#40
    accumsteps=1                                # на сколько элементов из батча аккумулировать лосс
    amp = True                                  # MixedPrecisionManager
    n_triplets = sum(1 for _ in open(triples))  # количество строк в triples.json
    steps_per_epoch = int(n_triplets/bsize)     # количество батчей в эпохе. ColBERT обучается по всем строкам файла один раз без эпох
    warmup=0                                    # через сколько шагов сделать warmup до изначального lr

    experiment = "colbertv2_lr04_bs_50_grad_clip_doc_maxlen"
    with Run().context(RunConfig(nranks=1, experiment=experiment)): # nranks - число видеокарт
        config = ColBERTConfig(bsize=bsize, 
                                lr=lr, 
                                warmup=warmup, 
                                doc_maxlen=doc_maxlen, 
                                dim=dim, 
                                nway=nway, 
                                accumsteps=accumsteps, 
                                amp=amp,
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

# config = ColBERTConfig(bsize=32, lr=1e-05, warmup=20_000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False, nway=64, accumsteps=1, similarity='cosine', use_ib_negatives=True)
# trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

# trainer.train(checkpoint='colbert-ir/colbertv1.9') 