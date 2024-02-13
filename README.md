### Dataset format

The file documents_train.tsv contains the training documents and has the following format:
```
0 text0
1 text1
2 text2
...
```

The file queries.tsv contains the queries and has the following format:
```
0 query0
1 query1
2 query2
...
```

The file triples.json contains the training triples and has the following format:
```
[0, 12692, 48199]
[0, 12693, 17664]
[1, 23783, 21169]
[1, 3796, 17104]
[1, 47897, 30084]
[2, 52241, 54085]
...
```

The first component is the id of the query, the second one the id of a relevant 
document and the third one the id of an irrelevant document.

### Venv

We have also included a new environment file specifically for CPU-only environments (conda_env_cpu.yml), but note that if you are testing CPU execution on a machine that includes GPUs you might need to specify os.environ["CUDA_VISIBLE_DEVICES"] = "-1" as part of your command. Note that a GPU is required for training and indexing.

```
conda env create -f conda_env[_cpu].yml
conda activate colbert
```

### Hyperparameters

There is no description yet, but all hyperparameters could be seen [there](https://github.com/IgorSondors/ColBERT/blob/main/colbert/infra/config/settings.py)

### Settings

To set cuda/cpu usage and stepts of ckpt to save go [there](https://github.com/IgorSondors/ColBERT/blob/main/colbert/parameters.py) 