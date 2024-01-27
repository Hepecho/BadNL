# Lab 3.2: BadNL
## Introduction
this project is a reproduction code for [BadNL](https://arxiv.org/pdf/2006.01043v1.pdf), 
aimed at reproducing experiments on the IMDB dataset in the paper
## Usage
### Dataset
please download the [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) dataset in directory `./data/aclImdb`
### Example
Before conducting any other poisoning experiments, please prepare a wf.json file 
(a dictionary composed of words and frequency)
```
python src/load_dataset.py --action 0
```
if you'd like to test single word-level attack, firstly generate the poisoned dataset.
you need to set model.TextCNN.config.mode equals 'word', then run this:
```
python src/load_dataset.py --action 1
```
the poisoned data file will be saved in directory `./data/PoisonedIMDB/word/`

after that, run:
```
python src/main.py --action 1
```
The result will be output in the console and saved in directory `./log/word/`:

If you need to test other experiments, please check the `action` parameter in `./src/main.py` and `./src/load_dataset.py`