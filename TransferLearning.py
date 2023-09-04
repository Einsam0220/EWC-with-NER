import torch
import flair
import random
import os
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from torch.optim import SGD
from flair.datasets import DataLoader
from flair.models import SequenceTagger
from flair.embeddings import CharacterEmbeddings, WordEmbeddings, StackedEmbeddings, TokenEmbeddings, PooledFlairEmbeddings, RoBERTaEmbeddings, BertEmbeddings
from typing import List
from flair.data import MultiCorpus, Corpus

def train_process(HosA_corpus: Corpus, HosB_corpus: Corpus, model_path: str, save_model_name: str, ewc_bool: bool):
    torch.cuda.empty_cache()
    random.seed(10)

    HosB_tag_dictionary = HosB_corpus.make_tag_dictionary(tag_type='ner')

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
        RoBERTaEmbeddings()
    ]
    HosB_embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256, embeddings=HosB_embeddings, tag_dictionary=HosB_tag_dictionary, 
                                            tag_type='ner', use_crf=True)

    parameters = dict(tagger.named_parameters())

    pre_trained_tagger: SequenceTagger = SequenceTagger.load(model_path)

    tagger.load_state_dict(parameters,strict = False)

    trainer: ModelTrainer = ModelTrainer(tagger, Pre_corpus=HosA_corpus, corpus=HosB_corpus, optimizer=SGD)

    save_model_path = f'trained_model_path'

    if ewc_bool == True:
        trainer.ewc_train(save_model_path,
                  mini_batch_size=256, max_epochs=150, embeddings_storage_mode='gpu', checkpoint=True,
                  learning_rate=0.1, momentum = 0.9, weight_decay=1e-5,importance=400)
    else:
        trainer.train(save_model_path,
                  mini_batch_size=256, max_epochs=150, embeddings_storage_mode='gpu', checkpoint=True,
                  learning_rate=0.1, momentum = 0.9, weight_decay=1e-5)
    
    tagger: SequenceTagger = SequenceTagger.load('your_best_model')

    Test_DataLoader = DataLoader(HosA_corpus.test, batch_size=256)

    predict_txt_path = "your_target_path"
    
    result, score = tagger.evaluate(Test_DataLoader.dataset, out_path= predict_txt_path + "predictions.txt")

    with open(predict_txt_path + 'Score.txt', 'w') as f:
        f.write('micro: \n'+ result.log_header +'\n'+result.log_line+'\n')
        f.write(result.detailed_results)
############### EXAMPLE ###############
columns = {0: 'text', 1: 'ner'}
data_folder = './'

HosA_corpus: Corpus = ColumnCorpus(data_folder, columns, train_file='train_file', 
                              test_file='test_file',
                              dev_file='dev_file')

HosB_corpus: Corpus = ColumnCorpus(data_folder, columns, train_file='train_file', 
                              test_file='test_file',
                              dev_file='dev_file')

model_name = f'SourceDomain_best_model'

save_model_name = 'TargetDomain_model'

train_process(HosA_corpus=HosA_corpus, HosB_corpus=HosB_corpus, model_path=model_name, save_model_name=save_model_name, ewc_bool=False)
