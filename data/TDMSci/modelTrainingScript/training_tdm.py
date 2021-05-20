from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, CharacterEmbeddings, BertEmbeddings
from typing import List


# define columns
columns = {0: 'text', 1: 'pos', 2: 'ner'} 

# this is the folder in which train, test and dev files reside
data_folder = '../conllFormat/'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train_1500_v2.conll',
                              test_file='dev_150_v2.conll',
                              dev_file='test_500_v2.conll')



# 2. what tag do we want to predict?
tag_type = 'ner'



# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)


# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    BertEmbeddings('bert-base-cased'), 	
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('../model/flair/tdm-bert',
              learning_rate=0.1,
              mini_batch_size=16,
              max_epochs=150)i

