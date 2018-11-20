from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)


reader = Seq2SeqDatasetReader()
train_dataset = reader.read('/home/6/18M31289/groupdisk/matsumaru/data/wmt14_ende_sp/train.ende')
validation_dataset = reader.read('/home/6/18M31289/groupdisk/matsumaru/data/wmt14_ende_sp/valid.ende')
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
EMBEDDING_DIM = 400
HIDDEN_DIM = 400
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = SimpleSeq2Seq(vocab, word_embeddings, lstm)
optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)
trainer.train()
logits = model.forward()
ids = np.argmax(logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'tokens') for i in ids])