# Spacy Setup
```
pip install -U pip setuptools wheel
pip install spacy

python -m spacy download en_core_web_sm
```
# Feature-based attention: The Key, Value, and Query

* The main difference between attention and retrieval systems is that we introduce a more abstract and smooth notion of 'retrieving' an object.
* By defining a degree of similarity between our representations, we can weight our query.
* Instead of choosing where to look according to the position within a sequence, we now attend to the content that we wanna look at!
* We use the keys to define the attention weights to look at the data and the values as the information we will actually get.
* We can associate the similarity between vectors that represent anything by calculating the scaled dot product, namely the cosine of the angle.

# Self-Attention: The Transformer Encoder
* Self-attention is an attention mechanism to compute a representation of the sequence relating different positions of a single sequence.
* Self-attention enables us to find correlations between different words of the input indicating the syntactic and contextual structure of the sentence.

# BERT
* BERT is a method of pretraining language representations that was used to create models that NLP practicioners can download and use for free.
* You can either use these models to extract high quality language features from your data, or you can fine-tune these models on a specific task.

* In this tutuorial, we will use BERT to extract features, namely word and sentence embedding vectors, from text data.
* These embeddings are useful for keyword/search expansion, semantic search and information retrieval.
* For example, if you want to match customer questions against already answered questions, these representations will hlep you accurately retrieve results matching the customer's intent and contextual meaning.
* Second and most importantly, these vectors are used as high-quality feature inputs to downstream models. BERT offers an advantage over models like Word2Vec, regardless of the context within which the word appears, BERT produces word representations that are dynamically informed by the words around them.
* Aside from capturing obvious differences like polysemy, the context-informed word embeddings capture other forms of information that result in more accurate feature representations, ehich in turn result in better model performance.
```
import torch
from transformers import BertTokenizer, BertModel
import logging 
import matplotlib.pyplot as plt

model = BertModel.from_pretrained('bert-base'uncased', output_hidden_states=True,)
model.eval()


with torch.no_grad():
outputs = model(tokens_tensor, segments_tensors)
```
* In brief, the training is done by masking a few owrds in a sentence and tasking the model to predict the masked words. And as the model trains to predict, it learns to produce a powerful internal representation of words as word embedding.


