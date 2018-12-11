Overview:
This repository contains the code for our RBE595: Deep Learning for Robotic Perception final project.
In this project, we investigate solutions to several problems with state-of-the-art open-domain
dialogue systems. Namely, we strive to increase the consistency of dialogue agents' personas, reducing
the likelihood of contradictions in their responses. To do this, we add the agent's persona as an additional 
input to our deep learning model, using Facebook's Persona-Chat dataset (https://github.com/DeepPavlov/convai) 
as the foundation of our training data. Further, we examine methods for decoding more
interesting responses that stimulate conversation, even when tasked with answering a prompt that
has not been seen before in the training data. We compare greedy decoding, beam search, and Diverse
Beam Search (https://arxiv.org/abs/1610.02424). Lastly, we propose a new method for expanding dialogue system's 
vocabularies without increasing model complexity or training time. This method applies the k-means algorithm to 
GloVe word embeddings in order to cluster words semantically and create a mapping from a significantly larger 
vocabulary down to a smaller synonymous vocabulary that can be used to train models on setups with less resources 
and in a shorter amount of time.

Authors:
Albert Enyedy and Brian Zylich

Contents:
/data/ - Contains our processed versions of data from Persona-Chat and Cornell Movie-Dialogs Corpus, as well as
scripts for processing them.

/persona/ - Contains the files needed to generate our LSTM-Persona model that adds a persona as an additional
input when predicting responses for the dialogue agent.

/persona_generation/ - Explores the prediction of personas from a person/character's dialogue lines. Uses a similar
approach to the LSTM model without personas. The goal is to be able to generate personas for datasets other than
Persona-Chat, so that they can be used as training data for our LSTM-persona model. 

/practice/ - Scripts used to understand Persona-Chat dataset and Keras library.

/topic/ - Builds on basic LSTM sequence-to-sequence model, adding a dense layer on top of the LSTM encoder layer
with the motivation that this might allow the model to better capture topic or phrase information.

/vanilla/ - The original model from Oswaldo Ludwig (https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras)
applied to the Persona-Chat dataset.

/vocab_clustering/ - Uses k-means to cluster GloVe word embeddings to create a vocabulary mapping for use as a
preprocessing step before training and testing of models. Allows a model to "understand" many more words than 
are actually in its vocabulary.

References:
Much of our code is built on top of the framework developed by Oswaldo Ludwig 
(https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras). In addition, we use the
Persona-Chat (https://github.com/DeepPavlov/convai) and Cornell Movie-Dialogs Corpus
(https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).