# -*- coding: utf-8 -*-

__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import Adam 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
from keras.layers import concatenate

import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility
import pickle
import os.path
import sys
import nltk
import re
import time
import random
import copy

from keras.utils import plot_model

word_embedding_size = 200
sentence_embedding_size = 300
topic_embedding_size = 50
persona_embedding_size = 100
dictionary_size = 8000
maxlen_input = 50

vocabulary_file = 'vocabulary_personas_clustered'
weights_file = 'my_model_weights20_personas_clustered.h5'
unknown_token = 'something'
file_saved_context = 'saved_context'
file_saved_answer = 'saved_answer'
name_of_computer = 'john'

def greedy_decoder(question, persona):
    # question = np.reshape(question, (maxlen_input,))
    # print(np.shape(question), np.shape(persona))

    flag = 0
    prob = 1
    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = vocab_words.index("BOS")  #  the index of the symbol BOS (begin of sentence)
    for k in range(maxlen_input - 1):
        ye = model.predict([question, persona, ans_partial])
        # print(np.shape(ye))
        yel = ye[0,:]
        # print(yel)
        p = np.max(yel)
        mp = np.argmax(ye)
        ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, -1] = mp
        if mp == vocab_words.index("EOS"):  #  he index of the symbol EOS (end of sentence)
            flag = 1
        if flag == 0:    
            prob = prob * p
    text = ''
    for k in ans_partial[0]:
        k = k.astype(int)
        if k < (dictionary_size-2):
            w = vocabulary[k]
            text = text + w[0] + ' '
    return(text, prob)
    
def beam_decoder(question, persona, k):
    # flag = 0
    # prob = 1
    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = vocab_words.index("BOS")  #  the index of the symbol BOS (begin of sentence)
    candidates = [ans_partial]
    candidate_probabilities = [1]
    candidates_finished = [False]
    for x in range(maxlen_input - 1):
        new_candidates = []
        new_candidate_probabilities = []
        new_candidates_finished = []

        for c in range(len(candidates)):
            if not candidates_finished[c]:
                ye = model.predict([question, persona, candidates[c]])
                yel = ye[0,:]
                # p = np.max(yel)

                word_indices = np.argpartition(yel, len(yel) - k)[len(yel) - k:]
                word_probabilities = yel[word_indices]
                # print(word_indices)
                # print(word_probabilities)
                
                # mp = np.argmax(ye)
                for i in range(len(word_indices)):
                    word_index = word_indices[i]
                    word_probability = word_probabilities[i]

                    new_candidate = copy.deepcopy(candidates[c])
                    new_candidate[0, 0:-1] = new_candidate[0, 1:]
                    new_candidate[0, -1] = word_index
                    new_candidate_finished = (word_index == vocab_words.index("EOS"))
                    new_candidate_probability = candidate_probabilities[c] * word_probability
                    new_candidates.append(new_candidate)
                    new_candidate_probabilities.append(new_candidate_probability)
                    new_candidates_finished.append(new_candidate_finished)
            else:
                new_candidates.append(candidates[c])
                new_candidate_probabilities.append(candidate_probabilities[c])
                new_candidates_finished.append(candidates_finished[c])
        new_candidates = np.array(new_candidates)
        new_candidate_probabilities = np.array(new_candidate_probabilities)
        new_candidates_finished = np.array(new_candidates_finished)

        new_candidate_indices = np.argpartition(new_candidate_probabilities, len(new_candidate_probabilities) - k)[len(new_candidate_probabilities) - k:]
        # print(new_candidate_indices)
        candidates = new_candidates[new_candidate_indices]
        candidate_probabilities = new_candidate_probabilities[new_candidate_indices]
        candidates_finished = new_candidates_finished[new_candidate_indices]

    # for c in candidates:
    #     text = ''
    #     for x in c[0]:
    #         x = x.astype(int)
    #         if x < (dictionary_size-2):
    #             w = vocabulary[x]
    #             text = text + w[0] + ' '
    #     print(text)

    max_index = np.argmax(candidate_probabilities)
    final_cand = candidates[max_index]
    final_prob = candidate_probabilities[max_index]
    text = ''
    for x in final_cand[0]:
        x = x.astype(int)
        if x < (dictionary_size-2):
            w = vocabulary[x]
            text = text + w[0] + ' '
    return(text, final_prob)
    
def dbs_decoder(question, persona, b, g, diverse):
    b_prime = int(b / g) # pick g to divide b
    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = vocab_words.index("BOS")  #  the index of the symbol BOS (begin of sentence)
    group_candidates = {i:[ans_partial] for i in range(g)}
    group_candidate_probabilities = {i:[1] for i in range(g)}
    group_candidates_finished = {i:[False] for i in range(g)}
    for t in range(maxlen_input - 1):
        diversity_counts = np.zeros(dictionary_size)

        # one step of beam search for first group (without diversity)
        new_candidates, new_candidate_probabilities, new_candidates_finished = beam_search_step(question, persona, group_candidates[0], 
            group_candidate_probabilities[0], group_candidates_finished[0], b_prime)
        new_candidate_indices = np.argpartition(new_candidate_probabilities, len(new_candidate_probabilities) - b_prime)[len(new_candidate_probabilities) - b_prime:]
        # print(new_candidate_indices)
        group_candidates[0] = new_candidates[new_candidate_indices]
        for gc in group_candidates[0]:
            # print(gc[0, -1])
            diversity_counts[int(gc[0, -1])] += 1
        group_candidate_probabilities[0] = new_candidate_probabilities[new_candidate_indices]
        group_candidates_finished[0] = new_candidates_finished[new_candidate_indices]

        # run other groups using diversity function
        for group in range(1, g):
            new_candidates, new_candidate_probabilities, new_candidates_finished = beam_search_step(question, persona, group_candidates[group], 
                group_candidate_probabilities[group], group_candidates_finished[group], b_prime, diversity_counts=diversity_counts, diversity=diverse)
            new_candidate_indices = np.argpartition(new_candidate_probabilities, len(new_candidate_probabilities) - b_prime)[len(new_candidate_probabilities) - b_prime:]
            # print(new_candidate_indices)
            group_candidates[group] = new_candidates[new_candidate_indices]
            for gc in group_candidates[group]:
                diversity_counts[int(gc[0, -1])] += 1
            group_candidate_probabilities[group] = new_candidate_probabilities[new_candidate_indices]
            group_candidates_finished[group] = new_candidates_finished[new_candidate_indices]

    all_candidates = []
    all_probabilities = []
    all_finished = []
    for group in range(g):
        all_candidates.extend(group_candidates[group])
        all_probabilities.extend(group_candidate_probabilities[group])
        all_finished.extend(group_candidates_finished[group])
        print("group", group)
        for c in group_candidates[group]:
            text = ''
            for x in c[0]:
                x = x.astype(int)
                if x < (dictionary_size-2):
                    w = vocabulary[x]
                    text = text + w[0] + ' '
            print(text)

    max_index = np.argmax(all_probabilities)
    final_cand = all_candidates[max_index]
    final_prob = all_probabilities[max_index]
    text = ''
    for x in final_cand[0]:
        x = x.astype(int)
        if x < (dictionary_size-2):
            w = vocabulary[x]
            text = text + w[0] + ' '
    return(text, final_prob)

def beam_search_step(question, persona, candidates, candidate_probabilities, candidates_finished, b_prime, diversity_counts=None, diversity=False):
    new_candidates = []
    new_candidate_probabilities = []
    new_candidates_finished = []
    for c in range(len(candidates)):
        if not candidates_finished[c]:
            ye = model.predict([question, persona, candidates[c]])
            yel = ye[0,:]

            if diversity_counts is not None:
                yel -= diversity * diversity_counts

            word_indices = np.argpartition(yel, len(yel) - b_prime)[len(yel) - b_prime:]
            word_probabilities = yel[word_indices]
            
            for i in range(len(word_indices)):
                word_index = word_indices[i]
                word_probability = word_probabilities[i]

                new_candidate = copy.deepcopy(candidates[c])
                new_candidate[0, 0:-1] = new_candidate[0, 1:]
                new_candidate[0, -1] = word_index
                new_candidate_finished = (word_index == vocab_words.index("EOS"))
                new_candidate_probability = candidate_probabilities[c] * word_probability
                new_candidates.append(new_candidate)
                new_candidate_probabilities.append(new_candidate_probability)
                new_candidates_finished.append(new_candidate_finished)
        else:
            new_candidates.append(candidates[c])
            new_candidate_probabilities.append(candidate_probabilities[c])
            new_candidates_finished.append(candidates_finished[c])
    new_candidates = np.array(new_candidates)
    new_candidate_probabilities = np.array(new_candidate_probabilities)
    new_candidates_finished = np.array(new_candidates_finished)
    return new_candidates, new_candidate_probabilities, new_candidates_finished

def preprocess(raw_word, name):
    
    l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
    l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
    l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
    l4 = ['jeffrey','fred','benjamin','paula','walter','rachel','andy','helen','harrington','kathy','ronnie','carl','annie','cole','ike','milo','cole','rick','johnny','loretta','cornelius','claire','romeo','casey','johnson','rudy','stanzi','cosgrove','wolfi','kevin','paulie','cindy','paulie','enzo','mikey','i\97','davis','jeffrey','norman','johnson','dolores','tom','brian','bruce','john','laurie','stella','dignan','elaine','jack','christ','george','frank','mary','amon','david','tom','joe','paul','sam','charlie','bob','marry','walter','james','jimmy','michael','rose','jim','peter','nick','eddie','johnny','jake','ted','mike','billy','louis','ed','jerry','alex','charles','tommy','bobby','betty','sid','dave','jeffrey','jeff','marty','richard','otis','gale','fred','bill','jones','smith','mickey']    

    raw_word = raw_word.lower()
    # raw_word = raw_word.replace(', ' + name_of_computer, '')
    # raw_word = raw_word.replace(name_of_computer + ' ,', '')

    for j, term in enumerate(l1):
        raw_word = raw_word.replace(term,l2[j])
        
    for term in l3:
        raw_word = raw_word.replace(term,' ')
    
    # for term in l4:
    #     raw_word = raw_word.replace(', ' + term, ', ' + name)
    #     raw_word = raw_word.replace(' ' + term + ' ,' ,' ' + name + ' ,')
    #     raw_word = raw_word.replace('i am ' + term, 'i am ' + name_of_computer)
    #     raw_word = raw_word.replace('my name is' + term, 'my name is ' + name_of_computer)
    
    # for j in range(30):
    #     raw_word = raw_word.replace('. .', '')
    #     raw_word = raw_word.replace('.  .', '')
    #     raw_word = raw_word.replace('..', '')
       
    # for j in range(5):
    #     raw_word = raw_word.replace('  ', ' ')
        
    # if raw_word[-1] !=  '!' and raw_word[-1] != '?' and raw_word[-1] != '.' and raw_word[-2:] !=  '! ' and raw_word[-2:] != '? ' and raw_word[-2:] != '. ':
    #     raw_word = raw_word + ' .'
    
    # if raw_word == ' !' or raw_word == ' ?' or raw_word == ' .' or raw_word == ' ! ' or raw_word == ' ? ' or raw_word == ' . ':
    #     raw_word = 'what ?'
    
    # if raw_word == '  .' or raw_word == ' .' or raw_word == '  . ':
    #     raw_word = 'i do not want to talk about it .'
      
    return raw_word

def tokenize(sentences):

    # Tokenizing the sentences into words:
    tokenized_sentences = nltk.word_tokenize(sentences)
    index_to_word = [x[0] for x in vocabulary]
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    new_tokenized_sentences = []
    for w in tokenized_sentences:
        if w in word_to_index:
            new_tokenized_sentences.append(w)
        else:
            print("(", w, "not in vocabulary)")
            new_tokenized_sentences.append(unknown_token)
    # tokenized_sentences = [w if w in word_to_index else unknown_token for w in tokenized_sentences]
    X = np.asarray([word_to_index[w] for w in new_tokenized_sentences])
    s = X.size
    Q = np.zeros((1,maxlen_input))
    if s < (maxlen_input + 1):
        Q[0,- s:] = X
    else:
        Q[0,:] = X[- maxlen_input:]
    
    return Q

 # Open files to save the conversation for further training:
qf = open(file_saved_context, 'w')
af = open(file_saved_answer, 'w')

print('Starting the model...')

# *******************************************************************
# Keras model of the chatbot: 
# *******************************************************************

ad = Adam(lr=0.00005) 

input_context = Input(shape=(maxlen_input,), dtype='int32')#, name='the context text')
input_persona = Input(shape=(maxlen_input,), dtype='int32')
input_answer = Input(shape=(maxlen_input,), dtype='int32')#, name='the answer text up to the current token')
LSTM_encoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform')#, name='Encode context')
LSTM_encoder_persona = LSTM(persona_embedding_size, kernel_initializer= 'lecun_uniform')
LSTM_decoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform')#, name='Encode answer up to the current token')
if os.path.isfile(weights_file):
    Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input)#, name='Shared')
else:
    Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, weights=[embedding_matrix], input_length=maxlen_input)#, name='Shared')
word_embedding_context = Shared_Embedding(input_context)
context_embedding = LSTM_encoder(word_embedding_context)

word_embedding_answer = Shared_Embedding(input_answer)
answer_embedding = LSTM_decoder(word_embedding_answer)

word_embedding_persona = Shared_Embedding(input_persona)
persona_embedding = LSTM_encoder_persona(word_embedding_persona)

# LSTM_encoder_topic = LSTM(topic_embedding_size, kernel_initializer='lecun_uniform')
LSTM_encoder_topic = Dense(topic_embedding_size, activation="relu")
topic_embedding = LSTM_encoder_topic(context_embedding)

merge_layer = concatenate([topic_embedding, context_embedding, persona_embedding, answer_embedding], axis=1)#, name='concatenate the embeddings of the context and the answer up to current token')
out = Dense(int(dictionary_size/2), activation="relu")(merge_layer)#, name='relu activation')(merge_layer)
out = Dense(dictionary_size, activation="softmax")(out)#, name='likelihood of the current token using softmax activation')(out)

model = Model(inputs=[input_context, input_persona, input_answer], outputs = [out])

model.compile(loss='categorical_crossentropy', optimizer=ad)

# plot_model(model, to_file='model_graph.png')    

if os.path.isfile(weights_file):
    model.load_weights(weights_file)


# Loading the data:
vocabulary = pickle.load(open(vocabulary_file, 'rb'))
vocab_words = list(map(lambda l: l[0], vocabulary))

print("\n \n \n \n    CHAT:     \n \n")


# select persona
persona_pool = {}
selected_persona = None

# read in possible personas
with open("personas") as persona_file:
    count = 0
    for line in persona_file:
        persona_pool[line.strip()] = count
        count += 1

# randomly pick personas from file
persona_selections = random.sample(list(persona_pool.items()), 3)

count = 0
for p, p_id in persona_selections:
    print(count, ":", p)
    count += 1

selected_persona = int(input("Which persona do you want to talk with?"))
selected_persona = persona_selections[selected_persona]


with open("Padded_personas_clustered", "rb") as padded_persona_file:
    padded_personas = pickle.load(padded_persona_file)
    selected_persona = np.array([padded_personas[selected_persona[1]]])

text = ''
for k in list(list(selected_persona)[0]):
    k = int(k)
    if k < (dictionary_size-2):
        w = vocabulary[k]
        text = text + w[0] + ' '
print(text)

# Processing the user query:
prob = 0
que = ''
last_query  = ' '
last_last_query = ''
text = ' '
last_text = ''
# print('computer: hi ! please type your name.\n')
name = ""
# print('computer: hi , ' + name +' ! My name is ' + name_of_computer + '.\n') 


while que != 'goodbye .':
    
    que = input('user: ')
    que = preprocess(que, name_of_computer)
    # Collecting data for training:
    q = last_query + ' ' + text
    a = que
    qf.write(q + '\n')
    af.write(a + '\n')
    # Composing the context:
    if prob > 0.001:
        query = text + ' ' + que
    else:    
        query = que
   
    last_text = text
    
    Q = tokenize(query)
    
    # Using the trained model to predict the answer:
    predout, prob = beam_decoder(Q[0:1], selected_persona, 10)
    response = predout.strip().split('EOS')[0]
    response = response.split('BOS')[1]
    # print(response)
    text = preprocess(response, name)
    print ('computer (beam): ' + text + '    (with probability of %f)'%prob)

    predout_dbs, prob_dbs = dbs_decoder(Q[0:1], selected_persona, 10, 10, 0.4)
    response_dbs = predout_dbs.strip().split('EOS')[0]
    response_dbs = response_dbs.split('BOS')[1]
    # print(response)
    text_dbs = preprocess(response_dbs, name)
    print ('computer (dbs): ' + text_dbs + '    (with probability of %f)'%prob_dbs)

    predout_greedy, prob_greedy = greedy_decoder(Q[0:1], selected_persona)
    response_greedy = predout_greedy.strip().split('EOS')[0]
    response_greedy = response_greedy.split('BOS')[1]
    text_greedy = preprocess(response_greedy, name)
    print ('computer (greedy): ' + text_greedy + '    (with probability of %f)'%prob_greedy)
    
    last_last_query = last_query    
    last_query = que

qf.close()
af.close()
