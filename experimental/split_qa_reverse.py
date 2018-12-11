# -*- coding: utf-8 -*-
# editor -- me
__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

import numpy as np

text = open('training_dialogue_plus_personas.txt', 'r')

q = open('context_dialog', 'w')
a = open('answers_personas', 'w')
person1=''
person2=''
person = ' '
previous_person=' '

l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']

personID = 2

for i, raw_word in enumerate(text):
    pos = raw_word.find('+++$+++')

    if pos > -1:
        person = raw_word[pos+7:pos+10]
        raw_word = raw_word[pos+8:]
    while pos > -1:
        pos = raw_word.find('+++$+++')
        raw_word = raw_word[pos+2:]
        
    raw_word = raw_word.replace('$+++','')
    previous_person = person

    for j, term in enumerate(l1):
        raw_word = raw_word.replace(term,l2[j])
        
    for term in l3:
        raw_word = raw_word.replace(term,' ')
    
    raw_word = raw_word.lower()

    
    if(raw_word[0] == '<'):
        a.write(raw_word[7:-1]+ '\n')
        if(person1 != '' and person2 != ''):
            q.write(person1 + '\n' + person2 + '\n')  # python will convert \n to os.linese
            personID = 2
            person1 = ''
            person2 = ''
    else: 
        if(raw_word[0] == '_'):
            personID = 1
        if(personID == 1):
            person1 += raw_word[:-1] + ' '
            personID = 2
        elif(personID == 2):
            person2 += raw_word[:-1] + ' ' 
            personID = 1

    

q.close()
a.close()
