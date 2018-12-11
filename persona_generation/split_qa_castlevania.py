# -*- coding: utf-8 -*-
# Modified by: Albert Enyedy and Brian Zylich

# prepares castlevania dataset for persona prediction

__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

import numpy as np

text = open('castlevania_symphony_dataset.txt', 'r', encoding="utf-8")

q = open('context_castlevania', 'w', encoding="utf-8")
a = open('characters_castlevania', 'w', encoding="utf-8")
person1=''
person2=''
line_number = 0
person = ' '
previous_person=' '
# wordlimit = 120 don't need because truncates in next file

l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']

first = True
previous_scriptline = 0
previous_movie = ''
previous_person = ''
personID = True # true is person 1 false is person 2

personas = {}

for line in (list(text)):
    #print(line.rstrip())
    raw_word = line
    parts = raw_word.split(': ')

    person = parts[0]
    if person not in personas:
        personas[person] = ""
    raw_word = parts[1].strip()
    
    raw_word = raw_word.lower()

    for j, term in enumerate(l1):
        raw_word = raw_word.replace(term,l2[j])
        
    for term in l3:
        raw_word = raw_word.replace(term,' ')

    personas[person] += raw_word + ' '

for p, text in personas.items():
    a.write(p + '\n')
    q.write(text + '\n')
    

q.close()
a.close()
