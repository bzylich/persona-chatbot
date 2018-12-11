# -*- coding: utf-8 -*-
# Modified by: Albert Enyedy and Brian Zylich

# splits Persona-Chat dataset such that personas are included with each line of dialogue

__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

import numpy as np

text = open('training_dialogue_plus_personas.txt', 'r')
p = open('personas', "w")
q = open('context', 'w')
a = open('answers', 'w')
pre_pre_previous_raw=''
pre_previous_raw=''
previous_raw=''
person = ' '
previous_person=' '

l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']

persona1 = ""
persona2 = ""
in_persona = False
count = 0

for i, raw_word in enumerate(text):
    original_word = raw_word
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

    if "<p1>:" in original_word or "<p2>:" in original_word:
        if not in_persona:
            persona1 = raw_word.replace("<p1> : ", "")
            persona2 = None
            in_persona = True
        else:
            persona2 = raw_word.replace("<p2> : ", "")
    else:
        if in_persona:
            in_persona = False
            pre_pre_previous_raw = ""
            pre_previous_raw = ""
            previous_raw = ""
            count = 0
        else:
            if count % 2 == 0:
                current_persona = persona2
            else:
                current_persona = persona1

            if previous_raw != "":
                p.write(current_persona)
                q.write((pre_previous_raw[:-1] + ' ' + previous_raw[:-1]).strip() + '\n')  # python will convert \n to os.linese
                a.write(raw_word[:-1]+ '\n')

    
        pre_pre_previous_raw = pre_previous_raw    
        pre_previous_raw = previous_raw
        if "__SILENCE__" not in original_word:
            previous_raw = raw_word
        else:
            previous_raw = ""
        count += 1

p.close()
q.close()
a.close()
