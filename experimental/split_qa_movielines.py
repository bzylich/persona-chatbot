# -*- coding: utf-8 -*-
# editor -- me
__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

import numpy as np

text = open('movie_lines.txt', 'r', encoding="iso-8859-1")

q = open('context_movie', 'w', encoding="utf-8")
a = open('characters', 'w', encoding="utf-8")
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

for line in reversed(list(text)):
    #print(line.rstrip())
    raw_word = line
    parts = raw_word.split(' +++$+++ ')

    person = parts[1]
    if person not in personas:
        personas[person] = ""
    scriptline = parts[0]
    scriptline = int(scriptline[1:])
    movie = parts[2]
    raw_word = parts[4].strip()
    
    if(first):
        # reset when new set of dialog partners or a new movie
        previous_scriptline = scriptline - 1
        previous_person = person
        previous_movie = movie
        personID = True
        person1 = ''
        person2 = ''
        first = False
    


    raw_word = raw_word.lower()



    for j, term in enumerate(l1):
        raw_word = raw_word.replace(term,l2[j])
        
    for term in l3:
        raw_word = raw_word.replace(term,' ')

    personas[person] += raw_word + ' '
    
   


    # print(str(scriptline) + " " + person + " " + movie + " " + raw_word + "\n")
    
    # reset variables on new movie
    # if(movie != previous_movie):
    #     if(person1 != '' and person2 != ''):
    #         q.write(person1 + '\n' + person2 + '\n')  # python will convert \n to os.linese
    #     previous_scriptline = scriptline - 1
    #     previous_person = person
    #     previous_movie = movie
    #     person1 = ''
    #     person2 = ''
    # # reset variables on new set of dialog partners
    # elif(scriptline != previous_scriptline + 1):
    #     if(person1 != '' and person2 != ''):
    #         q.write(person1 + '\n' + person2 + '\n')  # python will convert \n to os.linese
    #     previous_scriptline = scriptline - 1
    #     previous_person = person
    #     person1 = ''
    #     person2 = ''
    # # parse the conversation between conversation partners otherwise
    # else:
    #     if(person != previous_person):
    #         personID = not (personID)
    #         print("person = " + person + " previous person = " + previous_person + '\n')
        
    #     # only have to worry about two people for each dialog it seems
    #     if(personID): # person 1
    #         person1 += raw_word + ' '
    #         print("person1: " + person1 + '\n')
    #     else: # person 2
    #         person2 += raw_word + ' '
    #         print("person2: " + person2 + '\n') 



    # previous_scriptline = scriptline
    # previous_person = person
    # previous_movie = movie

    # change this to actually interpret za warudos
    # if(raw_word[0] == '<'):
    #     a.write(raw_word[7:-1]+ '\n')
    #     if(person1 != '' and person2 != ''):
    #         q.write(person1 + '\n' + person2 + '\n')  # python will convert \n to os.linese
    #         personID = 2
    #         person1 = ''
    #         person2 = ''
    # else: 
    #     if(raw_word[0] == '_'):
    #         personID = 1
    #     if(personID == 1):
    #         person1 += raw_word[:-1] + ' '
    #         personID = 2
    #     elif(personID == 2):
    #         person2 += raw_word[:-1] + ' ' 
    #         personID = 1

for p, text in personas.items():
    a.write(p + '\n')
    q.write(text + '\n')
    

q.close()
a.close()
