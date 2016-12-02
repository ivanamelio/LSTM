#!/usr/bin/python


# coding: utf-8

#
# Extract Markov transition probabilities (r=2) from the first 39 a, c, e, h  of wikipedia
# and generate a fictious text, storing it in markov_text.txt
#


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import datetime
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
import datetime
import sys


argv = sys.argv
assert len(sys.argv) == 2

############## Markov states ############

def read_data(filename):
	f = zipfile.ZipFile(filename)
	for name in f.namelist():
		return tf.compat.as_str(f.read(name))
	f.close()
markov_text_0 = read_data('/home/ivan/Desktop/udacity_courses/corso_DL/tensorflow/tensorflow/examples/udacity/text8.zip')


state_dict = ['a','c','e','h']        ##   <<<<<<<<---------------
n_states = len(state_dict)
r = argv[1] #memory


def togli_altre_lettere(text):
	markov_text = ''
	for i in range(len(text)):
		if text[i] in state_dict:
			markov_text+=text[i]
	return markov_text
markov_text = togli_altre_lettere(markov_text_0)


# Utility functions to map characters to vocabulary IDs and back.

# In[10]:

############   character <-> number  functions

vocabulary_size = n_states

def char2id(char):
	if len(char) == 1:
		return state_dict.index(char)
	if len(char) == 2:
		return state_dict.index(char[0])*n_states+state_dict.index(char[1])
	else:
		idn = 0
		for i in range(len(char)):
			idn += state_dict.index(char[len(char)-i-1])*(n_states**i)
		return idn

def id2char(num_id):  # inversa di char2id
	return state_dict[num_id]

def matrix_element2char(i):  # inversa di char2id
	s = ''
	for j in range(r):
		k = r-j-1
		divs = n_states**k
		s += state_dict[i/divs]
		i = i%divs
	return s







# In[11]:

################   create Markov matrix from a short text, so to have some forbidden states and some forbidden transitions
# warning! a state is forbidden iff has zero frequency in freq_dist
# but this iff statement holds if the last two letters of the text had already appeared...

#def togli_coda(text):
#    coda = len(text)
#    while coda >len(text)/2:
#        for j in range(coda-2):
#            if text[j]==text[coda-2] and text[j+1]==text[coda-1]:
#                return text[:coda]
#        coda = coda - 1
#    assert 0 >1
#print(togli_coda(markov_text[:40]))

def estrai_matrice(markov_text):
	M_2_to_1 = np.zeros((n_states**r, n_states))
	for i in range(len(markov_text)-r):
		s = markov_text[i:(i+r)]
		M_2_to_1[char2id(s),char2id(markov_text[i+r])]+=1.


	freq_dict = {}
	zeri_list = []
	for i in range(len(M_2_to_1)):
		freq_dict[matrix_element2char(i)] =  np.sum(M_2_to_1[i,:])/np.sum(M_2_to_1)
		sys.stderr.write('frequency  ' + matrix_element2char(i)+ '  ->  ' + str(freq_dict[matrix_element2char(i)]) +'\n')
		if freq_dict[matrix_element2char(i)] == 0:   #pulisce eventuali righe di zeri
			zeri_list.append(matrix_element2char(i))
			for j in range(len(M_2_to_1[0])):
				M_2_to_1[i,j] = 1.


	M_2_to_1 = M_2_to_1/np.sum(M_2_to_1, axis = 1)[:,None]

	for i in range(len(M_2_to_1)):
		for j in range(len(M_2_to_1[0])):
			sys.stderr.write('prob  ' + matrix_element2char(i) + '  ->  ' + id2char(j) + ' :  ' + '%.2f'  % M_2_to_1[i,j]+'\n')
	#print(M_2_to_1)

	Markov_dict = {}
	for i in range(len(M_2_to_1)):
		Markov_dict[matrix_element2char(i)] =  M_2_to_1[i,:]
	#print(Markov_dict)
	return M_2_to_1, Markov_dict, freq_dict, zeri_list

M_2_to_1, Markov_dict, freq_dict, zeri_list = estrai_matrice(markov_text[:39])



# In[13]:

def Markov_choice(s, n_states, state_dict, Markov_dict):
	rr = -np.random.rand()
	for x in range(n_states):
		rr += Markov_dict[s][x]
		if rr >= 0:
			return state_dict[x]


def genera_dati(L, n_states, state_dict, r, Markov_dict, inizio_markov_text):
	text = inizio_markov_text
	for i in range(r,L-1):
		nuovo=Markov_choice(text[(i-r):i], n_states, state_dict, Markov_dict)
		text+=nuovo
	return text
text = genera_dati(10000000, n_states, state_dict, r, Markov_dict, markov_text[:r])   #.........  deve essere una stringa
#datasize want to be order 10^8
#print(text[:1000])
_,_,_,_ = estrai_matrice(text[:40000])  # e' essenziale per capire 1) se partendo da stato permesso a caso catena tocca tutti stati ammessi
							#2) non tocca stati proibiti
# le frequenze di text non sono le stesse di


with open('markov_text_mem=' + str(r) +'.txt', 'w') as myfile:
    myfile.write(text)


zeri_id = np.array(map(char2id, zeri_list))
np.savez('chain_stats_mem=' + str(r) +'.npz', a=M_2_to_1, b=zeri_id)
