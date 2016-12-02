from dependenceszero import *


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



def estrai_matrice(markov_text):
	M_2_to_1 = np.zeros((n_states**r, n_states))
	for i in range(len(markov_text)-r):
		s = markov_text[i:(i+r)]
		M_2_to_1[char2id(s),char2id(markov_text[i+r])]+=1.


	freq_dict = {}
	zeri_list = []
	for i in range(len(M_2_to_1)):
		freq_dict[matrix_element2char(i)] =  np.sum(M_2_to_1[i,:])/np.sum(M_2_to_1)
		#sys.stderr.write('frequency  ' + matrix_element2char(i)+ '  ->  ' + str(freq_dict[matrix_element2char(i)]) +'\n')
		if freq_dict[matrix_element2char(i)] == 0:   #pulisce eventuali righe di zeri
			zeri_list.append(matrix_element2char(i))
			for j in range(len(M_2_to_1[0])):
				M_2_to_1[i,j] = 1.


	M_2_to_1 = M_2_to_1/np.sum(M_2_to_1, axis = 1)[:,None]

	#for i in range(len(M_2_to_1)):
		#for j in range(len(M_2_to_1[0])):
			#sys.stderr.write('prob  ' + matrix_element2char(i) + '  ->  ' + id2char(j) + ' :  ' + '%.2f'  % M_2_to_1[i,j]+'\n')
	#print(M_2_to_1)

	Markov_dict = {}
	for i in range(len(M_2_to_1)):
		Markov_dict[matrix_element2char(i)] =  M_2_to_1[i,:]
	#print(Markov_dict)
	return M_2_to_1, Markov_dict, freq_dict, zeri_list
