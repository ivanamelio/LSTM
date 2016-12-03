from dependences import *


def L2_modif(tr_M_ref, tr_M, zeri_list_ref):
	assert tr_M_ref.shape == tr_M.shape
	matrice = []
	for i in range(len(tr_M_ref)):
		if matrix_element2char(i) not in zeri_list_ref:
			matrice.append(tr_M_ref[i]-tr_M[i])
	matrice = np.array(matrice)
	return np.sqrt((matrice**2).mean())


def z_err(zeri_list, LSTM_freq):
	f_error = 0.
	for z in zeri_list:
		f_error += LSTM_freq[z]
	return f_error

def stampa_err(argv, L2_modif, z_err, L):
	orig = sys.stdout
	f = open("6_risultati_serial.dat", "a")
	sys.stdout = f
	print int(argv[1]),  L2_modif,  z_err, L,  argv[4], argv[2], float(argv[3])
	sys.stdout = orig
	f.close()
