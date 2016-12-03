# LSTM for learning Markov chains (|q|=4, r=6).
# The LSTM is trained presenting a random chain-like text.
# Then a sequence is generated by feeding the sampled letter as new input.
# =============
# arg required in input:  hidden_units 'output_gate=yes'/'output_gate=no' dropout_param     n_train_steps
#
from package.dependences import *
import package.metriche as mtr

argv = sys.argv
assert len(sys.argv) == 5
assert argv[2] in ['output_gate=yes','output_gate=no']

Lgt = 100000
num_nodes = int(argv[1])
out_gate = argv[2]
dropout_par = float(argv[3])
n_train_steps = int(argv[4])
batch_size=64
num_unrollings=10

with open('package/markov_data/markov_text_mem='+str(r)+'.txt', 'r') as myfile:
    text=myfile.read()
chain_stats = np.load('package/markov_data/chain_stats_mem='+str(r)+'.npz')
M_2_to_1 = chain_stats['a']
zeri_list = map(matrix_element2char, chain_stats['b'].tolist())


########### cuore del machine learning ##########
valid_size = 2000  #qual e' il tempo in cui Markov chain non dipende piu' da stato iniziale
valid_text = text[:valid_size]
train_text = text[valid_size:]
#train_size = len(train_text)
#pars = lstm.LSTM_parms(num_nodes, num_unrollings,  out_gate, batch_size, n_train_steps, dropout_par)
#graph = lstm.crea_grafo_lstm(pars)
execfile('package/lstm.py')
save_path = train_lstm(pars, graph, train_text, valid_text)
LSTM_text = genera_testo(Lgt, graph, save_path)
############################


LSTM_2_to_1, LSTM_dict, LSTM_freq, _ = estrai_matrice(LSTM_text)


L2_modif = mtr.L2_modif(M_2_to_1, LSTM_2_to_1, zeri_list)
z_err = mtr.z_err(zeri_list, LSTM_freq)
mtr.stampa_err(argv, L2_modif, z_err, Lgt)
