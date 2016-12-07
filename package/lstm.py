
class LSTM_parms:

    def __init__(self,num_nodes, num_unrollings, out_gate, batch_size, n_train_steps, dropout_par):
        self.num_nodes=num_nodes
        self.num_unrollings=num_unrollings
        self.out_gate=out_gate
        self.batch_size=batch_size
        self.n_train_steps=n_train_steps
        self.dropout_par=dropout_par

pars = LSTM_parms(num_nodes, num_unrollings,  out_gate, batch_size, n_train_steps, dropout_par)

class BatchGenerator(object): # "new-style" classes... to formally identify class and type
	def __init__(self, text, batch_size, num_unrollings):
		self._text = text
		self._text_size = len(text)
		self._batch_size = batch_size
		self._num_unrollings = num_unrollings
		segment = self._text_size // batch_size
		self._cursor = [ offset * segment for offset in range(batch_size)]
		self._last_batch = self._next_batch()

	def _next_batch(self):
		"""Generate a single batch from the current cursor position in the data."""
		batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
		for b in range(self._batch_size):
			batch[b, char2id(self._text[self._cursor[b]])] = 1.0
			self._cursor[b] = (self._cursor[b] + 1) % self._text_size
		return batch

	def next(self):
		"""Generate the next array of batches from the data. The array consists of
		the last batch of the previous array, followed by num_unrollings new ones.
		"""
		batches = [self._last_batch]
		for step in range(self._num_unrollings):
			batches.append(self._next_batch())
		self._last_batch = batches[-1]
		return batches

def characters(probabilities):
	"""Turn a 1-hot encoding or a probability distribution over the possible
	characters back into its (most likely) character representation."""
	return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
	"""Convert a sequence of batches back into their (most likely) string
	representation."""
	s = [''] * batches[0].shape[0]
	for b in batches:
		s = [''.join(x) for x in zip(s, characters(b))]
	return s

def logprob(predictions, labels):
	"""Log-probability of the true labels in a predicted batch."""
	predictions[predictions < 1e-10] = 1e-10
	return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
	"""Sample one element from a distribution assumed to be an array of normalized
	probabilities.
	"""
	r = random.uniform(0, 1)
	s = 0
	for i in range(len(distribution)):
		s += distribution[i]
		if s >= r:
			return i
	return len(distribution) - 1

def sample(prediction):
	"""Turn a (column) prediction into 1-hot encoded samples."""
	p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
	p[0, sample_distribution(prediction[0])] = 1.0
	return p

def random_distribution():
	"""Generate a random column of probabilities."""
	b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
	return b/np.sum(b, 1)[:,None]



#def crea_grafo_lstm(pars):
#PARTE DEL MAIN
    #num_nodes = int(sys.argv[1]),   arg[2] is output gate

graph = tf.Graph()
with graph.as_default():

    keep_prob = tf.placeholder(tf.float32)
    # Parameters:
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([vocabulary_size, pars.num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([pars.num_nodes, pars.num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, pars.num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([vocabulary_size, pars.num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([pars.num_nodes, pars.num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, pars.num_nodes]))
    # Memory cell: input, state and bias.
    cx = tf.Variable(tf.truncated_normal([vocabulary_size, pars.num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([pars.num_nodes, pars.num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, pars.num_nodes]))
    # Output gate: input, previous output, and bias.


    if pars.out_gate == 'output_gate=yes':

        ox = tf.Variable(tf.truncated_normal([vocabulary_size, pars.num_nodes], -0.1, 0.1))
        om = tf.Variable(tf.truncated_normal([pars.num_nodes, pars.num_nodes], -0.1, 0.1))
        ob = tf.Variable(tf.zeros([1, pars.num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([pars.batch_size, pars.num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([pars.batch_size, pars.num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([pars.num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))

    saver = tf.train.Saver()

    input_gate, forget_gate, output_gate = [0.,0.,0.]
    # Definition of the cell computation.
    def lstm_cell(i, o, state):
	    #"""Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
	    #Note that in this formulation, we omit the various connections between the
	    #previous state and the gates."""
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        if pars.out_gate=='output_gate=yes':
            output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
            return output_gate * tf.tanh(state), state
        else:
            output_gate = 0.
            return tf.tanh(state), state, input_gate, forget_gate, output_gate

    # Input data.
    train_data = list()
    for _ in range(pars.num_unrollings + 1):
        train_data.append(tf.placeholder(tf.float32, shape=[pars.batch_size,vocabulary_size]))
    train_inputs = train_data[:pars.num_unrollings] #delle 11 lettere le prime 10 servono per predire l' ultima
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.

    # Unrolled LSTM loop. Logits chiama outputs e questo loop viene eseguito! (giusto??)
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state, input_gate, forget_gate, output_gate = lstm_cell(i, output, state)
        outputs.append(output)

    # State saving across unrollings. in this way the recurrent state flows continuously in time, even
	#if the weight update occurs any num_unrollings time steps. hink it is strictly necessary only after validation stuff
    with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
		# state credo che sia num_node x batch_size... viene tramandato a nuovo batch
    	# Classifier.
        logits = tf.nn.xw_plus_b(tf.nn.dropout(tf.concat(0, outputs), keep_prob), w, b)  # applica softmax a tutti i 10 output e li confronta con i labels
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

    # Optimizer. Clippa il gradiente e tuna il learning rate
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
    saved_sample_output = tf.Variable(tf.zeros([1, pars.num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, pars.num_nodes]))
    reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, pars.num_nodes])),saved_sample_state.assign(tf.zeros([1, pars.num_nodes])))
    sample_output, sample_state, input_gate, forget_gate, output_gate = lstm_cell(sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

#return graph

def train_lstm(pars, graph, train_text, valid_text):

    num_steps = pars.n_train_steps   #dropout_par = float(argv[3])
    summary_frequency = 400
    valid_size= len(valid_text)

    train_batches = BatchGenerator(train_text, pars.batch_size, pars.num_unrollings)
    valid_batches = BatchGenerator(valid_text, 1, 1)

    with tf.Session(graph=graph) as session:
    	tf.initialize_all_variables().run()
    	sys.stderr.write('Initialized'+'\n')
    	mean_loss = 0
    	for step in range(num_steps):
    		batches = train_batches.next()
    		feed_dict = dict()
    		feed_dict[keep_prob]=pars.dropout_par
    		for i in range(pars.num_unrollings + 1):
    			feed_dict[train_data[i]] = batches[i]
    		_, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    		mean_loss += l
    		if step % summary_frequency == 0:
    			if step > 0:
    				mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few batches.
    			sys.stderr.write('Average loss at step %d: %f learning rate: %f   \n' % (step, mean_loss, lr))
    			mean_loss = 0
    			labels = np.concatenate(list(batches)[1:])
    			sys.stderr.write('Minibatch perplexity: %.2f  \n' % float(np.exp(logprob(predictions, labels))))
    			#if step % (summary_frequency * 10) == 0:
    		#backup
    				#save_path = saver.save(session, "check_point_training.ckpt")
    		# Generate some samples.
    				#print('=' * 80)
    				#for _ in range(5):
    					#feed = sample(random_distribution())
    					#sentence = characters(feed)[0]
    					#reset_sample_state.run()  ####   <<<<<-----------
    					#for _ in range(79):
    						#prediction = sample_prediction.eval({sample_input: feed})
    						#feed = sample(prediction)
    						#sentence += characters(feed)[0]
    					#print(sentence)
    				#print('=' * 80)
    	# Measure validation set perplexity.
    			reset_sample_state.run()    ######   <<<<<<<<<---------
    			valid_logprob = 0
    			for _ in range(valid_size):
    				b = valid_batches.next()
    				predictions = sample_prediction.eval({sample_input: b[0], keep_prob: 1.})
    				valid_logprob = valid_logprob + logprob(predictions, b[1])
    			sys.stderr.write('Validation set perplexity: %.2f    \n' % float(np.exp(valid_logprob / valid_size)))

    	ora = datetime.datetime.now()
    	save_path = saver.save(session, 'backup_training/backup_training_'+ str(ora.day)+'_'+str(ora.month)+'_alle_'+str(ora.hour) +'.ckpt')

    return save_path

def genera_testo(L, graph, save_path):
	with tf.Session(graph=graph) as session:
		saver.restore(session, save_path)
		for _ in range(1):
			feed = sample(random_distribution())
			sentence = characters(feed)[0]
			reset_sample_state.run()  ####   <<<<<-----------
			for _ in range(L):
				prediction = sample_prediction.eval({sample_input: feed, keep_prob: 1.})  ### the sampled letter is fed as new input
				feed = sample(prediction)
				sentence += characters(feed)[0]
	return sentence

def get_weights(graph, save_path, pars):
    with tf.Session(graph=graph) as session:
        saver.restore(session, save_path)
        if pars.out_gate == 'output_gate=yes':
            pesi_and_bias = session.run([cx, cm, cb, ix, im, ib, fx, fm, fb, ox, om, ob, w, b])
        else:
            pesi_and_bias = session.run([cx, cm, cb, ix, im, ib, fx, fm, fb, w, b])
        return pesi_and_bias

def neural_recordings(graph, save_path, valid_text):
    with tf.Session(graph=graph) as session:
        saver.restore(session, save_path)
        valid_batches = BatchGenerator(valid_text, 1, 1)
        reset_sample_state.run()    ######   <<<<<<<<<---------
        rec = {}

        for i in range(valid_size):
            b = valid_batches.next()
            #predictions, input_sw, for_sw, activ = session.run([sample_prediction, input_gate, forget_gate, state],{sample_input: b[0], keep_prob: 1.})
            predictions, input_sw, for_sw, activ = session.run([sample_prediction, input_gate, forget_gate, sample_state],{sample_input: b[0], keep_prob: 1.})
            if i == 0:
                rec['in_sw'] = input_sw
                rec['f_sw'] = for_sw
                rec['activity'] = activ
                rec['sftmx'] = predictions
            else:
                rec['in_sw'] = np.append(rec['in_sw'], input_sw, axis = 0)
                rec['f_sw'] = np.append(rec['f_sw'], for_sw, axis = 0)
                rec['activity'] = np.append(rec['activity'], activ, axis = 0)
                rec['sftmx'] = np.append(rec['sftmx'], predictions, axis = 0)
        return rec
