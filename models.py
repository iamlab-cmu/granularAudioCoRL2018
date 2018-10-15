import tensorflow as tf

class CNN(object):
	scope = 'critic'

	def __init__(self, sess, state_shape, lr=1e-5, decay=0.99, train_mass=True, train_spill=True):
		self._sess = sess
		self._state_shape = state_shape
		self._lr = lr
		# self._target_inputs, self._target_action, self._target_mass = self._build_network()
		# self._target_network = tf.trainable_variables()[(len(self._network) + var_offset):]
		self._target_mass = tf.placeholder(tf.float32, shape=[None])
		self._target_spill = tf.placeholder(tf.bool, shape=[None])
		self._dropout_prob = tf.placeholder_with_default(1.0, shape=())
		self._inputs, self._mass_out, self._spill_out = self._build_network()
		
		# self._update_target = [self._target_network[i].assign((self._network[i] * tau) + (self._target_network[i] * (1 - tau))) for i in range(len(self._target_network))]

		self._mass_loss = train_mass * tf.Print(tf.losses.mean_squared_error(
			labels=tf.stop_gradient(self._target_mass),
			predictions=self._mass_out,
		), [tf.shape(self._inputs), tf.shape(self._mass_out), tf.shape(self._target_mass), tf.reduce_max(self._inputs), self._mass_out, self._target_mass], first_n=0, summarize=5)

		self._spill_loss = train_spill * tf.Print(tf.losses.sigmoid_cross_entropy(
			multi_class_labels=tf.cast(self._target_spill, tf.int32),
			logits=self._spill_out,
		), [self._spill_out, self._target_spill], first_n=0, summarize=10)
		self._spill_prediction = self._spill_out > 0
		self._spill_accuracy = tf.Print(1.0 - tf.reduce_sum(tf.abs(tf.cast(self._spill_prediction, tf.float32) - tf.cast(self._target_spill, tf.float32))) / tf.cast(tf.shape(self._target_spill), tf.float32)[0],
			[tf.cast(self._spill_prediction, tf.int32), tf.cast(self._target_spill, tf.int32)], first_n=0, summarize=20)
		true_positive = tf.reduce_sum(tf.cast(tf.logical_and(self._spill_prediction, self._target_spill), tf.float32))
		false_positive = tf.reduce_sum(tf.cast((tf.cast(self._spill_prediction, tf.float32) - tf.cast(self._target_spill, tf.float32)) > 0.1, tf.float32))
		false_negative = tf.reduce_sum(tf.cast((tf.cast(self._target_spill, tf.float32) - tf.cast(self._spill_prediction, tf.float32)) > 0.1, tf.float32))
		self._spill_precision = true_positive / (true_positive + false_positive)
		self._spill_recall = true_positive / (true_positive + false_negative)

		self._loss = self._mass_loss + self._spill_loss

		epochs = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(self._lr, epochs,
										   1, decay, staircase=True)

		self._optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self._loss)

		self._increment_epochs = tf.Print(tf.assign(epochs, epochs+1), [learning_rate])

		self._saver = tf.train.Saver()

	def _build_network(self):
		s_inputs = tf.placeholder(tf.float32, shape=(None,) + self._state_shape)
		reshaped = tf.reshape(s_inputs, shape=([-1] + list(self._state_shape + (1,))))
		filters = [8, 16, 32]
		strides = [2, 2, 2]
		conv1 = tf.layers.conv2d(
		  inputs=reshaped,
		  filters=filters[0],
		  kernel_size=[3, 3],
		  padding='same',
		  activation=tf.nn.relu,
		  name='conv1')
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[strides[0], strides[0]], strides=strides[0])
		conv2 = tf.layers.conv2d(
		  inputs=pool1,
		  filters=filters[1],
		  kernel_size=[4, 4],
		  padding='same',
		  activation=tf.nn.relu,
		  name='conv2')
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[strides[1], strides[1]], strides=strides[1])
		conv3 = tf.layers.conv2d(
		  inputs=pool2,
		  filters=filters[2],
		  kernel_size=[4, 4],
		  padding='same',
		  activation=tf.nn.relu,
		  name='conv3')
		pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[strides[2], strides[2]], strides=strides[2])
		width = int(int(self._state_shape[1] / strides[0])/strides[1]/strides[2])
		height = int(int(self._state_shape[0] / strides[0])/strides[1]/strides[2])
		# pool3 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=strides[2])
		# width = int(int(self._state_shape[1]  /strides[2]))
		# height = int(int(self._state_shape[0] /strides[2]))
		pool3_flat = tf.reshape(pool3, [-1, height * width * filters[2]])
		fc1 = tf.Print(tf.layers.dense(pool3_flat, 256, 
				name='fc1', activation=tf.nn.relu), [tf.shape(reshaped)], first_n=0, summarize=5)
		fc2 = tf.layers.dense(tf.nn.dropout(fc1, self._dropout_prob), 256, 
				name='fc2', activation=tf.nn.relu)
		mass_out = tf.layers.dense(fc2, 1, 
				name='mass_output')
		spill_out = tf.layers.dense(fc2, 1, 
				name='spill_output')
		# out = tf.layers.dense(tf.nn.dropout(fc1, self._dropout_prob), 1, 
		# 		name='conv_output')
		return s_inputs, tf.squeeze(mass_out), tf.squeeze(spill_out)

	def train(self, state, mass, spill, dropout=0.5):
		return self._sess.run([self._optimizer, self._loss, self._mass_loss, self._spill_loss],
			{self._inputs: state, self._target_mass: mass, self._target_spill: spill, self._dropout_prob:dropout})[1:]

	def test(self, state, mass, spill):
		return self._sess.run([self._mass_loss, self._spill_accuracy, self._spill_precision, self._spill_recall], {self._inputs: state, self._target_mass: mass, self._target_spill: spill})

	def predict(self, state):
		return self._sess.run([self._mass_out, self._spill_prediction], {self._inputs: state})

	def save_model_weights(self, prefix): 
		self._saver.save(self._sess, './'+prefix+'.ckpt')

	def load_model_weights(self, weight_file):
		self._saver.restore(self._sess, weight_file)

	def get_state_shape(self):
		return self._state_shape

	def increment_step(self):
		return self._sess.run(self._increment_epochs)

class RNN(CNN):

	def predict_all_outputs(self, state):
		return self._sess.run([self._all_mass_out, self._all_spill_out], {self._inputs: state})

class LSTM(RNN):
	def _build_network(self):
		s_inputs = tf.placeholder(tf.float32, shape=(None,) + self._state_shape)
		batch_size = tf.shape(s_inputs)[0]
		time_aligned = tf.transpose(s_inputs, perm=[0, 2, 1])
		lstm_cell = tf.nn.rnn_cell.LSTMCell(512)
		initial_state = lstm_cell.zero_state(batch_size, tf.float32)
		outputs, states = tf.nn.dynamic_rnn(lstm_cell, time_aligned, initial_state=initial_state, dtype=tf.float32)
		mass_out0 = tf.layers.dense(tf.nn.dropout(outputs, self._dropout_prob), 512, activation=tf.nn.relu)
		self._all_mass_out = tf.layers.dense(tf.nn.dropout(mass_out0, self._dropout_prob), 1)
		self._all_spill_out = tf.layers.dense(tf.nn.dropout(outputs, self._dropout_prob), 1)
		final_mass_out = tf.slice(self._all_mass_out, begin=[0, tf.shape(outputs)[1] - 1, 0], size=[batch_size, 1, 1])
		final_spill_out = tf.slice(self._all_spill_out, begin=[0, tf.shape(outputs)[1] - 1, 0], size=[batch_size, 1, 1])
		return s_inputs, tf.squeeze(final_mass_out), tf.squeeze(final_spill_out)

class GRU(RNN):
	def _build_network(self):
		s_inputs = tf.placeholder(tf.float32, shape=(None,) + self._state_shape)
		batch_size = tf.shape(s_inputs)[0]
		time_aligned = tf.transpose(s_inputs, perm=[0, 2, 1])
		lstm_cell = tf.nn.rnn_cell.GRUCell(512)
		initial_state = lstm_cell.zero_state(batch_size, tf.float32)
		outputs, states = tf.nn.dynamic_rnn(lstm_cell, time_aligned, initial_state=initial_state, dtype=tf.float32)
		mass_out0 = tf.layers.dense(tf.nn.dropout(outputs, self._dropout_prob), 512, activation=tf.nn.relu)
		self._all_mass_out = tf.layers.dense(tf.nn.dropout(mass_out0, self._dropout_prob), 1)
		self._all_spill_out = tf.layers.dense(tf.nn.dropout(outputs, self._dropout_prob), 1)
		final_mass_out = tf.slice(self._all_mass_out, begin=[0, tf.shape(outputs)[1] - 1, 0], size=[batch_size, 1, 1])
		final_spill_out = tf.slice(self._all_spill_out, begin=[0, tf.shape(outputs)[1] - 1, 0], size=[batch_size, 1, 1])
		return s_inputs, tf.squeeze(final_mass_out), tf.squeeze(final_spill_out)


class TimeFC(RNN):
	def _build_network(self):
		s_inputs = tf.placeholder(tf.float32, shape=(None,) + self._state_shape)
		batch_size = tf.shape(s_inputs)[0]
		time_aligned = tf.transpose(s_inputs, perm=[0, 2, 1])
		shared0 = tf.layers.dense(time_aligned, 512, activation=tf.nn.relu)
		shared1 = tf.layers.dense(tf.nn.dropout(shared0, self._dropout_prob), 512, activation=tf.nn.relu)
		shared2 = tf.layers.dense(tf.nn.dropout(shared1, self._dropout_prob), 1, activation=tf.nn.relu)
		cumul = tf.cumsum(shared2, axis=1)
		self._all_mass_out = cumul
		self._all_spill_out = tf.layers.dense(cumul, 1)
		final_mass_out = tf.slice(self._all_mass_out, begin=[0, tf.shape(cumul)[1] - 1, 0], size=[batch_size, 1, 1])
		final_spill_out = tf.slice(self._all_spill_out, begin=[0, tf.shape(cumul)[1] - 1, 0], size=[batch_size, 1, 1])
		return s_inputs, tf.squeeze(final_mass_out), tf.squeeze(final_spill_out)

class SumGRU(RNN):
	def _build_network(self):
		s_inputs = tf.placeholder(tf.float32, shape=(None,) + self._state_shape)
		batch_size = tf.shape(s_inputs)[0]
		time_aligned = tf.transpose(s_inputs, perm=[0, 2, 1])
		lstm_cell = tf.nn.rnn_cell.GRUCell(512)
		initial_state = lstm_cell.zero_state(batch_size, tf.float32)
		outputs, states = tf.nn.dynamic_rnn(lstm_cell, time_aligned, initial_state=initial_state, dtype=tf.float32)
		shared0 = tf.layers.dense(tf.nn.dropout(outputs, self._dropout_prob), 1, activation=tf.nn.relu)
		cumul = tf.cumsum(shared0, axis=1)
		self._all_mass_out = cumul
		self._all_spill_out = tf.layers.dense(cumul, 1)
		final_mass_out = tf.slice(self._all_mass_out, begin=[0, tf.shape(cumul)[1] - 1, 0], size=[batch_size, 1, 1])
		final_spill_out = tf.slice(self._all_spill_out, begin=[0, tf.shape(cumul)[1] - 1, 0], size=[batch_size, 1, 1])
		return s_inputs, tf.squeeze(final_mass_out), tf.squeeze(final_spill_out)

class Logistic(CNN):
	def _build_network(self):
		s_inputs = tf.placeholder(tf.float32, shape=(None,) + self._state_shape)
		summed = tf.reduce_sum(s_inputs, axis=2)
		mass_out = tf.layers.dense(summed, 1)
		spill_out = tf.layers.dense(summed, 1)
		return s_inputs, tf.squeeze(mass_out), tf.squeeze(spill_out)