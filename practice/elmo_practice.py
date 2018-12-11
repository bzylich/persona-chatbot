import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Embedding, TimeDistributed, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import os.path
import sys

conv1 = ["hi", "hello", "how are you", "very good and you", "not bad", "what have you been doing today", "not much",
		"same", "i have to go", "good bye", "see you later"]

print(conv1)

vocab = {}
max_len = 0

def build_vocab(conv, expand_vocab=True):
	global max_len
	vocab['<s>'] = 0
	vocab['<e>'] = 1
	vocab['<unk>'] = 2
	new_conv = []
	for utter in conv:
		tokens = utter.split(" ")
		new_utter = [vocab['<s>']]
		num_tokens = 1
		for t in tokens:
			if len(t) > 0:
				if t not in vocab:
					if expand_vocab:
						vocab[t] = len(vocab)
						new_utter.append(vocab[t])
					else:
						new_utter.append(vocab['<unk>'])
				else:
					new_utter.append(vocab[t])
				num_tokens += 1
				if not expand_vocab and num_tokens > max_len - 2:
					break
		new_utter.append(vocab['<e>'])
		new_conv.extend(new_utter)
		if expand_vocab and len(new_utter) > max_len:
			max_len = len(new_utter)
	return new_conv

conv1_tokenized = build_vocab(conv1)
reverse_vocab = {key: word for (word, key) in vocab.items()}

def create_input_sequences(conv):
	new_conv = []
	for i in range(len(conv)):
		new_seq = []
		for j in range(max(0, i - max_len + 1), i + 1):
			new_seq.append(reverse_vocab[conv[j]])
		new_conv.append(new_seq)
	return new_conv

conv1_padded = pad_sequences(create_input_sequences(conv1_tokenized), padding="pre", value='<unk>', dtype=object)

print("vocab size:", len(vocab))
print("max seq len:", max_len)

X_train = conv1_padded[:-1] 
print(X_train)
y_train = to_categorical(conv1_tokenized[1:], num_classes=len(vocab)) 

batch_size = 5

# Initialize session
sess = tf.Session()
K.set_session(sess)

# Now instantiate the elmo model
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)),
							"sequence_len": tf.constant(batch_size * [max_len])
							}, 
					signature="tokens", as_dict=True)["elmo"]

# define the model architecture
model = Sequential()
model.add(Lambda(ElmoEmbedding, input_shape=(max_len,), output_shape=(max_len, 1024), dtype=object))
model.add(LSTM(512, return_sequences=True, stateful=False))
model.add(LSTM(256, return_sequences=False, stateful=False))
model.add((Dense(256, activation='relu')))
model.add((Dense(len(vocab), activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics=['accuracy'])


filepath="./saved_models/elmo_practice.hdf5"
if os.path.exists(filepath):
	model.load_weights(filepath)

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# only train the model if a saved version is not found or an extra argument is given on the command line
if len(sys.argv) < 2 or not os.path.exists(filepath):
	model.fit(X_train, y_train, epochs=30, batch_size=batch_size, verbose=1, callbacks=callbacks_list)

# scores = model.evaluate(X_train, y_train, verbose=1)

prompt = ""
prompt_tokenized = []
while True:
	# prime the agent with the entire prompt
	prompt_tokenized = np.array(list(prompt_tokenized) + build_vocab([prompt], expand_vocab=False))
	prompt_padded = np.array([pad_sequences(create_input_sequences(prompt_tokenized), maxlen=max_len, padding="pre", value='<unk>', dtype=object)[-1]])
	response = ""
	while True:
		next_word = model.predict(prompt_padded)
		prompt_tokenized = np.array(list(prompt_tokenized) + [np.argmax(next_word)])
		next_word = reverse_vocab[np.argmax(next_word)]
		if next_word == "<e>":
			break
		prompt_padded = np.array([pad_sequences(create_input_sequences(prompt_tokenized), maxlen=max_len, padding="pre", value='<unk>', dtype=object)[-1]])
		if next_word == "<s>" or next_word == "<unk>":
			continue
		if response != "":
			response += " "
		response += next_word


	# say something
	print('chatbot>', response)

	# get input
	prompt = input('user>')
