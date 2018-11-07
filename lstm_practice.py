import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Embedding, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
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
		new_conv.append(new_utter)
		if expand_vocab and len(new_utter) > max_len:
			max_len = len(new_utter)
	return new_conv

conv1_tokenized = build_vocab(conv1)
reverse_vocab = {key: word for (word, key) in vocab.items()}
# print(vocab)
# print(conv1_tokenized)

conv1_padded = list(pad_sequences(conv1_tokenized, padding="post", value=vocab['<unk>']))
# print(conv1_padded)
# print(np.shape(conv1_padded))

def vectorize_conv(conv):
	new_conv = []
	for utter in conv:
		new_utter = []
		for t in utter:
			new_token = []
			for i in range(len(vocab)):
				if i == t:
					new_token.append(1)
				else:
					new_token.append(0)
			new_utter.append(new_token)
		new_conv.append(new_utter)
	return new_conv

conv1_padded = np.array(conv1_padded)
conv1_vectorized = np.array(vectorize_conv(conv1_padded))

print("vocab size:", len(vocab))
print("max seq len:", max_len)

X_train = conv1_padded[:-1] #conv1_vectorized[:-1]
y_train = conv1_vectorized[1:] #conv1_padded[1:]

# define the model architecture
model = Sequential()
model.add(Embedding(len(vocab), 64, input_length=max_len))
model.add(LSTM(16, return_sequences=True, stateful=False))
model.add(LSTM(16, return_sequences=True, stateful=False))
# model.add(TimeDistributed(Dense(10, activation='relu')))
model.add(TimeDistributed(Dense(len(vocab), activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics=['accuracy'])

filepath="./saved_models/lstm_practice.hdf5"
if os.path.exists(filepath):
	model.load_weights(filepath)

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# only train the model if a saved version is not found or an extra argument is given on the command line
if len(sys.argv) < 2 or not os.path.exists(filepath):
	model.fit(X_train, y_train, epochs=200, batch_size=5, validation_split=0.05, verbose=1, callbacks=callbacks_list)

scores = model.evaluate(X_train, y_train, verbose=1)


prompt = ""
while True:
	# prime the agent with the entire prompt
	prompt_tokenized = build_vocab([prompt], expand_vocab=False)
	prompt_padded = pad_sequences(prompt_tokenized, maxlen=max_len, padding="post", value=vocab['<unk>'])
	response = ""
	while True:
		next_words = list(model.predict(prompt_padded)[0])
		# print(next_words)
		# print(np.shape(next_words))
		for w in next_words:
			next_word = reverse_vocab[np.argmax(w)]
			if next_word == "<s>":
				continue
			if next_word == "<e>":
				break
			if response != "":
				response += " "
			response += next_word
		break


	# say something
	print('chatbot>', response)

	# get input
	prompt = input('user>')
