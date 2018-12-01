import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Embedding, TimeDistributed, Lambda, Dropout
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import os.path
import sys
import pickle
# from convai_generator import ConvAISequence
from multiprocessing import freeze_support

def store_vocab(file_name, conv, max_len):
	global vocab
	# global max_len
	vocab['<s>'] = 0
	vocab['<e>'] = 1
	vocab['<unk>'] = 2
	for utter in conv:
		tokens = utter.split(" ")
		# num_tokens = 2
		for t in tokens:
			if t not in vocab:
				vocab[t] = len(vocab)
			# num_tokens += 1
		# if num_tokens > max_len:
		# 	max_len = num_tokens

	with open(file_name, 'wb') as pkl_file:
		pickle.dump({"vocab": vocab, "max_len": max_len, "num_examples": num_training_samples}, pkl_file)

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


if __name__ == '__main__':
	freeze_support()

	conversations = []

	vocab_filepath = "vocab_info.pkl"
	dialogue_filepath = "training_dialogue_plus_personas.txt"

	if not os.path.exists(vocab_filepath):
		print("Loading data...")
		with open(dialogue_filepath, "w") as dialogue_out:
			for convai_file in ["./train_both_original_no_cands.txt", "./valid_both_original_no_cands.txt"]:
				with open(convai_file) as raw_convai:
					your_persona = ""
					partner_persona = ""
					in_persona = False
					for line in raw_convai:
						line = line.replace("\n", "")
						if not "your persona:" in line and not "partner's persona:" in line:
							if in_persona:
								dialogue_out.write("<p1>:" + your_persona + "\n")
								dialogue_out.write("<p2>:" + partner_persona + "\n")
							in_persona = False
							words = line.split(" ")
							line = " ".join(words[1:])
							parts = line.split("\t")
							for p in parts:
								if "__SILENCE__" not in p:
									# conversations.append(p)
									dialogue_out.write(p + '\n')
						elif "your persona:" in line:
							if not in_persona:
								your_persona = ""
								partner_persona = ""
								in_persona = True
							your_persona += line.split("your persona:")[1]
						else:
							if not in_persona:
								your_persona = ""
								partner_persona = ""
								in_persona = True
							partner_persona += line.split("partner's persona:")[1]

		print("Data loaded :)")

		# num_training_samples = len(conversations)

	# vocab = {}
	# max_len = 25

	# if not os.path.exists(vocab_filepath):
	# 	store_vocab(vocab_filepath, conversations, max_len)
	# else:
	# 	info = pickle.load(open(vocab_filepath, 'rb'))
	# 	vocab = info['vocab']
	# 	max_len = info['max_len']
	# 	num_training_samples = info['num_examples']

	# # conversations = conversations[:2000]

	# print("Building vocab...")

	# # conv1_tokenized = build_vocab(conversations, expand_vocab=False)
	# reverse_vocab = {key: word for (word, key) in vocab.items()}

	# # print("Padding sequences...")

	# # conv1_padded = list(pad_sequences(conv1_tokenized, padding="post", value=vocab['<unk>']))

	# # print("Vectorizing sequences...")

	# # conv1_padded = np.array(conv1_padded)
	# # conv1_vectorized = np.array(vectorize_conv(conv1_padded))

	# print("vocab size:", len(vocab))
	# print("max seq len:", max_len)

	# # X_train = conv1_padded[:-1] #conv1_vectorized[:-1]
	# # y_train = conv1_vectorized[1:] #conv1_padded[1:]

	# batch_size = 50

	# # define the model architecture
	# model = Sequential()
	# model.add(Embedding(len(vocab), 512, input_length=max_len))
	# model.add(LSTM(512, return_sequences=True, stateful=False))
	# model.add(LSTM(512, return_sequences=True, stateful=False))
	# model.add(TimeDistributed(Dense(256, activation='relu')))
	# model.add(Dropout(0.2))
	# model.add(TimeDistributed(Dense(len(vocab), activation='softmax')))

	# model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics=['accuracy'])

	# filepath="./saved_models/convai_practice.hdf5"
	# if os.path.exists(filepath):
	# 	model.load_weights(filepath)

	# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	# callbacks_list = [checkpoint]


	# # only train the model if a saved version is not found or an extra argument is given on the command line
	# if len(sys.argv) < 2 or not os.path.exists(filepath):
	# 	# model.fit(X_train, y_train, epochs=200, batch_size=batch_size, validation_split=0.0, verbose=1, callbacks=callbacks_list)
	# 	my_training_generator = ConvAISequence(dialogue_filepath, [i for i in range(num_training_samples)], batch_size)
	# 	model.fit_generator(generator=my_training_generator, steps_per_epoch=(num_training_samples // batch_size),
	# 		epochs=200, verbose=1, use_multiprocessing=True, workers=3, callbacks=callbacks_list)

	# # scores = model.evaluate(X_train, y_train, verbose=1)

	# prompt = ""
	# while True:
	# 	# prime the agent with the entire prompt
	# 	prompt_tokenized = build_vocab([prompt], expand_vocab=False)
	# 	prompt_padded = pad_sequences(prompt_tokenized, maxlen=max_len, padding="post", value=vocab['<unk>'])
	# 	response = ""
	# 	while True:
	# 		next_words = list(model.predict(prompt_padded)[0])
	# 		# print(next_words)
	# 		# print(np.shape(next_words))
	# 		for w in next_words:
	# 			next_word = reverse_vocab[np.argmax(w)]
	# 			if next_word == "<s>":
	# 				continue
	# 			if next_word == "<e>":
	# 				break
	# 			if response != "":
	# 				response += " "
	# 			response += next_word
	# 		break


	# 	# say something
	# 	print('chatbot>', response)

	# 	# get input
	# 	prompt = input('user>')
	# 	# prompt = prompt.lower().replace(".", " .").replace("?", " ?").replace("!", " !").replace(",", " ,")
