# Created by: Albert Enyedy and Brian Zylich

# generates batches from convai data

from keras.utils import Sequence
import numpy as np
import linecache
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class ConvAISequence(Sequence):

	def __init__(self, file_name, line_numbers, batch_size):
		self.line_numbers = line_numbers
		self.batch_size = batch_size
		self.file_name = file_name

	def __len__(self):
		return int(np.ceil((len(self.line_numbers) - 1)/ float(self.batch_size)))

	def __getitem__(self, idx):
		info = pickle.load(open("vocab_info.pkl", 'rb'))
		self.vocab = info['vocab']
		self.max_len = info['max_len']

		batch_x = self.line_numbers[idx * self.batch_size: ((idx + 1) * self.batch_size) + 1]
		batch_x = [self.process_line(linecache.getline(self.file_name, line_x)) for line_x in batch_x]
		batch_x = pad_sequences(batch_x, maxlen=self.max_len, padding="post", value=self.vocab['<unk>'])

		batch_y = to_categorical(batch_x[1:], num_classes=len(self.vocab))
		batch_x = batch_x[:-1]

		return np.array(batch_x, dtype=np.int32), np.array(batch_y, dtype=np.int32)

	def process_line(self, line):
		tokens = line.replace("\n", "").lower().split(" ")
		new_tokens = [self.vocab['<s>']]
		for t in tokens:
			if len(new_tokens) > self.max_len - 2:
				break
			if t in self.vocab:
				new_tokens.append(self.vocab[t])
			else:
				new_tokens.append(self.vocab['<unk>'])
		new_tokens.append(self.vocab['<e>'])
		return new_tokens
		