import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import matplotlib.pyplot as plt

conv1 = ["hi", "hello", "how are you", "very good and you", "not bad", "what have you been doing today", "not much"
		"same", "i have to go", "good bye", "see you later"]

vocab = {}

def build_vocab(conv):
	new_conv = []
	for utter in conv:
		tokens = utter.split(" ")
		new_utter = []
		for t in tokens:
			if t not in vocab:
				vocab[t] = len(vocab)
			new_utter.append(vocab[t])
		new_conv.append(new_utter)
	return new_conv

conv1_tokenized = build_vocab(conv1)

def vectorize_conv(conv):
	return
			

#Generate 2 sets of X variables
#LSTMs have unique 3-dimensional input requirements 
seq_length=5
X =[[i+j for j in range(seq_length)] for i in range(100)]
X_simple =[[i for i in range(4,104)]]
X =np.array(X)
X_simple=np.array(X_simple)

y =[[ i+(i-1)*.5+(i-2)*.2+(i-3)*.1 for i in range(4,104)]]
y =np.array(y)
X_simple=X_simple.reshape((100,1))
X=X.reshape((100,5,1))
y=y.reshape((100,1))

model = Sequential()
model.add(LSTM(8,input_shape=(5,1),return_sequences=False))#True = many to many
model.add(Dense(2,kernel_initializer='normal',activation='linear'))
model.add(Dense(1,kernel_initializer='normal',activation='linear'))

model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
model.fit(X,y,epochs=2000,batch_size=5,validation_split=0.05,verbose=1)
scores = model.evaluate(X,y,verbose=1,batch_size=5)

print('Accuracy: {}'.format(scores[1])) 

predict=model.predict(X)
plt.plot(y, predict-y, 'C2')
plt.ylim(ymax = 3, ymin = -3)
plt.show()