import numpy as np

def string_2_oneHot(string):
	vocab = list(set(string))
	oneHot = np.zeros((len(string), len(vocab)))
	print vocab

	for x in np.arange(len(string)):
		for v in np.arange(len(vocab)):
			if string[x] == vocab[v]:
				oneHot[x][v] = 1
	
	return oneHot

def softmax(x):
    	e_x = np.exp(x - np.max(x))
    	return e_x / e_x.sum()

class rNetwork(object):
	def __init__(self, string):
		#Hyper params
		self.iL = len(list(set(string))) #input size of string
		self.hL = 10
		self.oL = self.iL #input and output layers same size

		#Weights
		self.W_ih = np.random.uniform(-np.sqrt(1/self.iL), -np.sqrt(1/self.iL), (self.iL, self.hL))
		self.W_hh = np.random.uniform(-np.sqrt(1/self.hL), -np.sqrt(1/self.hL), (self.hL, self.hL))
		self.W_ho = np.random.uniform(-np.sqrt(1/self.hL), -np.sqrt(1/self.hL), (self.hL, self.oL))

	def forward(self, x):
		hS = np.zeros((len(x), self.hL)) #size x array lenght, hidden layer size
		hS[-1] = np.zeros(self.hL) #initial hidden state

		o = np.zeros((len(x), self.oL))

		for t in np.arange(len(x)):
			hS[t] = np.tanh(np.dot(x[t], self.W_ih) + np.dot(hS[t-1], self.W_hh))
			o[t]  = softmax(np.dot(hS[t], self.W_ho))
		return [o, hS]

string = "hello"
np.random.seed(10)
rnn = rNetwork(string)

o, s = rnn.forward(string_2_oneHot(string))
print o
print s
