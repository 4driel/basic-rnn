import numpy as np

# string_2_oneHot
# returns a one-hot vector of string #
###################################################
def string_2_oneHot(string):
    vocab = list(set(string)) #entire character set used on string
    oneHot = np.zeros((len(string), len(vocab)))
    print vocab

    for x in np.arange(len(string)):
        for v in np.arange(len(vocab)):
	    if string[x] == vocab[v]:
		oneHot[x][v] = 1
    return oneHot

# softmax
# calculates the softmax function to the recieved vector #
##########################################################
def softmax(x):
    	e_x = np.exp(x - np.max(x))
    	return e_x / e_x.sum()

# rNewtwork
# models rnn - character model like#
####################################
class rNetwork(object):

	def __init__(self, string, vocab):
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

	def lossFunc(self, x):
		L = 0

		o, s = self.forward(x)

		for i in np.arange(len(x)):
			L += x[i+1]*np.log(o[i])

		L = -L/len(x)

string = "hello"
np.random.seed(10)
rnn = rNetwork(string)

x = string_2_oneHot(string)
o, s = rnn.forward(x[0:(len(x)-1)])
print o
print s


