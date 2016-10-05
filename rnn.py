import numpy as np

def oneHot(string, charSet): #returns one-hot vector
    oneHot = np.zeros((len(string), len(charSet))) #init

    for i in np.arange(len(string)):
        for c in np.arange(len(charSet)):
	    if string[i] == charSet[c]:
		oneHot[i][c] = 1
    return oneHot

def getCharSet(string): #returns char set
    charSet = list(set(string))
    return charSet

def softmax(x): #calculates softmax function of vector
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class rnn(object): #character-model-like rnn
    def __init__(self, charSet):
	#Hyper params
	self.iL = len(charSet)
	self.hL = 10
	self.oL = self.iL #input and output layers same size
	#Weights
	self.W_ih = np.random.uniform(-np.sqrt(1/self.iL), -np.sqrt(1/self.iL), (self.iL, self.hL))
	self.W_hh = np.random.uniform(-np.sqrt(1/self.hL), -np.sqrt(1/self.hL), (self.hL, self.hL))
	self.W_ho = np.random.uniform(-np.sqrt(1/self.hL), -np.sqrt(1/self.hL), (self.hL, self.oL))

    def forward(self, x): #forward propagation
	hS = np.zeros((len(x), self.hL)) #size x array lenght, hidden layer size
	hS[-1] = np.zeros(self.hL) #initial hidden state

	o = np.zeros((len(x), self.oL)) #initialize output

	for t in np.arange(len(x)):
	    hS[t] = np.tanh(np.dot(x[t], self.W_ih) + np.dot(hS[t-1], self.W_hh))
	    o[t]  = softmax(np.dot(hS[t], self.W_ho))
	return [o, hS]

    def lossFunc(self, y, o): # calculate cross-entropy loss
	sumt = 0

	for n in np.arange(len(o[:,0])):
	    sumt += np.multiply(y[n], np.log(o[n]))
	loss = -np.sum(sumt/len(o[:,0]))
	return loss

#define string
string = "hello"
#calc character set
charSet = getCharSet(string)	
#get one-hot input and output vectors
x = oneHot(string[:-1], charSet)
y = oneHot(string[1:], charSet)

#create rnn
np.random.seed(10)
rnn = rnn(charSet)

#forward propagation and loss
o, s = rnn.forward(x)
loss = rnn.lossFunc(y, o)

print string
print charSet
print "y"
print y
print "o"
print o
print "loss"
print loss
