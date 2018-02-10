import numpy as np
import h5py 
import matplotlib.pyplot as plt


#Going to build a DNN with 3 layers, 2 hidden and 1 output layer

#sigmoid activation function make
def sigmoid(z):
	"""
	Arguments:
	Z -- numpy array of any shape
	Returns:
	A -- output of sigmoid(z), same shape as Z
	"""
	return 1/(1+np.exp(-z))

#relu activation function make
def relu(z):
	"""
	Arguments:
	Z -- Output of the linear layer, of any shape
	Returns:
	A -- Post-activation parameter, of the same shape as Z
	"""
	return max(0,z)

#sigmoid unit for back propagation
def sigmoid_backward(dA, cache):
	"""
	Implement the backward propagation for a single SIGMOID unit.
	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently
	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""
	Z= cache
	#while computing backward prop, we get differential of sigmoid function as backward activation function
	#so, we are calculating the differential of sigmoid this way
	term= 1/(1+ np.exp(-Z))
	dZ= dA* term * (1- term)
	return dZ


#relu function during backward propagation with input dA and output dZ
def relu_backward(dA, cache):
	"""
	Implement the backward propagation for a single RELU unit.
	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently
	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""
	Z= cache
	#dZ initialize as array same as dA and then changed values where dA is negative
	dZ= np.array(dA, copy=True)
	dZ[dZ <= 0]= 0
	return dZ


#layer_dims= [12288, 20, 7, 1]  # 3 layer DNN; input of size (64,64,3)
#initialize parameters W and b


def initialize_parameters(layer_dims):
	"""
	Arguments:
	layer_dims -- python array (list) containing the dimensions of each layer in our network    
	Returns:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
					bl -- bias vector of shape (layer_dims[l], 1)
	"""
	# m is no. of examples
	np.random.seed(7)
	L= len(layer_dims)
	parameters={}

	for l in range(1, L):
		parameters["W"+ str(l)]= np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
		parameters["b"+str(l)]= np.zeros((layer_dims[l], 1))
	return parameters


def linear_activation_forward(A_prev, W, b, activation):
	"""
	Implement the forward propagation for the LINEAR->ACTIVATION layer

	Arguments:
	A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

	Returns:
	A -- the output of the activation function, also called the post-activation value 
	"""
	if(activation =="relu"):
		Z= np.dot(W, A_prev) + b
		linear_cache= (A_prev, W, b)
		A= relu(Z)
		activation_cache= Z

	elif(activation == "sigmoid"):
		Z= np.dot(W, A_prev) + b
		linear_cache= (A_prev, W, b)
		A=sigmoid(Z)
		activation_cache= Z
	cache= (linear_cache, activation_cache)	

	return A, cache


def forward_prop(X, parameters):
	"""
	Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
	
	Arguments:
	X -- data, numpy array of shape (input size, number of examples)
	parameters -- output of initialize_parameters_deep()
	
	Returns:
	AL -- last post-activation value
	"""
	caches = []
	# L is the number of layers in the neural network
	L= len(parameters)//2 
	A= X
	# Implement [LINEAR -> RELU]*(L-1)
	for i in range(1,L):
		A_prev= A
		A, cache= linear_activation_forward(A_prev, parameters["W"+str(i)],parameters["b"+str(i)], "relu")
		caches.append(cache)
	# Implement LINEAR -> SIGMOID
	AL, cache = linear_activation_forward(A, parameters["W"+str(L)],parameters["W"+str(L)],"sigmoid")
	caches.append(cache)
	return AL, caches


def compute_cost(AL, Y):
	"""
	Arguments:
	AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
	Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

	Returns:
	cost -- cross-entropy cost
	"""
	m= Y.shape[1]
	cost= (-1/m)*(np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T))
	cost= np.squeeze(cost)	#this turns [[10]] into 10
	return cost


def linear_backward(dZ, cache):
	"""
	Implement the linear portion of backward propagation for a single layer (layer l)

	Arguments:
	dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	A_prev, W, b= cache
	m= A_prev.shape[1]
	dW= 1./m * np.dot(dZ, A_prev.T)
	db= 1./m * np.sum(dZ, axis=1, keepdims= True)
	dA_prev= np.dot(W.T, dZ)
	return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer.
	
	Arguments:
	dA -- post-activation gradient for current layer l 
	cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	
	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	linear_cache, activation_cache= cache
	if(activation == "relu"):
		dZ= relu_backward(dA, activation_cache)
		dA_prev, dW, db=linear_backward(dZ, linear_cache)
	elif(activation=='sigmoid'):
		dZ= sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db= linear_backward(dZ, linear_cache)
	return dA_prev, dW, db


def backward_prop(AL, Y, caches):
	"""
	Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	
	Arguments:
	AL -- probability vector, output of the forward propagation (L_model_forward())
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	caches -- list of caches containing:
				every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
				the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
	
	Returns:
	grads -- A dictionary with the gradients
			 grads["dA" + str(l)] = ... 
			 grads["dW" + str(l)] = ...
			 grads["db" + str(l)] = ... 
	"""
	grads= {}
	L= len(caches)
	m= AL.shape[1]
	Y= Y.reshape(AL.shape)		 # after this line, Y is the same shape as AL
	#initializing the backpropagation
	dAL= - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
	# Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
	current_cache= caches[L-1]
	grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)]= linear_activation_backward(dAL, current_cache, activation= "sigmoid")
	for l in reversed(range(L-1)):
		# lth layer: (RELU -> LINEAR) gradients.
		current_cache= caches[l]
		dA_prev_temp, dW_temp, db_temp= linear_activation_backward(grads["dA"+str(l+2)] , current_cache, activation= "relu")
		grads["dA"+str(l+1)]= dA_prev_temp
		grads["dW"+str(l+1)]= dW_temp
		grads["db"+str(l+1)]= db_temp 
	return grads


def update_parameters(parameters, grads, learning_rate):
	"""
	Update parameters using gradient descent
	
	Arguments:
	parameters -- python dictionary containing your parameters 
	grads -- python dictionary containing your gradients, output of L_model_backward
	
	Returns:
	parameters -- python dictionary containing your updated parameters 
				  parameters["W" + str(l)] = ... 
				  parameters["b" + str(l)] = ...
	"""
	L= len(parameters)//2
	for i in range(1,L):
		parameters["W"+str(i)]-= learning_rate*grads["dW"+str(i)]
		parameters["b"+str(i)]-= learning_rate*grads["db"+str(i)]
	return parameters


def predict(X, y, parameters):
	"""
	This function is used to predict the results of a  L-layer neural network.
	
	Arguments:
	X -- data set of examples you would like to label
	parameters -- parameters of the trained model
	
	Returns:
	p -- predictions for the given dataset X
	"""

	m= X.shape[1]
	# // is used to floor divide the operand i.e. divide the operand and floor the value
	L= len(parameters) //2       
	predict= np.zeros((1, m))
	probab= forward_prop(X, parameters)
	for i in range(0, probab.shape[1]):
		if(probab[0,i] >=0.5):
			predict[0,i]=1
		else:
			predict[0,i]= 0

	print("Accuracy : "+str(np.sum(predict == y)/m))			
	return predict

