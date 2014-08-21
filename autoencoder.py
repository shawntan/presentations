import math
import theano
import theano.tensor as T
import numpy         as np
import utils         as U
import matplotlib.pyplot as plt
def build_network(input_size,hidden_size):
	X = T.imatrix('X')
	W_input_to_hidden  = U.create_shared(U.initial_weights(input_size,hidden_size))
	W_hidden_to_output = U.create_shared(U.initial_weights(hidden_size,input_size))
	b_output = U.create_shared(U.initial_weights(input_size))

	hidden = T.nnet.sigmoid(T.dot(X,W_input_to_hidden))
	output = T.nnet.softmax(T.dot(hidden,W_input_to_hidden.T) + b_output)

	parameters = [W_input_to_hidden,b_output]

	return X,output,parameters

def build_error(X,output,params):
	return T.mean((X - output)**2) + sum(0.0001*T.sum(p**2) for p in params)

def hinton(matrix, max_weight=None, ax=None):
	"""Draw Hinton diagram for visualizing a weight matrix."""
	ax = ax if ax is not None else plt.gca()

	if not max_weight:
		max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

	ax.patch.set_facecolor('gray')
	ax.set_aspect('equal', 'box')
	ax.xaxis.set_major_locator(plt.NullLocator())
	ax.yaxis.set_major_locator(plt.NullLocator())

	for (x,y),w in np.ndenumerate(matrix):
		color = 'white' if w > 0 else 'black'
		size = np.sqrt(np.abs(w))
		rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
				facecolor=color, edgecolor=color)
		ax.add_patch(rect)

	ax.autoscale_view()
	ax.invert_yaxis()
	return plt.figure()

def run_experiment():
	X,output,parameters = build_network(8,3)
	error = build_error(X,output,parameters)
	grads = T.grad(error,wrt=parameters)
	updates = [ (W,W-grad) for W,grad in zip(parameters,grads) ]
	train = theano.function(
			inputs=[X],
			outputs=error,
			updates=updates
			)
	test = theano.function(
			inputs=[X],
			outputs=output,
			)
	data = np.eye(8,dtype=np.int32)
#	data = np.vstack((data,))
	print "Training..."
	for _ in xrange(100000):
		np.random.shuffle(data)
		train(data)
	#print_arr(test(np.eye(8,dtype=np.int32)))
	#print_arr(1/(1 + np.exp(-parameters[0].get_value())),1)
	return hinton((1/(1 + np.exp(-parameters[0].get_value()))).T)
