import numpy as np
import matplotlib.pyplot as plt

D = np.random.randn(1000,500)
hidden_layer_sizes = [500]*10
nonlinearities = ['tanh']*len(hidden_layer_sizes)

act = {'relu': lambda x:np.maximum(0,x), 'tanh':lambda x:np.tanh(x)}
Hs = {}
for i in xrange(len(hidden_layer_sizes)):
    x = D if i==0 else Hs[i-1]  #input at this layer
    fan_in = x.shape[1]
    fan_out = hidden_layer_sizes[i]
    W = np.random.randn(fan_in,fan_out) * 1.0     #layer initialization

    H = np.dot(x,W)
    H = act[nonlinearities[i]](H)     # nonlinearity
    Hs[i] = H #cache result on this layer

print 'input layer had mean %f and std %f' % (np.mean(D),np.std(D))
layer_means = [np.mean(H) for i,H in Hs.iteritems()]
layer_stds = [np.std(H) for i,H in Hs.iteritems()]
for i,H in Hs.iteritems():
    print 'hidden layer %d had mean %f and std %f' %(i+1, layer_means[i], layer_stds[i])

#plot the means and standard deviations
plt.figure()
plt.subplot(121)
plt.plot(Hs.keys(), layer_means, 'ob--')
plt.title('layer means')
plt.show()
plt.subplot(121)
plt.plot(Hs.keys(), layer_stds, 'or--')
plt.title('layer std')
plt.show()

#plt the raw distributions
plt.figure()
for i,H in Hs.iteritems():
    plt.subplot(1,len(Hs),i+1)
    plt.hist(H.ravel(), 30, range=(-1,1))
plt.show()

