import numpy as np
import pickle
from scipy.special import softmax


dataset_folder = "datasets/mnist"
zero_prob=0.85
inputs=32 + 1
hidden=50
outputs=10
batch_size=32
ticks = 3
epochs = 10
lr = 0.001


train_pkl = pickle.load(open(dataset_folder+"/train.pkl", "rb"))
test_pkl = pickle.load(open(dataset_folder+"/test.pkl", "rb"))

train_X = np.stack(train_pkl["x"], axis=0)
train_Y = np.array(train_pkl["y"])


test_X = np.stack(test_pkl["x"], axis=0)
test_Y = np.array(test_pkl["y"])

def f(x, derivative=False):
    gt = (x > 0)
    if derivative:
        return 1 * gt
    else:
        return x * gt

def ce_loss(out, y, grad):
    y = y.astype(int)
    m = y.shape[0]
    p = softmax(out, axis=1) + np.finfo(float).eps
    log_likelihood = -np.log(p[range(m),y])
    loss = np.mean(log_likelihood, axis=0)

    de = p
    de[range(m),y] -= 1
    de = de/m
    de = np.expand_dims(np.expand_dims(de, axis=2), axis=2)
    grad = de*grad

    return (loss, grad.sum(axis=1).sum(axis=0))

class Adam:
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0., **kwargs):
        
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))

        self.__dict__.update(kwargs)
        self.iterations = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay

    def step(self, params, grads):
        original_shapes = [x.shape for x in params]
        params = [x.flatten() for x in params]
        grads = [x.flatten() for x in grads]
        

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))

        if not hasattr(self, 'ms'):
            self.ms = [np.zeros(p.shape) for p in params]
            self.vs = [np.zeros(p.shape) for p in params]
    
        ret = [None] * len(params)
        for i, p, g, m, v in zip(range(len(params)), params, grads, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
            self.ms[i] = m_t
            self.vs[i] = v_t
            ret[i] = p_t
        
        self.iterations += 1
        
        for i in range(len(ret)):
            ret[i] = ret[i].reshape(original_shapes[i])
        
        return ret

def derivate(grad, t):
    sn = np.zeros(shape=grad.shape)
    sn[:,np.eye(neurons).astype(bool)] = state[:,np.newaxis]
    sn = np.transpose(sn, (0,3,2,1))
    return f(t, derivative=True)[:, np.newaxis, np.newaxis] * (np.matmul(grad, A) + sn)


def net(state, grad, x):
    # set inputs neurons state with input vector
    state[:,0:inputs] = np.concatenate((x, np.ones((batch_size,1))), axis=1)
    
    # compute ti
    t = np.matmul(state, A)
    
    # Forward propagate the gradients
    grad = derivate(grad, t)
    
    # Compute new state
    state = f(t)
    
    return state, grad


neurons = inputs + hidden + outputs
A = np.random.rand(neurons, neurons) # Adjacency Matrix

A_mask = np.random.rand(neurons, neurons)
A_mask = (A_mask <= zero_prob).astype(int)
A_mask[:, :inputs] = 0
A_mask[inputs+hidden:, :] = 0

A = A*A_mask

optimizer = Adam(lr=lr)
for epoch in range(0, epochs):
    losses = 0
    for i in range(0, len(train_X)//batch_size):
        # Training batches
        batch_X = train_X[i*batch_size:(i+1)*batch_size]
        batch_Y = train_Y[i*batch_size:(i+1)*batch_size]
        
        state = np.zeros(shape=(batch_size, neurons)) # Init state
        grad = np.zeros(shape=(batch_size, neurons, neurons, neurons)) # Init gradients

        # Update MNN in time
        for t in range(0, ticks):
            state, grad = net(state, grad, batch_X)
        
        # Permute gradients for error function
        grad = np.transpose(grad, (0,3,1,2))
        
        # Slice output values and gradients
        outs = state[:,inputs+hidden:neurons]
        outs_grad = grad[:,inputs+hidden:neurons]
        
        # Compute loss and error gradients
        loss, err_grad = ce_loss(outs, batch_Y, outs_grad)
        losses += loss

        # Update weights
        A = optimizer.step(A, err_grad)
        
        # Zero pruned weights
        A = A*A_mask

    #if epoch % 50 == 0:
        print("epoch: %d  perc: %.2f loss: %f" % (epoch, (100*i/(len(train_X)//batch_size)), loss))


rights = 0
tots = 0
for i in range(0, len(test_X)//batch_size):
    # Testing batches
    batch_X = test_X[i*batch_size:(i+1)*batch_size]
    batch_Y = test_Y[i*batch_size:(i+1)*batch_size]
    
    # Init MNN
    state = np.zeros(shape=(batch_size, neurons))
    grad = np.zeros(shape=(batch_size, neurons, neurons, neurons))
    
    for t in range(0, ticks):
        state, grad = net(state, grad, batch_X)
    outs = state[:,inputs+hidden:neurons]
    
    rights += (np.argmax(outs, axis=1) == batch_Y).astype(int).sum()   
    tots += batch_size
    
print("Accuracy: %f" % (rights/tots) )
