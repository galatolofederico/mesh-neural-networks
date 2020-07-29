import numpy as np
import torch
import random

inputs = 2
hiddens = 2
outputs = 2
batch_size = 10


def f(x, derivative=False):
    gt = (x > 0)
    if derivative:
        return 1 * gt
    else:
        return x * gt


def derivate(grad, state, t):
    sn = np.zeros(shape=grad.shape)
    sn[:,np.eye(neurons).astype(bool)] = state[:,np.newaxis]
    sn = np.transpose(sn, (0,3,2,1))
    return f(t, derivative=True)[:, np.newaxis, np.newaxis] * (np.matmul(grad, np_A) + sn)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.expand_dims(np.sum(exps, axis=1), axis=1)


def ce_loss(out, y, grad):
    y = y.astype(int)
    m = y.shape[0]
    p = softmax(out)
    log_likelihood = -np.log(p[range(m),y])
    loss = np.mean(log_likelihood, axis=0)

    de = p
    de[range(m),y] -= 1
    de = de/m
    de = np.expand_dims(np.expand_dims(de, axis=2), axis=2)
    grad = de*grad

    return (loss, grad.sum(axis=1).sum(axis=0))

def np_net(state, grad, x, bs=batch_size, compute_grad=True):
    state[:,0:inputs] = x
    t = np.matmul(state, np_A)
    grad = derivate(grad, state, t) if compute_grad else None
    state = f(t)
    return state, grad

def tc_net(state, x):
    state[:,0:inputs] = x
    t = torch.matmul(state, tc_A)
    state = torch.relu(t)
    return state


def run_test():
    ticks = random.randint(1, 3)
    np_input_batch = np.random.rand(ticks, batch_size, inputs)
    tc_input_batch = torch.tensor(np_input_batch).float()

    np_output_batch = np.random.randint(0, outputs, (batch_size,))
    tc_output_batch = torch.tensor(np_output_batch)

    np_state = np.zeros((batch_size, neurons))
    tc_state = torch.zeros(batch_size, neurons)

    grad = np.zeros((batch_size, neurons, neurons, neurons))

    for i in range(0, ticks):
        np_state, grad = np_net(np_state, grad, np_input_batch[i])
        tc_state = tc_net(tc_state, tc_input_batch[i])
    
    grad = np.transpose(grad, (0,3,1,2))
    
    state_err = (tc_state.detach().numpy() - np_state).sum()

    np_outs = np_state[:,inputs+hiddens:neurons]
    tc_outs = tc_state[:,inputs+hiddens:neurons]

    outs_grad = grad[:,inputs+hiddens:neurons]

    np_loss, err_grad = ce_loss(np_outs, np_output_batch, outs_grad)

    tc_loss_fn = torch.nn.CrossEntropyLoss()
    tc_loss = tc_loss_fn(tc_outs, tc_output_batch)
    tc_loss.backward()

    err_grad[:,:inputs] = 0
    # Ignore input weigths gradients
    err_grad[:inputs, :] = 0
    tc_A.grad[:inputs, :] = 0

    grad_err = (tc_A.grad.numpy() - err_grad).sum()

    return state_err, grad_err


neurons = inputs + hiddens + outputs
tests = 100

for i in range(0, tests):
    np_A = np.random.rand(neurons, neurons)
    tc_A = torch.tensor(np_A).float().requires_grad_(True)

    state_err, grad_err = run_test()
    print("Test: %d\tState absolute error: %f\tGradient absolute error: %f" % (i, state_err, grad_err))
