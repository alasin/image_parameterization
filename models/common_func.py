import torch
# import torch.nn as nn
# import numpy as np
import torch.nn.functional as F
import torch_sparse


def act_f_leaky(x):
    x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
    return x

def act_f(x):
    x = F.relu(x, inplace=True)
    return x


def conv_f_new(x, stride, kernel_size, layer, subspace, pad='reflection'):
    weight = layer['weight'].view(layer['w_shape'])
    bias = layer['bias'].view(layer['b_shape'])
    
    to_pad = int((kernel_size - 1) / 2)
    
    x = F.pad(x, (to_pad, to_pad, to_pad, to_pad), mode='reflect')
    x = F.conv2d(x, weight, bias, stride)
  
    return x

def conv_f(x, stride, kernel_size, layer, transform, subspace, pad='reflection'):
    #weight = torch.matmul(layer['weight'], subspace).view(layer['w_shape'])
    #bias = torch.matmul(layer['bias'], subspace).view(layer['b_shape'])
    w0 = layer.weight
    b0 = layer.bias

    i, v = transform['weight']._indices(), transform['weight']._values()
    weight = torch_sparse.spmm(i, v, transform['w_num'], subspace).view(transform['w_shape'])
    # weight += w0

    i, v = transform['bias']._indices(), transform['bias']._values()
    bias = torch_sparse.spmm(i, v, transform['b_num'], subspace).view(transform['b_shape'])
    # bias += b0
    # weight = torch.sparse.mm(layer['weight'], subspace).view(layer['w_shape'])
    # bias = torch.sparse.mm(layer['bias'], subspace).view(layer['b_shape'])
    
    to_pad = int((kernel_size - 1) / 2)
    
    x = F.pad(x, (to_pad, to_pad, to_pad, to_pad), mode='reflect')
    x = F.conv2d(x, weight, bias, stride)
    
    return x


def bn_f_new(x, layer, bn_module, subspace):
    run_mean = bn_module.running_mean
    run_var = bn_module.running_var
    
    weight = layer['weight'].view(layer['w_shape'])
    bias = layer['bias'].view(layer['b_shape'])
    y = F.batch_norm(x, run_mean, run_var, weight, bias, training=True)
    dummy_y = bn_module(x)

    return y

def bn_f(x, transform, bn_module, subspace):
    run_mean = bn_module.running_mean
    run_var = bn_module.running_var
    
    w0 = bn_module.weight
    b0 = bn_module.bias
    #weight = torch.matmul(layer['weight'], subspace).view(layer['w_shape'])
    #bias = torch.matmul(layer['bias'], subspace).view(layer['b_shape'])
    # weight = torch.sparse.mm(layer['weight'], subspace).view(layer['w_shape'])
    # bias = torch.sparse.mm(layer['bias'], subspace).view(layer['b_shape'])
    i, v = transform['weight']._indices(), transform['weight']._values()
    weight = torch_sparse.spmm(i, v, transform['w_num'], subspace).view(transform['w_shape'])
    # weight += w0

    i, v = transform['bias']._indices(), transform['bias']._values()
    bias = torch_sparse.spmm(i, v, transform['b_num'], subspace).view(transform['b_shape'])
    # bias += b0
    
    y = F.batch_norm(x, run_mean, run_var, weight, bias, training=True)
    dummy_y = bn_module(x)

    return y