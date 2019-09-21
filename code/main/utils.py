import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable, Function
import random


"""
 Log-Sum-Exp-Pairwise Loss 
 - Reference: https://arxiv.org/pdf/1704.03135.pdf
"""
def to_one_hot(y, num_classes):
    y_tensor = y.data.type(torch.LongTensor).view(-1, 1)
    return Variable(torch.zeros(y_tensor.size()[0], num_classes).scatter_(1, y_tensor, 1))

class LSEP(Function): 

    @staticmethod
    def forward(ctx, input, target):

        loss = 0.0
        batch_size,num_classes = target.size()[0], target.size()[1]
        positive_indices_boolean , negative_indices_boolean = target.gt(0).float(), target.eq(0).float()
        for i in range(batch_size): 
            pos_indices = np.array([j for j,pos in enumerate(positive_indices_boolean[i]) if pos != 0])
            neg_indices = np.array([j for j,neg in enumerate(negative_indices_boolean[i]) if neg != 0])
            for p_value in pos_indices:
                for n_value in neg_indices:
                    loss += np.exp(input[i,n_value]-input[i,p_value])
        
        loss = torch.from_numpy(np.array([np.log(1 + loss)])).float()
        
        ctx.save_for_backward(input, target)
        ctx.loss = loss
        ctx.positive_indices_boolean , ctx.negative_indices_boolean= positive_indices_boolean, negative_indices_boolean
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_variables
        batch_size , num_classes = input.size()[0], input.size()[1]
        loss = Variable(ctx.loss, requires_grad = False)
        positive_indices_boolean , negative_indices_boolean = ctx.positive_indices_boolean, ctx.negative_indices_boolean
        grad_input = torch.zeros(input.size())
        
        one_hot_pos, one_hot_neg = [],[]
        for i in range(batch_size): 
            pos_indices = np.array([j for j,pos in enumerate(positive_indices_boolean[i]) if pos != 0])
            neg_indices = np.array([j for j,neg in enumerate(negative_indices_boolean[i]) if neg != 0])

            if len(pos_indices) !=0 :
                one_hot_pos.append(to_one_hot(torch.from_numpy(pos_indices), num_classes))
                one_hot_neg.append(to_one_hot(torch.from_numpy(neg_indices), num_classes))
            else:
                one_hot_pos.append( Variable(torch.FloatTensor()) )
                one_hot_neg.append(  Variable(torch.FloatTensor())  )
        ## grad
        for i in range(batch_size):
            for phot in one_hot_pos[i]:
                for nhot in one_hot_neg[i]:
                    grad_input[i] += (phot-nhot)*torch.exp(-input[i].data*(phot-nhot))
        grad_input = Variable(grad_input)* (grad_output * (-1.0 / loss))

        return grad_input, None, None
    
class LSEPLoss(nn.Module): 
    def __init__(self): 
        super(LSEPLoss, self).__init__()
        
    def forward(self, input, target): 
        return LSEP.apply(input.cpu(), target.cpu())



