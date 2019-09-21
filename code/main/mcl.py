import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.models
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score,precision_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import itertools
import plot
from plot import create_plot


def compute_aucs(y_gt, pred):
    num_classes =  y_gt.shape[1]
    aurocs = []
    y_gt_np = y_gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(num_classes):
        aurocs.append(roc_auc_score(y_gt_np[:, i], pred_np[:, i]))
    return aurocs


def compute_accs(y_gt, pred):
    num_classes =  y_gt.shape[1]
    accs = []
    if args["use_gpu"]:
       y_gt_np = y_gt.cpu().numpy()
       pred_np = pred.cpu().numpy()
    else:
       y_gt_np = y_gt.numpy()
       pred_np = pred.numpy()
    pred_np = pred_np >0.5
    pred_np = pred_np.astype(np.float32)
    for i in range(num_classes):
        accs.append(accuracy_score(y_gt_np[:, i], pred_np[:, i]))
    return accs





# ====== prepare dataset ======
class CXRDataSet(Dataset):
    def __init__(self, mask = False, train_or_valid = "train", transform=None):
        data_path = "preprocessed_data/" 
        self.train_or_valid = train_or_valid
        if train_or_valid == "train":
            self.X = np.uint8(np.load(data_path + "train_features.npy")*255)
            self.y = np.uint8(np.load(data_path + "train_onehot_label.npy"))
            if mask:
                sub_sample = (self.y.sum(axis=1)!=0)
                self.X = self.X[sub_sample,:]
                self.y = self.y[sub_sample,:]
        else:
            self.X = np.uint8(np.load(data_path + "valid_features.npy")*255)
            self.y = np.uint8(np.load(data_path + "valid_onehot_label.npy"))
        self.transform = transform

    def __getitem__(self, index):
        current_features = np.tile(self.X[index],3) 
        if self.transform is not None:
            image = self.transform(current_features)
        label = self.y[index]
        return image, torch.from_numpy(label).type(torch.FloatTensor) 

    def __len__(self):
        return len(self.y)

def cxr(batch_sz=16, num_workers=2,mask=False):
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    train_dataset = CXRDataSet(mask, train_or_valid="train", 
                                    transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)
                                        ]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batch_sz, shuffle=False, num_workers=4)
    train_loader.num_classes = 8
    # prepare validation set
    valid_dataset = CXRDataSet(train_or_valid="valid",
                                    transform=transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)
                                            ]))

    eval_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size = batch_sz, shuffle=False, num_workers=4)
    eval_loader.num_classes = 8
    return train_loader, eval_loader

class densenet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_dim):
        super(densenet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_dim),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2


#======start : from learners.classification import Learner_Classification
# This file provides the template Learner. The Learner is used in training/evaluation loop
# The Learner implements the training procedure for specific task.
# The default Learner is from classification task.

class Learner_Classification(nn.Module):
    def __init__(self, model, criterion, optimizer, scheduler):
        #2
        super(Learner_Classification, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.KPI = -1  # An non-negative index, larger is better.

    def learn(self, inputs, targets,reg_param=0, **kwargs):
        #4
        loss, out = self.forward_with_criterion(inputs,targets,**kwargs)
        l2_reg = 0.0
        for w in self.model.parameters():
            l2_reg = l2_reg + w.norm(2)
        loss+= reg_param*l2_reg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.detach()
        out = out.detach()
        return loss, out

    def step_schedule(self, epoch):
        #3
        self.epoch = epoch
        self.scheduler.step(self.epoch)
        for param_group in self.optimizer.param_groups:
            print('LR:',param_group['lr'])


class Learner_Clustering(Learner_Classification):
    @staticmethod
    def create_model(model_type,model_name,out_dim):
        #1
        model = densenet121(out_dim=out_dim)

        return model

    def forward(self, x):
        #6
        logits = self.model.forward(x)
        prob = logits
        return prob

    def forward_with_criterion(self, inputs, simi, mask=None, **kwargs):
        #5
        raw_prob = self.forward(inputs)
        prob = torch.sigmoid(raw_prob)
        #normalized_prob =  F.softmax(raw_prob, dim =1) 
        sum_prob = torch.sum(prob, dim=1).view(-1,1)
        normalized_prob = prob/sum_prob
        prob1, prob2 = PairEnum(normalized_prob, mask)
        return self.criterion(prob1, prob2, simi),prob
#====== end: from learners.clustering import Learner_Clustering


# Meta Classification Likelihood (MCL)
class MCL(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        P1 = prob1.mul_(prob2)
        P1 = P1.sum(1)
        L1 = P1.add(MCL.eps)
        L1 = L1.log_() 
        if torch.cuda.is_available():
            L1 = -L1.mul(simi.cuda())
        else:
            L1 = -L1.mul(simi)
        P2 = -P1
        L2 = P2.add(MCL.eps+1.0).log_()
        if torch.cuda.is_available():
            L2 = -L2.mul(1.0-simi.cuda())
        else:    
            L2 = -L2.mul(1.0-simi)
        L = L1 + L2
        return L.mean()
#==== end: import modules.criterion
def onehot_prepare_task_target(onehot_target, args):
    # Prepare the target for different criterion/tasks
    if args["loss"]   == 'MCL':  
        n = onehot_target.shape[0]
        out1 = np.ones((n,n))
        for i in range(0,n):
            for j in range(0,n):
                y_i = np.array(onehot_target[i,:])
                y_j = np.array(onehot_target[j,:])
                # mask is array of elements indices that are eq. to 1
                mask_i = np.where(y_i == 1)[0]
                mask_j = np.where(y_j == 1)[0]
                if np.sum(onehot_target[i,:])== 0 and np.sum(onehot_target[j,:])== 0 :
                    out1[i,j] = 1
                else:
                    length = len(list(set(mask_i) & set(mask_j))) 
                    max_length = max(len(mask_i), len(mask_j)) 
                    out1[i,j] = length / max_length  
        out1 = torch.from_numpy(out1).float()
        train_target = out1.view(-1)
        eval_target = onehot_target 
    return train_target.detach(), eval_target  # Make sure no gradients


def train(epoch, train_loader, learner, args):
    print('==== Epoch:{0} ====')
    print("===== Training Starts=====")
    learner.train()
    learner.step_schedule(epoch)

    # The optimization loop
    train_dataset = train_loader.dataset
    augmented_img = []
    augmented_label = []
    augmented_weight = []
    #print("length",len(train_dataset))
    for j in range(len(train_dataset)):
        single_img, single_label = train_dataset[j]
        augmented_img.append(single_img)
        augmented_label.append(single_label)

    # shuffle data
    perm_index = torch.randperm(len(augmented_label))
    augmented_img = torch.stack(augmented_img)[perm_index]
    augmented_label = torch.stack(augmented_label)[perm_index]

    total_length = len(augmented_img)
    running_loss = 0.0
    perm_index = torch.randperm(len(augmented_label))
    augmented_img = augmented_img[perm_index]
    augmented_label = augmented_label[perm_index]
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    if args["use_gpu"] == True:
        gt = gt.cuda()
        pred = pred.cuda()
    num_classes = args["num_classes"] 
    loss_history = []
    for index in range(0, total_length , args["batch_size"]):
        if index+args["batch_size"] > total_length:
        #if index+args["batch_size"] > 200:
            break
        # zero the parameter gradients
        inputs_sub = augmented_img[index:index+args["batch_size"]]
        labels_sub = augmented_label[index:index+args["batch_size"]]
        inputs_sub, labels_sub = Variable(inputs_sub), Variable(labels_sub)
        # Prepare the inputs
        if args["use_gpu"]:
            inputs_sub = inputs_sub.cuda()
            labels_sub = labels_sub.cuda()
        np_labels_sub = labels_sub.cpu().numpy()
        train_target, eval_target = onehot_prepare_task_target(np_labels_sub, args)
        # train_target has similarity pair
        loss, output  = learner.learn(inputs_sub, train_target, args["reg_param"] )
        gt = torch.cat((gt, labels_sub),0)            
        pred = torch.cat((pred, output),0)            
        loss_history.append(loss)

    N_CLASSES = gt.size()[1]
    #print("===== computing auc ====== ")
    best_aurocs = compute_aucs(gt, pred)
    avg_best_aurocs = np.array(best_aurocs).mean()
    best_perm_auc = range(0,train_loader.num_classes)
    all_permutations = list(itertools.permutations(range(0,train_loader.num_classes)))
    for p in all_permutations:
        idx = list(p)
        aurocs = compute_aucs(gt, pred[:,idx])
        avg_aurocs = np.array(aurocs).mean()
        if(avg_aurocs>avg_best_aurocs):
            best_aurocs = aurocs
            avg_best_aurocs = avg_aurocs
            best_perm_auc = idx
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=avg_best_aurocs))
    CLASS_NAMES = ['Atelectasis', 'Cardiomegaly','Effusion', 'Infiltration',
                        'Mass','Nodule', 'Pneumonia', 'Pneumothorax']
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], best_aurocs[i]))
    #print("avg_best_aurocs", avg_best_aurocs)
    #print("best_perm", best_perm_auc)


    best_accs = compute_accs(gt, pred)
    avg_best_acc = np.array(best_accs).mean()
    best_perm_acc = range(0,train_loader.num_classes) 
    for p in all_permutations:
        idx = list(p)
        #print("p", idx)
        accs = compute_accs(gt, pred[:,idx])
        #print("aucros", acc)
        avg_acc = np.array(accs).mean()
        #print("avg_aucros", avg_accs)
        if(avg_acc>avg_best_acc):
            avg_best_acc = avg_acc
            best_perm_acc = idx
            best_accs = accs
    #print("avg_best_acc", avg_best_acc)
    #print("best_perm", best_perm_acc)
    print("===== compute acc ====== ")
    print('The average ACC is {ACC_avg:.3f}'.format(ACC_avg=avg_best_acc))
    for i in range(N_CLASSES):
        print('The ACC of {} is {}'.format(CLASS_NAMES[i], best_accs[i]))
    print("=====Training ends=====")
    return best_perm_acc, best_perm_auc, loss_history, avg_best_acc,avg_best_aurocs 
        

def evaluate(eval_loader, model, args, best_perm_auc, best_perm_acc):
    #print("model", model)
    # Initialize all meters
    print('====== Validation Starts======')
    model.eval()
    encoder = MultiLabelBinarizer()
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    if args["use_gpu"] == True:
            gt = gt.cuda()
            pred = pred.cuda()
    for i, (input, target) in enumerate(eval_loader):
        # Prepare the inputs
        if args["use_gpu"]:
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda()
        gt = torch.cat((gt, target),0) 

        if args["use_gpu"]:
            with torch.no_grad():
                input_var = Variable(input.view(-1,3,224,224).cuda())
        else:    
            input_var = Variable(input.view(-1,3,224,224))
        #_, eval_target = onehot_prepare_task_target(target, args)
        # Inference
        output = torch.sigmoid(model(input_var)).detach()
        pred = torch.cat((pred, output))
    #print("pred", pred)
    #print("output", output)
    CLASS_NAMES = ['Atelectasis', 'Cardiomegaly','Effusion', 'Infiltration',
                        'Mass','Nodule', 'Pneumonia', 'Pneumothorax']
    num_classes = gt.size()[1]
    aurocs = compute_aucs(gt, pred[:,best_perm_auc])
    #print("aucros", aurocs)
    avg_aurocs = np.array(aurocs).mean()
    #print("avg_aucros", avg_aurocs)
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=avg_aurocs))
    for i in range(num_classes):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], aurocs[i]))

    accs = compute_accs(gt, pred[:,best_perm_acc])
    #print("accs", accs)
    avg_accs = np.array(accs).mean()
    #print("avg_accs", avg_accs)
    print('The average ACC is {ACC_avg:.3f}'.format(ACC_avg=avg_accs))
    for i in range(num_classes):
        print('The ACC of {} is {}'.format(CLASS_NAMES[i], accs[i]))
    print('====== Evaluation ends======')
    return avg_accs, avg_aurocs

def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # Select the optimization criterion/task
    LearnerClass = Learner_Clustering
    criterion = MCL()
    # Prepare dataloaders
    train_loader , eval_loader = cxr(batch_sz = args["batch_size"], num_workers = args["workers"], mask = args["mask"])
    # Prepare the model
    if args["num_classes"]<0:  # Use ground-truth number of classes/clusters
        args["num_classes"] = train_loader.num_classes
    model = LearnerClass.create_model(args["model_type"],args["model"],args["num_classes"])
    model = torch.nn.DataParallel(model)

    # GPU
    if args["use_gpu"]:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.cuda.manual_seed(1)
    print('Criterion:', criterion)


    optim_args = {'lr':args["lr"]}
    #optimizer = torch.optim.__dict__[args["optimizer"]](model.parameters(), **optim_args)
    #optimizer = torch.optim.Adam(model.parameters(),lr=0.0002, betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"], betas=(0.9, 0.999))
    scheduler = MultiStepLR(optimizer, milestones=args["schedule"], gamma=0.1)
    learner = LearnerClass(model, criterion, optimizer, scheduler)
    all_loss_history = []
    train_acc_history = []
    train_auc_history = []
    val_acc_history = []
    val_auc_history = []
    for epoch in range(args["start_epoch"], args["epochs"]):
        best_perm_acc, best_perm_auc,loss_history, avg_best_acc,avg_best_auc = train(epoch, train_loader, learner, args)
        train_acc_history.append(avg_best_acc) 
        train_auc_history.append(avg_best_auc) 
        all_loss_history = all_loss_history + loss_history 
        #KPI = 0 
        val_acc, val_auc = evaluate(eval_loader, model, args,  best_perm_auc, best_perm_acc)
        val_acc_history.append(val_acc) 
        val_auc_history.append(val_auc) 
    print("==========================Summary==================")
    #print("model", args["model"])
    print("lr", args["lr"])
    print("loss", args["loss"])
    print("number of epoch", args["epochs"])
    print("weight_decay", args["weight_decay"])
    print("mask", args["mask"])
    print("train_auc_history", train_auc_history)
    print("val_auc_history", val_auc_history)
    print("train_acc_history", train_acc_history)
    print("val_acc_history", val_acc_history)
    create_plot(all_loss_history, args, train_acc_history, val_acc_history,train_auc_history,val_auc_history)
    torch.save(model, "model/" +args["loss"]+"_mask"+str(args["mask"])+"_wd"+str(args["weight_decay"]) + '.pt')

#def get_args(argv):
if __name__ == '__main__':
    args = {}
    args["gpuid"] = [0]
    args["model_type"] = 'densenet' 
    args["model"] = 'densenet121' 
    args["num_classes"] = 8
    args["workers"] = 4
    args["epochs"] = 2 
    args["batch_size"] = 16 
    args["lr"] = 0.0001 
    args["loss"] = 'MCL' 
    args["schedule"] = [10,20] 
    args["optimizer"] = 'Adam'
    args["print_freq"] = 100
    args["resume"] = ''
    args["use_gpu"] = torch.cuda.is_available()
    args["start_epoch"] = 0
    args["reg_param"] = 0.0 
    args["weight_decay"] = 0.0
    args["mask"]=True
    args["drop_rate"]= 0.0
    print("number of epochs: ", args["epochs"])
    print("number of classes: ", args["num_classes"])
    print("batch size", args["batch_size"])
    print("model:", args["model"])
    print("loss: ", args["loss"])
    print("learning rate: ", args["lr"])
    #print("schedule: ", args["schedule"])
    print("weight_decay: ", args["weight_decay"]) 
    print("mask: ", args["mask"])
    run(args)

