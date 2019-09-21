import os, sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score,precision_score, precision_recall_fscore_support, accuracy_score, roc_curve
import utils
from plot import create_plot


def compute_auc(y_gt, pred):
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


def compute_all(y_gt, pred, thresholds = 0.5*np.ones((1, 8))):
    if args["use_gpu"]:
        y_gt_np = y_gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
    else:
        y_gt_np = y_gt.numpy()
        pred_np = pred.numpy()
    num_classes =  y_gt.shape[1]
    #pred_np = pred_np >0.5
    pred_np = pred_np > thresholds
    pred_np = pred_np.astype(np.float32)
    percs = []
    recalls = []
    FBetas = []
    precision, recall , fbeta_score , support  = precision_recall_fscore_support(y_gt_np, pred_np)
    return precision,recall, fbeta_score, support

def WBCE_loss(pred, y_gt, weights=None):
    pred = pred.clamp(min=1e-5, max=1-1e-5)
    y_gt = y_gt.float()
    if weights is not None:
        loss = -weights * (y_gt * torch.log(pred)) - weights * ((1 - y_gt) * torch.log(1 - pred))
    return torch.sum(loss)

class densenet121(nn.Module):
    def __init__(self, out_size, drop_r):
        super(densenet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True, drop_rate = drop_r)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(num_ftrs, out_size),  nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x

class vgg16(nn.Module):
    def __init__(self, out_size):
        super(vgg16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        num_ftrs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, out_size),nn.Sigmoid())

    def forward(self, x):
        x = self.vgg16(x)
        return x

class res50(nn.Module):
    def __init__(self, out_size):
        super(res50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, out_size),nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        return x

class CXRDataSet(Dataset):
    def __init__(self, train_or_valid = "train", transform=None, mask = False, loss = "WBC"):
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
        if loss == "WBC":
            self.weight_pos = (len(self.y)-self.y.sum(axis=0))/len(self.y)
            self.weight_neg = (self.y.sum(axis=0))/len(self.y)
        else :
            self.weight_pos = len(self.y)/(self.y.sum(axis=0))
            self.weight_neg = len(self.y)/(len(self.y)-self.y.sum(axis=0))
        self.transform = transform

    def __getitem__(self, index):
        current_features = np.tile(self.X[index],3) 
        if self.transform is not None:
            image = self.transform(current_features)
        label = self.y[index]
        weight = np.add(((1-label) * self.weight_neg),(label * self.weight_pos))
        return image, torch.from_numpy(label).type(torch.FloatTensor), torch.from_numpy(weight).type(torch.FloatTensor)

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":

    if not os.path.exists("model"):
        os.makedirs("model")

    if not os.path.exists("plots"):
        os.makedirs("plots")

    args = {}
    args["use_gpu"] = torch.cuda.is_available()  
    torch.manual_seed(1)
    if args["use_gpu"]:
        torch.cuda.manual_seed(1)
    args["num_classes"] = 8
    #args["scheduler"] = False
    args["scheduler"] = True 
    #args["loss"] = "BCE"
    #args["loss"] = "WBCE"
    #args["loss"] = "WBCE2"
    args["loss"] = "LSEP"
    #args["loss"] = "MCL"
    #args["reg"] = 0.0 
    args["mask"] = False 
    #args["mask"] = True 
    args["weight_decay"] = 0.0
    #args["weight_decay"] = 0.001
    #args["weight_decay"] = 0.0005
    #args["drop_rate"]=0.0
    args["drop_rate"]=0.3
    #args["model"] = "dense121"
    #args["model"] = "vgg16"
    args["model"] = "resnet50"
    args["lr"] = 0.0002 
    args["epoch"] = 1 
    args["batch_size"] = 16
    loss_history = []
    train_acc_history = []
    train_auc_history = []
    train_recall_history = []
    train_perc_history = []
    val_acc_history = []
    val_auc_history = []
    val_recall_history = []
    val_perc_history = []
    print("number of epoch: ", args["epoch"])
    print("number of classes: ", args["num_classes"])
    print("batch size", args["batch_size"])
    print("model:", args["model"])
    print("loss: ", args["loss"])
    print("learning rate: ", args["lr"])
    print("scheduler: ", args["scheduler"])
    print("weight_decay: ", args["weight_decay"]) 
    print("drop rate: ", args["drop_rate"]) 
    print("mask: ", args["mask"]) 
    # prepare training set
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_dataset = CXRDataSet(train_or_valid="train",
			                   mask = args["mask"],
                               loss=args["loss"],
                               transform = transforms.Compose([
                                           transforms.ToPILImage(),
                                           transforms.RandomCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean,std)
                                            ]))
    augmented_image = []
    augmented_label = []
    augmented_weight = []
    train_length = len(train_dataset)
    for index in range(train_length):
        single_img, single_label, single_weight = train_dataset[index]
        augmented_image.append(single_img)
        augmented_label.append(single_label)
        augmented_weight.append(single_weight)

    shuffled_index = torch.randperm(len(augmented_label))
    augmented_image = torch.stack(augmented_image)[shuffled_index]
    augmented_label = torch.stack(augmented_label)[shuffled_index]
    augmented_weight = torch.stack(augmented_weight)[shuffled_index]

    # prepare validation set
    valid_dataset = CXRDataSet(train_or_valid="valid",
                                mask = args["mask"], loss=args["loss"],
            		            transform=transforms.Compose([
            				    transforms.ToPILImage(),
            				    transforms.CenterCrop(224),
            				    transforms.ToTensor(),
            				    transforms.Normalize(mean,std)
            				]))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=16, shuffle=False, num_workers=4)
    print("============== training starts =================")
    cudnn.benchmark = True
    # initialize and load the model
    if args["model"] == "dense121":
        model = densenet121(args["num_classes"],args["drop_rate"])
        model = torch.nn.DataParallel(model)
    elif args["model"] == "vgg16":
        model = vgg16(args["num_classes"])
        model = torch.nn.DataParallel(model)
    elif args["model"] == "resnet50":
        model = res50(args["num_classes"])
        model = torch.nn.DataParallel(model)

    if args["use_gpu"] :
        model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(),lr = args["lr"], betas=(0.9, 0.999), weight_decay=args["weight_decay"])
    if(args["scheduler"] == True):
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    total_length = len(augmented_image)

    # initialize the ground truth and output tensor
    train_gt = torch.FloatTensor()
    if args["use_gpu"]:
        train_gt = train_gt.cuda()
    train_pred = torch.FloatTensor()
    if args["use_gpu"]:
            train_pred = train_pred.cuda()
    all_thresholds = []
    for epoch in range(args["epoch"]):  # loop over the dataset multiple times
        print("Epoch:",epoch)
        running_loss = 0.0
        # shuffle
        shuffled_index = torch.randperm(len(augmented_label))
        augmented_image = augmented_image[shuffled_index]
        augmented_label = augmented_label[shuffled_index]
        augmented_weight = augmented_weight[shuffled_index]

        for index in range(0, total_length , args["batch_size"]):
            if index+args["batch_size"] > total_length:
            #if index+args["batch_size"] > 200:
            	break
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs_sub = augmented_image[index:index+args["batch_size"]]
            labels_sub = augmented_label[index:index+args["batch_size"]]
            weights_sub = augmented_weight[index:index+args["batch_size"]]
            if args["use_gpu"] :
                inputs_sub, labels_sub = Variable(inputs_sub.cuda()), Variable(labels_sub.cuda())
            else:
                inputs_sub, labels_sub = Variable(inputs_sub), Variable(labels_sub)
            if args["use_gpu"] :
                weights_sub = Variable(weights_sub.cuda())
            else:
                weights_sub = Variable(weights_sub)
            outputs = model(inputs_sub)
            if args["loss"] == "BCE":
                criterion = nn.BCELoss()
                loss = criterion(outputs, labels_sub)
            elif args["loss"] == "WBCE" or args["loss"] == "WBCE2":
                loss = WBCE_loss(outputs, labels_sub, weights_sub)
            elif args["loss"] == "LSEP":
                loss = utils.LSEPLoss()(outputs,labels_sub)

            #l2_reg = 0.0 
            #print("model.parameters",model.parameters()) 
            #for W in model.parameters():
            #    #l2_reg = l2_reg + W.norm(2)
            #    l2_reg = l2_reg + W.norm(2)
            #loss +=  args["reg"]*l2_reg.cpu() 
            loss_history.append(loss)
            loss.backward()
            optimizer.step()
            #running_loss += loss.data[0]
            running_loss += loss.item()

            train_gt = torch.cat((train_gt, labels_sub), 0)
            train_pred = torch.cat((train_pred, outputs), 0)
        train_sum_ones = torch.sum(train_gt, dim=0)
        print("sum_ones for training set", train_sum_ones)
        class_names = ["Atelectasis", "Cardiomegaly","Effusion", "Infiltration","Mass","Nodule", "Pneumonia", "Pneumothorax"]
        train_aurocs = compute_auc(train_gt, train_pred.detach())
        train_auroc_avg = np.array(train_aurocs).mean()
        train_auc_history.append(train_auroc_avg)

        print("The average AUROC is {auroc_avg:.3f}".format(auroc_avg=train_auroc_avg))
        for i in range(args["num_classes"]):
            print("The AUROC of {} is {}".format(class_names[i], train_aurocs[i]))
        train_accs = compute_accs(train_gt, train_pred.detach())
        train_acc_avg = np.array(train_accs).mean()
        train_acc_history.append(train_acc_avg)
        print("The average ACC is {acc_avg:.3f}".format(acc_avg=train_acc_avg))
        for i in range(args["num_classes"]):
            print("The ACC of {} is {}".format(class_names[i], train_accs[i]))
        train_pred = train_pred.detach()
        #======= Find best threshold ====== 
        train_optimal_thresholds = []
        for i in range(0, args["num_classes"]): 
            if args["use_gpu"]:
                fpr, tpr, thresholds = roc_curve(train_gt[:,i].cpu(), train_pred[:,i].cpu())
            else:
                fpr, tpr, thresholds = roc_curve(train_gt[:,i], train_pred[:,i])
            optimal_idx = np.argmax(tpr - fpr)
            t = thresholds[optimal_idx]
            train_optimal_thresholds.append(t)
        #======= End best threshold ====== 

        train_percs, train_recalls , train_FBetas, train_support = compute_all(train_gt, train_pred, train_optimal_thresholds)
        print("support", train_support)
        train_perc_avg = np.array(train_percs).mean()
        train_recall_avg = np.array(train_recalls).mean()
        train_FBeta_avg = np.array(train_FBetas).mean()
        train_perc_history.append(train_perc_avg)
        train_recall_history.append(train_recall_avg)

        print("The average PERCs is {perc_avg:.3f}".format(perc_avg=train_perc_avg))
        for i in range(args["num_classes"]):
            print("The PERC of {} is {}".format(class_names[i], train_percs[i]))

        print("The average RECALL is {recall_avg:.3f}".format(recall_avg=train_recall_avg))
        for i in range(args["num_classes"]):
            print("The RECALL of {} is {}".format(class_names[i], train_recalls[i]))

        print("The average FBeta is {FBeta_avg:.3f}".format(FBeta_avg=train_FBeta_avg))
        for i in range(args["num_classes"]):
            print("The FBeta of {} is {}".format(class_names[i], train_FBetas[i]))
        print("==========training ends ========")

        print("==========validations starts========")
        model.eval()


        # initialize the ground truth and output tensor
        gt = torch.FloatTensor()
        if args["use_gpu"]:
            gt = gt.cuda()
        pred = torch.FloatTensor()
        if args["use_gpu"]:
            pred = pred.cuda()
        for i, (inp, target, weight) in enumerate(valid_loader):
            if args["use_gpu"]:
                target = target.cuda()
            gt = torch.cat((gt, target), 0)
            #     bs, n_crops, c, h, w = inp.size()
            if args["use_gpu"]:
                with torch.no_grad():            
                    input_var = Variable(inp.view(-1, 3, 224, 224).cuda())
            else:
                with torch.no_grad():            
                    input_var = Variable(inp.view(-1, 3, 224, 224))
            output = model(input_var)
            #output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output.data), 0)

        #class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia"                        , "Infiltration", "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]
        class_names = ["Atelectasis", "Cardiomegaly","Effusion", "Infiltration","Mass","Nodule", "Pneumonia", "Pneumothorax"]


        #===================== metrics========================================
        aurocs = compute_auc(gt, pred)
        auroc_avg = np.array(aurocs).mean()
        val_auc_history.append(auroc_avg)
        print("The average AUROC is {auroc_avg:.3f}".format(auroc_avg=auroc_avg))
        for i in range(args["num_classes"]):
            print("The AUROC of {} is {}".format(class_names[i], aurocs[i]))

        accs = compute_accs(gt, pred)
        acc_avg = np.array(accs).mean()
        val_acc_history.append(acc_avg)
        print("The average ACC is {acc_avg:.3f}".format(acc_avg=acc_avg))
        for i in range(args["num_classes"]):
            print("The ACC of {} is {}".format(class_names[i], accs[i]))

        #======= Find best threshold ====== 
        optimal_thresholds = []
        for i in range(0,args["num_classes"]): 
            if(args["use_gpu"]):
                fpr, tpr, thresholds = roc_curve(gt[:,i].cpu(), pred[:,i].cpu())
            else:
                fpr, tpr, thresholds = roc_curve(gt[:,i], pred[:,i])
            optimal_idx = np.argmax(tpr - fpr)
            t = thresholds[optimal_idx]
            optimal_thresholds.append(t)
        print("optimal_thresholds",optimal_thresholds) 
        all_thresholds.append(optimal_thresholds)
        #======= End best threshold ====== 

        percs, recalls , FBetas, support = compute_all(gt, pred, optimal_thresholds)
        print("support", support)
        perc_avg = np.array(percs).mean()
        recall_avg = np.array(recalls).mean()
        FBeta_avg = np.array(FBetas).mean()
        val_perc_history.append(perc_avg)
        val_recall_history.append(recall_avg)

        print("The average PERCs is {perc_avg:.3f}".format(perc_avg=perc_avg))
        for i in range(args["num_classes"]):
            print("The PERC of {} is {}".format(class_names[i], percs[i]))


        print("The average RECALL is {recall_avg:.3f}".format(recall_avg=recall_avg))
        for i in range(args["num_classes"]):
            print("The RECALL of {} is {}".format(class_names[i], recalls[i]))


        print("The average FBeta is {FBeta_avg:.3f}".format(FBeta_avg=FBeta_avg))
        for i in range(args["num_classes"]):
            print("The FBeta of {} is {}".format(class_names[i], FBetas[i]))

        print("==========Validation Ends ========")
        model.train()

        model_saved_path = "model/"+args["model"]+"_mask"+str(args["mask"])+"_"+args["loss"]+"_lr"+str(args["lr"])+"_wd" + str(args["weight_decay"])+"_drop_rate"+str(args["drop_rate"])+"_"+str(auroc_avg)+".pkl"
        torch.save(model.state_dict(),model_saved_path)
    print("==========================Summary==================")
    print("number of epoch", args["epoch"])
    print("model", args["model"])
    print("batch size", args["batch_size"])
    print("mask", args["mask"])
    print("learning rate: ", args["lr"])
    print("scheduler: ", args["scheduler"])
    print("weight_decay", args["weight_decay"]) 
    print("loss function", args["loss"])
    print("drop rate: ", args["drop_rate"]) 
    print("train_auc_history", train_auc_history)
    print("val_auc_history", val_auc_history)
    print("train_acc_history", train_acc_history)
    print("val_acc_history", val_acc_history)
    print("train_perc_history", train_perc_history)
    print("val_perc_history", val_perc_history)
    print("train_recall_history", train_recall_history)
    print("val_recall_history", val_recall_history)
    create_plot(loss_history, args, train_acc_history, val_acc_history,train_auc_history,val_auc_history)
    #all_thresholds
    best_t_index = val_auc_history.index(max(val_auc_history))  
    #print("best_t_index",best_t_index)
    best_t =  all_thresholds[best_t_index] 
    #print("best_t", best_t)
