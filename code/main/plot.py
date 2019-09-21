import numpy as np
import matplotlib.pyplot as plt
import  matplotlib

#plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
def create_plot(loss_history, args, train_acc, val_acc,train_auc,val_auc):
    fig0 = plt.figure(0)
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("plots/loss_"+args["model"]+"_mask"+str(args["mask"])+"_"+args["loss"]+"_lr"+str(args["lr"])+'_'+"_wd" + str(args["weight_decay"])+"_droprate"+str(args["drop_rate"])+".png") 

    fig1 = plt.figure(1)
    plt.plot(train_acc)
    plt.title("Accuracy")
    plt.plot(val_acc)
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.savefig("plots/acc_"+args["model"]+"_mask"+str(args["mask"])+"_"+args["loss"]+"_lr"+str(args["lr"])+'_'+"_wd" + str(args["weight_decay"])+"_droprate"+str(args["drop_rate"])+".png") 


    
    fig2 = plt.figure(2)
    plt.plot(train_auc)
    plt.title("AUROC")
    plt.plot(val_auc)
    plt.legend(['Training AUROC', 'Validation AUROC'], loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.savefig("plots/auc_"+args["model"]+"_mask"+str(args["mask"])+"_"+args["loss"]+"_lr"+str(args["lr"])+'_'+"_wd" + str(args["weight_decay"])+"_droprate"+str(args["drop_rate"])+".png") 


