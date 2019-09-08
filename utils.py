import torchvision
import os
import torch
from pylab import plt


def load_resnet50_param(model, stop_layer='layer4'):
    resnet50 = torchvision.models.resnet50(pretrained=True)
    saved_state_dict = resnet50.state_dict()
    new_params = model.state_dict().copy()

    for i in saved_state_dict:  # copy params from resnet50,except layers after stop_layer

        i_parts = i.split('.')

        if not i_parts[0] == stop_layer:

            new_params['.'.join(i_parts)] = saved_state_dict[i]
        else:
            break
    model.load_state_dict(new_params)
    model.train()
    return model


def check_dir(checkpoint_dir):#create a dir if dir not exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(os.path.join(checkpoint_dir,'model'))
        os.makedirs(os.path.join(checkpoint_dir,'pred_img'))




def optim_or_not(model, yes):
    for param in model.parameters():
        if yes:
            param.requires_grad = True
        else:
            param.requires_grad = False


def turn_off(model):
    optim_or_not(model.module.conv1, False)
    optim_or_not(model.module.layer1, False)
    optim_or_not(model.module.layer2, False)
    optim_or_not(model.module.layer3, False)




def get_10x_lr_params(model):
    """
    get layers for optimization
    """

    b = []
    b.append(model.module.layer5.parameters())
    b.append(model.module.layer55.parameters())
    b.append(model.module.layer6_0.parameters())
    b.append(model.module.layer6_1.parameters())
    b.append(model.module.layer6_2.parameters())
    b.append(model.module.layer6_3.parameters())
    b.append(model.module.layer6_4.parameters())
    b.append(model.module.layer7.parameters())
    b.append(model.module.layer9.parameters())
    b.append(model.module.residule1.parameters())
    b.append(model.module.residule2.parameters())
    b.append(model.module.residule3.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i




def loss_calc_v1(pred, label, gpu):

    label = label.long()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda(gpu)

    return criterion(pred, label)



def plot_loss(checkpoint_dir,loss_list,save_pred_every):
    x=range(0,len(loss_list)*save_pred_every,save_pred_every)
    y=loss_list
    plt.switch_backend('agg')
    plt.plot(x,y,color='blue',marker='o',label='Train loss')
    plt.xticks(range(0,len(loss_list)*save_pred_every+3,(len(loss_list)*save_pred_every+10)//10))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir,'loss_fig.pdf'))
    plt.close()


def plot_iou(checkpoint_dir,iou_list):
    x=range(0,len(iou_list))
    y=iou_list
    plt.switch_backend('agg')
    plt.plot(x,y,color='red',marker='o',label='IOU')
    plt.xticks(range(0,len(iou_list)+3,(len(iou_list)+10)//10))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(checkpoint_dir,'iou_fig.pdf'))
    plt.close()



def get_iou_v1(query_mask,pred_label,mode='foreground'):#pytorch 1.0 version
    if mode=='background':
        query_mask=1-query_mask
        pred_label=1-pred_label
    num_img=query_mask.shape[0]#batch size
    num_predict_list,inter_list,union_list,iou_list=[],[],[],[]
    for i in range(num_img):
        num_predict=torch.sum((pred_label[i]>0).float()).item()
        combination = (query_mask[i] + pred_label[i]).float()
        inter = torch.sum((combination == 2).float()).item()
        union = torch.sum((combination ==1).float()).item()+torch.sum((combination ==2).float()).item()
        if union!=0:
            inter_list.append(inter)
            union_list.append(union)
            iou_list.append(inter/union)
            num_predict_list.append(num_predict)
        else:
            inter_list.append(inter)
            union_list.append(union)
            iou_list.append(0)
            num_predict_list.append(num_predict)
    return inter_list,union_list,iou_list,num_predict_list