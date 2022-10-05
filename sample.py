import torch
from torch import nn
import torchvision 
from torchvision import transforms
import cfg
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
import PIL

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        backbone=torchvision.models.vgg19(pretrained=True)
        for params in backbone.parameters():
            params.requires_grad=False
        self.backbone=backbone.features
        #1_2,2_2,3_2,4_2,5_2
        self.layers=[3,8,13,22,31]
    def forward(self,x):
        outs=[]
        for i,layer in enumerate(self.backbone):
            x=layer(x)
            if i in self.layers:
                outs.append(x)
        return outs

def show_act(X):
    act=torch.linalg.norm(X,dim=0)
    act=act.reshape([X.shape[1],X.shape[2]])
    plt.imshow(act.cpu(),cmap='gray')

def feat_transform(X,Y):
    H=X.shape[1]
    W=X.shape[2]
    X=X.flatten(1)
    Y=Y.flatten(1)
    A=torch.matmul(Y, X.T).cpu()
    U,sigma,V=torch.svd(A)
    P=torch.matmul(U, V.T)
    P=P.to(cfg.device)
    X=torch.matmul(P, X)
    X=X.reshape([-1,H,W])
    return X

def patch_dist(patch_x,Y):
    patch_x=patch_x.repeat(1,16,16)
    diff=torch.linalg.norm(patch_x-Y,dim=0)
    diffs=torch.zeros([16,16]).to(cfg.device)
    for i in range(4):
        for j in range(4):
            diffs[:,:]+=diff[i::4,j::4]
    return diffs

def get_prob_map(content,style):
    extractor=VGG().to(cfg.device)
    X=extractor(content)[-2].squeeze()
    Y=extractor(style)[-2].squeeze()
    X=X/torch.linalg.norm(X,dim=0)
    Y=Y/torch.linalg.norm(Y,dim=0)
    C=X.shape[0]
    H=X.shape[1]
    W=X.shape[2]
    X=X.unsqueeze(0)
    Y=Y.unsqueeze(0)
    X_pad=nn.ReflectionPad2d(1)(X).squeeze()
    Y_pad=nn.ReflectionPad2d(1)(Y).squeeze()
    
    Y_map=torch.zeros([C,H*3,W*3])
    for i in range(H):
        for j in range(W):
            Y_map[:,i*3:i*3+3,j*3:j*3+3]=Y_pad[:,i:i+3,j:j+3]
    
    Y_map=Y_map.to(cfg.device)    
    probs=torch.zeros([H,W])  
    for i in range(H):
        for j in range(W):
            patch=X_pad[:,i:i+3,j:j+3]
            patch=patch.repeat(1,H,W)
            diff=torch.linalg.norm(patch-Y_map,dim=0)
            total_diff=torch.zeros(H,W).to(cfg.device)
            for k in range(3):
                for l in range(3):
                    total_diff+=diff[k::3,l::3]
            index=torch.argmin(total_diff)
            x=index//H
            y=index%W
            probs[x,y]+=1
    probs=probs.unsqueeze(0).unsqueeze(0)
    probs=F.upsample(probs,size=(content.shape[-2],content.shape[-1])).squeeze()
    probs_map=[]
    for i in range(content.shape[-2]):
        for j in range(content.shape[-1]):
            while probs[i,j]>0:
                probs_map.append([i,j])
                probs[i,j]-=1
    return probs_map



