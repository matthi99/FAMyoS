# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:56:14 2024

@author: A0067501
"""

import yaml
import torch
import os
import pydicom
import numpy as np
from skimage.measure import label  
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import logging
import sys
from nets import UNet, UNet2D

def get_network(architecture="unet", device="cuda:0", **kwargs):
    architecture = architecture.lower()
    if architecture == "unet":
        net = UNet(**kwargs)
    elif architecture == "unet2d":
        net = UNet2D(**kwargs)

    else:
        net = UNet(**kwargs)
    return net.to(device=device)

def constant_pad(x, size, c=2048):
    padding_size = ((0, size - x.shape[0]), (0, size - x.shape[1]))
    return np.pad(x, padding_size, mode='constant', constant_values=c)

def get_data(root, filenames):
    data=[]
    for file in filenames:
        data.append(pydicom.dcmread(os.path.join(root, file)))
    data = sorted(data, key = lambda k: abs(k.get('SliceLocation')))
    pixel_spacing=[]
    zspacing=[]
    X=np.zeros((len(data),256,256))
    for i in range(len(data)):
        X[i,...]=normalize(constant_pad(data[i].pixel_array, 256, c=np.mean(data[i].pixel_array) ))
        pixel_spacing.append(data[i].PixelSpacing)
        try:
            zspacing.append(float(data[i].SpacingBetweenSlices))
        except:
            if i<(len(data)-1):
                cosines=data[0].ImageOrientationPatient
                normal=np.zeros(3)
                normal[0] = cosines[1]*cosines[5] - cosines[2]*cosines[4]
                normal[1] = cosines[2]*cosines[3] - cosines[0]*cosines[5]
                normal[2] = cosines[0]*cosines[4] - cosines[1]*cosines[3]
                dist=abs(np.sum(normal*data[i].ImagePositionPatient)-np.sum(normal*data[i+1].ImagePositionPatient))
                zspacing.append(dist)
            else:
                zspacing.append(dist)
                
    area_elements =[np.prod(p) for p in pixel_spacing]
    
    return X, area_elements, zspacing



def load_network(architecture, fold, device):
    if architecture == "unet":
        path  = "weights/3d-net/"
    elif architecture == "unet2d":
        path = "weights/2d-net/"
    elif architecture == "roi":
        path = "weights/roi-net/"
    else:
        print("ERROR")
    params= yaml.load(open(os.path.join(path, "config.json"), 'r'), Loader=yaml.SafeLoader)['network']
    if architecture =="roi":
        weights = torch.load(os.path.join(path,"best_weights.pth"), weights_only=True,   map_location=torch.device(device))
    else: 
        weights = torch.load(os.path.join(path,f"best_weights_{fold}.pth"), weights_only=True,   map_location=torch.device(device))
    net = get_network(architecture=architecture, **params).to(device)
    net.load_state_dict(weights)
    net.eval()
    return net

def calculate_center(X, net_roi, device):
    X =torch.from_numpy(X)
    X=X[None, None,...]
    X=X.type(torch.FloatTensor)
    X=X.to(device)
    z_dim=X.shape[2]
    pred= net_roi(X)[0]
    mask = pred >= 0.5
    middle=mask[0,0,z_dim//2,:,:].cpu().numpy()
    labels = label(middle)
    try:
        middle = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        center=ndimage.measurements.center_of_mass(middle*1)
    #if isNaN(center[0]):
    except:
        print("Problem!:")
        #center=np.array([116, 105])
        center=np.array([0, 0])
    return(int(center[0]), int(center[1]))

def normalize(img):
    if len(img.shape)==3:
        for i in range(img.shape[0]):
            img[i,...]=(img[i,...] - np.mean(img[i,...])) / np.std(img[i,...])
    else:
        img = (img - np.mean(img))/np.std(img)
    return img


def predict2d(im,net2d, num_classes):
    in2d=torch.moveaxis(im,2,0)[:,0,...]
    temp=net2d(in2d)[0]
    temp=torch.moveaxis(temp,0,1)
    temp=torch.argmax(temp,0).long()
    temp=torch.nn.functional.one_hot(temp,num_classes)
    temp=torch.moveaxis(temp,-1,0)
    return temp

def predict3d(im, out2d, net3d, num_classes):
    out2d = out2d[3:,...]
    out2d = out2d[None,...]
    in3d=torch.cat((im, out2d),1)
    out3d=net3d(in3d)[0]
    out3d=torch.argmax(out3d,1).long()
    out3d=torch.nn.functional.one_hot(out3d,num_classes)
    out3d=torch.moveaxis(out3d,-1,1).float()[0,...]
    return out3d

def predict(X, center, device, width = 48):
    shape= (256,256)
    im=X.copy()
    im=normalize(im[:,center[0]-width : center[0]+width, center[1]-width: center[1]+width])
    im = torch.from_numpy(im[None,None, ...].astype("float32")).to(device)
    result = torch.zeros((5, im.shape[2], im.shape[3], im.shape[4])).to(device)
    for i in range(5):
        net2d = load_network(architecture='unet2d', fold =i, device=device)
        net3d = load_network(architecture='unet', fold =i, device=device)
    
        out2d = predict2d(im, net2d, 5)
        result += predict3d(im, out2d, net3d, 5)
        
        out2d = torch.flip(predict2d(torch.flip(im, dims=[3]), net2d, 5), dims=[2])
        result += predict3d(im, out2d, net3d, 5)
        
        out2d = torch.flip(predict2d(torch.flip(im, dims=[4]), net2d,5), dims=[3])
        result += predict3d(im, out2d, net3d, 5)
        
    result= torch.argmax(result,0).cpu().detach().numpy()
    result=np.pad(result, ((0,0),(center[0]-width, shape[0]-(center[0]+width)), (center[1]-width, shape[1]-(center[1]+width))), 
                            constant_values=0)
    
    
    return result


def get_orig_data(root, filenames):
    data=[]
    for file in filenames:
        data.append(pydicom.dcmread(os.path.join(root, file)))
    data = sorted(data, key = lambda k: abs(k.get('SliceLocation')))
    X=np.zeros((len(data),256,256))
    for i in range(len(data)):
        X[i,...]=constant_pad(data[i].pixel_array, 256, c=0) 
    return X


    
def plot(X, segmentation, patient, plotfolder):
    classes= ["bloodpool", "healthy muscle", "scar", "mvo"]
    segmentation -=1
    segmentation=np.ma.masked_where(segmentation ==-1, segmentation)
    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)
    savefolder= os.path.join(plotfolder, patient) 
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    for i in range(len(segmentation)):
        
        fig, axs = plt.subplots(1,2,  constrained_layout=True, dpi=500)
            
        axs[0].imshow(X[i,...], cmap='gray')
        axs[0].axis('off')
        axs[0].set_title("Input")
        axs[1].imshow(X[i,...], cmap='gray')
        mat=axs[1].imshow(segmentation[i],'jet', interpolation='none', alpha=0.5, vmin = 0, vmax = 3)
        axs[1].set_title("NN Segmentation")
        axs[1].axis('off')
        values = np.array([0,1,2,3])
        colors = [ mat.cmap(mat.norm(value)) for value in values]
        patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i]) ) for i in range(len(values)) ]
        plt.legend(handles=patches, loc='lower right',  bbox_to_anchor=(0.85, -0.4, 0.2, 0.2) )
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        plt.savefig(os.path.join(savefolder,f"slice_{i+1}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        
def save_dicom(segmentation, root, filenames, patient, savefolder):
    """Save segmentation as dicom
    """
    if not os.path.exists(os.path.join(savefolder, "padded_imgs", patient)):
        os.makedirs(os.path.join(savefolder, "padded_imgs", patient))
    if not os.path.exists(os.path.join(savefolder, "segmentations", patient)):
        os.makedirs(os.path.join(savefolder, "segmentations", patient))
    data=[]
    for file in filenames:
        data.append(pydicom.dcmread(os.path.join(root, file)))
    data = sorted(data, key = lambda k: abs(k.get('SliceLocation')))
    for i in range(len(data)):
        padded = constant_pad(data[i].pixel_array, 256, c=0) 
        padded = padded.astype(data[i].pixel_array.dtype)
        data[i].Rows, data[i].Columns = (256,256)
        data[i].PixelData = padded.tobytes()
        data[i].save_as(os.path.join(savefolder, "padded_imgs", patient, f"slice_{i}"))
        
        seg = segmentation[i].astype(data[i].pixel_array.dtype)
        data[i].PixelData = seg.tobytes()
        data[i].save_as(os.path.join(savefolder, "segmentations", patient, f"slice_{i}"))
        
    
def get_logger(name, level=logging.INFO, formatter = '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s'):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.handler_set = True
    return logger