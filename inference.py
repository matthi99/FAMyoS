# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:03:27 2023

@author: matthias
"""

import torch
import os
import numpy as np
import argparse



from utils import *


parser = argparse.ArgumentParser(description='Define parameters for inference.')
parser.add_argument('--patient_folder', help= "Path folder where DICOM files of patients are stored", type= str, 
                    default="dicoms/")
parser.add_argument('--save', help= "Should segmentation masks be saved", type=bool, default=True)
parser.add_argument('--save_folder',  help= "Path folder where results should be stored", type= str, default="segmentations")
parser.add_argument('--plots', help= "Plot segmentations", type=bool, default=False)
args = parser.parse_args()



if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

if args.plots ==True:
    plotfolder=os.path.join(args.save_folder, "plots")
    
#define logger and device
logger= get_logger("Inference")
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#get patients
patients= [f for f in os.listdir(args.patient_folder) if os.path.isdir(os.path.join(args.patient_folder,f))]
patients.sort(key=lambda x: int(x.split("_")[1]))


net_roi= load_network(architecture='roi', fold = 0, device=device)

for patient in patients: 
    if not os.path.exists(os.path.join(args.save_folder, "segmentations", patient)):
        print(patient)
        files = [x for x in os.walk(os.path.join(args.patient_folder,patient))][-1]
        root = files[0]
        filenames = files[-1]    
        X, area_elements, zspacing = get_data(root, filenames) 
        center =calculate_center(X, net_roi, device)
        segmentation=predict(X, center, device)
        if args.save == True:
            save_dicom(segmentation, root, filenames, patient, args.save_folder)
        if args.plots == True:
            plot(X, segmentation, patient, plotfolder)


logger.info(f"All predctions saved in {args.save_folder}")





