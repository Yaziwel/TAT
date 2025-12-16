import os 
os.environ['CUDA_VISIBLE_DEVICES']='0' 
from model.TAT import TAT
from loss.losses import CharbonnierLoss
from evaluation.evaluation_metric import compute_measure
from data.common import transformData, dataIO
from data.MedicalDataUniform import Train_Data, Test_Data, DataSampler 
from loss.AutomaticWeightedLoss import AutomaticWeightedLoss
import numpy as np

import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
import time 
from tqdm import tqdm
import random
from tools import set_seeds, mkdir  
import pdb 
import pandas as pd 
from ema_pytorch import EMA

transformData = transformData()
io=dataIO() 
set_seeds(42)



def build_train_sampler(modality_list, data_root, batch_size, shuffle=True, sampling='uniform'):
    dataloader_list = [] 
    if sampling=='uniform':
        for modality in modality_list:
            dataset = Train_Data(root_dir = data_root, modality_list = [modality]) 
            print("{}: {}".format(modality, dataset.length))
            dataloader_list.append(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)) 
    else:
        dataset = Train_Data(root_dir = data_root, modality_list = modality_list) 
        print("All Modality: {}".format(modality, dataset.length))
        dataloader_list.append(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)) 
    sampler = DataSampler(dataloader_list) 
    
    return sampler

def save_model(G_net_model, save_dir, optimizer_G=None, ex=""):
    save_path=os.path.join(save_dir, "Model")
    mkdir(save_path)
    G_save_path = os.path.join(save_path,'Generator{}.pth'.format(ex))
    torch.save(G_net_model.cpu().state_dict(), G_save_path)
    G_net_model.cuda()

    if optimizer_G is not None:
        opt_G_save_path = os.path.join(save_path,'Optimizer_G{}.pth'.format(ex))
        torch.save(optimizer_G.state_dict(), opt_G_save_path)

total_iteration = 400000
val_iteration = 1e3

batch_size = 4 ## 4 per task for 'uniform' sampling
eps=1e-8

psnr_max=0

save_dir = "experiment/TAT" 
data_root = "/home/data/zhiwen/dataset/All-in-One/" 
modality_list = ["PET", "CT", "MRI"] 



Generator = TAT() 
# load_parameters(os.path.join(save_dir, "Model","Generator_iteration_8000.pth"), Generator)
Generator.cuda() 

AWL = AutomaticWeightedLoss(task_list=modality_list)
AWL.cuda()

model_ema = EMA(
    Generator,
    beta = 0.999,              # exponential moving average factor
    update_after_step = 0,    # only after this number of .update() calls will it start updating
    update_every = 1,          # how often to actually update, to save on compute (updates every 10th .update() call) 
    # update_model_with_ema_every = 1e4,
    move_ema_to_online_device = True
)



train_sampler = build_train_sampler(modality_list, data_root, batch_size, shuffle=True) 
valid_loader = DataLoader(Test_Data(root_dir=data_root, use_num=16), batch_size=1, shuffle=False) 


# optimizer_G = torch.optim.Adam(Generator.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08) 
optimizer_G = torch.optim.AdamW(
    [
        {'params': Generator.parameters()},
        {'params': AWL.parameters()}
    ],
    lr=2e-4,
    betas=(0.9, 0.99)
)
lr_scheduler_G = CosineAnnealingLR(optimizer_G, total_iteration, eta_min=1.0e-7)

# CE = nn.CrossEntropyLoss().cuda() 
criterion = nn.L1Loss(reduction='none').cuda()

running_loss = []
eval_metrics={
    "psnr":[],
    "ssim":[], 
    "rmse":[]
    }


pbar = tqdm(total=int(total_iteration))

print("################ Train ################")
for iteration in list(range(1, int(total_iteration)+1)):


    l_G=[]
    in_pic,  label_pic, m_list = next(train_sampler)
    
    in_pic = in_pic.type(torch.FloatTensor).cuda()
    label_pic = label_pic.type(torch.FloatTensor).cuda() 



    #################
    #     train G
    #################
    Generator.train()
    optimizer_G.zero_grad() 
    


    restored_pic = Generator(in_pic) 
    loss_G = AWL(in_pic, label_pic, restored_pic, m_list) 
    
    
    loss_G.backward()
    optimizer_G.step() 
    model_ema.update()

    l_G.append(loss_G.item())
    torch.cuda.empty_cache() 
    lr_scheduler_G.step()
            
    

    
    if iteration % val_iteration == 0: 
        psnr=0
        ssim=0
        rmse=0
        model_ema.ema_model.eval() 
        for counter,data in enumerate(tqdm(valid_loader)):
            v_in_pic, v_label_pic, modality, file_name = data 
            modality = modality[0] 
            file_name = file_name[0] 

            # pdb.set_trace()


            v_in_pic = v_in_pic.type(torch.FloatTensor).cuda() 
            v_label_pic = v_label_pic.type(torch.FloatTensor)
            
            # pdb.set_trace()
            with torch.no_grad():
                gen_img = model_ema(v_in_pic) 
            
            gen_img = transformData.denormalize(gen_img, modality).detach().cpu() 
            
            v_label_pic = transformData.denormalize(v_label_pic, modality) 
            
            
            
            '''
            truncation for test image 
            CT:[-160, 240]
            '''
            
            gen_img = transformData.truncate_test(gen_img, modality) 
            v_label_pic = transformData.truncate_test(v_label_pic, modality) 
            
            data_range = v_label_pic.max()-v_label_pic.min()
            oneEval = compute_measure(gen_img, v_label_pic, data_range = data_range) 
            
            psnr+=oneEval[0]
            ssim+=oneEval[1] 
            rmse+=oneEval[2] 
    
            io.save(gen_img.clone().numpy().squeeze(), os.path.join(save_dir, "Gimg", "{}_{}.nii".format(file_name, modality) ))
            # import pdb 
            # pdb.set_trace()
            
            torch.cuda.empty_cache()
        c_psnr=psnr/(counter+1)
        c_ssim=ssim/(counter+1) 
        c_rmse=rmse/(counter+1) 
        
        eval_metrics['psnr'].append(c_psnr)
        eval_metrics['ssim'].append(c_ssim)  
        eval_metrics['rmse'].append(c_rmse) 
    
        save_model(G_net_model=Generator, save_dir=save_dir, optimizer_G=None, ex="_latest")
        if c_psnr>=psnr_max:
            psnr_max=c_psnr
            io.save("Best Iteration: {}, PSNR: {}, SSIM:{}, RMSE:{}".format(iteration, c_psnr, c_ssim, c_rmse),os.path.join(save_dir, "best.txt"))
            save_model(G_net_model=model_ema.ema_model, save_dir=save_dir, optimizer_G = optimizer_G, ex="_best")
        io.save(
            {'eval_metrics':eval_metrics},
            os.path.join(save_dir, "evaluationLoss.bin")
            ) 


    pbar.set_description("loss_G:{:6}, psnr:{:6}".format(loss_G.item(), eval_metrics['psnr'][-1] if len(eval_metrics['psnr'])>0 else 0)) 
    pbar.update() 



####################
#
# Code for testing
#
####################

Generator.load_state_dict(torch.load(os.path.join(save_dir, "Model","Generator_best.pth"))) 
Generator.eval()
for modality_name in modality_list:
    test_loader = DataLoader(Test_Data(root_dir=data_root, modality_list = [modality_name], target_folder="test"), batch_size=1, shuffle=False) 
    psnr_list=[]
    ssim_list=[]
    rmse_list=[]
    name_list=[]
    for counter, data in enumerate(tqdm(test_loader)):
        v_in_pic, v_label_pic, modality, file_name = data 
        modality = modality[0] 
        file_name = file_name[0] 
    
        v_in_pic = v_in_pic.type(torch.FloatTensor).cuda() 
        v_label_pic = v_label_pic.type(torch.FloatTensor) 
        
        # pdb.set_trace()
        
        with torch.no_grad():
            gen_img = Generator(v_in_pic) 
        
        gen_img = transformData.denormalize(gen_img, modality).detach().cpu() 
        
        v_label_pic = transformData.denormalize(v_label_pic, modality) 
        
        
        '''
        truncation for test image 
        CT:[-160, 240]
        '''
        
        gen_img = transformData.truncate_test(gen_img, modality) 
        v_label_pic = transformData.truncate_test(v_label_pic, modality) 
        
        data_range = v_label_pic.max()-v_label_pic.min()
        oneEval = compute_measure(gen_img, v_label_pic, data_range = data_range) 
        
        psnr_list.append(oneEval[0])
        ssim_list.append(oneEval[1])
        rmse_list.append(oneEval[2])
        name_list.append(file_name)

        io.save(gen_img.clone().numpy().squeeze(), os.path.join(save_dir, "test_result", modality, "{}.nii".format(file_name)))  
    
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list) 
    rmse_list = np.array(rmse_list)
    name_list = np.array(name_list)
    c_psnr = psnr_list.mean()
    c_ssim = ssim_list.mean()
    c_rmse = rmse_list.mean()
    print(" ^^^Final Test  {}   psnr:{:.6}, ssim:{:.6}, rmse:{:.6} ".format(modality_name, c_psnr,c_ssim, c_rmse))
    result_dict={
        "NAME":name_list,
        "PSNR":psnr_list,
        "SSIM":ssim_list, 
        "RMSE":rmse_list,
        }
    result=pd.DataFrame({ key:pd.Series(value) for key, value in result_dict.items() })
    result.to_csv(os.path.join(save_dir, "test_result", "{}_result.csv".format(modality_name)))

