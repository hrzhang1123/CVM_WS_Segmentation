
import os
import glob
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import cv2
from scipy.ndimage import gaussian_filter, morphology
import skfmm
import scipy.io as sio
from skimage.filters import threshold_otsu, threshold_multiotsu
from numpy import linalg as LA
from model.ResNet_UNet import ResNet_UNet
from torchvision import transforms
import torch
import pickle
import torch.backends.cudnn as cudnn
import argparse
import random
import time
import torch.nn.functional as F
from openpyxl import Workbook
from multiprocessing import Pool


from Variational_utils import Residual, AOS_v2, get_relu_residual


cudnn.benchmark = True


parser = argparse.ArgumentParser(description='None')
parser.add_argument('--name', default='None', type=str)     ##############

parser.add_argument('--network', default='resnet34', type=str)     ##############
parser.add_argument('--pretrain', action='store_false', default=True)
parser.add_argument('--frozen', action='store_true', default=False)
parser.add_argument('--isRotate', action='store_false', default=True)
parser.add_argument('--isColorJitter', action='store_false', default=True)
parser.add_argument('--fully_supervised', action='store_true', default=False) ##########
parser.add_argument('--isNormalize', action='store_true', default=False)

parser.add_argument('--coord_ref_size', default='[1024,1024]', type=str)
parser.add_argument('--image_resize', default='[4096, 4096]', type=str)
parser.add_argument('--heatmap_resize', default='[512, 512]', type=str)
parser.add_argument('--mask_resize', default='[256, 256]', type=str)
parser.add_argument('--numCoord', default=5, type=int)
parser.add_argument('--nClass', default=2, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--lr', default='0.0001', type=float)
parser.add_argument('--epoch', default=31, type=int)
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--subset_index', default=0, type=int)
parser.add_argument('--Iters', default=5, type=int)
parser.add_argument('--num_thread', default=12, type=int)
parser.add_argument('--alpha', default=1, type=float) ## weight for variational mask  if <0, no
parser.add_argument('--eta', default=0.3, type=float) ## weight for variational mask
parser.add_argument('--vm_freq', default=5, type=int) ## weight for variational mask






parser.add_argument('--subset_list_path', default='', type=str)
parser.add_argument('--log_dir', default='', type=str)  # resnet34_pretrainFeature
parser.add_argument('--image_dir', default='', type=str)   ## Slide_for_test
parser.add_argument('--heatmap_dir', default='', type=str)   ## Slide_for_test
parser.add_argument('--feature_dir', default='', type=str)   ## 4096_resnet34_forTest
parser.add_argument('--mask_dir', default='', type=str)
parser.add_argument('--dot_dir', default='', type=str)



performance_metrics = ['dice', 'acc', 'sen', 'spec', 'prec', 'F1', 'kappa', 'auc']

#TYPE = "SELECTIVE"
TYPE = "GLOBAL"

LOAD = sio.loadmat('TestFilesPython.mat')
z = np.array(LOAD['z'])

lambda1 = np.array(LOAD['lambda'])[0][0]   ###############
tau = np.array(LOAD['tau'])[0][0]

#Iters = np.array(LOAD['Iters'])[0][0]
#Iters = 30

utol = 0.001 # np.array(LOAD['utol'])[0][0]
sigma1 = np.array(LOAD['sigma1'])[0][0]
lambda3 = np.array(LOAD['lambda3'])[0][0]    ################

eps2 = np.array(LOAD['eps2'])[0][0]    ##

w = np.array(LOAD['w'])
lambda_TV = np.array(LOAD['lambda_TV'])[0][0]
theta_tai = np.array(LOAD['theta_tai'])[0][0]
cols = np.array(LOAD['cols'])
rows = np.array(LOAD['rows'])
#Mask = np.array(LOAD['Mask'])

c1 = 0
c2 = 1
xi = 0.1

if TYPE == "SELECTIVE":
    theta = np.array(LOAD['theta'])[0][0]
    CV_type = 1

elif TYPE == "GLOBAL":
    CV_type = 0
    theta = 0









def main():
    params = parser.parse_args()

    coord_ref_size = json.loads(params.coord_ref_size)
    #image_resize = json.loads(params.image_resize)
    mask_resize = json.loads(params.mask_resize)
    heatmap_resize = json.loads(params.heatmap_resize)

    
    model = ResNet_UNet(network=params.network, pretrain=params.pretrain, nClass=params.nClass, frozen=params.frozen).to(params.device)
    writer = None

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    log_dir = os.path.join(params.log_dir, 'log.txt')
    log_file = open(log_dir, 'a')

    z = vars(params).copy()
    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))

    ce_loss = nn.CrossEntropyLoss(reduction='none').to(params.device)


    train_paths, test_paths, val_paths = get_img_paths_subset_withVal(params.image_dir, params.subset_list_path, params.subset_index, params.dot_dir)


    train_dataset = image_dataset_camelyon(train_paths, params.mask_dir, params.heatmap_dir, params.dot_dir, coord_ref_size, mask_resize, heatmap_resize,
                                           params.isRotate, numCoord=params.numCoord, isColorJitter=params.isColorJitter)

    test_dataset = image_dataset_camelyon(test_paths, params.mask_dir, params.heatmap_dir, params.dot_dir,
                                           coord_ref_size, mask_resize, heatmap_resize, isRotate=False, numCoord=params.numCoord,
                                           isColorJitter=False)

    val_dataset = image_dataset_camelyon(val_paths, params.mask_dir, params.heatmap_dir, params.dot_dir,
                                           coord_ref_size, mask_resize, heatmap_resize, isRotate=False, numCoord=params.numCoord,
                                           isColorJitter=False)

    vm_dataset = image_dataset_camelyon(train_paths, params.mask_dir, params.heatmap_dir, params.dot_dir,
                                           coord_ref_size, mask_resize, heatmap_resize, isRotate=False, numCoord=params.numCoord,
                                           isColorJitter=False)



    DataLoader_train = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
    DataLoader_test = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=params.num_workers)
    DataLoader_val = DataLoader(val_dataset, batch_size=params.batch_size, num_workers=params.num_workers)
    DataLoader_vm = DataLoader(vm_dataset, batch_size=params.batch_size, num_workers=params.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    vMap_dict = None    ##
    best_auc = 0
    best_val_loss = 1000



    best_epoch = 0
    for epoch in range(params.epoch):

        tstart = time.time()
        if params.fully_supervised:
            train_FullySupervised(model, DataLoader_train, ce_loss, optimizer, params, log_file, epoch, writer)
        else:
            train(model, DataLoader_train, ce_loss, optimizer, params, log_file, epoch, writer, vMap_dict)
        

        val_partial_loss = val(model, DataLoader_val, ce_loss, params, log_file, epoch, writer, best_auc)



        if val_partial_loss < best_val_loss:

            if params.alpha >= 0:
                #if epoch % params.vm_freq == 0:
                vMap_dict = get_Variational_Mask(model, DataLoader_vm, ce_loss, params, log_file, epoch, writer)

            best_val_loss = val_partial_loss

            test(model, DataLoader_test, ce_loss, params, log_file, epoch, writer, 0)

            print(f'{epoch}: {time.time()-tstart}')
            print_log(f'    best test auc {best_auc} from epoch {epoch} ', log_file)

            torch.save(model.state_dict(), os.path.join(params.log_dir, 'save_model.pth'))



def train(model, dataloader, loss_cri, optim, params, f_log, epoch, writer, vm_masks=None):
    model.train()

    partial_LOSS = AverageMeter()
    full_LOSS = AverageMeter()
    full_AUC = AverageMeter()
    full_VM_LOSS = AverageMeter()


    tstep = len(dataloader)
    dataiter = iter(dataloader)

    mData = []

    if epoch % 5 == 0:
        heatmap_save_dir = os.path.join(params.log_dir, 'train_heatmap',  str(epoch))
        if not os.path.exists(heatmap_save_dir):
            os.makedirs(heatmap_save_dir)

    for idx in range(tstep):

        data = next(dataiter)

        images = data['image'].to(params.device)
        full_masks = data['full_mask'].to(params.device)
        partial_masks = data['partial_mask'].to(params.device)
        valid_masks = data['valid_mask'].to(params.device)
        image_names = data['image_name']
        image_paths = data['image_path']
        heatmap_np = data['heatmap_np']

        rot_rnd = data['rnd'][0].item()


        in_coords_ts = data['in_coords']
        out_coords_ts = data['in_coords']

        bs = images.shape[0]

        output, feat0, feat1 = model(images)


        output_vec = output.view(bs, params.nClass, -1)
        output_vec = torch.transpose(output_vec, 1, 2).reshape(-1, params.nClass)

        full_vec = full_masks.view(-1)
        partial_vec = partial_masks.view(-1)
        valid_vec = valid_masks.view(-1)

        partial_ce_loss = loss_cri(output_vec, partial_vec)
        partial_ce_loss *= valid_vec

        partial_ce_loss = torch.sum(partial_ce_loss) / torch.sum(valid_vec)
        
        if vm_masks == None:
            aloss = partial_ce_loss
        else:
            tvm = vm_masks[image_names[0]]
            tvm_np = np.zeros(tvm.shape)

            tvm_np[tvm>params.threshold] = 1

            

            tvm_np = np.rot90(tvm_np, rot_rnd, axes=(0, 1))
            tvm_np = np.ascontiguousarray(tvm_np)
            tvm_ts = torch.from_numpy(tvm_np).long().to(params.device)
            tvm_vec_ts = tvm_ts.view(-1)

            vm_loss = torch.mean(loss_cri(output_vec, tvm_vec_ts))

            full_VM_LOSS.update(vm_loss.item(), 1)

            aloss = partial_ce_loss + params.alpha * vm_loss

        optim.zero_grad()
        aloss.backward()
        optim.step()

        full_ce_loss = loss_cri(output_vec, full_vec)
        full_ce_loss = torch.mean(full_ce_loss)

        try:
            full_auc = roc_auc_score(full_vec.cpu().detach().numpy(), torch.softmax(output_vec, dim=1)[:,1].cpu().detach().numpy())
            full_AUC.update(full_auc, 1)
        except:
            #raise  RuntimeError(f'{image_names} has error')
            print((f'------------{image_names} has error'))


        output_softmax = torch.softmax(output, dim=1)
        output_mask_np = output_softmax.detach().cpu()[0].numpy()[1,:,:]

        partial_LOSS.update(partial_ce_loss.item(), bs)
        full_LOSS.update(full_ce_loss.item(), bs)

        mData.append( {'name': image_names[0], 'gt_np': full_masks[0].detach().cpu().numpy(), 'pred_np':output_mask_np, 'heatmap_np': heatmap_np[0]})

    tstr = f'---train epoch: {epoch}, partial ce loss {partial_LOSS.avg}, full ce loss {full_LOSS.avg}, full auc {full_AUC.avg}, VM loss {full_VM_LOSS.avg}'
    print_log(tstr, f_log)


    if epoch % params.vm_freq == 0:
        ### process the mData
        train_save_dir = os.path.join(params.log_dir, 'train_eval', str(epoch))
        if not os.path.exists(train_save_dir):
            os.makedirs(train_save_dir)

        train_save_map_dir = os.path.join(train_save_dir, 'train_map')
        if not os.path.exists(train_save_map_dir):
            os.makedirs(train_save_map_dir)

        mMetric_all = {}
        for sst in performance_metrics:
            mMetric_all[sst] = []
        mMetric_all['name'] = []


        ### process the mData
        for tdict in mData:
            tname = tdict['name']
            tpred_np = tdict['pred_np']
            theatmap_np = tdict['heatmap_np']
            tgt_np = tdict['gt_np']

            save_image_set(theatmap_np, tpred_np, train_save_map_dir, tname, params.threshold)

            eval_value_dict = eval_metric(tpred_np, tgt_np, params.threshold)

            for sst in performance_metrics:
                mMetric_all[sst].append(eval_value_dict[sst])
            mMetric_all['name'].append(tname)

        book = Workbook()
        # item_names = ['Name', 'Dice', 'Acc', 'Sen']
        item_names = ['Name'] + performance_metrics
        tsheet = book.create_sheet("sheet_", 0)
        tsheet.append(item_names)

        for ii in range(len(mMetric_all['name'])):
            temp = [ str(mMetric_all[sst][ii]) for sst in performance_metrics ]
            trow = [mMetric_all['name'][ii]] + temp
            tsheet.append(trow)

        allMean = [ str(np.array(mMetric_all[sst]).mean()) for sst in performance_metrics]
        trow = ['mean'] + allMean
        tsheet.append(trow)
        book.save(os.path.join(train_save_dir, 'all_eval_metrics.xlsx'))


#####################------------------------------------------->>>>>
## features_np: FS x H x W
def constrastive_Variational_proc(args):

    features_ts, in_coords_np, out_coords_np, isNormalize, eta, params, save_path = args

    def get_gaussian_feature(tcoord, sigma=1):
        return features_ts[:, tcoord[1], tcoord[0]]

    def get_sim_map(inFeat):
        outfeat = torch.einsum('cab,c->ab', features_ts, inFeat)
        if isNormalize:
            outfeat = (outfeat - torch.min(outfeat)) / (torch.max(outfeat) - torch.min(outfeat))
        return outfeat

    all_in_sim_maps = []
    all_out_sim_maps = []

    for ii in range(in_coords_np.shape[0]):
        cc = in_coords_np[ii]
        tfeat = get_gaussian_feature(cc)
        tsim_map = get_sim_map(tfeat)
        all_in_sim_maps.append(tsim_map)

    for ii in range(out_coords_np.shape[0]):
        cc = out_coords_np[ii]
        tfeat = get_gaussian_feature(cc)
        tsim_map = get_sim_map(tfeat)
        all_out_sim_maps.append(tsim_map)

    all_map = []
    for in_map in all_in_sim_maps:
        in_mean = []
        for out_map in all_out_sim_maps:
            t_residual = get_relu_residual(in_map, out_map, eta=eta)
            in_mean.append(t_residual)

        in_mean = [ sst.unsqueeze(0) for sst in in_mean]
        in_mean =  torch.cat(in_mean, dim=0)  #np.concatenate(in_mean, axis=0)
        in_mean =  torch.mean(in_mean, dim=0) #np.mean(in_mean, axis=0)
        in_mean = (in_mean - torch.min(in_mean))/(torch.max(in_mean) - torch.min(in_mean))
        all_map.append(in_mean)

    all_u = []
    for tmap in all_map:
        tmap_np = tmap.numpy()
        tu = ConvexSeg_Run_CVOnly(tmap_np, c1, c2, lambda1, tau, params.Iters, utol, theta, sigma1, CV_type, lambda3, eps2, w,
                                  lambda_TV, theta_tai, cols, rows, xi, tmap_np)
        tu = (tu - np.min(tu)) / (np.max(tu) - np.min(tu))
        all_u.append(tu)

    mean_tu = [ np.expand_dims(sst, 0) for sst in all_u]
    mean_tu = np.concatenate(mean_tu, axis=0)
    mean_tu = np.mean(mean_tu, axis=0)
    mean_tu = (mean_tu - np.min(mean_tu)) / (np.max(mean_tu)-np.min(mean_tu))
    mean_tu_img = Image.fromarray((mean_tu*255).astype(np.uint8))
    mean_tu_img.save(save_path)

#########################-------------------------->
def ConvexSeg_Run_CVOnly(z,c1,c2,lambda1,tau,Iters,utol,theta,sigma1,CV_type,lambda3,eps2,w,lambda_TV,theta_tai,cols,rows,xi,Mask, ):

    m , n = np.shape(z)

    ## FLAG KEY
    # 0 - Spencer-Chen
    # 1 - Geodesic

    for flag in [0]:

        if theta == 0 or lambda1 == 0:
            SSF = np.zeros(np.shape(z))
            u = z.copy()

        else:

            if flag == 0:

                # TODO: Code interface for input a binary mask
                print('FLAG = 0')
                SSF = np.zeros(np.shape(z))

            elif flag ==1:

                Pd1 = 0 #calculate_geodesic_dist(z, rows, cols)
                SSF = theta * Pd1/np.max(Pd1.flatten())

                # TODO: Code geodesic distance calculator

            u = Mask.copy().astype(np.float32)

            if np.shape(cols)[0]*np.shape(cols)[1] == 1 and np.shape(rows)[0]*np.shape(rows)[1] == 1:
                Mask[rows:rows+2,cols:cols+2] = 1

        #### Additional parameters
        mu = 1; # regularisation term
        varsigma = 1e-2; # parameter in penalty function
        as1 = 2; # multiply minimum alpha value by as1
        beta = eps2; # parameter in curvature
        b = 161.7127690; # from Taylor expansion of fpen fn.

        sig = np.maximum(1.0,np.sqrt(100.0*sigma1))
        z_sm = gaussian_filter(z, sigma=sig)

        gy, gx = np.gradient(z_sm)

        nab_z = np.sqrt((gx**2)+(gy**2))
        beta1 = 10

        g = np.divide(1.0, (np.ones(np.shape(nab_z))+(beta1*(nab_z**2))))

        #### For recording progression of residual
        res = []
        #### Calculate fitting term and set alpha

        c1 = np.sum(z[Mask>0.5])/np.sum(Mask>0.5)
        c2 = np.sum(z[Mask<0.5])/np.sum(Mask<0.5)

        if CV_type == 0: # Original Chan-Vese

            f1 = (z-c1)**2-lambda3 *(z-c2)**2 + SSF

        elif CV_type==1: # Roberts-Spencer

            N = 3

            K11 = np.expand_dims(np.array(threshold_multiotsu(z,classes = N)),0)
            K = np.concatenate((np.zeros((1,1)),K11),1)
            K = np.concatenate((K,np.ones((1,1))),1)[0]

            # LOWER THRESHOLD
            L_vect = np.maximum(c1 - K,0.0)
            L_vect2 = L_vect.copy()
            TF_vect = np.expand_dims(np.arange(np.shape(L_vect)[0]),0)[0]
            TF_vect = TF_vect[L_vect>0.0]
            L_vect2 = L_vect2[L_vect>0.0]
            TF2 = np.where(L_vect2==np.min(L_vect2))[0][0]
            L = K[TF_vect[TF2]]

            # UPPER THRESHOLD
            H_vect = np.maximum(K - c1,0.0)
            H_vect2 = H_vect.copy()

            TF_vect = np.expand_dims(np.arange(np.shape(L_vect)[0]),0)[0]
            TF_vect = TF_vect[H_vect>0.0]
            H_vect2 = H_vect2[H_vect>0.0]

            if not H_vect2.all():
                H = K[-1];
            else:
                TF2 = np.where(H_vect2==np.min(H_vect2))[0]
                H = K[TF_vect[TF2[0]]];

            gamma1 = c1 - L;
            gamma2 = H - c1;

            TF3 = np.logical_and(z>=c1-gamma1, z<=c1)
            TF4 = np.logical_and(z<=c1+gamma2, z>c1)

            f3 = ( 1+((z-c1)/gamma1) )*TF3 + ( 1-((z-c1)/gamma2) )*TF4

            f1 = ((z-c1)**2) - lambda3 * f3 + SSF 


        for l in range(Iters):

            if (l+1) % 50 == 0:
                tau = np.maximum(1.0e-3,tau*0.9)

            if l % 2 ==0:

                if CV_type==1: # Roberts-Spencer

                    # LOWER THRESHOLD
                    L_vect = np.maximum(c1 - K,0.0)
                    L_vect2 = L_vect.copy()
                    TF_vect = np.expand_dims(np.arange(np.shape(L_vect)[0]),0)[0]
                    TF_vect = TF_vect[L_vect>0.0]
                    L_vect2 = L_vect2[L_vect>0.0]
                    TF2 = np.where(L_vect2==np.min(L_vect2))[0][0]
                    L = K[TF_vect[TF2]]

                    # UPPER THRESHOLD
                    H_vect = np.maximum(K - c1,0.0)
                    H_vect2 = H_vect.copy()

                    TF_vect = np.expand_dims(np.arange(np.shape(L_vect)[0]),0)[0]
                    TF_vect = TF_vect[H_vect>0.0]
                    H_vect2 = H_vect2[H_vect>0.0]

                    if not H_vect2.all():
                        H = K[-1]
                    else:
                        TF2 = np.where(H_vect2==np.min(H_vect2))[0]
                        H = K[TF_vect[TF2[0]]]

                    gamma1 = c1 - L
                    gamma2 = H - c1

                    TF3 = np.logical_and(z>=c1-gamma1, z<=c1)
                    TF4 = np.logical_and(z<=c1+gamma2, z>c1)

                    f3 = ( 1+((z-c1)/gamma1) )*TF3 + ( 1-((z-c1)/gamma2) )*TF4

                    f1 = ((z-c1)**2) - lambda3 * f3 + SSF

            oldu = u.copy()

            f0 = lambda1*f1/np.max(f1.flatten());
            f11 = f0.flatten()
            A = (1/2) * LA.norm(f11, np.inf) # minimum for alpha from Chan paper
            alpha = as1*A

            #### Calculate penalty term
            N1 = np.sqrt((2*u-1)**2+varsigma)-1
            Hnu = (1/2)+(1/np.pi)*np.arctan(N1/varsigma)
            dN_num1 = np.divide((4*u-2.0),(N1+1.0))
            dN = dN_num1 * (np.divide(varsigma*N1,np.pi*(varsigma**2+N1**2))+ Hnu)
            dNu = alpha*dN

            #### Perform AOS iteration
            #### v2 is from CMS 15 paper
            u = AOS_v2(u,n,g,tau,mu,f0,beta,dNu,b,alpha,varsigma,theta_tai)
            #### Check residual and store
            R = Residual(u,oldu)
            res.append(R)

            #if l % 50 ==0:
            #    print('It:'+str(l)+', Res:'+str(np.round(R,3)))


            #### Stopping criterion for u
            if R < utol:
                break

        #print(' == Done ('+str(l+1)+' iters) \n')

        Res_Final = res
        Iters_Final = l
        TC = 1
        u_data = u

        return u #TC,Iters_Final,Res_Final,u,cols,rows,u_data,R_min,R_max,C_min,C_max



############################ ---- >
def get_Variational_Mask(model, dataloader, loss_cri, params, f_log, epoch, writer):
    model.eval()

    map_save_dir = os.path.join(params.log_dir, 'vm_eval' , 'vm_maps_buff', str(epoch))
    if not os.path.exists(map_save_dir):
        os.makedirs(map_save_dir)

    tstep = len(dataloader)
    dataiter = iter(dataloader)

    mData = []

    for idx in range(tstep):

        data = next(dataiter)

        images = data['image'].to(params.device)
        full_masks = data['full_mask'].to(params.device)
        partial_masks = data['partial_mask'].to(params.device)
        valid_masks = data['valid_mask'].to(params.device)
        image_names = data['image_name']
        img_paths = data['image_path']
        heatmap_np = data['heatmap_np']

        in_coords_ts = data['in_coords']
        out_coords_ts = data['out_coords']
        

        bs = images.shape[0]

        with torch.no_grad():
            output, feat0, feat1 = model(images)

        contFeat = feat1[0]

        contFeat = F.normalize(contFeat, p=2, dim=0)

        contFeat_ts = contFeat.detach().cpu()

        in_coords_np = in_coords_ts[0].numpy()
        out_coords_np = out_coords_ts[0].numpy()

        mData.append({'feature_ts': contFeat_ts, 'in_coords_np': in_coords_np, 'out_coords_np': out_coords_np,
                      'img_name': image_names[0], 'full_mask_np': full_masks[0].cpu().numpy(), 'heatmap_np': heatmap_np[0]})


    pool = Pool(processes=params.num_thread)
    arg_list = []
    for tdata in mData:
        targs = [ tdata['feature_ts'], tdata['in_coords_np'], tdata['out_coords_np'], params.eta, params.isNormalize, params, os.path.join(map_save_dir, tdata['img_name']+'.bmp')]

        arg_list.append(targs)

    pool.map(constrastive_Variational_proc, arg_list)
    pool.close()
    pool.join()

    ## Now do the evaluation

    vmap_eval_dir = os.path.join(params.log_dir, 'vm_eval', str(epoch))
    if not os.path.exists(vmap_eval_dir):
        os.makedirs(vmap_eval_dir)
    vmap_savemap_dir = os.path.join(vmap_eval_dir, 'heatmap')
    if not os.path.exists(vmap_savemap_dir):
        os.makedirs(vmap_savemap_dir)

    mMetric_all = {}
    for sst in performance_metrics:
        mMetric_all[sst] = []
    mMetric_all['name'] = []

    return_vmap_dict = {}

    for tdict in mData:
        timgName = tdict['img_name']
        tgt = tdict['full_mask_np']

        theatmap_np = tdict['heatmap_np']

        vpred_img = Image.open(os.path.join(map_save_dir, timgName+'.bmp')).convert('L')
        vpred_np = np.array(vpred_img) / 255

        return_vmap_dict[timgName] = vpred_np

        save_image_set(theatmap_np, vpred_np, vmap_savemap_dir, timgName, params.threshold)

        eval_value_dict = eval_metric(vpred_np, tgt, params.threshold)

        for sst in performance_metrics:
            mMetric_all[sst].append(eval_value_dict[sst])
        mMetric_all['name'].append(timgName)

    book = Workbook()
    # item_names = ['Name', 'Dice', 'Acc', 'Sen']
    item_names = ['Name'] + performance_metrics
    tsheet = book.create_sheet("sheet_", 0)
    tsheet.append(item_names)

    for ii in range(len(mMetric_all['name'])):
        temp = [ str(mMetric_all[sst][ii]) for sst in performance_metrics ]
        trow = [mMetric_all['name'][ii]] + temp
        tsheet.append(trow)

    allMean = [ str(np.array(mMetric_all[sst]).mean()) for sst in performance_metrics]

    tstr = ''
    for tidx, sst in enumerate(performance_metrics):
        tstr += f' {sst}: {allMean[tidx]} '
    tstr = ' vm on train set  ' + tstr

    print_log(tstr, f_log)

    trow = ['mean'] + allMean
    tsheet.append(trow)
    book.save(os.path.join(vmap_eval_dir, 'all_eval_metrics.xlsx'))

    return return_vmap_dict



def train_FullySupervised(model, dataloader, loss_cri, optim, params, f_log, epoch, writer):
    model.train()

    partial_LOSS = AverageMeter()
    full_LOSS = AverageMeter()
    full_AUC = AverageMeter()

    tstep = len(dataloader)
    dataiter = iter(dataloader)

    for idx in range(tstep):

        data = next(dataiter)

        images = data['image'].to(params.device)
        full_masks = data['full_mask'].to(params.device)
        partial_masks = data['partial_mask'].to(params.device)
        valid_masks = data['valid_mask'].to(params.device)
        image_names = data['image_name']
        img_paths = data['image_path']

        bs = images.shape[0]

        output, feat0, feat1 = model(images)

        output_vec = output.view(bs, params.nClass, -1)
        output_vec = torch.transpose(output_vec, 1, 2).reshape(-1, params.nClass)

        full_vec = full_masks.view(-1)
        partial_vec = partial_masks.view(-1)
        valid_vec = valid_masks.view(-1)

        partial_ce_loss = loss_cri(output_vec, partial_vec)
        partial_ce_loss *= valid_vec

        partial_ce_loss = torch.sum(partial_ce_loss) / torch.sum(valid_vec)

        full_ce_loss = loss_cri(output_vec, full_vec)
        full_ce_loss = torch.mean(full_ce_loss)

        optim.zero_grad()
        full_ce_loss.backward()
        optim.step()


        try:
            full_auc = roc_auc_score(full_vec.cpu().detach().numpy(), output_vec[:,1].cpu().detach().numpy())
            full_AUC.update(full_auc, 1)
        except:
            #raise  RuntimeError(f'{image_names} has error')
            print((f'------------{image_names} has error'))



        partial_LOSS.update(partial_ce_loss.item(), bs)
        full_LOSS.update(full_ce_loss.item(), bs)

    tstr = f'---train epoch: {epoch}, partial ce loss {partial_LOSS.avg}, full ce loss {full_LOSS.avg}, full auc {full_AUC.avg}'
    print_log(tstr, f_log)


def test(model, dataloader, loss_cri, params, f_log, epoch, writer, best_auc):
    model.eval()

    partial_LOSS = AverageMeter()
    full_LOSS = AverageMeter()

    tstep = len(dataloader)
    dataiter = iter(dataloader)

    mData = []

    with torch.no_grad():

        for idx in range(tstep):

            data = next(dataiter)

            images = data['image'].to(params.device)
            full_masks = data['full_mask'].to(params.device)
            partial_masks = data['partial_mask'].to(params.device)
            valid_masks = data['valid_mask'].to(params.device)
            image_names = data['image_name']
            img_paths = data['image_path']

            heatmap_np = data['heatmap_np']

            bs = images.shape[0]

            output, feat0, feat1 = model(images)

            output_vec = output.view(bs, params.nClass, -1)
            output_vec = torch.transpose(output_vec, 1, 2).reshape(-1, params.nClass)

            full_vec = full_masks.view(-1)
            partial_vec = partial_masks.view(-1)
            valid_vec = valid_masks.view(-1)

            partial_ce_loss = loss_cri(output_vec, partial_vec)
            partial_ce_loss *= valid_vec

            partial_ce_loss = torch.sum(partial_ce_loss) / torch.sum(valid_vec)

            full_ce_loss = loss_cri(output_vec, full_vec)
            full_ce_loss = torch.mean(full_ce_loss)

            partial_LOSS.update(partial_ce_loss.item(), bs)
            full_LOSS.update(full_ce_loss.item(), bs)

            output_softmax = torch.softmax(output, dim=1)
            output_mask_np = output_softmax.detach().cpu()[0].numpy()[1,:,:]
            gt_mask_np = full_masks.cpu()[0].numpy()
            eval_value_dict = eval_metric(output_mask_np, gt_mask_np, params.threshold)

            mData.append( {'img_name': image_names[0], 'metric': eval_value_dict, 'heatmap_np': heatmap_np[0], 'gt_mask_np': gt_mask_np, 'pred_mask_np': output_mask_np})


        tstr = f'test epoch: {epoch}, partial ce loss {partial_LOSS.avg}, full ce loss {full_LOSS.avg}'
        print_log(tstr, f_log)


        aDict = {'name': []}
        for sst in performance_metrics:
            aDict[sst] = []
        for tdata in mData:
            for ti in performance_metrics:
                aDict[ti].append(tdata['metric'][ti])

        tmean_auc = np.array(aDict['auc']).mean()
        if tmean_auc > best_auc:
            book = Workbook()
            # item_names = ['Name', 'Dice', 'Acc', 'Sen']
            item_names = ['Name'] + performance_metrics
            tsheet = book.create_sheet("sheet_", 0)
            tsheet.append(item_names)

            #trow = []
            for tdata in mData:
                trow = [ tdata['metric'][tm] for tm in performance_metrics]
                trow = [tdata['img_name']] + trow
                tsheet.append(trow)
                #for tm in performance_metrics:
                #    trow += []

            heatmap_dir = os.path.join(params.log_dir, 'heatmap')
            if not os.path.exists(heatmap_dir):
                os.makedirs(heatmap_dir)

            trow = [np.array(aDict[ti]).mean() for ti in performance_metrics]

            trow = ['mean'] + trow
            tsheet.append(trow)
            book.save(os.path.join(params.log_dir, 'all_eval_metrics.xlsx'))

            for tdata in mData:
                theatmap_np = tdata['heatmap_np']
                tpred_np = tdata['pred_mask_np']
                tname = tdata['img_name']

                save_image_set(theatmap_np, tpred_np, heatmap_dir, tname, params.threshold)


        trow = [np.array(aDict[ti]).mean() for ti in performance_metrics]
        tstr = f'test {epoch} '
        for tmetric, tvalue in zip(performance_metrics, trow):
            tstr += f' mean {tmetric}: {tvalue}'
        print_log(tstr, f_log)

        return tmean_auc


def val(model, dataloader, loss_cri, params, f_log, epoch, writer, best_auc):
    model.eval()

    partial_LOSS = AverageMeter()
    full_LOSS = AverageMeter()

    tstep = len(dataloader)
    dataiter = iter(dataloader)

    mData = []

    with torch.no_grad():

        for idx in range(tstep):

            data = next(dataiter)

            images = data['image'].to(params.device)
            full_masks = data['full_mask'].to(params.device)
            partial_masks = data['partial_mask'].to(params.device)
            valid_masks = data['valid_mask'].to(params.device)
            image_names = data['image_name']
            img_paths = data['image_path']

            heatmap_np = data['heatmap_np']

            bs = images.shape[0]

            output, feat0, feat1 = model(images)

            output_vec = output.view(bs, params.nClass, -1)
            output_vec = torch.transpose(output_vec, 1, 2).reshape(-1, params.nClass)

            full_vec = full_masks.view(-1)
            partial_vec = partial_masks.view(-1)
            valid_vec = valid_masks.view(-1)

            partial_ce_loss = loss_cri(output_vec, partial_vec)
            partial_ce_loss *= valid_vec

            partial_ce_loss = torch.sum(partial_ce_loss) / torch.sum(valid_vec)

            full_ce_loss = loss_cri(output_vec, full_vec)
            full_ce_loss = torch.mean(full_ce_loss)

            partial_LOSS.update(partial_ce_loss.item(), bs)
            full_LOSS.update(full_ce_loss.item(), bs)

            output_softmax = torch.softmax(output, dim=1)
            output_mask_np = output_softmax.detach().cpu()[0].numpy()[1,:,:]
            gt_mask_np = full_masks.cpu()[0].numpy()
            eval_value_dict = eval_metric(output_mask_np, gt_mask_np, params.threshold)

            mData.append( {'img_name': image_names[0], 'metric': eval_value_dict, 'heatmap_np': heatmap_np[0], 'gt_mask_np': gt_mask_np, 'pred_mask_np': output_mask_np})


        tstr = f'val epoch: {epoch}, partial ce loss {partial_LOSS.avg}, full ce loss {full_LOSS.avg}'
        print_log(tstr, f_log)

        return partial_LOSS.avg



#############------------------->
def save_image_set(orig_np, pred_mask_np, folder_dir, tname, threshold):
    cam_np = get_overlaid_images(orig_np, pred_mask_np)
    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8))
    cam_img.save(os.path.join(folder_dir, tname + '_overlaid.jpg'))

    cut_pred_map_np = np.zeros(pred_mask_np.shape)
    cut_pred_map_np[pred_mask_np>threshold] = 1
    cut_pred_map_img = Image.fromarray((cut_pred_map_np*255).astype(np.uint8))
    cut_pred_map_img.save(os.path.join(folder_dir, tname+'_threshold.jpg'))

    pred_mask_img = Image.fromarray((pred_mask_np*255).astype(np.uint8))
    pred_mask_img.save(os.path.join(folder_dir, tname+'_pred.jpg'))



#############------------------->
def get_overlaid_images(orig_np, pred_mask_np):
    pred_mask_img = Image.fromarray((pred_mask_np*255).astype(np.uint8))
    pred_mask_img = pred_mask_img.resize((orig_np.shape[1], orig_np.shape[0]) )
    pred_mask_np = np.array(pred_mask_img)

    #orig_np = np.array(orig_img)

    heatmap = cv2.applyColorMap(np.uint8(pred_mask_np), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    cam = heatmap * 0.3 + np.float32(orig_np) * 0.5
    cam = cam / np.max(cam)
    cam = cam[:,:,::-1]
    return cam





def eval_metric(pred_np, gt_np, threshold, ):

    def auc_func(tpred_np, tgt_np):
        tpred_np = np.reshape(tpred_np, (tpred_np.shape[0]*tpred_np.shape[1]))
        tgt_np = np.reshape(tgt_np, (gt_np.shape[0]*gt_np.shape[1]))
        return roc_auc_score(tgt_np, tpred_np)

    def dice(im1, im2):
        """
        Computes the Dice coefficient, a measure of set similarity.
        Parameters
        ----------
        im1 : array-like, bool
            Any array of arbitrary size. If not boolean, will be converted.
        im2 : array-like, bool
            Any other array of identical size. If not boolean, will be converted.
        Returns
        -------
        dice : float
            Dice coefficient as a float on range [0,1].
            Maximum similarity = 1
            No similarity = 0

        Notes
        -----
        The order of inputs for `dice` is irrelevant. The result will be
        identical if `im1` and `im2` are switched.
        """
        smooth = 1
        im1 = np.asarray(im1).astype(np.bool)
        im2 = np.asarray(im2).astype(np.bool)
        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)
        return (2. * intersection.sum() + smooth) / (im1.sum() + im2.sum() + smooth)

    def performance(pred_mask, label):
        '''
        acc=(TP+TN)/(TP+FN+TN+FP)
        '''
        pred_mask = pred_mask.astype(np.uint8)
        label = label.astype(np.uint8)
        TP, FN, TN, FP = [0, 0, 0, 0]
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if label[i][j] == 1:
                    if pred_mask[i][j] == 1:
                        TP += 1
                    elif pred_mask[i][j] == 0:
                        FN += 1
                elif label[i][j] == 0:
                    if pred_mask[i][j] == 1:
                        FP += 1
                    elif pred_mask[i][j] == 0:
                        TN += 1
        acc = (TP + TN) / (TP + FN + TN + FP)
        kappa = 2.0*(TP*TN-FN*FP) / ( (TP+FP)*(FP+TN) + (TP+FN)*(FN+TN) )


        try:
            sen = TP / (TP + FN)
            spec = TN / (TN + FP)
            prec =  TP / (TP + FP)
            F1 = 2 * (sen * prec) / (sen + prec)
        except:
            sen = 0
            spec = 0
            prec = 0
            F1 = 0

        return acc, sen, spec, prec, F1, kappa

    tauc = auc_func(pred_np, gt_np)

    pred_mask = np.zeros(pred_np.shape)
    pred_mask[pred_np>threshold] = 1

    vdice = dice(pred_mask, gt_np)
    acc, sen, spec, prec, F1, kappa = performance(pred_mask, gt_np)

    temp_list = [ vdice, acc, sen, spec, prec, F1, kappa, tauc ]
    assert len(performance_metrics) == len(temp_list)

    metric_dict = {}

    for tval, tkey in zip(temp_list, performance_metrics):
        metric_dict[tkey] = tval

    #return vdice, acc, sen, spec, prec, F1, kappa
    return metric_dict


###############################------------------->
## coord_ref_size: (x, y)
class image_dataset_camelyon(Dataset):
    def __init__(self, img_paths, mask_dir, heatmap_dir, dot_dir, coord_ref_size, mask_resize, heatmap_resize, isRotate, numCoord=-1, isColorJitter=True):

        self.mask_dir = mask_dir
        #self.img_resize = img_resize
        self.mask_resize = mask_resize
        #self.patch_dirs = img_paths
        #self.transform = transform
        self.dot_dir = dot_dir
        self.isRotate = isRotate
        self.isColorJitter = isColorJitter

        self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.mData = []

        self.ColorJitter = transforms.ColorJitter(0.4, 0.4, 0.4)   # ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1))

        def get_mask_path(img_name):
            return os.path.join(mask_dir, img_name+'.tif')

        ## Get the dot coordinate w.r.t the mask
        imgNames = [ os.path.basename(sst).split('.')[0] for sst in img_paths]

        for tname, timg_path in zip(imgNames, img_paths):

            with open(os.path.join(dot_dir, tname, 'coords.pickle'), 'rb') as f:
                tcoord_dict = pickle.load(f)

            tdata_dict = {}

            tdata_dict['img_name'] = tname     #===
            tdata_dict['img_path'] = timg_path     #===

            ### {'coords': tcoords, 'labels': tlabels}
            in_coords = [ tcoord_dict['coords'][ii] for ii in range(len(tcoord_dict['coords']) ) if tcoord_dict['labels'][ii] ==0]
            out_coords = [ tcoord_dict['coords'][ii] for ii in range(len(tcoord_dict['coords']) ) if tcoord_dict['labels'][ii] ==1]

            if numCoord != -1:
                if len(in_coords) > numCoord:
                    in_coords = in_coords[:numCoord]
                if len(out_coords) > numCoord:
                    out_coords = out_coords[:numCoord]

            in_coords = [ (float(sst[0]) / coord_ref_size[0], float(sst[1]) / coord_ref_size[1]) for sst in in_coords ]
            out_coords = [ (float(sst[0]) / coord_ref_size[0], float(sst[1]) / coord_ref_size[1]) for sst in out_coords ]

            in_coords = [ ( int(sst[0]*mask_resize[0]), int(sst[1]*mask_resize[1]) ) for sst in in_coords ]
            out_coords = [ ( int(sst[0]*mask_resize[0]), int(sst[1]*mask_resize[1]) ) for sst in out_coords ]

            tdata_dict['in_coords'] = in_coords    #===
            tdata_dict['out_coords'] = out_coords   #===

            partial_mask_np = np.zeros(mask_resize)
            valid_mask_np = np.zeros(mask_resize)

            for tcood in in_coords:
                partial_mask_np[tcood[1], tcood[0]] = 1
                valid_mask_np[tcood[1], tcood[0]] = 1
            for tcood in out_coords:
                valid_mask_np[tcood[1], tcood[0]] = 1

            tdata_dict['partial_mask_np'] = partial_mask_np    #===
            tdata_dict['valid_mask_np'] = valid_mask_np    #===


            orig_timage = Image.open(timg_path).convert('RGB')
            timage = orig_timage.copy()

            heatmap_img = Image.open(os.path.join(heatmap_dir, tname+'.tif'))
            heatmap_img = heatmap_img.resize((heatmap_resize[0], heatmap_resize[1]))
            heatmap_np = np.array(heatmap_img)
            tdata_dict['heatmap_np'] = heatmap_np

            tdata_dict['image_img'] = timage     #===


            tmask_path = get_mask_path(tname)
            tmask_img = Image.open(tmask_path).convert('L')
            tmask_img = tmask_img.resize((self.mask_resize[0], self.mask_resize[1]))
            full_mask_np = np.array(tmask_img) // 255

            tdata_dict['full_mask_np'] = full_mask_np     #===

            self.mData.append(tdata_dict)


    def __getitem__(self, index):

        tdata = self.mData[index]

        timg = tdata['image_img']
        if self.isColorJitter:
            timg = self.ColorJitter(timg)
        timg_np = np.array(timg)

        tfull_mask_np = tdata['full_mask_np']
        tpartial_mask_np = tdata['partial_mask_np']
        tvadlie_mask_np = tdata['valid_mask_np']
        tname = tdata['img_name']
        tpath = tdata['img_path']
        in_coords = tdata['in_coords']
        out_coords = tdata['out_coords']
        theapmap_np = tdata['heatmap_np']


        rnd = random.randint(0, 3)
        if self.isRotate:
            timg_np = np.rot90(timg_np, rnd, axes=(0,1))
            tfull_mask_np = np.rot90(tfull_mask_np, rnd, axes=(0,1))
            tpartial_mask_np = np.rot90(tpartial_mask_np, rnd, axes=(0,1))
            tvadlie_mask_np = np.rot90(tvadlie_mask_np, rnd, axes=(0,1))

        timg_np = timg_np.transpose((2,0,1)).copy()
        tfull_mask_np = tfull_mask_np.copy()
        tpartial_mask_np = tpartial_mask_np.copy()
        tvadlie_mask_np = tvadlie_mask_np.copy()

        timg_ts = torch.from_numpy(timg_np).float()

        timg_ts = self.normalization(timg_ts)

        tfull_mask_ts = torch.from_numpy(tfull_mask_np).long()
        tpartial_mask_ts = torch.from_numpy(tpartial_mask_np).long()
        tvadlie_mask_ts = torch.from_numpy(tvadlie_mask_np).float()

        in_coords_ts = torch.LongTensor(in_coords)
        out_coords_ts = torch.LongTensor(out_coords)


        return {'image': timg_ts, 'full_mask': tfull_mask_ts, 'partial_mask': tpartial_mask_ts,
                'valid_mask': tvadlie_mask_ts, 'image_name': tname, 'image_path': tpath, 'in_coords': in_coords_ts, 'out_coords': out_coords_ts, 'heatmap_np':theapmap_np, 'rnd': torch.LongTensor([rnd]) }

    def __len__(self):
        return len(self.mData)





####################------------------->>
def get_img_paths_subset_withVal(img_dir, subSetList_path, index, dot_dir, gap=5):

    dot_names = os.listdir(dot_dir)

    with open(subSetList_path, 'rb') as f:
        subSetList = pickle.load(f)

    test_set = subSetList.pop(index)

    train_set = []
    for sst in subSetList:
        train_set.extend(sst)

    test_set = [sst for sst in test_set if sst in dot_names]
    train_set = [sst for sst in train_set if sst in dot_names]


    train_set.sort()

    tidx = list(range(0,len(train_set), gap))

    val_set = [ train_set[sst] for sst in range(len(train_set)) if sst in tidx ]
    train_set = [ train_set[sst] for sst in range(len(train_set)) if sst not in tidx ]


    test_set = [ os.path.join(img_dir, sst+'.tif') for sst in test_set ]
    train_set = [ os.path.join(img_dir, sst+'.tif') for sst in train_set ]
    val_set = [ os.path.join(img_dir, sst+'.tif') for sst in val_set ]

    print(f' trainset: {len(train_set)}, test_set: {len(test_set)}, val_set: {len(val_set)}')

    return train_set, test_set, val_set


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




if __name__ == "__main__":
    main()
