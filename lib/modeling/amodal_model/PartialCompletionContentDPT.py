import multiprocessing as mp
import argparse
import yaml
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# from deocclusion import utils
import numpy as np
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from lib.modeling.amodal_model import amodal_utils
from lib.modeling.amodal_model.dpt import DPTDepthModel
from lib.modeling.amodal_model import unet

class PartialCompletionContentDPT(nn.Module):

    def __init__(self):
        super(PartialCompletionContentDPT, self).__init__()

        self.with_modal = True



        self.model = DPTDepthModel(
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )


        param = {'in_channels': 2, 'n_classes': 2}
        self.amodal_mask=unet.unet2(**param)
        self.amodal_mask.cuda()
        self.model.cuda()


        cudnn.benchmark = True
    #return depth_input, depth_gt,img_input
    def set_input(self, depth_gt,rgb,mask,hint):
        self.depth_gt = depth_gt.cuda()
        self.rgb = rgb.cuda()
        self.mask = mask.cuda()#eraser
        self.loss_mask = torch.ones_like(self.mask).cuda()

        self.hint = hint.cuda()#modal


    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            amodal_output = self.amodal_mask(torch.cat([self.hint, self.mask], dim=1))
            comp = amodal_output.argmax(dim=1, keepdim=True).float()
            comp[self.mask == 0] = (self.hint > 0).float()[self.mask == 0]

            if self.with_modal:
                #rgb+depth
                output = self.model(torch.cat([self.rgb, comp], dim=1), comp)
            else:
                output, _ = self.model(self.rgb, self.visible_mask3)
            if output.shape[-1] != self.rgb.shape[-1]:
                output = nn.functional.interpolate(
                    output, size=self.rgb.shape[2:4],
                    mode="bilinear", align_corners=True)

            output_comp = (1-self.mask) * self.depth_gt + (self.mask) * output

        if self.with_modal:
            mask_tensors = [self.mask, self.hint,comp]
        else:
            mask_tensors = [self.mask,self.rgb]
        ret_tensors = {'common_tensors': [ output,output_comp, self.depth_gt],
                       'mask_tensors': mask_tensors}
        if ret_loss:
            loss_dict = {}
            loss_dict['l1'] = self.l1_loss(output, self.depth_gt,self.loss_mask)
            loss_dict['sig'] = self.sig_loss(output, self.depth_gt, self.loss_mask)
            for k in loss_dict.keys():
                loss_dict[k] /= self.world_size
            return ret_tensors, loss_dict
        else:
            return ret_tensors

    def step(self):
        with torch.no_grad():
            amodal_output = self.amodal_mask(torch.cat([self.hint, self.mask], dim=1))
            comp = amodal_output.argmax(dim=1, keepdim=True).float()
            comp[self.mask == 0] = (self.hint > 0).float()[self.mask == 0]
        # output
        if self.with_modal:
            #rgb + depth
            output = self.model(torch.cat([self.rgb, comp], dim=1), comp)
        else:
            output, _ = self.model(self.rgb, self.visible_mask3)
        if output.shape[2] != self.rgb.shape[2]:
            output = nn.functional.interpolate(
                output, size=self.rgb.shape[2:4],
                mode="bilinear", align_corners=True)
        loss_dict = {}
        gen_loss = 0
        loss_dict['sig']=self.sig_loss(output, self.depth_gt, self.loss_mask)
        loss_dict['l1'] = self.l1_loss(output, self.depth_gt, self.loss_mask)
        for key, coef in self.params['lambda_dict'].items():
            value = coef * loss_dict[key]
            gen_loss += value

        # # update
        self.optim.zero_grad()
        gen_loss.backward()
        self.optim.step()


        return loss_dict


    def load_state(self, root, Iter, resume=False):
        path = os.path.join(root, "ckpt_iter_{}.pth.tar".format(Iter))
        netD_path = os.path.join(root, "D_iter_{}.pth.tar".format(Iter))

        if resume:
            amodal_utils.load_state(path, self.model, self.optim)
            #utils.load_state(netD_path, self.netD, self.optimD)
        else:
            amodal_utils.load_state(path, self.model)
           # utils.load_state(netD_path, self.netD)

    def save_state(self, root, Iter):
        path = os.path.join(root, "ckpt_iter_{}.pth.tar".format(Iter))
        netD_path = os.path.join(root, "D_iter_{}.pth.tar".format(Iter))

        torch.save({
            'step': Iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict()}, path)

    def switch_to(self, phase):
        if phase == 'train':
            self.model.train()
            self.amodal_mask.eval()
        else:
            self.model.eval()
            #if not self.demo:
            self.amodal_mask.eval()

    def load_model_demo(self, path,amodal_path):
        amodal_utils.load_state(path, self.model)
        amodal_utils.load_weights(amodal_path, self.amodal_mask)

    def inf(self,rgb,hint,mask):
        with torch.no_grad():
            amodal_output = self.amodal_mask(torch.cat([hint, mask], dim=1))
            comp = amodal_output.argmax(dim=1, keepdim=True).float()
            comp[mask == 0] = (hint > 0).float()[mask == 0]
        # output

            #rgb + depth
            output,d_feat = self.model(torch.cat([rgb, comp], dim=1), comp)

        return output,comp,d_feat
