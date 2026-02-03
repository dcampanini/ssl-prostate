import os
import math
import time
from functools import partial
from matplotlib.pyplot import grid
import numpy as np
from numpy import nanmean, nonzero, percentile
from torchprofile import profile_macs

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import sys
sys.path.append('..')

import lib.models as models
import lib.networks as networks
from lib.utils import SmoothedValue, concat_all_gather, LayerDecayValueAssigner

import wandb

from lib.data.med_transforms import get_scratch_train_transforms, get_val_transforms, get_post_transforms, get_vis_transforms, get_raw_transforms
from lib.data.med_datasets import get_msd_trainset, get_train_loader, get_val_loader, idx2label_all, btcv_8cls_idx
from lib.tools.visualization import patches3d_to_grid, images3d_to_grid
from lib.models import UNet
from .base_trainer import BaseTrainer

from timm.data import Mixup
from timm.utils import accuracy
from lib.utils import to_3tuple

from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import compute_dice, compute_hausdorff_distance

from collections import defaultdict, OrderedDict

import json
from report_guided_annotation import extract_lesion_candidates
from picai_eval import evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F

from picai_baseline.unet.training_setup.data_generator import prepare_datagens, get_testing_datagen
from picai_baseline.unet.training_setup.augmentations.nnUNet_DA import \
    apply_augmentations



class FocalLossCustom(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum"):
        super(FocalLossCustom, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets, num_classes=self.num_classes).float()
        #print('targets1',targets.shape)
        targets = torch.moveaxis(targets, (0, 1, 2, 3, 4), (0, 2, 3, 4, 1))
        #print('targets2', targets.shape)
        #print('inputs', inputs.shape)
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = (inputs[-1] * targets[-1]) + ((1 - inputs[-1]) * (1 - targets[-1]))
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets[-1] + (1 - self.alpha) * (1 - targets[-1])
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class FocalLossWithMMD(nn.Module):
    """Focal loss with MMD penalty for regularized fine-tuning."""

    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum", 
                 mmd_weight=0.1, mmd_sigma=10000):
        super(FocalLossWithMMD, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        self.mmd_weight = mmd_weight
        self.mmd_sigma = mmd_sigma

    def gaussian_kernel(self, x, y, sigma):
        """RBF/Gaussian kernel."""
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)

        
        kernel_input = (tiled_x - tiled_y).pow(2).sum(2)
        return torch.exp(-kernel_input / (2 * sigma ** 2))

    def compute_mmd(self, x, y):
        """
        Compute MMD between two sets of samples.
        x: features from pretrained model
        y: features from finetuned model
        """
        # Flatten spatial dimensions: [B, C, D, H, W] -> [B, C*D*H*W]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if y.dim() > 2:
            y = y.view(y.size(0), -1)
        
        x_kernel = self.gaussian_kernel(x, x, self.mmd_sigma)
        y_kernel = self.gaussian_kernel(y, y, self.mmd_sigma)
        xy_kernel = self.gaussian_kernel(x, y, self.mmd_sigma)


        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    def forward(self, inputs, targets, pretrained_features=None, finetuned_features=None):
        """
        Args:
            inputs: model predictions
            targets: ground truth labels
            pretrained_features: features from frozen pre-trained model
            finetuned_features: features from fine-tuning model
        """
        # Compute focal loss
        import pdb; pdb.set_trace()
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets, num_classes=self.num_classes).float()
        targets = torch.moveaxis(targets, (0, 1, 2, 3, 4), (0, 2, 3, 4, 1))
        
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = (inputs * targets) + ((1 - inputs) * (1 - targets))
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            focal_loss = loss.mean()
        elif self.reduction == "sum":
            focal_loss = loss.sum()
        else:
            focal_loss = loss

        # Add MMD penalty if features are provided
        total_loss = focal_loss
        mmd_loss = torch.tensor(0.0, device=focal_loss.device)
        
        if pretrained_features is not None and finetuned_features is not None:
            mmd_loss = self.compute_mmd(pretrained_features, finetuned_features)
            total_loss = focal_loss + self.mmd_weight * mmd_loss

        return total_loss, focal_loss, mmd_loss


class FeatureExtractor:
    """Helper class to extract features from any layer using hooks"""
    def __init__(self, model, layer_name):
        self.features = None
        self.hook = None
        self.layer_name = layer_name
        
        # Register hook on the specified layer
        layer = self._get_layer_by_name(model, layer_name)
        self.hook = layer.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture layer output"""
        self.features = output
    
    def _get_layer_by_name(self, model, layer_name):
        """Get layer by dot-separated name (e.g., 'encoder.blocks.11')"""
        layers = layer_name.split('.')
        layer = model
        for l in layers:
            layer = getattr(layer, l)
        return layer
    
    def get_features(self):
        return self.features
    
    def remove(self):
        """Remove the hook"""
        if self.hook is not None:
            self.hook.remove()


def compute_avg_metric(metric, meters, metric_name, batch_size, args):
    assert len(metric.shape) == 2
    if args.dataset == 'btcv':
        # cls_avg_metric = np.nanmean(np.nanmean(metric, axis=0))
        cls_avg_metric = np.mean(np.ma.masked_invalid(np.nanmean(metric, axis=0)))
        # cls8_avg_metric = np.nanmean(np.nanmean(metric[..., btcv_8cls_idx], axis=0))
        cls8_avg_metric = np.nanmean(np.ma.masked_invalid(np.nanmean(metric[..., btcv_8cls_idx], axis=0)))
        meters[metric_name].update(value=cls_avg_metric, n=batch_size)
        meters[f'cls8_{metric_name}'].update(value=cls8_avg_metric, n=batch_size)
    else:
        cls_avg_metric = np.nanmean(np.nanmean(metric, axis=0))
        meters[metric_name].update(value=cls_avg_metric, n=batch_size)

class SegTrainerMMD(BaseTrainer):
    r"""
    General Segmentation Trainer
    """
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.proj_name
        print('DEVICES count', torch.cuda.device_count(), 'is available', torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Add MMD-related attributes
        self.pretrained_model = None
        self.feature_extractor = None
        self.pretrained_feature_extractor = None
        self.mmd_feature_layer = None
        
        if args.test:
            self.metric_funcs = OrderedDict([
                                        ('Dice', 
                                        compute_dice),
                                        ('HD',
                                        partial(compute_hausdorff_distance, percentile=95))
                                        ])
        else:
            self.metric_funcs = OrderedDict([
                                        ('Dice', 
                                        compute_dice)
                                        ])
    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name}")

            # Setup loss functions based on dataset
            if args.dataset == 'btcv':
                args.num_classes = 14
                self.loss_fn = DiceCELoss(to_onehot_y=True,
                                        softmax=True,
                                        squared_pred=True,
                                        smooth_nr=args.smooth_nr,
                                        smooth_dr=args.smooth_dr)
                self.using_custom_focal = False
            elif args.dataset == 'msd_brats':
                args.num_classes = 3
                self.loss_fn = DiceLoss(to_onehot_y=False, 
                                        sigmoid=True, 
                                        squared_pred=True, 
                                        smooth_nr=args.smooth_nr, 
                                        smooth_dr=args.smooth_dr)
                self.using_custom_focal = False
            elif args.dataset == 'prostate':
                args.num_classes = 2
                device = 'cuda' if args.gpu else 'cpu'
                
                # Check if using MMD
                use_mmd = getattr(args, 'use_mmd', True)
                
                if use_mmd:
                    print("=> Using Focal Loss with MMD penalty")
                    self.using_custom_focal = True
                    self.loss_fn = FocalLossWithMMD(
                        alpha=0.69857891, 
                        gamma=1.0,
                        num_classes=2,
                        reduction="sum",
                        mmd_weight=getattr(args, 'mmd_weight', 0.1),
                        mmd_sigma=getattr(args, 'mmd_sigma', 1.0)
                    ).to(device)
                else:
                    print("=> Using standard Focal Loss")
                    self.using_custom_focal = True
                    self.loss_fn = FocalLossCustom(alpha=0.69857891, gamma=1.0).to(device)
            else:
                raise ValueError(f"Unsupported dataset {args.dataset}")
            
            self.post_pred, self.post_label = get_post_transforms(args)

            # setup mixup and loss functions
            if args.mixup > 0:
                raise NotImplemented("Mixup for segmentation has not been implemented.")
            else:
                self.mixup_fn = None
            
            # Create model
            if self.model_name == 'UNet':
                self.model = UNet(
                                spatial_dims=len(args.image_shape),
                                in_channels=args.num_channels,
                                out_channels=args.out_chans,
                                strides=args.model_strides,
                                channels=args.model_features
                                )
            else:
                self.model = getattr(models, self.model_name)(
                    encoder=getattr(networks, args.enc_arch),
                    decoder=getattr(networks, args.dec_arch),
                    args=args
                )

            # Load pretrained weights
            print('DEVICES', torch.cuda.device_count())
            if hasattr(args, 'test') and args.test and args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Start loading the model weights from {args.pretrain} for test")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                state_dict = checkpoint['state_dict']
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")
                
            elif args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Start loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load based on model type
                if self.model_name == 'UNETR3D' or self.model_name == 'UNETR3DFusion':
                    for key in list(state_dict.keys()):
                        if key.startswith('encoder.'):
                            state_dict[key[len('encoder.'):]] = state_dict[key]
                            del state_dict[key]
                        if key == 'encoder_pos_embed':
                            pe = torch.zeros([1, 1, state_dict[key].size(-1)])
                            state_dict['pos_embed'] = torch.cat([pe, state_dict[key]], dim=1)
                            del state_dict[key]
                        if key == 'patch_embed.proj.weight' and \
                            state_dict['patch_embed.proj.weight'].shape != self.model.encoder.patch_embed.proj.weight.shape:
                            del state_dict['patch_embed.proj.weight']
                            del state_dict['patch_embed.proj.bias']
                        if key == 'pos_embed' and \
                            state_dict['pos_embed'].shape != self.model.encoder.pos_embed.shape:
                            del state_dict[key]
                    msg = self.model.encoder.load_state_dict(state_dict, strict=False)
                    
                elif self.model_name == 'DynSeg3d':
                    if args.pretrain_load == 'enc+dec':
                        for key in list(state_dict.keys()):
                            if key.startswith('decoder.head.') or (key.startswith('decoder.blocks.') and int(key[15]) > 7):
                                del state_dict[key]
                    elif args.pretrain_load == 'enc':
                        for key in list(state_dict.keys()):
                            if key.startswith('decoder.'):
                                del state_dict[key]
                    msg = self.model.load_state_dict(state_dict, strict=False)
                    
                elif self.model_name == 'UNet':
                    print('Printing keys to remove')
                    for key, value in list(state_dict.items()):
                        if '2' in key:
                            print(key, value.shape)
                            del state_dict[key]
                    msg = self.model.load_state_dict(state_dict, strict=False)
                
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")
            
            # CREATE FROZEN MMD REFERENCE MODEL (only if use_mmd is True)
            if getattr(args, 'use_mmd', True):
                #mmd_reference_path = getattr(args, 'mmd_reference', None)
                mmd_reference_path = args.mmd_reference
                
                if mmd_reference_path is not None and os.path.exists(mmd_reference_path):
                    print(f"\n{'='*60}")
                    print(f"=> Creating frozen reference model for MMD from {mmd_reference_path}")
                    
                    # Create reference model with same architecture
                    if self.model_name == 'UNet':
                        self.pretrained_model = UNet(
                                        spatial_dims=len(args.image_shape),
                                        in_channels=args.num_channels,
                                        out_channels=args.out_chans,
                                        strides=args.model_strides,
                                        channels=args.model_features
                                        )
                    else:
                        self.pretrained_model = getattr(models, self.model_name)(
                            encoder=getattr(networks, args.enc_arch),
                            decoder=getattr(networks, args.dec_arch),
                            args=args
                        )
                    
                    # Load reference weights from mmd_reference
                    checkpoint_ref = torch.load(mmd_reference_path, map_location='cpu')
                    if 'state_dict' in checkpoint_ref:
                        state_dict_ref = checkpoint_ref['state_dict']
                    else:
                        state_dict_ref = checkpoint_ref
                    
                    # Process state_dict based on model type
                    if self.model_name == 'UNETR3D' or self.model_name == 'UNETR3DFusion':
                        for key in list(state_dict_ref.keys()):
                            if key.startswith('encoder.'):
                                state_dict_ref[key[len('encoder.'):]] = state_dict_ref[key]
                                del state_dict_ref[key]
                            if key == 'encoder_pos_embed':
                                pe = torch.zeros([1, 1, state_dict_ref[key].size(-1)])
                                state_dict_ref['pos_embed'] = torch.cat([pe, state_dict_ref[key]], dim=1)
                                del state_dict_ref[key]
                            if key == 'patch_embed.proj.weight' and \
                                'patch_embed.proj.weight' in state_dict_ref and \
                                state_dict_ref['patch_embed.proj.weight'].shape != self.pretrained_model.encoder.patch_embed.proj.weight.shape:
                                del state_dict_ref['patch_embed.proj.weight']
                                del state_dict_ref['patch_embed.proj.bias']
                            if key == 'pos_embed' and \
                                'pos_embed' in state_dict_ref and \
                                state_dict_ref['pos_embed'].shape != self.pretrained_model.encoder.pos_embed.shape:
                                del state_dict_ref[key]
                        msg_ref = self.pretrained_model.encoder.load_state_dict(state_dict_ref, strict=False)
                        
                    elif self.model_name == 'DynSeg3d':
                        if args.pretrain_load == 'enc+dec':
                            for key in list(state_dict_ref.keys()):
                                if key.startswith('decoder.head.') or (key.startswith('decoder.blocks.') and int(key[15]) > 7):
                                    del state_dict_ref[key]
                        elif args.pretrain_load == 'enc':
                            for key in list(state_dict_ref.keys()):
                                if key.startswith('decoder.'):
                                    del state_dict_ref[key]
                        msg_ref = self.pretrained_model.load_state_dict(state_dict_ref, strict=False)
                        
                    elif self.model_name == 'UNet':
                        for key in list(state_dict_ref.keys()):
                            if '2' in key:
                                del state_dict_ref[key]
                        msg_ref = self.pretrained_model.load_state_dict(state_dict_ref, strict=False)
                    
                    print(f'Reference model loading messages: \n {msg_ref}')
                    
                    # Freeze the reference model
                    self.pretrained_model.eval()
                    for param in self.pretrained_model.parameters():
                        param.requires_grad = False
                    
                    print(f"=> Reference model frozen successfully")
                    
                    # Determine feature extraction layer
                    if self.model_name == 'UNETR3D' or self.model_name == 'UNETR3DFusion':
                        self.mmd_feature_layer = getattr(args, 'mmd_feature_layer', 'encoder.blocks.11')
                    elif self.model_name == 'DynSeg3d':
                        self.mmd_feature_layer = getattr(args, 'mmd_feature_layer', 'encoder.blocks.11')
                    elif self.model_name == 'UNet':
                        self.mmd_feature_layer = getattr(args, 'mmd_feature_layer', 'bottleneck')
                    else:
                        self.mmd_feature_layer = 'encoder'
                    
                    print(f"=> MMD will use features from layer: {self.mmd_feature_layer}")
                    print(f"{'='*60}\n")
                else:
                    print(f"WARNING: --use_mmd is set but --mmd_reference is not provided or doesn't exist")
                    if mmd_reference_path is not None:
                        print(f"         Path: {mmd_reference_path}")
                    print(f"         MMD will be disabled")
                    self.pretrained_model = None
            else:
                # MMD not requested
                self.pretrained_model = None

            # Wrap model (must be done before feature extractors)
            self.wrap_model()
            
            # Setup feature extractors after wrapping and GPU transfer
            if getattr(args, 'use_mmd', True) and self.pretrained_model is not None:
                # Move pretrained model to GPU
                if args.gpu is not None:
                    self.pretrained_model = self.pretrained_model.cuda(args.gpu)
                
                # Get the actual model (unwrap if DataParallel/DistributedDataParallel)
                actual_model = self.model.module if hasattr(self.model, 'module') else self.model
                
                # Create feature extractors
                try:
                    self.feature_extractor = FeatureExtractor(
                        actual_model,
                        self.mmd_feature_layer
                    )
                    self.pretrained_feature_extractor = FeatureExtractor(
                        self.pretrained_model,
                        self.mmd_feature_layer
                    )
                    print("=> Feature extractors created successfully")
                except Exception as e:
                    print(f"=> Warning: Could not create feature extractors: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"=> MMD will be disabled")
                    self.pretrained_model = None
                    self.feature_extractor = None
                    self.pretrained_feature_extractor = None
            else:
                self.feature_extractor = None
                self.pretrained_feature_extractor = None
                    
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert(self.model is not None and self.wrapped_model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args
        model = self.model

        num_layers = model.get_num_layers()
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))

        optim_params = self.get_parameter_groups(get_layer_id=partial(assigner.get_layer_id, prefix='encoder.'), 
                                                 get_layer_scale=assigner.get_scale, 
                                                 verbose=True)
        self.optimizer = torch.optim.AdamW(optim_params, 
                                            lr=args.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)

    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating train dataloader")
            args = self.args

            if args.dataset in ['btcv', 'msd_brats', 'prostate']:
                if not args.eval_test:
                    train_gen, valid_gen, class_weights = prepare_datagens(args=args, fold_id=args.fold_id, seed_for_shuffle=args.data_seed)

                    # integrate data augmentation pipeline from nnU-Net
                    self.dataloader = apply_augmentations(
                                dataloader=train_gen,
                                num_threads=args.workers,
                                disable=False,
                                seeds_train=args.augmentation_seeds
                                                )
                    self.iters_per_epoch = 200
                    print(f"==> Length of train dataloader is {self.iters_per_epoch}")
                    self.val_dataloader = valid_gen
                else:
                    if args.test_file:
                        self.val_dataloader = get_testing_datagen(args=args, test_file=args.test_file)
                    else:
                        self.val_dataloader = get_testing_datagen(args=args)
                
            elif args.dataset == 'brats20':
                raise NotImplementedError("brats20 transforms and dataloaders on MONAI has not been implemented yet.")
            else:
                raise ValueError("Currently only support brats2020 dataset")
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        print("=> finish creating dataloader")

    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        best_metric = 0
        best_ts_metric = 0

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            
            if epoch == args.start_epoch:
                self.evaluate(epoch=epoch, niters=niters)

            # train for one epoch
            niters = self.epoch_train(epoch, niters)
        
            # evaluate after each epoch training
            if (epoch + 1) % args.eval_freq == 0:
                metric_list = self.evaluate(epoch=epoch, niters=niters)
                metric = metric_list[1].score if args.dataset == 'prostate' else metric_list[0]
                if len(metric_list) == 3:
                    ts_metric = metric_list[2]
                elif len(metric_list) == 2 and args.dataset != 'prostate':
                    ts_metric = metric_list[1]
                else:
                    ts_metric = None
                if metric > best_metric:
                    print(f"=> New val best metric: {metric} | Old val best metric: {best_metric}!")
                    best_metric = metric
                    if ts_metric is not None:
                        print(f"=> New ts best metric: {ts_metric} | Old ts best metric: {best_ts_metric}!")
                        best_ts_metric = ts_metric
                    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                        self.save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'arch': args.arch,
                                'state_dict': self.model.state_dict(),
                                'optimizer' : self.optimizer.state_dict(),
                                'scaler': self.scaler.state_dict(),
                                'metric':metric
                            }, 
                            is_best=False, 
                            filename=f'{args.ckpt_dir}/best_model.pth.tar'
                        )
                        print("=> Finish saving best model.")
                else:
                    print(f"=> Still old val best metric: {best_metric}")
                    if ts_metric is not None:
                        print(f"=> Still old ts best metric: {best_ts_metric}")

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if (epoch + 1) % args.save_freq == 0:
                    if args.save_intermediate:
                        self.save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'arch': args.arch,
                                'state_dict': self.model.state_dict(),
                                'optimizer' : self.optimizer.state_dict(),
                                'scaler': self.scaler.state_dict(),
                            }, 
                            is_best=False, 
                            filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar'
                        )
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(),
                        }, 
                        is_best=False, 
                        filename=f'{args.ckpt_dir}/last_check.pth.tar'
                    )

    @staticmethod
    def train_class_batch(model, samples, target, criterion, 
                         pretrained_model=None, 
                         feature_extractor=None,
                         pretrained_feature_extractor=None):
        """Modified to handle MMD penalty with hooks"""
        
        # Forward pass through fine-tuned model
        outputs = model(samples)
        #import pdb; pdb.set_trace()
        # Check if using MMD
        if (pretrained_model is not None and 
            isinstance(criterion, FocalLossWithMMD) and
            feature_extractor is not None and
            pretrained_feature_extractor is not None):
            
            # Get features from fine-tuned model (already computed in forward)
            finetuned_features = feature_extractor.get_features()
            
            # Get features from pretrained model
            with torch.no_grad():
                _ = pretrained_model(samples)
                pretrained_features = pretrained_feature_extractor.get_features()
            
            # Compute loss with MMD
            total_loss, focal_loss, mmd_loss = criterion(
                outputs, 
                target, 
                pretrained_features.detach(), 
                finetuned_features
            )
            return total_loss, focal_loss, mmd_loss
        else:
            # Standard focal loss
            loss = criterion(outputs, target)

            return loss, torch.tensor(0.0, device=loss.device), torch.tensor(0.0, device=loss.device)

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler
        mixup_fn = self.mixup_fn
        loss_fn = self.loss_fn

        # switch to train mode
        model.train()
        if self.pretrained_model is not None:
            self.pretrained_model.eval()  # Keep pretrained model in eval mode

        load_start_time = time.time()
        step = 0
        for i, batch_data in enumerate(train_loader):
            step += 1
            load_time = time.time() - load_start_time
            # adjust learning at the beginning of each iteration
            if args.start_from and args.start_from_lr:
                pass
            else:
                self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            if args.gpu is not None:
                try:
                    image = batch_data['data'].to(args.gpu, non_blocking=True)
                    target = batch_data['seg'].to(args.gpu, non_blocking=True)
                except Exception:
                    image = torch.from_numpy(batch_data['data']).to(args.gpu, non_blocking=True)
                    target = torch.from_numpy(batch_data['seg']).to(args.gpu, non_blocking=True)
            else:
                print('No gpu selected, stopping')
                break

            if mixup_fn is not None:
                image, target = mixup_fn(image, target)

            # compute output and loss
            forward_start_time = time.time()
            
            if self.using_custom_focal:
                loss, focal_loss, mmd_loss = self.train_class_batch(
                    model, 
                    image, 
                    target[:, 0, ...].long(), 
                    loss_fn,
                    pretrained_model=self.pretrained_model,
                    feature_extractor=self.feature_extractor,
                    pretrained_feature_extractor=self.pretrained_feature_extractor
                )
            else:
                loss, focal_loss, mmd_loss = self.train_class_batch(model, image, target, loss_fn)
                
            forward_time = time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time = time.time() - bp_start_time

            # Log to the screen
            if i % args.print_freq == 0:
                if 'lr_scale' in optimizer.param_groups[0]:
                    last_layer_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_scale']
                else:
                    last_layer_lr = optimizer.param_groups[0]['lr']

                log_str = (f"Epoch: {epoch:03d}/{args.epochs} | "
                          f"Iter: {i:05d}/{self.iters_per_epoch} | "
                          f"TotalIter: {niters:06d} | "
                          f"Init Lr: {self.lr:.05f} | "
                          f"Lr: {last_layer_lr:.05f} | "
                          f"Load Time: {load_time:.03f}s | "
                          f"Forward Time: {forward_time:.03f}s | "
                          f"Backward Time: {bp_time:.03f}s | "
                          f"Loss: {loss.item():.03f}")
                
                # Add MMD info if using it
                if isinstance(mmd_loss, torch.Tensor) and mmd_loss.item() > 0:
                    log_str += f" | Focal: {focal_loss.item():.03f} | MMD: {mmd_loss.item():.05f}"
                
                print(log_str)
                
                if args.rank == 0 and not args.disable_wandb:
                    log_dict = {
                        "lr": last_layer_lr,
                        "Loss": loss.item(),
                    }
                    if isinstance(mmd_loss, torch.Tensor) and mmd_loss.item() > 0:
                        log_dict["Focal_Loss"] = focal_loss.item()
                        log_dict["MMD_Loss"] = mmd_loss.item()
                    
                    wandb.log(log_dict, step=niters)

            niters += 1
            load_start_time = time.time()
            if step >= self.iters_per_epoch: 
                break
        return niters

    @torch.no_grad()
    def evaluate(self, epoch=0, niters=0, get_only_output=False):
        print("=> Start Evaluating")
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader
        
        if args.spatial_dim == 3:
            roi_size = (args.roi_x, args.roi_y, args.roi_z)
        elif args.spatial_dim == 2:
            roi_size = (args.roi_x, args.roi_y)
        else:
            raise ValueError(f"Do not support this spatial dimension (={args.spatial_dim}) for now")

        meters = defaultdict(SmoothedValue)
        if hasattr(args, 'ts_ratio') and args.ts_ratio != 0:
            assert args.batch_size == 1, "Test mode requires batch size 1"
            #ts_samples = int(len(val_loader) * args.ts_ratio)
            #val_samples = len(val_loader) - ts_samples
            val_samples = 300
            ts_meters = defaultdict(SmoothedValue)
        else:
            ts_samples = 0
            val_samples = 300
            #val_samples = len(val_loader)
            ts_meters = None
        #print(f"val samples: nose XD and test samples: {ts_samples}")

        # switch to evaluation mode
        model.eval()
        all_valid_preds, all_valid_labels = [], [] # Esto es solo para prostate
        for i, batch_data in enumerate(val_loader):
            #image, target = batch_data['image'].to(args.gpu, non_blocking=True), batch_data['label'].to(args.gpu, non_blocking=True)
            
            if args.gpu is not None:
                try:
                    image = batch_data['data'].to(args.gpu, non_blocking=True)
                    target = batch_data['seg'].to(args.gpu, non_blocking=True)
                except Exception:
                    image = torch.from_numpy(batch_data['data']).to(args.gpu, non_blocking=True)
                    target = torch.from_numpy(batch_data['seg']).to(args.gpu, non_blocking=True)
            
            # compute output
            if args.use_test_augmentation:
                images = [image, torch.flip(image, [4])]
                preds = []
                for im in images:
                    with torch.cuda.amp.autocast():
                        output = sliding_window_inference(im,
                                                  roi_size=roi_size,
                                                  sw_batch_size=4,
                                                  predictor=model,
                                                  overlap=args.infer_overlap)
                    output_convert = torch.stack([self.post_pred(output_tensor) for output_tensor in decollate_batch(output)], dim=0)
                    preds.append(output_convert)
                preds[1] = torch.flip(preds[1], [4])
                output_convert = torch.mean(torch.stack(preds), dim=0)
                target_convert = torch.stack([self.post_label(target_tensor) for target_tensor in decollate_batch(target)], dim=0)
                
            else:
                with torch.cuda.amp.autocast():
                    output = sliding_window_inference(image,
                                                  roi_size=roi_size,
                                                  sw_batch_size=4,
                                                  predictor=model,
                                                  overlap=args.infer_overlap)
                target_convert = torch.stack([self.post_label(target_tensor) for target_tensor in decollate_batch(target)], dim=0)
                output_convert = torch.stack([self.post_pred(output_tensor) for output_tensor in decollate_batch(output)], dim=0)

            batch_size = image.size(0)
            idx2label = idx2label_all[args.dataset]
            for metric_name, metric_func in self.metric_funcs.items():
                if i < val_samples:
                    log_meters = meters
                else:
                    log_meters = ts_meters
                metric = metric_func(y_pred=output_convert, y=target_convert, include_background=False if args.dataset == 'btcv' else True)
                metric = metric.cpu().numpy()
                compute_avg_metric(metric, log_meters, metric_name, batch_size, args)
                for k in range(metric.shape[-1]):
                    cls_metric = np.nanmean(metric, axis=0)[k]
                    if np.isnan(cls_metric) or np.isinf(cls_metric):
                        continue
                    log_meters[f'{idx2label[k]}.{metric_name}'].update(value=cls_metric, n=batch_size)
            # Aplicar evaluacion tipo PICAI
            if args.dataset == 'prostate':
                #perm_pred = torch.permute(output_convert, (0,1,4,2,3))[:, 1, ...]
                #perm_target = torch.permute(target_convert, (0,1,4,2,3))[:, 0, ...]
                perm_pred = output_convert[:, 1, ...]
                perm_target = target_convert[:, 0, ...]
                all_valid_preds += [perm_pred.detach().cpu().numpy()]
                all_valid_labels += [perm_target.detach().cpu().numpy()]
                print(perm_pred.shape, perm_target.shape, len(all_valid_preds), len(all_valid_labels))
            print(f'==> Evaluating on the {i+1}th batch is finished.')
        
        if args.dataset == 'prostate':
            #import pdb;pdb.set_trace()
            # Juntamos las preds y las mandamos a evaluate de picai
            try:
                final_det = np.concatenate([x for x in np.array(all_valid_preds)], axis=0)
                final_target = np.concatenate([x for x in np.array(all_valid_labels)], axis=0)
                if get_only_output:
                    return final_det, final_target
                valid_metrics = evaluate(y_det=iter(final_det),
                            y_true=iter(final_target),
                            y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])
                print('Metrics for fold_id : ', args.fold_id)
                print('data_path =',args.data_path)
                print('overviews_dir =',args.overviews_dir)
                print('conf_file =',args.conf_file)
                print('resume model =', args.resume)
                #print(valid_metrics)
                auroc = round(valid_metrics.auroc,3)
                ap = round(valid_metrics.AP,3)
                thr = self.args.metrics_threshold
                acc = valid_metrics.accuracy_at_thr(thr)
                precision, recall = valid_metrics.calculate_precision_recall_at_thr(thr)
                gmean = valid_metrics.gmean_at_thr(thr)
                print(f'AUROC: {auroc}')
                print(f'AP: {ap}')
                print(f'Accuracy at thr ({thr}): ', acc)
                print(f'Sensitivity (recall) at thr ({thr}): ', recall)
                print(f'G-mean at thr ({thr}): ', gmean)
                if args.save_roc_data:
                    roc_curve_data = valid_metrics.calculate_ROC()
                    roc_curve_data_json = {
                        "FPR": roc_curve_data["FPR"].tolist(),     # numpy array → list
                        "TPR": roc_curve_data["TPR"].tolist(),     # numpy array → list
                        "AUROC": float(roc_curve_data["AUROC"])    # np.float64 → float
                    }
                    save_path = args.metrics_dir
                    os.makedirs(save_path, exist_ok=True)
                    with open(f"{save_path}/roc_curve_data.json", "w") as f:
                        json.dump(roc_curve_data_json, f, indent=4)

            except:
                print('Valid Failed! Printing preds')
                print(final_det)
                print('Nan Index', np.argwhere(np.isnan(final_det)))
                self.save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'arch': args.arch,
                                'state_dict': self.model.state_dict(),
                                'optimizer' : self.optimizer.state_dict(),
                                'scaler': self.scaler.state_dict(), # additional line compared with base imple
                            }, 
                            is_best=False, 
                            filename=f'{args.ckpt_dir}/failed_eval_{epoch}.pth.tar'
                        )
                print("=> Finish saving failed model.")
                return
        # gather the stats from all processes
        if args.distributed:
            for k, v in meters.items():
                print(f'==> start synchronizing meter {k}...')
                v.synchronize_between_processes()
                print(f'==> finish synchronizing meter {k}...')
            if ts_meters is not None:
                for k, v in ts_meters.items():
                    print(f'==> start synchronizing meter {k}...')
                    v.synchronize_between_processes()
                    print(f'==> finish synchronizing meter {k}...')
        # pdb.set_trace()
        log_string = f"==> Epoch {epoch:04d} val results: \n"
        for k, v in meters.items():
            global_avg_metric = v.global_avg
            new_line = f"===> {k}: {global_avg_metric:.05f} \n"
            log_string += new_line
        print(log_string)
        if ts_meters is not None:
            log_string = f"==> Epoch {epoch:04d} test results: \n"
            for k, v in ts_meters.items():
                global_avg_metric = v.global_avg
                new_line = f"===> {k}: {global_avg_metric:.05f} \n"
                log_string += new_line
            print(log_string)

        if args.rank == 0 and not args.disable_wandb:
            wandb_log_dict = {}
            for k, v in meters.items():
                wandb_log_dict[k] = v.global_avg
            if args.dataset == 'prostate':
                wandb_log_dict['AUROC'] = valid_metrics.auroc
                wandb_log_dict['AP'] = valid_metrics.AP
            wandb.log(wandb_log_dict, step=niters)
        print("=> Finish Evaluating")

        if args.dataset == 'btcv':
            assert ts_meters is None
            return [meters['cls8_Dice'].global_avg]
        elif args.dataset == 'msd_brats':
            if ts_meters is None:
                return [meters['Dice'].global_avg]
            else:
                return [meters['Dice'].global_avg, ts_meters['Dice'].global_avg]
        # De momento devuelvo la misma metrica dice
        elif args.dataset == 'prostate':
            if ts_meters is None:
                return [meters['Dice'].global_avg, valid_metrics]
            else:
                return [meters['Dice'].global_avg, ts_meters['Dice'].global_avg, valid_metrics]

    def resume(self, resume = None):
        args = self.args
        if resume == None:
            resume = args.resume
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            if args.gpu is None:
                checkpoint = torch.load(resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler']) # additional line compared with base imple
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    def start_from(self, resume = None):
        args = self.args
        if resume == None:
            resume = args.start_from
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            if args.gpu is None:
                checkpoint = torch.load(resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(resume, map_location=loc)
            #args.start_epoch = checkpoint['epoch']
            self.lr_checkpoint = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if args.start_from_lr:
                for param_group in self.optimizer.param_groups:
                    if 'lr_scale' in param_group:
                        param_group['lr'] = args.start_from_lr * param_group['lr_scale']
                    else:
                        param_group['lr'] = args.start_from_lr
            self.scaler.load_state_dict(checkpoint['scaler']) # additional line compared with base imple
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = cur_lr * param_group['lr_scale']
            else:
                param_group['lr'] = cur_lr

    def save_best_model(self):
        """Esta funcion evaluará todos los checkpoints y guardara el mejor"""
        args = self.args
        # Compute iterations when resuming
        ckpt_dir = args.ckpt_dir
        best_metric = 0
        others_best = None
        best_metric_checkpoint = ''
        for checkpoint_path in os.listdir(ckpt_dir):
            if checkpoint_path != 'best_model.pth.tar':
                path = os.path.join(ckpt_dir, checkpoint_path)
                # Cargamos el checkpoint
                print('Cargando checkpoint:', path)
                self.resume(resume=path)
                niters = args.start_epoch * self.iters_per_epoch
                epoch = args.start_epoch
                metric_list = self.evaluate(epoch=epoch, niters=niters)
                print('Metric:', metric_list[1].score, metric_list[1].auroc, metric_list[1].AP)
                metric = metric_list[1].score if args.dataset == 'prostate' else metric_list[0]
                if len(metric_list) == 3:
                    ts_metric = metric_list[2]
                elif len(metric_list) == 2 and args.dataset != 'prostate':
                    ts_metric = metric_list[1]
                else:
                    ts_metric = None
                if metric > best_metric:
                    print(f"=> New val best metric: {metric} | Old val best metric: {best_metric}!")
                    best_metric = metric
                    others_best = metric_list[1]
                    best_metric_checkpoint = checkpoint_path
                    print(f'=> Best metric checkpoint: {best_metric_checkpoint}')
                    if ts_metric is not None:
                        print(f"=> New ts best metric: {ts_metric} | Old ts best metric: {best_ts_metric}!")
                        best_ts_metric = ts_metric
                    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                        self.save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'arch': args.arch,
                                'state_dict': self.model.state_dict(),
                                'optimizer' : self.optimizer.state_dict(),
                                'scaler': self.scaler.state_dict(), # additional line compared with base imple
                                'metric':metric
                            }, 
                            is_best=False, 
                            filename=f'{args.ckpt_dir}/best_model.pth.tar'
                        )
                        print("=> Finish saving best model.")
                else:
                    print(f"=> Still old val best metric: {best_metric}")
                    print('Others: ', others_best)
                    print(f'=> Best metric checkpoint: {best_metric_checkpoint}')
                    if ts_metric is not None:
                        print(f"=> Still old ts best metric: {best_ts_metric}")

    def evaluate_test(self):
        print('Comenzando Evaluacion...')
        args = self.args
        models_root_dir = args.models_root_dir
        models_to_ensemble = args.models_to_ensemble
        preds = []
        for model_dir in models_to_ensemble:
            model_path = os.path.join(models_root_dir, model_dir, 'ckpts', 'best_model.pth.tar')
            print(f'Cargando Checkpoint {model_path}')
            self.resume(resume=model_path)
            final_det, final_target = self.evaluate(get_only_output=True)
            preds.append(final_det)

        ensemble_output = np.mean(preds, axis=0).astype('float32')
        valid_metrics = evaluate(y_det=iter(ensemble_output),
                            y_true=iter(final_target),
                            y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])
        print(valid_metrics)
        #valid_metrics.save_full(args.metrics_save_path)

        


