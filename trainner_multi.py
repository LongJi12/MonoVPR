import logging
import os, math
import time
import subprocess

import torch
from torch import nn
from tqdm import tqdm
from accelerate import Accelerator
from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import EventStorage
import detectron2.utils.comm as comm
from detectron2.engine import default_writers
from detectron2.data.samplers import TrainingSampler
from detectron2.solver import build_lr_scheduler, build_optimizer
from fvcore.common.timer import Timer

from torch.utils.data.distributed import DistributedSampler

from data_loader import apollo_mapper, apollo_3dpose_loader
from tester import save_results, test_model
from pytorch3d.io import load_obj
import torch.optim as optim
import numpy as np

import cv2

p3d_weights = {
    'loss_keypoint': 0.1,
    'loss_rotate': 1, 'loss_trans': 0.5, 'loss_mesh': 3,
    'loss_R': 0.1, 'loss_T': 0.01, 'loss_RT': 0.01
}


def train_model(cfg, model,eval=False):
    from accelerate import DistributedDataParallelKwargs

    # 修改 Accelerator 的初始化代码
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerate = Accelerator(kwargs_handlers=[ddp_kwargs])
    model.train()

    # 新增：初始化最小损失跟踪器
    best_loss = float('inf')      # 初始化为无穷大
    best_iter = 0                # 对应最佳损失的迭代次数
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Load optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR)

    # Load logger and checkpointer
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer)
    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    # writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []c
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if accelerate.is_main_process else []

    # Check if there is a checkpoint to resume from
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, 'model_0014999.pth')
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        # Load checkpoint if it exists
        checkpointer.load(checkpoint_path)
        # Retrieve the starting iteration
        start_iter =15000
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Load train data
    dataset = apollo_3dpose_loader(cfg.DATASETS.TRAIN[0])
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=2)
    # train_sampler = (
    # DistributedSampler(dataset, shuffle=True) 
    # if num_gpus > 1 
    # else None
    # )
    from detectron2.data import build_detection_train_loader
    from torch.utils.data.distributed import DistributedSampler


    # data_loader = build_detection_train_loader(
    #     dataset,
    #     mapper=apollo_mapper(cfg.DATASETS.RESIZE),
    #     sampler=DistributedSampler(
    #         dataset,
    #         num_replicas=accelerate.num_processes,
    #         rank=accelerate.process_index,
    #         shuffle=True
    #     ),
    #     total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #     aspect_ratio_grouping=True,
    #     num_workers=cfg.DATALOADER.NUM_WORKERS,
    # )

    data_loader = build_detection_train_loader(
        dataset,
        mapper=apollo_mapper(cfg.DATASETS.RESIZE),
        
        sampler=TrainingSampler(len(dataset),shuffle=True, seed=cfg.SEED),
        total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    print('seed',cfg.SEED)
    model, optimizer, data_loader = accelerate.prepare(model, optimizer, data_loader)
    # set rescaling factor. BAAM uses them at 3d pose estimation.
    # model.roi_heads.rx = cfg.DATASETS.RESIZE[1] / dataset[0]['width']
    # model.roi_heads.ry = cfg.DATASETS.RESIZE[0] / dataset[0]['height']    
    model.module.roi_heads.rx = cfg.DATASETS.RESIZE[1] / dataset[0]['width']
    model.module.roi_heads.ry = cfg.DATASETS.RESIZE[0] / dataset[0]['height']



    # Train setting
    # print("-------训练开始-----")
    # print(f"----datatsets数量:{len(dataset)}----")
    # print(f"----cfg.SOLVER.IMS_PER_BATCH数量:{cfg.SOLVER.IMS_PER_BATCH}----")
    epoch_iter = math.ceil((len(dataset) / cfg.SOLVER.IMS_PER_BATCH))   # 向上取整
    print("{} iters per epoch".format(epoch_iter))

    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
       
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            # pre-time check
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()    

            ###ddp
            # current_epoch = iteration // epoch_iter
            # if current_epoch != last_epoch:
            #     data_loader.sampler.set_epoch(current_epoch)
            #     last_epoch = current_epoch


            if iteration == cfg.SOLVER.STEPS[0]:
                for g in optimizer.param_groups: g['lr'] *= 0.1
            # if iteration == cfg.SOLVER.STEPS[1]:
            #     for g in optimizer.param_groups: g['lr'] *= 0.1
            # calculate losses
            loss_dict = model(data)

            for k in loss_dict.keys():
                if k in p3d_weights.keys():
                    # loss_dict[k] *= p3d_weights[k]
                    loss_dict[k] = loss_dict[k] * p3d_weights[k]

            losses = sum(loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict
            # loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            accelerate.gather(loss_dict)  # 同步所有进程的损失
            loss_dict_reduced = {
                    k: accelerate.gather(v).mean().item()  # 显式聚合
                    for k, v in loss_dict.items()
                }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            
            # if comm.is_main_process():
            #     storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            if accelerate.is_main_process:
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
             ### 新增：保存最低损失权重 ====
            if comm.is_main_process():
                current_loss = losses_reduced
                # 只在主进程保存
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_iter = iteration
                    # 保存完整模型状态字典
                    checkpointer.save(f"best_model_{iteration}.pth")
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, f'best_model_iter_{best_iter}_loss_{best_loss:.4f}.pth')
                    )
                    print(f"[Epoch {iteration//epoch_iter}] Saved best model at iter {best_iter} with loss {best_loss:.4f}")
            # backward
            optimizer.zero_grad()
            # losses.backward()
            accelerate.backward(losses)
            optimizer.step()

             #Save model checkpoint every 10000 iterations for the first 20000
            if iteration <= 20000 and iteration % 10000 == 0:
                # 每1000次迭代保存一次完整状态
                checkpointer.save(f"model_{iteration}.pth")
                # # 同时保存一个只包含模型参数的轻量版本
                # torch.save(
                #     model.state_dict(),
                #     os.path.join(cfg.OUTPUT_DIR, f"model_light_{iteration}.pth")
                # )
                
            if iteration > 20000 and iteration<=26000 and iteration % 100 == 0:
                
                checkpointer.save(f"model_{iteration}.pth")
                # torch.save(
                #     model.state_dict(),
                #     os.path.join(cfg.OUTPUT_DIR, f"model_light_{iteration}.pth")
                # )
            if iteration > 26000 and iteration % 500 == 0:
                checkpointer.save(f"model_{iteration}.pth")
                torch.cuda.empty_cache()

            # Output loss and other stats to storage
            storage.put_scalars(total_loss=losses.item())

            # recode learning rate
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            # check post-time
            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()


            if iteration - start_iter > 5 and (
                    (iteration + 1) % 10 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

