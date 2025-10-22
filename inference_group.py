import os
import glob
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist

from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Dataset

#modified for pickle
import re
#from dataset_pickle import TSPDataset, collate_fn, make_tgt_mask
#from dataset_inference_pickle import TSPDataset as TSPDataset_Val
#from dataset_inference_pickle import collate_fn as collate_fn_val

from dataset_inference import TSPDataset as TSPDataset_Val
from dataset_inference import collate_fn as collate_fn_val

from model import make_model, subsequent_mask
from loss import SimpleLossCompute, LabelSmoothing

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))



class TSPModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = make_model(
            src_sz=cfg.node_size, 
            enc_num_layers = cfg.enc_num_layers,
            dec_num_layers = cfg.dec_num_layers,
            d_model=cfg.d_model, 
            d_ff=cfg.d_ff, 
            h=cfg.h, 
            dropout=cfg.dropout,
            encoder_pe = cfg.encoder_pe,
            decoder_pe = cfg.decoder_pe,
            decoder_lut = cfg.decoder_lut,
            feat_dim=cfg.feat_dim,

        )
        self.automatic_optimization = False
        
        criterion = LabelSmoothing(size=cfg.node_size, smoothing=cfg.smoothing)
        
        self.loss_compute = SimpleLossCompute(self.model.generator, criterion, cfg.node_size)
        
        self.set_cfg(cfg)
        self.train_outputs = []
        
        self.val_optimal_tour_distances = []
        self.val_predicted_tour_distances = []
        
        # 메모리 효율적인 변수로 교체
        self.test_optimal_sum = 0.0
        self.test_predicted_sum = 0.0
        self.test_sample_count = 0

        self.optgap = 0.0
        
    def set_cfg(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg)  # save config file with pytorch lightening

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr*len(self.cfg.gpus), betas=self.cfg.betas, eps=self.cfg.eps)
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(step, model_size=self.cfg.d_model, factor=self.cfg.factor, warmup=self.cfg.warmup),
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def train_dataloader(self):
        train_dataset = TSPDataset(self.cfg.train_data_path, self.cfg.fuzzy_depth)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size = self.cfg.train_batch_size, 
            shuffle = True, 
            collate_fn = collate_fn,
            pin_memory=True
        )
        return train_dataloader

    def val_dataloader(self):
        self.val_dataset = TSPDataset_Val(self.cfg.val_data_path, self.cfg.fuzzy_depth)
        val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size = self.cfg.val_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn_val,
            pin_memory=True
        )
        return val_dataloader
    
    def training_step(self, batch):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        tgt_y = batch["tgt_y"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]

        opt = self.optimizers() # manual backprop
        opt.zero_grad() # manual backprop
        
        self.model.train()
        out = self.model(src, tgt, tgt_mask) # [B, V, E]
        
        if self.cfg.comparison_matrix == "memory":
            comparison_matrix = self.model.memory
        elif self.cfg.comparison_matrix == "encoder_lut":
            comparison_matrix = self.model.encoder_lut
        elif self.cfg.comparison_matrix == "decoder_lut":
            comparison_matrix = self.model.decoder_lut
        else:
            assert False
        
        loss = self.loss_compute(out, tgt_y, visited_mask, ntokens, comparison_matrix) # check! 
        

        training_step_outputs = [l.item() for l in loss]
        self.train_outputs.extend(training_step_outputs)

        loss = loss.mean()
        self.manual_backward(loss) # manual backprop
        
        opt.step() # manual backprop

        if self.trainer.is_global_zero:
            self.log(
                name = "train_loss",
                value = loss,
                prog_bar = True,
            )
        
        return {"loss": loss}

    def on_train_epoch_start(self) -> None:
        if self.trainer.is_global_zero:
            self.train_start_time = time.time()
    
    def on_train_epoch_end(self):
        outputs = self.all_gather(sum(self.train_outputs))
        lengths = self.all_gather(len(self.train_outputs))
        self.train_outputs.clear()
        
        lr_scheduler = self.lr_schedulers() # manual backprop
        lr_scheduler.step() # manual backprop
        
        if self.trainer.is_global_zero:
            train_loss = outputs.sum() / lengths.sum()
            train_time = time.time() - self.train_start_time
            self.print(
                f"##############Train: Epoch {self.current_epoch}###################",
                "train_loss={:.03f}, ".format(train_loss),
                "train time={:.03f}".format(train_time),
                f"##################################################################\n",
            )
            
    def validation_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        tsp_tours = batch["tsp_tours"]
        #for fuzzy
        src_fuzzy = batch["src_fuzzy"]

        
        batch_size = tsp_tours.shape[0]
        self.model.eval()
        with torch.no_grad():
            memory = self.model.encode(src, src_fuzzy)
            ys = tgt.clone()
            visited_mask = visited_mask.clone()
            for i in range(self.cfg.node_size - 1):
                # memory, tgt, tgt_mask
                tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool).to(src.device)
                out = self.model.decode(memory, src_fuzzy , ys, tgt_mask)
                
                if self.cfg.comparison_matrix == "memory":
                    comparison_matrix = self.model.memory
                elif self.cfg.comparison_matrix == "encoder_lut":
                    comparison_matrix = self.model.encoder_lut
                elif self.cfg.comparison_matrix == "decoder_lut":
                    comparison_matrix = self.model.decoder_lut
                else:
                    assert False
                    
                prob = self.model.generator(out[:, -1].unsqueeze(1), visited_mask, comparison_matrix)
                
                _, next_word = torch.max(prob, dim=-1)
                next_word = next_word.squeeze(-1)
                
                visited_mask[torch.arange(batch_size), 0, next_word] = True
                
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        
        
        optimal_tour_distance = self.get_tour_distance(src, tsp_tours)
        predicted_tour_distance = self.get_tour_distance(src, ys)
        
        result = {
            "optimal_tour_distance": optimal_tour_distance.tolist(),
            "predicted_tour_distance": predicted_tour_distance.tolist(),
            }    
        
        self.val_optimal_tour_distances.extend(result["optimal_tour_distance"])
        self.val_predicted_tour_distances.extend(result["predicted_tour_distance"])
        
        return result
    
    def get_tour_distance(self, graph, tour):
        # graph.shape = [B, N, 2]
        # tour.shape = [B, N]

        shp = graph.shape
        gathering_index = tour.unsqueeze(-1).expand(*shp)
        ordered_seq = graph.gather(dim = 1, index = gathering_index)
        rolled_seq = ordered_seq.roll(dims = 1, shifts = -1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(-1).sqrt() # [B, N]
        group_travel_distances = segment_lengths.sum(-1)
        return group_travel_distances

    def on_validation_epoch_start(self) -> None:
        if self.trainer.is_global_zero:
            self.validation_start_time = time.time()

    def on_validation_epoch_end(self):
        optimal_tour_distances = self.all_gather(sum(self.val_optimal_tour_distances))
        predicted_tour_distances = self.all_gather(sum(self.val_predicted_tour_distances))
        
        self.val_optimal_tour_distances.clear()
        self.val_predicted_tour_distances.clear()
        
        total = self.cfg.node_size * len(self.val_dataset)
        opt_gaps = (predicted_tour_distances - optimal_tour_distances) / optimal_tour_distances
        mean_opt_gap = opt_gaps.mean().item() * 100
       

        self.log(
            name = "opt_gap",
            value = mean_opt_gap,
            prog_bar = True,
            sync_dist=True
        )
        
        if self.trainer.is_global_zero:
            validation_time = time.time() - self.validation_start_time
            self.print(
                f"##############Validation: Epoch {self.current_epoch}##############",
                "validation time={:.03f}".format(validation_time),
                f"\ntotal={total}",
                f"\nmean_opt_gap = {mean_opt_gap}  %",
                f"##################################################################\n",
            )
            
    def test_dataloader(self):
        self.test_dataset = TSPDataset_Val(self.cfg.val_data_path, self.cfg.fuzzy_depth)
        test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size = self.cfg.test_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn_val,
            pin_memory=False,  # Train 코드와 동일하게 False로 변경
            num_workers = 0,  # Train 코드와 동일하게 0으로 변경
            persistent_workers = False,
        )
        return test_dataloader
    
    def points2adj(self, points):
            """
            return distance matrix
            Args:
            points: b, n, 2
            Returns: b, n, n
            """
            assert points.dim() == 3
            return torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1) ** 0.5
      

    def test_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        tsp_tours = batch["tsp_tours"]
        
        #for fuzzy
        src_fuzzy = batch["src_fuzzy"]

        
        batch_size = tsp_tours.shape[0]
        node_size = tsp_tours.shape[1]
        src_original = src.clone()
        fuzzy_emb_size = src_fuzzy.shape[-1]

        G = self.cfg.G
        
        # Train 코드와 동일한 방식으로 G개씩 나누어 처리
        optimal_tour_distance = self.get_tour_distance(src_original, tsp_tours)
        best_predicted_distances = torch.full((batch_size,), float('inf'), device=src.device)
        best_tours = torch.zeros_like(tsp_tours)
        
        # G개를 한번에 처리하지 않고 배치로 나누어 처리
        g_batch_size = min(16, G)  # 한 번에 처리할 G 샘플 수
        
        for g_start in range(0, G, g_batch_size):
            g_end = min(g_start + g_batch_size, G)
            current_g = g_end - g_start
            
            with torch.no_grad():
                # 현재 배치만큼만 복제
                src_g = src.unsqueeze(1).repeat(1, current_g, 1, 1).reshape(batch_size * current_g, node_size, 2)
                src_fuzzy_g = src_fuzzy.unsqueeze(1).repeat(1, current_g, 1, 1).reshape(batch_size * current_g, node_size, fuzzy_emb_size)
                tgt_g = torch.arange(g_start, g_end).to(src.device).unsqueeze(0).repeat(batch_size, 1).reshape(batch_size * current_g, 1)
                
                visited_mask_g = torch.zeros(batch_size, current_g, 1, node_size, dtype=torch.bool, device=src.device)
                visited_mask_g[:, torch.arange(current_g), :, g_start + torch.arange(current_g)] = True
                visited_mask_g = visited_mask_g.reshape(batch_size * current_g, 1, node_size)
                
                # 인코딩 및 디코딩
                memory = self.model.encode(src_g, src_fuzzy_g)
                ys = tgt_g.clone()
                
                for i in range(self.cfg.node_size - 1):
                    tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool).to(src.device)
                    out = self.model.decode(memory, src_fuzzy_g, ys, tgt_mask, i)
                    
                    if self.cfg.comparison_matrix == "memory":
                        comparison_matrix = self.model.memory
                    elif self.cfg.comparison_matrix == "encoder_lut":
                        comparison_matrix = self.model.encoder_lut
                    elif self.cfg.comparison_matrix == "decoder_lut":
                        comparison_matrix = self.model.decoder_lut
                    
                    prob = self.model.generator(out[:, -1].unsqueeze(1), visited_mask_g, comparison_matrix)
                    _, next_word = torch.max(prob, dim=-1)
                    next_word = next_word.squeeze(-1)
                    
                    visited_mask_g[torch.arange(batch_size * current_g), 0, next_word] = True
                    ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
                    
                    # 중간 텐서 정리
                    del out, prob
                
                # 거리 계산
                predicted_distances = self.get_tour_distance(src_g, ys).reshape(batch_size, current_g)
                
                # 각 배치에서 최선의 투어 업데이트
                for b in range(batch_size):
                    min_idx = predicted_distances[b].argmin()
                    if predicted_distances[b, min_idx] < best_predicted_distances[b]:
                        best_predicted_distances[b] = predicted_distances[b, min_idx]
                        best_tours[b] = ys.reshape(batch_size, current_g, node_size)[b, min_idx]
                
                # 메모리 정리
                #del src_g, src_fuzzy_g, tgt_g, visited_mask_g, memory, ys, predicted_distances
                #torch.cuda.empty_cache()
        
        # 최종 결과 계산
        predicted_tour_distance = self.get_tour_distance(src_original, best_tours)
        
        # 즉시 합계에 추가
        self.test_optimal_sum += optimal_tour_distance.sum().item()
        self.test_predicted_sum += predicted_tour_distance.sum().item()
        self.test_sample_count += batch_size
        
        return {}
    
    def on_test_epoch_end(self):
        # 로컬 합계
        local_sums = torch.tensor([
            self.test_optimal_sum,
            self.test_predicted_sum,
            float(self.test_sample_count)
        ]).cuda()
        
        # All-gather (Train 코드와 동일)
        if dist.is_initialized() and self.trainer.world_size > 1:
            gathered = [torch.zeros_like(local_sums) for _ in range(self.trainer.world_size)]
            dist.all_gather(gathered, local_sums)
            total_optimal = sum(g[0].item() for g in gathered)
            total_predicted = sum(g[1].item() for g in gathered)
            total_count = sum(g[2].item() for g in gathered)
        else:
            total_optimal = local_sums[0].item()
            total_predicted = local_sums[1].item()
            total_count = local_sums[2].item()
        
        if self.trainer.is_global_zero and total_count > 0:
            mean_opt_gap = ((total_predicted - total_optimal) / total_optimal) * 100
            self.optgap = mean_opt_gap
            self.print(f"\nmean_opt_gap = {mean_opt_gap}  %")
            

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--ckpt", default="/root/FuzzyFormer/logs/lightning_logs/version_599/checkpoints/TSP20-epoch=00-opt_gap=115.3660.ckpt", help="Path to ckpt.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    cfg = OmegaConf.load(args.config)
    pl.seed_everything(cfg.seed)
    
    tsp_model = TSPModel.load_from_checkpoint(cfg.resume_checkpoint, strict = False) #, map_location=torch.device('cuda:0'))
    #tsp_model = TSPModel.load_from_checkpoint(args.ckpt, strict = False)
    match = re.search(r'version_(\w+)', cfg.resume_checkpoint)


    tsp_model.set_cfg(cfg)
    
    # build trainer
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=cfg.gpus,
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.max_epochs,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    
    )
    
    print("\n Gpu  ", cfg.gpus)

    print("\n data path ", cfg.val_data_path)
    print("\n G ", cfg.G)
    s2 = time.time()
    trainer.test(tsp_model)
    e2 = time.time()
    #elapsed_time2 = e2 - s2
    #print("\n elapsed _time ", elapsed_time2)
    print(f"\n optgap : {tsp_model.optgap} \t time : {e2 - s2}")

