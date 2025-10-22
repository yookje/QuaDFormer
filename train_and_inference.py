import os
import glob
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy

from torch.utils.data import DataLoader, Dataset


from dataset import TSPDataset, collate_fn, make_tgt_mask
from dataset_inference import TSPDataset as TSPDataset_Val
from dataset_inference import collate_fn as collate_fn_val

from model import make_model, subsequent_mask
from loss import SimpleLossCompute, LabelSmoothing

#import torch.profiler
import torch.distributed as dist
import gc
import psutil
import traceback

# 메모리 모니터링 콜백
class MemoryMonitorCallback(Callback):
    def __init__(self, threshold_gb=490, check_interval=100):
        self.threshold = threshold_gb * 1024**3
        self.check_interval = check_interval
        self.process = psutil.Process(os.getpid())
        self.step_count = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1
        if self.step_count % self.check_interval == 0:
            self._check_memory(trainer, "train", batch_idx)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.check_interval == 0:
            self._check_memory(trainer, "validation", batch_idx)
            
    def on_validation_epoch_end(self, trainer, pl_module):
        # Validation 종료 시 강제 메모리 정리
        self._force_cleanup()
        mem_info = self.process.memory_info()
        print(f"[Memory] After validation cleanup: {mem_info.rss/(1024**3):.1f}GB")
    
    def _check_memory(self, trainer, phase, batch_idx):
        mem_info = self.process.memory_info()
        rss_gb = mem_info.rss / (1024**3)
        
        if trainer.is_global_zero and batch_idx % 500 == 0:
            print(f"[Memory] {phase} batch {batch_idx}: {rss_gb:.1f}GB")
        
        if mem_info.rss > self.threshold:
            print(f"⚠️ WARNING: {phase} - Memory {rss_gb:.1f}GB exceeds threshold!")
            self._force_cleanup()
            
            # 재확인
            mem_info_after = self.process.memory_info()
            if mem_info_after.rss > self.threshold:
                print(f"❌ Memory still high: {mem_info_after.rss/(1024**3):.1f}GB. Stopping.")
                trainer.should_stop = True
    
    def _force_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass


def rate(step, model_size, factor, warmup):
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
        
        # 메모리 효율적인 변수들로 교체
        self.val_optimal_sum = 0.0
        self.val_predicted_sum = 0.0
        self.val_sample_count = 0
        
        self.test_optimal_sum = 0.0
        self.test_predicted_sum = 0.0
        self.test_sample_count = 0
        
        self.c = 0

    def set_cfg(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg)
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr*len(self.cfg.gpus), betas=self.cfg.betas, eps=self.cfg.eps)
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(step, model_size=self.cfg.d_model, factor=self.cfg.factor, warmup=self.cfg.warmup),
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    """
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg.lr*len(self.cfg.gpus), 
            betas=self.cfg.betas, 
            eps=self.cfg.eps
        )

        #modified for tsp1000     
        #fixed
        #lr_scheduler = LambdaLR(
        #optimizer=optimizer,
        #lr_lambda=lambda step: min(step / self.cfg.warmup, 1.0) if step > 0 else 0.001
        
        
        
        #완만한 optimizer
        def custom_rate(step): # 1부터 0.003
            if step < self.cfg.warmup:
                # warmup phase
                return step / self.cfg.warmup
            else:
                # 목표: step 160000 (epoch 100)에서 0.056
                # 현재: step 400에서 1.0 시작
                # 지수 감소: decay_rate^(step-400) = 0.056
                decay_per_step = (0.056) ** (1.0 / (160000 - 400))
                return decay_per_step ** (step - 400)
        
        
        def custom_rate(step): #0.2에서 0.093
            if step < self.cfg.warmup:
                return step / self.cfg.warmup
            else:
                # warmup 직후 0.2 시작
                # 0.5 * 2 * rate = 0.2이므로 rate = 0.2
                start_rate = 0.2
                end_rate = 0.0056
                
                start_step = self.cfg.warmup
                end_step = 320000  # 100 epochs * 3200 steps
                
                if step >= end_step:
                    return end_rate
                
                progress = (step - start_step) / (end_step - start_step)
                current_rate = start_rate - (start_rate - end_rate) * progress
                return current_rate
        
        
        def custom_rate(step): # 1~0.093

            if step < self.cfg.warmup:
                return step / self.cfg.warmup  # 0에서 1.0까지
            else:
                # warmup 직후 1.0 시작
                start_rate = 1.0
                end_rate = 0.093
                
                start_step = self.cfg.warmup
                end_step = 320000  # 100 epochs * 3200 steps
                
                if step >= end_step:
                    return end_rate
                
                progress = (step - start_step) / (end_step - start_step)
                current_rate = start_rate - (start_rate - end_rate) * progress
                return current_rate
        
        lr_scheduler = LambdaLR(optimizer, lr_lambda=custom_rate)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    """

    def train_dataloader(self):
        train_dataset = TSPDataset(self.cfg.train_data_path, self.cfg.fuzzy_depth)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size = self.cfg.train_batch_size, 
            shuffle = True, 
            collate_fn = collate_fn,
            pin_memory = False,  # False로 변경
            num_workers = 0,  # 0으로 설정
            persistent_workers = False,
        )
        return train_dataloader
    
    def val_dataloader(self):
        self.val_dataset = TSPDataset_Val(self.cfg.val_data_path, self.cfg.fuzzy_depth)
        val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size = self.cfg.val_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn_val,
            pin_memory = False,  # False로 변경
            num_workers = 0,  # 0으로 설정
            persistent_workers = False,
        )
        return val_dataloader

    def training_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        tgt_y = batch["tgt_y"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        src_fuzzy = batch["src_fuzzy"]

        opt = self.optimizers()
        opt.zero_grad()
        
        self.model.train()
        out = self.model(src, src_fuzzy, tgt, tgt_mask)
        
        if self.cfg.comparison_matrix == "memory":
            comparison_matrix = self.model.memory
        elif self.cfg.comparison_matrix == "encoder_lut":
            comparison_matrix = self.model.encoder_lut
        elif self.cfg.comparison_matrix == "decoder_lut":
            comparison_matrix = self.model.decoder_lut
        else:
            assert False
            
        loss = self.loss_compute(out, tgt_y, visited_mask, ntokens, comparison_matrix)
        
        # 즉시 스칼라로 변환
        training_step_outputs = [l.item() for l in loss]
        self.train_outputs.extend(training_step_outputs)

        loss = loss.mean()
        self.manual_backward(loss, retain_graph=True)
        
        # Gradient clipping
        #if (cfg.grad_clip == True) : #(cfg.node_size >=1000) :
        #    self.clip_gradients(opt, gradient_clip_val=0.25, gradient_clip_algorithm='value')
        
        opt.step()

        # 메모리 정리 (매 500 스텝마다)
        if batch_idx % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

      
        if self.trainer.is_global_zero:
            self.log(
                name = "train_loss",
                value = loss,
                prog_bar = True,
            )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        tsp_tours = batch["tsp_tours"]
        src_fuzzy = batch["src_fuzzy"]
        
        batch_size = tsp_tours.shape[0]
        decoding_step = self.cfg.node_size if self.cfg.use_start_token else self.cfg.node_size - 1
        
        self.model.eval()
        
        with torch.no_grad():
            memory = self.model.encode(src, src_fuzzy)
            ys = tgt.clone()
            visited_mask = visited_mask.clone()
            visited_mask = visited_mask.unsqueeze(1)

            for i in range(decoding_step):
                tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool).to(src.device).requires_grad_(False)
                out = self.model.decode(memory, src_fuzzy, ys, tgt_mask, i)
                
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
                
                # 중간 텐서 정리
                del out, prob
        
        optimal_tour_distance = self.get_tour_distance(src, tsp_tours)
        predicted_tour_distance = self.get_tour_distance(src, ys)
        
        # 즉시 합계에 추가 (리스트 사용 안함)
        self.val_optimal_sum += optimal_tour_distance.sum().item()
        self.val_predicted_sum += predicted_tour_distance.sum().item()
        self.val_sample_count += batch_size
        
        # 메모리 정리
        del memory, ys, visited_mask, optimal_tour_distance, predicted_tour_distance
        
        return {}

    def on_validation_epoch_start(self):
        self.val_optimal_sum = 0.0
        self.val_predicted_sum = 0.0
        self.val_sample_count = 0
        
        if self.trainer.is_global_zero:
            self.validation_start_time = time.time()

    def on_validation_epoch_end(self):
        # 각 프로세스의 합계
        local_sums = torch.tensor([
            self.val_optimal_sum,
            self.val_predicted_sum,
            float(self.val_sample_count)
        ]).cuda()
        
        # All-reduce
        if dist.is_initialized() and self.trainer.world_size > 1:
            dist.all_reduce(local_sums, op=dist.ReduceOp.SUM)
        
        total_optimal = local_sums[0].item()
        total_predicted = local_sums[1].item()
        total_count = int(local_sums[2].item())
        
        if total_count > 0:
            mean_opt_gap = ((total_predicted - total_optimal) / total_optimal) * 100
        else:
            mean_opt_gap = 0.0
        
        self.log(
            name = "opt_gap",
            value = mean_opt_gap,
            prog_bar = True,
            sync_dist = False  # 이미 수동으로 동기화함
        )
        
        if self.trainer.is_global_zero:
            validation_time = time.time() - self.validation_start_time
            total = self.cfg.node_size * len(self.val_dataset)
            self.print(
                f"##############Validation: Epoch {self.current_epoch}##############",
                "validation time={:.03f}".format(validation_time),
                f"\ntotal={total}",
                f"\nmean_opt_gap = {mean_opt_gap}  %",
                f"##################################################################\n",
            )
        
        # 강제 메모리 정리
        self._force_memory_cleanup()

    def test_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        tsp_tours = batch["tsp_tours"]
        src_fuzzy = batch["src_fuzzy"]
        
        batch_size = tsp_tours.shape[0]
        node_size = tsp_tours.shape[1]
        src_original = src.clone()
        fuzzy_emb_size = src_fuzzy.shape[-1]

        self.G = self.cfg.G
        
        # G개씩 나누어 처리하여 메모리 절약
        optimal_tour_distance = self.get_tour_distance(src_original, tsp_tours)
        best_predicted_distances = torch.full((batch_size,), float('inf'), device=src.device)
        best_tours = torch.zeros_like(tsp_tours)
        
        # G개를 한번에 처리하지 않고 배치로 나누어 처리
        g_batch_size = min(16, self.G)  # 한 번에 처리할 G 샘플 수
        
        for g_start in range(0, self.G, g_batch_size):
            g_end = min(g_start + g_batch_size, self.G)
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
                del src_g, src_fuzzy_g, tgt_g, visited_mask_g, memory, ys, predicted_distances
                torch.cuda.empty_cache()
        
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
        
        # All-gather
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

    def get_tour_distance(self, graph, tour):
        shp = graph.shape
        gathering_index = tour.unsqueeze(-1).expand(*shp)
        ordered_seq = graph.gather(dim = 1, index = gathering_index)
        rolled_seq = ordered_seq.roll(dims = 1, shifts = -1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(-1).sqrt()
        group_travel_distances = segment_lengths.sum(-1)
        return group_travel_distances

    def _force_memory_cleanup(self):
        """강제 메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass

    # 나머지 메서드들은 동일...
    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            self.train_start_time = time.time()
            

    def on_train_epoch_end(self):
        outputs = self.all_gather(sum(self.train_outputs))
        lengths = self.all_gather(len(self.train_outputs))
        self.train_outputs.clear()
        
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        
        if self.trainer.is_global_zero:
            train_loss = outputs.sum() / lengths.sum()
            train_time = time.time() - self.train_start_time
            self.print(
                f"##############Train: Epoch {self.current_epoch}###################",
                "train_loss={:.03f}, ".format(train_loss),
                "train time={:.03f}".format(train_time),
                f"##################################################################\n",
            )
            
          
    def test_dataloader(self):
        self.test_dataset = TSPDataset_Val(self.cfg.val_data_path, self.cfg.fuzzy_depth)
        test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size = self.cfg.test_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn_val,
            pin_memory = False,
            num_workers = 0,
            persistent_workers = False,
        )
        return test_dataloader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_arguments()
    cfg = OmegaConf.load(args.config)

    pl.seed_everything(cfg.seed)
    
    # 모델 로드
    if cfg.resume_training == True:
        tsp_model = TSPModel.load_from_checkpoint(cfg.resume_checkpoint, strict = False)
    else:
        tsp_model = TSPModel(cfg)

    # 체크포인트 콜백 - top3 저장 (전체 상태 포함)
    checkpoint_callback = ModelCheckpoint(
        monitor = "opt_gap",
        filename = f'TSP{cfg.node_size}-' + "{epoch:02d}-{opt_gap:.4f}",
        save_top_k = 3,  # top 3개 유지
        mode = "min",
        save_weights_only = False,  # 전체 상태 저장 (optimizer, lr_scheduler 포함)
        save_on_train_epoch_end = False,  # validation 후에만 저장
    )

    # 메모리 모니터 콜백 추가 (490GB 임계값)
    memory_monitor = MemoryMonitorCallback(threshold_gb=490)

    # Logger
    tb_logger = TensorBoardLogger("logs")

    """
    # DDP 전략 - find_unused_parameters 제거
    ddp_strategy = DDPStrategy(
        find_unused_parameters=False,  # True에서 False로
        static_graph=True,
        gradient_as_bucket_view=True,
    )
    """
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=cfg.gpus,
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.max_epochs,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
        logger=[tb_logger],
        callbacks=[checkpoint_callback, memory_monitor],
        strategy="ddp_find_unused_parameters_true",
        #strategy=ddp_strategy,  # 변경된 전략
        
        accumulate_grad_batches=1,  # 필요시 증가
        enable_model_summary=True,
        enable_progress_bar=True,
        detect_anomaly=False,  # 프로덕션에서는 False
    )
    
    if trainer.is_global_zero:
        items = trainer.progress_bar_callback.get_metrics(trainer, tsp_model)
  
    # 학습
    try:
        s = time.time()
        trainer.fit(tsp_model)
        e = time.time()
        elapsed_time = e - s
        
        # 테스트
        best_model_dir = os.path.join(trainer.default_root_dir, checkpoint_callback.best_model_path)
        tsp_model = TSPModel.load_from_checkpoint(best_model_dir, strict = False)
        s2 = time.time()
        trainer.test(tsp_model)
        e2 = time.time()
        elapsed_time2 = e2 - s2
        
        if trainer.is_global_zero:
            model_dir = os.path.join(trainer.default_root_dir, checkpoint_callback.dirpath)
            
            metrics = []
            trainer_files = glob.glob(os.path.join(model_dir, "*"))
            for file in trainer_files:
                metrics.append([float(str(file).split("=")[-1].split(".ckpt")[0]), file])
            metrics.sort()
            
         
            
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        traceback.print_exc()
       
