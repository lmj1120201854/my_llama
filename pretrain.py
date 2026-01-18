# import os
# import platform
# import argparse
# import time
# import warnings
# import math
# import pandas as pd
# import torch
# from torch import optim
# from torch.utils.data import DataLoader
# from contextlib import nullcontext

# from transformers import AutoTokenizer

# from models import Transformer
# from config import ModelConfig
# from dataset import PretrainDataset

# import wandb

# warnings.filterwarnings('ignore')

# def Logger(content):
#     print(content)
#     if args.use_wandb and wandb.run is not None:
#         wandb.log({"logs/message": content})

# def get_lr(it, all_it):
#     warmup_iters = args.warmup_iters
#     lr_decay_iters = all_it
#     min_lr = args.learning_rate / 10

#     if it < warmup_iters:
#         return args.learning_rate * it / warmup_iters

#     if it > 

import os
import platform
import argparse
import time
import warnings
import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from transformers import AutoTokenizer

from models import Transformer
from config import ModelConfig
from dataset import PretrainDataset

import wandb

warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    # 输出目录
    out_dir: str = "base_model_215M"
    
    # 训练参数
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 2e-4
    accumulation_steps: int = 8
    grad_clip: float = 1.0
    warmup_iters: int = 0
    
    # 设备和精度
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    
    # 数据
    data_path: str = "data/seq_monkey_datawhale.jsonl"
    num_workers: int = 8
    max_seq_len: int = 512
    
    # 日志和保存
    log_interval: int = 100
    save_interval: int = 1000
    milestone_interval: int = 20000
    
    # WandB 配置
    use_wandb: bool = False
    wandb_project: str = "My-Llama"
    wandb_run_name: str = "Pretrain"
    wandb_watch_model: bool = False
    wandb_watch_freq: int = 1000
    
    # 多 GPU
    gpus: str = "0,1"
    
    # Tokenizer 路径
    tokenizer_path: str = "./bpe_tokenizer/"
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps
    
    @property
    def save_dir(self) -> str:
        return self.out_dir


class PretrainTrainer:
    def __init__(self, config: TrainingConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        
        self._setup_gpu_environment()
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scaler = None
        self.train_loader = None
        self.ctx = None
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_start_time = None
    
        os.makedirs(self.config.save_dir, exist_ok=True)
    
    def _setup_gpu_environment(self):
        if self.config.gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpus
            if torch.cuda.is_available():
                self.config.device = "cuda:0"
            else:
                self.config.device = "cpu"
        
        self.device_type = "cuda" if "cuda" in self.config.device else "cpu"
    
    def setup(self):
        self.log("Setting up training components...")
        
        torch.manual_seed(42)
        
        self.ctx = (
            nullcontext() 
            if self.device_type == "cpu" 
            else torch.cuda.amp.autocast()
        )
        
        # 初始化模型和分词器
        self._init_model()
        
        # 初始化数据加载器
        self._init_dataloader()
        
        # 初始化优化器和 scaler
        self._init_optimizer()
        
        # 初始化 WandB
        if self.config.use_wandb:
            self._init_wandb()
        
        self.log("Setup completed!")
        self._log_training_info()
    
    def _init_model(self):
        self.log("Initializing model and tokenizer...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        
        # 创建模型
        self.model = Transformer(self.model_config)
        
        # 多 GPU 设置
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.log(f"Using {num_gpus} GPUs with DataParallel!")
            self.model = torch.nn.DataParallel(self.model)
        
        # 移动到设备
        self.model = self.model.to(self.config.device)
        
        # 记录参数量
        param_count = self._count_parameters()
        self.log(f"Model parameters: {param_count / 1e6:.3f}M")
    
    def _init_dataloader(self):
        self.log("Initializing dataloader...")
        
        train_ds = PretrainDataset(
            self.config.data_path,
            self.tokenizer,
            max_length=self.model_config.max_seq_len
        )
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        self.iter_per_epoch = len(self.train_loader)
        self.total_steps = self.config.epochs * self.iter_per_epoch
        
        self.log(f"Dataset size: {len(train_ds)}")
        self.log(f"Steps per epoch: {self.iter_per_epoch}")
    
    def _init_optimizer(self):
        self.log("Initializing optimizer...")
        
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.config.dtype in ['float16', 'bfloat16'])
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def _init_wandb(self):
        self.log("Initializing WandB...")
        
        # 系统信息
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [
                torch.cuda.get_device_name(i) 
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
        
        # 初始化
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config={
                # 训练配置
                **self.config.__dict__,
                # 模型配置
                "model_dim": self.model_config.dim,
                "model_n_layers": self.model_config.n_layers,
                "model_max_seq_len": self.model_config.max_seq_len,
                # 系统信息
                **system_info,
            },
            tags=["pretraining", f"dim-{self.model_config.dim}", f"layers-{self.model_config.n_layers}"],
        )
        
        # 定义指标
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("gradients/*", step_metric="train/global_step")
        wandb.define_metric("performance/*", step_metric="train/global_step")
        wandb.define_metric("system/*", step_metric="train/global_step")
        
        # 监控模型
        if self.config.wandb_watch_model:
            wandb.watch(
                self.model,
                log="all",
                log_freq=self.config.wandb_watch_freq,
                log_graph=True
            )
    
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _get_lr(self, step: int) -> float:
        warmup_iters = self.config.warmup_iters
        lr_decay_iters = self.total_steps
        min_lr = self.config.learning_rate / 10

        if step < warmup_iters:
            return self.config.learning_rate * step / warmup_iters
        
        if step > lr_decay_iters:
            return min_lr
        
        decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (self.config.learning_rate - min_lr)
    
    def _get_gpu_memory_info(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        
        memory_info = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            
            memory_info[f"system/gpu_{i}_allocated_gb"] = allocated
            memory_info[f"system/gpu_{i}_reserved_gb"] = reserved
            memory_info[f"system/gpu_{i}_utilization_pct"] = allocated / total * 100
        
        return memory_info
    
    def _compute_gradient_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def _get_model_state_dict(self) -> Dict[str, Any]:
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()
    
    def save_checkpoint(
        self,
        path: str,
        loss: float,
        is_milestone: bool = False
    ):
        checkpoint = {
            'model_state_dict': self._get_model_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'loss': loss,
            'config': {
                'dim': self.model_config.dim,
                'n_layers': self.model_config.n_layers,
            }
        }
        
        torch.save(checkpoint, path)
        self.log(f"Checkpoint saved to {path}")
        
        # # 记录到 WandB
        # if self.config.use_wandb:
        #     artifact_name = f'model-{"milestone" if is_milestone else "checkpoint"}-{self.global_step}'
        #     artifact = wandb.Artifact(
        #         name=artifact_name,
        #         type='model',
        #         description=f'Checkpoint at step {self.global_step}, loss {loss:.4f}',
        #         metadata={
        #             'epoch': self.current_epoch,
        #             'global_step': self.global_step,
        #             'loss': loss,
        #         }
        #     )
        #     artifact.add_file(path)
        #     wandb.log_artifact(artifact)
    
    def load_checkpoint(self, path: str):
        self.log(f"Loading checkpoint from {path}")
        
        checkpoint = torch.load(path, map_location=self.config.device)
        
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        self.log(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def log(self, message: str):
        print(message)
        if self.config.use_wandb and wandb.run is not None:
            wandb.log({"logs/message": message})
    
    def _log_training_info(self):
        self.log("=" * 50)
        self.log("Training Configuration:")
        self.log(f"  - Epochs: {self.config.epochs}")
        self.log(f"  - Batch size: {self.config.batch_size}")
        self.log(f"  - Effective batch size: {self.config.effective_batch_size}")
        self.log(f"  - Learning rate: {self.config.learning_rate}")
        self.log(f"  - Total steps: {self.total_steps}")
        self.log(f"  - Device: {self.config.device}")
        self.log("=" * 50)
    
    def _train_step(self, batch) -> Dict[str, float]:
        X, Y, loss_mask = batch
        X = X.to(self.config.device)
        Y = Y.to(self.config.device)
        loss_mask = loss_mask.to(self.config.device)
        
        # 更新学习率
        lr = self._get_lr(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # 前向传播
        with self.ctx:
            out = self.model(X, Y)
            loss = out.last_loss / self.config.accumulation_steps
            loss_mask_flat = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()
        
        # 反向传播
        self.scaler.scale(loss).backward()
        
        metrics = {
            'loss': loss.item() * self.config.accumulation_steps,
            'lr': lr,
        }
        
        return metrics
    
    def _optimizer_step(self) -> Dict[str, float]:
        self.scaler.unscale_(self.optimizer)
        
        # 计算梯度范数
        grad_norm_before = self._compute_gradient_norm()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip
        )
        
        grad_norm_after = self._compute_gradient_norm()
        
        # 优化器步骤
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        
        return {
            'grad_norm_before_clip': grad_norm_before,
            'grad_norm_after_clip': grad_norm_after,
            'was_clipped': grad_norm_before > self.config.grad_clip,
        }
    
    def _log_metrics(
        self,
        step: int,
        metrics: Dict[str, Any],
        epoch_start_time: float
    ):
        elapsed = time.time() - epoch_start_time
        samples_per_sec = (step + 1) * self.config.batch_size / elapsed if elapsed > 0 else 0
        tokens_per_sec = samples_per_sec * self.model_config.max_seq_len
        
        # 预估剩余时间
        remaining_steps = self.iter_per_epoch - step
        eta_minutes = remaining_steps * (elapsed / (step + 1)) / 60 if step > 0 else 0
        
        # 打印日志
        log_msg = (
            f"Epoch:[{self.current_epoch + 1}/{self.config.epochs}]"
            f"({step}/{self.iter_per_epoch}) "
            f"loss:{metrics['loss']:.4f} "
            f"lr:{metrics['lr']:.2e} "
            f"speed:{samples_per_sec:.1f}samples/s "
            f"ETA:{eta_minutes:.1f}min"
        )
        print(log_msg)
        
        # 记录到 WandB
        if self.config.use_wandb:
            wandb_metrics = {
                "train/loss": metrics['loss'],
                "train/learning_rate": metrics['lr'],
                "train/epoch": self.current_epoch + 1,
                "train/step": step,
                "train/global_step": self.global_step,
                "train/epoch_progress": step / self.iter_per_epoch,
                "performance/samples_per_second": samples_per_sec,
                "performance/tokens_per_second": tokens_per_sec,
                "performance/eta_minutes": eta_minutes,
                "amp/scale": self.scaler.get_scale(),
            }
            
            # 添加梯度指标
            if 'grad_norm_before_clip' in metrics:
                wandb_metrics.update({
                    "gradients/norm_before_clip": metrics['grad_norm_before_clip'],
                    "gradients/norm_after_clip": metrics['grad_norm_after_clip'],
                    "gradients/was_clipped": metrics['was_clipped'],
                })
            
            # 添加 GPU 内存信息
            wandb_metrics.update(self._get_gpu_memory_info())
            
            wandb.log(wandb_metrics, step=self.global_step)
    
    def train_epoch(self):
        self.model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(self.train_loader):
            # 训练步骤
            metrics = self._train_step(batch)
            running_loss += metrics['loss']
            num_batches += 1
            
            # 梯度累积后更新
            if (step + 1) % self.config.accumulation_steps == 0:
                grad_metrics = self._optimizer_step()
                metrics.update(grad_metrics)
            
            # 日志记录
            if step % self.config.log_interval == 0:
                metrics['avg_loss'] = running_loss / num_batches
                self._log_metrics(step, metrics, epoch_start_time)
            
            # 保存检查点
            if (step + 1) % self.config.save_interval == 0:
                self.model.eval()
                ckp_path = (
                    f"{self.config.save_dir}/pretrain_"
                    f"{self.model_config.dim}_{self.model_config.n_layers}.pth"
                )
                self.save_checkpoint(ckp_path, metrics['loss'])
                self.model.train()
            
            # 里程碑检查点
            if (step + 1) % self.config.milestone_interval == 0:
                self.model.eval()
                ckp_path = (
                    f"{self.config.save_dir}/pretrain_"
                    f"{self.model_config.dim}_{self.model_config.n_layers}_"
                    f"step{self.global_step}.pth"
                )
                self.save_checkpoint(ckp_path, metrics['loss'], is_milestone=True)
                self.model.train()
            
            self.global_step += 1
        
        # Epoch 结束统计
        epoch_time = time.time() - epoch_start_time
        epoch_avg_loss = running_loss / num_batches
        
        self.log(f"Epoch {self.current_epoch + 1} completed in {epoch_time / 60:.1f} min, avg_loss: {epoch_avg_loss:.4f}")
        
        if self.config.use_wandb:
            wandb.log({
                "epoch/number": self.current_epoch + 1,
                "epoch/avg_loss": epoch_avg_loss,
                "epoch/time_minutes": epoch_time / 60,
            })
        
        return epoch_avg_loss
    
    def train(self, resume_from: Optional[str] = None):
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        self.training_start_time = time.time()
        training_status = "completed"
        
        self.log("Starting training...")
        
        if self.config.use_wandb:
            wandb.log({
                "train/total_steps": self.total_steps,
                "train/iter_per_epoch": self.iter_per_epoch,
            })
        
        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch
                self.train_epoch()
        
        except KeyboardInterrupt:
            self.log("Training interrupted by user")
            training_status = "interrupted"
        
        except Exception as e:
            self.log(f"Training failed with error: {e}")
            training_status = "failed"
            if self.config.use_wandb:
                wandb.log({"error": str(e)})
            raise
        
        finally:
            self._finalize_training(training_status)
    
    def _finalize_training(self, status: str):
        total_time = time.time() - self.training_start_time
        
        # 保存最终模型
        final_path = (
            f"{self.config.save_dir}/pretrain_"
            f"{self.model_config.dim}_{self.model_config.n_layers}_final.pth"
        )
        torch.save(self._get_model_state_dict(), final_path)
        
        self.log(f"Training {status}!")
        self.log(f"Total time: {total_time / 3600:.2f} hours")
        self.log(f"Final model saved to: {final_path}")
        
        if self.config.use_wandb:
            wandb.run.summary["training_status"] = status
            wandb.run.summary["total_time_hours"] = total_time / 3600
            
            # 保存最终模型为 artifact
            artifact = wandb.Artifact(
                name='final-model',
                type='model',
                description=f'Final model ({status})',
                metadata={'status': status, 'time_hours': total_time / 3600}
            )
            artifact.add_file(final_path)
            wandb.log_artifact(artifact)
            
            wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    parser.add_argument("--out_dir", type=str, default="base_model_215M")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./data/seq_monkey_datawhale.jsonl")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Happy-LLM")
    parser.add_argument("--wandb_run_name", type=str, default="Pretrain-215M")
    parser.add_argument("--wandb_watch_model", action="store_true")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # 创建配置
    training_config = TrainingConfig(
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        accumulation_steps=args.accumulation_steps,
        grad_clip=args.grad_clip,
        warmup_iters=args.warmup_iters,
        data_path=args.data_path,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_watch_model=args.wandb_watch_model,
        gpus=args.gpus,
    )
    
    model_config = ModelConfig(
        dim=1024,
        n_layers=18,
    )
    
    # 创建 Trainer 并训练
    trainer = PretrainTrainer(training_config, model_config)
    trainer.setup()
    trainer.train(resume_from=args.resume)