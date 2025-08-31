#!/usr/bin/env python3

import torch
import wandb
from lightning import LightningModule
from torchmetrics.text import CharErrorRate as CER

class TestWandBLogging(LightningModule):
    def __init__(self):
        super().__init__()
        self.cer = CER()
        self.step_count = 0
        
    def training_step(self, batch, batch_idx):
        # Simulate some training data
        batch_size = 4
        images = torch.randn(batch_size, 3, 68, 1800)
        labels = torch.randint(0, 100, (batch_size, 10))
        
        # Simulate predictions
        preds = torch.randint(0, 100, (batch_size, 10))
        
        # Calculate CER
        total_cer = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            # Simulate CER calculation
            cer = torch.rand(1).item() * 0.5  # Random CER between 0 and 0.5
            total_cer += cer
            valid_samples += 1
        
        avg_cer = total_cer / valid_samples if valid_samples > 0 else 0.0
        
        # Log using PyTorch Lightning
        self.log("train/cer", avg_cer, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log using direct WandB
        if hasattr(self, 'loggers') and self.loggers:
            for logger in self.loggers:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log'):
                    logger.experiment.log({"train/cer_direct": avg_cer}, step=self.step_count)
                    print(f'Direct WandB log: train/cer_direct = {avg_cer:.4f} at step {self.step_count}')
        
        # Also try direct wandb.log
        try:
            wandb.log({"train/cer_wandb": avg_cer}, step=self.step_count)
            print(f'WandB wandb.log: train/cer_wandb = {avg_cer:.4f} at step {self.step_count}')
        except Exception as e:
            print(f"Direct wandb.log failed: {e}")
        
        self.step_count += 1
        
        # Return dummy loss
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

def test_wandb_logging():
    """Test WandB logging functionality"""
    print("Testing WandB logging...")
    
    # Initialize WandB
    wandb.init(
        project="test-cer-logging",
        name="test-run",
        config={
            "learning_rate": 0.001,
            "batch_size": 4,
            "epochs": 1
        }
    )
    
    # Create test module
    model = TestWandBLogging()
    
    # Simulate a few training steps
    for step in range(5):
        loss = model.training_step(None, step)
        print(f"Step {step}: Loss = {loss.item()}")
    
    wandb.finish()
    print("WandB logging test completed!")

if __name__ == "__main__":
    test_wandb_logging()
