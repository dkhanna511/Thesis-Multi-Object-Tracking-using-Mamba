import torch
import torch.nn as nn


class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, initial_lr, warmup_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.warmup_lr = warmup_lr
        self.current_step = 0

    def step(self):
        # Linear warmup
        if self.current_step < self.warmup_steps:
            lr = self.warmup_lr + (self.initial_lr - self.warmup_lr) * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.current_step += 1

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    

# model = modenum_blocksl.float()
# criterion = nn.SmoothL1Loss()  # Mean squared error loss
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas = (0.9, 0.98), )

    
# from torch.optim.lr_scheduler import StepLR

# # Define the warmup scheduler
# warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=4000, initial_lr=0.001, warmup_lr=1e-6)

# # Define a standard scheduler to use after warmup
# scheduler_after_warmup = StepLR(optimizer, step_size=30, gamma=0.1)





class CustomWarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        """
        Custom learning rate scheduler with warmup steps.

        Args:
        - optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        - d_model (int): The token dimension (input token size, or model dimension size).
        - warmup_steps (int): The number of warmup steps.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """Update the learning rate based on the current step."""
        self.current_step += 1
        lr = self.compute_lr()

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_lr(self):
        """Compute the learning rate using the formula."""
        scale = self.d_model ** -0.5
        step_factor = min(self.current_step ** -0.5, self.current_step * (self.warmup_steps ** -1.5))
        return scale * step_factor



# Example usage
# d_model = embedding_dim  # Token dimension size (input token size)
# warmup_steps = 4000

# model = torch.nn.Linear(d_model, d_model)  # Example model
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-8)

# scheduler = CustomWarmupScheduler(optimizer, d_model, warmup_steps)
