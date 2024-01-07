from dataclasses import dataclass

from model import UNet, ModelConfig
from ddpm import DDPM, DDPMConfig
from utils import load_config

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


@dataclass
class TrainConfig:
    name: str
    model_config: ModelConfig
    ddpm_config: DDPMConfig
    epochs: int
    batch_size: int
    learning_rate: float


def train(model, dataset, config: TrainConfig):
    writer = SummaryWriter(f'runs/{config.name}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    ddpm = DDPM(model, config.ddpm_config, device).to(device)
    ddpm.train()
    
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
    )

    for epoch in range(config.epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, labels) in progress_bar:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = ddpm(data, labels)
            loss.backward()
            optimizer.step()

            if batch_idx == 0:
                running_loss = loss
            else:
                running_loss = 0.99 * running_loss + 0.01 * loss

            progress_bar.set_description(f'Epoch {epoch} Loss: {running_loss:.4f}')

            if batch_idx % 10 == 0:
                iter = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', loss, iter)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iter)

        scheduler.step()

    return ddpm


def main():
    try:
        train_config = load_config('config.json', TrainConfig)
    except TypeError as e:
        raise ValueError(f"Invalid configuration: {e}")

    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    model = UNet(train_config.model_config, num_classes=10)

    train(model, dataset, train_config)

    torch.save(model.state_dict(), f'models/{train_config.name}.pt')

    print("Done.")

if __name__ == "__main__":
    main()