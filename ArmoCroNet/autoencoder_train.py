import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import YooxDatasetHistogram
from model_encoder import ColorAutoencoder

from utils import save_checkpoint, load_checkpoint, save_some_examples
import config

from tqdm import tqdm


def train_fn(enc : ColorAutoencoder, loader : DataLoader, opt_gen : optim.Adam, l1_loss : nn.MSELoss):
    loop = tqdm(loader, leave=True)

    enc.train()

    i = 0
    for idx, (x, y) in enumerate(loop):
        i += 1
        if i > 10000:
            break

        x = x.to(config.DEVICE).float()
        y = y.to(config.DEVICE).float()

        for _ in range(1):
            # Train Discriminator
            y_fake = enc(x)
       
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = L1

            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()


        if idx % 17 == 0:
            loop.set_postfix(
                L1=L1.item(),
            )
    return L1


def main():
    enc = ColorAutoencoder(in_channels=3, levels=8).to(config.DEVICE)

    opt_gen = optim.Adam(enc.parameters(), lr=config.GEN_LEARNING_RATE, betas=(0.5, 0.999))
    
    L1_LOSS = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, enc, opt_gen, config.GEN_LEARNING_RATE)
    
    train_dataset = YooxDatasetHistogram()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        # num_workers=config.NUM_WORKERS,
    )
    val_dataset = YooxDatasetHistogram()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # for epoch in range(config.NUM_EPOCHS):
    for epoch in range(10):
        loss = train_fn(enc, train_loader, opt_gen, L1_LOSS)

        if config.SAVE_MODEL and epoch % 1 == 0:
            if loss != loss:
                load_checkpoint(config.CHECKPOINT_GEN, enc, opt_gen, config.GEN_LEARNING_RATE)
            else:
                if config.LOAD_MODEL == False:
                    print("#"*20)
                    risposta = input("Vuoi sovrascrivere perdendo il modello precedente? SI - NO:\n")
                    print("#"*20)
                    if risposta == "SI":
                        save_checkpoint(enc, opt_gen, filename=config.CHECKPOINT_GEN)
                        config.LOAD_MODEL = True
                else:
                    save_checkpoint(enc, opt_gen, filename=config.CHECKPOINT_GEN)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()