import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import save_checkpoint, load_checkpoint, save_some_examples
import config

from dataset import YooxDatasetEdge
from model_generator import Generator
from model_discriminator import Discriminator

from tqdm import tqdm


def train_fn(disc : Discriminator, gen : Generator, loader : DataLoader, opt_disc : optim.Adam, opt_gen  : optim.Adam, l1_loss : nn.MSELoss, bce : nn.BCEWithLogitsLoss):
    loop = tqdm(loader, leave=True)

    disc.train()
    gen.train()
    i = 0
    for idx, (x, y) in enumerate(loop):
        i += 1
        if i > 10000:
            break

        x = x.to(config.DEVICE).float()
        y = y.to(config.DEVICE).float()

        h = None
        for b in y:
            h0 = torch.histc(b[0], 256)
            h1 = torch.histc(b[1], 256)
            h2 = torch.histc(b[2], 256)
            h_3c = torch.stack([h0,h1,h2])
            if h is None:
                h = h_3c[None,:]
            else:
                h = torch.concat([h, h_3c[None,:]], 0)
        h = h.to(config.DEVICE).float()

        for _ in range(1):
            # Train Discriminator
            y_fake = gen(x[:,None,:,:], h)
            D_fake = disc(y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_real = disc(y.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_loss = D_real_loss + D_fake_loss

            opt_disc.zero_grad()
            D_loss.backward()
            opt_disc.step()

            # Train Generator
            D_fake = disc(y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA / (10 if i % 2 == 1 else 1)
            #L1 = hist_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()


        if idx % 17 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                G_fake_loss=G_fake_loss.item(),
                L1=L1.item(),
            )
    return G_loss + D_loss


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=1,levels=6).to(config.DEVICE)
    #gen = Generator(2,1).to(config.DEVICE)

    opt_disc = optim.Adam(disc.parameters(), lr=config.DISC_LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.GEN_LEARNING_RATE, betas=(0.5, 0.999))
    
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.DISC_LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.GEN_LEARNING_RATE)
    
    train_dataset = YooxDatasetEdge()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        # num_workers=config.NUM_WORKERS,
    )
    val_dataset = YooxDatasetEdge()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # for epoch in range(config.NUM_EPOCHS):
    for epoch in range(10):
        loss = train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE)

        if config.SAVE_MODEL and epoch % 1 == 0:
            if loss != loss:
                load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.DISC_LEARNING_RATE)
                load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.GEN_LEARNING_RATE)
            else:
                if config.LOAD_MODEL == False:
                    print("#"*20)
                    risposta = input("Vuoi sovrascrivere perdendo il modello precedente? SI - NO:\n")
                    print("#"*20)
                    if risposta == "SI":
                        save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
                        save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                        config.LOAD_MODEL = True
                else:
                    save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
                    save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)

        save_some_examples(gen, val_loader)
    #save_some_examples(gen, val_loader)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()