import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from modules.networks import *
from modules.datasets import *

def validate(generator, discriminator, data):
    is_first = True

    avg_loss_gen = 0.0
    avg_loss_disc = 0.0

    with torch.no_grad():
        generator.eval()
        discriminator.eval()

        for input_image, real_image in data:
            input_image = input_image.to(torch.float32).to(device) / 255
            real_image = real_image.to(torch.float32).to(device) / 255

            disc_out_real = discriminator(torch.cat((input_image, real_image), dim=1))
            gen_out = generator(input_image)
            disc_out_fake = discriminator(torch.cat((input_image, gen_out), dim=1))

            # Выведем результаты по самому первому изображению из датасета
            if is_first:
                plt.subplot(2, 3, 1)
                plt.axis('off')
                plt.imshow(np.moveaxis(input_image[0].detach().cpu().numpy(), 0, 2))
                plt.subplot(2, 3, 2)
                plt.axis('off')
                plt.imshow(np.moveaxis(real_image[0].detach().cpu().numpy(), 0, 2))
                plt.subplot(2, 3, 5)
                plt.axis('off')
                plt.imshow(torch.sigmoid(disc_out_real[0, 0]).detach().cpu().numpy(), cmap='gray')
                plt.subplot(2, 3, 3)
                plt.axis('off')
                plt.imshow(np.moveaxis(gen_out[0].detach().cpu().numpy(), 0, 2))
                plt.subplot(2, 3, 6)
                plt.axis('off')
                plt.imshow(torch.sigmoid(disc_out_fake[0, 0]).detach().cpu().numpy(), cmap='gray')
                plt.show()
                is_first = False

            disc_loss = discriminator_loss(disc_out_real, disc_out_fake)
            gen_loss_total, _, _ = generator_loss(disc_out_fake, gen_out, real_image)

            avg_loss_disc += disc_loss.item()
            avg_loss_gen += gen_loss_total.item()

        avg_loss_disc /= len(data)
        avg_loss_gen /= len(data)

    return avg_loss_disc, avg_loss_gen

def train_step_generator(generator, discriminator, optim_gen, data):
    avg_loss_gen = 0.0

    generator.train()
    discriminator.eval()

    for input_image, real_image in data:
        input_image = input_image.to(torch.float32).to(device) / 255
        real_image = real_image.to(torch.float32).to(device) / 255

        gen_out = generator(input_image)
        disc_out_fake = discriminator(torch.cat((input_image, gen_out), dim=1))
        optim_gen.zero_grad()
        gen_loss_total, _, _ = generator_loss(disc_out_fake, gen_out, real_image)
        avg_loss_gen += gen_loss_total.item()
        gen_loss_total.backward()
        optim_gen.step()

    avg_loss_gen /= len(data)
    return avg_loss_gen

def train_step_discriminator(generator, discriminator, optim_disc, data):
    avg_loss_disc = 0.0

    generator.eval()
    discriminator.train()

    for input_image, real_image in data:
        input_image = input_image.to(torch.float32).to(device) / 255
        real_image = real_image.to(torch.float32).to(device) / 255

        gen_out = generator(input_image)
        disc_out_real = discriminator(torch.cat((input_image, real_image), dim=1))
        disc_out_fake = discriminator(torch.cat((input_image, gen_out), dim=1))

        optim_disc.zero_grad()
        disc_loss = discriminator_loss(disc_out_real, disc_out_fake)
        avg_loss_disc += disc_loss.item()
        disc_loss.backward()
        optim_disc.step()

    avg_loss_disc /= len(data)
    return avg_loss_disc

batch_size_facades = 10

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', '-t', action='store_true')
parser.add_argument('--epoch', '-e', type=int, default=300)
parser.add_argument('--dataset', '-d', default='facades')
args = parser.parse_args()

dataset_facades_train = FacadesDataset('src/dataset/%s' % args.dataset, 'train')
dataset_facades_test  = FacadesDataset('src/dataset/%s' % args.dataset, 'test')
dataset_facades_val   = FacadesDataset('src/dataset/%s' % args.dataset, 'val')

from torch.utils.data import DataLoader

dataloader_facades_train = DataLoader(dataset_facades_train, batch_size=batch_size_facades)
dataloader_facades_test  = DataLoader(dataset_facades_test,  batch_size=batch_size_facades)
dataloader_facades_val   = DataLoader(dataset_facades_val,   batch_size=batch_size_facades)

from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import pandas as pd

if args.train:
    facades_generator = MyGenerator().to(device)
    facades_discriminator = MyDiscriminator().to(device)
    facades_generator_optim = torch.optim.Adam(facades_generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    facades_discriminator_optim = torch.optim.Adam(facades_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    history = []
    for i in trange(args.epoch):
        if (i + 1) % 10 == 0:
            torch.save(facades_generator.cpu(), 'saved/%s_generator_%03d.pth' % (args.dataset, i + 1))
            torch.save(facades_discriminator.cpu(), 'saved/%s_discriminator_%03d.pth' % (args.dataset, i + 1))
            facades_generator.to(device)
            facades_discriminator.to(device)

        for j in range(5):
            disc_loss = train_step_discriminator(facades_generator, facades_discriminator,
                                                facades_discriminator_optim, dataloader_facades_train)
        gen_loss = train_step_generator(facades_generator, facades_discriminator,
                                        facades_generator_optim, dataloader_facades_train)
        val_disc_loss, val_gen_loss = validate(facades_generator, facades_discriminator, dataloader_facades_val)
        history.append((disc_loss, gen_loss, val_disc_loss, val_gen_loss))
    pd.DataFrame(history).to_csv('history.csv')
else:
    facades_generator = torch.load('src/saved/%s_generator_%03d.pth' % (args.dataset, args.epoch)).to(device)
    facades_discriminator = torch.load('src/saved/%s_discriminator_%03d.pth' % (args.dataset, args.epoch)).to(device)

    with torch.no_grad():
        facades_generator.eval()

        for input_image, real_image in tqdm(dataloader_facades_val):
            input_image = input_image.to(torch.float32).to(device) / 255
            real_image = real_image.to(torch.float32).to(device) / 255
            gen_out = facades_generator(input_image)

            for i in range(input_image.shape[0]):
                plt.subplot(1, 3, 1)
                plt.axis('off')
                plt.imshow(np.moveaxis(input_image[i].detach().cpu().numpy(), 0, 2))
                plt.subplot(1, 3, 2)
                plt.axis('off')
                plt.imshow(np.moveaxis(real_image[i].detach().cpu().numpy(), 0, 2))
                plt.subplot(1, 3, 3)
                plt.axis('off')
                plt.imshow(np.moveaxis(gen_out[i].detach().cpu().numpy(), 0, 2))
                plt.show()
