"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import weights_init, compute_acc
from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10
from folder import ImageFolder


class ACGAN_Adv:
    def __init__(self, params):
        self.params = params
        # specify the gpu id if using only 1 gpu
        if params.ngup == 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)
        try:
            os.makedirs(params.outf)
        except OSError:
            pass
        if params.manualSeed is None:
            params.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", params.manualSeed)
        random.seed(params.manualSeed)
        torch.manual_seed(params.manualSeed)
        if params.cuda:
            torch.cuda.manual_seed_all(params.manualSeed)

        cudnn.benchmark = True

        if torch.cuda.is_available() and not params.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda")

        # datase t
        if params.dataset == 'imagenet':
            # folder dataset
            self.dataset = ImageFolder(
                root=params.dataroot,
                transform=transforms.Compose([
                    transforms.Scale(params.imageSize),
                    transforms.CenterCrop(params.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                classes_idx=(10, 20)
            )
        elif params.dataset == 'cifar10':
            self.dataset = dset.CIFAR10(
                root=params.dataroot, download=True,
                transform=transforms.Compose([
                    transforms.Scale(params.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        else:
            raise NotImplementedError(
                "No such dataset {}".format(params.dataset))

        assert self.dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=params.batchSize,
                                                      shuffle=True, num_workers=int(params.workers))

        # some hyper parameters
        self.ngpu = int(params.ngpu)
        self.nz = int(params.nz)
        self.ngf = int(params.ngf)
        self.ndf = int(params.ndf)
        self.num_classes = int(params.num_classes)
        self.nc = 3

        # Define the generator and initialize the weights
        if params.dataset == 'imagenet':
            self.netG = _netG(ngpu, nz)
        else:
            self.netG = _netG_CIFAR10(ngpu, nz)
        self.netG.apply(weights_init)
        if params.netG != '':
            self.netG.load_state_dict(torch.load(params.netG))
        print(self.netG)

        # loss functions
        self.dis_criterion = nn.BCELoss()
        self.aux_criterion = nn.NLLLoss()

        # tensor placeholders
        self.input = torch.FloatTensor(
            params.batchSize, 3, params.imageSize, params.imageSize)
        self.noise = torch.FloatTensor(params.batchSize, nz, 1, 1)
        self.eval_noise = torch.FloatTensor(
            params.batchSize, nz, 1, 1).normal_(0, 1)
        self.dis_label = torch.FloatTensor(params.batchSize)
        self.aux_label = torch.LongTensor(params.batchSize)
        self.real_label = 1
        self.fake_label = 0

        # if using cuda
        if params.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.dis_criterion.cuda()
            self.aux_criterion.cuda()
            self.input, self.dis_label, self.aux_label = self.input.cuda(
            ), self.dis_label.cuda(), self.aux_label.cuda()
            self.noise, self.eval_noise = self.noise.cuda(), self.eval_noise.cuda()

        # define variables
        self.input = Variable(self.input)
        self.noise = Variable(self.noise)
        self.eval_noise = Variable(self.eval_noise)
        self.dis_label = Variable(self.dis_label)
        self.aux_label = Variable(self.aux_label)
        # noise for evaluation
        self.eval_noise_ = np.random.normal(0, 1, (params.batchSize, self.nz))
        self.eval_label = np.random.randint(
            0, self.num_classes, params.batchSize)
        self.eval_onehot = np.zeros((params.batchSize, self.num_classes))
        self.eval_onehot[np.arange(params.batchSize), self.eval_label] = 1
        self.eval_noise_[np.arange(params.batchSize),
                         :self.num_classes] = self.eval_onehot[np.arange(params.batchSize)]
        self.eval_noise_ = (torch.from_numpy(self.eval_noise_))
        self.eval_noise.data.copy_(
            self.eval_noise_.view(params.batchSize, nz, 1, 1))
        # setup optimizer
        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=params.lr, betas=(params.beta1, 0.999))

    def train_batch(self, images, labels):
        # optimize D
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        for i in range(1):
            # train with real
            self.netD.zero_grad()
            batch_size = images.size(0)
            if params.cuda:
                images = images.cuda()
                labels = labels.cuda()
            self.input.data.resize_as_(images).copy_(images)
            self.dis_label.data.resize_(batch_size).fill_(self.real_label)
            self.aux_label.data.resize_(batch_size).copy_(labels)
            dis_output, aux_output = self.netD(self.input)

            dis_errD_real = self.dis_criterion(dis_output, self.dis_label)
            aux_errD_real = self.aux_criterion(aux_output, self.aux_label)
            errD_real = dis_errD_real + aux_errD_real
            errD_real.backward()
            D_x = dis_output.data.mean()

            # compute the current classification accuracy
            accuracy = compute_acc(aux_output, aux_label)

            # train with fake
            self.noise.data.resize_(batch_size, self.nz, 1, 1).normal_(0, 1)
            labels = np.random.randint(0, self.num_classes, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, self.nz))
            class_onehot = np.zeros((batch_size, self.num_classes))
            class_onehot[np.arange(batch_size), labels] = 1
            noise_[np.arange(batch_size),
                   :self.num_classes] = class_onehot[np.arange(batch_size)]
            noise_ = (torch.from_numpy(noise_))
            self.noise.data.copy_(noise_.view(batch_size, self.nz, 1, 1))
            self.aux_label.data.resize_(batch_size).copy_(
                torch.from_numpy(labels))

            fake = self.netG(noise)
            self.dis_label.data.fill_(self.fake_label)
            dis_output, aux_output = self.netD(fake.detach())

            dis_errD_fake = self.dis_criterion(dis_output, self.dis_label)
            aux_errD_fake = self.aux_criterion(aux_output, self.aux_label)
            errD_fake = dis_errD_fake + aux_errD_fake
            errD_fake.backward()
            D_G_z1 = dis_output.data.mean()
            errD = errD_real + errD_fake
            self.optimizerD.step()
        # optimize G
         ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        for i in range(1):
            self.netG.zero_grad()
            # fake labels are real for generator cost
            self.dis_label.data.fill_(real_label)
            dis_output, aux_output = self.netD(fake)
            dis_errG = dis_criterion(dis_output, dis_label)
            aux_errG = aux_criterion(aux_output, aux_label)
            errG = dis_errG + aux_errG
            errG.backward()
            D_G_z2 = dis_output.data.mean()
            #todo 
            #dav_loss and pert_loss
            C = 0.1
            loss_perturb = torch.mean(torch.norm(fake.view(fake.shape[0],-1),2,dim=1))
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            self.optimizerG.step()
        return errD_real, errD_fake, errD, errG, D_x, D_G_z1, D_G_z2,accuracy

    def train(self, params):

        avg_loss_D = 0.0
        avg_loss_G = 0.0
        avg_loss_A = 0.0
        adv_loss_sum = 0.0
        for epoch in range(params.epochs):
            for i, data in enumerate(self.dataloader, 0):
                images, labels = data
                # loss_G_batch, loss_D_batch, loss_A_batch, loss_adv_batch, loss_perturb_bath,accuracy = self.train_batch(images, labels)
                errD_real, errD_fake, errD, errG,D_x, D_G_z1, D_G_z2, accuracy = train_batch(images,labels)
                print(epoch)
                # compute the average loss
                curr_iter = epoch * len(self.dataloader) + i
                all_loss_G = avg_loss_G * curr_iter
                all_loss_D = avg_loss_D * curr_iter
                all_loss_A = avg_loss_A * curr_iter
                all_loss_G += errG.data.item()
                all_loss_D += errD.data.item()
                all_loss_A += accuracy
                avg_loss_G = all_loss_G / (curr_iter + 1)
                avg_loss_D = all_loss_D / (curr_iter + 1)
                avg_loss_A = all_loss_A / (curr_iter + 1)
                print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
                  % (epoch, params.epochs, i, len(self.dataloader),
                     errD.data.item(), avg_loss_D, errG.data.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
                if i % 100 == 0:
                    vutils.save_image(
                        images, '%s/real_samples.png' % params.outf)
                    print('Label for eval = {}'.format(self.eval_label))
                    fake = self.netG(self.eval_noise)
                    vutils.save_image(
                        fake.data,
                        '%s/fake_samples_epoch_%03d.png' % (params.outf, epoch)
                    )
            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' %
                    (params.outf, epoch))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' %
                    (params.outf, epoch))

        

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int,
                        default=1, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128,
                        help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=110,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--netG', default='',
                        help="path to self.netG (to continue training)")
    parser.add_argument('--netD', default='',
                        help="path to self.netD (to continue training)")
    parser.add_argument('--outf', default='.',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes for AC-GAN')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='The ID of the specified GPU')

    params = parser.parse_args()
    print(params)
    ACGAN_Adv = ACGAN_Adv(params)
    ACGAN_Adv.train(params)
