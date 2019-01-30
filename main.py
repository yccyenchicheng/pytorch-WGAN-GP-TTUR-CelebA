from __future__ import print_function
import os
import sys
import random
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
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

from models import model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='celeba | mnist')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--seed', type=int, default=1, help='Set seed for reproducible experiment.')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--use_cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")

# gp
parser.add_argument('--Lambda', type=float, default=10.0, help='Gradient penalty lambda hyperparameter')

# ttur
parser.add_argument('--lrD', type=float, default=0.0004, help='learning rate for Critic, default=0.0004')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for Generator, default=0.0001')

parser.add_argument('--gen_iter', type=int, default=50, help='num of iters to sample images from generator')
parser.add_argument('--exp', default="experient1", help='Where to store samples and models')
args = parser.parse_args()

def calc_gradient_penalty(netD, real_data, fake_data, args):
    # Follows the implementation at: https://github.com/bioinf-jku/TTUR/blob/master/WGAN_GP/gan_64x64_FID.py#L576-L590
    
    # alpha for interpolation
    alpha = torch.rand(args.batch_size, 1).to(real_data)
    alpha = alpha.expand(args.batch_size, real_data.nelement()//args.batch_size).contiguous()
    alpha = alpha.view(args.batch_size, args.nc, args.image_size, args.image_size)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(real_data),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    # record slopes
    slopes = gradients.norm(2, dim=1)
    
    args.slopes = slopes.clone().data

    gradient_penalty = ((slopes - 1.) ** 2).mean() * args.Lambda

    return gradient_penalty

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def makedirs(directory):
    """like mkdir -p"""
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    makedirs(args.exp)
    makedirs('{}/fake_samples'.format(args.exp))
    makedirs('{}/ckptG'.format(args.exp))
    makedirs('{}/ckptD'.format(args.exp))

    # writer for TensorboardX
    log_dir = os.path.join(args.exp, 'tboard')
    writer = SummaryWriter(log_dir=log_dir)
    args.writer = writer # no need to pass writer to every function

    logger = get_logger(logpath=os.path.join(args.exp, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    # seed
    logger.info("Random Seed: {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    if args.dataset == 'celeba':
        # follows the preprocess step from here: 
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/celeba.py#L175-L205
        # image: c x 218 x 178
        # Remove 40 pixels each side vertically
        # Remove 20 pixels each side horizontally
        h_start = 40
        h_end = 218 - 40
        w_start = 20
        w_end = 178 - 20
        crop_trans = lambda x: x[:, h_start:h_end, w_start:w_end]

        # x: [0, 1] -> [-1, 1]
        linear_trans = lambda x: 2*x -1

        # to -1 and 1
        dataset = dset.ImageFolder(root=args.dataroot,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Lambda(crop_trans),
                                    transforms.ToPILImage(),
                                    transforms.Scale(size=(args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(linear_trans),
                                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # or Normalize to (0.5, 0.5) without linear_trans
                                ]))
    elif args.dataset == 'mnist':
        dataset = dset.MNIST(root=args.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Scale(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        )
    assert dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=int(args.workers),
                                            drop_last=True)

    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)
    nc = int(args.nc)

    # ref: https://github.com/LynnHo/WGAN-GP-DRAGAN-Celeba-Pytorch/blob/master/train_celeba_wgan_gp.py
    netD = model.Discriminator(3, dim=args.image_size)
    netG = model.Generator(nz, dim=args.image_size)

    # load checkpoint
    if args.netG != '': # load checkpoint if needed
        netG.load_state_dict(torch.load(args.netG))
    logger.info(netG)

    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))
    logger.info(netD)

    # data placeholder
    input = torch.FloatTensor(args.batch_size, 3, args.image_size, args.image_size)
    noise = torch.FloatTensor(args.batch_size, nz)
    fixed_noise = torch.FloatTensor(64, nz).normal_(0, 1)

    if args.use_cuda:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, 0.999))

    gen_iterations = 0
    for epoch in range(args.niter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ###########################
            # (1) Update D network    #
            ###########################
            for p in netD.parameters(): # update D
                p.requires_grad = True  

            for p in netG.parameters(): # freeze G
                p.requires_grad = False

            data = data_iter.next()
            i += 1

            # train with real
            real_cpu, _ = data
            netD.zero_grad()
            batch_size = real_cpu.size(0)

            if args.use_cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            var_input = Variable(input)

            logits_D_real = netD(var_input)
            loss_D_real = -logits_D_real.mean()

            # train with fake
            noise.resize_(args.batch_size, nz).normal_(0, 1)
            var_noise = Variable(noise, volatile=True) # totally freeze netG

            fake = Variable(netG(var_noise).data)
            var_input = fake
            logits_D_fake = netD(var_input)
            loss_D_fake = logits_D_fake.mean()

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_cpu.data, fake.data, args)
            args.writer.add_histogram('Slopes', args.slopes.clone().cpu().data.numpy(), epoch * len(dataloader) + i)

            # calculate loss for D
            Wasserstein_D = loss_D_fake + loss_D_real
            loss_D = Wasserstein_D + gradient_penalty
            loss_D.backward()
            optimizerD.step()

            ###########################
            # (2) Update G network
            ###########################
            for p in netG.parameters(): # update G
                p.requires_grad = True

            for p in netD.parameters(): # freeze D
                p.requires_grad = False

            netG.zero_grad()
            noise.resize_(args.batch_size, nz).normal_(0, 1)
            var_noise = Variable(noise)
            fake = netG(var_noise)
            logits_G = netD(fake)

            loss_G = -logits_G.mean()
            loss_G.backward()

            optimizerG.step()
            gen_iterations += 1

            logger.info('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake: %f Wass._D: %f'
                % (epoch, args.niter, i, len(dataloader), gen_iterations,
                loss_D.item(), loss_G.item(), loss_D_real.item(), loss_D_fake.item(), Wasserstein_D.item()))
            
            # tensorboard
            # TODO: logging for Loss_D and Loss_G might have some bug here.
            if writer is not None:
                niter = epoch * len(dataloader) + i
                writer.add_scalar('Train/Loss_D', loss_D.item(), niter)
                writer.add_scalar('Train/Loss_G', loss_G.item(), niter)
                writer.add_scalar('Train/Loss_D_real', loss_D_real.item(), niter)
                writer.add_scalar('Train/Loss_D_fake', loss_D_fake.item(), niter)
                writer.add_scalar('Train/GP', gradient_penalty.item(), niter)
                writer.add_scalar('Train/Wasserstein_D', Wasserstein_D.item(), niter)
            
            if gen_iterations % args.gen_iter == 0:
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, '{0}/real_samples.png'.format(args.exp))

                # fixed_fake
                fixed_fake = netG(Variable(fixed_noise, volatile=True))
                fixed_fake.data = fixed_fake.data.mul(0.5).add(0.5)
                vutils.save_image(fixed_fake.data, '{0}/fake_samples/fixed_{1}.png'.format(args.exp, gen_iterations))

                # sampled_fake
                sample_noise = torch.FloatTensor(64, nz).normal_(0, 1).cuda()
                random_fake = netG(Variable(sample_noise, volatile=True))
                random_fake.data = random_fake.data.mul(0.5).add(0.5)
                vutils.save_image(random_fake.data, '{0}/fake_samples/random_{1}.png'.format(args.exp, gen_iterations))

                # tboard
                fixed_fake_grid = vutils.make_grid(fixed_fake.data, normalize=True, scale_each=True)
                random_fake_grid = vutils.make_grid(random_fake.data, normalize=True, scale_each=True)

                writer.add_image('FakeSamples/fixed', fixed_fake_grid, gen_iterations)
                writer.add_image('FakeSamples/random', random_fake_grid, gen_iterations)

        # save checkpoint
        if epoch % 3 == 0:
            torch.save(netG.state_dict(), '{0}/ckptG/netG_epoch_{1}.pth'.format(args.exp, epoch))
            torch.save(netD.state_dict(), '{0}/ckptD/netD_epoch_{1}.pth'.format(args.exp, epoch))
