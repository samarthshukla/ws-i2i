import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class HybridModel(BaseModel):
    def name(self):
        return 'HybridModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=10, norm='instance')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=150.0, help='weight for L1 loss')
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=10.0, help='weight for identity loss')
            parser.add_argument('--dataroot_unaligned', required=True, help='path to unaligned images (should have subfolders trainA, trainB, valA, valB, etc)')
            parser.add_argument('--pool_size_unaligned', type=int, default=50, help='the size of image buffer that stores previously generated images')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_A']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        # load/define networks
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = False
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_B_pool_aligned = ImagePool(opt.pool_size)
            self.fake_A_pool_aligned = ImagePool(opt.pool_size)
            self.real_A_pool_aligned = ImagePool(opt.pool_size)
            self.real_B_pool_aligned = ImagePool(opt.pool_size)

            self.fake_B_pool_unaligned = ImagePool(opt.pool_size_unaligned)
            self.fake_A_pool_unaligned = ImagePool(opt.pool_size_unaligned)
            self.real_A_pool_unaligned = ImagePool(opt.pool_size_unaligned)
            self.real_B_pool_unaligned = ImagePool(opt.pool_size_unaligned)

            self.criterionGAN = networks.GANLoss(case=2).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterion_identity = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.L1Loss()
            self.lambda_L1 = self.opt.lambda_L1

            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            ##Temporary
            self.loss_D_real = 0.0
            self.loss_D_fake = 0.0

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, mode='aligned'):
        self.fake_B = self.netG_A(self.real_A)
        self.same_B = self.netG_A(self.real_B)

        self.fake_A = self.netG_B(self.real_B)
        self.same_A = self.netG_B(self.real_A)

        #if(mode == 'unaligned'):
        with torch.no_grad():
            self.rec_A = self.netG_B(self.fake_B)
            self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        if(self.opt.loss_case=='rel_bce'):
            loss_D = self.criterionGAN(pred_real - pred_fake, True)
        else:
            loss_D = (torch.mean((pred_real - torch.mean(pred_fake) - 1.0) ** 2) + torch.mean((pred_fake - torch.mean(pred_real) + 1.0) ** 2))/2
        loss_D.backward()
        return loss_D

    def backward_D_A(self, mode='aligned'):
        if(mode=='aligned'):
            fake_B = self.fake_B_pool_aligned.query(self.fake_B)
        elif(mode=='unaligned'):
            fake_B = self.fake_B_pool_unaligned.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self, mode='aligned'):
        if(mode=='aligned'):
            fake_A = self.fake_A_pool_aligned.query(self.fake_A)
        elif(mode=='unaligned'):
            fake_A = self.fake_A_pool_unaligned.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self, mode='aligned'):

        if(mode=='aligned'):
            real_B = self.real_B_pool_aligned.query(self.real_B)
            real_A = self.real_A_pool_aligned.query(self.real_A)

        elif(mode=='unaligned'):
            real_B = self.real_B_pool_unaligned.query(self.real_B)
            real_A = self.real_A_pool_unaligned.query(self.real_A)

        pred_fake_B = self.netD_A(self.fake_B)
        pred_real_B = self.netD_A(real_B)

        pred_fake_A = self.netD_B(self.fake_A)
        pred_real_A = self.netD_B(real_A)

        if(self.opt.loss_case=='rel_bce'):
            self.loss_G_GAN_A = self.criterionGAN(pred_fake_B - pred_real_B, True)
            self.loss_G_GAN_B = self.criterionGAN(pred_fake_A - pred_real_A, True)
        else:
            self.loss_G_GAN_A = (torch.mean((pred_real_A - torch.mean(pred_fake_A) + 1.0) ** 2) + torch.mean((pred_fake_A - torch.mean(pred_real_A) - 1.0) ** 2))/2
            self.loss_G_GAN_B = (torch.mean((pred_real_B - torch.mean(pred_fake_B) + 1.0) ** 2) + torch.mean((pred_fake_B - torch.mean(pred_real_B) - 1.0) ** 2))/2

        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B

        self.loss_cycle = self.loss_cycle_A + self.loss_cycle_B

        if(mode == 'aligned'):
            self.loss_G_L1_B = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
            self.loss_G_L1_A = self.criterionL1(self.fake_A, self.real_A) * self.lambda_L1

            self.loss_G_L1 = self.loss_G_L1_A + self.loss_G_L1_B

            self.l1_counter.append(self.loss_G_L1_B.item())

        self.loss_G_identity_A = self.criterion_identity(self.same_A, self.real_A) * self.opt.lambda_identity
        self.loss_G_identity_B = self.criterion_identity(self.same_B, self.real_B) * self.opt.lambda_identity
        self.loss_G_identity = self.loss_G_identity_A + self.loss_G_identity_B

        self.loss_G_GAN = self.loss_G_GAN_A + self.loss_G_GAN_B

        self.loss_G = self.loss_G_GAN + self.loss_G_identity

        self.loss_G += self.loss_cycle

        if(mode == 'aligned'):

            self.loss_G += self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self, mode='aligned'):
        self.forward(mode)

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A(mode)
        self.backward_D_B(mode)
        self.optimizer_D.step()

        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G(mode)
        self.optimizer_G.step()
