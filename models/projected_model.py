

import torch
import torch.nn as nn

from .base_model import BaseModel
from .fs_networks_fix import Generator_Adain_Upsample

from pg_modules.projected_discriminator import ProjectedDiscriminator

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

class fsModel(BaseModel):
    def name(self):
        return 'fsModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
        self.isTrain = opt.isTrain

        # Generator network
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=opt.Gdeep)
        self.netG.cuda()

        # Id network
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
        self.netArc = netArc_checkpoint
        self.netArc = self.netArc.cuda()
        self.netArc.eval()
        self.netArc.requires_grad_(False)
        if not self.isTrain:
            pretrained_path =  opt.checkpoints_dir
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            return
        self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
        # self.netD.feature_network.requires_grad_(False)
        self.netD.cuda()


        if self.isTrain:
            # define loss functions
            self.criterionFeat  = nn.L1Loss()
            self.criterionRec   = nn.L1Loss()


           # initialize optimizers

            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)

        # load networks
        if opt.continue_train:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # print (pretrained_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)
        torch.cuda.empty_cache()

    def cosin_metric(self, x1, x2):
        #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))



    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)
        self.save_optim(self.optimizer_G, 'G', which_epoch)
        self.save_optim(self.optimizer_D, 'D', which_epoch)
        '''if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)'''

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

import onnxruntime

class fsModel_onnx(BaseModel):
    def name(self):
        return 'fsModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.isONNX = opt.pretrained_model_path.endswith('.onnx')

        if self.isONNX:
            self.ort_session = onnxruntime.InferenceSession(opt.pretrained_model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            print("ONNX model loaded.")
        else:
            # Generator network
            self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=opt.Gdeep)
            self.netG.cuda()

            # Id network
            netArc_checkpoint = opt.Arc_path
            netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
            self.netArc = netArc_checkpoint
            self.netArc = self.netArc.cuda()
            self.netArc.eval()
            self.netArc.requires_grad_(False)

            if not self.isTrain:
                pretrained_path = opt.checkpoints_dir
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
                return   
            
            self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
            self.netD.cuda()

            if self.isTrain:
                # define loss functions
                self.criterionFeat = nn.L1Loss()
                self.criterionRec = nn.L1Loss()

                # initialize optimizers
                params = list(self.netG.parameters())
                self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8)

                params = list(self.netD.parameters())
                self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8)

            # load networks
            if opt.continue_train:
                pretrained_path = '' if not self.isTrain else opt.load_pretrain
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
                self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)
            torch.cuda.empty_cache()

    def cosin_metric(self, x1, x2):
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def forward(self, x):
        if self.isONNX:
            ort_inputs = {self.input_name: x.cpu().numpy()}
            ort_outs = self.ort_session.run([self.output_name], ort_inputs)
            return torch.tensor(ort_outs[0]).cuda()
        else:
            return self.netG(x)

    def save(self, which_epoch):
        if not self.isONNX:
            self.save_network(self.netG, 'G', which_epoch)
            self.save_network(self.netD, 'D', which_epoch)
            self.save_optim(self.optimizer_G, 'G', which_epoch)
            self.save_optim(self.optimizer_D, 'D', which_epoch)

    def update_fixed_params(self):
        if not self.isONNX:
            params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            if self.opt.verbose:
                print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        if not self.isONNX:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            if self.opt.verbose:
                print('update learning rate: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr

# # import torch
# # import torch.nn as nn
# # import onnx
# # import onnxruntime

# # from .base_model import BaseModel
# # from .fs_networks_fix import Generator_Adain_Upsample

# # from pg_modules.projected_discriminator import ProjectedDiscriminator

# # def compute_grad2(d_out, x_in):
# #     batch_size = x_in.size(0)
# #     grad_dout = torch.autograd.grad(
# #         outputs=d_out.sum(), inputs=x_in,
# #         create_graph=True, retain_graph=True, only_inputs=True
# #     )[0]
# #     grad_dout2 = grad_dout.pow(2)
# #     assert(grad_dout2.size() == x_in.size())
# #     reg = grad_dout2.view(batch_size, -1).sum(1)
# #     return reg

# # class fsModel(BaseModel):
# #     def name(self):
# #         return 'fsModel'

# #     def initialize(self, opt):
# #         BaseModel.initialize(self, opt)
# #         self.isTrain = opt.isTrain

# #         # Generator network
# #         self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=opt.Gdeep)
# #         self.netG.cuda()

# #         # Id network
# #         netArc_checkpoint = opt.Arc_path
# #         netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
# #         self.netArc = netArc_checkpoint
# #         self.netArc = self.netArc.cuda()
# #         self.netArc.eval()
# #         self.netArc.requires_grad_(False)

# #         if not self.isTrain:
# #             pretrained_path = opt.checkpoints_dir
# #             self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
# #             return
        
# #         self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
# #         self.netD.cuda()

# #         if self.isTrain:
# #             self.criterionFeat  = nn.L1Loss()
# #             self.criterionRec   = nn.L1Loss()
# #             self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8)
# #             self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8)

# #         if opt.continue_train:
# #             pretrained_path = '' if not self.isTrain else opt.load_pretrain
# #             self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
# #             self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
# #             self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
# #             self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)
# #         torch.cuda.empty_cache()

# #         # Load pretrained weights if specified
# #         if opt.fine_tuning:
# #             self.load_pretrained_weights(opt)
# #             self.freeze_layers()

# #     def load_pretrained_weights(self, opt):
# #         if opt.pretrained_model_path.endswith('.pth'):
# #             checkpoint = torch.load(opt.pretrained_model_path)
# #             self.netG.load_state_dict(checkpoint['netG'])
# #             self.netD.load_state_dict(checkpoint['netD'])
# #             print("Pretrained weights loaded.")
# #         elif opt.pretrained_model_path.endswith('.onnx'):
# #             self.ort_session = onnxruntime.InferenceSession(opt.pretrained_model_path)
# #             self.input_name = self.ort_session.get_inputs()[0].name
# #             self.output_name = self.ort_session.get_outputs()[0].name
# #             print("ONNX model loaded.")
# #         else:
# #             raise ValueError("Unsupported model format: {}".format(opt.pretrained_model_path))

# #     def freeze_layers(self):
# #         for param in self.netG.parameters():
# #             param.requires_grad = False
# #         for param in self.netD.parameters():
# #             param.requires_grad = False
# #         for param in self.netG.layer_to_train.parameters():
# #             param.requires_grad = True

# #     def forward(self, x):
# #         if hasattr(self, 'ort_session'):
# #             ort_inputs = {self.input_name: x.cpu().numpy()}
# #             ort_outs = self.ort_session.run([self.output_name], ort_inputs)
# #             return torch.tensor(ort_outs[0]).cuda()
# #         else:
# #             return self.netG(x)

# #     def cosin_metric(self, x1, x2):
# #         return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

# #     def save(self, which_epoch):
# #         self.save_network(self.netG, 'G', which_epoch)
# #         self.save_network(self.netD, 'D', which_epoch)
# #         self.save_optim(self.optimizer_G, 'G', which_epoch)
# #         self.save_optim(self.optimizer_D, 'D', which_epoch)

# #     def update_fixed_params(self):
# #         params = list(self.netG.parameters())
# #         if self.gen_features:
# #             params += list(self.netE.parameters())
# #         self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
# #         if self.opt.verbose:
# #             print('------------ Now also finetuning global generator -----------')

# #     def update_learning_rate(self):
# #         lrd = self.opt.lr / self.opt.niter_decay
# #         lr = self.old_lr - lrd
# #         for param_group in self.optimizer_D.param_groups:
# #             param_group['lr'] = lr
# #         for param_group in self.optimizer_G.param_groups:
# #             param_group['lr'] = lr
# #         if self.opt.verbose:
# #             print('update learning rate: %f -> %f' % (self.old_lr, lr))
# #         self.old_lr = lr

# import torch
# import torch.nn as nn
# import onnx
# import onnxruntime

# from .base_model import BaseModel
# from .fs_networks_fix import Generator_Adain_Upsample
# from pg_modules.projected_discriminator import ProjectedDiscriminator

# def compute_grad2(d_out, x_in):
#     batch_size = x_in.size(0)
#     grad_dout = torch.autograd.grad(
#         outputs=d_out.sum(), inputs=x_in,
#         create_graph=True, retain_graph=True, only_inputs=True
#     )[0]
#     grad_dout2 = grad_dout.pow(2)
#     assert(grad_dout2.size() == x_in.size())
#     reg = grad_dout2.view(batch_size, -1).sum(1)
#     return reg

# class fsModel(BaseModel):
#     def name(self):
#         return 'fsModel'

#     def initialize(self, opt):
#         BaseModel.initialize(self, opt)
#         self.isTrain = opt.isTrain

#         # Generator network
#         self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=opt.Gdeep)
#         self.netG.cuda()

#         # Id network
#         netArc_checkpoint = opt.Arc_path
#         netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
#         self.netArc = netArc_checkpoint
#         self.netArc = self.netArc.cuda()
#         self.netArc.eval()
#         self.netArc.requires_grad_(False)

#         self.is_onnx_model = opt.pretrained_model_path.endswith('.onnx')

#         if self.is_onnx_model:
#             self.ort_session = onnxruntime.InferenceSession(opt.pretrained_model_path)
#             self.input_name = self.ort_session.get_inputs()[0].name
#             self.output_name = self.ort_session.get_outputs()[0].name
#             print("ONNX model loaded.")
#         else:
#             if not self.isTrain:
#                 pretrained_path =  opt.checkpoints_dir
#                 self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
#                 return
            
#             self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
#             self.netD.cuda()

#             if self.isTrain:
#                 # define loss functions
#                 self.criterionFeat  = nn.L1Loss()
#                 self.criterionRec   = nn.L1Loss()

#                 # initialize optimizers
#                 # optimizer G
#                 params = list(self.netG.parameters())
#                 self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8)

#                 # optimizer D
#                 params = list(self.netD.parameters())
#                 self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8)

#             # load networks
#             if opt.continue_train:
#                 pretrained_path = '' if not self.isTrain else opt.load_pretrain
#                 self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
#                 self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
#                 self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
#                 self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)
#             torch.cuda.empty_cache()

#     def cosin_metric(self, x1, x2):
#         return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

#     def forward(self, x):
#         if hasattr(self, 'ort_session'):
#             ort_inputs = {self.input_name: x.cpu().numpy()}
#             ort_outs = self.ort_session.run([self.output_name], ort_inputs)
#             return torch.tensor(ort_outs[0]).cuda()
#         else:
#             return super().forward(x)

#     def save(self, which_epoch):
#         self.save_network(self.netG, 'G', which_epoch)
#         self.save_network(self.netD, 'D', which_epoch)
#         self.save_optim(self.optimizer_G, 'G', which_epoch)
#         self.save_optim(self.optimizer_D, 'D', which_epoch)

#     def update_fixed_params(self):
#         params = list(self.netG.parameters())
#         if self.gen_features:
#             params += list(self.netE.parameters())
#         self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
#         if self.opt.verbose:
#             print('------------ Now also finetuning global generator -----------')

#     def update_learning_rate(self):
#         lrd = self.opt.lr / self.opt.niter_decay
#         lr = self.old_lr - lrd
#         for param_group in self.optimizer_D.param_groups:
#             param_group['lr'] = lr
#         for param_group in self.optimizer_G.param_groups:
#             param_group['lr'] = lr
#         if self.opt.verbose:
#             print('update learning rate: %f -> %f' % (self.old_lr, lr))
#         self.old_lr = lr
