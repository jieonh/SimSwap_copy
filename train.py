import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.backends import cudnn

import wandb

# hydra 설정 관련
import hydra
from omegaconf import DictConfig, OmegaConf, errors
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

# 내부 패키지
from util import util
from util.plot import plot_batch

from models.projected_model import fsModel
from data.data_loader_Swapping import GetLoader

# 문자열을 boolean으로 변환하는 함수
def str2bool(v):
    return v.lower() in ('true')

# 학습 함수
def train(opt, model, log_name, iter_path, sample_path):
    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

    loss_avg        = 0
    refresh_count   = 0
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)

    # 데이터 로더 설정
    train_loader    = GetLoader(
        dataset_roots=opt.dataset,
        batch_size=opt.batchSize,
        dataloader_workers=16,
        random_seed=opt.random_seed)

    randindex = [i for i in range(opt.batchSize)]
    random.shuffle(randindex)

    # 학습 재개 여부에 따른 시작 지점 설정
    if not opt.continue_train:
        start   = 0
    else:
        start   = int(opt.which_epoch)
    total_step  = opt.total_step
    import datetime
    print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    wandb.init(project=opt.project_name, config=OmegaConf.to_container(opt, resolve=True))

    model.netD.feature_network.requires_grad_(False)

    # 학습 루프
    for step in tqdm(range(start, total_step)):
        model.netG.train()
        for interval in range(2):
            random.shuffle(randindex)
            src_image1, src_image2  = train_loader.next()
            
            if step%2 == 0:
                img_id = src_image2
            else:
                img_id = src_image2[randindex]

            img_id_112      = F.interpolate(img_id,size=(112,112), mode='bicubic')
            latent_id       = model.netArc(img_id_112)
            latent_id       = F.normalize(latent_id, p=2, dim=1)
            if interval:
                
                img_fake        = model.netG(src_image1, latent_id)
                gen_logits,_    = model.netD(img_fake.detach(), None)
                loss_Dgen       = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                real_logits,_   = model.netD(src_image2,None)
                loss_Dreal      = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                loss_D          = loss_Dgen + loss_Dreal
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
            else:
                
                # model.netD.requires_grad_(True)
                img_fake        = model.netG(src_image1, latent_id)
                # G 손실 계산
                #print(img_fake.shape) #torch.Size([4, 3, 1016, 1920])
                gen_logits,feat = model.netD(img_fake, None)
                
                loss_Gmain      = (-gen_logits).mean()
                img_fake_down   = F.interpolate(img_fake, size=(112,112), mode='bicubic')
                latent_fake     = model.netArc(img_fake_down)
                latent_fake     = F.normalize(latent_fake, p=2, dim=1)
                loss_G_ID       = (1 - model.cosin_metric(latent_fake, latent_id)).mean()
                real_feat       = model.netD.get_feature(src_image1)
                feat_match_loss = model.criterionFeat(feat["3"],real_feat["3"]) 
                loss_G          = loss_Gmain + loss_G_ID * opt.lambda_id + feat_match_loss * opt.lambda_feat
                

                if step%2 == 0:
                    ### 추가 - 손실 함수 계산 전 크기 맞추기
                    img_fake_resized = F.interpolate(img_fake, size=src_image1.shape[2:], mode='bilinear', align_corners=True)
                    
                    
                    #G_Rec  
                    loss_G_Rec  = model.criterionRec(img_fake_resized, src_image1) * opt.lambda_rec
                    loss_G      += loss_G_Rec

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                

        # 결과 및 오류 출력
        # 로그 정보 출력
        if (step + 1) % opt.log_frep == 0:
            # errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            errors = {
                "G_Loss":loss_Gmain.item(),
                "G_ID":loss_G_ID.item(),
                "G_Rec":loss_G_Rec.item(),
                "G_feat_match":feat_match_loss.item(),
                "D_fake":loss_Dgen.item(),
                "D_real":loss_Dreal.item(),
                "D_loss":loss_D.item()
            }
            wandb.log(errors, step=step)
            message = '( step: %d, ) ' % (step)
            for k, v in errors.items():
                message += '%s: %.3f ' % (k, v)

            print(message)
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)

        # 출력 이미지 표시
        if (step + 1) % opt.sample_freq == 0:
            model.netG.eval()
            with torch.no_grad():
                imgs        = list()
                zero_img    = (torch.zeros_like(src_image1[0,...]))
                imgs.append(zero_img.cpu().numpy())
                save_img    = ((src_image1.cpu())* imagenet_std + imagenet_mean).numpy()
                for r in range(opt.batchSize):
                    imgs.append(save_img[r,...])
                arcface_112     = F.interpolate(src_image2,size=(112,112), mode='bicubic')
                id_vector_src1  = model.netArc(arcface_112)
                id_vector_src1  = F.normalize(id_vector_src1, p=2, dim=1)

                for i in range(opt.batchSize):
                    
                    imgs.append(save_img[i,...])
                    image_infer = src_image1[i, ...].repeat(opt.batchSize, 1, 1, 1)
                    img_fake    = model.netG(image_infer, id_vector_src1).cpu()
                    
                    img_fake    = img_fake * imagenet_std
                    img_fake    = img_fake + imagenet_mean
                    img_fake    = img_fake.numpy()
                    for j in range(opt.batchSize):
                        imgs.append(img_fake[j,...])
                print("Save test data")
                
                
                # # imgs 리스트에 있는 배열들의 형상을 출력하여 확인
                # for i, img in enumerate(imgs):
                #     print(f"img[{i}].shape: {img.shape}")

                # 배열 크기를 맞추기 위해 보간 사용
                target_shape = imgs[0].shape  # 첫 번째 이미지의 크기로 맞추기

                aligned_imgs = []
                for img in imgs:
                    if img.shape != target_shape:
                        img = F.interpolate(torch.tensor(img).unsqueeze(0), size=target_shape[1:], mode='bilinear', align_corners=True).squeeze(0).numpy()
                    aligned_imgs.append(img)
                
                # 배열을 스택
                imgs = np.stack(aligned_imgs, axis=0).transpose(0, 2, 3, 1)
                #imgs = np.stack(imgs, axis = 0).transpose(0,2,3,1)
                plot_batch(imgs, os.path.join(sample_path, 'step_'+str(step+1)+'.jpg'))

        # 최신 모델 저장
        if (step+1) % opt.model_freq==0:
            print('saving the latest model (steps %d)' % (step+1))
            model.save(step+1)            
            np.savetxt(iter_path, (step+1, total_step), delimiter=',', fmt='%d')
    wandb.finish()
    
# hydra 메인 함수
@hydra.main(
    config_path="configs", 
    config_name="train.yaml",
    version_base=None
)
def main(opt: DictConfig):
    #opt         = TrainOptions().parse()
    iter_path   = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    
    log_path = os.path.join(opt.checkpoints_dir, opt.name, 'summary')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
        
    else:    
        start_epoch, epoch_iter = 1, 0

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    
    print(f"GPU used : {torch.cuda.is_available()}")
    print("GPU number : ", str(opt.gpu_ids))
    
    cudnn.benchmark = True

    model = fsModel()
    model.initialize(opt)

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)
    
    train(opt, model, log_name, iter_path, sample_path)
        
if __name__ == '__main__':
    main()
