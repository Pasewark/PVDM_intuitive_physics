import os
import random
import numpy as np
import sys; sys.path.extend([sys.path[0][:-4], '/app'])

import time
import tqdm
import copy
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


from utils import AverageMeter
from evals.eval import test_psnr, test_ifvd, test_fvd_ddpm
from models.ema import LitEma
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR

from torchvision import transforms as ttransforms
import tensorflow as tf


_NUM_PROBE_TFRECORDS = 20
_NUM_FREEFORM_TRAIN_TFRECORDS = 100
_NUM_FREEFORM_TEST_TFRECORDS = 10

_FREEFORM_FEATURES = dict(
     image=tf.io.FixedLenFeature(dtype=tf.string, shape=()),                                                                                                                                             
     mask=tf.io.FixedLenFeature(dtype=tf.string, shape=()),                                                                                                                                              
     camera_pose=tf.io.FixedLenFeature(dtype=tf.float32, shape=(15, 6)),                                                                                                                              
)

_PROBE_FEATURES = dict(                                                                                                                                                                                                
    possible_image=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    possible_mask=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    possible_camera_pose=tf.io.FixedLenFeature(dtype=tf.float32,
                                               shape=(2, 15, 6)),
    impossible_image=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    impossible_mask=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    impossible_camera_pose=tf.io.FixedLenFeature(dtype=tf.float32,
                                                 shape=(2, 15, 6)),                                                                                                                            
)

def _parse_freeform_row(row):                                                                                                                                                                                             
  row = tf.io.parse_example(row, _FREEFORM_FEATURES)                                                                                                                                                                      
  row['image'] = tf.reshape(tf.io.decode_raw(row['image'], tf.uint8),
                            [15, 64, 64, 3])                                                                                                  

  row['mask'] = tf.reshape(tf.io.decode_raw(row['mask'], tf.uint8),
                           [15, 64, 64])                                                                                                       
  return row                                                                                                                                                                                                     
                                                                                                                                                                                                                 
def _parse_probe_row(row):                                                                                                                                                                                             
  row = tf.io.parse_example(row, _PROBE_FEATURES)                                                                                                                                                                      
  for prefix in ['possible', 'impossible']:                                                                                                                                                                      
    row[f'{prefix}_image'] = tf.reshape(
        tf.io.decode_raw(row[f'{prefix}_image'], tf.uint8),
        [2, 15, 64, 64, 3])                                                                                                  
    row[f'{prefix}_mask'] = tf.reshape(
        tf.io.decode_raw(row[f'{prefix}_mask'], tf.uint8),
        [2, 15, 64, 64])                                                                                                       
  return row                                                                                                                                                                                                     


def _make_tfrecord_paths(dir_name, subdir_name, num_records):
  root = f'gs://physical_concepts/{dir_name}/{subdir_name}/data.tfrecord'
  paths = [f'{root}-{i:05}-of-{num_records:05}' for i in range(num_records)]
  return paths

def make_freeform_tfrecord_dataset(is_train, shuffle=False):
  """Returns a TFRecordDataset for freeform data."""
  if is_train:
    subdir_str = 'train'
    num_records = _NUM_FREEFORM_TRAIN_TFRECORDS
  else:
    subdir_str = 'test'
    num_records = _NUM_FREEFORM_TEST_TFRECORDS

  tfrecord_paths = _make_tfrecord_paths('freeform', subdir_str, num_records)
  ds = tf.data.TFRecordDataset(tfrecord_paths, compression_type='GZIP')
  ds = ds.map(_parse_freeform_row)                        
  if shuffle:
    ds = ds.shuffle(buffer_size=50)                                                                                                                                                                
  return ds

def make_probe_tfrecord_dataset(concept_name, shuffle=False):
  """Returns a TFRecordDataset for probes data."""
  tfrecord_paths = _make_tfrecord_paths('probes', concept_name, 20)
  ds = tf.data.TFRecordDataset(tfrecord_paths, compression_type='GZIP')
  ds = ds.map(_parse_probe_row)
  if shuffle:
    ds = ds.shuffle(buffer_size=20)                                                                                                                                                                                         
  return ds

def concat_frames_horizontally(v):
  """Arrange a video as horizontally aligned frames."""
  num_frames = v.shape[0]
  # [F, H, W, C] --> [H, W*F, C].
  return np.concatenate([v[x] for x in range(num_frames)], axis=1)

def describe(d):
  """Describe the contents of a dict of np arrays."""
  for k, v in d.items():
    print(f'\'{k}\' has shape: {v.shape}')
    print(f'===================')
    print(f'min: {v.min()}, max: {v.max()}, type: {v.dtype}\n')

def colorize_mask(m):
  """Adds color channel to mask of unique object ids."""
  m = m[..., np.newaxis]
  min_val = np.max(m)
  max_val = np.min(m)
  # Use three different mappings into range [0-1] to form color.
  c1 = (m - min_val)/(max_val - min_val)
  c2 = np.abs((m - max_val)/(min_val - max_val))
  c3 = (c1+c2)/2.
  mask = np.concatenate([c1, c2, c3], axis=-1)
  return mask

def plot_video(v, name=''):
  """Plots something of the form [num_frames, height, width, channel]."""
  num_frames = v.shape[0]
  width = v.shape[2]
  v = concat_frames_horizontally(v)
  plt.figure(figsize=(30,5))
  plt.imshow(v)
  plt.xticks(ticks=[i*width+width/2 for i in range(num_frames)],
            labels=range(1,num_frames+1))
  plt.yticks([])
  plt.xlabel('Frame Number')
  plt.title(name)
  plt.show()

class MyDataset(torch.utils.data.IterableDataset):
    def __init__(self, data,batch_size,size=64):
        super(MyDataset).__init__()

        self.data=data
        transforms = [ttransforms.Resize((size, size))]
        frame_transform = ttransforms.Compose(transforms)
        self.frame_transform = frame_transform
        self.batch_size=batch_size

    def __iter__(self):
        while True:
            videos=[]
            for _ in range(self.batch_size):
                full_data=next(self.data)
                # I think the data should be (b t c h w)
                mask=torch.from_numpy(full_data['mask'].astype(float)/10).unsqueeze(3).permute(0,3,1,2).type(torch.float32)
                vid=torch.from_numpy(full_data['image']).permute(0,3,1,2).type(torch.float32)
                # combine mask and vid
                #combined=torch.cat((mask,vid),dim=1)
                #assert combined.shape==(15,4,mask.shape[-1],mask.shape[-1])
                #videos.append(combined)
                videos.append(vid)
                
                #videos.append(self.frame_transform(torch.from_numpy(vid).unsqueeze(3).permute(3,0,1,2)).type(torch.float32))
            yield torch.stack(videos,0)


def latentDDPM(rank, first_stage_model, model, opt, criterion, train_loader, test_loader, scheduler, ema_model=None, cond_prob=0.3, logger=None):
    scaler = GradScaler()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if rank == 0:
        rootdir = logger.logdir

    device = torch.device('cuda', rank)

    losses = dict()
    losses['diffusion_loss'] = AverageMeter()
    check = time.time()

    #lr_scheduler = LambdaLR(opt, scheduler)
    if ema_model == None:
        ema_model = copy.deepcopy(model)
        ema = LitEma(ema_model)
        ema_model.eval()
    else:
        ema = LitEma(ema_model)
        ema.num_updates = torch.tensor(11200,dtype=torch.int)
        ema_model.eval()

    first_stage_model.eval()
    model.train()

    for it, (x, _) in enumerate(train_loader):
        x = x.to(device)
        x = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w') # videos
        c = None

        # conditional free guidance training
        model.zero_grad()

        if model.module.diffusion_model.cond_model:
            p = np.random.random()

            if p < cond_prob:
                c, x = torch.chunk(x, 2, dim=2)
                mask = (c+1).contiguous().view(c.size(0), -1) ** 2
                mask = torch.where(mask.sum(dim=-1) > 0, 1, 0).view(-1, 1, 1)

                with autocast():
                    with torch.no_grad():
                        z = first_stage_model.module.extract(x).detach()
                        c = first_stage_model.module.extract(c).detach()
                        c = c * mask + torch.zeros_like(c).to(c.device) * (1-mask)

            else:
                c, x_tmp = torch.chunk(x, 2, dim=2)
                mask = (c+1).contiguous().view(c.size(0), -1) ** 2
                mask = torch.where(mask.sum(dim=-1) > 0, 1, 0).view(-1, 1, 1, 1, 1)

                clip_length = x.size(2)//2
                prefix = random.randint(0, clip_length)
                x = x[:, :, prefix:prefix+clip_length, :, :] * mask + x_tmp * (1-mask)
                with autocast():
                    with torch.no_grad():
                        z = first_stage_model.module.extract(x).detach()
                        c = torch.zeros_like(z).to(device)

            (loss, t), loss_dict = criterion(z.float(), c.float())

        else:
            if it == 0:
                print("Unconditional model")
            with autocast():    
                with torch.no_grad():
                    z = first_stage_model.module.extract(x).detach()

            (loss, t), loss_dict = criterion(z.float())

        """
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        """
        loss.backward()
        opt.step()

        losses['diffusion_loss'].update(loss.item(), 1)

        # ema model
        if it % 25 == 0 and it > 0:
            ema(model)

        if it % 500 == 0:
            #psnr = test_psnr(rank, model, test_loader, it, logger)
            if logger is not None and rank == 0:
                logger.scalar_summary('train/diffusion_loss', losses['diffusion_loss'].average, it)

                log_('[Time %.3f] [Diffusion %f]' %
                     (time.time() - check, losses['diffusion_loss'].average))

            losses = dict()
            losses['diffusion_loss'] = AverageMeter()


        if it % 10000 == 0 and rank == 0:
            torch.save(model.module.state_dict(), rootdir + f'model_{it}.pth')
            ema.copy_to(ema_model)
            torch.save(ema_model.module.state_dict(), rootdir + f'ema_model_{it}.pth')
            fvd = test_fvd_ddpm(rank, ema_model, first_stage_model, test_loader, it, logger)


            if logger is not None and rank == 0:
                logger.scalar_summary('test/fvd', fvd, it)

                log_('[Time %.3f] [FVD %f]' %
                     (time.time() - check, fvd))


def first_stage_train(rank, model, opt, d_opt, criterion, train_loader, test_loader, first_model, fp, logger=None):
    tf.config.set_visible_devices([], 'GPU')
    #print('first stage train')
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if rank == 0:
        rootdir = logger.logdir

    device = torch.device('cuda', rank)
    
    losses = dict()
    losses['ae_loss'] = AverageMeter()
    losses['d_loss'] = AverageMeter()
    check = time.time()

    accum_iter = 3
    disc_opt = False

    if fp:
        scaler = GradScaler()
        scaler_d = GradScaler()

        try:
            scaler.load_state_dict(torch.load(os.path.join(first_model, 'scaler.pth')))
            scaler_d.load_state_dict(torch.load(os.path.join(first_model, 'scaler_d.pth')))
        except:
            print("Fail to load scalers. Start from initial point.")


    model.train()
    #disc_start = criterion.module.discriminator_iter_start
    disc_start=criterion.discriminator_iter_start
    
    #for it, (x, _) in enumerate(train_loader):
    it=-1
    mybatch_size=32
    for epoch in range(100):
        
        train_ds = make_freeform_tfrecord_dataset(is_train=True, shuffle=True)
        torch_dataset=MyDataset(train_ds.as_numpy_iterator(),batch_size=mybatch_size)
        for step,data in enumerate(torch_dataset):
            it+=1
            if step*mybatch_size>299000:break

            if it > 1000000:
                break
                
            x=data
            batch_size = x.size(0)

            x = x.to(device)
            x = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w') # videos

            if not disc_opt:
                with autocast():
                    x_tilde, vq_loss  = model(x)

                    if it % accum_iter == 0:
                        model.zero_grad()
                    ae_loss = criterion(vq_loss, x, 
                                        rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size),
                                        optimizer_idx=0,
                                        global_step=it)

                    ae_loss = ae_loss / accum_iter

                scaler.scale(ae_loss).backward()

                if it % accum_iter == accum_iter - 1:
                    scaler.step(opt)
                    scaler.update()

                losses['ae_loss'].update(ae_loss.item(), 1)

            else:
                if it % accum_iter == 0:
                    criterion.zero_grad()

                with autocast():
                    with torch.no_grad():
                        x_tilde, vq_loss = model(x)
                    d_loss = criterion(vq_loss, x, 
                             rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size),
                             optimizer_idx=1,
                             global_step=it)
                    d_loss = d_loss / accum_iter

                scaler_d.scale(d_loss).backward()

                if it % accum_iter == accum_iter - 1:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler_d.unscale_(d_opt)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(criterion.module.discriminator_2d.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(criterion.module.discriminator_3d.parameters(), 1.0)

                    scaler_d.step(d_opt)
                    scaler_d.update()

                losses['d_loss'].update(d_loss.item() * 3, 1)

            if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
                if disc_opt:
                    disc_opt = False
                else:
                    disc_opt = True

            if it % 2000 == 0:
                #fvd = test_ifvd(rank, model, test_loader, it, logger)
                #psnr = test_psnr(rank, model, test_loader, it, logger)
                if logger is not None and rank == 0:
                    logger.scalar_summary('train/ae_loss', losses['ae_loss'].average, it)
                    logger.scalar_summary('train/d_loss', losses['d_loss'].average, it)
                    #logger.scalar_summary('test/psnr', psnr, it)
                    #logger.scalar_summary('test/fvd', fvd, it)

                    #log_('[Time %.3f] [AELoss %f] [DLoss %f] [PSNR %f]' %
                    #     (time.time() - check, losses['ae_loss'].average, losses['d_loss'].average, psnr))
                    log_('[Time %.3f] [AELoss %f] [DLoss %f]' %
                         (time.time() - check, losses['ae_loss'].average, losses['d_loss'].average))

                    #torch.save(model.module.state_dict(), rootdir + f'model_last.pth')
                    #torch.save(criterion.module.state_dict(), rootdir + f'loss_last.pth')
                    torch.save(model.state_dict(), rootdir + f'model_last.pth')
                    torch.save(criterion.state_dict(), rootdir + f'loss_last.pth')
                    torch.save(opt.state_dict(), rootdir + f'opt.pth')
                    torch.save(d_opt.state_dict(), rootdir + f'd_opt.pth')
                    torch.save(scaler.state_dict(), rootdir + f'scaler.pth')
                    torch.save(scaler_d.state_dict(), rootdir + f'scaler_d.pth')

                losses = dict()
                losses['ae_loss'] = AverageMeter()
                losses['d_loss'] = AverageMeter()

            if it % 2000 == 0 and rank == 0:
                #torch.save(model.module.state_dict(), rootdir + f'model_{it}.pth')
                torch.save(model.state_dict(), rootdir + f'model_{it}.pth')

