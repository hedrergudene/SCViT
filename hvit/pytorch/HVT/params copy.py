import argparse
import sys
import json

parser = {}
parser['batch-size'] = 64
parser['epochs'] = 300

# Model parameters
parser['model'] = 'deit_base_patch16_224'
parser['input-size'] = 224

parser['drop'] = 0.0
parser['drop-path'] = 0.1

parser['model-ema'] = True
parser['no-model-ema'] = False
parser['model-ema-decay'] = 0.99996
parser['model-ema-force-cpu'] = False

# Optimizer parameters
parser['opt'] = 'adamw'
parser['opt-eps'] = 1e-8
parser['opt-betas'] = None
parser['clip-grad'] = None
parser['momentum'] = 0.9
parser['weight-decay'] = 0.05
# Learning rate schedule parameters
parser['sched'] = 'cosine'
parser['lr'] = 5e-4
parser['lr-noise'] = None
parser['lr-noise-pct'] = 0.67
parser['lr-noise-std'] = 1.0
parser['warmup-lr'] = 1e-6
parser['min-lr'] = 1e-5

parser['decay-epochs'] = 30
parser['warmup-epochs'] = 5
parser['cooldown-epochs'] = 10
parser['patience-epochs'] = 10
parser['decay-rate'] = 0.1

# Augmentation parameters
parser['color-jitter'] = 0.4
parser['aa'] = 'rand-m9-mstd0.5-inc1'
parser['smoothing'] = 0.1
parser['train-interpolation'] = 'bicubic'

parser['repeated-aug'] = True
parser['no-repeated-aug'] = False

# * Random Erase params
parser['reprob'] = 0.25
parser['remode'] = 'pixel'
parser['recount'] = 1
parser['resplit'] = False

# * Mixup params
parser['mixup'] = 0.8
parser['cutmix'] = 1.0
parser['cutmix-minmax'] = None
parser['mixup-prob'] = 1.0
parser['mixup-switch-prob'] = 0.5
parser['mixup-mode'] = 'batch'

# Distillation parameters
parser['teacher-model'] = 'regnety_160'
parser['teacher-path'] = ''
parser['distillation-type'] = 'none'
parser['distillation-alpha'] = 0.5
parser['distillation-tau'] = 1.0

# * Finetuning params
parser['finetune'] = ''

# Dataset parameters
parser['data-path'] = '/datasets01/imagenet_full_size/061417/'
parser['data-set'] = 'IMNET'
parser['inat-category'] = 'name'

parser['output_dir'] = ''
parser['device'] = 'cuda'
parser['seed'] = 0
parser['resume'] =''
parser['start_epoch'] =0
parser['eval'] = 'store_true'
parser['dist-eval'] = 'store_true'
parser['num_workers'] = 10
parser['pin-mem'] = 'store_true'
parser['no-pin-mem'] = 'store_false'

# distributed training parameters
parser['world_size'] =1
parser['dist_url'] ='env://'
parser['exp_name'] ='deit'
parser['config'] =None
parser['patch_size'] = 16
parser['num_heads'] = 3
parser['head_dim'] = 64
parser['num_blocks'] = 12
parser['input_size'] = 224

parser['pool_kernel_size'] = 3
parser['pool_block_width'] = 4
parser['pool_stride'] = 2