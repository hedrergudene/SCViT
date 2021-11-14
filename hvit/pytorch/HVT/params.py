import argparse
import sys
import json

args = {}
args['batch-size'] = 64
args['epochs'] = 300

# Model parameters
args['model'] = 'deit_base_patch16_224'
args['input-size'] = 224

args['drop'] = 0.0
args['drop-path'] = 0.1

args['model-ema'] = True
args['no-model-ema'] = False
args['model-ema-decay'] = 0.99996
args['model-ema-force-cpu'] = False

# Optimizer parameters
args['opt'] = 'adamw'
args['opt-eps'] = 1e-8
args['opt-betas'] = None
args['clip-grad'] = None
args['momentum'] = 0.9
args['weight-decay'] = 0.05
# Learning rate schedule parameters
args['sched'] = 'cosine'
args['lr'] = 5e-4
args['lr-noise'] = None
args['lr-noise-pct'] = 0.67
args['lr-noise-std'] = 1.0
args['warmup-lr'] = 1e-6
args['min-lr'] = 1e-5

args['decay-epochs'] = 30
args['warmup-epochs'] = 5
args['cooldown-epochs'] = 10
args['patience-epochs'] = 10
args['decay-rate'] = 0.1

# Augmentation parameters
args['color-jitter'] = 0.4
args['aa'] = 'rand-m9-mstd0.5-inc1'
args['smoothing'] = 0.1
args['train-interpolation'] = 'bicubic'

args['repeated-aug'] = True
args['no-repeated-aug'] = False

# * Random Erase params
args['reprob'] = 0.25
args['remode'] = 'pixel'
args['recount'] = 1
args['resplit'] = False

# * Mixup params
args['mixup'] = 0.8
args['cutmix'] = 1.0
args['cutmix-minmax'] = None
args['mixup-prob'] = 1.0
args['mixup-switch-prob'] = 0.5
args['mixup-mode'] = 'batch'

# Distillation parameters
args['teacher-model'] = 'regnety_160'
args['teacher-path'] = ''
args['distillation-type'] = 'none'
args['distillation-alpha'] = 0.5
args['distillation-tau'] = 1.0

# * Finetuning params
args['finetune'] = ''

# Dataset parameters
args['data-path'] = '/datasets01/imagenet_full_size/061417/'
args['data-set'] = 'IMNET'
args['inat-category'] = 'name'

args['output_dir'] = ''
args['device'] = 'cuda'
args['seed'] = 0
args['resume'] =''
args['start_epoch'] = 0
args['eval'] = 'store_true'
args['dist-eval'] = 'store_true'
args['num_workers'] = 10
args['pin-mem'] = 'store_true'
args['no-pin-mem'] = 'store_false'

# distributed training parameters
args['world_size'] =1
args['dist_url'] ='env://'
args['exp_name'] ='deit'
args['config'] = None
args['patch_size'] = 16
args['num_heads'] = 3
args['head_dim'] = 64
args['num_blocks'] = 12
args['input_size'] = 224

args['pool_kernel_size'] = 3
args['pool_block_width'] = 4
args['pool_stride'] = 2