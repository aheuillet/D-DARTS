import os
import sys
import time
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
from model_search import NetworkImageNet, NetworkCIFAR
from architect import Architect
from separate_loss import ConvSeparateLoss, TriSeparateLoss, ConvAblationLoss
from architecture_processing import hausdorff_metric

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default=os.path.join(os.path.expanduser('~'),'work/dataset/cifar/'), help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or cifar 100 for searching')
parser.add_argument('--arch_baseline', type=str, default=None, help='start search from existing baseline')
parser.add_argument('--batch_size', type=int, default=72, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--steps', type=int, default=4, help="Number of max steps in each cell")
parser.add_argument('--multiplier', type=int, default=4, help="Channel mutiplier applied at the output of every cell")
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--aux_loss_weight', type=float, default=9.0, help='weight decay')
parser.add_argument('--abl_loss_weight', type=float, default=0.7, help='abl loss weight')
parser.add_argument('--gpus', nargs='+', default=[0], type=int, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--pretrain_epochs', type=int, default=5, help='num of pre-training epochs for baseline arch')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--single_level', action='store_true', default=False, help='use single level')
parser.add_argument('--sep_loss', type=str, default='l2', help='loss to use to separate weight value')
parser.add_argument('--cell_loss', type=str, default='ablation_loss', help='loss to use for cell nested network training')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--parse_method', type=str, default='threshold_sparse', help='parse the code method')
parser.add_argument('--op_threshold', type=float, default=0.85, help='threshold for edges')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_lr_gamma', type=float, default=0.9, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--amp', action='store_true', default=False, help="train using Automatic Mixed Precision.")
parser.add_argument('--no_arch_metric', action='store_true', default=False, help="do not log the architectural metric between starting arch and current arc per epoch")
args = parser.parse_args()

args.save = './logs/search/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

if args.dataset == "cifar10":
  CLASSES = 10
elif args.dataset == "cifar100":
  CLASSES = 100
else:
  CLASSES = 1000

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  utils.create_exp_dir(args.save, scripts_to_save=None)

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)
  writer = SummaryWriter(f'./runs/search/search-{args.dataset}-{args.cell_loss}-abl{args.abl_loss_weight}-{time.strftime("%Y%m%d-%H%M%S")}')

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpus[0])
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpus[0])
  logging.info("args = %s", args)
  run_start = time.time()
  start_epoch = 0
  dur_time = 0

  if args.arch_baseline:
    logging.info(f"Starting search from baseline {args.arch_baseline}")
  
  scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

  if args.sep_loss == 'l2':
    criterion_train = ConvSeparateLoss(weight=args.aux_loss_weight)  
  else: 
    criterion_train = TriSeparateLoss(weight=args.aux_loss_weight)
  criterion_val = nn.CrossEntropyLoss()

  if args.cell_loss == "ablation_loss":
    criterion_block = ConvAblationLoss(weight=args.aux_loss_weight, abl_weight=args.abl_loss_weight)  
  else: 
    criterion_block = criterion_train 

  if 'cifar' in args.dataset:
    model = NetworkCIFAR(args.init_channels, CLASSES, args.layers, criterion_train, criterion_block, args.arch_learning_rate, args.arch_weight_decay,
                  steps=args.steps, multiplier=4, stem_multiplier=3,
                  arch_baseline=args.arch_baseline, op_threshold=args.op_threshold, epochs=args.epochs)
  else:
    model = NetworkImageNet(args.init_channels, CLASSES, args.layers, criterion_train, criterion_block, args.arch_learning_rate, args.arch_weight_decay,
                  steps=args.steps, multiplier=4, stem_multiplier=3,
                  arch_baseline=args.arch_baseline, op_threshold=args.op_threshold, epochs=args.epochs)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  model_optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  arch_optimizer = None

  train_transform, valid_transform = utils._data_transforms_cifar(args)
  if args.dataset == "cifar10":
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  elif args.dataset == "cifar100":
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  elif args.dataset == "imagenet":
    train_transform, valid_transform = utils._data_transforms_imagenet(args)
    train_data = dset.ImageNet(root=args.data, train=True, download=True, transform=train_transform)
  else:
    raise("Unsupported dataset")

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=4)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=4)

  architect = Architect(model, args)

  metrics = []

  # resume from checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      logging.info("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch']
      dur_time = checkpoint['dur_time']
      model_optimizer.load_state_dict(checkpoint['model_optimizer'])
      scaler.load_state_dict(checkpoint['model_scaler'])
      architect.set_optimizers_states(checkpoint['arch_optimizers'])
      model.restore(checkpoint['network_states'])
      logging.info('=> loaded checkpoint \'{}\'(epoch {})'.format(args.resume, start_epoch))
    else:
      logging.info('=> no checkpoint found at \'{}\''.format(args.resume))
  
  search = True
  if args.arch_baseline:
    args.epochs += args.pretrain_epochs
    logging.info(f"Pretraining baseline architecture for {args.pretrain_epochs} epochs...")
    search = False

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=-1 if start_epoch == 0 else start_epoch)
  if args.resume and os.path.isfile(args.resume):
    scheduler.load_state_dict(checkpoint['scheduler'])

  for epoch in range(start_epoch, args.epochs):
    if epoch == args.pretrain_epochs and args.arch_baseline:
      logging.info(f"Pretraining done.")
      search = True

    lr = scheduler.get_last_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    # training and search the model
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion_train, model_optimizer, arch_optimizer, scaler, args.amp, search=search)
    logging.info('train_acc %f', train_acc)

    # validation the model
    valid_acc, valid_obj = infer(valid_queue, model, criterion_val)
    logging.info('valid_acc %f', valid_acc)

    scheduler.step()
    architect.update_cell_schedulers()

    # save checkpoint
    utils.save_checkpoint({
      'epoch': epoch + 1,
      'dur_time': dur_time + time.time() - run_start,
      'scheduler': scheduler.state_dict(),
      'model_optimizer': model_optimizer.state_dict(),
      'model_scaler': scaler.state_dict(),
      'arch_optimizers': architect.get_optimizers_states(),
      'network_states': model.states(),
    }, is_best=False, save=args.save)

    if model.arch_baseline and not args.no_arch_metric and epoch >= args.pretrain_epochs + 5:
      current_arch = utils.parse_genotype(torch.sigmoid(model.arch_parameters()).data.cpu().numpy(), model._steps, model._multiplier, parse_method="threshold_edge", dartopti=True)
      metric = hausdorff_metric(model.arch_baseline, current_arch)
      writer.add_scalar('distance_metric', metric, global_step=epoch)
      logging.info(f'architectural distance metric with baseline: {metric}')
      if len(metrics) >= 4:
        if abs(metric - metrics[-1]) <= 0.001 and abs(metric - metrics[-2]) <= 0.001 and abs(metric - metrics[-3]) <= 0.001 and abs(metric - metrics[-4]) <= 0.001:
          logging.info('Architectural metric has not changed for the last 4 epochs. Stopping search process now.')
          break
      metrics.append(metric)


    writer.add_scalar('train_acc', train_acc, global_step=epoch)
    writer.add_scalar('valid_acc', valid_acc, global_step=epoch)
    writer.add_scalar('lr', lr, global_step=epoch)
    writer.add_scalar('global loss', train_obj, global_step=epoch)

    duration_time = utils.calc_time(dur_time + time.time() - run_start)

    logging.info('save checkpoint (epoch %d) in %s  dur_time: %s', epoch, args.save, duration_time)

  # save last operations
  prob_dist = torch.sigmoid(model.arch_parameters(to_parse=True)).data.cpu().numpy() 
  np.save(os.path.join(os.path.join(args.save, 'arch_weight.npy')), prob_dist)
  logging.info('save last weights done')
  model_name = f"{args.arch_baseline}_{args.dataset}" if args.arch_baseline else f"{args.dataset}"
  utils.parse_genotype(prob_dist, model._steps, model._multiplier, path='genotypes/', parse_method='threshold_sparse', reductions=model.reduction_indexes, model_name=model_name, dartopti=bool(model.arch_baseline))
  utils.parse_genotype(prob_dist, model._steps, model._multiplier, path='genotypes/', parse_method='threshold_edge', reductions=model.reduction_indexes, model_name=model_name, dartopti=bool(model.arch_baseline))
  logging.info('save genotypes done')
  writer.close()
  return {'type': "Search", 'val_acc': valid_acc, "arch_baseline": args.arch_baseline, "dataset": args.dataset, "duration_time": duration_time}


def train(train_queue, valid_queue, model, architect, criterion, model_optimizer, arch_optimizer, scaler, use_amp=True, search=True):
  objs = utils.AvgrageMeter()
  objs1 = utils.AvgrageMeter()
  objs2 = utils.AvgrageMeter()
  #objs3 = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  model.train()

  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    list_loss3 = []
    # Get a random minibatch from the search queue(validation set) with replacement
    # TODO: next is too slow
    if not args.single_level and search:
      input_search, target_search = next(iter(valid_queue))
      input_search = input_search.cuda(non_blocking=True)
      target_search = target_search.cuda(non_blocking=True)
      # Bi-level default
      loss1, loss2, list_loss3 = architect.step(input_search, target_search, scaler)

    with torch.cuda.amp.autocast(enabled=use_amp):
      logits = model(input)
      aux_input = torch.sigmoid(model.arch_parameters())
      if not args.single_level:
        loss, _, _ = criterion(logits, target, aux_input, cell_mc=None, mean_mc=None)
      else:
        loss, loss1, loss2 = criterion(logits, target, aux_input, cell_mc=None, mean_mc=None)
    
    scaler.scale(loss).backward()
    scaler.unscale_(model_optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    # Update the network parameters
    scaler.step(model_optimizer)

    ## if single level
    if args.single_level:
      arch_optimizer.step()
    
    scaler.update()

    model_optimizer.zero_grad(set_to_none=True)

    ## if single-level
    if args.single_level:
      arch_optimizer.zero_grad()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.detach().item(), n)
    if search:
      objs1.update(loss1, n)
      objs2.update(loss2, n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d loss: %e top1: %f top5: %f', step, objs.avg, top1.avg, top5.avg)
      if search:
        logging.info('val cls_loss %e; spe_loss %e', objs1.avg, objs2.avg)
        logging.info(f'ablation_loss_list: {list_loss3}')

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, use_amp=True):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)

      with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(input)
        loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

if __name__ == '__main__':
  main()
