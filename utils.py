import os
import numpy as np
import torch
import pandas as pd
import random
from tqdm import tqdm
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torch.autograd import Variable
from torchvision.transforms.transforms import Resize
from auto_augment import CIFAR10Policy, ImageNetPolicy
from typing import Any, Callable, Optional

class OpenImages(VisionDataset):

  def __init__(self, root: str, split: str, transforms: Optional[Callable], transform: Optional[Callable], target_transform: Optional[Callable]) -> None:
      super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
      assert(split in ['train', 'validation'])
      self.root = root
      self.split = split
      self.load_meta()

  def load_meta(self):
    df = pd.read_csv(self.split + "-annotations-bbox.csv")
    self.classes = df["LabelName"].unique()
    self.class_to_idx = {class_: i for i, class_ in enumerate(self.classes)}

    def process_classes(x):
      a = []
      print("Parsing OpenImages annotations...")
      for s in tqdm(x):
        i = self.class_to_idx[s]
        if i not in a:
          a.append(i)
      print("done")
      return a

    df = df[["ImageID", "LabelName"]]
    df = df.groupby("ImageID").agg(process_classes)
    self.targets = df.to_dict()
    self.images = list(self.targets.values())

class AvgrageMeter(object):
  """
    Keeps track of most recent, average, sum, and count of a metric.
  """

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  """Compute the top1 and top5 accuracy

  """
  maxk = max(topk)
  batch_size = target.size(0)

  # Return the k largest elements of the given input tensor
  # along a given dimension -> N * k
  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124] if args.dataset == 'cifar10' else [0.50707519, 0.48654887, 0.44091785]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768] if args.dataset == 'cifar10' else [0.26733428, 0.25643846, 0.27615049]

  normalize_transform = [
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]

  random_transform = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()]

  if args.auto_aug:
    random_transform += [CIFAR10Policy()]

  if args.cutout:
    cutout_transform = [Cutout(args.cutout_length)]
  else:
    cutout_transform = []

  train_transform = transforms.Compose(
      random_transform + normalize_transform + cutout_transform
  )

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_imagenet(args):
  IMAGENET_MEAN = [0.485, 0.456, 0.406]
  IMAGENET_STD = [0.229, 0.224, 0.225]

  normalize_transform = [
      transforms.ToTensor(),
      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

  random_transform = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),]

  if args.auto_aug:
    random_transform += [ImageNetPolicy()]

  train_transform = transforms.Compose(
      random_transform + normalize_transform
  )

  valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def calc_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    t, h = divmod(h, 24)
    return {'day': t, 'hour': h, 'minute': m, 'second': int(s)}

def parse(weights, operation_set,
           op_threshold, parse_method, steps):
  gene = []
  if parse_method == 'darts':
    n = 2
    start = 0
    for i in range(steps): # step = 4
      end = start + n
      W = weights[start:end].copy()
      edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
      for j in edges:
        k_best = None
        for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
        gene.append((operation_set[k_best], j)) # geno item : (operation, node idx)
      start = end
      n += 1
  elif 'threshold' in parse_method:
    n = 2
    start = 0
    for i in range(steps): # step = 4
      end = start + n
      W = weights[start:end].copy()
      if 'edge' in parse_method or 'complete' in parse_method:
        edges = list(range(i + 2))
      else: # select edges using darts methods
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

      for j in edges:
        if 'edge' in parse_method: # OP_{prob > T} AND |Edge| <= 2
          topM = sorted(enumerate(W[j]), key=lambda x: x[1])[-2:]
          for k, v in topM: # Get top M = 2 operations for one edge
            if W[j][k] >= op_threshold:
              gene.append((operation_set[k], i+2, j))
        elif 'complete' in parse_method: # OP_{prob > T}
          for k,_ in enumerate(W[j]):
            if W[j][k] >= op_threshold:
              gene.append((operation_set[k], i+2, j))
        elif 'sparse' in parse_method: # max( OP_{prob > T} ) and |Edge| <= 2
          k_best = None
          for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
          if W[j][k_best] >= op_threshold:
            gene.append((operation_set[k_best], i+2, j))
        else:
            raise NotImplementedError("Not support parse method: {}".format(parse_method))
      start = end
      n += 1
  return gene


from genotypes import Genotype_nested, PRIMITIVES, PRIMITIVES_DARTS, Genotype_opt
def parse_genotype(alphas, steps, multiplier, path = None,
                   parse_method='threshold_sparse', op_threshold=0.85, reductions=[], model_name='', dartopti=False):
  genes = []
  prims = PRIMITIVES if dartopti else PRIMITIVES_DARTS
  for i in range(alphas.shape[0]):
    genes.append(parse(alphas[i], prims, op_threshold, parse_method, steps))
  concat = range(2 + steps - multiplier, steps+2)
  genotype = Genotype_nested(genes=genes, concat=concat, reductions=reductions)

  if path is not None:
      if not os.path.exists(path):
          os.makedirs(path)
      print('Architecture parsing....\n', genotype)
      save_path = os.path.join(path, model_name + '_' + parse_method + '_' + str(op_threshold) + '.txt')
      with open(save_path, "w+") as f:
          f.write(str(genotype))
          print('Save in :', save_path)
  return genotype

import matplotlib.pyplot as plt
import json
def save_file(recoder, size = (14, 7), path='./'):
    fig, axs = plt.subplots(*size, figsize = (36, 98))
    num_ops = size[1]
    row = 0
    col = 0
    for (k, v) in recoder.items():
        axs[row, col].set_title(k)
        axs[row, col].plot(v, 'r+')
        if col == num_ops-1:
            col = 0
            row += 1
        else:
            col += 1
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, 'output.png'), bbox_inches='tight')
    plt.tight_layout()
    print('save history weight in {}'.format(os.path.join(path, 'output.png')))
    with open(os.path.join(path, 'history_weight.json'), 'w') as outf:
        json.dump(recoder, outf)
        print('save history weight in {}'.format(os.path.join(path, 'history_weight.json')))

def convert_genotype(genotype):
  if isinstance(genotype, Genotype_opt):
    genes = [genotype.genes[i] for i in genotype.seq]
    genotype = Genotype_nested(genes, genotype.concat, genotype.reductions)
  return genotype


if __name__ == "__main__":
  parse_genotype(np.load('cell_arch_weights.npy'), 4, 4, path='genotypes/') 