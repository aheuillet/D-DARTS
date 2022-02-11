from random import random
from traceback import print_tb
import torch
import torch.nn as nn
import numpy as np
import genotypes
from operations import *
from genotypes import PRIMITIVES, PRIMITIVES_DARTS, Genotype_nested, Genotype_opt
from architecture_processing import deserialize_architecture_to_alphas

class MixedOp(nn.Module):
  def __init__(self, C, stride, prims):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.stride = stride
    for primitive in prims:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights, cell_active):
    if cell_active:
      return sum(w * op(x) for w, op in zip(weights, self._ops))
    else:
      return sum(torch.tensor(random()).cuda() * op(x) for op in self._ops)
      

class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, lr, criterion, weight_decay, epochs, weights, dartopti):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.activated = True

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()

    prims = PRIMITIVES if dartopti else PRIMITIVES_DARTS

    if weights is not None:
      self.alphas = torch.tensor(weights, requires_grad=True, dtype=torch.float32, device="cuda")
    elif criterion is not None:
      k = sum(1 for i in range(self._steps) for n in range(2+i))
      self.alphas = torch.zeros(k, len(prims), requires_grad=True, device="cuda")

    self.criterion = criterion
    if criterion:
      self.optimizer = torch.optim.Adam([self.alphas],
        lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(epochs), eta_min=0.001)

    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, prims)
        self._ops.append(op)


  def loss(self, target1, input2, logits, cell, marginal_contributions):
    return self.criterion(logits, target1, input2, cell, marginal_contributions)

  def forward(self, s0, s1, weights=None):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    if weights is None:
      weights = self.alphas
    weights = torch.sigmoid(weights)

    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j], self.activated) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, criterion_cell, cell_optim_lr, cell_optim_weight_decay, steps=4, multiplier=4, stem_multiplier=3,
                arch_baseline=None, op_threshold=None, epochs=50):
    super(NetworkCIFAR, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.op_threshold = op_threshold
    self.arch_baseline = arch_baseline

    self.reduction_indexes = [self._layers//3, 2*self._layers//3]

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    self.cells = nn.ModuleList()

    baseline_alphas = None
    if arch_baseline:
      self.arch_baseline = eval(f"genotypes.{arch_baseline}")
      self._steps = self.arch_baseline.concat[-1] - 1
      self._layers = len(self.arch_baseline.seq)
      self.reduction_indexes = self.arch_baseline.reductions
      baseline_alphas = deserialize_architecture_to_alphas(self.arch_baseline)
      C_prev = self.init_cells_baseline(C_curr, baseline_alphas, epochs, cell_optim_lr, criterion_cell, cell_optim_weight_decay)
    else:
      C_prev = self.init_cells(C_curr, epochs, cell_optim_lr, criterion_cell, cell_optim_weight_decay)


    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
  

  def init_cells(self, C_curr, epochs, cell_optim_lr, criterion_cell, cell_optim_weight_decay):
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
    reduction_prev = False
    for i in range(self._layers):
      if i in self.reduction_indexes:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(self._steps, self._multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, cell_optim_lr, criterion_cell, cell_optim_weight_decay, epochs, None, dartopti=False)
      reduction_prev = reduction
      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, self._multiplier*C_curr
    return C_prev

  def init_cells_baseline(self, C_curr, baseline_alphas, epochs, cell_optim_lr, criterion_cell, cell_optim_weight_decay):
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
    reduction_prev = False
    seen = []
    self.cell_dict = {c: None for c in range(max(self.arch_baseline.seq))}
    for i in range(self._layers):
      if i in self.reduction_indexes:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      index = self.arch_baseline.seq[i]
      if index not in seen:
        crit = criterion_cell
        weights = baseline_alphas[index]
        self.cell_dict[index] = i
        seen.append(index)
      else:
        crit = None
        weights = None
      cell = Cell(self._steps, self._multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, cell_optim_lr, crit, cell_optim_weight_decay, epochs, weights, dartopti=True)
      reduction_prev = reduction
      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, self._multiplier*C_curr
    return C_prev

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      weights = self.cells[self.cell_dict[self.arch_baseline.seq[i]]].alphas if cell.criterion is None else None
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def arch_parameters(self, to_parse=False):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    prims = PRIMITIVES if self.arch_baseline else PRIMITIVES_DARTS
    num_ops = len(prims)
    length = len(self.cells) if to_parse or not(self.arch_baseline) else len(self.arch_baseline.genes)
    alphas = torch.zeros(length, k, num_ops, device='cuda')
    index = 0
    for i,c in enumerate(self.cells):
      if c.criterion:
        alphas[index] = c.alphas
        index += 1
      elif to_parse: # Output alphas so that we can parse the genotype like a standard one (Genotype_nested)
        alphas[index] = self.cells[self.cell_dict[self.arch_baseline.seq[i]]].alphas
        index += 1 
    return alphas
  
  def arch_state_dicts(self):
    state_dicts = []
    for i,c in enumerate(self.cells):
      state_dicts.append(c.state_dict())
    return state_dicts
  
  def arch_criterions(self):
    criterions = []
    for i,c in enumerate(self.cells):
      criterions.append(c.criterion)
    return criterions

  def states(self):
    return {
      'alphas': self.arch_parameters(),
      'arch_state_dicts': self.arch_state_dicts(),
      'arch_criterions': self.arch_criterions(),
      'criterion': self._criterion,
      'network_state_dict': self.state_dict()
    }

  def restore(self, states):
    self.load_state_dict(states['network_state_dict'])
    self._criterion = states['criterion']
    for i,c in enumerate(self.cells):
      c.load_state_dict(states['arch_state_dicts'][i])
      c.alphas = states['alphas'][i]
      c.alphas.requires_grad_()
      c.criterion = states['arch_criterions'][i]


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, criterion_cell, cell_optim_lr, cell_optim_weight_decay, steps=4, multiplier=4, stem_multiplier=3,
                arch_baseline=None, op_threshold=None, epochs=50):
    super(NetworkImageNet, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.op_threshold = op_threshold
    self.arch_baseline = arch_baseline

    self.reduction_indexes = [self._layers//3, 2*self._layers//3]

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    self.cells = nn.ModuleList()

    baseline_alphas = None
    if arch_baseline:
      self.arch_baseline = eval(f"genotypes.{arch_baseline}")
      self._steps = self.arch_baseline.concat[-1] - 1
      self._layers = len(self.arch_baseline.seq)
      self.reduction_indexes = self.arch_baseline.reductions
      baseline_alphas = deserialize_architecture_to_alphas(self.arch_baseline)
      C_prev = self.init_cells_baseline(C_curr, baseline_alphas, epochs, cell_optim_lr, criterion_cell, cell_optim_weight_decay)
    else:
      C_prev = self.init_cells(C_curr, epochs, cell_optim_lr, cell_optim_weight_decay)


    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
  

  def init_cells(self, C_curr, epochs, cell_optim_lr, criterion_cell, cell_optim_weight_decay):
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
    reduction_prev = True
    for i in range(self._layers):
      if i in self.reduction_indexes:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(self._steps, self._multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, cell_optim_lr, criterion_cell, cell_optim_weight_decay, epochs, None)
      reduction_prev = reduction
      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, self._multiplier*C_curr
    return C_prev

  def init_cells_baseline(self, C_curr, baseline_alphas, epochs, cell_optim_lr, criterion_cell, cell_optim_weight_decay):
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
    reduction_prev = True
    seen = []
    self.cell_dict = {c: None for c in range(max(self.arch_baseline.seq))}
    for i in range(self._layers):
      if i in self.reduction_indexes:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      index = self.arch_baseline.seq[i]
      if index not in seen:
        crit = criterion_cell
        weights = baseline_alphas[index]
        self.cell_dict[index] = i
        seen.append(index)
      else:
        crit = None
        weights = None
      cell = Cell(self._steps, self._multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, cell_optim_lr, crit, cell_optim_weight_decay, epochs, weights)
      reduction_prev = reduction
      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, self._multiplier*C_curr
    return C_prev

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      weights = self.cells[self.cell_dict[self.arch_baseline.seq[i]]].alphas if cell.criterion is None else None
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def arch_parameters(self, to_parse=False):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    prims = PRIMITIVES if self.arch_baseline else PRIMITIVES_DARTS
    num_ops = len(prims)
    length = len(self.cells) if to_parse else len(self.arch_baseline.genes)
    alphas = torch.zeros(length, k, num_ops, device='cuda')
    index = 0
    for i,c in enumerate(self.cells):
      if c.criterion:
        alphas[index] = c.alphas
        index += 1
      elif to_parse: # Output alphas so that we can parse the genotype like a standard one (Genotype_nested)
        alphas[index] = self.cells[self.cell_dict[self.arch_baseline.seq[i]]].alphas
        index += 1 
    return alphas
  
  def arch_state_dicts(self):
    state_dicts = []
    for i,c in enumerate(self.cells):
      state_dicts.append(c.state_dict())
    return state_dicts
  
  def arch_criterions(self):
    criterions = []
    for i,c in enumerate(self.cells):
      criterions.append(c.criterion)
    return criterions

  def states(self):
    return {
      'alphas': self.arch_parameters(),
      'arch_state_dicts': self.arch_state_dicts(),
      'arch_criterions': self.arch_criterions(),
      'criterion': self._criterion,
      'network_state_dict': self.state_dict()
    }

  def restore(self, states):
    self.load_state_dict(states['network_state_dict'])
    self._criterion = states['criterion']
    for i,c in enumerate(self.cells):
      c.load_state_dict(states['arch_state_dicts'][i])
      c.alphas = states['alphas'][i]
      c.alphas.requires_grad_()
      c.criterion = states['arch_criterions'][i]