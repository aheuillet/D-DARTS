import torch
import torch.nn as nn
from operations import *
from utils import drop_path, convert_genotype

class Cell(nn.Module):
  def __init__(self, genotype, gene_index, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

    if reduction:
      op_names, indices = zip(*genotype.genes[gene_index])
      concat = genotype.concat
    else:
      op_names, indices = zip(*genotype.genes[gene_index])
      concat = genotype.concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1) # N，C，H, W


class DCOCell(nn.Module):
  def __init__(self, gene, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(DCOCell, self).__init__()

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

    op_names, tos, froms = zip(*gene)
    self.inplanes = C_prev
    self._compile(C, op_names, tos, froms, reduction)

  def _compile(self, C, op_names, tos, froms, reduction):
    self._ops = nn.ModuleDict()
    for name_i, to_i, from_i in zip(op_names, tos, froms):
      stride = 2 if reduction and from_i < 2 else 1
      op = OPS[name_i](C, stride, True)
      if str(to_i) in self._ops.keys():
        if str(from_i) in self._ops[str(to_i)]:
          self._ops[str(to_i)][str(from_i)] += [op]
        else:
          self._ops[str(to_i)][str(from_i)] = nn.ModuleList()
          self._ops[str(to_i)][str(from_i)] += [op]
      else:
        self._ops[str(to_i)] = nn.ModuleDict()
        self._ops[str(to_i)][str(from_i)] = nn.ModuleList()
        self._ops[str(to_i)][str(from_i)] += [op]

    self.multiplier = len(self._ops)

  def forward(self, s0, s1, drop_prob=0):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = {}
    states['0'] = s0
    states['1'] = s1

    for to_i, ops in self._ops.items():
      h = []
      for from_i, op_i in ops.items():
        if from_i not in states:
          continue
        h += [sum([op(states[from_i]) for op in op_i if from_i in states])]
      out = sum(h)
      if self.training and drop_prob > 0:
        out = drop_path(out, drop_prob)
      states[to_i] = out

    return torch.cat([v for v in states.values()][2:], dim=1)

class AuxiliaryHeadCIFAR(nn.Module):
  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0), -1))
    return x

class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, parse_method='darts'):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self.drop_path_prob = 0.2

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

    genotype = convert_genotype(genotype)

    self.cells = nn.ModuleList()
    reduction_prev = False
    self.reduction_indexes = list(genotype.reductions)
    for i in range(layers):
      if i in self.reduction_indexes:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
     
      if parse_method == 'darts':
        cell = Cell(genotype, i, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      else:
        if layers > len(genotype.genes):
          if len(self.reduction_indexes) > 2:
            raise NotImplementedError
          if i < layers//3:
            n = i % len(genotype.genes)//3
          elif i == layers//3:
            n = len(genotype.genes)//3
          elif i > layers//3 and i < 2*layers//3:
            n = i % (2*len(genotype.genes)//3 - 1 - len(genotype.genes)//3) + len(genotype.genes)//3 + 1
          elif i == 2*layers//3:
            n = 2*len(genotype.genes)//3
          else:
            n = i % (len(genotype.genes) - 1 - 2*len(genotype.genes)//3) + 2*len(genotype.genes)//3 + 1
        else:
          n = i
        assert(n < len(genotype.genes))
        cell = DCOCell(genotype.genes[n], C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == self.reduction_indexes[-1]:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == self.reduction_indexes[-1]:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, parse_method='threshold', include_classifier=True):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self.drop_path_prob = 0.2
    self.include_classifier = include_classifier

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
  )

    C_prev_prev, C_prev, C_curr = C, C, C

    genotype = convert_genotype(genotype)

    self.cells = nn.ModuleList()
    reduction_prev = True
    self.reduction_indexes = list(genotype.reductions)
    for i in range(layers):
      if i in self.reduction_indexes:
        C_curr *= 2
        reduction = True
      else:
        reduction = False

      if parse_method == 'darts':
        cell = Cell(genotype, i, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      else:
        if layers > len(genotype.genes):
          if len(self.reduction_indexes) > 2:
            raise NotImplementedError
          if i < layers//3:
            n = i % len(genotype.genes)//3
          elif i == layers//3:
            n = len(genotype.genes)//3
          elif i > layers//3 and i < 2*layers//3:
            n = i % (2*len(genotype.genes)//3 - 1 - len(genotype.genes)//3) + len(genotype.genes)//3 + 1
          elif i == 2*layers//3:
            n = 2*len(genotype.genes)//3
          else:
            n = i % (len(genotype.genes) - 1 - 2*len(genotype.genes)//3) + 2*len(genotype.genes)//3 + 1
        else:
          n = i
        assert(n < len(genotype.genes))
        cell = DCOCell(genotype.genes[n], C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == self.reduction_indexes[-1]:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)
  
  def get_in_channels_list(self, layers):
    in_channels = []
    for name, c in self.named_modules():
      if name in layers:
        in_channels.append(c.inplanes)
    return in_channels

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == self.reduction_indexes[-1]:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    if self.include_classifier:
      logits = self.classifier(out.view(out.size(0), -1))
      return logits, logits_aux
    else:
      return out
