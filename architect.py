import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from numpy.linalg import eigvals
from separate_loss import ConvAblationLoss

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class Architect(object):
  def __init__(self, model, args):

    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.hessian = None
    self.grads = None
    self.use_amp = args.amp
    self.use_mc = (args.cell_loss == "ablation_loss")
    self.report_freq = args.report_freq
    self.step_cnt = 0


  def step(self, input_valid, target_valid, scaler):
    list_loss2 = []
    list_loss3 = []
    if self.use_mc:
      mc = self.compute_marginal_contributions(input_valid, target_valid)
      mean_mc = np.mean(mc)
    else:
      mean_mc = 0
    with torch.cuda.amp.autocast(enabled=self.use_amp):
      logits = self.model(input_valid)
    count = 0
    for i,c in enumerate(self.model.cells):
      if c.criterion:
        with torch.cuda.amp.autocast(enabled=self.use_amp):
          if isinstance(c.criterion, ConvAblationLoss):
            cell_mc = mc[count]
            loss, loss1, loss2, loss3 = c.loss(target_valid, torch.sigmoid(c.alphas), logits, cell_mc, mean_mc)
            list_loss3.append(loss3)
          else:
            loss, loss1, loss2 = c.loss(target_valid, torch.sigmoid(c.alphas), logits, c, mean_mc)
        retain_graph = True if count < len(mc) else False
        scaler.scale(loss).backward(retain_graph=retain_graph)
        scaler.step(c.optimizer)
        c.optimizer.zero_grad(set_to_none=True)
        list_loss2.append(loss2)
        count += 1
    self.step_cnt += 1
    return loss1, np.mean(list_loss2), list_loss3

  def get_optimizers_states(self):
    state_dicts = []
    for c in self.model.cells:
      if c.criterion:
        state_dicts.append(c.optimizer.state_dict())
    return state_dicts
  
  def set_optimizers_states(self, state_dicts):
    for i,c in enumerate(self.model.cells):
      if c.criterion:
        c.optimizer.load_state_dict(state_dicts[i])
  
  def update_cell_schedulers(self):
    for c in self.model.cells:
      if c.criterion:
        c.scheduler.step()

  def zero_grads(self, parameters):
    for p in parameters:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    
  def compute_marginal_contributions(self, input, target):
      """
      Compute the marginal contribution of each cell in regard to all the other cells.
      """
      with torch.cuda.amp.autocast(enabled=self.use_amp):
        mc = []
        logits = self.model(input)
        v_cell = F.cross_entropy(logits, target).cpu().detach().item()
        for cell in self.model.cells:
          if cell.criterion:
            cell.activated = False
            logits = self.model(input)
            v_no_cell = F.cross_entropy(logits, target).cpu().detach().item()
            cell.activated = True
            mc.append(v_cell - v_no_cell)
        return mc
