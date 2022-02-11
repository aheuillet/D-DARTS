import os
import sys
import utils as dutils
import argparse
import torch.utils
from genotypes import Genotype_nested
import torchvision.datasets as dset
from model import NetworkCIFAR, NetworkImageNet
from thop import profile

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--data', type=str, default='/home/alexandre/work/dataset/cifar/', help='location of the data corpus')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--parse_method', type=str, default='threshold_sparse', help='experiment name')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--layers', type=int, default=8)
parser.add_argument('--init_channels', type=int, default=16)
parser.add_argument('--gpu', type=int, default=0, help='GPU on which to deserialize the model.')


if __name__ == '__main__':

    args = parser.parse_args()
    args.auto_aug = False
    args.cutout = False
    if "cifar" in args.dataset:
        train_transform, valid_transform = dutils._data_transforms_cifar(args)
        if args.dataset == "cifar10":
            classes = 10
            valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
        else:
            classes = 100
            valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == "imagenet":
        classes = 1000
        train_transform, valid_transform = dutils._data_transforms_imagenet(args)
        valid_data = dset.ImageNet(root=args.data, split='val', transform=valid_transform)
    else:
        raise("Unsupported dataset")
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    with open(f'genotypes/{args.arch}.txt', 'r') as g:
        genotype = eval(g.read())
    print('Parsing Genotypes: {}'.format(genotype))
    if args.dataset == "imagenet":
        model = NetworkImageNet(args.init_channels, classes, len(genotype.genes), False, genotype, args.parse_method)
    else:
        model = NetworkCIFAR(args.init_channels, classes, len(genotype.genes), False, genotype, args.parse_method)

    inputs = (torch.randn(1, 3, 224, 224),) if args.dataset == "imagenet" else (torch.randn(1, 3, 32, 32),)
    flops, params = profile(model, inputs=inputs, verbose=False)
    print('flops = %fM' % (flops / 1e6))
    print('param size = %fM' %( params / 1e6))

    model = model.cuda()

    if args.model_path and os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=f'cuda:{args.gpu}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print('The Pre-Trained Model Is InValid!')
        sys.exit(-1)

    top1 = dutils.AvgrageMeter()
    top5 = dutils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits, _ = model(input)
            prec1, prec5 = dutils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                print('valid %03d %f %f' % (step, top1.avg, top5.avg))
        print("Final Mean Top1: {}, Top5: {}".format(top1.avg, top5.avg))