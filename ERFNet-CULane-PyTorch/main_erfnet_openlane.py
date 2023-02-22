import os
import os.path as ops
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import numpy as np
import models
# from models import sync_bn
import dataset.openlane_dataset as ds
from options.options import parser
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

best_mIoU = 0


def main():
    dataset_folder = '/home/liang/Datasets/OpenLane'
    train_file = dataset_folder + '/list_1000/training.txt'
    val_file = dataset_folder + '/list_1000/test.txt'
    global directory
    directory = 'predicts/openlane'
    if not os.path.exists(directory):
        os.makedirs(directory)
    global model_save_folder
    model_save_folder = 'trained_openlane'
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    global args, best_mIoU
    args = parser.parse_args()
    args.org_h = 1280
    args.org_w = 1920
    args.crop_y = 0
    args.resize_h = 400
    args.resize_w = 600
    args.vgg_mean = np.array([0.485, 0.456, 0.406])
    args.vgg_std = np.array([0.229, 0.224, 0.225])
    # args.vgg_mean = np.array([103.939, 116.779, 123.68])
    # args.vgg_std = np.array([1., 1., 1.])

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)
    args.evaluate = False
    # args.resume = model_save_folder + '/_erfnet_checkpoint_base.pth.tar'
    args.resume = False

    # if args.no_partialbn:
    #     sync_bn.Synchronize.init(args.gpus)

    args.dataset = 'sim3d'
    num_class = 23
    args.num_class = num_class-1
    ignore_label = 255

    args.start_epoch = 0
    args.epochs = 14
    args.print_freq = 1000
    args.batch_size = 8

    model = models.ERFNet(num_class, partial_bn=not args.no_partialbn)
    # input_mean = model.input_mean
    # input_std = model.input_std
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        ckpt_name = []
        cnt = 0
        for name, param in state_dict.items():
            if name not in list(own_state.keys()) or 'output_conv' in name:
                 ckpt_name.append(name)
                 # continue
            own_state[name].copy_(param)
            cnt += 1
        print('#reused param: {}'.format(cnt))
        return model

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model = load_my_state_dict(model, checkpoint['state_dict'])
            # torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code
    print('Open Lane dataset Loading')
    train_dataset = ds.LaneDataset(args, dataset_folder, train_file, data_aug=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)

    val_dataset = ds.LaneDataset(args, dataset_folder, val_file, data_aug=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, pin_memory=False)
    val_loader.is_testing = True

    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(num_class)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
    # criterion_exist = torch.nn.BCELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)
    log_dir = directory + '/tf_log'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    if args.evaluate:
        validate(val_loader, model, criterion, 0, evaluator, True)
        return

    for epoch in range(args.start_epoch, args.epochs): # args.start_epoch
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer)
        # break

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            mIoU = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), evaluator)
            # remember best mIoU and save checkpoint
            is_best = mIoU > best_mIoU
            best_mIoU = max(mIoU, best_mIoU)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_mIoU': best_mIoU,
            }, is_best)
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # losses_exist = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
        # sync_bn.convert_bn(model)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _, idx) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.contiguous()
        # print('i', input[0].shape)
        # print('t', target.shape)
        # # Convert tensor to numpy array
        # tar_array = target[0].permute(0, 1).numpy()
        # inp_array = input[0].permute(1, 2, 0).numpy()
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

        # # Show the first image in the left subplot
        # ax1.imshow(tar_array)
        # ax1.set_title('target')

        # # Show the second image in the right subplot
        # ax2.imshow(inp_array)
        # ax2.set_title('input')

        # # Display the figure
        # plt.show()
        # break
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var, no_lane_exist=True) # output_mid
        loss = criterion(torch.nn.functional.log_softmax(output, dim=1), target_var)
        # print(output_exist.data.cpu().numpy().shape)
        writer.add_scalar('training_loss/iteration', loss.item(), epoch * len(train_loader) + i)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, lr=optimizer.param_groups[-1]['lr'])))
            batch_time.reset()
            data_time.reset()
            losses.reset()


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def validate(val_loader, model, criterion, iter, evaluator, evaluate=False):

    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, img_name, idx) in enumerate(val_loader):
            target = target.cuda()
            input = input.contiguous()
            # input_var = torch.autograd.Variable(input, volatile=True)
            # target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input, no_lane_exist=True)
            loss = criterion(torch.nn.functional.log_softmax(output, dim=1), target)

            output = F.softmax(output, dim=1)
            pred = output.data.cpu().numpy()
            # save output visualization
            
            if evaluate:
                for cnt in range(len(idx)):
                    img_path = img_name[cnt].split('/')
                    img_file = img_path.pop()
                    save_dir = directory + '/output/' + ops.join(*img_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    prob_map = np.max(pred[cnt, 1:, :, :], axis=0)
                    prob_map = (prob_map * 255).astype(int)
                    # cv2.imwrite(directory + '/output/image_{}.png'.format(idx[cnt]), prob_map)
                    cv2.imwrite(ops.join(save_dir, img_file), prob_map)
            else:
                prob_map = np.max(pred[0, 1:, :, :], axis=0)
                prob_map = (prob_map * 255).astype(int)
                cv2.imwrite(directory + '/output/image_{}.png'.format(idx[0]), prob_map)
                # cv2.imwrite(ops.join(directory, str(img_name)), prob_map)

            # measure accuracy and record loss
            pred = pred.transpose(0, 2, 3, 1)
            pred = np.argmax(pred, axis=3).astype(np.uint8)
            IoU.update(evaluator(pred, target.cpu().numpy()))
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
                mIoU = np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum))
                mIoU = np.sum(mIoU) / len(mIoU)
                print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' 'Pixels Acc {acc:.3f}\t' 'mIoU {mIoU:.3f}'.format(i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc, mIoU=mIoU)))

    acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
    mIoU = np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum))
    mIoU = np.sum(mIoU) / len(mIoU)
    print(('Testing Results: Pixels Acc {acc:.3f}\tmIoU {mIoU:.3f} ({bestmIoU:.4f})\tLoss {loss.avg:.5f}'.format(acc=acc, mIoU=mIoU, bestmIoU=max(mIoU, best_mIoU), loss=losses)))

    return mIoU


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join(model_save_folder, '_'.join((args.snapshot_pref, args.method.lower(), filename)))
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(model_save_folder, '_'.join((args.snapshot_pref, args.method.lower(), 'model_best.pth.tar')))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = decay


if __name__ == '__main__':
    main()
