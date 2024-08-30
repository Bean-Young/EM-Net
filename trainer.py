import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from torch.utils.data import ConcatDataset
from networks.classify import AttentionModel
import torch.nn.functional as F

def trainer_ISIC2016(args, model, snapshot_path):
    from dataset.dataloader import Skin_dataset, RandomGenerator,CustomDataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = Skin_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    merged_dataset = []
    for i_batch, sampled_batch in enumerate(trainloader):
        for i in range(3):
            sampled_batch[i]['image'] = torch.squeeze(sampled_batch[i]['image'], 0)
            sampled_batch[i]['field'] = torch.squeeze(sampled_batch[i]['field'], 0)
            sampled_batch[i]['label'] = torch.squeeze(sampled_batch[i]['label'], 0)
            merged_dataset.append(sampled_batch[i])

    domains = ['ISIC','Waterloo']
    domain_datasets = {domain: [] for domain in domains}

    for item in merged_dataset:
        domain = item['domain'][0]
        if domain in domain_datasets:
            domain_datasets[domain].append(item)

    domain_custom_datasets = {domain: CustomDataset(data) for domain, data in domain_datasets.items()}
    domain_dataloaders = {domain: DataLoader(dataset, batch_size=batch_size, shuffle=True)
                          for domain, dataset in domain_custom_datasets.items()}

    train_class_dataset = ConcatDataset([domain_custom_datasets['ISIC'],domain_custom_datasets['Waterloo']])
    trainloader = DataLoader(train_class_dataset, batch_size=batch_size, shuffle=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    device = torch.device("cuda:0")
    weights = torch.tensor([1.0, 1.1], dtype=torch.float32, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=weights,reduction='none')
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    iter_num_class=0
    domain_classes=args.number_domain
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    cnn_model = AttentionModel(in_dim1=(args.batch_size, 2, 224, 224), in_dim2=(args.batch_size, 768, 14, 14),
                               n_doms=args.number_domain)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn_model = cnn_model.to(device)
    class_percentages = torch.zeros(domain_classes, device='cuda')
    for epoch_num in iterator:
        class_counts = torch.zeros(domain_classes, device='cuda')
        class_counts.fill_(0)
        if epoch_num<50 or  epoch_num>100:
            class_percentages = torch.zeros(domain_classes, device='cuda')
            weight_correct=1
        else:
            weight_correct=(domain_classes)/(domain_classes-1)
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, field_batch, label_batch,domain_batch = sampled_batch['image'], sampled_batch['field'], sampled_batch['label'],sampled_batch['domain']
            image_batch, field_batch,label_batch = image_batch.cuda(), field_batch.cuda(),label_batch.cuda()
            outputs,feature = model(image_batch,field_batch)
            feature = feature.permute(0, 2, 1).contiguous().view(feature.size(0), 768, 14, 14)
            cnn_model.train()
            outputs_class = cnn_model(outputs, feature)
            probabilities = F.softmax(outputs_class, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            one_hot_predictions = F.one_hot(predicted_classes, num_classes=probabilities.shape[1])
            class_percentages_for_batch = class_percentages.repeat(predicted_classes.shape[0], 1)
            image_weights = 1 - (one_hot_predictions* class_percentages_for_batch).sum(dim=1)
            image_weights = image_weights.clone()
            for i, domain in enumerate(domain_batch):
                if domain == 'Waterloo':
                    image_weights[i] = 1.0
            class_counts += one_hot_predictions.sum(dim=0)
            label_batch= torch.squeeze(label_batch, 1)
            expanded_weights = image_weights.view(-1, 1, 1).expand(-1, 224, 224)
            loss_ce = ce_loss(outputs, label_batch.long())
            weighted_loss_ce = loss_ce * expanded_weights
            loss_ce = weighted_loss_ce.mean()
            loss_dice = dice_loss(outputs, label_batch, softmax=True,sample_weight=image_weights)
            loss = 0.5 * loss_ce + 0.5*loss_dice
            loss=loss*weight_correct
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        class_percentages = class_counts / class_counts.sum()
        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
