import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from torchvision import transforms
import network,loss,get_weight,utils
import lr_schedule, data_list
import copy,random
import tqdm
import os
import argparse

def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in tqdm.trange(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def get_features(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm.trange(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feats, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_feature = feats.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_feature = torch.cat((all_feature,feats.float().cpu()),0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return all_feature, all_label, all_output

def train(args):
    ## prepare data
    train_bs, test_bs = args.batch_size, args.batch_size * 2

    if args.sampler == "subset_sampler":
        source_base_dataset_train = data_list.ImageList(open(args.s_dset_path).readlines(),
                                                        transform=image_train(), root=args.root)
        source_base_dataset_test = data_list.ImageList(open(args.s_dset_path).readlines(),
                                                       transform=image_test(), root=args.root)

    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train(),return_index=True,root=args.root)
    dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train(),return_index=True,root=args.root)
    dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test(),root=args.root)
    dsets["source_val"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_test(),root=args.root)

    dset_loaders = {}
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=args.worker)
    dset_loaders["source_val"] = DataLoader(dsets["source_val"], batch_size=test_bs, shuffle=False, num_workers=args.worker)

    ##prepare model
    if "ResNet" in args.net:
        params = {"resnet_name": args.net, "bottleneck_dim": args.bottleneck_dim,
                  'class_num': args.class_num,"radius":args.radius,"normalize_classifier":args.normalize_classifier}
        base_network = network.ResNetFc(**params)

    base_network = base_network.cuda()
    advnet=network.AdversarialNetwork(base_network.output_num(),1024).cuda()
    parameter_list = base_network.get_parameters()

    ## set optimizer
    optimizer_config = {"type": torch.optim.SGD, "optim_params":
        {'lr': args.lr, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": True},
                        "lr_type": "inv", "lr_param": {"lr": args.lr, "gamma": args.gamma, "power": 0.75}
                        }
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ##training
    best_acc = 0
    for i in range(args.max_iterations + 1):
        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        ##test
        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            base_network.train(False)
            temp_acc = image_classification(dset_loaders, base_network)
            if best_acc < temp_acc:
                best_acc = temp_acc
                best_model = base_network.state_dict()
            log_str = "\n {} iter: {:05d}, precision: {:.5f}, best_acc: {:.5f} \n".format(args.name,i, temp_acc, best_acc)
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str)

        ##update weight, loader
        if args.sampler == "weighted_sampler":
            if args.dset == "domainnet" :
                args.seed = None
            if i % args.weight_update_interval == 0 and i>0:
                base_network.train(False)
                all_source_features, _, _ = get_features(dset_loaders["source_val"], base_network)
                all_target_features, _, _ = get_features(dset_loaders["test"], base_network)
                weights = get_weight.get_weight(all_source_features, all_target_features, args.rho0, args.seed,
                                                args.max_iter_discriminator, args.automatical_adjust, args.up,
                                                args.low, i,args.multiprocess,args.c)
                weights = torch.Tensor(weights[:])
                dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs,
                                                    sampler=WeightedRandomSampler(weights, num_samples=len(weights),
                                                                                  replacement=True),
                                                    num_workers=args.worker, drop_last=True)
        if args.sampler == "subset_sampler":
            if i % args.weight_update_interval == 0 and i > 5000:
                indexes = np.random.permutation(len(source_base_dataset_test))[:train_bs * 2000]
                dsets["source"] = data_list.SubDataset(source_base_dataset_train, indexes)
                dsets["source_val"] = data_list.SubDataset(source_base_dataset_test, indexes)
                dset_loaders["source_val"] = DataLoader(dsets["source_val"], batch_size=test_bs, shuffle=False,
                                                        num_workers=args.worker)
                base_network.train(False)
                all_source_features, _, _ = get_features(dset_loaders["source_val"], base_network)
                all_target_features, _, _ = get_features(dset_loaders["test"], base_network)
                weights = get_weight.get_weight(all_source_features, all_target_features)
                weights = torch.Tensor(weights[:])
                dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs,
                                                    sampler=WeightedRandomSampler(weights, num_samples=len(weights),
                                                                                  replacement=True),
                                                    num_workers=args.worker, drop_last=True)
        if args.sampler == "uniform_sampler":
            early_start = False
            if args.dset == "office" and i==200:
                early_start = True
            if i == 0:
                weights = torch.ones(len(dsets["source_val"]))
            elif i % args.weight_update_interval == 0 or early_start:
                base_network.train(False)
                all_source_features, _, _ = get_features(dset_loaders["source_val"], base_network)
                all_target_features, _, _ = get_features(dset_loaders["test"], base_network)
                weights = get_weight.get_weight(all_source_features, all_target_features, args.rho0, args.seed,
                                                args.max_iter_discriminator, args.automatical_adjust, args.up,
                                                args.low, i,args.multiprocess,args.c)
                weights = torch.Tensor(weights[:])

        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])

        ##forward
        inputs_source, labels_source,ids_source = iter_source.next()
        inputs_target, _,_ = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        _, outputs_source = base_network(inputs_source)
        features_target, _ = base_network(inputs_target)

        ##source (smoothed) cross entropy loss
        if args.label_smooth:
            if args.sampler == "weighted_sampler" or args.sampler == "subset_sampler":
                src_loss = loss.weighted_smooth_cross_entropy(outputs_source, labels_source)
            else:
                weight = weights[ids_source].cuda()
                src_loss = loss.weighted_smooth_cross_entropy(outputs_source, labels_source, weight)
        else:
            if args.sampler == "weighted_sampler" or args.sampler == "subset_sampler":
                src_loss = loss.weighted_cross_entropy(outputs_source,labels_source)
            else:
                weight = weights[ids_source].cuda()
                src_loss = loss.weighted_cross_entropy(outputs_source, labels_source, weight)

        ##target entropy loss
        fc = copy.deepcopy(base_network.fc)
        for param in fc.parameters():
            param.requires_grad = False
        softmax_tar_out = torch.nn.Softmax(dim=1)(fc(features_target))
        tar_loss = torch.mean(loss.entropy(softmax_tar_out))

        total_loss = src_loss
        if i>=args.start_adapt:
            total_loss = total_loss + args.ent_weight*tar_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print("step:{:d} \t src_loss:{:.4f} \t tar_loss:{:.4f}"
              "".format(i,src_loss.item(),tar_loss.item()))

    torch.save(best_model, os.path.join(args.output_dir, "best_model.pt"))

    log_str = 'Acc: ' + str(np.round(best_acc * 100, 2)) + '\n'
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    return best_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Adversarial Reweighting for Partial Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='run')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet50"])
    parser.add_argument('--dset', type=str, default='imagenet_caltech',
                        choices=["office", "office_home", "imagenet_caltech", "domainnet","visda-2017"])
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--ent_weight', type=float, default=0.1)
    parser.add_argument('--radius', type=float, default=20.0)
    parser.add_argument('--root', type=str, default='data',help="root to data")
    parser.add_argument('--label_smooth', action='store_true', default=False, help="whether to smooth label")

    args = parser.parse_args()
    args.start_adapt = 0
    args.normalize_classifier = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    args.rho0 = 5.0
    args.up = 5.0
    args.low = -5.0
    args.c = 1.2
    args.automatical_adjust = True
    args.max_iter_discriminator = 6000
    args.multiprocess = False
    args.gamma = 0.001

    if args.dset == 'domainnet':
        names = ['clipart', 'painting', 'real', 'sketch']
        k = 40
        args.class_num = 126
        args.max_iterations = 20000
        args.test_interval = 1000
        args.weight_update_interval = 1000
        args.lr = 1e-3
        args.radius = 20.0
        args.start_adapt = 1000
        args.sampler = "weighted_sampler"

    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        k = 25
        args.class_num = 65
        args.max_iterations = 5000
        args.test_interval = 500
        args.weight_update_interval = 500
        args.max_iter_discriminator = 3000
        args.lr = 1e-3
        args.radius = 10.0
        args.sampler = "uniform_sampler"


    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        k = 10
        args.class_num = 31
        args.max_iterations = 3000
        args.test_interval = 200
        args.weight_update_interval = 500
        args.max_iter_discriminator = 3000
        if args.s == 0 and args.t == 1:
            args.lr = 3e-4
            args.start_adapt = 1000
            args.rho0 = 10.0
        if args.s == 0 and args.t == 2:
            args.lr = 3e-4
            args.start_adapt = 1000
            args.rho0 = 3.0
        args.radius = 8.5
        args.sampler = "uniform_sampler"


    if args.dset == 'imagenet_caltech':
        names = ['imagenet', 'caltech']
        k = 84
        if args.s == 1:
            args.class_num = 256
            args.max_iterations = 40000
            args.test_interval = 1000
            args.weight_update_interval = 1000
            args.lr = 7e-4
            args.sampler = "weighted_sampler"
        else:
            args.class_num = 1000
            args.max_iterations = 100000
            args.test_interval = 1000
            args.weight_update_interval = 2000
            args.lr = 1e-3
            args.sampler = "subset_sampler"
            args.gamma = 0.0004

        args.radius = 20.0

    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        k = 6
        args.class_num = 12
        args.max_iterations = 40000
        args.test_interval = 1000
        if args.s == 0:
            args.weight_update_interval = 3000
            args.lr = 1e-3
            args.sampler = "weighted_sampler"
            args.normalize_classifier = False
        else:
            args.weight_update_interval = 1000
            args.lr = 1e-4
            args.sampler = "uniform_sampler"
            args.low = 1.0
            args.multiprocess = True
        args.radius = 5.0
    args.bottleneck_dim = utils.recommended_bottleneck_dim(args.class_num)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    data_folder = './data/'
    args.s_dset_path = data_folder + args.dset + '/' + names[args.s] + '.txt'
    args.t_dset_path = data_folder + args.dset + '/' + names[args.t] + '_' + str(k) + '.txt'

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    args.output_dir = os.path.join('ckp/', args.dset, args.name, args.output)

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    args.out_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file.write(str(args) + '\n')
    args.out_file.flush()

    train(args)

