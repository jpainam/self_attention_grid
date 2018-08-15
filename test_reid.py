import torch
import numpy as np
from torchvision import datasets, models, transforms
import argparse
import os
from image_folder_loader import ImageFolderLoader
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model_path', default='model_best', type=str, help='Model path')
parser.add_argument('--test_dir', default='/home/paul/datasets/market1501/pytorch', type=str, help='./test_data')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--multi', action='store_true', help='use multiple query')
opt = parser.parse_args()

test_dir = opt.test_dir

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()
    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, data in enumerate(queryloader):
            imgs1, imgs2, labels, camids = data
            if use_gpu:
                imgs1, imgs2 = imgs1.cuda(), imgs2.cuda()
            features = model(imgs1)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(labels)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, data in enumerate(galleryloader):
            imgs1, imgs2, labels, camids = data
            if use_gpu:
                imgs1, imgs2 = imgs1.cuda(), imgs2.cuda()
            features = model(imgs1)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(labels)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")

    return cmc[0]


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

# VGG-16 Takes 224x224 images as input, so we resize all of them

num_class = 751
data_transforms_1 = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
])
data_transforms_2 = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
])

image_datasets = {x: ImageFolderLoader(os.path.join(test_dir, x),
                                       transform_1=data_transforms_1, transform_2=data_transforms_2)
                  for x in ['gallery', 'query', 'multi-query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=opt.batchsize,
                                              shuffle=False, num_workers=0)
               for x in ['gallery', 'query', 'multi-query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()


def load_network(network):
    save_path = os.path.join('./model', opt.model_path)
    network.load_state_dict(torch.load(save_path))
    return network


use_gpu = torch.cuda.is_available()


if __name__ == "__main__":
    from reid_attention import VGG_16
    from resnet_attention import ResNetAttention
    network = ResNetAttention(num_class)
    model = load_network(network)
    #removed = list(model.children())[:-1]
    #from  torch import nn
    #model = nn.Sequential(*removed)
    if use_gpu:
        model = model.cuda()
    test(model, queryloader=dataloaders['query'], galleryloader=dataloaders['gallery'],
         use_gpu=use_gpu)