import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0, tau = 1.0, amc_instance = None, amc_temporal = None, amc_margin = 0.5):
    
    # print(f'amc instance : {amc_instance}', f'amc temporal : {amc_temporal}')
    # print('Tau = ', tau)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, tau=tau, amc_margin= amc_margin, amc_coef= amc_instance)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, tau=tau, amc_coef= amc_temporal, amc_margin = amc_margin)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2,  tau=tau, amc_coef= amc_temporal, amc_margin = amc_margin)
        d += 1
    # print(loss / d)
    return loss / d

def instance_contrastive_loss(z1, z2, tau = 1.0, amc_coef = 0, amc_margin = 0.5): # EQ2 in the paper, z1 and z2 have the shape of BxTxC
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    

    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]   # THE ONLY THING THAT HAPPENS HERE IS THAT THE DIAGONLA IS REMOVED!! MAGE MA MASKHARE YE TOIM?
    logits = torch.div(logits, tau)
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2

    if amc_coef:
        # print('calculating amc!!!')
        amc_loss = amc3d_vectorized('cuda', z, amc_margin = amc_margin)
        # print(f'Loss Inst: {loss}, amc_loss : {amc_coef * amc_loss}')
        return loss + amc_coef * amc_loss
    else:
        return loss
    

def temporal_contrastive_loss(z1, z2, tau = 1, amc_coef = 0, amc_margin = 0.5): # EQ1 in the paper, z1 and z2 have the shape of BxTxC
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    logits = -F.log_softmax(logits, dim=-1)
    logits = torch.div(logits, tau)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    # print(loss, amc_loss)
    if amc_coef:
        amc_loss = amc3d_vectorized('cuda', z,  amc_margin = amc_margin)
        # with torch.no_grad():
        #     amc_loss_v = amc3d_vectorized('cuda', z,  amc_margin = amc_margin)
            # print(z.shape)
        # print('AMC : ', amc_loss)
        # print('AMC Vec : ', amc_loss_v)
        # print(f'Loss Temp : {loss}, amc_loss : {amc_coef * amc_loss}')
        return loss + amc_coef * amc_loss
    else:
        return loss



import torch
import torch.nn.functional as F

from copy import deepcopy
def amc3d(device, features, amc_margin = 0.5):
    total_loss = torch.tensor(0.0).to('cuda')
    # main_features = deepcopy(features)
    main_features = features
    for i in range(len(main_features)):
        features = main_features[i]
        bs = features.shape[0]/2

        labels = torch.cat([torch.arange(bs) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)
        

        similarity_matrix = torch.matmul(features, features.T)
        # print(similarity_matrix.shape)
        # print(labels.shape)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)


        # m = 0.5
        m = amc_margin
        negatives = torch.clamp(negatives,min=-1+1e-10,max=1-1e-10)
        clip = torch.acos(negatives)
        b1 = m - clip
        mask = b1>0
        l1 = torch.sum((mask*b1)**2)
        positives = torch.clamp(positives,min = -1+1e-10,max = 1-1e-10)
        l2 = torch.acos(positives)
        l2 = torch.sum(l2**2)
        
        loss = (l1 + l2)/25
        # print(loss, total_loss)
        total_loss = total_loss + loss

    return total_loss


def amc3d_vectorized(device, features, amc_margin=0.5):
    features = F.normalize(features, dim=-1)
    similarity_matrix = torch.matmul(features, features.transpose(-1, -2))
    bs = features.shape[1] // 2
    labels = torch.cat([torch.arange(bs) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    labels = labels.repeat(features.shape[0], 1, 1)
    mask = torch.eye(labels.shape[1], dtype=torch.bool).to(device)
    mask = mask.repeat(features.shape[0], 1, 1)
    labels = labels[~mask].view(features.shape[0], labels.shape[1], -1)
    similarity_matrix = similarity_matrix[~mask].view(features.shape[0], similarity_matrix.shape[1], -1)
    positives = similarity_matrix[labels.bool()].view(features.shape[0], labels.shape[1], -1)
    negatives = similarity_matrix[~labels.bool()].view(features.shape[0], labels.shape[1], -1)
    negatives = torch.clamp(negatives, min=-1+1e-10, max=1-1e-10)
    clip = torch.acos(negatives)
    b1 = amc_margin - clip
    mask = b1 > 0
    l1 = torch.sum((mask * b1) ** 2, dim=[1, 2])
    positives = torch.clamp(positives, min=-1+1e-10, max=1-1e-10)
    l2 = torch.acos(positives)
    l2 = torch.sum(l2 ** 2, dim=[1, 2])
    loss = (l1 + l2) / 25
    total_loss = torch.sum(loss)
    return total_loss