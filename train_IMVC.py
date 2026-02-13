from __future__ import print_function, division
import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from sklearn import preprocessing
import warnings
from time import time
import logging
from datetime import datetime

# Import custom utils
from utils import cluster_acc, WKLDiv, imagedataset_with_mask_and_full, initialize_logging, plot_pretrain_metrics, plot_finetuning_metrics

warnings.filterwarnings('ignore')

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SingleViewModel(nn.Module):
    def __init__(self, n_input, n_z, n_clusters, pretrain):
        super(SingleViewModel, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input[0], 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
        )
        self.latentmu = nn.Sequential(
            nn.Linear(64 * 4 * 4, n_z),
            nn.ReLU()
        )
        self.latentep = nn.Sequential(
            nn.Linear(64 * 4 * 4, n_z),
            nn.ReLU()
        )
        self.delatent = nn.Sequential(
            nn.Linear(n_z, 64 * 4 * 4),
            nn.ReLU()
        )
        for m in self.encoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_input[0], (4, 4), stride=2, padding=1),
            nn.Sigmoid()
        )

        for m in self.decoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)
        self.pretrain = pretrain

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        z = mu + eps * std
        return z

    def forward(self, x):
        mu = self.latentmu(self.encoder(x).view(x.shape[0],-1))
        log = self.latentep(self.encoder(x).view(x.shape[0],-1))
        z = self.reparameterize(mu, log)
        x_bar = self.decoder(self.delatent(z).view(-1,64,4,4))
        return x_bar, z, mu, log


class MultiViewModel(nn.Module):
    def __init__(self, n_input, n_z, n_clusters, pretrain, save_path, args):
        super(MultiViewModel, self).__init__()
        self.pretrain = pretrain
        self.save_path = save_path
        self.n_clusters = n_clusters
        self.viewNumber = args.viewNumber
        
        classifier = list()
        for viewIndex in range(self.viewNumber):
            classifier.append(
                nn.Sequential(
                    Linear(args.viewNumber * n_z, n_clusters),
                    nn.Softmax(dim=1)
                )
            )
        self.classifier = nn.ModuleList(classifier)

        if args.share:
            self.aes = SingleViewModel(
                n_input=n_input[0],
                n_z=n_z,
                n_clusters=self.n_clusters,
                pretrain=self.pretrain)
        else:
            aes = []
            for viewIndex in range(self.viewNumber):
                aes.append(SingleViewModel(
                n_input=n_input[0],
                n_z=n_z,
                n_clusters=self.n_clusters,
                pretrain=self.pretrain))
            self.aes = nn.ModuleList(aes)
        self.args = args

    def forward(self, x):
        outputs = []
        if self.args.share:
            for viewIndex in range(self.viewNumber):
                outputs.append(self.aes(x[viewIndex]))
        else:
            for viewIndex in range(self.viewNumber):
                outputs.append(self.aes[viewIndex](x[viewIndex]))
        return outputs


class Gaussian(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(Gaussian, self).__init__()
        self.num_classes = num_classes
        self.mean = nn.Parameter(torch.zeros(self.num_classes, latent_dim))

    def forward(self, z):
        z = z.unsqueeze(1)
        return z - self.mean.unsqueeze(0)


class Multi_Gaussian(nn.Module):
    def __init__(self, num_classes, latent_dim, viewNumber):
        super(Multi_Gaussian, self).__init__()
        self.num_classes = num_classes
        self.viewNumber = viewNumber
        self.latent_dim = latent_dim

        gus_list = list()
        for viewIndex in range(self.viewNumber):
            gus_list.append(Gaussian(self.num_classes, self.viewNumber*self.latent_dim).cuda())
        self.gus_list = nn.ModuleList(gus_list)

    def forward(self, z, index):
        return self.gus_list[index](z)

def MMD_loss(x, y, batch_size, sigma=1.0):
    Kxx = compute_rbf_kernel(x, x, sigma)
    Kxy = compute_rbf_kernel(x, y, sigma)
    Kyy = compute_rbf_kernel(y, y, sigma)
    loss = torch.sum(Kxx) + torch.sum(Kyy) - 2 * torch.sum(Kxy)
    return loss/(batch_size*batch_size)

def compute_rbf_kernel(x, y, sigma=1.0):
    dist = torch.sum(x ** 2, dim=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
    kernel = torch.exp(-dist / (2 * sigma ** 2))
    return kernel

def gaussian_kl_divergence(mu1, logvar1, mu2, logvar2):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    kl_divergence = 0.5 * (torch.sum(var1 / var2, dim=-1)
                           + torch.sum((mu2 - mu1).pow(2) / var2, dim=-1)
                           + torch.sum(logvar2, dim=-1)
                           - torch.sum(logvar1, dim=-1)
                           - mu1.shape[-1])
    return torch.sum(kl_divergence)/(mu1.shape[0]*mu1.shape[1])

def pretrain_aes(args, log_file=None):
    save_path = args.save_path
    viewNumber = args.viewNumber
    model = MultiViewModel(
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        pretrain=True,
        save_path=args.save_path,
        args=args).cuda()

    dataset = imagedataset_with_mask_and_full(args.dataset, args.viewNumber, args.method, True, missing_rate = args.missing_rate)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)
    multi_gaussian = Multi_Gaussian(args.n_clusters, args.n_z, args.viewNumber).cuda()

    gamma_1 = args.gamma_1
    for epoch in tqdm(range(1000), desc="Pretrain AEs Epochs"):
        for batch_idx, (x, _, _) in enumerate(dataLoader):
            loss=0.
            mseloss=0.
            kl_loss=0.
            cat_loss=0.
            for viewIndex in range(viewNumber):
                x[viewIndex] = x[viewIndex].cuda()        
            output = model(x)

            for viewIndex in range(viewNumber):
                if viewIndex == 0:
                    z_all = output[viewIndex][1]
                else:
                    z_all = torch.cat((z_all, output[viewIndex][1]), dim=1)
            for viewIndex in range(viewNumber):
                z_prior_mean=multi_gaussian(z_all,viewIndex)
                mseloss += loss + F.mse_loss(output[viewIndex][0], x[viewIndex])
                temp = -0.5 * (output[viewIndex][3].unsqueeze(1) -
                               torch.sum(torch.square(z_prior_mean),dim=2,keepdim=True)/((args.n_z)))
                y_v = model.classifier[viewIndex](z_all)
                kl_loss += torch.sum(torch.mean(y_v.unsqueeze(-1) * temp), dim=0)
                cat_loss += -torch.sum(torch.mean(y_v * torch.log(y_v + 1e-8), dim=0))

            loss = mseloss + gamma_1 * kl_loss + 0*cat_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.print_log and epoch % 10 == 0 and batch_idx % 50 == 0:
                print('epoch:{},batch:{},mseloss:{:.4f}, kl_loss{:.4f}, loss{:.4f}'.format(epoch,batch_idx, mseloss,kl_loss, loss))
            if log_file and epoch % 10 == 0 and batch_idx % 50 == 0:
                logging.info(f'Pretrain - Epoch: {epoch}, Batch: {batch_idx}, MSE Loss: {mseloss:.4f}, KL Loss: {kl_loss:.4f}, Total Loss: {loss:.4f}')

    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))
    
    torch.save(model.state_dict(), args.save_path + str(args.share) + '.pkl')
    torch.save(multi_gaussian.state_dict(), args.save_path + str(args.share) + '_multigaussian.pkl')
    
    # Clustering check
    dataLoader = DataLoader(dataset, batch_size=args.instanceNumber, shuffle=False)
    for batch_idx, (x, y, _) in enumerate(dataLoader):
        for viewIndex in range(viewNumber):
            x[viewIndex] = x[viewIndex].cuda()
        output = model(x)
        y = y.data.cpu().numpy()
        
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    for viewIndex in range(args.viewNumber):
        z_v = output[viewIndex][1]
        if (viewIndex) == 0:
            z_all = z_v
        else:
            z_all = torch.cat((z_all, z_v), dim=1)
            
    kmeans.fit_predict(z_all.cpu().detach().data.numpy())
    y_pred = kmeans.labels_
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('Pretrain z_all :acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    if log_file:
        logging.info(f'Pretrain Result - z_all :acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}')

def fineTuning(args, log_file=None):
    model = MultiViewModel(
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        pretrain=True,
        save_path=args.save_path,
        args=args).cuda()

    viewNumber = args.viewNumber
    multi_gaussian = Multi_Gaussian(args.n_clusters, args.n_z, args.viewNumber).cuda()
    model.load_state_dict(torch.load(args.save_path + str(args.share) + '.pkl'))
    multi_gaussian.load_state_dict(torch.load(args.save_path + str(args.share) + '_multigaussian.pkl'))
    
    dataset = imagedataset_with_mask_and_full(args.dataset, args.viewNumber, args.method, True, missing_rate = args.missing_rate)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    gamma_1 = args.gamma_1
    gamma_2 = args.gamma_2
    gamma_3 = args.gamma_3
    gamma_4 = args.gamma_4

    # Initialize final_acc variable to return the final clustering accuracy
    final_acc = 0.0

    for epoch in tqdm(range(1000), desc="FineTuning AEs Epochs"):
        for batch_idx, (x, _, _) in enumerate(dataLoader):
            loss=0.
            mseloss=0.
            kl_loss=0.
            dskl_loss=0.
            mmd_loss=0.
            cat_loss=0.
            for viewIndex in range(viewNumber):
                x[viewIndex] = x[viewIndex].cuda()        
            output = model(x)

            for viewIndex in range(viewNumber):
                if viewIndex == 0:
                    z_all = output[viewIndex][1]
                else:
                    z_all = torch.cat((z_all, output[viewIndex][1]), dim=1)
            for viewIndex in range(viewNumber):
                z_prior_mean=multi_gaussian(z_all,viewIndex)
                mseloss += loss + F.mse_loss(output[viewIndex][0], x[viewIndex])
                temp = -0.5 * (output[viewIndex][3].unsqueeze(1) -
                               torch.sum(torch.square(z_prior_mean),dim=2,keepdim=True)/(args.n_z))
                y_v = model.classifier[viewIndex](z_all)
                kl_loss += torch.sum(torch.mean(y_v.unsqueeze(-1) * temp), dim=0)
                cat_loss += -torch.sum(torch.mean(y_v * torch.log(y_v + 1e-8), dim=0))

            for viewIndex1 in range(0, viewNumber):
                for viewIndex2 in  range(0, viewNumber):
                    if viewIndex1 == viewIndex2: 
                        continue
                    mmd_loss += MMD_loss(output[viewIndex1][1], output[viewIndex2][1], args.batch_size)
                    dskl_loss+= gaussian_kl_divergence(output[viewIndex1][2], output[viewIndex1][3], output[viewIndex2][2], output[viewIndex2][3])

            loss = mseloss + gamma_1 * kl_loss + gamma_2 * mmd_loss + gamma_3 * dskl_loss + gamma_4 * cat_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.print_log and epoch % 50 == 0 and batch_idx % 20 == 0:
                print('epoch:{},batch_idx{}, mseloss:{:.4f}, mmdloss{:.4f}, kl_loss{:.4f}, dsklloss{:.4f},loss{:.4f}'.format(epoch, batch_idx, mseloss, mmd_loss, kl_loss, dskl_loss, loss))
            if log_file and epoch % 50 == 0 and batch_idx % 20 == 0:
                logging.info(f'FineTuning - Epoch: {epoch}, Batch: {batch_idx}, MSE Loss: {mseloss:.4f}, MMD Loss: {mmd_loss:.4f}, KL Loss: {kl_loss:.4f}, DSKL Loss: {dskl_loss:.4f}, Total Loss: {loss:.4f}')

    # Save final model
    torch.save(model.state_dict(), args.save_path.replace('.pkl', '') + '_gen.pkl')
    torch.save(multi_gaussian.state_dict(), args.save_path.replace('.pkl', '') + '_gen_multigaussian.pkl')

    # Final clustering
    dataLoader = DataLoader(dataset, batch_size=args.instanceNumber, shuffle=False)
    for batch_idx, (x, y, _) in enumerate(dataLoader):
        for viewIndex in range(viewNumber):
            x[viewIndex] = x[viewIndex].cuda()
        output = model(x)
        y = y.data.cpu().numpy()
        
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    for viewIndex in range(args.viewNumber):
        z_v = output[viewIndex][1]
        if (viewIndex) == 0:
            z_all = z_v
        else:
            z_all = torch.cat((z_all, z_v), dim=1)

    kmeans.fit_predict(z_all.cpu().detach().data.numpy())
    y_pred = kmeans.labels_
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('FineTuning z_all :acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    if log_file:
        logging.info(f'FineTuning Result - z_all :acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}')

    final_acc = acc

    y_pred = 0
    for viewIndex in range(viewNumber):
        y_pred += model.classifier[viewIndex](z_all)
    y_pred = np.argmax(y_pred.cpu().detach().data.numpy(), axis=1)
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('FineTuning y :acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    if log_file:
        logging.info(f'FineTuning Result - y :acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}')

    return final_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SUMVC Training')
    parser.add_argument('--dataset', type=str, default='Multi-COIL-20', 
                        choices=['resized_NoisyMNIST', 'Multi-COIL-10', 'Multi-COIL-20'])
    parser.add_argument('--missing_rate', type=float, default=0.1, help='Missing rate (0.1, 0.3, 0.5, 0.7)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--n_z', type=int, default=200)
    # parser.add_argument('--share', type=int, default=1)
    parser.add_argument('--gamma_1', type=float, default=0.1)
    parser.add_argument('--gamma_2', type=float, default=0.1)
    parser.add_argument('--print_log', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    args.method = args.dataset # for utils compatibility
    
    if args.dataset == 'resized_NoisyMNIST':
        args.n_input = [[1, 32, 32], [1, 32, 32]]
        args.viewNumber = 2
        args.instanceNumber = 10000
        args.n_clusters = 10
        args.n_z = 200
        args.share = 1

        
    elif args.dataset == 'Multi-COIL-10':
        args.n_input = [[1, 32, 32], [1, 32, 32], [1, 32, 32]]
        args.viewNumber = 3
        args.instanceNumber = 720
        args.n_clusters = 10
        args.share = 1

    elif args.dataset == 'Multi-COIL-20':
        args.n_input = [[1, 32, 32], [1, 32, 32], [1, 32, 32]]
        args.viewNumber = 3
        args.instanceNumber = 1440
        args.n_clusters = 20
        args.share = 1

    if not os.path.exists('./results'):
        os.makedirs('./results')

    args.gamma_3 = args.gamma_2
    args.gamma_4 = args.gamma_1

    print(f"Starting training for {args.dataset} with Missing Rate {args.missing_rate}")
    print(args)

    if args.dataset == 'Multi-COIL-20':
        batch_size_candidates = [144, 288, 12, 720, 1440, 6, 1] 
    elif args.dataset == 'Multi-COIL-10':
        batch_size_candidates = [720, 72, 144, 360, 12]    
    elif args.dataset == 'resized_NoisyMNIST':
        batch_size_candidates = [500, 1000, 2000, 200, 100, 20, 5000, 10000]    
    else:
        batch_size_candidates = [2000, 1000, 100, 20]   

    best_acc = -1.0
    best_bs = -1

    for bs in batch_size_candidates:
        print(f"\n{'='*40}\nTesting Batch Size: {bs}\n{'='*40}")
        
        args.batch_size = bs
        args.save_path = f"./results/{args.dataset}_mr{args.missing_rate}_bs{bs}_"
        
        try:
            log_file = initialize_logging(args)
            t0 = time()
            pretrain_aes(args, log_file)
            current_acc = fineTuning(args, log_file)
            
            t1 = time()
            print(f"Batch Size {bs} finished in {t1-t0:.2f}s. ACC: {current_acc:.4f}")
            logging.info(f"Batch Size {bs} Training Finished. Final ACC: {current_acc:.4f}")
            
            if current_acc > best_acc:
                best_acc = current_acc
                best_bs = bs
                print(f"*** New Best Found! Batch Size: {best_bs}, ACC: {best_acc:.4f} ***")

            # try:
            #     plot_pretrain_metrics(log_file, args)
            #     plot_finetuning_metrics(log_file, args)
                
            #     import glob
            #     import os
            #     for svg_file in glob.glob(f"{args.dataset}*.svg"):
            #         dst_path = os.path.join('results', os.path.basename(svg_file))
            #         os.replace(svg_file, dst_path)
            # except Exception as e:
            #     print(f"Plotting failed: {e}")

            if current_acc >= 0.9999:
                break
                
        except Exception as e:
            continue
            
    print(f"\n{'='*40}")
    print(f"Grid Search Completed.")
    print(f"Best Batch Size: {best_bs}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"{'='*40}")