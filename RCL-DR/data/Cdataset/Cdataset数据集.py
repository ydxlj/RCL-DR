import time
import scipy.io as sio
import pandas as pd
import os
import torch
from tensorboard.backend.event_processing.event_file_inspector import PRINT_SEPARATOR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import numpy as np
import faiss
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp

# 导入自定义的NCL模型和其他函数
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss

from scipy.io import loadmat
from scipy.sparse import csr_matrix

def load_data(filepath):
    # 读取 CSV 文件
    df = pd.read_csv(filepath, header=None)
    # 将 DataFrame 转换为 NumPy 数组
    drug_disease_matrix = df.values
    return drug_disease_matrix


class NCL(torch.nn.Module):
    def __init__(self, embedding_dim, n_layers, reg_weight, ssl_temp, ssl_reg, hyper_layers, alpha, proto_reg,
                 num_clusters, drug_disease_matrix, clip_grad_norm=None):
        super(NCL, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.drug_disease_matrix = drug_disease_matrix
        self.n_drugs, self.n_diseases = self.drug_disease_matrix.shape
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.hyper_layers = hyper_layers
        self.alpha = alpha
        self.proto_reg = proto_reg
        self.num_clusters = num_clusters
        self.clip_grad_norm = clip_grad_norm  # 添加 clip_grad_norm 属性

        # 初始化嵌入层
        self.drug_embedding = torch.nn.Embedding(self.n_drugs, embedding_dim)
        self.disease_embedding = torch.nn.Embedding(self.n_diseases, embedding_dim)

        # 初始化损失函数
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # 用于全排序评估加速的存储变量
        self.restore_drug_e = None
        self.restore_disease_e = None
        self.norm_adj_mat = self.get_norm_adj_mat(self.drug_disease_matrix).to(self.device)

        # 参数初始化
        self.other_parameter_name = ['restore_drug_e', 'restore_disease_e']
        self.apply(xavier_uniform_initialization)
        self.drug_centroids = None
        self.drug_2cluster = None
        self.disease_centroids = None
        self.disease_2cluster = None

    def e_step(self):
        drug_embeddings = self.drug_embedding.weight.detach().cpu().numpy()
        disease_embeddings = self.disease_embedding.weight.detach().cpu().numpy()
        self.drug_centroids, self.drug_2cluster = self.run_kmeans(drug_embeddings)
        self.disease_centroids, self.disease_2cluster = self.run_kmeans(disease_embeddings)

        self.drug_centroids = self.drug_centroids.to(self.drug_embedding.weight.device)
        self.drug_2cluster = self.drug_2cluster.to(self.drug_embedding.weight.device)
        self.disease_centroids = self.disease_centroids.to(self.disease_embedding.weight.device)
        self.disease_2cluster = self.disease_2cluster.to(self.disease_embedding.weight.device)

    def run_kmeans(self, x):
        kmeans = faiss.Kmeans(d=self.embedding_dim, k=self.num_clusters, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        centroids = torch.Tensor(cluster_cents).to(self.drug_embedding.weight.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.drug_embedding.weight.device)
        return centroids, node2cluster

    def get_norm_adj_mat(self, data):
        A = sp.dok_matrix((self.n_drugs + self.n_diseases, self.n_drugs + self.n_diseases), dtype=np.float32)
        A[:self.n_drugs, self.n_drugs:] = data
        A[self.n_drugs:, :self.n_drugs] = data.T

        # 规范化邻接矩阵
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D

        # 将规范化的邻接矩阵转换为稀疏张量
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        drug_embeddings = self.drug_embedding.weight
        disease_embeddings = self.disease_embedding.weight
        ego_embeddings = torch.cat([drug_embeddings, disease_embeddings], dim=0)
        return ego_embeddings

    def forward(self, norm_adj):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
            all_embeddings = torch.sparse.mm(norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        drug_all_embeddings, disease_all_embeddings = torch.split(lightgcn_all_embeddings,
                                                                  [self.n_drugs, self.n_diseases])
        return drug_all_embeddings, disease_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, drug, disease):
        drug_embeddings_all, disease_embeddings_all = torch.split(node_embedding, [self.n_drugs, self.n_diseases])

        drug_embeddings = drug_embeddings_all[drug]
        norm_drug_embeddings = F.normalize(drug_embeddings)

        drug2cluster = self.drug_2cluster[drug]
        drug2centroids = self.drug_centroids[drug2cluster]
        pos_score_drug = torch.mul(norm_drug_embeddings, drug2centroids).sum(dim=1)
        pos_score_drug = torch.exp(pos_score_drug / self.ssl_temp)
        ttl_score_drug = torch.matmul(norm_drug_embeddings, self.drug_centroids.transpose(0, 1))
        ttl_score_drug = torch.exp(ttl_score_drug / self.ssl_temp).sum(dim=1)

        proto_nce_loss_drug = -torch.log(pos_score_drug / ttl_score_drug).sum()

        disease_embeddings = disease_embeddings_all[disease]
        norm_disease_embeddings = F.normalize(disease_embeddings)

        disease2cluster = self.disease_2cluster[disease]
        disease2centroids = self.disease_centroids[disease2cluster]
        pos_score_disease = torch.mul(norm_disease_embeddings, disease2centroids).sum(dim=1)
        pos_score_disease = torch.exp(pos_score_disease / self.ssl_temp)
        ttl_score_disease = torch.matmul(norm_disease_embeddings, self.disease_centroids.transpose(0, 1))
        ttl_score_disease = torch.exp(ttl_score_disease / self.ssl_temp).sum(dim=1)
        proto_nce_loss_disease = -torch.log(pos_score_disease / ttl_score_disease).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_drug + proto_nce_loss_disease)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, drug, disease):
        current_drug_embeddings, current_disease_embeddings = torch.split(current_embedding,
                                                                          [self.n_drugs, self.n_diseases])
        previous_drug_embeddings_all, previous_disease_embeddings_all = torch.split(previous_embedding,
                                                                                    [self.n_drugs, self.n_diseases])

        current_drug_embeddings = current_drug_embeddings[drug]
        previous_drug_embeddings = previous_drug_embeddings_all[drug]
        norm_drug_emb1 = F.normalize(current_drug_embeddings)
        norm_drug_emb2 = F.normalize(previous_drug_embeddings)
        norm_all_drug_emb = F.normalize(previous_drug_embeddings_all)
        pos_score_drug = torch.mul(norm_drug_emb1, norm_drug_emb2).sum(dim=1)
        ttl_score_drug = torch.matmul(norm_drug_emb1, norm_all_drug_emb.transpose(0, 1))
        pos_score_drug = torch.exp(pos_score_drug / self.ssl_temp)
        ttl_score_drug = torch.exp(ttl_score_drug / self.ssl_temp).sum(dim=1)

        ssl_loss_drug = -torch.log(pos_score_drug / ttl_score_drug).sum()

        current_disease_embeddings = current_disease_embeddings[disease]
        previous_disease_embeddings = previous_disease_embeddings_all[disease]
        norm_disease_emb1 = F.normalize(current_disease_embeddings)
        norm_disease_emb2 = F.normalize(previous_disease_embeddings)
        norm_all_disease_emb = F.normalize(previous_disease_embeddings_all)
        pos_score_disease = torch.mul(norm_disease_emb1, norm_disease_emb2).sum(dim=1)
        ttl_score_disease = torch.matmul(norm_disease_emb1, norm_all_disease_emb.transpose(0, 1))
        pos_score_disease = torch.exp(pos_score_disease / self.ssl_temp)
        ttl_score_disease = torch.exp(ttl_score_disease / self.ssl_temp).sum(dim=1)

        ssl_loss_disease = -torch.log(pos_score_disease / ttl_score_disease).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_drug + self.alpha * ssl_loss_disease)
        return ssl_loss

    # 在 NCL 类中修改 calculate_loss 方法，使其返回单个损失值
    def calculate_loss(self, interaction, norm_adj):
        # clear the storage variable when training
        if self.restore_drug_e is not None or self.restore_disease_e is not None:
            self.restore_drug_e, self.restore_disease_e = None, None

        drug = interaction['drug']
        pos_disease = interaction['pos_disease']
        neg_disease = interaction['neg_disease']

        drug_all_embeddings, disease_all_embeddings, embeddings_list = self.forward(norm_adj)

        center_embedding = embeddings_list[0]
        context_embedding = embeddings_list[self.hyper_layers * 2]

        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, drug, pos_disease)
        proto_loss = self.ProtoNCE_loss(center_embedding, drug, pos_disease)

        r_embeddings = drug_all_embeddings[drug]
        pos_embeddings = disease_all_embeddings[pos_disease]
        neg_embeddings = disease_all_embeddings[neg_disease]

        # calculate BPR Loss
        pos_scores = torch.mul(r_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(r_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        r_ego_embeddings = self.drug_embedding(drug)
        pos_ego_embeddings = self.disease_embedding(pos_disease)
        neg_ego_embeddings = self.disease_embedding(neg_disease)

        reg_loss = self.reg_loss(r_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        total_loss = mf_loss + self.reg_weight * reg_loss + ssl_loss + proto_loss

        return total_loss

    def predict(self, interaction):
        drug = interaction['drug']
        disease = interaction['disease']
        drug_all_embeddings, disease_all_embeddings, embeddings_list = self.forward()
        r_embeddings = drug_all_embeddings[drug]
        d_embeddings = disease_all_embeddings[disease]
        scores = torch.mul(r_embeddings, d_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        drug = interaction['drug']
        if self.restore_drug_e is None or self.restore_disease_e is None:
            self.restore_drug_e, self.restore_disease_e, embedding_list = self.forward()
        r_embeddings = self.restore_drug_e[drug]
        scores = torch.matmul(r_embeddings, self.restore_disease_e.transpose(0, 1))
        return scores.view(-1)




    def get_norm_adj(self):
        return self.norm_adj_mat


# 获取当前时间
def get_current_time():
    return time.time()


import numpy as np
from torch.utils.data import Dataset


def load_data(filepath):
    data_dict = sio.loadmat(filepath)
    drug_disease_matrix = data_dict['didr']
    return drug_disease_matrix


from imblearn.over_sampling import SMOTE


def load_data(filepath):
    data_dict = sio.loadmat(filepath)
    drug_disease_matrix = data_dict['didr']
    return drug_disease_matrix

class SparseMatrixDataset(Dataset):
    def __init__(self, drug_disease_matrix, apply_smote=None):
        self.drug_disease_matrix = drug_disease_matrix
        self.n_drugs, self.n_diseases = drug_disease_matrix.shape
        self.positive_pairs = self._get_positive_pairs()
        self.negative_pairs = self._get_negative_pairs()

        if apply_smote:
            self.apply_smote()

    def _get_positive_pairs(self):
        positive_pairs = []
        drug_indices, disease_indices = self.drug_disease_matrix.nonzero()
        for drug, disease in zip(drug_indices, disease_indices):
            positive_pairs.append((drug, disease))
        return positive_pairs

    def _get_negative_pairs(self):
        negative_pairs = []
        for drug in range(self.n_drugs):
            for disease in range(self.n_diseases):
                if self.drug_disease_matrix[drug, disease] == 0:
                    negative_pairs.append((drug, disease))
        return negative_pairs

    def apply_smote(self):
        X_pos = np.array(self.positive_pairs)
        X_neg = np.array(self.negative_pairs)
        y_pos = np.ones(len(X_pos))
        y_neg = np.zeros(len(X_neg))
        X = np.vstack((X_pos, X_neg))
        y = np.hstack((y_pos, y_neg))
        smote = SMOTE(random_state=56)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        self.positive_pairs = X_resampled[y_resampled == 1]
        self.negative_pairs = X_resampled[y_resampled == 0]

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        drug, pos_disease = self.positive_pairs[idx]
        _, neg_disease = self.negative_pairs[idx]
        return {
            'drug': torch.tensor(drug, dtype=torch.long),
            'pos_disease': torch.tensor(pos_disease, dtype=torch.long),
            'neg_disease': torch.tensor(neg_disease, dtype=torch.long)
        }

class Trainer:
    def __init__(self, model, optimizer, num_m_step, device, early_stop=None, verbose=False, callbacks=None,
                 patience=10):
        self.model = model
        self.optimizer = optimizer
        self.num_m_step = num_m_step
        self.device = device
        self.early_stop = early_stop
        self.verbose = verbose
        self.callbacks = callbacks if callbacks is not None else []
        self.best_valid_score = -np.inf
        self.best_valid_result = None
        self.stop_training = False
        self.patience = patience
        self.patience_counter = 0

    def fit(self, train_data, val_data=None, epochs=100, batch_size=256, eval_step=1, save_model=True,
            show_progress=False, apply_smote=True):
        train_loader = self._create_data_loader(train_data, batch_size, apply_smote=apply_smote)
        val_loader = self._create_data_loader(val_data, batch_size,
                                              apply_smote=apply_smote) if val_data is not None else None
        if save_model and epochs >= epochs:
            self._save_model()

        for epoch_idx in range(epochs):
            if epoch_idx % self.num_m_step == 0:
                self.model.e_step()

            train_start_time = get_current_time()
            train_loss = self._train_epoch(train_loader, epoch_idx, show_progress=show_progress)
            train_end_time = get_current_time()

            print(
                f"Epoch {epoch_idx + 1}/{epochs}, Time: {train_end_time - train_start_time:.2f}s")

            train_auroc, train_aupr, _, _ = calculate_auroc_aupr(self.model, train_data)

            if eval_step <= 0 or val_data is None:
                continue

            if epoch_idx % eval_step == 0:
                valid_start_time = get_current_time()
                valid_score, valid_result = self._valid_epoch(val_loader, epoch_idx, show_progress=show_progress)
                valid_end_time = get_current_time()

                if self.early_stop:
                    self._update_best_valid(valid_score, valid_result)

                valid_info = f"Epoch {epoch_idx + 1}/{epochs}, Valid Score: {valid_score:.4f}, Time: {valid_end_time - valid_start_time:.2f}s"
                if self.verbose:
                    print(valid_info)

                if self.best_valid_score == valid_score:
                    if save_model:
                        self._save_model()

                if self.stop_training:
                    print("Early stopping triggered. Stopping training.")
                    break

                if val_data is not None:
                    val_auroc, val_aupr, _, _ = calculate_auroc_aupr(self.model, val_data)
                    print(
                        f"Epoch {epoch_idx + 1}/{epochs}, Validation AUROC: {val_auroc:.4f}, Validation AUPR: {val_aupr:.4f}")

        return self.best_valid_score, self.best_valid_result

    def _train_epoch(self, train_loader, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        if loss_func is None:
            loss_func = self.model.calculate_loss

        total_loss = 0
        if show_progress:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch_idx + 1} Training", leave=True)

        for batch in train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            norm_adj = self.model.norm_adj_mat
            loss = loss_func(batch, norm_adj)
            if isinstance(loss, tuple):
                loss = loss[1]
            if torch.isnan(loss):
                raise ValueError("Loss is NaN.")
            loss.backward()
            if self.model.clip_grad_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.model.clip_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _valid_epoch(self, val_loader, epoch_idx, show_progress=False):
        self.model.eval()
        total_score = 0
        if show_progress:
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch_idx + 1} Validation", leave=False)

        with torch.no_grad():
            norm_adj = self.model.get_norm_adj()
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                drug_all_embeddings, disease_all_embeddings, _ = self.model.forward(norm_adj)
                scores = torch.matmul(drug_all_embeddings, disease_all_embeddings.t())
                total_score += scores.mean().item()

        valid_score = total_score / len(val_loader)
        valid_result = {
            'valid_score': valid_score,
        }
        return valid_score, valid_result

    def _update_best_valid(self, valid_score, valid_result):
        if valid_score > self.best_valid_score:
            self.best_valid_score = valid_score
            self.best_valid_result = valid_result
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.stop_training = True

    def _save_model(self):
        save_dir = "../Ldataset"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    def _create_data_loader(self, data, batch_size, apply_smote=True):
        dataset = SparseMatrixDataset(data, apply_smote=apply_smote)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return data_loader

def calculate_auroc_aupr(model, val_data, optimizer):
    model.train()  # 设置为训练模式
    with torch.enable_grad():  # 允许梯度计算
        if isinstance(val_data, np.ndarray):
            val_data = sp.csr_matrix(val_data)

        full_drug_disease = sp.csr_matrix((model.n_drugs, model.n_diseases), dtype=np.float32)
        full_drug_disease[:val_data.shape[0], :val_data.shape[1]] = val_data

        norm_adj = model.get_norm_adj_mat(full_drug_disease).to(model.device)
        drug_embeddings, disease_embeddings, _ = model.forward(norm_adj)

        val_drug_embeddings = drug_embeddings[:val_data.shape[0]]
        val_disease_embeddings = disease_embeddings[:val_data.shape[1]]

        scores = torch.matmul(val_drug_embeddings, val_disease_embeddings.t())
        scores = torch.sigmoid(scores)

        true_labels = val_data.toarray().flatten()
        pred_scores = scores.cpu().numpy().flatten()
        true_labels_binary = (true_labels > 0).astype(int)

        roc_auc = roc_auc_score(true_labels_binary, pred_scores)
        prauc = average_precision_score(true_labels_binary, pred_scores)

        # 计算损失并进行反向传播和优化器更新
        loss = model.calculate_loss(batch, norm_adj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return roc_auc, prauc, true_labels_binary, pred_scores


def run_kfold_experiment(data, model_params, optimizer_params, kfold_params, num_repeats=2, apply_smote=True):
    results = []
    all_auroc_scores = []
    all_aupr_scores = []
    all_y_true = []
    all_y_pred = []

    output_dir = "experiment_results12"
    os.makedirs(output_dir, exist_ok=True)

    for repeat in range(num_repeats):
        print(f"Starting repeat {repeat + 1}/{num_repeats}")
        kf = KFold(**kfold_params)

        for fold, (train_index, val_index) in enumerate(kf.split(data)):
            print(f"Starting fold {fold + 1}/{kf.n_splits}")
            train_data = data[train_index]
            val_data = data[val_index]
            print(f"Validation data shape: {val_data.shape}")

            if apply_smote:
                train_dataset = SparseMatrixDataset(train_data, apply_smote=False)
                train_data = train_dataset.drug_disease_matrix
                print("SMOTE applied. New train data shape:", train_data.shape)
                print("Sample SMOTE data:", train_data[:5, :5])

                val_dataset = SparseMatrixDataset(val_data, apply_smote=False)
                print("SMOTE applied. New train data shape:", train_data.shape)
                val_data = val_dataset.drug_disease_matrix

            if isinstance(val_data, np.ndarray):
                val_data = sp.csr_matrix(val_data)

            val_positive_samples = np.sum(val_data.toarray() > 0)
            val_negative_samples = np.sum(val_data.toarray() == 0)
            print(f"Validation set positive samples: {val_positive_samples}")
            print(f"Validation set negative samples: {val_negative_samples}")

            overlap = set(train_index) & set(val_index)
            if overlap:
                print("Overlap detected:", overlap)
            else:
                print("No overlap detected.")

            model = NCL(**model_params)
            optimizer = optim.Adam(model.parameters(), **optimizer_params)
            trainer = Trainer(model, optimizer, num_m_step=5,
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), patience=10)

            trainer.fit(train_data, val_data, epochs=100, batch_size=256, eval_step=1, save_model=True,
                        show_progress=True)

            val_auroc, val_aupr, y_true, y_pred = calculate_auroc_aupr(model, val_data)
            all_auroc_scores.append(val_auroc)
            all_aupr_scores.append(val_aupr)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

            result_file = os.path.join(output_dir, f"repeat_{repeat}_fold_{fold}.txt")
            with open(result_file, 'w') as f:
                f.write("True Labels\tPredicted Scores\n")
                for true_label, pred_score in zip(y_true, y_pred):
                    f.write(f"{true_label}\t{pred_score}\n")

            results.append({
                'repeat': repeat,
                'fold': fold,
                'best_valid_score': trainer.best_valid_score,
                'best_valid_result': trainer.best_valid_result
            })

    avg_auroc = np.mean(all_auroc_scores)
    std_auroc = np.std(all_auroc_scores)
    avg_aupr = np.mean(all_aupr_scores)
    std_aupr = np.std(all_aupr_scores)

    print(f"Average AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")
    print(f"Average AUPR: {avg_aupr:.4f} ± {std_aupr:.4f}")

    return results, avg_auroc, avg_aupr, all_y_true, all_y_pred


def main():
    # 读取药物疾病关联矩阵
    data_path = "D:\\13645583245\\文献代码\\DRNCL\\dataset\\Fdataset\\Fdataset.mat"
    drug_disease_matrix = load_data(data_path)
    drug_disease_matrix = drug_disease_matrix.T

    embedding_dim = 32
    n_layers = 3
    reg_weight = 1e-5
    ssl_temp = 0.1
    ssl_reg = 1e-6
    hyper_layers = 1
    alpha = 0.5
    proto_reg = 8e-8
    num_clusters = 5
    clip_grad_norm = 5

    model_params = {
        'embedding_dim': embedding_dim,
        'n_layers': n_layers,
        'reg_weight': reg_weight,
        'ssl_temp': ssl_temp,
        'ssl_reg': ssl_reg,
        'hyper_layers': hyper_layers,
        'alpha': alpha,
        'proto_reg': proto_reg,
        'num_clusters': num_clusters,
        'drug_disease_matrix': drug_disease_matrix,
        'clip_grad_norm': clip_grad_norm
    }

    optimizer_params = {'lr': 0.001}

    kfold_params = {
        'n_splits': 10,
        'shuffle': True,
        'random_state': 56
    }

    results, avg_auroc, avg_aupr, all_y_true, all_y_pred = run_kfold_experiment(drug_disease_matrix, model_params,
                                                                                optimizer_params,
                                                                                kfold_params, num_repeats=10,
                                                                                apply_smote=True)

    # 汇总结果
    best_valid_scores = [result['best_valid_score'] for result in results]
    best_valid_results = [result['best_valid_result'] for result in results]

    # 打印平均 AUROC 和 AUPR
    print(f"Average AUROC: {avg_auroc:.4f} ± {np.std(best_valid_scores):.4f}")
    print(f"Average AUPR: {avg_aupr:.4f} ± {np.std(best_valid_results):.4f}")

if __name__ == "__main__":
    main()