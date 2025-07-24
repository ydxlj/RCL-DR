import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import faiss
from torch.nn.utils import clip_grad_norm_
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss

class NCL(torch.nn.Module):
    def __init__(self, embedding_dim, n_layers, reg_weight, ssl_temp, ssl_reg,
                 hyper_layers, alpha, proto_reg, num_clusters, drug_disease_matrix,
                 clip_grad_norm=None):
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
        self.clip_grad_norm = clip_grad_norm

        self.drug_embedding = torch.nn.Embedding(self.n_drugs, embedding_dim)
        self.disease_embedding = torch.nn.Embedding(self.n_diseases, embedding_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.restore_drug_e = None
        self.restore_disease_e = None
        self.norm_adj_mat = self.get_norm_adj_mat(self.drug_disease_matrix).to(self.device)

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

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D

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

    def calculate_loss(self, interaction, norm_adj):
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