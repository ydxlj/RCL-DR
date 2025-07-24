import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision, TP, FP, FN, TN]

def evaluate(predict, label):
    aupr = metrics.average_precision_score(y_true=label, y_score=predict)
    auroc = metrics.roc_auc_score(y_true=label, y_score=predict)
    result = {"aupr": aupr, "auroc": auroc}
    return result

def calculate_auroc_aupr(model, val_data, drug_names, disease_names):
    model.eval()
    with torch.no_grad():
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

        metrics_result = get_metrics(true_labels_binary, pred_scores)
        evaluate_result = evaluate(pred_scores, true_labels_binary)

        result = {
            "roc_auc": evaluate_result["auroc"],
            "prauc": evaluate_result["aupr"],
            "aupr": evaluate_result["aupr"],
            "auroc": evaluate_result["auroc"],
            "f1_score": metrics_result[2],
            "accuracy": metrics_result[3],
            "recall": metrics_result[4],
            "specificity": metrics_result[5],
            "precision": metrics_result[6],
            "TP": metrics_result[7],
            "FP": metrics_result[8],
            "FN": metrics_result[9],
            "TN": metrics_result[10],
            "pred_scores": pred_scores
        }

    return result