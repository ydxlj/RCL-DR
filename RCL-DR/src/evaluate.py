import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn import metrics
import logging


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.sort(np.unique(predict_score.flatten()))
    thresholds = sorted_predict_score[np.int32(
        np.linspace(0, len(sorted_predict_score) - 1, 1000)
    )]

    tpr_list, fpr_list, precision_list = [], [], []

    for thresh in thresholds:
        binary_pred = (predict_score >= thresh).astype(int)

        tp = np.sum((binary_pred == 1) & (real_score == 1))
        fp = np.sum((binary_pred == 1) & (real_score == 0))
        fn = np.sum((binary_pred == 0) & (real_score == 1))
        tn = np.sum((binary_pred == 0) & (real_score == 0))
        tpr = tp / (tp + fn + 1e-7)
        fpr = fp / (fp + tn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        precision_list.append(precision)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr = metrics.auc(tpr_list, precision_list)

    f1_scores = 2 * np.array(precision_list) * np.array(tpr_list) / (
            np.array(precision_list) + np.array(tpr_list) + 1e-7)
    best_idx = np.argmax(f1_scores)

    best_f1 = f1_scores[best_idx]
    best_acc = (tp + tn) / (tp + tn + fp + fn)
    best_recall = tpr_list[best_idx]
    best_precision = precision_list[best_idx]
    best_specificity = 1 - fpr_list[best_idx]

    return {
        "auroc": auroc,
        "aupr": aupr,
        "f1": best_f1,
        "accuracy": best_acc,
        "recall": best_recall,
        "precision": best_precision,
        "specificity": best_specificity
    }


def evaluate_model(model, data_matrix, drug_names, disease_names):
    model.eval()
    with torch.no_grad():
        if isinstance(data_matrix, np.ndarray):
            data_matrix = sp.csr_matrix(data_matrix)

        full_matrix = sp.csr_matrix((model.n_drugs, model.n_diseases))
        full_matrix[:data_matrix.shape[0], :data_matrix.shape[1]] = data_matrix

        drug_emb, disease_emb, _ = model.forward()

        scores = torch.sigmoid(torch.mm(drug_emb, disease_emb.t())).cpu().numpy()

        true_labels = full_matrix.toarray().flatten()
        pred_scores = scores.flatten()
        metrics = get_metrics(true_labels, pred_scores)

        disease_scores = scores[:, disease_idx]
        top_drug_indices = np.argsort(-disease_scores)[:10]
        disease_id = disease_names[disease_idx][0] if disease_names.size > 0 else f"D{disease_idx}"

        logging.info(f"Top predictions for disease {disease_id}:")
        for rank, drug_idx in enumerate(top_drug_indices):
            drug_id = drug_names[drug_idx][0] if drug_names.size > 0 else f"Drug{drug_idx}"
            logging.info(f"Rank {rank + 1}: {drug_id} - Score: {disease_scores[drug_idx]:.4f}")

        return metrics