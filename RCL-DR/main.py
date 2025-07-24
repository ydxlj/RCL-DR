import os
import time
import yaml
import scipy.io as sio
import numpy as np
import torch
from sklearn.model_selection import KFold
from models.ncl import NCL
from utils.dataloader import load_data
from utils.trainer import Trainer
from utils.evaluate import calculate_auroc_aupr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_kfold_experiment(data, model_params, optimizer_params, kfold_params,
                         training_params, drug_names, disease_names):
    results = []
    all_auroc_scores = []
    all_aupr_scores = []
    all_y_true = []
    all_y_pred = []

    output_dir = training_params.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    num_repeats = training_params.get('num_repeats', 1)
    apply_smote = training_params.get('apply_smote', True)

    for repeat in range(num_repeats):
        logger.info(f"Starting repeat {repeat + 1}/{num_repeats}")
        kf = KFold(**kfold_params)

        for fold, (train_index, val_index) in enumerate(kf.split(data)):
            logger.info(f"Starting fold {fold + 1}/{kf.n_splits}")
            train_data = data[train_index]
            val_data = data[val_index]

            # 确保数据是稀疏矩阵格式
            if not isinstance(train_data, sp.csr_matrix):
                train_data = sp.csr_matrix(train_data)
            if not isinstance(val_data, sp.csr_matrix):
                val_data = sp.csr_matrix(val_data)

            model = NCL(**model_params)
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trainer = Trainer(
                model, optimizer,
                num_m_step=5,
                device=device,
                patience=training_params.get('patience', 10)
            )

            trainer.fit(
                train_data,
                val_data,
                epochs=training_params.get('epochs', 100),
                batch_size=training_params.get('batch_size', 256),
                eval_step=training_params.get('eval_step', 1),
                apply_smote=apply_smote,
                drug_names=drug_names,
                disease_names=disease_names
            )

            val_metrics = calculate_auroc_aupr(model, val_data, drug_names, disease_names)
            all_auroc_scores.append(val_metrics["roc_auc"])
            all_aupr_scores.append(val_metrics["prauc"])
            all_y_true.append(val_data.toarray().flatten())
            all_y_pred.append(val_metrics["pred_scores"])

            result_file = os.path.join(output_dir, f"repeat_{repeat}_fold_{fold}.txt")
            with open(result_file, 'w') as f:
                f.write("True Labels\tPredicted Scores\n")
                for true_label, pred_score in zip(val_data.toarray().flatten(), val_metrics["pred_scores"]):
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

    logger.info(f"Average AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")
    logger.info(f"Average AUPR: {avg_aupr:.4f} ± {std_aupr:.4f}")

    return results, avg_auroc, avg_aupr, all_y_true, all_y_pred


def main():
    config = load_config("config/config.yaml")

    # 加载数据
    data_path = config['data']['path']
    drug_disease_matrix, drug_names, disease_names = load_data(data_path)
    drug_disease_matrix = drug_disease_matrix.T

    # 运行实验
    results, avg_auroc, avg_aupr, all_y_true, all_y_pred = run_kfold_experiment(
        drug_disease_matrix,
        model_params=config['model_params'],
        optimizer_params=config['optimizer_params'],
        kfold_params=config['kfold_params'],
        training_params={
            'num_repeats': config['training']['num_repeats'],
            'output_dir': config['data']['output_dir'],
            'apply_smote': config['training']['apply_smote'],
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'eval_step': config['training']['eval_step'],
            'patience': config['training']['patience']
        },
        drug_names=drug_names,
        disease_names=disease_names
    )

    # 保存最终结果
    summary_file = os.path.join(config['data']['output_dir'], "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Average AUROC: {avg_auroc:.4f}\n")
        f.write(f"Average AUPR: {avg_aupr:.4f}\n")
        for i, result in enumerate(results):
            f.write(
                f"Repeat {result['repeat']}, Fold {result['fold']}: Valid Score = {result['best_valid_score']:.4f}\n")

    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()