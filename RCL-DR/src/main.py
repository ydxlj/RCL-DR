import os
import yaml
import logging
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold
from .dataloader import load_data
from .model import NCL
from .trainer import Trainer
import torch.optim as optim

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiment(config):
    drug_disease_matrix, drug_names, disease_names = load_data(config['data_path'])
    drug_disease_matrix = drug_disease_matrix.T
    all_auroc = []
    all_aupr = []
    results = []
    os.makedirs(config['output_dir'], exist_ok=True)

    for repeat in range(config['num_repeats']):
        logger.info(f"Starting repeat {repeat + 1}/{config['num_repeats']}")

        kf = KFold(
            n_splits=config['k_folds'],
            shuffle=True,
            random_state=56 + repeat
        )

        for fold, (train_idx, val_idx) in enumerate(kf.split(drug_disease_matrix)):
            logger.info(f"Fold {fold + 1}/{config['k_folds']}")

            train_data = drug_disease_matrix[train_idx]
            val_data = drug_disease_matrix[val_idx]

            model = NCL(config, train_data)
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )

            trainer = Trainer(model, optimizer, config)
            trainer.fit(
                train_data,
                val_data,
                drug_names,
                disease_names
            )

            model.load_state_dict(torch.load(trainer.best_model_path))
            from .evaluate import evaluate_model
            val_metrics = evaluate_model(model, val_data, drug_names, disease_names)

            all_auroc.append(val_metrics['auroc'])
            all_aupr.append(val_metrics['aupr'])
            results.append({
                'repeat': repeat,
                'fold': fold,
                'auroc': val_metrics['auroc'],
                'aupr': val_metrics['aupr']
            })

            logger.info(f"Fold {fold + 1} - AUROC: {val_metrics['auroc']:.4f}, AUPR: {val_metrics['aupr']:.4f}")

    mean_auroc = np.mean(all_auroc)
    std_auroc = np.std(all_auroc)
    mean_aupr = np.mean(all_aupr)
    std_aupr = np.std(all_aupr)

    logger.info("=" * 50)
    logger.info(f"Final Results - AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    logger.info(f"Final Results - AUPR: {mean_aupr:.4f} ± {std_aupr:.4f}")
    logger.info("=" * 50)
    result_file = os.path.join(config['output_dir'], 'results.txt')
    with open(result_file, 'w') as f:
        f.write(f"AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}\n")
        f.write(f"AUPR: {mean_aupr:.4f} ± {std_aupr:.4f}\n")
        f.write("\nPer fold results:\n")
        for res in results:
            f.write(f"Repeat {res['repeat']} Fold {res['fold']}: "
                    f"AUROC={res['auroc']:.4f} AUPR={res['aupr']:.4f}\n")

    return results


if __name__ == "__main__":
    with open("../configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    results = run_experiment(config)