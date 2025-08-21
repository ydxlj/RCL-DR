import time
import torch
import logging
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = model.device
        self.best_valid_score = -np.inf
        self.patience_counter = 0
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.output_dir, 'best_model.pth')

    def fit(self, train_data, val_data, drug_names, disease_names):
        train_loader = self._create_data_loader(
            train_data,
            self.config['batch_size'],
            self.config['apply_smote']
        )

        val_loader = None
        if val_data is not None:
            val_loader = self._create_data_loader(
                val_data,
                self.config['batch_size'],
                apply_smote=False
            )
        for epoch in range(self.config['epochs']):
            start_time = time.time()
            if epoch % self.config['num_m_step'] == 0:
                self.model.e_step()
            train_loss = self._train_epoch(train_loader, epoch)
            epoch_time = time.time() - start_time
            if (epoch + 1) % self.config['eval_step'] == 0 or epoch == self.config['epochs'] - 1:
                log_msg = f"Epoch {epoch + 1}/{self.config['epochs']} - " \
                          f"Time: {epoch_time:.1f}s - " \
                          f"Train Loss: {train_loss:.4f}"

                if val_loader is not None:
                    from .evaluate import evaluate_model
                    val_metrics = evaluate_model(self.model, val_data, drug_names, disease_names)

                    log_msg += f" - Val AUROC: {val_metrics['auroc']:.4f} - " \
                               f"Val AUPR: {val_metrics['aupr']:.4f}"
                    if val_metrics['auroc'] > self.best_valid_score:
                        self.best_valid_score = val_metrics['auroc']
                        self.patience_counter = 0
                        torch.save(self.model.state_dict(), self.best_model_path)
                        logger.info("Saved new best model")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config['patience']:
                            logger.info("Early stopping triggered")
                            break

                logger.info(log_msg)

    def _train_epoch(self, data_loader, epoch_idx):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(
            data_loader,
            desc=f"Epoch {epoch_idx + 1}",
            leave=False,
            mininterval=1.0
        )

        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(batch)
            loss.backward()
            if self.model.clip_grad_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.model.clip_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(data_loader)

    def _create_data_loader(self, data, batch_size, apply_smote):
        from .dataloader import SparseMatrixDataset
        dataset = SparseMatrixDataset(data, apply_smote)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )