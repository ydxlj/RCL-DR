import torch
import time
import numpy as np
import logging
from tqdm import tqdm
import os
from torch.nn.utils import clip_grad_norm_
from torch import optim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, optimizer, num_m_step, device, early_stop=None, verbose=False, callbacks=None,
                 patience=10):
        assert num_m_step is not None, "num_m_step cannot be None"
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
            show_progress=False, apply_smote=True, drug_names=None, disease_names=None):
        try:
            train_loader = self._create_data_loader(train_data, batch_size, apply_smote=apply_smote)
            val_loader = self._create_data_loader(val_data, batch_size, apply_smote=False) if val_data is not None else None

            for epoch_idx in range(epochs):
                if epoch_idx % self.num_m_step == 0:
                    self.model.e_step()

                train_start_time = time.time()
                train_loss = self._train_epoch(train_loader, epoch_idx, show_progress=show_progress)
                train_end_time = time.time()

                logger.info(f"Epoch {epoch_idx + 1}/{epochs}, Loss: {train_loss:.4f}, Time: {train_end_time - train_start_time:.2f}s")

                if eval_step <= 0 or val_data is None:
                    continue

                if epoch_idx % eval_step == 0:
                    valid_start_time = time.time()
                    valid_score, valid_result = self._valid_epoch(val_loader, epoch_idx, show_progress=show_progress)
                    valid_end_time = time.time()

                    if self.early_stop:
                        self._update_best_valid(valid_score, valid_result)

                    valid_info = f"Epoch {epoch_idx + 1}/{epochs}, Valid Score: {valid_score:.4f}, Time: {valid_end_time - valid_start_time:.2f}s"
                    if self.verbose:
                        logger.info(valid_info)

                    if self.best_valid_score == valid_score and save_model:
                        self._save_model()

                if self.stop_training:
                    logger.info("Early stopping triggered. Stopping training.")
                    break

            return self.best_valid_score, self.best_valid_result

        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            raise

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

        valid_score = total_score / len(val_loader) if val_loader else 0
        valid_result = {'valid_score': valid_score}
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
        save_dir = "saved_models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    def _create_data_loader(self, data, batch_size, apply_smote=True):
        from utils.dataloader import create_data_loader
        return create_data_loader(data, batch_size, apply_smote=apply_smote)