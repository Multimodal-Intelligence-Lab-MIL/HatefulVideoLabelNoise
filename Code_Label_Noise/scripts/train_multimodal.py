import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.multimodal_config import MultimodalConfig
from src.utils.seed import set_seed
from src.data.dataset_multimodal import MultimodalHybridDataLoader
from src.modules.multimodal_models import MultimodalModelFactory, MultimodalModelFactory_MHC
from src.training.trainer_multimodal import MultimodalTrainer
from src.evaluation.metrics_multimodal import MultimodalMetricsCalculator


class MultimodalHybridEvaluator:


    def __init__(self):
        self.config = MultimodalConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        set_seed(self.config.SEED)

        print("CUDA available:", torch.cuda.is_available())
        print("Using device:", self.device)


        self._print_experiment_config()


        self.data_loader = MultimodalHybridDataLoader(self.config)


        self._load_fold_data()

    def _print_experiment_config(self):

        print(f"\n{'=' * 60}")
        print(f"Multimodal mixed feature experimental configuration")
        print(f"{'=' * 60}")
        print("Training data configuration:")
        print(f"  - Hate video feature types: {self.config.TRAIN_HATE_TYPE}")
        print(f"  - Non-Hate Video Feature Types: {self.config.TRAIN_NONHATE_TYPE}")
        print(f"  - Includes additional Pure Non-Hate footage: {self.config.TRAIN_INCLUDE_PURE_NONHATE}")
        print("Test Data Configuration:")
        print(f"  - Hate video feature types: {self.config.TEST_HATE_TYPE}")
        print(f"  - Does it contain non-hate videos?: {self.config.TEST_INCLUDE_NONHATE}")
        print(f"{'=' * 60}")

    def _load_fold_data(self):

        with open(os.path.join(self.config.DATA_PATH, self.config.FOLD_DATA_PATH), 'rb') as f:
            self.fold_data = pickle.load(f)

    def run_evaluation(self):

        fold_results = []


        for fold_id in range(1, 6):
            print(f"\n--- 多模态 Fold {fold_id} ---")


            train_list, train_label = self.fold_data[f'Fold_{fold_id}']['train']
            val_list, val_label = self.fold_data[f'Fold_{fold_id}']['val']
            test_list, test_label = self.fold_data[f'Fold_{fold_id}']['test']


            train_dataset = self.data_loader.create_dataset(train_list, train_label, is_test=False, is_val=False)
            val_dataset = self.data_loader.create_dataset(val_list, val_label, is_test=False, is_val=True)
            test_dataset = self.data_loader.create_dataset(test_list, test_label, is_test=True)


            if fold_id == 1:
                feature_dims = train_dataset.get_feature_dims()
                print(f"Actual feature dimension: {feature_dims}")


            train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE)
            test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE)


            model = MultimodalModelFactory_MHC.create_multimodal_model(self.config, self.device)


            if fold_id == 1:
                MultimodalModelFactory_MHC.get_model_info(model)


            trainer = MultimodalTrainer(model, self.config, self.device)


            for epoch in range(self.config.NUM_EPOCHS):

                train_loss = trainer.train_epoch(train_loader)


                val_loss = trainer.validate(val_loader)
                val_acc, val_macro_f1, val_weighted_f1 = MultimodalMetricsCalculator.evaluate_validation(
                    model, val_loader, self.device
                )
                test_acc, test_macro_f1, test_weighted_f1 = MultimodalMetricsCalculator.evaluate_validation(model, test_loader, self.device)


                should_print = (
                        self.config.VERBOSE or
                        epoch % self.config.PRINT_EVERY == 0 or
                        epoch == self.config.NUM_EPOCHS - 1
                )

                if should_print:
                    print(f"Epoch {epoch + 1:2d}/{self.config.NUM_EPOCHS}: "
                          f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                          f"Val Accuracy={val_acc:.4f}, Macro F1={val_macro_f1:.4f}"
                          f"Test Accuracy={test_acc:.4f}, Macro F1={test_macro_f1:.4f}")


                is_best = trainer.update_best_model(val_macro_f1)
                if is_best:
                    print(f"  --> Best model updated at epoch {epoch + 1}! (Macro F1: {val_macro_f1:.4f})")


            model_path = os.path.join(
                self.config.BASE_DIR,
                f"multimodal_fold_{fold_id}.pth"
            )
            trainer.save_model(model_path)
            print(f"[*] Best model saved to {model_path}")


            trainer.load_best_model()
            test_metrics = MultimodalMetricsCalculator.evaluate_test(model, test_loader, self.device)

            acc, macro_f1, f1_h, auc_score, precision_h, recall_h = test_metrics
            print(f"==> [Fold {fold_id} Test] "
                  f"Acc={acc:.4f}, MacroF1={macro_f1:.4f}, F1(H)={f1_h:.4f}, "
                  f"AUC={auc_score:.4f}, Prec(H)={precision_h:.4f}, Recall(H)={recall_h:.4f}")


            fold_results.append({
                'accuracy': acc, 'macro_f1': macro_f1, 'f1_h': f1_h,
                'auc': auc_score, 'precision_h': precision_h, 'recall_h': recall_h
            })


        self._print_summary(fold_results)

        return fold_results

    def _print_summary(self, fold_results):
        print(f"\n{'=' * 60}")

        metrics = ['accuracy', 'macro_f1', 'f1_h', 'auc', 'precision_h', 'recall_h']
        metric_names = ['Accuracy', 'Macro F1', 'F1 (H)', 'AUC', 'Precision (H)', 'Recall (H)']

        for metric, name in zip(metrics, metric_names):
            values = [result[metric] for result in fold_results]
            avg = np.mean(values)
            std = np.std(values, ddof=1)
            print(f"{name:12s}: {avg:.4f} ± {std:.4f}")

        print(f"\n{'=' * 60}")
        print("Finish")
        print(f"{'=' * 60}")

def main():
    evaluator = MultimodalHybridEvaluator()
    results = evaluator.run_evaluation()
    return results


if __name__ == "__main__":
    main()