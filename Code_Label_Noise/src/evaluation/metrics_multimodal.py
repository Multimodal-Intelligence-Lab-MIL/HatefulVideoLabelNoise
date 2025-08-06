import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc


class MultimodalMetricsCalculator:


    @staticmethod
    def evaluate_validation(model, dataloader, device):

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for text_features, video_features, audio_features, labels in dataloader:

                text_features = text_features.to(device)
                video_features = video_features.to(device)
                audio_features = audio_features.to(device)


                outputs = model(text_features, video_features, audio_features)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())


        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

        return acc, macro_f1, weighted_f1

    @staticmethod
    def evaluate_test(model, dataloader, device):

        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for text_features, video_features, audio_features, labels in dataloader:

                text_features = text_features.to(device)
                video_features = video_features.to(device)
                audio_features = audio_features.to(device)


                outputs = model(text_features, video_features, audio_features)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.numpy())


        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        f1_h = f1_score(all_labels, all_preds, average='binary', pos_label=1)


        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        auc_score = auc(fpr, tpr)

        precision_h = precision_score(all_labels, all_preds, pos_label=1)
        recall_h = recall_score(all_labels, all_preds, pos_label=1)

        return acc, macro_f1, f1_h, auc_score, precision_h, recall_h