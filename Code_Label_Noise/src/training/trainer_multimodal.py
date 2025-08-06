import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy


class MultimodalTrainer:


    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device


        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


        self.class_weights = torch.FloatTensor([0.41, 0.59]).to(device) #same to hatemm


        self.best_model_weights = None
        self.best_val_macro_f1 = float('-inf')

    def train_epoch(self, dataloader):

        self.model.train()
        total_loss = 0

        for text_features, video_features, audio_features, labels in dataloader:

            text_features = text_features.to(self.device)
            video_features = video_features.to(self.device)
            audio_features = audio_features.to(self.device)
            labels = labels.to(self.device)


            self.optimizer.zero_grad()


            outputs = self.model(text_features, video_features, audio_features)


            loss = F.cross_entropy(outputs, labels, weight=self.class_weights)


            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):

        self.model.eval()
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for text_features, video_features, audio_features, labels in dataloader:

                text_features = text_features.to(self.device)
                video_features = video_features.to(self.device)
                audio_features = audio_features.to(self.device)
                labels = labels.to(self.device)


                outputs = self.model(text_features, video_features, audio_features)


                loss = criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def update_best_model(self, val_macro_f1):

        if val_macro_f1 > self.best_val_macro_f1:
            self.best_val_macro_f1 = val_macro_f1
            self.best_model_weights = copy.deepcopy(self.model.state_dict())
            return True
        return False

    def load_best_model(self):

        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)

    def save_model(self, path):

        if self.best_model_weights is not None:
            torch.save(self.best_model_weights, path)