import torch
import torch.nn as nn


class TextModel(nn.Module):


    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Linear(fc2_hidden, output_size),
        )

    def forward(self, xb):
        return self.network(xb)


class VideoLSTM(nn.Module):


    def __init__(self, input_emb_size=768, no_of_frames=100, lstm_hidden=128, fc_output=64):
        super(VideoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_emb_size, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden * no_of_frames, fc_output)
        self.no_of_frames = no_of_frames

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        x, _ = self.lstm(x)

        x = x.contiguous().view(x.shape[0], -1)

        if x.shape[1] != self.fc.in_features:

            x = x.view(x.shape[0], -1, x.shape[1] // self.no_of_frames)
            x = torch.mean(x, dim=2)
            x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class AudioModel(nn.Module):


    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Linear(fc2_hidden, output_size),
        )

    def forward(self, xb):
        return self.network(xb)


class MultimodalFusionModel(nn.Module):


    def __init__(self, text_model, video_model, audio_model, num_classes=2):
        super().__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        self.video_model = video_model

        #  (3 * 64 = 192 -> num_classes)
        self.fc_output = nn.Linear(3 * 64, num_classes)

    def forward(self, x_text, x_video, x_audio):

        text_out = self.text_model(x_text)
        video_out = self.video_model(x_video)
        audio_out = self.audio_model(x_audio)


        fused_features = torch.cat((text_out, video_out, audio_out), dim=1)


        output = self.fc_output(fused_features)
        return output


class MultimodalModelFactory:


    @staticmethod
    def create_multimodal_model(config, device):



        text_model = TextModel(
            input_size=config.TEXT_FEATURE_DIM,
            fc1_hidden=config.TEXT_FC1_HIDDEN,
            fc2_hidden=config.TEXT_FC2_HIDDEN,
            output_size=config.TEXT_OUTPUT_SIZE
        ).to(device)

        video_model = VideoLSTM(
            input_emb_size=config.VIDEO_FEATURE_DIM,
            no_of_frames=config.VIDEO_SEQ_LEN,
            lstm_hidden=config.VIDEO_LSTM_HIDDEN,
            fc_output=config.VIDEO_OUTPUT_SIZE
        ).to(device)

        audio_model = AudioModel(
            input_size=config.AUDIO_FEATURE_DIM,
            fc1_hidden=config.AUDIO_FC1_HIDDEN,
            fc2_hidden=config.AUDIO_FC2_HIDDEN,
            output_size=config.AUDIO_OUTPUT_SIZE
        ).to(device)


        multimodal_model = MultimodalFusionModel(
            text_model=text_model,
            video_model=video_model,
            audio_model=audio_model,
            num_classes=config.NUM_CLASSES
        ).to(device)

        return multimodal_model

    @staticmethod
    def get_model_info(model):

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel parameter statistics:")
        print(f"  Total number of parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")


        text_params = sum(p.numel() for p in model.text_model.parameters())
        video_params = sum(p.numel() for p in model.video_model.parameters())
        audio_params = sum(p.numel() for p in model.audio_model.parameters())
        fusion_params = sum(p.numel() for p in model.fc_output.parameters())

        print(f"  Text: {text_params:,}")
        print(f"  Video: {video_params:,}")
        print(f"  Audio: {audio_params:,}")
        print(f"  Fusion: {fusion_params:,}")

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'text_params': text_params,
            'video_params': video_params,
            'audio_params': audio_params,
            'fusion_params': fusion_params
        }


import torch
import torch.nn as nn


class TextModel_MHC(nn.Module):


    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU()
        )

        self.fc_output = nn.Linear(fc2_hidden, output_size)

    def forward(self, xb):
        x = self.fc1(xb)
        x = self.fc2(x)
        x = self.fc_output(x)
        return x

class VideoModel_MHC(nn.Module):


    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)


        self.fc1 = nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU()
        )

        self.fc_output = nn.Linear(fc2_hidden, output_size)

    def forward(self, x):

        if len(x.shape) == 3:

            x = x.transpose(1, 2)

            x = self.temporal_pool(x)

            x = x.squeeze(-1)
        elif len(x.shape) > 3:

            batch_size = x.shape[0]
            x = x.view(batch_size, x.shape[1], -1)
            x = x.transpose(1, 2)
            x = self.temporal_pool(x)
            x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_output(x)
        return x


class AudioModel_MHC(nn.Module):


    def __init__(self, input_size, output_size):
        super().__init__()
        # 单个FC层直接输出
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, xb):
        return self.fc(xb)


class MultimodalFusionModel_MHC(nn.Module):


    def __init__(self, text_model, video_model, audio_model, fusion_hidden=128, num_classes=2):
        super().__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        self.video_model = video_model


        self.fusion_fc1 = nn.Sequential(
            nn.Linear(3 * 128, fusion_hidden),
            nn.ReLU()
        )


        self.fusion_fc2 = nn.Linear(fusion_hidden, num_classes)

    def forward(self, x_text, x_video, x_audio):

        text_out = self.text_model(x_text)
        video_out = self.video_model(x_video)
        audio_out = self.audio_model(x_audio)


        fused_features = torch.cat((text_out, video_out, audio_out), dim=1)


        x = self.fusion_fc1(fused_features)
        output = self.fusion_fc2(x)

        return output


class MultimodalModelFactory_MHC:


    @staticmethod
    def create_multimodal_model(config, device):


        text_model = TextModel_MHC(
            input_size=config.TEXT_FEATURE_DIM,
            fc1_hidden=config.TEXT_FC1_HIDDEN,
            fc2_hidden=config.TEXT_FC2_HIDDEN,
            output_size=config.TEXT_OUTPUT_SIZE
        ).to(device)

        video_model = VideoModel_MHC(
            input_size=config.VIDEO_FEATURE_DIM,
            fc1_hidden=config.VIDEO_FC1_HIDDEN,
            fc2_hidden=config.VIDEO_FC2_HIDDEN,
            output_size=config.VIDEO_OUTPUT_SIZE
        ).to(device)

        audio_model = AudioModel_MHC(
            input_size=config.AUDIO_FEATURE_DIM,
            output_size=config.AUDIO_OUTPUT_SIZE
        ).to(device)


        multimodal_model = MultimodalFusionModel_MHC(
            text_model=text_model,
            video_model=video_model,
            audio_model=audio_model,
            fusion_hidden=config.FUSION_HIDDEN if hasattr(config, 'FUSION_HIDDEN') else 128,
            num_classes=config.NUM_CLASSES
        ).to(device)

        return multimodal_model

    @staticmethod
    def get_model_info(model):

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel parameter statistics:")
        print(f"  Total number of parameters: {total_params:,}")
        print(f"  trainable parameters: {trainable_params:,}")

        # 各子模型参数统计
        text_params = sum(p.numel() for p in model.text_model.parameters())
        video_params = sum(p.numel() for p in model.video_model.parameters())
        audio_params = sum(p.numel() for p in model.audio_model.parameters())
        fusion_params = sum(p.numel() for p in model.fusion_fc1.parameters()) + \
                        sum(p.numel() for p in model.fusion_fc2.parameters())

        print(f"  Text: {text_params:,}")
        print(f"  Video: {video_params:,}")
        print(f"  Audio: {audio_params:,}")
        print(f"  Fusion: {fusion_params:,}")

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'text_params': text_params,
            'video_params': video_params,
            'audio_params': audio_params,
            'fusion_params': fusion_params
        }


