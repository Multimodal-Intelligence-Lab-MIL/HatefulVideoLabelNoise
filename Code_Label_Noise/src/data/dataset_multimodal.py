import torch
import numpy as np
import pickle
import os
from torch.utils.data import Dataset
import re

class MultimodalHybridDataset(Dataset):


    def __init__(self, data_list, labels, features_dict, config,
                 is_test=False, is_val=False, hate_video_ids=None):
        """
        Initialise multimodal dataset

        Args:
            data_list: List of video IDs
            labels: List of labels
            features_dict: Dictionary containing all modal features
            config: Configuration object
            is_test: Whether it is the testing phase
            is_val: Whether it is the validation phase
            hate_video_ids: Set of hate video IDs
        """
        self.config = config
        self.is_test = is_test
        self.is_val = is_val
        self.features_dict = features_dict

        # Automatically infer hate video IDs
        if hate_video_ids is None:
            self.hate_video_ids = set()
            for video_id, label in zip(data_list, labels):
                if label == 1:
                    self.hate_video_ids.add(video_id)
        else:
            self.hate_video_ids = hate_video_ids

        self.expanded_data = []
        self.expanded_labels = []

        self._expand_dataset(data_list, labels)
        self._detect_feature_formats()

    def _expand_dataset(self, data_list, labels):
        """Extend data set according to configuration"""
        is_training = not self.is_test and not self.is_val
        subset_type = "Train" if is_training else ("Val" if self.is_val else "Test")


        pure_hate_ids = []
        pure_nonhate_ids = []
        full_ids = []

        is_training = not self.is_test and not self.is_val

        for video_id, label in zip(data_list, labels):
            is_hate_video = video_id in self.hate_video_ids

            if is_training:
                # Training phase: Decide which features to use based on the configuration.
                if is_hate_video:
                    if self.config.TRAIN_HATE_TYPE == "pure":

                        segments = self._find_video_segments(video_id, 'text_pure_hate')

                        if segments:
                            for segment_id in segments:
                                self.expanded_data.append(segment_id)
                                self.expanded_labels.append(label)
                        else:
                            self.expanded_data.append(video_id)
                            self.expanded_labels.append(label)
                    else:  # TRAIN_HATE_TYPE == "full"

                        self.expanded_data.append(video_id)
                        self.expanded_labels.append(label)
                else:
                    # non-hate视频
                    if self.config.TRAIN_NONHATE_TYPE == "pure":
                        # Use pure non-hate clips (non-hate parts extracted from hate videos).
                        #
                        segments = self._find_video_segments(video_id, 'text_pure_nonhate')
                        if segments:
                            for segment_id in segments:
                                self.expanded_data.append(segment_id)
                                self.expanded_labels.append(label)
                        else:

                            self.expanded_data.append(video_id)
                            self.expanded_labels.append(label)
                    else:  # TRAIN_NONHATE_TYPE == "full"

                        self.expanded_data.append(video_id)
                        self.expanded_labels.append(label)

                # Add additional pure non-hate segments (if enabled in configuration)
                if self.config.TRAIN_INCLUDE_PURE_NONHATE and not is_hate_video:
                    # Add an additional pure non-hate clip corresponding to each non-hate video.
                    segments = self._find_video_segments(video_id, 'text_pure_nonhate')
                    for segment_id in segments:
                        self.expanded_data.append(segment_id)
                        self.expanded_labels.append(0)

            else:
                # Testing/verification phase: Decide which features to use based on the configuration.
                if is_hate_video:
                    if self.config.TEST_HATE_TYPE == "pure":

                        segments = self._find_video_segments(video_id, 'text_pure_hate')

                        if segments:
                            for segment_id in segments:
                                self.expanded_data.append(segment_id)
                                self.expanded_labels.append(label)
                        else:
                            self.expanded_data.append(video_id)
                            self.expanded_labels.append(label)
                    else:  # TEST_HATE_TYPE == "full"

                        self.expanded_data.append(video_id)
                        self.expanded_labels.append(label)
                else:

                    if self.config.TEST_INCLUDE_NONHATE:

                        self.expanded_data.append(video_id)
                        self.expanded_labels.append(label)


        print(f"Finish: {len(data_list)} -> {len(self.expanded_data)} samples")
        # ======= Statistical output =======
        total = len(self.expanded_data)
        unique_videos = set([x.split('_')[0] if '_' in x else x for x in self.expanded_data])
        #print(f"Total: {len(unique_videos)}")
        #print(f"Pure Hate segment: {len(pure_hate_ids)}")
        #print(f"Pure Non-Hate segment: {len(pure_nonhate_ids)}")
        #print(f"Full video-level: {len(full_ids)}")
        #print(f"First 10 sample ID: {self.expanded_data[:10]}")

        # ======= 写入文件 =======
        filename = f"{subset_type.lower()}_ids.txt"
        with open(filename, "w") as f:
            for vid in self.expanded_data:
                f.write(f"{vid}\n")
        print(f"ID list has been written：{filename}\n")
    '''
    def _find_video_segments(self, video_id, feature_type):
        
        segments = []
        feature_data = self.features_dict[feature_type]
        for key in feature_data.keys():
            if key.startswith(video_id + "_seg_"):
                segments.append(key)
        return sorted(segments)'''

    def _find_video_segments(self, video_id, feature_type):
        segments = []
        feature_data = self.features_dict[feature_type]
        pattern = re.compile(rf"^{re.escape(video_id)}(_hate)?_seg_\d+")
        for key in feature_data.keys():
            if pattern.match(key):
                segments.append(key)
        return sorted(segments)

    def _detect_feature_formats(self):

        self.feature_dims = {}


        if len(self.features_dict['text_original']) > 0:
            sample_key = next(iter(self.features_dict['text_original']))
            sample_feature = self.features_dict['text_original'][sample_key]
            self.feature_dims['text'] = np.array(sample_feature).shape[-1]
        else:
            self.feature_dims['text'] = self.config.TEXT_FEATURE_DIM


        if len(self.features_dict['audio_original']) > 0:
            sample_key = next(iter(self.features_dict['audio_original']))
            sample_feature = self.features_dict['audio_original'][sample_key]
            feature_shape = np.array(sample_feature).shape
            if len(feature_shape) == 2:
                self.feature_dims['audio'] = feature_shape[-1]
                #self.feature_dims['audio'] = feature_shape  # If it is (60, 40)
            else:
                self.feature_dims['audio'] = feature_shape[0]
        else:
            self.feature_dims['audio'] = self.config.AUDIO_FEATURE_DIM


        if len(self.features_dict['video_original']) > 0:
            sample_key = next(iter(self.features_dict['video_original']))
            sample_feature = self.features_dict['video_original'][sample_key]
            feature_shape = np.array(sample_feature).shape
            if len(feature_shape) == 2:
                self.feature_dims['video'] = feature_shape
            else:
                self.feature_dims['video'] = (1, feature_shape[0])
        else:
            self.feature_dims['video'] = (self.config.VIDEO_SEQ_LEN, self.config.VIDEO_FEATURE_DIM)


        print(f"  Text: {self.feature_dims['text']}")
        print(f"  Audio: {self.feature_dims['audio']}")
        print(f"  Video: {self.feature_dims['video']}")

    def get_feature_dims(self):

        return self.feature_dims

    def __len__(self):
        return len(self.expanded_data)

    def __getitem__(self, idx):

        video_id = self.expanded_data[idx]
        label = self.expanded_labels[idx]


        text_feature = self._get_modality_feature(video_id, 'text')


        audio_feature = self._get_modality_feature(video_id, 'audio')


        video_feature = self._get_modality_feature(video_id, 'video')

        #if text_feature.shape != (768,):
        #    print(f"[unnormal] text_feature shape: {text_feature.shape} | video_id: {video_id}")
        #if audio_feature.shape != (40,):
        #    print(f"[unnormal] audio_feature shape: {audio_feature.shape} | video_id: {video_id}")
        #if video_feature.shape != (60, 768):
        #    print(f"[unnormal] video_feature shape: {video_feature.shape} | video_id: {video_id}")

        return (
            torch.tensor(text_feature, dtype=torch.float32),
            torch.tensor(video_feature, dtype=torch.float32),
            torch.tensor(audio_feature, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

    def _get_modality_feature(self, video_id, modality):

        if "_seg_" in video_id:

            if video_id in self.features_dict[f'{modality}_pure_hate']:
                feature_data = self.features_dict[f'{modality}_pure_hate']
            else:
                feature_data = self.features_dict[f'{modality}_pure_nonhate']
        else:

            feature_data = self.features_dict[f'{modality}_original']


        if video_id in feature_data:
            feature = np.array(feature_data[video_id])


            if modality == 'text':

                if feature.ndim == 2:
                    feature = feature.mean(axis=0)
                    pass
                elif feature.ndim == 1:
                    pass
                else:
                    raise ValueError(f"Unsupported Text feature dimensions: {feature.shape}")

            elif modality == 'audio':

                if feature.ndim == 2:
                    #pass
                    feature = feature.mean(axis=0) #Use MHC and HateMM to modify back.
                elif feature.ndim == 1:
                    if feature.shape[0] != self.feature_dims['audio']:
                        print(f"[Error] Audio feature dimensions are inconsistent!shape={feature.shape}")
                        feature = np.zeros((self.feature_dims['audio'],))
                else:
                    raise ValueError(f"Unsupported audio feature dimensions: {feature.shape}")

            elif modality == 'video':

                if feature.ndim == 1:

                    target_seq_len = self.config.VIDEO_SEQ_LEN
                    feature_dim = len(feature)

                    if feature_dim >= target_seq_len:
                        feature = feature[:target_seq_len]
                    else:
                        repeat_times = target_seq_len // feature_dim + 1
                        feature = np.tile(feature, repeat_times)[:target_seq_len]
                    feature = feature.reshape(target_seq_len, 1)
                elif feature.ndim == 2:

                    seq_len, feat_dim = feature.shape
                    target_seq_len = self.config.VIDEO_SEQ_LEN
                    if seq_len != target_seq_len:

                        indices = np.linspace(0, seq_len - 1, target_seq_len).astype(int)
                        feature = feature[indices]
                else:
                    raise ValueError(f"Unsupported video feature dimensions: {feature.shape}")
        else:

            if modality == 'text':
                feature = np.zeros((self.feature_dims['text'],))
            elif modality == 'audio':
                feature = np.zeros((self.feature_dims['audio'],))
            elif modality == 'video':
                feature = np.zeros(self.feature_dims['video'])

        return feature


class MultimodalHybridDataLoader:


    def __init__(self, config):
        self.config = config
        self.features_dict = {}
        self.hate_video_ids = set()

        self._load_all_features()
        self._identify_hate_videos()

    def _load_all_features(self):

        self._load_modality_features('text')

        self._load_modality_features('audio')

        self._load_modality_features('video')

    def _load_modality_features(self, modality):

        modality_upper = modality.upper()


        pure_hate_path = getattr(self.config, f'{modality_upper}_PURE_HATE_FEATURE_PATH')
        with open(os.path.join(self.config.DATA_PATH, pure_hate_path), 'rb') as f:
            self.features_dict[f'{modality}_pure_hate'] = pickle.load(f)
        print(
            f"{modality.capitalize()} pure hate: {len(self.features_dict[f'{modality}_pure_hate'])} ")


        original_path = getattr(self.config, f'{modality_upper}_ORIGINAL_FEATURE_PATH')
        with open(os.path.join(self.config.DATA_PATH, original_path), 'rb') as f:
            self.features_dict[f'{modality}_original'] = pickle.load(f)
        print(f"{modality.capitalize()} original: {len(self.features_dict[f'{modality}_original'])} ")


        pure_nonhate_path = getattr(self.config, f'{modality_upper}_PURE_NONHATE_FEATURE_PATH')
        with open(os.path.join(self.config.DATA_PATH, pure_nonhate_path), 'rb') as f:
            self.features_dict[f'{modality}_pure_nonhate'] = pickle.load(f)
        print(
            f"{modality.capitalize()} pure non-hate: {len(self.features_dict[f'{modality}_pure_nonhate'])} ")

    def _identify_hate_videos(self):

        for seg_id in self.features_dict['text_pure_hate'].keys():
            # hate_zvqFZAYn4fg_hate_seg_1
            if "_seg_" in seg_id:
                # remove "_hate_seg_x" or "_seg_x"
                if "_hate_seg_" in seg_id:
                    video_id = seg_id.rsplit("_hate_seg_", 1)[0]
                else:
                    video_id = seg_id.rsplit("_seg_", 1)[0]

                #print(f"[Debugging] Extracted from seg_id: {seg_id} => video_id: {video_id}")
                self.hate_video_ids.add(video_id)

        print(f"[Debugging] hate_video_ids total identified: {len(self.hate_video_ids)} ")

    def create_dataset(self, data_list, labels, is_test=False, is_val=False):

        return MultimodalHybridDataset(
            data_list=data_list,
            labels=labels,
            features_dict=self.features_dict,
            config=self.config,
            is_test=is_test,
            is_val=is_val,
            hate_video_ids=self.hate_video_ids
        )