import os


class MultimodalConfig:


    # Basic path configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "Data_HateMM")

    # =================================================================
    # Multimodal feature configuration
    # =================================================================

    # Text modality feature path
    TEXT_PURE_HATE_FEATURE_PATH = 'Feature_embedding/Trimmed_hate_video/bert_pure_hate_features.p'
    TEXT_ORIGINAL_FEATURE_PATH = 'Feature_embedding/Trimmed_hate_video/bert_medium_seq_100_feature_embedding.p'
    TEXT_PURE_NONHATE_FEATURE_PATH = 'Feature_embedding/Trimmed_hate_video/bert_pure_nonhate_in_hate_features.p'

    # Audio modality feature path
    AUDIO_PURE_HATE_FEATURE_PATH = 'Feature_embedding/Trimmed_hate_video/mfcc_pure_hate_seg_features.p'
    AUDIO_ORIGINAL_FEATURE_PATH = 'Feature_embedding/Trimmed_hate_video/all_audio_mfcc_features_float32_CMFusion.p'
    AUDIO_PURE_NONHATE_FEATURE_PATH = 'Feature_embedding/Trimmed_hate_video/mfcc_pure_nonhate_in_hate_seg_features.p'

    # Video modality feature path
    VIDEO_PURE_HATE_FEATURE_PATH = 'Feature_embedding/Trimmed_hate_video/vit_pure_hate_seg_features.p'
    VIDEO_ORIGINAL_FEATURE_PATH = 'Feature_embedding/Trimmed_hate_video/all_vit_features_CMFusion.p'
    VIDEO_PURE_NONHATE_FEATURE_PATH = 'Feature_embedding/Trimmed_hate_video/vit_pure_nonhate_in_hate_seg_features.p'

    # Feature dimension configuration
    TEXT_FEATURE_DIM = 768      # BERT
    AUDIO_FEATURE_DIM = 40      # MFCC
    VIDEO_FEATURE_DIM = 768     # ViT
    VIDEO_SEQ_LEN = 100         # Sequence length

    # =================================================================
    # Manually modify here to switch experiments
    # ‘Full’ here refers to untrimmed raw video, while ‘pure’ refers to trimmed video.
    # =================================================================

    # Training data combination control
    TRAIN_HATE_TYPE = "pure"        # What features should be used for hate videos: ‘full’ or ‘pure
    TRAIN_NONHATE_TYPE = "full"     # What features should be used for non-hate videos: ‘full’ or ‘pure’?
    TRAIN_INCLUDE_PURE_NONHATE = False  # Whether to add additional trimmed non-hate segments

    # Test data combination control (validation set also follows test set)
    TEST_HATE_TYPE = "pure"         # What features should be used for hate videos during testing: ‘full’ or ‘pure’?
    TEST_INCLUDE_NONHATE = True   # Does the test include non-hate videos (set to False to test only hate videos)?

    # =================================================================
    # odel Architecture Configuration
    # =================================================================

    # Text
    TEXT_FC1_HIDDEN = 128
    TEXT_FC2_HIDDEN = 128
    TEXT_OUTPUT_SIZE = 128 #HateMM changed to 64, MHC to 128

    # Audi
    AUDIO_FC1_HIDDEN = 128
    AUDIO_FC2_HIDDEN = 128
    AUDIO_OUTPUT_SIZE = 128 #HateMM changed to 64, MHC to 128
    AUDIO_LSTM_HIDDEN = 128

    # Video LSTM模型配置
    VIDEO_LSTM_HIDDEN = 128
    VIDEO_FC_HIDDEN = 128
    VIDEO_OUTPUT_SIZE = 128 #HateMM changed to 64, MHC to 128

    VIDEO_FC1_HIDDEN = 128
    VIDEO_FC2_HIDDEN =  128

    # 融合模型配置
    FUSION_INPUT_SIZE = 3 * 128  # text_out + video_out + audio_out
    NUM_CLASSES = 2

    # =================================================================
    # Training Configuration
    # =================================================================

    # Random seed
    SEED = 2021

    # five fold data file path
    FOLD_DATA_PATH = 'five_folds.pickle'

    # Training hyperparameters
    NUM_EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    DROPOUT = 0.0

    # Output control
    PRINT_EVERY = 10
    VERBOSE = True

    # MLP model structure (consistent with single modality)
    HIDDEN_DIM1 = 128
    HIDDEN_DIM2 = 128