import os
import cv2
import torch
import pickle
import numpy as np
import whisper
import threading
import time
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import moviepy.editor as mp
from contextlib import contextmanager



def process_with_simple_timeout(func, args, timeout_seconds):

    import threading
    import time

    result = [None]
    exception = [None]
    finished = [False]

    def target():
        try:
            result[0] = func(*args)
            finished[0] = True
        except Exception as e:
            exception[0] = e
            finished[0] = True


    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()


    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if finished[0]:
            break
        time.sleep(0.1)

    if not finished[0]:

        raise TimeoutError(f"Operation timeout ({timeout_seconds}s)")

    if exception[0]:
        raise exception[0]

    return result[0]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



whisper_model = whisper.load_model("base", device=device)


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()


hatexplain_tokenizer = BertTokenizer.from_pretrained('Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two')
hatexplain_model = BertModel.from_pretrained('Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two').to(device)
hatexplain_model.eval()

# 参数设置
NUM_FRAMES = 100
FEATURE_DIM = 768
VIDEO_ROOT = 'set your path'
TIMEOUT_SECONDS = 120


OUTPUT_TRANSCRIPTS_HATE = 'whisper_base_pure_hate_transcripts.p'
OUTPUT_TRANSCRIPTS_NONHATE = 'whisper_base_pure_nonhate_transcripts.p'
OUTPUT_BERT_FEATURES_HATE = 'bert_pure_hate_features.p'
OUTPUT_BERT_FEATURES_NONHATE = 'bert_pure_nonhate_in_hate_features.p'
OUTPUT_HATEXPLAIN_FEATURES_HATE = 'hatexplain_pure_hate_features.p'
OUTPUT_HATEXPLAIN_FEATURES_NONHATE = 'hatexplain_pure_nonhate_in_hate_features.p'
ERROR_LOG_FILE = 'xxxx/xxxx/text_feature_extraction_errors.txt'


hate_video_paths = []
nonhate_video_paths = []


all_folder_path = os.path.join(VIDEO_ROOT, "ALL")


pure_hate_path = os.path.join(all_folder_path, "pure_hate")
if os.path.isdir(pure_hate_path):
    for file in os.listdir(pure_hate_path):
        if file.endswith(".mp4") and not file.startswith("._"):
            full_path = os.path.join(pure_hate_path, file)
            hate_video_paths.append(full_path)


pure_non_hate_path = os.path.join(all_folder_path, "pure_non_hate")
if os.path.isdir(pure_non_hate_path):
    for file in os.listdir(pure_non_hate_path):
        if file.endswith(".mp4") and not file.startswith("._"):
            full_path = os.path.join(pure_non_hate_path, file)
            nonhate_video_paths.append(full_path)


def generate_key_name(video_path, video_type):

    filename = os.path.basename(video_path)


    name_without_ext = filename.replace(".mp4", "")


    if name_without_ext.startswith("videoID_"):
        new_name = name_without_ext.replace("videoID_", "hate_", 1)
    else:

        new_name = f"hate_{name_without_ext}"

    return new_name


def extract_audio_and_transcribe(video_path):

    try:

        video = mp.VideoFileClip(video_path)
        audio = video.audio


        temp_audio_path = "temp_audio.wav"
        audio.write_audiofile(temp_audio_path, verbose=False, logger=None)


        result = whisper_model.transcribe(temp_audio_path)
        transcript = result["text"]


        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        video.close()
        audio.close()

        return transcript

    except Exception as e:
        raise Exception(f"Failed: {str(e)}")


def text_to_bert_features(text, tokenizer, model, max_length=100):

    try:
        if not text or text.strip() == "":

            return np.zeros((max_length, FEATURE_DIM))


        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return np.zeros((max_length, FEATURE_DIM))

        features_list = []

        with torch.no_grad():
            for sentence in sentences:
                if len(sentence) > 0:

                    inputs = tokenizer(sentence,
                                       return_tensors="pt",
                                       padding=True,
                                       truncation=True,
                                       max_length=512).to(device)


                    outputs = model(**inputs)

                    sentence_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [1, 768]
                    features_list.append(sentence_features[0])

        if not features_list:
            return np.zeros((max_length, FEATURE_DIM))


        features_array = np.array(features_list)  # [num_sentences, 768]


        if len(features_array) >= max_length:

            features_final = features_array[:max_length]
        else:

            features_final = np.zeros((max_length, FEATURE_DIM))
            features_final[:len(features_array)] = features_array

        return features_final

    except Exception as e:
        raise Exception(f"Failed: {str(e)}")


def process_single_video(video_path, video_type):

    key_name = generate_key_name(video_path, video_type)


    transcript = extract_audio_and_transcribe(video_path)


    bert_features = text_to_bert_features(transcript, bert_tokenizer, bert_model)


    hatexplain_features = text_to_bert_features(transcript, hatexplain_tokenizer, hatexplain_model)

    return {
        'name': key_name,
        'transcript': transcript,
        'bert_features': bert_features,
        'hatexplain_features': hatexplain_features
    }


def process_videos(video_paths, error_log_path, video_type):

    transcripts_dict = {}
    bert_features_dict = {}
    hatexplain_features_dict = {}
    error_paths = []

    for video_path in tqdm(video_paths, desc=f"Porcess {video_type} video ({len(video_paths)} )"):
        try:

            result = process_with_simple_timeout(
                process_single_video,
                (video_path, video_type),
                TIMEOUT_SECONDS
            )

            transcripts_dict[result['name']] = result['transcript']
            bert_features_dict[result['name']] = result['bert_features']
            hatexplain_features_dict[result['name']] = result['hatexplain_features']

        except TimeoutError as e:
            error_paths.append(f"{video_path} | Error: Processing timeout ({TIMEOUT_SECONDS}s)")
        except Exception as e:
            error_paths.append(f"{video_path} | Error: {str(e)}")


    if error_paths:
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"\n=== {video_type} Video processing error ===\n")
            ef.write("\n".join(error_paths) + "\n")

    return transcripts_dict, bert_features_dict, hatexplain_features_dict



os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True)
with open(ERROR_LOG_FILE, "w", encoding="utf-8") as ef:
    ef.write("Text feature extraction error log\n")
    ef.write("=" * 50 + "\n")


hate_transcripts, hate_bert_features, hate_hatexplain_features = process_videos(
    hate_video_paths, ERROR_LOG_FILE, "hate"
)


nonhate_transcripts, nonhate_bert_features, nonhate_hatexplain_features = process_videos(
    nonhate_video_paths, ERROR_LOG_FILE, "non_hate"
)




with open(OUTPUT_TRANSCRIPTS_HATE, "wb") as f:
    pickle.dump(hate_transcripts, f)

with open(OUTPUT_TRANSCRIPTS_NONHATE, "wb") as f:
    pickle.dump(nonhate_transcripts, f)


with open(OUTPUT_BERT_FEATURES_HATE, "wb") as f:
    pickle.dump(hate_bert_features, f)

with open(OUTPUT_BERT_FEATURES_NONHATE, "wb") as f:
    pickle.dump(nonhate_bert_features, f)


with open(OUTPUT_HATEXPLAIN_FEATURES_HATE, "wb") as f:
    pickle.dump(hate_hatexplain_features, f)

with open(OUTPUT_HATEXPLAIN_FEATURES_NONHATE, "wb") as f:
    pickle.dump(nonhate_hatexplain_features, f)
