import os
import cv2
import numpy as np
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel
import torch
import pickle
from datetime import datetime


try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
    model.eval()
    print("ViT model loaded successfully")
except Exception as e:
    print(f"Failed to initialize ViT model: {str(e)}")
    raise


NUM_FRAMES = 60
FEATURE_DIM = 768
VIDEO_ROOT = ''
OUTPUT_FEATURES_HATE = 'vit_pure_hate_seg_features.p'
OUTPUT_FEATURES_NONHATE = 'vit_pure_nonhate_in_hate_seg_features.p'
ERROR_LOG_FILE = '/vit_feature_extraction_errors.txt'



def collect_video_paths(root_dir):

    hate_video_paths = []
    nonhate_video_paths = []

    try:
        if not os.path.exists(root_dir):
            print(f"Root directory not found: {root_dir}")
            return hate_video_paths, nonhate_video_paths


        all_folder_path = os.path.join(root_dir, "ALL")
        if not os.path.exists(all_folder_path):
            print(f"videos folder not found: {all_folder_path}")
            return hate_video_paths, nonhate_video_paths


        pure_hate_path = os.path.join(all_folder_path, "pure_hate")
        if os.path.isdir(pure_hate_path):
            for file in os.listdir(pure_hate_path):
                if file.endswith(".mp4") and not file.startswith("._"):
                    full_path = os.path.join(pure_hate_path, file)
                    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                        hate_video_paths.append(full_path)


        pure_non_hate_path = os.path.join(all_folder_path, "pure_non_hate")
        if os.path.isdir(pure_non_hate_path):
            for file in os.listdir(pure_non_hate_path):
                if file.endswith(".mp4") and not file.startswith("._"):
                    full_path = os.path.join(pure_non_hate_path, file)
                    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                        nonhate_video_paths.append(full_path)

    except Exception as e:
        print(f"Error collecting video paths: {str(e)}")

    print(f"Collected {len(hate_video_paths)} hate videos and {len(nonhate_video_paths)} non-hate videos")
    return hate_video_paths, nonhate_video_paths



def generate_key_name(video_path, video_type):

    filename = os.path.basename(video_path)


    name_without_ext = filename.replace(".mp4", "")


    if name_without_ext.startswith("videoID_"):
        new_name = name_without_ext.replace("videoID_", "hate_video_", 1)
    else:

        new_name = f"hate_video_{name_without_ext}"

    return new_name



def extract_features(video_paths, error_log_path, video_type=""):

    features_dict = {}
    error_paths = []
    successful_count = 0
    skipped_count = 0


    if video_type == "hate":
        try:
            os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
            with open(error_log_path, "w", encoding="utf-8") as ef:
                ef.write(f"ViT feature extraction started at: {datetime.now()}\n")
                ef.write("=" * 50 + "\n")
        except Exception as e:
            print(f"Cannot create error log file: {str(e)}")

    for i, video_path in enumerate(tqdm(video_paths, desc=f"Processing {video_type} videos")):
        try:

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            if os.path.getsize(video_path) == 0:
                raise ValueError(f"Video file is empty: {video_path}")


            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames <= 0:
                cap.release()
                raise ValueError(f"Video has {total_frames} frames")

            if fps <= 0:
                print(f"Invalid FPS ({fps}) for video: {video_path}")


            if total_frames <= NUM_FRAMES:
                frame_indices = list(range(total_frames))
                frame_indices.extend([total_frames - 1] * (NUM_FRAMES - total_frames))
            else:
                step = max(1, total_frames // NUM_FRAMES)
                frame_indices = [min(i * step, total_frames - 1) for i in range(NUM_FRAMES)]

            frames = []
            valid_frame_count = 0

            for idx in frame_indices:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()

                    if ret and frame is not None:

                        if frame.shape[0] > 0 and frame.shape[1] > 0:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                            valid_frame_count += 1
                        else:
                            frames.append(np.ones((224, 224, 3), dtype=np.uint8) * 255)
                    else:
                        frames.append(np.ones((224, 224, 3), dtype=np.uint8) * 255)

                except Exception as frame_error:
                    print(f"Error reading frame {idx} from {video_path}: {str(frame_error)}")
                    frames.append(np.ones((224, 224, 3), dtype=np.uint8) * 255)

            cap.release()


            if valid_frame_count == 0:
                raise ValueError(f"No valid frames extracted from video: {video_path}")

            if valid_frame_count < NUM_FRAMES * 0.1:
                print(f"Only {valid_frame_count}/{NUM_FRAMES} valid frames for: {video_path}")


            try:
                with torch.no_grad():
                    inputs = extractor(frames, return_tensors="pt")

                    if inputs['pixel_values'].shape[0] != NUM_FRAMES:
                        raise ValueError(f"Unexpected input shape: {inputs['pixel_values'].shape}")

                    inputs = inputs.to(device)
                    outputs = model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [100, 768]


                    if features.shape != (NUM_FRAMES, FEATURE_DIM):
                        raise ValueError(f"Unexpected feature shape: {features.shape}")

                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        raise ValueError("Features contain NaN or Inf values")

            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory for video: {video_path}")
                torch.cuda.empty_cache()
                raise
            except Exception as model_error:
                raise ValueError(f"Model inference error: {str(model_error)}")


            key_name = generate_key_name(video_path, video_type)
            features_dict[key_name] = features
            successful_count += 1


            if (i + 1) % 50 == 0:
                print(
                    f"Processed {i + 1}/{len(video_paths)} {video_type} videos. Success: {successful_count}, Skipped: {skipped_count}")

        except KeyboardInterrupt:
            print("Processing interrupted by user")
            break

        except Exception as e:
            error_msg = f"Video: {video_path} | Error: {str(e)} | Type: {type(e).__name__}"
            error_paths.append(error_msg)
            skipped_count += 1

            print(f"Skipping {video_type} video {i + 1}/{len(video_paths)}: {os.path.basename(video_path)} - {str(e)}")


            if "CUDA" in str(e) or "memory" in str(e).lower():
                torch.cuda.empty_cache()


    if error_paths:
        try:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"\n{video_type.upper()} videos processing completed at: {datetime.now()}\n")
                ef.write(f"Total {video_type} videos: {len(video_paths)}\n")
                ef.write(f"Successful: {successful_count}\n")
                ef.write(f"Failed: {skipped_count}\n")
                ef.write("-" * 30 + "\n")
                ef.write("\n".join(error_paths) + "\n")
                ef.write("=" * 50 + "\n")
        except Exception as e:
            print(f"Cannot write to error log file: {str(e)}")

    print(
        f"{video_type.capitalize()} feature extraction completed. Success: {successful_count}/{len(video_paths)}, Failed: {skipped_count}")
    return features_dict



def main():
    try:

        hate_video_paths, nonhate_video_paths = collect_video_paths(VIDEO_ROOT)

        if not hate_video_paths and not nonhate_video_paths:
            print("No video files found to process")
            return


        print("Starting hate video feature extraction...")
        hate_features = {}
        if hate_video_paths:
            hate_features = extract_features(hate_video_paths, ERROR_LOG_FILE, "hate")
        else:
            print("No hate videos found")

        print("Starting non-hate video feature extraction...")
        nonhate_features = {}
        if nonhate_video_paths:
            nonhate_features = extract_features(nonhate_video_paths, ERROR_LOG_FILE, "non_hate")
        else:
            print("No non-hate videos found")

        if hate_features:
            print(f"Saving {len(hate_features)} hate features to {OUTPUT_FEATURES_HATE}")
            with open(OUTPUT_FEATURES_HATE, "wb") as f:
                pickle.dump(hate_features, f)
            print(f"Successfully saved hate features")
        else:
            print("No hate features to save")

        if nonhate_features:
            print(f"Saving {len(nonhate_features)} non-hate features to {OUTPUT_FEATURES_NONHATE}")
            with open(OUTPUT_FEATURES_NONHATE, "wb") as f:
                pickle.dump(nonhate_features, f)
            print(f"Successfully saved non-hate features")
        else:
            print("No non-hate features to save")


        try:
            if hate_features:
                with open(OUTPUT_FEATURES_HATE, "rb") as f:
                    loaded_hate = pickle.load(f)
                    print(f"Verification: Loaded {len(loaded_hate)} hate features from saved file")

                    print("Hate video key:")
                    for i, key in enumerate(list(loaded_hate.keys())[:3]):
                        print(f"  {key}")

            if nonhate_features:
                with open(OUTPUT_FEATURES_NONHATE, "rb") as f:
                    loaded_nonhate = pickle.load(f)
                    print(f"Verification: Loaded {len(loaded_nonhate)} non-hate features from saved file")

                    print("Non-hate video key:")
                    for i, key in enumerate(list(loaded_nonhate.keys())[:3]):
                        print(f"  {key}")
        except Exception as e:
            print(f"Failed to verify saved features: {str(e)}")

        print("Feature extraction completed successfully!")

    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Fatal error in main program: {str(e)}")
        raise


if __name__ == "__main__":
    main()