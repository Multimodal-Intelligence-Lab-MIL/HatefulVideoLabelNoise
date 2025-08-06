import os
import numpy as np
import pickle
from tqdm import tqdm
import librosa
import subprocess
import tempfile
from datetime import datetime


NUM_FRAMES = 60  # For hatemm 100, mhc 60
FEATURE_DIM = 40
SAMPLE_RATE = 16000
VIDEO_ROOT = 'set your path'
OUTPUT_FEATURES_HATE = 'mfcc_pure_hate_seg_features.p'
OUTPUT_FEATURES_NONHATE = 'mfcc_pure_nonhate_in_hate_seg_features.p'
ERROR_LOG_FILE = '/setyourpath/mfcc_feature_extraction_errors.txt'



def extract_audio_with_ffmpeg(video_path, sample_rate=16000):

    try:

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio_path = temp_file.name


        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            '-y',
            '-loglevel', 'error',
            temp_audio_path
        ]


        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise ValueError(f"FFmpeg failed with return code {result.returncode}: {result.stderr}")


        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            raise ValueError("FFmpeg produced empty output file")


        try:
            audio_array, _ = librosa.load(temp_audio_path, sr=sample_rate)
            return audio_array
        except Exception as e:
            raise ValueError(f"Failed to load extracted audio with librosa: {str(e)}")

    except subprocess.TimeoutExpired:
        raise ValueError("FFmpeg operation timed out")
    except FileNotFoundError:
        raise ValueError("FFmpeg not found. Please install FFmpeg and add it to PATH")
    except Exception as e:
        raise ValueError(f"FFmpeg extraction failed: {str(e)}")
    finally:

        try:
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        except:
            pass



def check_ffmpeg():

    try:
        result = subprocess.run(['ffmpeg', '-version'],
                                capture_output=True,
                                text=True,
                                timeout=10)
        if result.returncode == 0:
            print("FFmpeg is available")
            return True
        else:
            print("FFmpeg check failed")
            return False
    except FileNotFoundError:
        print("FFmpeg not found in PATH")
        return False
    except Exception as e:
        print(f"Error checking FFmpeg: {str(e)}")
        return False



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
        new_name = name_without_ext.replace("videoID_", "hate_", 1)
    else:

        new_name = f"hate_video_{name_without_ext}"

    return new_name


def extract_mfcc_features_ffmpeg(video_paths, error_log_path, video_type=""):

    features_dict = {}
    error_paths = []
    successful_count = 0
    skipped_count = 0


    if video_type == "hate":
        try:
            os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
            with open(error_log_path, "w", encoding="utf-8") as ef:
                ef.write(f"MFCC feature extraction with FFmpeg started at: {datetime.now()}\n")
                ef.write("=" * 50 + "\n")
        except Exception as e:
            print(f"Cannot create error log file: {str(e)}")

    for i, video_path in enumerate(tqdm(video_paths, desc=f"Extracting MFCCs from {video_type} videos")):
        try:

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            if os.path.getsize(video_path) == 0:
                raise ValueError("Video file is empty")


            try:
                audio_array = extract_audio_with_ffmpeg(video_path, SAMPLE_RATE)
            except Exception as e:
                raise ValueError(f"Audio extraction failed: {str(e)}")


            if audio_array is None or audio_array.size == 0:
                raise ValueError("Empty audio array")


            if np.all(audio_array == 0):
                print(f"Warning: Audio contains only silence: {os.path.basename(video_path)}")
                mfcc = np.zeros((NUM_FRAMES, FEATURE_DIM))
            else:

                try:
                    mfcc = librosa.feature.mfcc(
                        y=audio_array,
                        sr=SAMPLE_RATE,
                        n_mfcc=FEATURE_DIM,
                        hop_length=SAMPLE_RATE
                    )
                except Exception as e:
                    raise ValueError(f"MFCC extraction failed: {str(e)}")


                mfcc = mfcc.T


                if mfcc.size == 0:
                    raise ValueError("MFCC features are empty")

                if np.any(np.isnan(mfcc)) or np.any(np.isinf(mfcc)):
                    raise ValueError("MFCC features contain NaN or Inf values")


                num_available = mfcc.shape[0]
                if num_available < NUM_FRAMES:

                    padded = np.zeros((NUM_FRAMES, FEATURE_DIM))
                    padded[:num_available] = mfcc
                    mfcc = padded
                elif num_available > NUM_FRAMES:

                    indices = np.linspace(0, num_available - 1, NUM_FRAMES, dtype=int)
                    mfcc = mfcc[indices]


            if mfcc.shape != (NUM_FRAMES, FEATURE_DIM):
                raise ValueError(f"Unexpected MFCC shape: {mfcc.shape}, expected: ({NUM_FRAMES}, {FEATURE_DIM})")


            key_name = generate_key_name(video_path, video_type)
            features_dict[key_name] = mfcc
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

        if not check_ffmpeg():
            print("ERROR: FFmpeg is not available. Please install FFmpeg first.")
            print("Installation instructions:")
            print("- Windows: Download from https://ffmpeg.org/download.html or use 'winget install ffmpeg'")
            print("- Linux: sudo apt-get install ffmpeg (Ubuntu/Debian) or sudo yum install ffmpeg (CentOS/RHEL)")
            print("- macOS: brew install ffmpeg")
            return


        hate_video_paths, nonhate_video_paths = collect_video_paths(VIDEO_ROOT)

        if not hate_video_paths and not nonhate_video_paths:
            print("No video files found to process")
            return


        print("Starting hate video feature extraction with FFmpeg...")
        hate_features = {}
        if hate_video_paths:
            hate_features = extract_mfcc_features_ffmpeg(hate_video_paths, ERROR_LOG_FILE, "hate")
        else:
            print("No hate videos found")


        print("Starting non-hate video feature extraction with FFmpeg...")
        nonhate_features = {}
        if nonhate_video_paths:
            nonhate_features = extract_mfcc_features_ffmpeg(nonhate_video_paths, ERROR_LOG_FILE, "non_hate")
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