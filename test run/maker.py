from ultralytics import YOLO
import numpy as np
import cv2
from scipy.signal import savgol_filter
import os
import matplotlib.pyplot as plt
import sys

# Check for DTW library
try:
    from dtw import dtw
    HAS_DTW = True
    print("✅ DTW library imported successfully")
except ImportError:
    HAS_DTW = False
    print("❌ DTW library not found - install with 'pip install dtw-python'")

# Print versions for debugging
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

# Load the model with explicit verification
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Function to normalize keypoints sequence
def normalize_keypoints_sequence(keypoints_seq):
    """
    Normalize a sequence of keypoints for comparison.
    Each keypoint set is centered and scaled to a standard size.
    """
    if not keypoints_seq or len(keypoints_seq) == 0:
        print("Warning: Empty keypoints sequence passed to normalization function")
        return []

    normalized_seq = []

    for keypoints in keypoints_seq:
        # Skip if keypoints is empty
        if keypoints.size == 0:
            continue

        try:
            # Center the keypoints
            center = np.mean(keypoints, axis=0)
            centered = keypoints - center

            # Scale to standard size
            scale = np.max(np.ptp(centered, axis=0))
            if scale > 0:
                normalized = centered / scale
            else:
                normalized = centered

            normalized_seq.append(normalized)
        except Exception as e:
            print(f"Error normalizing keypoints: {e}")
            print(f"Keypoints shape: {keypoints.shape}")
            # Add the original keypoints as fallback
            normalized_seq.append(keypoints)

    return normalized_seq

# Compare two sitting sequences using DTW
def compare_sitting_sequences(seq1, seq2):
    """
    Compare two sequences of keypoints using Dynamic Time Warping.
    Returns a distance score (lower is more similar).
    """
    if not seq1 or not seq2 or len(seq1) == 0 or len(seq2) == 0:
        print("Warning: Empty sequence passed to comparison function")
        return float('inf')  # Return infinity as the distance for empty sequences

    if not isinstance(seq1, list) or not isinstance(seq2, list):
        print(f"Error: Expected lists, got {type(seq1)} and {type(seq2)}")
        return float('inf')

    try:
        # Convert sequences to feature vectors
        features1 = []
        features2 = []

        # For each frame, flatten the keypoints
        for kpts in seq1:
            features1.append(kpts.flatten())

        for kpts in seq2:
            features2.append(kpts.flatten())

        # Convert to numpy arrays
        features1 = np.array(features1)
        features2 = np.array(features2)

        # Use DTW to find optimal alignment and distance
        alignment = dtw(features1, features2, dist=lambda x, y: np.linalg.norm(x - y))

        return alignment.distance
    except Exception as e:
        print(f"Error in DTW comparison: {e}")

        # Fallback to simple comparison if DTW fails
        min_len = min(len(seq1), len(seq2))
        distances = []
        for i in range(min_len):
            try:
                dist = np.mean(np.linalg.norm(seq1[i] - seq2[i], axis=1))
                distances.append(dist)
            except Exception as e2:
                print(f"Error in fallback comparison at frame {i}: {e2}")
                continue

        if distances:
            return np.mean(distances)
        else:
            return float('inf')

# Improved function to extract sitting motion sequence from video
def extract_sitting_sequence(video_path, model):
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    frames = []
    keypoints_sequence = []
    heights = []
    processed_frames = 0
    keypoints_found = 0

    frame_idx = 0

    # Process every frame instead of every 3rd
    sample_rate = 1  # Changed from 3 to 1

    # Try different confidence thresholds
    confidence_threshold = 0.25  # Lowered from default to detect more keypoints

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            processed_frames += 1
            frames.append(frame.copy())

            # Add debug info to frame
            debug_frame = frame.copy()
            cv2.putText(debug_frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Run inference with explicit try/except
            try:
                # Set confidence threshold explicitly
                results = model.predict(source=frame, conf=confidence_threshold, verbose=False)[0]

                if len(results.keypoints) > 0 and results.keypoints.xy[0].shape[0] > 0:
                    keypoints_found += 1
                    kpts = results.keypoints.xy[0].cpu().numpy()
                    keypoints_sequence.append(kpts)

                    # Draw keypoints on debug frame
                    for point in kpts:
                        cv2.circle(debug_frame, (int(point[0]), int(point[1])),
                                   5, (0, 0, 255), -1)

                    y_values = kpts[:, 1]
                    height = np.max(y_values) - np.min(y_values)
                    heights.append(height)

                    cv2.putText(debug_frame, f"Height: {height:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(debug_frame, "No keypoints", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Interpolate missing keypoints if we have previous data
                    if keypoints_sequence and len(keypoints_sequence) > 0:
                        print(f"Frame {frame_idx}: No keypoints detected, using previous frame")
                        keypoints_sequence.append(keypoints_sequence[-1])  # Use last frame's keypoints
                        if heights:
                            heights.append(heights[-1])  # Use last height
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                cv2.putText(debug_frame, f"Error: {str(e)[:20]}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show debug frame
            cv2.imshow("Processing", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed {processed_frames} frames, found keypoints in {keypoints_found} frames")

    if not keypoints_sequence:
        print("❌ No keypoints detected in video")
        return None, None, None

    # Modified: Apply smoothing if we have enough data points, otherwise skip
    if len(heights) > 0:
        if len(heights) > 5:
            # Use a smaller window for Savitzky-Golay filter
            window_size = min(len(heights), 5)
            # Ensure window_size is odd
            window_size = window_size if window_size % 2 == 1 else window_size + 1
            heights = savgol_filter(heights, window_size, 2)
        else:
            print("Not enough height data points for smoothing, using raw data")
    else:
        print("❌ No height data available")
        return frames, keypoints_sequence, []

    # Plot height variation
    plt.figure(figsize=(10, 6))
    plt.plot(heights)
    plt.title("Dog Height Variation")
    plt.xlabel("Frame")
    plt.ylabel("Height")
    plt.savefig(f"{os.path.basename(video_path)}_heights.png")
    plt.close()

    return frames, keypoints_sequence, heights

# Modified detect_sitting_segments with more relaxed requirements
def detect_sitting_segments(heights, video_name):
    # Modified: Lower the minimum required height data points
    if len(heights) < 3:  # Changed from 10 to 3
        print("❌ Not enough height data to detect sitting")
        return None

    heights = np.array(heights)
    heights_norm = (heights - np.min(heights)) / (np.max(heights) - np.min(heights))

    # Calculate height differences
    height_diff = np.diff(heights_norm)

    # Plot for debugging
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(heights_norm)
    plt.title("Normalized Height")
    plt.subplot(212)
    plt.plot(height_diff)
    plt.title("Height Change Rate")

    # More relaxed thresholds for sitting detection
    plt.axhline(y=-0.01, color='r', linestyle='--', label='-0.01 threshold')  # Changed from -0.015
    plt.axhline(y=-0.005, color='g', linestyle='--', label='-0.005 threshold')  # Changed from -0.01
    plt.axhline(y=0.003, color='b', linestyle='--', label='0.003 threshold')  # Changed from 0.005
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{video_name}_height_analysis.png")
    plt.close()

    sit_start_candidates = []
    sit_end_candidates = []

    # Print thresholds for checking
    print("Looking for height decreases below -0.01")  # Changed from -0.015

    # More relaxed window size for detecting sitting
    window_size = 3  # Changed from 5

    # Find sitting start candidates
    for i in range(len(height_diff) - window_size + 1):
        window = height_diff[i:i+window_size]
        if np.mean(window) < -0.01:  # Changed from -0.015
            sit_start_candidates.append(i)

    # Find sitting end candidates (when the dog stabilizes)
    for i in range(len(height_diff) - window_size + 1):
        window = height_diff[i:i+window_size]
        if i > 0 and height_diff[i-1] < -0.005 and np.abs(np.mean(window)) < 0.003:  # Changed thresholds
            sit_end_candidates.append(i)

    print(f"Found {len(sit_start_candidates)} potential sitting starts")
    print(f"Found {len(sit_end_candidates)} potential sitting ends")

    # If we don't find candidates with our approach, use simple min/max height
    if len(sit_start_candidates) == 0 or len(sit_end_candidates) == 0:
        print("Not enough candidates found with threshold approach, trying simple min/max approach")

        # Find point of maximum height (standing)
        max_height_idx = np.argmax(heights_norm)

        # Find point of minimum height (sitting)
        min_height_idx = np.argmin(heights_norm)

        # Make sure max comes before min for a sitting motion
        if max_height_idx < min_height_idx:
            sit_start_candidates = [max(0, max_height_idx - 1)]
            sit_end_candidates = [min(len(heights_norm) - 1, min_height_idx + 1)]
        else:
            # Try to find local maxima and minima
            for i in range(1, len(heights_norm) - 1):
                if heights_norm[i] > heights_norm[i-1] and heights_norm[i] > heights_norm[i+1]:
                    sit_start_candidates.append(i)  # Local maximum
                if heights_norm[i] < heights_norm[i-1] and heights_norm[i] < heights_norm[i+1]:
                    sit_end_candidates.append(i)  # Local minimum

    # Find the best segment
    best_segment = None
    max_drop = 0

    for start in sit_start_candidates:
        for end in sit_end_candidates:
            # More flexible duration constraints
            if end > start + 1 and end < start + min(30, len(heights_norm) - 1):  # Changed from +3 to +1
                height_drop = heights_norm[start] - heights_norm[end]
                if height_drop > max_drop:
                    max_drop = height_drop
                    best_segment = (start, end)

    # If still no segment found, use beginning and end if there's a height decrease
    if best_segment is None and len(heights_norm) >= 2:
        if heights_norm[0] > heights_norm[-1]:
            best_segment = (0, len(heights_norm) - 1)
            max_drop = heights_norm[0] - heights_norm[-1]
            print(f"Using full sequence as sitting segment with height drop {max_drop:.4f}")

    if best_segment:
        start, end = best_segment
        print(f"✅ Found sitting segment from frame {start} to {end} with height drop {max_drop:.4f}")

        # Visualize the detected segment on the height plot
        plt.figure(figsize=(10, 6))
        plt.plot(heights_norm)
        plt.axvspan(start, end, color='green', alpha=0.3)
        plt.title(f"Detected Sitting Segment (frames {start}-{end})")
        plt.savefig(f"{video_name}_sitting_segment.png")
        plt.close()

        return best_segment
    else:
        print("❌ Could not identify a clear sitting segment")
        return None

# Main function with full implementation
def main():
    # Define video paths
    reference_video = "dogSitting.MOV"  # Video with correct sitting
    test_video = "IMG_6258.MOV"      # Video to evaluate

    # Verify files exist
    for video_path in [reference_video, test_video]:
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return

    # Load model with verification
    model = load_model(".idea/yolo11n-pose.pt")
    if model is None:
        return

    print("\n===== Processing Reference Video =====")
    ref_frames, ref_keypoints, ref_heights = extract_sitting_sequence(reference_video, model)

    print("\n===== Processing Test Video =====")
    test_frames, test_keypoints, test_heights = extract_sitting_sequence(test_video, model)

    if ref_keypoints is None or test_keypoints is None:
        print("❌ Failed to detect keypoints in one or both videos")
        return

    print("\n===== Analyzing Reference Video Sitting =====")
    ref_segment = detect_sitting_segments(ref_heights, "reference")

    print("\n===== Analyzing Test Video Sitting =====")
    test_segment = detect_sitting_segments(test_heights, "test")

    if ref_segment is None or test_segment is None:
        print("❌ Failed to detect sitting motion in one or both videos")
        return

    ref_start, ref_end = ref_segment
    test_start, test_end = test_segment

    ref_sitting_seq = ref_keypoints[ref_start:ref_end+1]
    test_sitting_seq = test_keypoints[test_start:test_end+1]

    print(f"\nReference sitting sequence: {len(ref_sitting_seq)} frames")
    print(f"Test sitting sequence: {len(test_sitting_seq)} frames")

    # Compare sitting sequences
    print("\n===== Comparing Sitting Motions =====")

    # Verify sequences are valid
    if not ref_sitting_seq or len(ref_sitting_seq) == 0 or not test_sitting_seq or len(test_sitting_seq) == 0:
        print("❌ Empty sitting sequences detected")
        return

    try:
        print(f"Normalizing reference sequence ({len(ref_sitting_seq)} frames)...")
        norm_ref_seq = normalize_keypoints_sequence(ref_sitting_seq)

        print(f"Normalizing test sequence ({len(test_sitting_seq)} frames)...")
        norm_test_seq = normalize_keypoints_sequence(test_sitting_seq)

        if not norm_ref_seq or not norm_test_seq:
            print("❌ Normalization failed, sequences are empty")
            return

        print(f"Normalized sequences: Ref={len(norm_ref_seq)} frames, Test={len(norm_test_seq)} frames")

        print("Comparing sequences...")
        if HAS_DTW:
            print("Using DTW comparison method")
            distance = compare_sitting_sequences(norm_ref_seq, norm_test_seq)
            print(f"DTW Sitting Motion Similarity Score: {distance:.4f}")
        else:
            print("Using simple comparison method (DTW not available)")
            # Simple distance calculation
            min_len = min(len(norm_ref_seq), len(norm_test_seq))
            distances = []
            for i in range(min_len):
                dist = np.mean(np.linalg.norm(norm_ref_seq[i] - norm_test_seq[i], axis=1))
                distances.append(dist)
            distance = np.mean(distances)
            print(f"Average Sitting Motion Similarity Score: {distance:.4f}")

        # Adjusted threshold for better sensitivity
        threshold = 7.5  # Changed from 5.0 to be more lenient
        print(f"Using similarity threshold: {threshold}")
        if distance < threshold:
            print("✅ Exercise performed correctly! The dog's sitting motion is proper.")
        else:
            print("⚠️ Form needs improvement. The sitting motion differs from the reference.")

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()

    # Visualize key frames from both videos
    if len(ref_frames) > ref_start and len(ref_frames) > ref_end:
        ref_start_frame = ref_frames[ref_start]
        ref_end_frame = ref_frames[ref_end]

        # Draw keypoints on frames
        for point in ref_sitting_seq[0]:
            cv2.circle(ref_start_frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        for point in ref_sitting_seq[-1]:
            cv2.circle(ref_end_frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        cv2.imwrite("ref_start.jpg", ref_start_frame)
        cv2.imwrite("ref_end.jpg", ref_end_frame)
        cv2.imshow("Reference Start", ref_start_frame)
        cv2.imshow("Reference End", ref_end_frame)
    else:
        print("⚠️ Reference frame indices out of range")

    if len(test_frames) > test_start and len(test_frames) > test_end:
        test_start_frame = test_frames[test_start]
        test_end_frame = test_frames[test_end]

        # Draw keypoints on frames
        for point in test_sitting_seq[0]:
            cv2.circle(test_start_frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        for point in test_sitting_seq[-1]:
            cv2.circle(test_end_frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

        cv2.imwrite("test_start.jpg", test_start_frame)
        cv2.imwrite("test_end.jpg", test_end_frame)
        cv2.imshow("Test Start", test_start_frame)
        cv2.imshow("Test End", test_end_frame)
    else:
        print("⚠️ Test frame indices out of range")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()