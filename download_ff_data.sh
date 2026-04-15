#!/bin/bash
# =============================================================================
# FaceForensics++ Dataset Downloader + Frame Extractor
# Downloads: original, Deepfakes, Face2Face, FaceSwap, NeuralTextures, DFD real, DFD fake
# Then extracts all videos into frames
# Usage:
#   Full download:    bash download_ff_data.sh
#   Sample (1 video): bash download_ff_data.sh --sample
#   N videos:         bash download_ff_data.sh --num_videos 5
#   High quality:     bash download_ff_data.sh --hq
#   Custom compress:  bash download_ff_data.sh --compression c0  (c0=raw, c23=hq, c40=lq)
#   Skip frames:      bash download_ff_data.sh --frame_skip 50  (keep frame 0, 50, 100, ...)
# =============================================================================

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/FaceForensics/download.py"
EXTRACT_SCRIPT="$SCRIPT_DIR/FaceForensics/dataset/extract_compressed_videos.py"
OUTPUT_BASE=~/Documents/INF-Deep_Learning/FF_data
COMPRESSION="c40"
SERVER="EU2"
NUM_VIDEOS=""   # empty = download all
FRAME_SKIP=""   # empty = keep all frames

# --- Parse flags ---
SAMPLE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample)
            SAMPLE=true
            NUM_VIDEOS=1
            shift
            ;;
        --num_videos)
            NUM_VIDEOS="$2"
            shift 2
            ;;
        --hq)
            COMPRESSION="c23"
            shift
            ;;
        --compression)
            COMPRESSION="$2"
            shift 2
            ;;
        --frame_skip)
            FRAME_SKIP="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash download_ff_data.sh [--sample] [--num_videos N] [--hq] [--compression c0|c23|c40] [--frame_skip N]"
            exit 1
            ;;
    esac
done

# --- Print settings ---
echo "============================================="
echo "  FaceForensics++ Downloader + Extractor"
echo "============================================="
if [ "$SAMPLE" = true ]; then
    echo "  Mode:        SAMPLE (1 video each)"
elif [ -n "$NUM_VIDEOS" ]; then
    echo "  Mode:        PARTIAL ($NUM_VIDEOS videos each)"
else
    echo "  Mode:        FULL (all videos)"
fi
echo "  Compression: $COMPRESSION  (c0=raw, c23=high quality, c40=low quality)"
echo "  Frame skip:  $([ -n "$FRAME_SKIP" ] && echo "keep every $FRAME_SKIP frames (0, $FRAME_SKIP, $((FRAME_SKIP*2)), ...)" || echo 'all frames')"
echo "  Server:      $SERVER"
echo "  Output:      $OUTPUT_BASE"
echo "============================================="
echo ""

# --- Check scripts exist ---
if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "ERROR: download.py not found at $SCRIPT_DIR"
    echo "Make sure download.py is in the same folder as this script."
    exit 1
fi

if [ ! -f "$EXTRACT_SCRIPT" ]; then
    echo "ERROR: extract_compressed_videos.py not found at $SCRIPT_DIR/dataset/"
    echo "Make sure extract_compressed_videos.py is in the dataset/ subfolder."
    exit 1
fi

# --- Download function ---
run_download() {
    local label=$1
    local dataset=$2
    local output=$3

    echo "----------------------------------------------"
    echo "Downloading: $label"
    echo "  Dataset: $dataset"
    echo "  Output:  $output"
    echo "----------------------------------------------"

    if [ -n "$NUM_VIDEOS" ]; then
        echo "" | python3 "$DOWNLOAD_SCRIPT" "$output" \
            -d "$dataset" \
            -c "$COMPRESSION" \
            --server "$SERVER" \
            --num_videos "$NUM_VIDEOS"
    else
        echo "" | python3 "$DOWNLOAD_SCRIPT" "$output" \
            -d "$dataset" \
            -c "$COMPRESSION" \
            --server "$SERVER"
    fi

    if [ $? -eq 0 ]; then
        echo "✓ Done: $label"
    else
        echo "✗ Failed: $label"
    fi
    echo ""
}

# --- Frame skip function ---
# Walks every per-video subfolder under data_path/**/<compression>/images/
# and deletes all PNGs whose 0-based sorted index is not a multiple of FRAME_SKIP.
run_skip_frames() {
    local label=$1
    local data_path=$2

    echo "----------------------------------------------"
    echo "Skipping frames: $label  (keep every $FRAME_SKIP)"
    echo "  Path: $data_path"
    echo "----------------------------------------------"

    local kept=0
    local deleted=0

    # Frames live at: data_path/**/<compression>/images/<video_id>/<frame>.png
    while IFS= read -r video_dir; do
        local idx=0
        while IFS= read -r frame; do
            if (( idx % FRAME_SKIP == 0 )); then
                (( kept++ ))
            else
                rm "$frame"
                (( deleted++ ))
            fi
            (( idx++ ))
        done < <(find "$video_dir" -maxdepth 1 -name "*.png" | sort)
    done < <(find "$data_path" -type d -name "images" | xargs -I{} find {} -mindepth 1 -maxdepth 1 -type d 2>/dev/null)

    echo "  Kept: $kept  |  Deleted: $deleted"
    echo "✓ Done: $label"
    echo ""
}

# --- Extract function ---
run_extract() {
    local label=$1
    local dataset=$2
    local data_path=$3

    echo "----------------------------------------------"
    echo "Extracting frames: $label"
    echo "  Dataset: $dataset"
    echo "  Path:    $data_path"
    echo "----------------------------------------------"

    python3 "$EXTRACT_SCRIPT" \
        --data_path "$data_path" \
        --dataset "$dataset" \
        --compression "$COMPRESSION"

    if [ $? -eq 0 ]; then
        echo "✓ Done: $label"
    else
        echo "✗ Failed: $label"
    fi
    echo ""
}

# =============================================================================
# STEP 1: Downloads
# =============================================================================
echo "============================================="
echo "  STEP 1: Downloading videos..."
echo "============================================="
echo ""

run_download "Original (YouTube real videos)"   "original"                   "$OUTPUT_BASE/real"
run_download "Face2Face (manipulated)"          "Face2Face"                  "$OUTPUT_BASE/Face2Face"
run_download "FaceSwap (manipulated)"           "FaceSwap"                   "$OUTPUT_BASE/FaceSwap"
run_download "NeuralTextures (manipulated)"     "NeuralTextures"             "$OUTPUT_BASE/NeuralTextures"
run_download "Deepfakes (manipulated)"          "Deepfakes"                  "$OUTPUT_BASE/fake"
run_download "DFD Real (Google actor videos)"   "DeepFakeDetection_original" "$OUTPUT_BASE/DFD_real"
run_download "DFD Fake (Google deepfakes)"      "DeepFakeDetection"          "$OUTPUT_BASE/DFD_fake"

# =============================================================================
# STEP 2: Frame Extraction
# =============================================================================
echo "============================================="
echo "  STEP 2: Extracting frames..."
echo "============================================="
echo ""

run_extract "Original (YouTube real videos)"   "original"                   "$OUTPUT_BASE/real"
run_extract "Face2Face (manipulated)"          "Face2Face"                  "$OUTPUT_BASE/Face2Face"
run_extract "FaceSwap (manipulated)"           "FaceSwap"                   "$OUTPUT_BASE/FaceSwap"
run_extract "NeuralTextures (manipulated)"     "NeuralTextures"             "$OUTPUT_BASE/NeuralTextures"
run_extract "Deepfakes (manipulated)"          "Deepfakes"                  "$OUTPUT_BASE/fake"
run_extract "DFD Real (Google actor videos)"   "DeepFakeDetection_original" "$OUTPUT_BASE/DFD_real"
run_extract "DFD Fake (Google deepfakes)"      "DeepFakeDetection"          "$OUTPUT_BASE/DFD_fake"

# =============================================================================
# STEP 3: Frame Skipping (only if --frame_skip was given)
# =============================================================================
if [ -n "$FRAME_SKIP" ]; then
    echo "============================================="
    echo "  STEP 3: Skipping frames (keep every $FRAME_SKIP)..."
    echo "============================================="
    echo ""

    run_skip_frames "Original (YouTube real videos)"   "$OUTPUT_BASE/real"
    run_skip_frames "Face2Face (manipulated)"          "$OUTPUT_BASE/Face2Face"
    run_skip_frames "FaceSwap (manipulated)"           "$OUTPUT_BASE/FaceSwap"
    run_skip_frames "NeuralTextures (manipulated)"     "$OUTPUT_BASE/NeuralTextures"
    run_skip_frames "Deepfakes (manipulated)"          "$OUTPUT_BASE/fake"
    run_skip_frames "DFD Real (Google actor videos)"   "$OUTPUT_BASE/DFD_real"
    run_skip_frames "DFD Fake (Google deepfakes)"      "$OUTPUT_BASE/DFD_fake"
fi

echo "============================================="
echo "  All done!"
echo "  Data saved to: $OUTPUT_BASE"
echo "============================================="