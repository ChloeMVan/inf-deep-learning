#!/bin/bash
# =============================================================================
# FaceForensics++ Dataset Downloader + Frame Extractor
# Downloads: original, Deepfakes, Face2Face, FaceSwap, NeuralTextures, DFD real, DFD fake
# Then extracts all videos into frames
# Usage:
#   Full download:    bash download_ff_data.sh
#   Sample (1 video): bash download_ff_data.sh --sample
# =============================================================================

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/FaceForensics/download.py"
EXTRACT_SCRIPT="$SCRIPT_DIR/FaceForensics/dataset/extract_compressed_videos.py"
OUTPUT_BASE=~/Documents/INF-Deep_Learning/FF_data
COMPRESSION="c40"
SERVER="EU2"
NUM_VIDEOS=1  # used only in sample mode

# --- Parse flags ---
SAMPLE=false
for arg in "$@"; do
    case $arg in
        --sample)
            SAMPLE=true
            NUM_VIDEOS=1
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash download_ff_data.sh [--sample]"
            exit 1
            ;;
    esac
done

# --- Print settings ---
echo "============================================="
echo "  FaceForensics++ Downloader + Extractor"
echo "============================================="
echo "  Mode:        $([ "$SAMPLE" = true ] && echo 'SAMPLE (1 video each)' || echo 'FULL (all videos)')"
echo "  Compression: $COMPRESSION"
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

    if [ "$SAMPLE" = true ]; then
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

echo "============================================="
echo "  All done!"
echo "  Data saved to: $OUTPUT_BASE"
echo "============================================="