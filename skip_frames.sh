#!/bin/bash
# =============================================================================
# FaceForensics++ Frame Skipper
# Walks already-extracted frames and deletes those not on the keep interval.
# Usage:
#   All categories:   bash skip_frames.sh --frame_skip 20
#   One category:     bash skip_frames.sh --frame_skip 20 --category DFD_real
#   Valid names:      real, fake, Face2Face, FaceSwap, NeuralTextures, DFD_real, DFD_fake
# =============================================================================

OUTPUT_BASE=~/Documents/INF-Deep_Learning/FF_data
FRAME_SKIP=""
CATEGORY=""

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --frame_skip)
            FRAME_SKIP="$2"
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash skip_frames.sh --frame_skip N [--category NAME]"
            echo "Valid category names: real, fake, Face2Face, FaceSwap, NeuralTextures, DFD_real, DFD_fake"
            exit 1
            ;;
    esac
done

if [ -z "$FRAME_SKIP" ]; then
    echo "ERROR: --frame_skip N is required."
    echo "Usage: bash skip_frames.sh --frame_skip N [--category NAME]"
    exit 1
fi

# --- Print settings ---
echo "============================================="
echo "  FaceForensics++ Frame Skipper"
echo "============================================="
echo "  Keep every: $FRAME_SKIP frames (0, $FRAME_SKIP, $((FRAME_SKIP*2)), ...)"
echo "  Category:   $([ -n "$CATEGORY" ] && echo "$CATEGORY only" || echo 'all')"
echo "  Data root:  $OUTPUT_BASE"
echo "============================================="
echo ""

# --- Category filter ---
should_run() { [ -z "$CATEGORY" ] || [ "$CATEGORY" = "$1" ]; }

# --- Frame skip function ---
# Walks every per-video subfolder under data_path and deletes PNGs
# whose 0-based sorted index is not a multiple of FRAME_SKIP.
run_skip_frames() {
    local label=$1
    local data_path=$2

    if [ ! -d "$data_path" ]; then
        echo "⚠ Skipping (directory not found): $data_path"
        echo ""
        return
    fi

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

# --- Run ---
should_run "real"           && run_skip_frames "Original (YouTube real videos)"   "$OUTPUT_BASE/real"
should_run "Face2Face"      && run_skip_frames "Face2Face (manipulated)"          "$OUTPUT_BASE/Face2Face"
should_run "FaceSwap"       && run_skip_frames "FaceSwap (manipulated)"           "$OUTPUT_BASE/FaceSwap"
should_run "NeuralTextures" && run_skip_frames "NeuralTextures (manipulated)"     "$OUTPUT_BASE/NeuralTextures"
should_run "fake"           && run_skip_frames "Deepfakes (manipulated)"          "$OUTPUT_BASE/fake"
should_run "DFD_real"       && run_skip_frames "DFD Real (Google actor videos)"   "$OUTPUT_BASE/DFD_real"
should_run "DFD_fake"       && run_skip_frames "DFD Fake (Google deepfakes)"      "$OUTPUT_BASE/DFD_fake"

echo "============================================="
echo "  All done!"
echo "  Data root: $OUTPUT_BASE"
echo "============================================="
