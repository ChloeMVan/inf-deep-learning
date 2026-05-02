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
#   Skip frames:      bash download_ff_data.sh --frame_skip 20  (keep frame 0, 20, 40, ...)
#   One category:     bash download_ff_data.sh --category DFD_real
#                     Valid names: real, fake, Face2Face, FaceSwap, NeuralTextures, DFD_real, DFD_fake
# =============================================================================

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="/content/download.py"
EXTRACT_SCRIPT="/content/extract_compressed_videos.py"
OUTPUT_BASE=/content/FF_data
COMPRESSION="c40"
SERVER="EU2"
NUM_VIDEOS=""   # empty = download all
FRAME_SKIP=""   # empty = keep all frames
CATEGORY=""     # empty = all categories
BATCH_SIZE=10   # extract this many videos before skipping frames

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
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash download_ff_data.sh [--sample] [--num_videos N] [--hq] [--compression c0|c23|c40] [--frame_skip N] [--category NAME]"
            echo "Valid category names: real, fake, Face2Face, FaceSwap, NeuralTextures, DFD_real, DFD_fake"
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
if [ -n "$FRAME_SKIP" ]; then
    echo "  Frame skip:  keep every $FRAME_SKIP frames — applied after every $BATCH_SIZE videos"
else
    echo "  Frame skip:  all frames kept"
fi
echo "  Category:    $([ -n "$CATEGORY" ] && echo "$CATEGORY only" || echo 'all')"
echo "  Server:      $SERVER"
echo "  Output:      $OUTPUT_BASE"
echo "============================================="
echo ""

# --- Check scripts exist ---
if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "ERROR: download.py not found at $DOWNLOAD_SCRIPT"
    exit 1
fi

if [ ! -f "$EXTRACT_SCRIPT" ]; then
    echo "ERROR: extract_compressed_videos.py not found at $EXTRACT_SCRIPT"
    exit 1
fi

# --- Category filter ---
should_run() { [ -z "$CATEGORY" ] || [ "$CATEGORY" = "$1" ]; }

# --- Download function ---
run_download() {
    local label=$1
    local dataset=$2
    local output=$3

    echo "----------------------------------------------"
    echo "Downloading: $label"
    echo "  Dataset: $dataset  |  Output: $output"
    echo "----------------------------------------------"

    if [ -n "$NUM_VIDEOS" ]; then
        echo "" | python3 "$DOWNLOAD_SCRIPT" "$output" \
            -d "$dataset" -c "$COMPRESSION" --server "$SERVER" --num_videos "$NUM_VIDEOS"
    else
        echo "" | python3 "$DOWNLOAD_SCRIPT" "$output" \
            -d "$dataset" -c "$COMPRESSION" --server "$SERVER"
    fi

    if [ $? -eq 0 ]; then echo "✓ Downloaded: $label"
    else               echo "✗ Download failed: $label"; fi
    echo ""
}

# --- Batched extract + immediate frame skip ---
#
# The extract script sees only BATCH_SIZE videos at a time via a temp dir of
# symlinks — the original video files are never moved. After each batch is
# extracted the frames are immediately skipped/deleted before the next batch
# starts, keeping peak PNG storage to ~BATCH_SIZE videos at a time.
#
# The videos directory is located dynamically under data_path (FF++ uses
# different sub-paths per category, e.g. original_sequences/actors/c40/videos).
# The images output dir is derived by replacing "videos" with "images" at the
# same level, matching what the extract script produces.
run_batched_extract_and_skip() {
    local label=$1
    local dataset=$2
    local data_path=$3

    echo "----------------------------------------------"
    echo "Extracting: $label"
    echo "  Path: $data_path  |  Batch size: $BATCH_SIZE"
    echo "----------------------------------------------"

    # Locate the videos directory — FF++ layout varies per category
    local src_video_dir
    src_video_dir=$(find "$data_path" -type d -name "videos" 2>/dev/null | head -1)

    if [ -z "$src_video_dir" ]; then
        echo "  No 'videos' directory found under $data_path"
        echo "✗ Failed: $label"
        echo ""
        return 1
    fi

    echo "  Videos dir: $src_video_dir"

    # Images are written alongside videos/ at the same level
    local dst_images_dir="${src_video_dir%/videos}/images"

    # Relative sub-path from data_path to the videos dir parent
    # e.g. data_path=.../DFD_real  src_video_dir=.../DFD_real/original_sequences/actors/c40/videos
    # → rel_parent = original_sequences/actors/c40
    local rel_parent
    rel_parent=$(dirname "${src_video_dir#"$data_path"/}")

    # Collect all video files for this category
    local all_videos=()
    while IFS= read -r f; do all_videos+=("$f"); done \
        < <(find "$src_video_dir" -maxdepth 1 -type f 2>/dev/null | sort)

    if [ ${#all_videos[@]} -eq 0 ]; then
        echo "  No video files found in $src_video_dir"
        echo "✗ Failed: $label"
        echo ""
        return 1
    fi

    echo "  Found ${#all_videos[@]} videos"

    # Temp dir mirrors the exact sub-path structure so the extract script
    # finds videos and writes images in the expected locations.
    local tmp_root
    tmp_root=$(mktemp -d)
    mkdir -p "$tmp_root/$rel_parent/videos"
    mkdir -p "$dst_images_dir"

    local i=0 total=${#all_videos[@]}
    local total_kept=0 total_deleted=0

    while (( i < total )); do
        local batch_end=$(( i + BATCH_SIZE ))
        (( batch_end > total )) && batch_end=$total

        echo ""
        echo "  --- Batch $((i / BATCH_SIZE + 1)): videos $((i+1))–$batch_end of $total ---"

        # Symlink this batch's video files into the temp dir
        local batch_ids=()
        for (( j = i; j < batch_end; j++ )); do
            local f="${all_videos[$j]}"
            local base; base=$(basename "$f")
            ln -sf "$f" "$tmp_root/$rel_parent/videos/$base"
            batch_ids+=("${base%%.*}")   # strip extension → video id
        done

        # Extract — script only sees BATCH_SIZE symlinked videos
        python3 "$EXTRACT_SCRIPT" \
            --data_path "$tmp_root" \
            --dataset   "$dataset" \
            --compression "$COMPRESSION"

        # Move extracted frame dirs to real destination; skip frames immediately
        for vid_id in "${batch_ids[@]}"; do
            local src="$tmp_root/$rel_parent/images/$vid_id"
            [ -d "$src" ] || continue

            mv "$src" "$dst_images_dir/"

            if [ -n "$FRAME_SKIP" ]; then
                local vid_frames="$dst_images_dir/$vid_id"
                local idx=0 kept=0 deleted=0
                while IFS= read -r frame; do
                    if (( idx % FRAME_SKIP == 0 )); then
                        (( kept++ ))
                    else
                        rm "$frame"
                        (( deleted++ ))
                    fi
                    (( idx++ ))
                done < <(find "$vid_frames" -maxdepth 1 -name "*.png" | sort)

                echo "    $vid_id: kept $kept / deleted $deleted frames"
                (( total_kept    += kept    ))
                (( total_deleted += deleted ))
            fi
        done

        # Remove this batch's symlinks before the next iteration
        for (( j = i; j < batch_end; j++ )); do
            rm -f "$tmp_root/$rel_parent/videos/$(basename "${all_videos[$j]}")"
        done

        (( i += BATCH_SIZE ))
    done

    rm -rf "$tmp_root"

    echo ""
    if [ -n "$FRAME_SKIP" ]; then
        echo "  Total frames kept: $total_kept  |  deleted: $total_deleted"
    fi
    echo "✓ Done: $label"
    echo ""
}

# --- Process one category: download then batch-extract+skip ---
process_category() {
    local label=$1
    local dataset=$2
    local output=$3

    run_download              "$label" "$dataset" "$output"
    run_batched_extract_and_skip "$label" "$dataset" "$output"
}

# =============================================================================
# Main
# =============================================================================
echo "============================================="
echo "  Processing categories..."
echo "============================================="
echo ""

should_run "real"           && process_category "Original (YouTube real videos)"   "original"                   "$OUTPUT_BASE/real"
should_run "Face2Face"      && process_category "Face2Face (manipulated)"          "Face2Face"                  "$OUTPUT_BASE/Face2Face"
should_run "FaceSwap"       && process_category "FaceSwap (manipulated)"           "FaceSwap"                   "$OUTPUT_BASE/FaceSwap"
should_run "NeuralTextures" && process_category "NeuralTextures (manipulated)"     "NeuralTextures"             "$OUTPUT_BASE/NeuralTextures"
should_run "fake"           && process_category "Deepfakes (manipulated)"          "Deepfakes"                  "$OUTPUT_BASE/fake"
should_run "DFD_real"       && process_category "DFD Real (Google actor videos)"   "DeepFakeDetection_original" "$OUTPUT_BASE/DFD_real"
should_run "DFD_fake"       && process_category "DFD Fake (Google deepfakes)"      "DeepFakeDetection"          "$OUTPUT_BASE/DFD_fake"

echo "============================================="
echo "  All done!"
echo "  Data saved to: $OUTPUT_BASE"
echo "============================================="
