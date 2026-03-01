#!/bin/bash
#
# IC Pipeline - Full automation script
# Runs: append_ytd_prices -> momentum scoring -> presentation generation
#
# Usage:
#   ./run_ic_pipeline.sh              # Use latest date from price data
#   ./run_ic_pipeline.sh 2026-03-04   # Use specific date
#
# Alias (add to ~/.zshrc or ~/.bashrc):
#   alias ic="cd /Users/larazanella/Desktop/GitHub/Projects/ic_technical && ./run_ic_pipeline.sh"
#

set -e  # Exit on any error

# === CONFIGURATION ===
IC_REPO="/Users/larazanella/Desktop/GitHub/Projects/ic_technical"
IC_DROPBOX="/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/ic"
MOMENTUM_REPO="/Users/larazanella/Desktop/GitHub/Projects/momentum"
CONDA_ENV="ptf_opt"

EXCEL="$IC_DROPBOX/ic_file.xlsx"
MASTER="$IC_DROPBOX/master_prices.csv"
TEMPLATE="$IC_DROPBOX/template.pptx"
HISTORY="$IC_DROPBOX/score_history/history.json"
OUTPUT="$IC_DROPBOX"

# === PARSE ARGUMENTS ===
# Optional date argument (YYYY-MM-DD), defaults to "latest"
DATE_ARG="${1:-latest}"

# === SETUP ===
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATE_DISPLAY=$(date +"%Y-%m-%d %H:%M:%S")
LOG_DIR="$IC_DROPBOX/logs"
LOG_FILE="$LOG_DIR/ic_pipeline_${TIMESTAMP}.log"

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Start logging (tee to both terminal and file)
exec > >(tee -a "$LOG_FILE") 2>&1

# === HELPER FUNCTIONS ===
format_duration() {
    local seconds=$1
    if [ $seconds -lt 60 ]; then
        echo "${seconds}s"
    else
        local minutes=$((seconds / 60))
        local secs=$((seconds % 60))
        if [ $minutes -lt 60 ]; then
            echo "${minutes}m ${secs}s"
        else
            local hours=$((minutes / 60))
            local mins=$((minutes % 60))
            echo "${hours}h ${mins}m ${secs}s"
        fi
    fi
}

notify() {
    local title="$1"
    local message="$2"
    osascript -e "display notification \"$message\" with title \"$title\"" 2>/dev/null || true
}

print_header() {
    echo "============================================================"
    echo "IC PIPELINE - $DATE_DISPLAY"
    echo "Data date: $DATE_ARG"
    echo "============================================================"
}

print_footer() {
    local status="$1"
    local total_duration="$2"
    local output_file="$3"

    echo "============================================================"
    if [ "$status" = "success" ]; then
        echo "✅ COMPLETE in $(format_duration $total_duration)"
        if [ -n "$output_file" ]; then
            echo "Output: $output_file"
        fi
    else
        echo "❌ FAILED after $(format_duration $total_duration)"
    fi
    echo "Log: $LOG_FILE"
    echo "============================================================"
}

# === PRE-FLIGHT CHECKS ===
preflight_check() {
    local missing=0

    echo ""
    echo "Pre-flight checks..."

    # Check input files
    for file in "$EXCEL" "$MASTER" "$TEMPLATE"; do
        if [ ! -f "$file" ]; then
            echo "  ❌ Missing: $file"
            missing=1
        else
            echo "  ✓ Found: $(basename "$file")"
        fi
    done

    # Check directories
    for dir in "$IC_REPO" "$MOMENTUM_REPO"; do
        if [ ! -d "$dir" ]; then
            echo "  ❌ Missing directory: $dir"
            missing=1
        else
            echo "  ✓ Found: $dir"
        fi
    done

    # Check history directory exists (create if needed)
    local history_dir=$(dirname "$HISTORY")
    if [ ! -d "$history_dir" ]; then
        echo "  ⚠ Creating history directory: $history_dir"
        mkdir -p "$history_dir"
    fi

    if [ $missing -eq 1 ]; then
        echo ""
        echo "❌ Pre-flight check failed. Aborting."
        return 1
    fi

    echo "  ✓ All checks passed"
    echo ""
    return 0
}

# === CONDA ACTIVATION ===
activate_conda() {
    # Try common conda locations
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    elif [ -f "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        source "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    else
        echo "❌ Could not find conda installation"
        return 1
    fi

    conda activate "$CONDA_ENV"
    echo "Activated conda environment: $CONDA_ENV"
}

# === MAIN PIPELINE ===
main() {
    local pipeline_start=$(date +%s)
    local step_start
    local step_end
    local step_duration
    local output_pptx=""

    print_header

    # Activate conda
    activate_conda || {
        notify "IC Pipeline" "❌ Failed to activate conda environment"
        exit 1
    }

    # Pre-flight checks
    preflight_check || {
        notify "IC Pipeline" "❌ Pre-flight check failed"
        exit 1
    }

    # Step 1: Append YTD prices
    echo -n "[1/3] Appending YTD prices..."
    step_start=$(date +%s)

    cd "$IC_REPO"
    python append_ytd_prices.py --excel "$EXCEL" --master "$MASTER" > /tmp/ic_step1.log 2>&1 || {
        echo "       ❌ Failed"
        cat /tmp/ic_step1.log
        notify "IC Pipeline" "❌ Failed at step 1: append_ytd_prices"
        print_footer "failed" $(($(date +%s) - pipeline_start)) ""
        exit 1
    }

    step_end=$(date +%s)
    step_duration=$((step_end - step_start))

    # Extract summary from output
    local new_dates=$(grep -o "Added [0-9]* new date" /tmp/ic_step1.log 2>/dev/null | head -1 || echo "")
    if [ -n "$new_dates" ]; then
        echo "       ✅ ${new_dates}s ($(format_duration $step_duration))"
    elif grep -q "No new dates" /tmp/ic_step1.log 2>/dev/null; then
        echo "       ✅ No new dates ($(format_duration $step_duration))"
    else
        echo "       ✅ Done ($(format_duration $step_duration))"
    fi

    # Step 2: Run momentum scoring
    echo -n "[2/3] Running momentum scoring..."
    step_start=$(date +%s)

    cd "$MOMENTUM_REPO"
    python run_momentum.py --prices "$MASTER" --output "$EXCEL" --history-years 1 > /tmp/ic_step2.log 2>&1 || {
        echo "   ❌ Failed"
        cat /tmp/ic_step2.log
        notify "IC Pipeline" "❌ Failed at step 2: momentum scoring"
        print_footer "failed" $(($(date +%s) - pipeline_start)) ""
        exit 1
    }

    step_end=$(date +%s)
    step_duration=$((step_end - step_start))
    echo "   ✅ Done ($(format_duration $step_duration))"

    # Step 3: Generate presentation
    echo -n "[3/3] Generating presentation..."
    step_start=$(date +%s)

    cd "$IC_REPO"
    python cli_generate.py \
        --excel "$EXCEL" \
        --prices "$MASTER" \
        --template "$TEMPLATE" \
        --history "$HISTORY" \
        --output "$OUTPUT" \
        --date "$DATE_ARG" > /tmp/ic_step3.log 2>&1 || {
        echo "    ❌ Failed"
        cat /tmp/ic_step3.log
        notify "IC Pipeline" "❌ Failed at step 3: presentation generation"
        print_footer "failed" $(($(date +%s) - pipeline_start)) ""
        exit 1
    }

    step_end=$(date +%s)
    step_duration=$((step_end - step_start))
    echo "    ✅ Done ($(format_duration $step_duration))"

    # Find output file
    output_pptx=$(grep -o "[0-9]*_Herculis_Partners_Technical_Update.pptx" /tmp/ic_step3.log 2>/dev/null | tail -1 || echo "")
    if [ -n "$output_pptx" ]; then
        output_pptx="$OUTPUT/$output_pptx"
    fi

    # Calculate total duration
    local pipeline_end=$(date +%s)
    local total_duration=$((pipeline_end - pipeline_start))

    print_footer "success" $total_duration "$output_pptx"

    # macOS notification
    notify "IC Pipeline" "✅ Complete in $(format_duration $total_duration)"

    # Cleanup temp files
    rm -f /tmp/ic_step1.log /tmp/ic_step2.log /tmp/ic_step3.log
}

# Run main function
main "$@"
