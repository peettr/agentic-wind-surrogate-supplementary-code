#!/bin/bash
# preflight_check.sh â€” Run before every Condor job submission
# Usage: bash preflight_check.sh <submit_file> <campaign_dir>
# Returns 0 if all checks pass, 1 if issues found (with auto-fix where possible)

set -euo pipefail

SUBMIT_FILE="${1:?Usage: preflight_check.sh <submit_file> <campaign_dir>}"
CAMPAIGN_DIR="${2:?Usage: preflight_check.sh <submit_file> <campaign_dir>}"
ISSUES=0
FIXES=0

echo "=== Preflight Check ==="
echo "Submit: $SUBMIT_FILE"
echo "Campaign: $CAMPAIGN_DIR"
echo ""

# 1. Check submit file exists
if [ ! -f "$SUBMIT_FILE" ]; then
    echo "âŒ FAIL: Submit file not found: $SUBMIT_FILE"
    exit 1
fi
echo "âœ… Submit file exists"

# 2. Parse all run names from submit file
NAMES=$(grep "^Name = " "$SUBMIT_FILE" | sed 's/Name = //' | sort)
N_TOTAL=$(echo "$NAMES" | wc -l)
echo "âœ… Found $N_TOTAL runs in submit file"

# 3. Check each run has a train_config.json
MISSING_CONFIG=0
for name in $NAMES; do
    CFG="$CAMPAIGN_DIR/$name/train_config.json"
    if [ ! -f "$CFG" ]; then
        echo "âŒ MISSING: $CFG"
        MISSING_CONFIG=$((MISSING_CONFIG + 1))
        ISSUES=1
    fi
done
if [ $MISSING_CONFIG -eq 0 ]; then
    echo "âœ… All $N_TOTAL train_config.json present"
fi

# 4. Check output/error file permissions & stale files
PERM_ISSUES=0
STALE_FILES=0
for name in $NAMES; do
    for ext in out err log; do
        F="$CAMPAIGN_DIR/$name.$ext"
        if [ -f "$F" ]; then
            # Check writable
            if [ ! -w "$F" ]; then
                echo "âš ï¸  FIX: Removing unwritable $F"
                rm -f "$F" 2>/dev/null || true
                PERM_ISSUES=$((PERM_ISSUES + 1))
                FIXES=$((FIXES + 1))
            else
                # Stale file from previous run â€” truncate
                echo "âš ï¸  FIX: Truncating stale $F"
                > "$F"
                STALE_FILES=$((STALE_FILES + 1))
                FIXES=$((FIXES + 1))
            fi
        fi
    done
    
    # Also check run dir permissions
    RUNDIR="$CAMPAIGN_DIR/$name"
    if [ -d "$RUNDIR" ] && [ ! -w "$RUNDIR" ]; then
        echo "âš ï¸  FIX: chmod run dir $RUNDIR"
        chmod 755 "$RUNDIR" 2>/dev/null || true
        FIXES=$((FIXES + 1))
    fi
    
    # Check stale sentinel/model files from previous runs
    for stale in STARTED FINISHED FAILED HEARTBEAT.json model_best.pt checkpoint.pt train.log; do
        SF="$RUNDIR/$stale"
        if [ -f "$SF" ]; then
            echo "âš ï¸  FIX: Removing stale $SF"
            rm -f "$SF"
            STALE_FILES=$((STALE_FILES + 1))
            FIXES=$((FIXES + 1))
        fi
    done
done
if [ $PERM_ISSUES -eq 0 ] && [ $STALE_FILES -eq 0 ]; then
    echo "âœ… No permission/stale file issues"
fi

# 5. Validate train_config.json has required fields
BAD_CONFIG=0
for name in $NAMES; do
    CFG="$CAMPAIGN_DIR/$name/train_config.json"
    if [ -f "$CFG" ]; then
        for field in experiment_id seed epochs lr batch_size arch_name loss_name data_dir results_dir split_manifest_path; do
            if ! python3 -c "import json; c=json.load(open('$CFG')); assert '$field' in c or '$field' in c.get('arch_kwargs',{}).get('training',{})" 2>/dev/null; then
                echo "âŒ BAD CONFIG: $name missing field '$field'"
                BAD_CONFIG=$((BAD_CONFIG + 1))
                ISSUES=1
            fi
        done
        # Check compute_r2 is set
        if ! python3 -c "import json; c=json.load(open('$CFG')); assert c.get('compute_r2') == True" 2>/dev/null; then
            echo "âš ï¸  WARNING: $name compute_r2 not True"
        fi
    fi
done
if [ $BAD_CONFIG -eq 0 ]; then
    echo "âœ… All configs have required fields"
fi

# 6. Check data files exist
DATA_DIR=$(python3 -c "import json; c=json.load(open('$CAMPAIGN_DIR/$(echo $NAMES | head -1)/train_config.json')); print(c['data_dir'])" 2>/dev/null || echo "")
if [ -n "$DATA_DIR" ]; then
    for f in all_data.pt split_manifest.json; do
        if [ -f "$DATA_DIR/$f" ]; then
            echo "âœ… Data file exists: $DATA_DIR/$f"
        else
            echo "âŒ MISSING DATA: $DATA_DIR/$f"
            ISSUES=1
        fi
    done
fi

# 7. Check wrapper script exists
WRAPPER=$(grep "^executable = " "$SUBMIT_FILE" | head -1 | sed 's/executable = //')
if [ -n "$WRAPPER" ] && [ -f "$WRAPPER" ]; then
    echo "âœ… Wrapper exists: $WRAPPER"
else
    echo "âŒ MISSING WRAPPER: $WRAPPER"
    ISSUES=1
fi

# 8. Summary
echo ""
echo "=== Summary ==="
echo "Runs: $N_TOTAL"
echo "Fixes applied: $FIXES"
if [ $ISSUES -eq 0 ]; then
    echo "âœ… All checks passed â€” safe to submit"
    exit 0
else
    echo "âŒ Issues found â€” review before submitting"
    exit 1
fi



