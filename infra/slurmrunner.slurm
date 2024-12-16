#!/bin/zsh
#SBATCH --job-name=json_processing     # Job name
#SBATCH --output=slurm_%A_%a.out  # Standard output and error log
#SBATCH --error=slurm_%A_%a.err   # Error log
#SBATCH --array=1-253                 # Array range (adjust as needed)
#SBATCH --ntasks=1                      # Run one task per array job
#SBATCH --time=1-00:00:00                 # Max time per job (adjust as needed)
#SBATCH --account=rcac-cpu                # Slurm account
#SBATCH --mem=8G


# Load the environment
source ~/.zshrc

if [ -f "/home/dchawra/miniforge3/etc/profile.d/mamba.sh" ]; then
    . "/home/dchawra/miniforge3/etc/profile.d/mamba.sh"
fi

conda activate tlm

# Set paths
PROGRAM="/anvil/scratch/dchawra/tvm-gnn/lower_meas_recs.py"
INPUT_DIR="/anvil/scratch/dchawra/tlm/gen/gen_data/measure_data_v100"
OUTPUT_DIR="/anvil/scratch/dchawra/output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get the list of files
FILES=($(ls "$INPUT_DIR"/*.json))

# Determine the file to process for this array task
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

# Run the program for the specific file
if [ -n "$FILE" ]; then
    python "$PROGRAM" --file "$FILE" --output_dir "$OUTPUT_DIR"
else
    echo "No file for task ID $SLURM_ARRAY_TASK_ID"
    echo "file: $FILE"
fi
