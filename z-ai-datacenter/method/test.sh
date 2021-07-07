#!/bin/bash
 
#SBATCH -J  attn-comb          # Job name
#SBATCH -o  out.attn-comb.%j    # Name of stdout output file (%j expands to %jobId)
#SBATCH -p  base                        # queue or partiton name
#SBATCH -t  48:00:00                   # Max Run time (hh:mm:ss) - 1.5 hours

#SBATCH --gres=gpu:1                   # Total number of gpu requested
 
cd ~
 
module purge
module load cuda/11.1
 
# >>> conda initialize >>>
source $HOME/anaconda3/etc/profile.d/conda.sh
 
conda activate expert
 
cd /home/yujin731/rexpert/methods/adapter-combine
ECHO "TEST"
 
# End of File.