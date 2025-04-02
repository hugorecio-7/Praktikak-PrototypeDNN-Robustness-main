#!/bin/bash
####################################
#SBATCH --job-name=AutoAttack
# For job arrays, the %A represents the job ID and %a the array index
#SBATCH --output=logs/autoattack_%A_%a.out
#SBATCH --error=logs/autoattack_%A_%a.err
# Number of cores to be used. This option uses only one core
#SBATCH --ntasks=1
# Remember to ask for enough memory for your process. The default is 2Gb
#SBATCH --mem-per-cpu=2048
# #SBATCH --time=100:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
# #SBATCH --partition=ML-GPU
# #SBATCH --partition=ML-CPU
# #SBATCH --gres=gpu:rtxa5000:1
# #SBATCH --nodelist=nodo81
# Determine the number of repetitions of the process
#SBATCH --array=1-1
###################################

#Example:
# sbatch launch/probsenn_train_bnd.sh mnist None 40 cnn_32 70 0.001 250 False 0 False False False 30 0 results/res_senn_tf2/mnist_None_40_cnn_32_70_0.001_250_aFalse_p0_vFalse_wFalse_tFalse_n30_exp_0/

echo "Executing for: $1 $2"

outfile="logs/autoattack_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$1_$2.dat"
echo $outfile

#bnd -exec \
apptainer exec -B /pfs --nv /opt/ohpc/pub/containers/NGC-pytorch-23.12-py3.sif pipenv run  \
python run_test.py --models $1 --attacks $2 --max_eps 0.8 --step 0.025 > $outfile


