#!/bin/bash 
# #SBATCH --job-name=exp_misr_joint_srdiff_lcc_train_backbone_hr5_sr4
# #SBATCH --partition=gpu
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:a100m40:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=4G
# #SBATCH --time=48:00:00
# #SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --output logs/exp_misr_joint_srdiff_lcc_train_backbone_hr5_sr4_%j.out
# #SBATCH --error logs/exp_misr_joint_srdiff_lcc_train_backbone_hr5_sr4_%j.err
# source load_modules.sh

export CONDA_ENVS_PATH=$HOME/miniconda3/envs
DATA_DIR="/hubert_storage/"
export DATA_DIR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /hubert_storage/flair_venv
which python
cd $HOME/exp_2026/LCC_FLAIR_HUB_SRDiff_DDPM_V_SN_TSViT_PDM
python trainer.py --config_file=./configs/train_main/ --exp_name srdiff_maxvit_ltae_ckpt --reset
