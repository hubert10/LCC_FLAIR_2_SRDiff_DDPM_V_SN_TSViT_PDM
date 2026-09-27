#!/bin/bash 
#SBATCH --job-name=exp_misr_joint_srdiff_lcc_train_backbone
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100m40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/exp_misr_joint_srdiff_lcc_train_backbone_%j.out
#SBATCH --error logs/exp_misr_joint_srdiff_lcc_train_backbone_%j.err
source load_modules.sh
export CONDA_ENVS_PATH=$HOME/.conda/envs
export DATA_DIR=$BIGWORK
conda activate flair_venv
which python
cd $HOME/LCC_FLAIR_HUB_SRDiff_DDPM_V_SN_TSViT_PDM
# srun python trainer.py --config configs/diffsr_maxvit_ltae.yaml --config_file flair-config-server.yml --exp_name misr/srdiff_highresnet_ltae_ckpt --hparams="cond_net_ckpt=/bigwork/nhgnkany/Results/LCC_FLAIR_HUB_SRDiff_DDPM_V_SN_TSViT_PDM/results/checkpoints/misr/highresnet_ltae_ckpt" --reset
srun python trainer.py --config configs/diffsr_maxvit_ltae.yaml --config_file flair-config-server.yml --exp_name srdiff_maxvit_ltae_ckpt --hparams="cond_net_ckpt=/bigwork/nhgnkany/pretrain_weights/MISR_JOINT_SRDiff_HIGHRESNET_PRETRAINED/results/checkpoints/misr/maxvit_ltae_ckpt" --reset
