#!/bin/bash 
#SBATCH --job-name=exp_misr_srdiff_lcc_test_inference_maxvit_hr5_convformer_sr4_caf_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100m40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=30:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/exp_misr_srdiff_lcc_test_inference_maxvit_hr5_convformer_sr4_caf_all_%j.out
#SBATCH --error logs/exp_misr_srdiff_lcc_test_inference_maxvit_hr5_convformer_sr4_caf_all_%j.err
source load_modules.sh
export CONDA_ENVS_PATH=$HOME/.conda/envs
export DATA_DIR=$BIGWORK
conda activate flair_venv
which python
cd $HOME/LCC_FLAIR_HUB_SRDiff_DDPM_V_SN_TSViT_PDM
srun python trainer.py --config configs/diffsr_maxvit_ltae.yaml --config_file flair-config-server.yml --exp_name misr/srdiff_highresnet_ltae_ckpt --hparams="cond_net_ckpt=/bigwork/nhgnkany/Results/LCC_FLAIR_HUB_SRDiff_DDPM_V_SN_TSViT_PDM/results/checkpoints/misr/srdiff_highresnet_ltae_ckpt" --infer
