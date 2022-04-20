conda_env="/projects/rlmolecule/jlaw/envs/crystals_nfp0_3"
#run_id="20220314_volunrelax_dls1.5"
#run_id="20220314_batt_icsd_and_volunrelax"
#out_dir="outputs/$run_id"
run_id="rel_unrel_split"
base_dir="outputs/20220314_batt_icsd_and_volunrelax"
out_dir="$base_dir/$run_id"
mkdir -p $out_dir

echo "cp train_model.py $out_dir"
cp train_model.py $out_dir

submit_script="""#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=2-00
#SBATCH --job-name=$run_id
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=$out_dir/run.out
#SBATCH --mail-user=jlaw@nrel.gov
#SBATCH --mail-type=END
source ~/.bashrc
module load cudnn/8.1.1/cuda-11.2
conda activate $conda_env
echo \"\$PWD\"
echo \"Job started at: \$(date)\"
srun python $out_dir/train_model.py \
     --out-dir=$out_dir \
     --data-file=$base_dir/all_data.p
echo \"Job finished at: \$(date)\"
"""
     #--data-file=$out_dir/volunrelax_data.p

echo "$submit_script" > $out_dir/submit.sh
echo "sbatch $out_dir/submit.sh"
sbatch $out_dir/submit.sh
