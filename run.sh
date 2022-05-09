conda_env="/projects/rlmolecule/jlaw/envs/crystals_nfp0_3"
if [ "$1" == "" ]; then
    echo "Need to pass <out_dir> as first argument"
    exit
fi
out_dir="$1"
run_id=$(basename $out_dir)
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
     --data-file=$out_dir/all_data.p
echo \"Job finished at: \$(date)\"
"""
     #--data-file=$out_dir/volunrelax_data.p

echo "$submit_script" > $out_dir/submit.sh
echo "sbatch $out_dir/submit.sh"
sbatch $out_dir/submit.sh
