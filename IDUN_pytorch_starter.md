# IDUN Starter guide
## Login to server:
```
ssh haakohu@idun-login1.hpc.ntnu.no # Optionally, ssh haakohu@idun-login2.hpc.ntnu.no
```

## Transferring files

### Using rsync
Copying a local file named `RDD2020.zip` to a given folder on IDUN (in this case /cluster/home/haakohu/=HOME directory):
```
rsync -rvP RDD2022_Norway.zip haakohu@idun-login1.hpc.ntnu.no:/cluster/home/haakohu/
```
This can be used on files or directories (the 'r' flag = recursive copy of directories).

## Remotely connecting to the server via other tools:
- Connect to your NTNU home directory:
    - Windows: https://i.ntnu.no/wiki/-/wiki/English/Connect+to+your+home+directory+via+Windows
    - Mac: https://i.ntnu.no/wiki/-/wiki/English/Connect+to+your+home+directory+via+Mac+OS+X
- Using VSCODE: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh



## Running batch jobs (non-interactive, e.g. training jobs)
This tutorial is written for the username "haakohu". Replace all instances of "haakohu" with your own NTNU username (same as FEIDE username).

### Setting up your environment (PYTORCH TUTORIAL)
This can be done on the login node without starting a job.

```
# Load environment modules
module purge
module load cuDNN/8.2.1.32-CUDA-11.3.1 # This can be updated in the future, to see available modules type: 'module list cuDNN'. cuDNN/8.2.1.32-CUDA-11.3.1 is the most recent version as of 25/11/2022
module load Anaconda3/2020.07


conda create -n myEnvironmentName python=3.8 
conda activate myEnvironmentName
# Install all dependencies that you need.The following will install pytorch 1.12
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# You can also install other requirements, e.g. with pip:
pip install tqdm
```
Now that your environment is set up, you can start submitting jobs!

### Submitting jobs
Read [this](https://www.hpc.ntnu.no/idun/getting-started-on-idun/running-jobs/) introduction first.

Define your sbatch file:
```
#!/bin/sh
#SBATCH --partition=GPUQ # CHANGE THIS TO CPUQ IF YOU DON'T NEED GPUs!!
#SBATCH --account=ie-idi # READ THIS TO FIND YOUR ACCOUNT: https://www.hpc.ntnu.no/idun/getting-started-on-idun/accounting/
#SBATCH --mem=64GB # Total memory you need. Most DL jobs should do fine with 64GB.
#SBATCH --nodes=1 # Number of nodes. Leave this to 1 as long as you're not running multi-node/multi-gpu training
#SBATCH --cpus-per-task=4 # Number of CPU cores required. Most V100 nodes has 4 CPU cores per GPU, so to maximize usage of the nodes, leave it to 4 (unless you know that your specific node has more than 4 cores/GPU).
#SBATCH --job-name=my_training_job # Set this to whatever you want
#SBATCH --output=/cluster/home/haakohu/out.slurm # The output file. NOTE! The output directory (/cluster/home/haakohu/ in this case)  has to exist before you submit the job!
#SBATCH --time=8:00:00 # Maximum running time of your job.
#SBATCH --export=ALL # Export all environment variables.
#SBATCH --gres=gpu:V100:1 # GPU type and number of GPUs.
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total of GPUS: $CUDA_VISIBLE_DEVICES"
nvidia-smi
nvidia-smi nvlink -s
nvidia-smi topo -m
module purge
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Anaconda3/2020.07 
# >>> conda initialize >>> You might not need this, but in my case I need it for some reason....
__conda_setup="$('/cluster/apps/eb/software/Anaconda3/2020.07/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
   if [ -f "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh" ]; then
      . "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
   else
      export PATH="/cluster/apps/eb/software/Anaconda3/2020.07/bin:$PATH"
   fi
fi
unset __conda_setup
conda activate myEnvironmentName # NOTE! Change the environment variable
cd /cluster/home/haakohu # CD to the folder containing your files on the server
python train.py # NOTE! Change to your script to start training
```
Save this file to a file name, e.g. `/cluster/home/haakohu/train.sbatch`.
Then start train via:
```
sbatch /cluster/home/haakohu/train.sbatch
```

## Jupyter notebook
To connect use port forwarding when doing ssh:
```
ssh -o "ServerAliveInterval=60" -L 7050:localhost:7050 mamoonas@idun.hpc.ntnu.no
```
Request for a resource:
```
salloc --nodes=1 --partition=GPUQ --time=60:00:00 --gres=gpu:1 
```

Ssh into node using port forwarding:
```
ssh -L 7050:localhost:7050 idun-04-01
```
Activate your conda environment before running jupyter:  
```
conda activate my_env
```
Run jupyter notebook
```
jupyter notebook --no-browser --port=7050
```
Copy the provided link in your browser

To add a conda env in jupyter kernels:
#run in terminal in base env to add conda env in jupyter
``` 
python -m ipykernel install --user --name=[env_name]
```