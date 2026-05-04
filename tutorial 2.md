# How to Use CINECA: A Tutorial

Welcome to this tutorial on how to use CINECA's High-Performance Computing (HPC) system. This guide will walk you through the steps to install the STEP client, generate a temporary certificate, and log in to the CINECA HPC environment.

## Prerequisites

Before proceeding, please ensure that:

- You have created a CINECA account.
- Two-factor authentication (2FA) is set up.

If you haven't completed these steps, please refer to the [CINECA HPC User Guide](https://wiki.u-gov.it/confluence/display/SCAIUS/HPC+User+Guide) for detailed instructions.

## Installing the STEP Client

To access the CINECA HPC system, you'll need to install the STEP client to generate a temporary certificate. Here's a script designed to run on IIS machines:

```bash
wget https://dl.smallstep.com/gh-release/cli/gh-release-header/v0.27.2/step_linux_0.27.2_amd64.tar.gz
tar -xf step_linux_0.27.2_amd64.tar.gz
mkdir ~/bin
cp step_0.27.2/bin/step ~/bin/
```
This script performs the following actions:

1. Downloads the STEP client package.
2. Extracts the contents of the downloaded archive.
3. Creates a bin directory in your home folder.
4. Copies the step binary to your ~/bin/ directory.
After running this script, you're ready to create the temporary certificate.

## Logging in to CINECA
Next, we'll use the STEP client to generate a temporary certificate and log in to CINECA. To simplify this process, use the following script. Remember to replace <EMAIL_ADDRESS> and <CINECA_USERNAME> with your actual email address and CINECA username.

```bash
#!/bin/bash
step ca bootstrap -f --ca-url=https://sshproxy.hpc.cineca.it \
  --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
eval $(ssh-agent)
ssh-keygen -R login.leonardo.cineca.it
step ssh login '<EMAIL_ADDRESS>' --provisioner cineca-hpc
ssh-add
ssh <CINECA_USERNAME>@login.leonardo.cineca.it
```
#### Script Explanation:

1. Bootstrap the STEP Client:
 - Connects to the CINECA Certificate Authority (CA) to obtain configuration details.
2. Start the SSH Agent:
 - Initializes the SSH agent for key management.
3. Remove Old SSH Keys:
 - Clears any previous SSH keys associated with login.leonardo.cineca.it.
4. Login with STEP SSH:
 - Initiates the SSH login process using your email and the cineca-hpc provisioner.
5. Add SSH Key to Agent:
 - Adds your SSH key to the agent for authentication.
6. SSH into CINECA:
 - Connects you to the CINECA HPC system.

When you run this script, a web browser window will open, directing you to the CINECA login page. Follow these steps:

1. Authenticate:
 - Enter your CINECA login credentials.
2. Verify 2FA:
 - Provide your two-factor authentication code.

After completing these steps, the script will automatically log you into Leonardo, CINECA's HPC system.

Congratulations! You are now logged into CINECA's HPC system and can begin utilizing their computational resources. 

## Data Storage Locations on CINECA

CINECA provides several data storage locations classified into two main categories: **Permanent** and **Temporary**.

- **Permanent Storage**: Data is retained for the entire duration of the CINECA project plus an additional 6 months.
- **Temporary Storage**: Data is deleted either after 40 days from file creation or immediately after the batch script ends.

### Available Storage Locations

1. **`$HOME`** (50 GB limit) &mdash; *Permanent*
2. **`$WORK`** (1 TB limit) &mdash; *Permanent*
3. **`$FAST`** (1 TB limit) &mdash; *Permanent*
4. **`$CINECA_SCRATCH`** (No limit) &mdash; *Temporary*
5. **`$TMPDIR`** (Technically no limits, but excessive use during script execution will incur additional computing hours) &mdash; *Temporary*
6. **`$DRES`** &mdash; *Permanent (Not recommended for use)*

### User-Specific vs. Project-Specific Storage

- **User-Specific Locations**: Accessible only by the user who owns the directory.
  - **`$HOME`**
  - **`$CINECA_SCRATCH`**
  - **`$TMPDIR`**
- **Project-Specific Locations**: Accessible by all collaborators involved in the specific project.
  - **`$WORK`**
  - **`$FAST`**

### Where to Store Code and Data

- **Code and Environments**: Store in **`$HOME`**.
  - Regularly backed up.
  - Adequate space for code and environment files.
- **Data**: Store in **`$WORK`** or **`$FAST`**.
  - Both are project-shared, allowing collaborators to access the data. 
  - **`$FAST`**:
    - Ideal for applications where I/O operations are the bottleneck.
    - Performs well with both large data blocks and frequent small I/O operations.
  - **`$WORK`**:
    - Designed for hosting large working data files.
    - Offers high bandwidth suitable for parallel file systems.
    - Excels with large data blocks but not ideal for frequent small I/O operations.
  - If you do not want to share the data with other users on the same project then **`$CINECA_SCRATCH`** is an option but be aware that this location is not permanent.

### Moving data over to CINECA

Here is the example code I used to move data over in the tutorial video:

```bash
rsync -avz --progress <USERNAME>@<SERVER>:<INTERNAL_DIRECTORY> <PATH_ON_CINECA>
```
Make sure the <PATH_ON_CINECA> is either **`$WORK`** or **`$FAST`** (or **`$CINECA_SCRATCH`**)

## Creating a Python Virtual Environment on CINECA

After logging into CINECA's Leonardo system, we'll proceed to create a Python virtual environment (venv). We'll perform this task on the **login node** because it provides internet access (not guaranteed on compute nodes) and we don't require GPUs for creating a Python environment.

### Recommended Reading

For more detailed information, consider reading the [CINECA User Guide on Deep Learning](https://wiki.u-gov.it/confluence/display/SCAIUS/Leonardo+-+Scientific+Python+user+environment+and+tools+for+AI%3A+the+CINECA+Artificial+Intelligence+project). As recommended in the guide, we'll load predefined modules that CINECA has compiled for users.

### Loading Predefined Modules

Run the following commands in the terminal to load the necessary modules:

```bash
module load profile/deeplrn
module av cineca-ai
module load cineca-ai/4.3.0
```


**Explanation:**

- `module load profile/deeplrn`: Loads the deep learning profile.
- `module av cineca-ai`: Lists available versions of the `cineca-ai` module.
- `module load cineca-ai/4.3.0`: Loads version 4.3.0 of the `cineca-ai` module.

### Navigating to the Correct Directory

Ensure you're in the correct directory for storing code (`$HOME`):
```bash
cd $HOME
```

### Creating the Virtual Environment

Create a new Python virtual environment using the following command:

```bash
python -m venv timefm --system-site-packages
```

**Notes:**

- `timefm`: This is the name of your virtual environment. Feel free to choose a name that suits your project.
- `--system-site-packages`: This option allows your virtual environment to access the pre-compiled packages provided by CINECA.

### Activating the Virtual Environment

Activate your virtual environment with:
```bash
source timefm/bin/activate
```

### Installing Python Packages

With the virtual environment activated, you can now install any required Python packages using `pip`. In the video tutorial, the following commands were executed:
```bash
pip install -r requirements.txt 
pip install nvitop
```
- `pip install -r requirements.txt`: Installs all packages listed in your `requirements.txt` file.
- `pip install nvitop`: Installs the `nvitop` package, which can be used for monitoring NVIDIA GPUs.

---

At this point, your Python virtual environment is set up and ready for use. You can now proceed to develop and run your Python applications on the CINECA HPC system.

---

## Debugging GPU Usage of Your Code

With your Python virtual environment set up, the next step is to debug the GPU usage of your code. To accomplish this, we'll request an **interactive environment** from CINECA. This environment allocates the resources you specify and provides a terminal interface for debugging purposes.

### Requesting an Interactive Environment

Run the following command in the terminal to request an interactive session:


```bash
srun --nodes=1 --gpus=4 --ntasks-per-node=4 --cpus-per-task=8 --account <PUT_ACCOUNT> -p boost_usr_prod --pty /bin/bash
```


**Explanation of the `srun` Command:**

- `--nodes=1`: Requests **one node**.
- `--gpus=4`: Allocates **four GPUs**.
- `--ntasks-per-node=4`: Specifies **four tasks per node** (one per GPU).
- `--cpus-per-task=8`: Assigns **eight CPUs per task/GPU**.
- `--account <PUT_ACCOUNT>`: Indicates your account; replace `<PUT_ACCOUNT>` with your actual account name.
- `-p boost_usr_prod`: Selects the **`boost_usr_prod`** partition.
- `--pty /bin/bash`: Starts an interactive **bash shell** session.

**Important Notes:**

- **Node and GPU Allocation:**
  - It's recommended to request only **one node** for interactive debugging.
  - Each node has a maximum of **four GPUs**; hence, we request up to four GPUs.
- **Task Allocation:**
  - Ensure that `--ntasks-per-node` equals the number of GPUs requested per node.
- **Billing Information:**
  - Be aware of how resource usage is billed on CINECA (explained below).

### Understanding CINECA's Billing System

CINECA bills resource usage based on **CPU hours**, not GPU hours. Each GPU is considered equivalent to **eight CPUs**. Here are some billing examples:

1. **Requesting 2 GPUs and 8 CPUs per GPU:**
   - **Total CPUs:** 2 GPUs × 8 CPUs/GPU = **16 CPUs**
   - **Billing Rate:** Billed for **16 CPU hours per hour used**.

2. **Requesting 1 GPU and 4 CPUs:**
   - **Total CPUs:** 1 GPU × 8 CPUs/GPU = **8 CPUs**
   - **Billing Rate:** Even if fewer CPUs are requested, you are billed for **8 CPU hours per hour used** (minimum for one GPU).

3. **Requesting 2 GPUs and 32 CPUs:**
   - **Total CPUs Requested:** **32 CPUs**
   - **Billing Rate:** Since the requested CPUs (32) exceed the GPU allocation equivalent (16 CPUs), you are billed for **32 CPU hours per hour used**.

**Account and Partition Details:**

- **`--account`:**
  - Specify the correct account to ensure proper billing.
  - If you're part of multiple projects, it's crucial to select the right account.
- **Partition (`-p`):**
  - The `boost_usr_prod` partition is recommended for general use.
  - For alternative partitions or Quality of Service (QoS) options, refer to the [LEONARDO Booster User Guide](https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.2.1%3A+LEONARDO+Booster+UserGuide).

### Starting Your Debugging Session

After a short wait, your interactive environment will be ready, and you'll have access to the allocated resources. You can now proceed with debugging your Python code.

#### Commands Executed During Debugging

Navigate to your home directory and then to your code directory:

```bash
cd $HOME
# Here you can then activate your venv (if not already activated)
source timefm/bin/activate
cd $HOME/code/TimeFM
```
Run your Python script with the desired parameters:

```bash
python run_finetune.py +experiment=TSTT_finetune finetune_pretrained=False gpus=4 num_workers=8 batch_size=4096 > ~/timefm2.log 2>&1 & nvitop
```


**Explanation:**

- `python run_finetune.py`: Executes your Python script.
- `+experiment=TSTT_finetune`: Specifies the experiment configuration.
- `finetune_pretrained=False`: Sets a parameter for the script.
- `gpus=4`: Utilizes four GPUs.
- `num_workers=8`: Sets the number of worker processes.
- `batch_size=4096`: Defines the batch size.
- `> ~/timefm2.log 2>&1 &`: Redirects both standard output and error to `timefm2.log` and runs the process in the background.
- `nvitop`: Launches `nvitop` to monitor GPU utilization.

### Monitoring GPU Utilization

It's important to verify that your code efficiently utilizes the GPU resources. You should check:

- **Memory Usage:** Ensure that the GPU memory is being utilized effectively.
- **Compute Utilization:** Confirm that the GPUs are actively processing computations.

The `nvitop` utility provides real-time monitoring and should display high utilization metrics if your code is optimized.

### Exiting the Interactive Environment

Once you're satisfied with your debugging session:

1. **Terminate Your Python Process (Optional but Recommended):**

   - Gracefully stop your Python script to free up resources.

2. **Exit the Interactive Session:**

   - Type `exit` in the terminal to leave the interactive environment.

#### Verifying Termination of Interactive Environment

If you're unsure whether the interactive environment has been terminated:

- **Check Running Jobs:**
```bash
squeue --me
```

- This command lists all jobs associated with your user.

- **Cancel Running Jobs:**

If you have any running jobs that need to be terminated, use:
```bash
scancel <JOBID>
```

- Replace `<JOBID>` with the Job ID obtained from the `squeue` command.

---

By following these steps, you can effectively debug and optimize the GPU usage of your code on CINECA's HPC system.

## Creating and Submitting Batch Scripts on CINECA

Now that you have verified your Python code and ensured it fully utilizes the GPUs, it's time to create batch scripts. Batch scripts are essential for scheduling jobs, running code on multiple nodes simultaneously, and automating workflows by setting job dependencies.

### Benefits of Batch Scripts

- **Parallel Execution**: Run code on multiple nodes at once.
- **Job Scheduling**: Queue jobs to run sequentially or based on dependencies.
- **Automation**: Automate tasks like pre-training and fine-tuning models.

### Creating the Batch Script (`training.sbatch`)

Here's an example of a batch script named `training.sbatch`:

```bash
#!/bin/bash
#SBATCH -A <PUT_ACCOUNT>
#SBATCH -p boost_usr_prod
#SBATCH --time=00:10:00      # Format: HH:MM:SS
#SBATCH --nodes=1            # Number of nodes
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4         # Number of GPUs per node
#SBATCH --mem=494000         # Memory per node in MB (481GB)
#SBATCH --job-name=testing_gpus
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=<PUT_EMAIL_ADDRESS>

# Ensure the correct directory
cd $HOME

# Activate the virtual environment
source ./timefm/bin/activate

# Run the job
srun bash run_pretrain.sh
```

Explanation of the Script Parameters
`#!/bin/bash`: Specifies the script interpreter.
- `#SBATCH -A <PUT_ACCOUNT>`: Sets the account to charge for resource usage. Replace <PUT_ACCOUNT> with your account name.
- `#SBATCH -p boost_usr_prod`: Selects the partition (boost_usr_prod) to submit the job to.
- `#SBATCH --time=00:10:00`: Sets the maximum wall time to 10 minutes.
- `#SBATCH --nodes=1`: Requests one node.
- `#SBATCH --ntasks-per-node=4`: Specifies four tasks per node, typically one per GPU.
- `#SBATCH --cpus-per-task=8`: Allocates eight CPUs per task.
- `#SBATCH --gres=gpu:4`: Requests four GPUs on the node.
- `#SBATCH --mem=494000`: Allocates 494,000 MB (approximately 481 GB) of memory per node.
- `#SBATCH --job-name=testing_gpus`: Names the job testing_gpus.
- `#SBATCH --mail-type=BEGIN` and `#SBATCH --mail-type=END`: Sends email notifications at the start and end of the job.
- `#SBATCH --mail-user=<PUT_EMAIL_ADDRESS>`: Email address for notifications. Replace <PUT_EMAIL_ADDRESS> with your email.

**Script Commands:**

- `cd $HOME`: Changes directory to your home directory.
- `source ./timefm/bin/activate`: Activates your Python virtual environment named timefm.
- `srun bash run_pretrain.sh`: Executes the run_pretrain.sh script using srun.

**Creating the Internal Bash Script (`run_pretrain.sh`)**
Since we're using PyTorch Lightning, which integrates well with SLURM when using srun, we'll encapsulate our execution command within a separate bash script named run_pretrain.sh (feel free to change the name to something else):
```bash
#!/bin/bash

cd $HOME

# Activate the virtual environment
source timefm/bin/activate

cd $HOME/code/TimeFM

python run_finetune.py +experiment=TSTT_finetune finetune_pretrained=False gpus=4 num_workers=8 batch_size=4096
```

**Notes**:

The script navigates to your code directory and runs the Python script with the specified parameters.

Ensure the script is executable by running:

```bash
chmod +x run_pretrain.sh
```
#### Submitting the Batch Job
To submit the batch script to the scheduler, run:
```bash
sbatch training.sbatch
```
This command will queue your job for execution. If you've set up email notifications in your batch script, you'll receive emails when your job starts and ends.

#### Monitoring and Managing Jobs
**Check Job Status:**

```bash
squeue --me
```
This command lists all jobs associated with your user.

**Cancel a Job:**

```bash
scancel <JOBID>
```
Replace `<JOBID>` with the Job ID obtained from the squeue command.


## Conclusion

In this tutorial, we've covered the essential steps to effectively utilize CINECA's High-Performance Computing (HPC) resources for your computational projects. Here's a concise summary of the most important points:

### Access and Authentication

- **Account Setup:**
  - Created a CINECA account and configured two-factor authentication (2FA).
  - Referred to the [CINECA HPC User Guide](https://wiki.u-gov.it/confluence/display/SCAIUS/HPC+User+Guide) for detailed instructions.

- **Installing the STEP Client:**
  - Installed the STEP client to generate temporary certificates for secure login.
  - Used a provided script to simplify the installation process.

- **Logging into CINECA:**
  - Generated a temporary certificate and logged into the Leonardo system.
  - Understood the importance of interactive environments for debugging.

### Data Storage Management

- **Storage Locations:**
  - Learned about permanent and temporary storage options:
    - **Permanent:** `$HOME`, `$WORK`, `$FAST`.
    - **Temporary:** `$CINECA_SCRATCH`, `$TMPDIR`.

- **Storage Usage:**
  - Stored code and environments in `$HOME` for regular backups.
  - Placed data in `$WORK` or `$FAST` depending on I/O requirements.
  - Recognized the difference between user-specific and project-specific directories.

### Environment Setup

- **Loading Modules:**
  - Loaded predefined modules using `module load` commands.
  - Utilized the `cineca-ai` module for deep learning tools.

- **Creating a Virtual Environment:**
  - Created a Python virtual environment with access to system site packages.
  - Activated the environment and installed necessary Python packages.

### Debugging and Resource Monitoring

- **Interactive Environment:**
  - Requested an interactive session using `srun` for debugging purposes.
  - Allocated appropriate resources (nodes, GPUs, CPUs) while considering billing implications.

- **Monitoring GPU Usage:**
  - Used `nvitop` to monitor GPU memory and compute utilization.
  - Ensured code efficiently utilized allocated GPU resources.

- **Billing Awareness:**
  - Understood CINECA's billing system based on CPU hours.

### Batch Job Submission

- **Creating Batch Scripts:**
  - Wrote a batch script (`training.sbatch`) with appropriate `#SBATCH` directives.
  - Included resource specifications, job naming, and email notifications.

- **Internal Execution Scripts:**
  - Created an internal bash script (`run_pretrain.sh`) to run code with `srun`.
  - Made the script executable using `chmod +x`.

- **Submitting and Managing Jobs:**
  - Submitted batch jobs using `sbatch`.
  - Monitored job status with `squeue --me`.
  - Canceled jobs when necessary using `scancel <JOBID>`.

### Best Practices

- **Resource Optimization:**
  - Verified code performance before batch submission to optimize resource usage.
  - Used interactive sessions for thorough testing and debugging.

- **Documentation and Support:**
  - Referred to CINECA's documentation for detailed guidance.

---

For more information and advanced usage, consult the [CINECA HPC User Guide](https://wiki.u-gov.it/confluence/display/SCAIUS/HPC+User+Guide) and other related documentation.

**Thank you for following this tutorial!**


