# AutoRE\_ML\_NLP\_LLM\_Course\_Student

## Disclaimer:
Installation and use of this lab environment is _**only supported on AMD64 (x86_64) systems**_.
Due to dependencies such as VEXIR used in Blackfyre, installation on ARM-based systems (e.g., Apple Silicon/M1, Raspberry Pi) is not supported and may require significant troubleshooting.
As noted in the course requirements, please ensure you are using an AMD64-based virtual machine or host environment for the course labs.

## Lab Environment Setup

This repository provides the environment and instructions necessary for the hands-on portion of the course. Please follow each step carefully to ensure your setup matches the course requirements.

---

### 1. Download and Verify the Dataset

From the **root folder** of the repository, run the following commands to download and verify the dataset:

```bash
curl -L -o lab_datasets-v2.zip "https://www.dropbox.com/scl/fi/36tfewp71smsa54pzonqc/lab_datasets-v2.zip?rlkey=ndbefbecgl02sb84txyq8rlkm&st=kwr5ksk6&dl=0"
```

Verify the SHA-256 hash matches:

```bash
sha256sum lab_datasets-v2.zip
# Expected output: 6155cec22b283779ec998aeeacae138b9bebd24db41d69b15a078c3a08e47f90  lab_datasets-v2.zip
```

Unzip the dataset:

```bash
unzip lab_datasets-v2.zip
```

---

### 2. Clone the Blackfyre Repository

From the **root folder**, run:

```bash
git clone https://github.com/jonescyber-ai/Blackfyre
```

---

### 3. Enter the `student` Folder

Change into the `student` subdirectory (already present in this repo):

```bash
cd student
```

---

### 4. Create Required Symlinks

Within the `student` directory, create the following symbolic links:

```bash
ln -s ../env/ .
ln -s ../lab_common/ .
ln -s ../lab_datasets/ .
ln -s ../Blackfyre/ .
```

---

### 5. (If Needed) Install Python 3.10

> **Skip this step if Python 3.10 is already installed on your system.**
> The Tensor version used requires Python 3.10.
> (Adjust the following commands if you are not using Ubuntu/Debian.)

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

---

### 6. Create and Activate Virtual Environment (in `student` folder)

In the `student` directory, create and activate the virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

---

### 7. Install Python Dependencies

With the virtual environment activated, install requirements:

```bash
pip install -r env/requirements.txt
```

---

### 8. Install Blackfyre as an Editable Package

Navigate to the Blackfyre source directory and install in editable mode:

```bash
cd Blackfyre
cd src/python
pip install -e .
```

---

### 9. Set Script Permissions

Make the Ghidra example script executable:

```bash
chmod a+x Blackfyre/examples/ghidra/example_generate_bcc_headless.sh
```

---

## Ghidra 11.2.1 and Blackfyre Plugin Installation (Required for Lab 0)

The following steps are **required for Lab 0**.
These instructions assume you are working on a VM where you are the only user.

### 1. Install JDK 21

Ghidra 11.2.1 requires JDK 21:

```bash
sudo apt install openjdk-21-jdk -y
```

### 2. Set Permissions and Install Ghidra in `/opt`

Set ownership of `/opt` so your user can manage the Ghidra installation (recommended in a single-user VM):

```bash
sudo chown -R "$USER":"$USER" /opt
```

Download and unzip Ghidra 11.2.1 in `/opt`:

```bash
cd /opt
wget https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.2.1_build/ghidra_11.2.1_PUBLIC_20241105.zip
unzip ghidra_11.2.1_PUBLIC_20241105.zip
```

### 3. Download and Install the Blackfyre Plugin

Download the Blackfyre plugin release file in `/opt`:

```bash
cd /opt
wget https://github.com/jonescyber-ai/Blackfyre/releases/download/v1.0.1/ghidra_11.2.1_PUBLIC_20250111_Blackfyre.zip
```

 

#### Install the Plugin in Ghidra

1. Open Ghidra (from `/opt/ghidra_11.2.1_PUBLIC/ghidraRun`).
2. In the Ghidra menu, go to **File → Install Extensions**.
3. Click "Add" or "Install" and select the plugin file:
   `/opt/blackfyre-ghidra-plugin-v1.0.1.zip`
4. After installation, ensure the plugin is **enabled** in the Extensions manager.
5. Restart Ghidra if prompted.

---

**Note:** These steps assume you are the only user on the VM.
If you are working in a multi-user environment, consult your administrator before changing permissions in `/opt`.


## Additional Notes

* Ensure all commands are run from the appropriate directories as indicated.
* If you encounter issues related to missing packages or permissions, consult the course troubleshooting guide or contact the instructor.

---
