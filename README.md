# AutoRE\_ML\_NLP\_LLM\_Course\_Student

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


Certainly. Here’s the revised section, reflecting that you should set permissions on `/opt`, download and unzip Ghidra there, and noting the assumption that this is a VM with a single user.

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

Download Ghidra 11.2.1 to `/opt`:

```bash
cd /opt
wget https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.2.1_build/ghidra_11.2.1_PUBLIC_20241105.zip
unzip ghidra_11.2.1_PUBLIC_20241105.zip
```

### 3. Download and Install the Blackfyre Plugin for Ghidra

Follow the instructions and download the plugin from:

* [Blackfyre Ghidra Plugin v1.0.1 Release and Installation Instructions](https://github.com/jonescyber-ai/Blackfyre/releases/tag/v1.0.1)

Complete the plugin installation as described in the release notes.

---

**Note:** These steps assume you are the only user on the VM.
If you are working in a multi-user environment, consult your administrator before changing permissions in `/opt`.

Let me know if you need the Ghidra plugin installation steps detailed inline or have other environment constraints to document.


---

## Additional Notes

* Ensure all commands are run from the appropriate directories as indicated.
* If you encounter issues related to missing packages or permissions, consult the course troubleshooting guide or contact the instructor.

---
