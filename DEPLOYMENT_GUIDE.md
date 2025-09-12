# 🏥 KOOS-PS Model Deployment Guide for Medical Staff

## 📋 Overview

This guide will walk you through setting up the KOOS-PS prediction model on a remote computer, step by step. You don't need any programming experience - just follow the instructions carefully.

**What you'll accomplish:**
- Download the AI model to your computer
- Install the necessary software
- Set up your patient data
- Run the model to predict knee surgery outcomes

---

## 🎯 Prerequisites

### What you need:
- A computer with Windows, Mac, or Linux
- Internet connection
- Administrator access to install software
- Your patient data (X-ray images and CSV file)
- At least 8GB of RAM (16GB recommended)
- At least 10GB of free disk space

### What you'll get:
- A working AI model that predicts KOOS-PS scores
- Visual reports showing prediction accuracy
- Detailed results for each patient

---

## 📥 Step 1: Download and Install Git

Git is a tool that helps you download the AI model from the internet.

### For Windows:

#### 1.1 Download Git
1. Open your web browser
2. Go to: https://git-scm.com/download/win
3. Click "Download for Windows"
4. The download will start automatically

#### 1.2 Install Git
1. Find the downloaded file (usually in your Downloads folder)
2. Double-click on it to run the installer
3. Click "Next" through all the installation screens
4. **Important**: Keep all default settings
5. Click "Install" and wait for it to finish
6. Click "Finish"

#### 1.3 Verify Installation
1. Press `Windows + R` keys together
2. Type `cmd` and press Enter
3. Type `git --version` and press Enter
4. You should see something like "git version 2.x.x"
5. If you see an error, restart your computer and try again

### For Mac:

#### 1.1 Install Xcode Command Line Tools
1. Open Terminal (press `Cmd + Space`, type "Terminal", press Enter)
2. Type: `xcode-select --install`
3. Press Enter
4. Click "Install" when prompted
5. Wait for installation to complete

#### 1.2 Verify Installation
1. In Terminal, type: `git --version`
2. Press Enter
3. You should see something like "git version 2.x.x"

### For Linux (Ubuntu/Debian):

#### 1.1 Install Git
1. Open Terminal (press `Ctrl + Alt + T`)
2. Type: `sudo apt update`
3. Press Enter and enter your password when prompted
4. Type: `sudo apt install git`
5. Press Enter and wait for installation

#### 1.2 Verify Installation
1. Type: `git --version`
2. Press Enter
3. You should see something like "git version 2.x.x"

---

## 🐍 Step 2: Download and Install Python

Python is the programming language the AI model uses.

### For Windows:

#### 2.1 Download Python
1. Go to: https://www.python.org/downloads/
2. Click "Download Python 3.x.x" (latest version)
3. The download will start automatically

#### 2.2 Install Python
1. Find the downloaded file and double-click it
2. **IMPORTANT**: Check "Add Python to PATH" at the bottom
3. Click "Install Now"
4. Wait for installation to complete
5. Click "Close"

#### 2.3 Verify Installation
1. Press `Windows + R`, type `cmd`, press Enter
2. Type: `python --version`
3. Press Enter
4. You should see "Python 3.x.x"

### For Mac:

#### 2.1 Install Python
1. Go to: https://www.python.org/downloads/
2. Click "Download Python 3.x.x"
3. Open the downloaded file
4. Follow the installation instructions
5. **Important**: Make sure to check "Add Python to PATH"

#### 2.2 Verify Installation
1. Open Terminal
2. Type: `python3 --version`
3. Press Enter
4. You should see "Python 3.x.x"

### For Linux:

#### 2.1 Install Python
1. Open Terminal
2. Type: `sudo apt update`
3. Press Enter
4. Type: `sudo apt install python3 python3-pip`
5. Press Enter and wait for installation

#### 2.2 Verify Installation
1. Type: `python3 --version`
2. Press Enter
3. You should see "Python 3.x.x"

---

## 📁 Step 3: Download the AI Model

Now we'll download the KOOS-PS prediction model to your computer.

### 3.1 Create a Project Folder
1. Open File Explorer (Windows) or Finder (Mac)
2. Navigate to your Desktop
3. Right-click and select "New Folder"
4. Name it "koos_model"
5. Double-click to open it

### 3.2 Download the Model
1. Open Command Prompt (Windows) or Terminal (Mac/Linux)
2. Navigate to your project folder:
   - **Windows**: Type `cd Desktop\koos_model` and press Enter
   - **Mac/Linux**: Type `cd Desktop/koos_model` and press Enter
3. Download the model by typing:
   ```
   git clone https://github.com/jmorapclc/koos_mmmodel.git .
   ```
4. Press Enter and wait for download to complete
5. You should see many files being downloaded

### 3.3 Verify Download
1. Open your koos_model folder
2. You should see these folders and files:
   - `data/` folder
   - `model/` folder
   - `requirements.txt` file
   - `README.md` file
   - Other files

---

## 📦 Step 4: Install Required Software Packages

The AI model needs additional software packages to work.

### 4.1 Navigate to Project Directory
1. Open Command Prompt (Windows) or Terminal (Mac/Linux)
2. Navigate to your project folder:
   - **Windows**: `cd Desktop\koos_model`
   - **Mac/Linux**: `cd Desktop/koos_model`

### 4.2 Install Packages
1. Type this command and press Enter:
   ```
   pip install -r requirements.txt
   ```
2. Wait for installation to complete (this may take 10-15 minutes)
3. You'll see many packages being installed

### 4.3 Verify Installation
1. Type: `python model/check_device.py`
2. Press Enter
3. You should see information about your computer's capabilities

---

## 📊 Step 5: Prepare Your Data

Before running the model, you need to prepare your patient data.

### 5.1 Data Requirements

#### X-ray Images:
- **Format**: JPEG files (if you have DICOM files, convert them first)
- **Size**: Any size (the model will resize them automatically)
- **Naming**: Use 6-digit numbers (e.g., `111601.jpeg`, `111602.jpeg`)
- **Location**: Put all images in `data/img_repo/` folder

#### Patient Data (CSV file):
- **File name**: `HALS_Dataset_v1.csv`
- **Location**: `data/` folder
- **Required columns** (see table below):

| Column Name | Description | Example Values |
|-------------|-------------|----------------|
| `HALS_MRN` | Patient ID (6-digit number) | 111601, 111602 |
| `Sex` | Patient gender | 0 (Female), 1 (Male) |
| `Age` | Patient age | 65, 72, 58 |
| `BMI` | Body Mass Index | 28.5, 32.1, 25.8 |
| `Side` | Knee side | 0 (Left), 1 (Right) |
| `Type_of_TKA` | Surgery type | 0 or 1 |
| `Patellar_Replacement` | Patella surgery | 0 or 1 |
| `Preop_KOOS_PS` | Pre-surgery score | 45.2, 38.7, 52.1 |
| `1_Year_Postop_KOOS_PS` | Target score (what we predict) | 78.5, 82.3, 75.1 |

### 5.2 Data Preparation Steps

#### Step 5.2.1: Create Image Directory
1. Open your `koos_model` folder
2. Go to `data` folder
3. Create a new folder called `img_repo`
4. Put all your X-ray images in this folder

#### Step 5.2.2: Prepare CSV File
1. Open Excel or any spreadsheet program
2. Create a new file with the columns listed above
3. Fill in your patient data
4. Save as `HALS_Dataset_v1.csv` in the `data` folder

#### Step 5.2.3: Verify Data Structure
Your folder structure should look like this:
```
koos_model/
├── data/
│   ├── HALS_Dataset_v1.csv
│   └── img_repo/
│       ├── 111601.jpeg
│       ├── 111602.jpeg
│       └── ... (more images)
├── model/
└── ... (other files)
```

---

## 🚀 Step 6: Run the AI Model

Now you're ready to train and run the AI model!

### 6.1 Basic Training (Recommended for first time)

1. Open Command Prompt (Windows) or Terminal (Mac/Linux)
2. Navigate to your project folder:
   - **Windows**: `cd Desktop\koos_model`
   - **Mac/Linux**: `cd Desktop/koos_model`
3. Run the model with these commands:

```bash
# Test the setup first
python model/test_setup.py

# Run basic training (1 epoch for testing)
python model/train.py --epochs 1 --batch_size 4

# Run full training (50 epochs for real results)
python model/train.py --epochs 50 --batch_size 16
```

### 6.2 Understanding the Output

While the model is running, you'll see output like this:
```
Loading data...
Found 230 valid samples
Creating datasets...
Train: 161 samples, Val: 34 samples, Test: 35 samples
Starting training...
Epoch 1/50: 100%|████████| 41/41 [02:15<00:00, 3.30s/it]
Training completed!
```

### 6.3 What Happens During Training

1. **Data Loading**: The model loads your patient data and images
2. **Data Splitting**: Divides data into training (70%), validation (15%), and test (15%)
3. **Training**: The AI learns patterns from your data
4. **Validation**: Tests accuracy on unseen data
5. **Saving Results**: Creates output files with predictions

---

## 📈 Step 7: View Results

After training completes, you'll find results in the `model/outputs/` folder.

### 7.1 Finding Your Results

1. Open your `koos_model` folder
2. Go to `model/outputs/`
3. Look for a folder with today's date and time (e.g., `20250112_143022/`)
4. Open this folder

### 7.2 Understanding the Results

#### 7.2.1 Predictions File (`predictions.csv`)
- Shows predicted vs actual KOOS-PS scores for each patient
- Columns: Patient ID, Actual Score, Predicted Score, Error

#### 7.2.2 Evaluation Plots (`evaluation_plots/` folder)
- **`scatter.png`**: Shows how well predictions match actual scores
- **`residuals.png`**: Shows prediction errors
- **`bland_altman.png`**: Shows agreement between predicted and actual scores
- **`subgroup.png`**: Shows performance by age, sex, BMI groups

#### 7.2.3 Model Files (`model_artifacts/` folder)
- **`model.pth`**: The trained AI model (can be used for new predictions)
- **`config.json`**: Model settings
- **`model_summary.txt`**: Detailed model information

### 7.3 Interpreting Results

#### Good Results Look Like:
- **Scatter plot**: Points should be close to the diagonal line
- **Low error**: Most predictions within 10 points of actual scores
- **High correlation**: R² value above 0.7

#### If Results Look Poor:
- Check your data quality
- Ensure images are clear and properly labeled
- Verify CSV file has correct format
- Try training for more epochs

---

## 🔧 Step 8: Troubleshooting Common Issues

### Issue 1: "Command not found" or "Python not recognized"

**Solution:**
1. Restart your computer
2. Make sure Python is installed correctly
3. Try using `python3` instead of `python`

### Issue 2: "No module named 'torch'" or similar errors

**Solution:**
1. Make sure you're in the correct directory (`koos_model`)
2. Reinstall packages: `pip install -r requirements.txt`
3. If that doesn't work, try: `pip3 install -r requirements.txt`

### Issue 3: "File not found" errors

**Solution:**
1. Check that your CSV file is named exactly `HALS_Dataset_v1.csv`
2. Make sure images are in `data/img_repo/` folder
3. Verify file paths are correct

### Issue 4: "Out of memory" errors

**Solution:**
1. Reduce batch size: `python model/train.py --batch_size 8`
2. Close other programs to free up memory
3. Use fewer epochs for testing: `--epochs 10`

### Issue 5: Images not loading

**Solution:**
1. Check image file names match the HALS_MRN values in CSV
2. Ensure images are in JPEG format
3. Verify images are in the correct folder

---

## 📞 Step 9: Getting Help

### If You're Stuck:

#### 1. Check the Logs
- Look at `model/outputs/training.log` for error messages
- Check the console output for specific errors

#### 2. Verify Your Setup
- Run: `python model/check_device.py`
- Run: `python model/test_setup.py`

#### 3. Common Solutions
- Restart your computer
- Reinstall Python and packages
- Check file permissions
- Verify data format

#### 4. Contact Support
- Check the main README.md for more technical details
- Look at the model/README.md for advanced usage
- Contact the development team if needed

---

## 🎯 Step 10: Using the Model for New Predictions

Once your model is trained, you can use it to predict outcomes for new patients.

### 10.1 Adding New Patients

1. Add new X-ray images to `data/img_repo/`
2. Add new patient data to your CSV file
3. Run the model again to get predictions

### 10.2 Understanding Predictions

- **High scores (80-100)**: Patient likely to have good outcomes
- **Medium scores (60-79)**: Patient may need additional support
- **Low scores (0-59)**: Patient may need more intensive care

### 10.3 Clinical Use

- Use predictions to counsel patients about expected outcomes
- Identify patients who may need additional support
- Plan post-surgical care based on predicted scores
- Monitor actual outcomes vs predictions for model improvement

---

## ✅ Success Checklist

Before you start, make sure you have:

- [ ] Git installed and working
- [ ] Python installed and working
- [ ] AI model downloaded to your computer
- [ ] Required packages installed
- [ ] Patient data prepared (CSV + images)
- [ ] Data in correct folder structure
- [ ] At least 8GB RAM available
- [ ] 10GB free disk space

After setup, you should be able to:

- [ ] Run `python model/test_setup.py` without errors
- [ ] Run `python model/train.py --epochs 1` successfully
- [ ] See results in `model/outputs/` folder
- [ ] View prediction plots and files
- [ ] Use the model for new patient predictions

---

## 🎉 Congratulations!

You've successfully set up the KOOS-PS prediction model on your computer! 

The AI model is now ready to help you predict knee surgery outcomes for your patients. Remember to:

1. **Keep your data organized** in the correct folders
2. **Regularly update** the model with new patient data
3. **Monitor results** to ensure accuracy
4. **Use predictions responsibly** as a tool to support clinical decision-making

The model will continue to improve as you add more patient data and retrain it regularly.

---

*This guide was designed specifically for medical staff with no programming experience. If you encounter any issues not covered here, please refer to the main README.md or contact the development team for assistance.*
