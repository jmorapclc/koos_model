# DCM File Processor for HALS Dataset

This Python script processes DICOM files from a tree structure and renames them according to specific rules, then updates the corresponding CSV file with new HALS_MRN values.

## Overview

The script performs the following tasks:

1. **Scans the source directory** (`/Volumes/MGB-CRE3-IRB2024P000937`) for DCM files
2. **Renames files** using a 6-digit naming pattern:
   - First 4 digits: Correlative number starting from 1116
   - Last 2 digits: Correlative number for each file (01, 02, etc.)
   - Special handling for directories with multiple subdirectories (adds 'a' suffix)
3. **Copies renamed files** to a new `img_repo` directory
4. **Updates the CSV file** with new HALS_MRN values and duplicates rows as needed

## File Naming Rules

### Single Subdirectory
- Files are renamed sequentially: `111601.dcm`, `111602.dcm`, etc.

### Multiple Subdirectories
- **Lowest alphanumeric subdirectory**: Files get 'a' suffix: `111901a.dcm`, `111902a.dcm`
- **Highest alphanumeric subdirectory**: Files get normal naming: `111901.dcm`, `111902.dcm`

## Requirements

- Python 3.6 or higher
- Access to the source directory (`/Volumes/MGB-CRE3-IRB2024P000937`)
- CSV file with the correct structure (see `data/HALS_Dataset_v1.csv`)

## Usage

1. **Ensure you're in the project directory**:
   ```bash
   cd /path/to/koos_model
   ```

2. **Make sure the CSV file exists**:
   ```bash
   ls data/HALS_Dataset_v1.csv
   ```

3. **Run the script**:
   ```bash
   python dcm_file_processor.py
   ```

## Output

The script will create:

- **`img_repo/`** directory containing all renamed DCM files
- **`updated_hals_dataset.csv`** with updated HALS_MRN values
- **`dcm_processor.log`** with detailed processing logs

## Configuration

You can modify the following variables in the `main()` function:

```python
SOURCE_PATH = "/Volumes/MGB-CRE3-IRB2024P000937"  # Source directory
CSV_FILE = "data/HALS_Dataset_v1.csv"              # Input CSV file
OUTPUT_DIR = "img_repo"                            # Output directory for DCM files
```

## CSV Structure

The script expects a CSV file with the following columns:
- `EMPI`: Patient identifier
- `Date_of_Surgery`: Surgery date
- `Accession_Number`: Corresponds to main directory names
- `HALS_MRN`: Will be updated with new file names
- Other columns: Preserved as-is

## Error Handling

The script includes comprehensive error handling and logging:
- Logs all operations to `dcm_processor.log`
- Continues processing even if individual files fail
- Reports summary statistics at the end

## Example

For a directory structure like:
```
├── 0047140
│ └── E15047136
│     └── DX
│         ├── 1___AP_KNEES_BI_T_STANDING
│         │ └── 1352438653.dcm
│         └── 2__PA_FLEXION_B_T_STANDING
│             └── 1352438675.dcm
```

The files will be renamed to:
```
img_repo/
├── 111601.dcm
└── 111602.dcm
```

And the CSV will be updated with corresponding HALS_MRN values.

## Troubleshooting

1. **Permission errors**: Ensure you have read access to the source directory and write access to the current directory
2. **CSV not found**: Make sure you're running from the correct directory and the CSV file exists
3. **Source directory not found**: Verify the path `/Volumes/MGB-CRE3-IRB2024P000937` is accessible
4. **Check logs**: Review `dcm_processor.log` for detailed error information
