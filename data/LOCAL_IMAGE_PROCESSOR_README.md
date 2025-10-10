# Local Image Processor for HALS Dataset

This script processes local image files (JPEG or DICOM) and updates the CSV file with new HALS_MRN identifiers. It can convert DICOM files to JPEG format if needed.

## Features

- **Image Format Support**: Handles both JPEG and DICOM files
- **DICOM Conversion**: Automatically converts DICOM files to JPEG format
- **Filename Validation**: Validates that filenames match the expected 6-digit format
- **CSV Integration**: Updates the HALS dataset CSV with new identifiers
- **Interactive Mode**: Prompts user for directory path
- **Comprehensive Logging**: Detailed logging of all operations

## File Naming Convention

The script expects image files to follow this naming pattern:
- **Format**: `XXXXXX[optional_a].extension`
- **Examples**: 
  - `100101.jpeg` (6 digits + extension)
  - `100102a.jpg` (6 digits + 'a' suffix + extension)
  - `111501.dcm` (DICOM format)
  - `111502a.dicom` (DICOM with 'a' suffix)

## Supported File Formats

### Input Formats
- `.jpg` / `.jpeg` - JPEG images
- `.dcm` / `.dicom` - DICOM medical images

### Output Format
- All images are converted to `.jpeg` format for consistency

## Usage

### Basic Usage

```bash
cd /path/to/koos_model
python data/local_image_processor.py
```

The script will prompt you to enter the path to the directory containing your image files.

### Programmatic Usage

```python
from data.local_image_processor import LocalImageProcessor
from data.config import *

# Create processor instance
processor = LocalImageProcessor(
    local_image_path="path/to/your/images",
    csv_file_path="data/HALS_Dataset_v1.csv",
    output_dir="data/processed_images"
)

# Process all images
processor.process_all()
```

## Configuration

The script uses settings from `data/config.py`:

```python
# Local directory containing already renamed image files
LOCAL_DCM_FILE_PATH = "./img_repo"

# Input CSV file path
CSV_FILE = "data/HALS_Dataset_v1.csv"

# Output directory for processed images
OUTPUT_DIR = "img_repo"

# Output CSV filename
OUTPUT_CSV_FILE = "updated_hals_dataset.csv"
```

## Dependencies

The script requires the following Python packages:

```bash
pip install pydicom pillow
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Output

### Processed Images
- All images are saved to the output directory as `.jpeg` files
- Original filenames are preserved (with extension changed to `.jpeg`)

### Updated CSV
- New rows are added to the CSV for each processed image
- HALS_MRN column contains the filename (without extension)
- Other columns are left empty for manual completion

### Log Files
- `local_image_processor.log` - Detailed processing log

## Example Workflow

1. **Prepare Images**: Place your image files in a directory with proper naming
2. **Run Script**: Execute the processor script
3. **Enter Path**: Provide the path to your image directory when prompted
4. **Review Output**: Check the processed images and updated CSV
5. **Complete Data**: Fill in missing data in the CSV file

## Error Handling

The script handles various error conditions:

- **Invalid Filenames**: Warns about files that don't match the expected format
- **Missing Directories**: Checks if input directories exist
- **DICOM Conversion Errors**: Logs errors during DICOM to JPEG conversion
- **File I/O Errors**: Handles file reading/writing errors gracefully

## Testing

Run the test suite to verify functionality:

```bash
python data/test_local_processor.py
```

Run the demo to see the processor in action:

```bash
python data/demo_local_processor.py
```

## Logging

The script provides comprehensive logging:

- **INFO**: Normal processing steps
- **WARNING**: Non-critical issues (e.g., invalid filenames)
- **ERROR**: Critical errors that prevent processing

Logs are written to both console and `local_image_processor.log` file.

## Integration with Existing Workflow

This script complements the existing `dcm_file_processor.py`:

- **Original Processor**: Processes DICOM files from complex tree structure
- **Local Processor**: Processes already-renamed images from local directory

Both scripts update the same CSV format and can be used together in a complete workflow.
