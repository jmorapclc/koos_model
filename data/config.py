"""
Configuration file for DCM File Processor

Modify these settings as needed for your environment.
"""

# Source directory containing the DCM files
SOURCE_PATH = "/Volumes/MGB-CRE3-IRB2024P000937"

# Input CSV file path (relative to script location)
CSV_FILE = "data/HALS_Dataset_v1.csv"

# Output directory for renamed DCM files
OUTPUT_DIR = "img_repo"

# Starting counter for main directories (default: 1116)
STARTING_COUNTER = 1116

# Output CSV filename
OUTPUT_CSV_FILE = "updated_hals_dataset.csv"

# Log file name
LOG_FILE = "dcm_processor.log"

# CSV column names (adjust if your CSV has different column names)
CSV_COLUMNS = {
    'EMPI': 'EMPI',
    'DATE_OF_SURGERY': 'Date_of_Surgery',
    'ACCESSION_NUMBER': 'Accession_Number',
    'HALS_MRN': 'HALS_MRN',
    'SEX': 'Sex',
    'AGE': 'Age',
    'BMI': 'BMI',
    'SIDE': 'Side',
    'TYPE_OF_TKA': 'Type_of_TKA',
    'PATELLAR_REPLACEMENT': 'Patellar_Replacement',
    'PREOP_KOOS_PS': 'Preoperative_KOOS-PS',
    'POSTOP_KOOS_PS': '1-Year_Postop_KOOS-PS'
}
