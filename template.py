import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    '.github/workflows/.gitkeep',
    f'src/__init__.py',
    f'src/components/__init__.py',
    f'src/components/data_ingestion.py',
    f'src/components/data_transformation.py',
    f'src/components/model_train.py',
    f'src/utils/__init__.py',
    f'src/utils/logger.py',
    f'src/utils/exception.py',
    f'src/utils/utils.py',
    f'src/pipeline/__init__.py',
    f'src/pipeline/predict_pipeline.py',
    'requirements.txt',
    'setup.py'
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory: {filedir} for the file : {filename}')
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f'Creating empty file : {filepath}')
    else:
        logging.info(f'{filename} already exists!')