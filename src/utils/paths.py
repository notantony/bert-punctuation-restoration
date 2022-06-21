from pathlib import Path


DATA_DIR = Path('./data')
TRAIN_DATA = DATA_DIR / 'train.tags.en-fr.en'
TEST_XML = DATA_DIR / 'IWSLT12.TALK.tst2010.en-fr.en.xml'
DEV_XML = DATA_DIR / 'IWSLT12.TALK.dev2010.en-fr.en.xml'

MODELS_DIR = Path('./models')

RESULTS_DIR = Path('./results')
