import gdown

from utils.paths import DATA_DIR, DEV_XML, TRAIN_DATA, TEST_XML


_GDOWN_FILES = [
    ('1-3Qe5-KQ-gUCBETv3qfHYPL1ZC0AuHFQ', TRAIN_DATA),
    ('1-3tyNoQs1DND17thZ2lfUfhTT39XBQ5_', DEV_XML),
    ('1-4L3t-dcereVT_ePAy1Sp_Ni5w9kq8vS', TEST_XML),
]


def download_gdown():
    for file_id, target_path in _GDOWN_FILES:
        gdown.download(id=file_id, output=str(target_path))
        

if __name__ == '__main__':
    download_gdown()
