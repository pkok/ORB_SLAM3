import os.path

from evaluate_ate_scale import normalize_data

MODES = ["mono", "stereo", "monoi", "stereoi"]

DATASETS = ["MH01", "MH02", "MH03", "MH04", "MH05",
            "V101", "V102", "V103",
            "V201", "V202", "V203"]

DATA_DIR = os.path.abspath("../data/")
DATA_FILE_PATTERN = "errors-{dataset}_{mode}.csv"

def error_files():
    pattern = DATA_DIR + DATA_FILE_PATTERN
    return [pattern.format(dataset=d, mode=m) for m in MODES
                                              for d in DATASETS]
