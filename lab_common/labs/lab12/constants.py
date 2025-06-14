import os

from lab_common.common import ROOT_PROJECT_FOLDER_PATH

LAB_12_DATASET = os.path.join(ROOT_PROJECT_FOLDER_PATH, "lab_datasets", "lab12")

LAB_12_TEST_BENIGN_FOLDER = os.path.join(LAB_12_DATASET, "small_test_benign")

TEST_BCC_FILE_PATH = os.path.join(LAB_12_DATASET,
                                  "libgpuwork.so.bcc")


LAB_12_CACHE_FOLDER = os.path.join(ROOT_PROJECT_FOLDER_PATH,
                                   "labs",
                                   "lab11",
                                   "cache")