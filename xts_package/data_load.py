import enum
import os.path


class xts_dataset_source(enum.Enum):

    A = '/drives/drive2/Patient Safety Graph/out/A'
    B = '/drives/drive2/Patient Safety Graph/out/B'


def xts_txt_load(dir):

    with open(dir) as f:

        return [line.strip() for line in f.readlines()]

def xts_dir_join(parent_dir, child_dir):

    full_dir = os.path.join(parent_dir, child_dir)

    return full_dir