# functions to process poses

from input_libs import *

def quat_from_pose(trans):

    w = trans['transform']['rotation']['w']
    x = trans['transform']['rotation']['x']
    y = trans['transform']['rotation']['y']
    z = trans['transform']['rotation']['z']

    return [w,x,y,z]


def read_calib_yaml(calib_folder, file_name):
    with open(os.path.join(calib_folder, file_name), 'r') as stream:
        try:
            cur_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cur_yaml


def read_txt(root_folder, file_name):
    with open(os.path.join(root_folder, file_name)) as f:
        cur_file = f.readlines()
    return cur_file

def read_numpy(root_folder, file_name):
    with open(os.path.join(root_folder, file_name), 'rb') as f:
        cur_file = np.load(f)
    return cur_file

def write_numpy(root_folder, file_name, current_file):
    with open(os.path.join(root_folder, file_name), 'wb') as f:
        np.save(f, current_file)

def read_csv(root_folder, file_name):
    with open(os.path.join(root_folder, file_name), newline='') as f:
        reader = csv.reader(f)
        cur_file = list(reader)
    return cur_file