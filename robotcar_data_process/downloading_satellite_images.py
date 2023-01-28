# given the center geodestic coordinate of each satellite patch
# retrieve satellite patchs from the google map server

# Todo: using the viewing direction of forward camera to move the center of satellite patch
# without this, the satellite patch only share a small common FoV as the ground-view query image

# NOTE:
# You need to provide a key
keys = ['AIzaSyDC-pLZ5iR9TuSupwN6bHRvqhB0020AdD0']

import numpy as np
import requests
from io import BytesIO
import os
import time
from PIL import Image as PILI

root_folder = "/data/robotcar"

log_id = "2015-06-26-08-09-43"

url_head = 'https://maps.googleapis.com/maps/api/staticmap?'
zoom = 18
sat_size = [640, 640]
maptype = 'satellite'
scale = 2


def read_numpy(root_folder, file_name):
    with open(os.path.join(root_folder, file_name), 'rb') as f:
        cur_file = np.load(f)
    return cur_file

def write_numpy(root_folder, file_name, current_file):
    with open(os.path.join(root_folder, file_name), 'wb') as f:
        np.save(f, current_file)

log_folder = os.path.join(root_folder, log_id)

Geodetic = read_numpy(log_folder, 'satellite_gps_center.npy')

nb_keys = len(keys)

nb_satellites = Geodetic.shape[0]

satellite_folder = os.path.join(root_folder, log_id, "Satellite_Images")

if not os.path.exists(satellite_folder):
        os.makedirs(satellite_folder)


for i in range(nb_satellites):

    lat_a, long_a, height_a = Geodetic[i, 0], Geodetic[i, 1], Geodetic[i, 2]

    image_name = satellite_folder + "/satellite_" + str(i) + "_lat_" + str(lat_a) + "_long_" + str(
        long_a) + "_zoom_" + str(
        zoom) + "_size_" + str(sat_size[0]) + "x" + str(sat_size[0]) + "_scale_" + str(scale) + ".png"

    if os.path.exists(image_name):
        continue

    time.sleep(1)

    saturl = url_head + 'center=' + str(lat_a) + ',' + str(long_a) + '&zoom=' + str(
        zoom) + '&size=' + str(
        sat_size[0]) + 'x' + str(sat_size[1]) + '&maptype=' + maptype + '&scale=' + str(
        scale) + '&format=png32' + '&key=' + \
             keys[0]
    #f = requests.get(saturl, stream=True)

    try:
        f = requests.get(saturl, stream=True)
        f.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    bytesio = BytesIO(f.content)
    cur_image = PILI.open(bytesio)

    cur_image.save(image_name)
