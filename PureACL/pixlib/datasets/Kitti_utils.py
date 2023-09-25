# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 16:14:01 2020

@author: loocy
"""
import numpy as np
import torch

CameraGPS_shift = [1.08, 0.26]
Camera_height = 1.65 #meter
Camera_distance = 0.54 #meter

SatMap_original_edge = 1280 # 0.05 m per pixel
SatMap_process_edge = 640 # 0.1 m per pixel
Default_lat = 49.03315203474046

yaw_min = -40
yaw_fov = 82

def get_grd_fov():
    return yaw_fov, yaw_min

def get_camera_height():
    return Camera_height

def get_camera_distance():
    return Camera_distance

def get_original_satmap_edge():
    return SatMap_original_edge

def get_process_satmap_edge():
    return SatMap_process_edge

# x: east shift in meter, y: south shift in meter
# return lat and lon after shift
# Curvature formulas from https://en.wikipedia.org/wiki/Earth_radius#Meridional
def meter2latlon(lat, lon, x, y):
    r = 6378137 # equatorial radius
    flatten = 1/298257 # flattening
    E2 = flatten * (2- flatten)
    m = r * np.pi/180  
    coslat = np.cos(lat * np.pi/180)
    w2 = 1/(1-E2 *(1-coslat*coslat))
    w = np.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1-E2)
    lon += x / kx 
    lat -= y / ky
    
    return lat, lon   

def gps2meters(lat_s, lon_s, lat_d, lon_d ):
    r = 6378137 # equatorial radius
    flatten = 1/298257 # flattening
    E2 = flatten * (2- flatten)
    m = r * np.pi/180  
    lat = (lat_s+lat_d)/2
    coslat = np.cos(lat * np.pi/180)
    w2 = 1/(1-E2 *(1-coslat*coslat))
    w = np.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1-E2)
    x = (lon_d-lon_s)*kx
    y = (lat_s-lat_d)*ky # y: from top to bottom
    
    return [x,y]

def gps2meters_torch(lat_s, lon_s, lat_d, lon_d ):
    # inputs: torch array: [n]
    r = 6378137 # equatorial radius
    flatten = 1/298257 # flattening
    E2 = flatten * (2- flatten)
    m = r * np.pi/180  
    lat = lat_d[0]
    coslat = np.cos(lat * np.pi/180)
    w2 = 1/(1-E2 *(1-coslat*coslat))
    w = np.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1-E2)
    
    x = (lon_d-lon_s)*kx
    y = (lat_s-lat_d)*ky # y: from top to bottom
    
    return x,y


def gps2shiftmeters(latlon ):
    # torch array: [B,S,2]

    r = 6378137 # equatoristereoal radius
    flatten = 1/298257 # flattening
    E2 = flatten * (2- flatten)
    m = r * np.pi/180  
    lat = latlon[0,0,0]
    coslat = torch.cos(lat * np.pi/180)
    w2 = 1/(1-E2 *(1-coslat*coslat))
    w = torch.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1-E2)

    shift_x = (latlon[:,:1,1]-latlon[:,:,1])*kx #B,S east
    shift_y = (latlon[:,:,0]-latlon[:,:1,0])*ky #B,S south
    shift = torch.cat([shift_x.unsqueeze(-1),shift_y.unsqueeze(-1)],dim=-1) #[B,S,2] #shift from 0
    
    # shift from privious
    S = latlon.size()[1]
    shift = shift[:,1:,:]-shift[:,:(S-1),:]
    
    return shift


def gps2distance(lat_s, lon_s, lat_d, lon_d ):
    x,y = gps2meters_torch(lat_s, lon_s, lat_d, lon_d )
    dis = torch.sqrt(torch.pow(x, 2)+torch.pow(y,2))
    return dis

def get_meter_per_pixel( zoom, lat=Default_lat, scale = SatMap_process_edge/SatMap_original_edge):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)	
    meter_per_pixel /= 2 # because use scale 2 to get satmap 
    meter_per_pixel /= scale
    return meter_per_pixel

def gps2shiftscale(latlon):
    # torch array: [B,S,2]
    
    shift = gps2shiftmeters(latlon)
    
    # turn meter to -1~1
    meter_per_pixel = get_meter_per_pixel(scale=1)
    win_range = meter_per_pixel*SatMap_original_edge
    shift /= win_range//2
    
    return shift

def get_camera_max_meter_shift():
    return np.linalg.norm(CameraGPS_shift)

def get_camera_gps_shift(heading):
    shift_x = CameraGPS_shift[0] * np.cos(heading%(2*np.pi)) + CameraGPS_shift[1] * np.sin(heading%(2*np.pi))
    shift_y = CameraGPS_shift[1] * np.cos(heading%(2*np.pi)) - CameraGPS_shift[0] * np.sin(heading%(2*np.pi))
    return shift_x, shift_y

def get_shiftuv_from_latlon(heading, latlon):
    shift_x = latlon[0] * np.cos(heading%(2*np.pi)) + latlon[1] * np.sin(heading%(2*np.pi))
    shift_y = latlon[1] * np.cos(heading%(2*np.pi)) - latlon[0] * np.sin(heading%(2*np.pi))
    return shift_x, shift_y

def get_height_config():
    start = 0 #-15 -7 0
    end = 0 
    count = 1 #16 8 1
    return start, end, count

    
