import numpy as np
import math

semi_a = 6378137.0
semi_b = 6356752.31424518

ratio = semi_b / semi_a

f = (semi_a - semi_b) / semi_a
# eccentricity = np.sqrt(1.0 - ratio*ratio)
eccentricity_square = 2.0 * f - f * f
eccentricity = np.sqrt(eccentricity_square)

# this is the reference gps for all ford dataset
gps_ref_lat = 49.03315203474046*np.pi/180.0
gps_ref_long = 8.337345840397377*np.pi/180.0
gps_ref_height = 0.0

ru_m = semi_a * (1.0-eccentricity_square)/np.power(1.0-eccentricity_square*np.sin(gps_ref_lat)*np.sin(gps_ref_lat), 1.5)
ru_t = semi_a / np.sqrt(1.0-eccentricity_square*np.sin(gps_ref_lat)*np.sin(gps_ref_lat))


# Converts WGS-84 Geodetic point (lat, lon, h) to the
#     // Earth-Centered Earth-Fixed (ECEF) coordinates (x, y, z).
def GeodeticToEcef( lat,  lon,  h):

    sin_lambda = np.sin(lat)
    cos_lambda = np.cos(lat)
    cos_phi = np.cos(lon)
    sin_phi = np.sin(lon)
    N = semi_a / np.sqrt(1.0 - eccentricity_square * sin_lambda * sin_lambda)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1.0 - eccentricity_square) * N) * sin_lambda
    return x,y,z

# Converts the Earth-Centered Earth-Fixed (ECEF) coordinates (x, y, z) to
# East-North-Up coordinates in a Local Tangent Plane that is centered at the
# (WGS-84) Geodetic point (lat0, lon0, h0).
def EcefToEnu( x,  y,  z, lat0,  lon0,  h0):

    sin_lambda = np.sin(lat0)
    cos_lambda = np.cos(lat0)
    cos_phi = np.cos(lon0)
    sin_phi = np.sin(lon0)
    N = semi_a / np.sqrt(1.0 - eccentricity_square * sin_lambda * sin_lambda)
    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1.0 - eccentricity_square) * N) * sin_lambda
    xd = x - x0
    yd = y - y0
    zd = z - z0
    # This is the matrix multiplication
    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd
    return xEast,yNorth,zUp


# Inverse of EcefToEnu. Converts East-North-Up coordinates (xEast, yNorth, zUp) in a
# Local Tangent Plane that is centered at the (WGS-84) Geodetic point (lat0, lon0, h0)
# to the Earth-Centered Earth-Fixed (ECEF) coordinates (x, y, z).
def EnuToEcef( xEast,  yNorth, zUp, lat0,  lon0,  h0):

# Convert to radians in notation consistent with the paper:
    sin_lambda = np.sin(lat0)
    cos_lambda = np.cos(lat0)
    cos_phi = np.cos(lon0)
    sin_phi = np.sin(lon0)
    N = semi_a / np.sqrt(1.0 - eccentricity_square * sin_lambda * sin_lambda)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - eccentricity_square) * N) * sin_lambda

    xd = -sin_phi * xEast - cos_phi * sin_lambda * yNorth + cos_lambda * cos_phi * zUp
    yd = cos_phi * xEast - sin_lambda * sin_phi * yNorth + cos_lambda * sin_phi * zUp
    zd = cos_lambda * yNorth + sin_lambda * zUp

    x = xd + x0
    y = yd + y0
    z = zd + z0
    return x,y,z



# Converts the Earth-Centered Earth-Fixed (ECEF) coordinates (x, y, z) to
# (WGS-84) Geodetic point (lat, lon, h).
def EcefToGeodetic( x,  y,  z):

    eps = eccentricity_square / (1.0 - eccentricity_square)
    p = math.sqrt(x * x + y * y)
    q = math.atan2((z * semi_a), (p * semi_b))
    sin_q = np.sin(q)
    cos_q = np.cos(q)
    sin_q_3 = sin_q * sin_q * sin_q
    cos_q_3 = cos_q * cos_q * cos_q
    phi = math.atan2((z + eps * semi_b * sin_q_3), (p - eccentricity_square * semi_a * cos_q_3))
    lon = math.atan2(y, x) * 180.0 / np.pi
    v = semi_a / math.sqrt(1.0 - eccentricity_square * np.sin(phi) * np.sin(phi))
    h = (p / np.cos(phi)) - v

    lat = phi*180.0/np.pi

    return lat, lon, h



def angular_distance_to_xy_distance( lat1, long1):
    dx = ru_t * np.cos(gps_ref_lat) * (long1 - gps_ref_long)
    dy = ru_m * (lat1 - gps_ref_lat)
    return dx, dy


def angular_distance_to_xy_distance_v2(lat_ref, long_ref, lat1, long1):

    gps_ref_lat = lat_ref * np.pi / 180.0
    gps_ref_long = long_ref * np.pi / 180.0

    ru_m_local = semi_a * (1.0 - eccentricity_square) / np.power(
        1.0 - eccentricity_square * np.sin(gps_ref_lat) * np.sin(gps_ref_lat), 1.5)

    ru_t_local = semi_a / np.sqrt(1.0 - eccentricity_square * np.sin(gps_ref_lat) * np.sin(gps_ref_lat))

    dx = ru_t_local * np.cos(gps_ref_lat) * (long1* np.pi / 180.0 - gps_ref_long)

    dy = ru_m_local * (lat1* np.pi / 180.0 - gps_ref_lat)

    return dx, dy