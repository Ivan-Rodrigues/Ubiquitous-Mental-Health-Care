import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from sklearn.cluster import DBSCAN
import sklearn.cluster as skc
import warnings
warnings.filterwarnings("ignore")
count_dis = 0

def discovery_clusters(gps_data):
    # encontra os clusters
    coords = gps_data.as_matrix(columns=['latitude', 'longitude'])
    kms_per_radian = 6371.0088
    epsilon = 0.1 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    return clusters



#Fórmula de Haversine
def calc_distance(lat1,lon1,lat2,lon2):
    R = 6373.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return round(distance,4)

# viagem máxima permitida = 30 km
def calc_total_distance(df_location, limit_distance):
    distance_total = 0
    count = 0
    for i in range(len(df_location)-1):

        lat1 = df_location.iloc[i]['latitude']
        long1 = df_location.iloc[i]['longitude']

        lat2 = df_location.iloc[i + 1]['latitude']
        long2 = df_location.iloc[i + 1]['longitude']

        distance = calc_distance(lat1, long1, lat2, long2)

        if distance < limit_distance:
            distance_total += distance
            count += 1
        i += 1
    return distance_total, count

#variância
def calc_loc_var(gps_data):
    gps_daily = pd.DataFrame()
    gps_daily['latitude_var'] = gps_data[gps_data['travelstate'] == 'stationary'].groupby('date')['latitude'].var()
    gps_daily['longitude_var'] = gps_data[gps_data['travelstate'] == 'stationary'].groupby('date')['longitude'].var()
    #divide by zero encountered in log
    gps_daily['loc_var'] = np.log(gps_daily['latitude_var'] + gps_daily['longitude_var']+0.1)

    mean_var = gps_daily['loc_var'].mean()

    return mean_var

def calc_max_distance(df_location, limit_distance):
    distance_max = 0
    for i in range(len(df_location) - 1):

        lat1 = df_location.iloc[i]['latitude']
        long1 = df_location.iloc[i]['longitude']

        lat2 = df_location.iloc[i + 1]['latitude']
        long2 = df_location.iloc[i + 1]['longitude']

        distance = calc_distance(lat1, long1, lat2, long2)

        if distance < limit_distance:
            if distance > distance_max:
                distance_max = distance


        i += 1
    return distance_max

def calc_dis_mean(df_location,limit_distance, mean):
    distance_total = 0
    for i in range(len(df_location) - 1):

        lat1 = df_location.iloc[i]['latitude']
        long1 = df_location.iloc[i]['longitude']

        lat2 = df_location.iloc[i + 1]['latitude']
        long2 = df_location.iloc[i + 1]['longitude']

        distance = calc_distance(lat1, long1, lat2, long2)

        if distance < limit_distance:
            distance = (distance - mean) ** 2
            distance_total += distance
        i += 1
    return distance_total


def calc_std_loc(gps_data):
    distance,total_dis = calc_total_distance(gps_data,25)
    mean = distance / total_dis
    dis_mean = calc_dis_mean(gps_data,25,mean)

    std = sqrt(dis_mean/total_dis)

    return std

def get_cluster_home(clusters, gps_data_home):
    idx_home = None
    distance_Atual = 999999
    coords_home = gps_data_home.as_matrix(columns=['latitude', 'longitude'])
    kms_per_radian = 6371.0088
    epsilon = 0.5 / kms_per_radian
    db_home = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine').fit(np.radians(coords_home))
    cluster_labels_home = db_home.labels_
    num_clusters = len(set(cluster_labels_home))
    cluster_home = pd.Series([coords_home[cluster_labels_home == n] for n in range(num_clusters)])

    lat_home = cluster_home[0][0][0]
    lon_home = cluster_home[0][0][1]
    #print("home: {},{}".format(lat_home,lon_home))

    for i in range(len(clusters)-1):
        lat_cluster = clusters[i][0][0]
        lon_cluster = clusters[i][0][1]
        distance = calc_distance(lat_home,lon_home, lat_cluster, lon_cluster)
        if distance < distance_Atual:
            distance_Atual = distance
            idx_home = i
    #print("home encontrado: {},{}".format(lat_home,lon_home))
    return idx_home

def calc_total_obs_clusters(clusters):
    total = 0
    for i in range(len(clusters) - 1):
        size = len(clusters[i])
        total += size
    return total

def calc_home_stay(cluster_home, cluster):
    total_home = len(cluster_home)
    total_all = calc_total_obs_clusters(cluster)
    home_stay = total_home/total_all
    #print(home_stay)
    return home_stay


def calc_max_distance_home(cluster_home, df_location, limit_distance):
    lat_home = cluster_home[0][0]
    lon_home = cluster_home[0][1]
    max_distance = 0
    for i in range(len(df_location) - 1):
        lat1 = df_location.iloc[i]['latitude']
        long1 = df_location.iloc[i]['longitude']

        distance = calc_distance(lat_home,lon_home, lat1, long1)
        if distance < limit_distance:
            if distance > max_distance:
                max_distance = distance
    return max_distance

def calc_entropy_loc(clusters):
    entropy = 0
    total_all = calc_total_obs_clusters(clusters)
    for i in range(len(clusters) - 1):
        total_cluster_i = len(clusters[i])
        pi = total_cluster_i/total_all
        entropy += pi * np.log(pi)*-1
    return entropy


def calc_normalized_entropy_loc(clusters):
    entropy = calc_entropy_loc(clusters)
    number_clusters = len(clusters)
    normalized_entropy = entropy/np.log(number_clusters)
    return normalized_entropy

