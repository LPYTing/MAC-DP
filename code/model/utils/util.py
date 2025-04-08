import os, traceback
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from datetime import datetime, timedelta
from typing import Set, List, Literal
from collections import defaultdict
from utils.classes import BidirectionalDict, Intersection, Mcs
from collections import defaultdict
from copy import deepcopy
from loguru import logger
from config import Config
from utils.hex_environment import create_hexagons_polygon, located_grid


'''Mapping street in city to [lat,lng, cpid, charger]'''
fcs_trans = {
    '1520 Walnut Dr, Palo Alto, California, United States': [37.445492,-122.138931,0,2],
    '250 Hamilton Ave, Palo Alto, California, United States': [37.444572,-122.160309,1,2],
    '528 High St., Palo Alto, California, United States': [37.443127,-122.163033,2,1],
    '400 Cambridge Ave, Palo Alto, California, United States': [37.427349,-122.145676,3,2],
    '500-532 Webster St, Palo Alto, California, United States': [37.449348,-122.157768,4,2],
    '520 Webster St, Palo Alto, California, United States': [37.449348,-122.157768,4,2],
    '528 High St, Palo Alto, California, United States': [37.443676,-122.16288,5,2],
    '528 high st, Palo Alto, California, United States': [37.443146,-122.163017,6,1],
    '1514 Walnut Dr, Palo Alto, California, United States': [37.445492,-122.139,7,2],
    '532 Webster St, Palo Alto, California, United States': [37.448284,-122.158272,8,1],
    '3700 Middlefield Rd, Palo Alto, California, United States': [37.422104,-122.11351,9,1],
    'Bryant St, Palo Alto, California, United States': [37.446373,-122.162331,10,2],
    '445 Bryant St, Palo Alto, California, United States': [37.446583,-122.162109,11,1],
    '275 Cambridge Ave, Palo Alto, California, United States': [37.428318,-122.144188,12,2],
    '1213 Newell Rd, Palo Alto, California, United States': [37.445496,-122.138924,13,2],
    '533 Cowper St, Palo Alto, California, United States': [37.448284,-122.158218,14,2],
    '520 Cowper St, Palo Alto, California, United States': [37.448284,-122.158218,14,2],
    '350 Sherman Ave, Palo Alto, California, United States': [37.42667,-122.143257,15,1],
    '358 Sherman Ave, Palo Alto, California, United States': [37.426655,-122.143417,16,2],
    '475 Cambridge Ave, Palo Alto, California, United States': [37.426155,-122.146065,17,1]
}
fcs_data = list(set(tuple(sublist) for sublist in fcs_trans.values()))  # fcs_data: [lat, lng, cp id, connector]
fcs_loc = [s[:2] for s in fcs_data]


'''Define city boundary'''
# Define the latitude/longitude coordinate pairs of the polygon
COORDINATE = [
    (37.481197, -122.203006),
    (37.485756, -122.142098),
    (37.459112, -122.107508),
    (37.39102, -122.073788),
    (37.390241, -122.202958),
]
# Create a Polygon object using the coordinates
POLYGON = Polygon(COORDINATE)


# Define a function to check if a location is inside the polygon
def InCity(city: str, lat, lng):
    try:
        if city == "Palo_Alto":
            polygon = POLYGON
            result = polygon.contains(Point(lat, lng))
            return result
        else:
            raise ValueError(f"City {city} not found in POLYGON")
    except Exception as e:
        print(f"ERROR: {e}")

def city_speed(speed: int=40, unit: str="sec"):
    speed_hr = speed
    speed_min = speed_hr/60
    speed_sec = speed_min/60
    dic = {"hr":speed_hr, "min":speed_min, "sec":speed_sec}
    return dic[unit]

def traveling_time(distance, speed=40, unit="sec"):
    return timedelta(seconds = distance / city_speed(speed, unit))

def longest_path():
    from geopy.distance import distance
    # Get the envelope of the polygon and its exterior coordinates
    envelope = POLYGON.envelope
    exterior_coords = list(envelope.exterior.coords)

    # Calculate the longest distance between any two points in the exterior coordinates
    max_distance = 0
    for i in range(len(exterior_coords)):
        for j in range(i+1, len(exterior_coords)):
            dist = distance(exterior_coords[i], exterior_coords[j]).km
            if dist > max_distance:
                max_distance = dist
    return max_distance

def get_PaloAlto_intersection(by: List[Literal["txt","osmnx"]] = defaultdict(lambda: "txt")) -> List[float]:
    if by == "txt":
        arr = np.loadtxt("intersection-Palo_Alto.txt")
        intersections_coordinates = arr.tolist()

    elif by == "osmnx":
        import osmnx as ox
        # Define the city name
        city_name = ["Palo Alto, California, USA", "East Palo Alto, California, USA", "Menlo Park, California, USA", "Stanford, California, USA", "Los Altos, California, USA", "Mountain View, California, USA", "Atherton, California, USA"]

        # Use OSMnx to get the street network of the city
        G_list = []
        for city in city_name:
            G_list.append(ox.graph_from_place(city, network_type="drive"))

        # Find all the intersections by the cross point
        intersections_coordinates = []
        for G in G_list:
            for node, data in G.nodes(data=True):
                if InCity("Palo_Alto", G.nodes[node]["y"], G.nodes[node]["x"]):
                    intersections_coordinates.append([G.nodes[node]["y"], G.nodes[node]["x"]])
        arr = np.array(intersections_coordinates)
        np.savetxt("intersection-Palo_Alto.txt",arr)

    return intersections_coordinates

def haversine_distance_np(lat1, lon1, lat2, lon2):
    # Convert latitudes and longitudes from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calculate the differences between latitudes and longitudes
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Haversine formula (vectorized for NumPy arrays)
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth's radius in kilometers (approximately)
    earth_radius = 6371

    # Calculate the distance in kilometers
    distance = earth_radius * c

    return distance

def haversine_distance_table(set1, set2):
    """
    Calculate the distance between each pair of points from two sets of locations.

    Input:
    set1: List of locations (latitude, longitude) in the format [[lat1, lon1], [lat2, lon2], ...]
    set2: List of locations (latitude, longitude) in the format [[lat1, lon1], [lat2, lon2], ...]

    Output:
    A 2D NumPy array with shape (len(set1), len(set2)), where the value at position [i, j] represents
    the distance between the i-th location in set1 and the j-th location in set2.
    """
    set1 = np.array(set1)
    set2 = np.array(set2)

    # Separate the latitudes and longitudes for broadcasting
    lat1, lon1 = set1[:, 0][:, np.newaxis], set1[:, 1][:, np.newaxis]
    lat2, lon2 = set2[:, 0], set2[:, 1]

    # Calculate the distance table using the haversine_distance_np function
    distance_table = haversine_distance_np(lat1, lon1, lat2, lon2)

    return distance_table


def indicator(X: List[list], Y: List[list]):
    x_loc, y_loc = [x.location for x in X], [y.location for y in Y]
    distance_table = haversine_distance_table(x_loc,y_loc)
    bd_dict = BidirectionalDict({})
    inter_dist = dict()
    for id,s1 in enumerate(distance_table):
        min_idx, min_val = np.argmin(s1), np.min(s1)
        bd_dict[id] = min_idx
        inter_dist[id] = min_val
    return bd_dict, inter_dist

def get_df(f_name):
    df = pd.read_csv(f_name)
    df["Start DateTime"] = pd.to_datetime(df["Start DateTime"])
    df["Time Difference"] = pd.to_timedelta(df["Time Difference"])
    df["req_lat"] = df["req_lat"].round(6)
    df["req_lng"] = df["req_lng"].round(6)
    df = df[["Start DateTime", "Time Difference", "req_lat", "req_lng"]]
    return df

def get_demand(df, periods) -> List[list]:
    result_df = pd.DataFrame(columns=df.columns)
    for period in periods:
        starttime, endtime = period[0], period[1]
        temp_df = df.loc[lambda x: (x["Start DateTime"] >= starttime) & (x["Start DateTime"] <= endtime)]
        result_df = pd.concat([result_df, temp_df], axis=0)
    values = result_df.to_numpy()
    return values

def get_intet_demand(inter_loc: List[list], values) -> List[dict[datetime, List[timedelta]]]:
    # values: ["Start DateTime", "Time Difference", "req_lat", "req_lng"]
    if values == []:
        raise Exception("No value in this period")
    req_loc = [[val[2], val[3]] for val in values]
    distance_table = haversine_distance_table(req_loc, inter_loc)
    inter_dem = [defaultdict(list) for _ in range(len(inter_loc))]
    for idx, dists in enumerate(distance_table):
        inter_dem[np.argmin(dists)][values[idx][0]].append(values[idx][1])
    return inter_dem

def get_fcs_inter(inter_loc: List[list], fcs_loc):
    if fcs_loc == []:
        raise Exception("No fcs data")
    distance_table = haversine_distance_table(fcs_loc, inter_loc)
    fcs_inters = []
    for dists in distance_table:
        fcs_inter = np.argmin(dists)
        fcs_inters.append([fcs_inter, inter_loc[fcs_inter]])
    return fcs_inters

def perfect_service_inter(mcs, intersections, stations, starttime, endtime, timestep):
    logger.debug(f"{mcs.intersection=}")
    new_stations = deepcopy(stations)
    new_stations.append(mcs)
    return perfect_service(intersections, new_stations, starttime, endtime, timestep)

def perfect_service(intersections, stations, starttime, endtime, timestep):
    try:
        # prepare indicator, timestep setting, global goal(traveling+waiting time)
        INDICATOR, inter_dist = indicator(intersections, stations)
        total_traveling_time = timedelta(seconds=0)
        total_waiting_time = timedelta(seconds=0)

        for s in stations:
            traveling_time_s = timedelta(seconds=0)
            waiting_time_s = timedelta(seconds=0)
            time = starttime

            # over all time step in time window
            while time < endtime:
                queue = []
                time += timestep

                # get demand of s at t (from t ~ t+1)
                for v in INDICATOR.inverse.get(s.id, []):
                    traveling_time_v = traveling_time(distance=inter_dist[v])
                    for k, v in intersections[v].demand.items():
                        if k > time - timestep and k <= time:
                            for r in v:
                                traveling_time_s += traveling_time_v
                                queue.append([k+traveling_time_v, r, timedelta(seconds=0)])
                queue.sort(key=lambda x:x[0])
                cursor = len(s.queue)
                s.queue += queue

                # get charger status of s
                for idx, c_status in enumerate(s.avai_time):
                    if c_status <= time:
                        if s.queue:
                            # update s.avai_time and pop waiting queue d_s
                            s.avai_time[idx] = time+s.queue.pop(0)[1]
                            cursor -= 1 if cursor > 0 else 0
                
                # calculate the waiting time of request in this period
                s.queue, _ = station_queue_update(s, time, batch=True)

                if s.queue:
                    for r in range(cursor,len(s.queue)):
                        waiting_time_s += s.queue[r][2]
                    
            indicated_v = INDICATOR.inverse.get(s.id, [])
            total_traveling_time += traveling_time_s
            total_waiting_time += waiting_time_s
            logger.debug(f"target_inter={stations[-1].intersection}, {s.id=}, indicated={len(indicated_v)}, v={indicated_v}")
            logger.debug(f"{waiting_time_s=}, {traveling_time_s=}")
        total_perfect_service = total_traveling_time + total_waiting_time
        return total_perfect_service

    except Exception as e:
        logger.error("Exception occurred: {}", e, exc_info=True)
        raise

def station_queue_update(station, current_time, batch=False):
    if Config.DEBUG == "True":
        logger.debug(f"1: {station.queue=}")
        logger.debug(f"1: {station.avai_time=}")
    if not batch:
        # fulfil the queue
        while station.queue:
            charger_status = np.array(station.avai_time)
            next_c = np.argmin(charger_status)
            if charger_status[next_c] < current_time:
                next_req = station.queue.pop(0)
                station.avai_time[next_c] = max(next_req[0], station.avai_time[next_c]) + next_req[1]
            else:
                break
    # Update waiting time
    if station.queue:
        charger_remain_time = np.array([t-current_time for t in station.avai_time])
        for req_idx in range(len(station.queue)):
            # get min charger remain time
            c = np.argmin(charger_remain_time)
            # update req waiting time
            station.queue[req_idx][2] = charger_remain_time[c]
            # update charger remain time for next req
            charger_remain_time[c] += station.queue[req_idx][1]
    if Config.DEBUG == "True":
        logger.debug(f"2: {station.queue=}")
        logger.debug(f"2: {station.avai_time=}")
    return station.queue, station.avai_time