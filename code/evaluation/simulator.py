import random,math
from geopy.distance import geodesic
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from utils.classes import Fcs
from utils.util import fcs_trans, traveling_time, station_queue_update
from dateutil.relativedelta import relativedelta
from config import Config
from loguru import logger



def simulator(demand, stations, d_factor, w_factor, f_factor):
    # logger.debug(f"demand launch time: {demand['Start DateTime']}")
    launch_time = demand['Start DateTime'].to_pydatetime()
    request_time = demand['Time Difference']
    if Config.DEBUG == "True":
        logger.debug(f"{launch_time=}, {request_time=}")
    # print([(idx, s.avai_time) for idx, s in enumerate(stations)])
    # d_score: nearest
    distances = [geodesic([demand["req_lat"], demand["req_lng"]], station.location).km for station in stations]
    trans_d = [1/np.exp(d) for d in distances]
    norm_d = sum(trans_d)
    d_score = [ds/norm_d for ds in trans_d]

    # w_score: available
    w_score = [0.0 for _ in range(len(stations))]
    for idx,s in enumerate(stations):
        if any([not isinstance(charger, datetime) for charger in s.avai_time]):  # if any None -> available
            w_score[idx] = 1.0
        elif any([charger < launch_time for charger in s.avai_time if isinstance(charger, datetime)]) and not s.queue: # if any charger available
            w_score[idx] = 1.0
    norm_w = sum(w_score)
    try:
        w_score = [ws/norm_w for ws in w_score]
    except:
        pass

    # f_score: most frequestly charge
    f_score = [0.0 for _ in range(len(stations))]
    try:  # Should build "History" column
        for station, count in demand["History"].items():
            f_score[station.id] += count
        norm_f = sum(f_score)
        f_score = [fs/norm_f for fs in f_score]
    except:
        pass
    
    # Compute the final probability of each station
    final_scores = [d_score[i]*d_factor + w_score[i]*w_factor + f_score[i]*f_factor for i in range(len(stations))]

    while True:
        # select a station by probability
        target_station = random.choices(stations, weights=final_scores, k=1)[0]

        travel_time = traveling_time(distance=distances[target_station.id])
        ev_arrival_time = travel_time + launch_time
        
        # if no charger available, add to queue
        # queue: List[[arrival_time, request_time, waiting_time]]
        target_station.queue.append([ev_arrival_time, request_time, timedelta(seconds=0)])

        if Config.DEBUG == "True":
            logger.debug(f"{final_scores=}")
            sorted_dist = sorted(distances)
            logger.debug(f"{target_station.id=}, distance: {distances[target_station.id]}, dist rank: {sorted_dist.index(distances[target_station.id])+1}, final_score: {final_scores[target_station.id]}")
            logger.debug(f"{ev_arrival_time=}")

        # update the target station queue and refresh
        target_station.queue, target_station.avai_time = station_queue_update(target_station, ev_arrival_time)
        wait_time = target_station.queue[-1][2] if target_station.queue else timedelta(seconds=0)

        # set a waiting time threshold
        if wait_time < timedelta(days=14):
            break
        else:
            final_scores[target_station.id] = 0.0
            if any(final_scores) == False:
                break
            target_station.queue.pop(-1)

    perfect_score_ev = (travel_time + wait_time) / timedelta(days=1)
    if Config.DEBUG == "True":
        logger.debug(f"{perfect_score_ev=}",end="\n\n")
        logger.debug([(s.id, len(s.queue)) for s in stations])
    return stations, target_station.id, travel_time, wait_time, perfect_score_ev
