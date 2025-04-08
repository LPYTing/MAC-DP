from typing import List
from datetime import datetime, timedelta


class Intersection:
    def __init__(self, id: int, location: List[list], demand: dict[datetime, List[timedelta]] = []):
        self.id = id
        self.location = location
        self.demand = demand

class Mcs:
    def __init__(self, id: int, connector: int = 2, location: List[list] = None, intersection: int = None, avai_time: List[datetime]=None, queue: list=None, hex: int = None):
        if not avai_time:
            avai_time = [datetime.fromisoformat("2000-01-01")] * connector
        if queue is None:
            queue = []
        self.id = id
        self.connector = connector
        self.location = location
        self.intersection = intersection
        self.avai_time = avai_time
        self.queue = queue  # queue: List[[arrival_time, request_time, waiting_time]]
        self.hex = hex

class Fcs:
    def __init__(self, id: int, connector: int = 2, location: List[list] = None, intersection: int = None, avai_time: List[datetime]=None, queue: list=None, hex: int = None):
        if not avai_time:
            avai_time = [datetime.fromisoformat("2000-01-01")] * connector
        if queue is None:
            queue = []
        self.id = id
        self.connector = connector
        self.location = location
        self.intersection = intersection
        self.avai_time = avai_time
        self.queue = queue  # queue: List[[arrival_time, request_time, waiting_time]]
        self.hex = hex

class Hex:
    def __init__(self, id, polygon) -> None:
        self.id = id
        self.polygon = polygon
        self.centroid = [self.polygon.centroid.y, self.polygon.centroid.x]
        self.fcs_list = []

class BidirectionalDict(dict):
    def __init__(self, *args, **kwargs):
        super(BidirectionalDict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
            if not self.inverse[self[key]]:
                del self.inverse[self[key]]
        super(BidirectionalDict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(BidirectionalDict, self).__delitem__(key)

    def get_key(self, value, default=None):
        return self.inverse.get(value, [default])[0]

    def get_keys(self, value, default=None):
        return self.inverse.get(value, default)
