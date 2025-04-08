import h3
from shapely.geometry import Polygon, Point


def create_hexagons_polygon(polygon, resolution):
    # Get the H3 indices covering the bounding box
    covering_h3_indexes = h3.polyfill(
        {'type': 'Polygon', 'coordinates': [list(polygon.exterior.coords)]},
        resolution
    )

    # Create a list to store hexagon polygons and their corresponding H3 index
    hexagons = []

    # Iterate through the H3 indexes and create hexagonal polygons
    for h3_index in covering_h3_indexes:
        # Get hexagon vertices as a list of lat, lng tuples
        vertices = h3.h3_to_geo_boundary(h3_index, geo_json=True)

        # Create a Shapely Polygon from the vertices
        hexagon = Polygon(vertices)

        # If the hexagon intersects the original polygon, add it and its H3 index to the list
        hexagons.append(hexagon)

    if resolution == 8:
        hexagons.sort(key=lambda x: (x.centroid.y, -x.centroid.x), reverse=True)
        new_hexagons = [0 for _ in range(len(hexagons))]
        mannual_mapping = [4,2,0,12,9,7,5,3,1,21,18,15,13,10,8,6,28,25,22,19,16,14,11,38,35,32,29,26,23,20,17,49,46,43,40,36,33,30,27,24,61,57,54,51,47,44,41,37,34,31,69,66,63,59,55,52,48,45,42,39,81,77,74,71,67,64,60,56,53,50,94,90,86,83,79,75,72,68,65,62,58,103,99,96,92,88,84,80,76,73,70,116,112,108,105,101,97,93,89,85,82,78,120,118,114,110,106,102,98,95,91,87,119,115,111,107,104,100,117,113,109]
        for idx, hex in enumerate(hexagons):
            new_hexagons[mannual_mapping.index(idx)] = hex
        hexagons = new_hexagons
    return hexagons

def located_grid(location, hexagons):
    point = Point(location[::-1])
    min_distance = float('inf')
    nearest_idx = None
    for idx, hex in enumerate(hexagons):
        if hex.polygon.contains(point):
            return idx
        else:
            hex_center = hex.polygon.centroid
            distance = point.distance(hex_center)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = idx
    if nearest_idx is not None:
        return nearest_idx
    else:
        raise Exception(f"No hexagon found!")

