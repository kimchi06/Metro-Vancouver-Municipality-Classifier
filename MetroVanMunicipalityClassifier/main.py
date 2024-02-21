# ===== The dataset for this project has been taken from the Metro Vancouver Open Data Portal:
# ===== https://arcg.is/0z8m00
# =====
# ===== The terms of use of this data is outlined in the license below:
# ===== https://open-data-portal-metrovancouver.hub.arcgis.com/pages/Open%20Government%20Licence

import sys
import json
import math
import random
import pandas as pd
from pandas import json_normalize
from sklearn.pipeline import make_pipeline
from shapely.geometry import Point, Polygon # pip install shapely
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split



def extract(coordinates):
    return pd.Series({'latitude': coordinates[1], 'longitude': coordinates[0]})



def main():
    img = sys.argv[1]
    input_lat = sys.argv[2]
    input_lon = sys.argv[3]
    input_lat = float(input_lat)
    input_lon = float(input_lon)

    with open('Administrative_Boundaries.geojson') as file:
        geojson = json.load(file)

    csv = pd.read_csv('Administrative_Boundaries.csv')
    
    parsed = json_normalize(geojson['features'])

    filtered = pd.DataFrame()
    filtered['municipality'] = parsed.iloc[:]['properties.FullName']
    filtered['coordinates'] = parsed.iloc[:]['geometry.coordinates']
    filtered['area'] = csv['SHAPE_Area'] / 1000000

    area_points = []
    for i in range(len(filtered['municipality'])):
        municipality = filtered['municipality'][i]
        coordinates = filtered['coordinates'][i][0][0]
        area = filtered['area'][i]

        polygon = Polygon(coordinates)
        min_lat, min_lon, max_lat, max_lon = polygon.bounds

        # Number of points set to: 5 / 1 km^2
        n = 5 * int(math.ceil(area) / 1)
        # Add points lying within the polygon following a uniform distribution
        for j in range(n):
            # Longitude is x, Latitude is y
            point = Point(random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
            while (point.within(polygon) != True):
                point = Point(random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
            area_points.append([municipality, point.y, point.x])

    exploded = filtered.explode('coordinates').explode('coordinates').explode('coordinates')
    exploded[['latitude', 'longitude']] = exploded['coordinates'].apply(lambda x: extract(x))
    exploded.drop(columns=['coordinates'], inplace = True)
    exploded.drop(columns=['area'], inplace = True)

    area_points_df = pd.DataFrame(area_points, columns=['municipality', 'latitude', 'longitude'])
    # all_points = pd.concat([exploded, area_points_df])
    all_points = area_points_df

    X = all_points[['latitude', 'longitude']]
    y = all_points['municipality']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # Various models
    # model = RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_leaf=10)
    # model = KNeighborsClassifier(n_neighbors=5, metric='haversine')
    model = VotingClassifier([
        ('knn', KNeighborsClassifier(n_neighbors=5, metric='haversine')),
        ('forest', RandomForestClassifier(n_estimators=100, max_depth=100, min_samples_leaf=10))
    ])
    
    model.fit(X_train, y_train)

    print("size of dataset: ", len(all_points))
    print("train: ", model.score(X_train, y_train))
    print("valid: ", model.score(X_valid, y_valid))

    # Custom prediction (latitude, longitude)
    coords = [[input_lat, input_lon]]
    prediction = model.predict(coords)
    print("the municipality associated with the image is: ", prediction)



if __name__ == '__main__':
    main()