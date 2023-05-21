import csv
from prediction import train_test_split, getCoordinatesAndRssisFromRow
from trilateration import trilateration
import numpy as np

def test(rssis, coords):
    num_batches = len(coords)
    diff = 0
    oneMeterInPx = 66
    for i, (rssi, coord) in enumerate(zip(rssis, coords)):
        predCoords = trilateration(rssi[0], rssi[1], rssi[2])
        coord = coord.numpy()
        distanceInPx = np.linalg.norm(coord - predCoords)
        diff += distanceInPx

    avgAccInPx = diff / num_batches
    print('Average accuracy in px:', avgAccInPx)
    print('Average accuracy in meters:', avgAccInPx / oneMeterInPx)

dataFile = open('data.csv')
csvReader = csv.reader(dataFile)

rows = []
for row in csvReader:
    if row[0]=='X':
        continue
    rows.append(row)

dataFile.close()

trainRows, testRows = train_test_split(rows, 80)

trainCoords, trainRssis = getCoordinatesAndRssisFromRow(trainRows)
testCoords, testRssis = getCoordinatesAndRssisFromRow(testRows)
allCoords, allRssis = getCoordinatesAndRssisFromRow(rows)

test(testRssis, testCoords)
test(allRssis, allCoords)