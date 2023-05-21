import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
import csv
from prediction import getCoordinatesAndRssisFromRow
from trilateration import trilateration
import torch
from prediction import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork()
model.load_state_dict(torch.load('wifiModel.pth'))
model = model.to(device)

dataFile = open('dataSmall.csv')
csvReader = csv.reader(dataFile)

rows = []
for row in csvReader:
    if row[0]=='X':
        continue
    rows.append(row)

dataFile.close()

allCoords, allRssis = getCoordinatesAndRssisFromRow(rows)
distanceDiffs = []

oneMeterInPx = 66
for i, (rssi, coord) in enumerate(zip(allRssis, allCoords)):
    predCoords = model(torch.tensor([[abs(int(rssi[0])), abs(int(rssi[1])), abs(int(rssi[2]))]], dtype=torch.float32))[0].detach().numpy()
    coord = coord.numpy()
    distanceInPx = np.linalg.norm(coord - predCoords)
    distanceInMeters = distanceInPx / oneMeterInPx
    distanceDiffs.append(distanceInMeters)

x_list = allCoords[:, 0].numpy() * 1.178
z_list = allCoords[:, 1].numpy() * 1.178


C_I_list = np.array(distanceDiffs)
plt.figure(figsize=(8,8))

im = plt.imread('butoPlanasProper.png')
implot = plt.imshow(im)

scatterPlt = plt.scatter(x_list, z_list, c=C_I_list, s=150, cmap='viridis_r')
colorBar = plt.colorbar(scatterPlt)
colorBar.set_label('Paklaida (metrai)', rotation=270, labelpad=25)
plt.xlabel("Pikseliai (px)")
plt.ylabel("Pikseliai (px)")
plt.show()