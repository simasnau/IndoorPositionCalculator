import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import random
import numpy as np
from model import NeuralNetwork

class CustomLineDataset(Dataset):
    def __init__(self, rssiValues, coordinates):
        self.coordinates = coordinates
        self.rssis = rssiValues
    def __len__(self):
        return len(self.coordinates)
    def __getitem__(self, idx):
        coordinates = self.coordinates[idx]
        rssiValue = self.rssis[idx]
        return rssiValue, coordinates

# metodas atskiriantis duomenis į mokymo ir testavimo
def train_test_split(data_list, percentage):
    random.shuffle(data_list)

    train_item_count = int(len(data_list) * percentage / 100)
    
    return data_list[:train_item_count], data_list[train_item_count:]

def normalize(data):
    return [(element - np.mean(data)) / np.std(data) for element in data]

def getCoordinatesAndRssisFromRow(rows):
    coordinates = []
    rssis = []
    for row in rows:
        coordinates.append([int(row[0]), int(row[1])])
        rssis.append([abs(int(row[2])), abs(int(row[3])), abs(int(row[4]))])
    return torch.FloatTensor(coordinates), torch.FloatTensor(rssis)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (rssis, coords) in enumerate(dataloader):
        rssis, coords = rssis.to(device), coords.to(device)

        optimizer.zero_grad()
        # Compute prediction error
        predictedCoords = model(rssis)
        loss = loss_fn(predictedCoords, coords)
        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            running_loss, current = loss.item(), batch * len(rssis)
            print(f"loss: {running_loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

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

    trainDataSet = CustomLineDataset(trainRssis, trainCoords)
    testDataSet = CustomLineDataset(testRssis, testCoords)
    trainDataLoader = DataLoader(trainDataSet, batch_size=4, shuffle=True)
    testDataLoader = DataLoader(testDataSet, batch_size=1, shuffle=True)


    model = NeuralNetwork().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainDataLoader, model, loss_fn, optimizer)
    print("Done!")

    # make a prediction
    prediction = model(torch.tensor([[40, 50, 60]], dtype=torch.float32))
    print(prediction)