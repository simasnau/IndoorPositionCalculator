from fastapi import FastAPI
import torch
from model import NeuralNetwork
from trilateration import trilateration

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork()
model.load_state_dict(torch.load('wifiModel.pth'))
model = model.to(device)

@app.get("/neural/{firstRssi}/{secondRssi}/{thirdRssi}")
async def root(firstRssi, secondRssi, thirdRssi):
    prediction = model(torch.tensor([[abs(int(firstRssi)), abs(int(secondRssi)), abs(int(thirdRssi))]], dtype=torch.float32))
    prediction = prediction[0]
    return {"x": int(prediction[0].item()), "y": int(prediction[1].item())}

@app.get("/trilateration/{firstRssi}/{secondRssi}/{thirdRssi}")
async def root(firstRssi, secondRssi, thirdRssi):
    coords = trilateration(firstRssi, secondRssi, thirdRssi)
    return {"x": coords[0], "y": coords[1]}
