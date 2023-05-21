import numpy as np
from scipy.optimize import curve_fit
import math

def calcDistance(rssi):
    if rssi > 0:
        rssi = - rssi
    
    v = (100 * rssi) + 816
    a = 965
    virsus = (100 * pow(math.e, -(v/a))) - 653

    distance = virsus / 2768
    return distance


apCoordinates = [[446, 41], [288, 304], [159, 38]]

def rms(y, a, b):
    yfit = [a, b]
    return np.linalg.norm(y - yfit, axis=1)


def trilateration(firstRssi, secondRssi, thirdRssi):
    rssiValues = [int(firstRssi), int(secondRssi), int(thirdRssi)]
    rssiValues = [calcDistance(x) for x in rssiValues]
    popt2, pcov2 = curve_fit(rms, np.array(apCoordinates), rssiValues, sigma=rssiValues, absolute_sigma=True)
    return [int(popt2[0]), int(popt2[1])]