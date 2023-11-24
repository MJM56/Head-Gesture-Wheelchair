# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:06:38 2022q

@author: eko my
"""
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
#'Stop,Maju,Kiri,Kanan,mundur'
import ModulEkstraksiFitur1 as md 
sDirektoriData = "d:\\1 PraTA\\Dataset\\KursiRoda1"
sKelas = "Mundur"
NoKamera = 0
FrameRate = 5
md.CreateDataSet(sDirektoriData, sKelas, NoKamera, FrameRate)


