# -*- coding: utf-8 -*-
#離散ウェーブレットを実施するクラス
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Dwt:
	def __init__(self, iniScaleFi, iniWaveletFi):
        #スケーリングフィルタとそのフィルタ長
		#(x1, x2, x3, x4, ...)を想定
		self.scaleFi = np.array(iniScaleFi)
		self.scaleFiSize = np.shape(iniScaleFi)[0]
	
		#ウェーブレットフィルタとフィルタ長
		self.waveletFi = np.array(iniWaveletFi)
		self.waveletFiSize = np.shape(iniWaveletFi)[0]


	def getFilter(self):
    	#スケーリング・ウェーブレットフィルタを返すメソッド
		return self.scaleFi, self.waveletFi

	def do1dimTransform(self, inputSignal):
		"""1次元変換を行うメソッド．元信号の座標を1個飛ばしで見ていく"""
		#入力信号のサイズ
		inputSignalSize = np.shape(inputSignal)[0]
		#右端の境界処理のために，折り返してくっつける
		inputSignal = np.append(inputSignal, inputSignal[-2::-1])
		#低周波成分と高周波成分の初期化
		lowFreqComp = np.zeros(int(inputSignalSize/2))
		highFreqComp = np.zeros(int(inputSignalSize/2))
		for t in range(0, np.shape(lowFreqComp)[0], 1):
			lowFreqComp[t] = sum(self.scaleFi * inputSignal[2*t : 2*t + self.scaleFiSize])
			highFreqComp[t] = sum(self.waveletFi * inputSignal[2*t : 2*t + self.waveletFiSize])
		
		return lowFreqComp, highFreqComp

	def do2dimTransform(self, inputImg):
		"""2次元変換を行うメソッド，1次元変換を向きを変えて複数行う．"""
		#入力画像のサイズ，(height, width)を想定
		inputImgHeight = np.shape(inputImg)[0]
		inputImgWidth = np.shape(inputImg)[1]
		#一時的な結果を格納
		tmpLowFreqComp = np.zeros((inputImgHeight, int(inputImgWidth/2)))
		tmpHighFreqComp = np.zeros((inputImgHeight, int(inputImgWidth/2)))
		for t in range(0, inputImgHeight, 1):
			tmpLowFreqComp[t], tmpHighFreqComp[t] = self.do1dimTransform(inputImg[t, :])

		#各成分の初期化
		cA = np.zeros((int(inputImgHeight/2), int(inputImgWidth/2)))
		cV = np.zeros(np.shape(cA))
		cH = np.zeros(np.shape(cA))
		cD = np.zeros(np.shape(cA))
		#転置して再度1dimDwt
		tmpLowFreqComp = tmpLowFreqComp.T
		tmpHighFreqComp = tmpHighFreqComp.T
		for t in range(0, int(inputImgWidth/2), 1):
			cA[t, :], cH[t, :] = self.do1dimTransform(tmpLowFreqComp[t, :])
			cV[t, :], cD[t, :] = self.do1dimTransform(tmpHighFreqComp[t, :])

		return cA.T, cV.T, cH.T, cD.T
		