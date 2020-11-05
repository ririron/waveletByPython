# -*- coding: utf-8 -*-
#ダイアディック離散ウェーブレットを実施するクラス
import Dwt
import numpy as np


class DyDwt(Dwt.Dwt):

    def __init__(self, iniScaleFi, iniWaveletFi,
                 iniScaleFiStartInd, iniWaveletFiStartInd):
        #スケーリングフィルタ，ウェーブレットフィルタ，スケーリングフィルタのt=0のインデックス，ウェーブレットフィルタの~
        super(DyDwt, self).__init__(iniScaleFi, iniWaveletFi)
        self.scaleFiStartInd = iniScaleFiStartInd
        self.waveletFiStartInd = iniWaveletFiStartInd
        
    def do1dimTransform(self, inputSignal):
        "1次元ダイアディック変換を行うメソッド"
        #入力信号のサイズ
        inputSignalSize = np.shape(inputSignal)[0]
        #両端の境界のため，ミラーリング
        inputSignal = np.concatenate([inputSignal[-1:0:-1], inputSignal,
                                     inputSignal[-2::-1]])                  
        #低周波成分と高周波成分の初期化
        lowFreqComp = np.zeros(inputSignalSize)
        highFreqComp = np.zeros(inputSignalSize)
        for t in range(0, inputSignalSize, 1):
            #左側のインデックス，右側のインデックス
            leftInd = (t + inputSignalSize - 1) - self.scaleFiStartInd
            rightInd = leftInd + self.scaleFiSize
            lowFreqComp[t] = sum(self.scaleFi 
                                * inputSignal[leftInd : rightInd])

            #左側のインデックス，右側のインデックス
            leftInd = (t + inputSignalSize - 1) - self.waveletFiStartInd
            rightInd = leftInd + self.waveletFiSize 
            highFreqComp[t] = sum(self.waveletFi 
                                * inputSignal[leftInd : rightInd])

        return lowFreqComp, highFreqComp

    
    def do2dimTransform(self, inputImg):
        """2次元ダイアディック変換を行うメソッド，1次元変換を向きを変えて複数行う．"""
        #入力画像のサイズ，(height, width)を想定
        inputImgHeight = np.shape(inputImg)[0]
        inputImgWidth = np.shape(inputImg)[1]
        #一時的な結果を格納
        tmpLowFreqComp = np.zeros((inputImgHeight, inputImgWidth))
        tmpHighFreqComp = np.zeros((inputImgHeight, inputImgWidth))

        for t in range(0, inputImgHeight, 1):
            tmpLowFreqComp[t], tmpHighFreqComp[t] = self.do1dimTransform(inputImg[t, :])

        #各成分の初期化
        cA = np.zeros((inputImgWidth, inputImgHeight)) #転置するから順序が逆    
        cV = np.zeros(np.shape(cA))
        cH = np.zeros(np.shape(cA))
        cD = np.zeros(np.shape(cA))

        #転置して再度1dimDwt
        tmpLowFreqComp = tmpLowFreqComp.T
        tmpHighFreqComp = tmpHighFreqComp.T
        for t in range(0, inputImgWidth, 1):
            cA[t, :], cH[t, :] = self.do1dimTransform(tmpLowFreqComp[t, :])
            cV[t, :], cD[t, :] = self.do1dimTransform(tmpHighFreqComp[t, :])

        return cA.T, cV.T, cH.T, cD.T
		




