import Dwt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

		
if __name__ == "__main__":
    testObj = Dwt.Dwt([0.5, 0.5], [0.5, -0.5])
    lowComp, highComp = testObj.do1dimTransform([0,1,2,3,4,5,6,7,8,9,10])


    im = np.array(Image.open("./resource/LENNA.bmp"))
    cA, cV, cH, cD = testObj.do2dimTransform(im)
    plt.imshow(cV)
    plt.show()