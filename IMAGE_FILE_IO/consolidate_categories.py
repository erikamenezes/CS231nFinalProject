import numpy as np

# Script to consolidate data from sub categories into larger categories



type = 'shop'

array = np.load("/Users/ckanitkar/Desktop/img_npy/CLOTHING/Blouse/" + type + "_labels.npy")
array2 = np.load("/Users/ckanitkar/Desktop/img_npy/CLOTHING/Coat/" + type + "_labels.npy")
array3 = np.load("/Users/ckanitkar/Desktop/img_npy/CLOTHING/Polo_Shirt/" + type + "_labels.npy")
array4 = np.load("/Users/ckanitkar/Desktop/img_npy/CLOTHING/T_Shirt/" + type + "_labels.npy")
array5 = np.load("/Users/ckanitkar/Desktop/img_npy/CLOTHING/Tank_Top/" + type + "_labels.npy")



output = np.concatenate((array, array2, array3, array4, array5))

print (output.shape)
print(output[0])
#np.save("/Users/ckanitkar/Desktop/img_npy_final/CLOTHING/UpperBody/" + type + "_labels.npy", output)