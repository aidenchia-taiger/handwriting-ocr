import cv2
import argparse

from Denoiser import remove_shadow_grayscale
from WordSegmentator import prepareImg, wordSegmentation

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', help="path to input image")
	args = parser.parse_args()

	inp_path = args.i

	img = cv2.imread(inp_path, cv2.IMREAD_GRAYSCALE)
	#img = remove_shadow_grayscale(img)
	#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=27,C=20.0)
	img = prepareImg(img, 50)
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)[0]
	img = res[1] # tightly cropped word

	cv2.imwrite('out/custom-preprocessed.png', img)
	print('[INFO] Custom Preprocessed image written to out/custom-preprocessed.png')

if __name__ == "__main__":
	main()


