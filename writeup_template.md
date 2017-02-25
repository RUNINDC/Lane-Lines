Self Driving Car — Finding Lane Lines
The objective is to detect lane lines on a Road.
Tools:
Open CV

2. Numpy
3. MoviePy
I. First thing we do is convert to greyscale and then blur the image. The kernel size is the amount of blur to be applied.
cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

II. Then we apply the Canny Edge Detection 
cv2.Canny(img, low_threshold, high_threshold)

III. Region of Interest: Not all of the edges are important to us, so we apply a function that masks out most of the image and only retain the bottom part of the road in view.


Then we calculate a Left Region of Interest and Right Region of Interest


IV. Hough Transform Now we convert the pixel dots that were detected as edges into lines. 

cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
