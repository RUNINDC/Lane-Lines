Self Driving Car — Finding Lane Lines
The objective is to detect lane lines on a Road using Python 3.5 and OpenCV
Tools:
1. Open CV
2. Numpy
3. Matplotlib
3. MoviePy

I. First thing we do is convert to greyscale and then blur the image. The kernel size is the amount of blur to be applied.
cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

II. Then we apply the Canny Edge Detection 
cv2.Canny(img, low_threshold, high_threshold)

III. Region of Interest: Not all of the edges are important to us, so we apply a function that masks out most of the image and only retain the bottom part of the road in view.


Then we calculate a Left Region of Interest and Right Region of Interest


IV. Hough Transform Now we convert the pixel dots that were detected as edges into lines. 

cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
V. Superimpose lane lines on the test images

Then we overlap the detected lines on the test images. In order to draw a single line on both the left and right lanes, I modified the draw lines function by:

1. Calculated slope and center of each line — then based on the slope, sort it into left or right.
2. Calculate the avg. slope and the center of left and right lane
3. Then using the Y coordinates from ROI, determine the X coordinates using the avg. slope and center point of lane lines.
