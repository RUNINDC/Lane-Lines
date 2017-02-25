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

The Canny edge takes a low and high threshold which detects a minimum difference in intensity to detect edge and creates a continous extension of an edge.

III. Region of Interest: Not all of the edges are important to us, so we apply a function that masks out most of the image and only retain the bottom part of the road in view.


Then we calculate a Left Region of Interest and Right Region of Interest


IV. Hough Transform We retrieve Hough Lines by converting the pixel dots that were detected as edges into lines. 

cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

The Hough Transform, which took the most amount of time to perform, takes a resolution for line position and orientation, a min number of pts for a line, the min length of a line and the max gap between pts in a line.

This pipeline was effective in obtain lines along both sides of the lanes.

V. Apply  lane lines on the test images. 
Then we overlap the detected lines on the test images. In order to draw a single line on both the left and right lanes, I modified the draw lines function by:

1. Calculated slope and center of each line — then based on the slope, sort it into left or right.
2. Calculate the avg. slope and the center of left and right lane
3. Then using the Y coordinates from ROI, determine the X coordinates using the avg. slope and center point of lane lines.

Shortcomings: There are potential shortcomings. The ROI is static so it can only be used in specific scenarios. Additionally, the slope conditions used for determining lanes do not work for curved roads.

Improvements: modify the mask selection to be dynamic so that it can work in many different scenarios. Adjust the slope conditions so that they can work on curved roads.
