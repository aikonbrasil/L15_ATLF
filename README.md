# Advanced Techniques for Lane Finding Lines
### L15_ATLF

This notes are used to sumarize some important topics of the train.

## Finding Corners
The main OpenCV functions that we are going to use are

```python
findChessboardCorners()
drawChessboardCorners()
```
to automatically find and draw corners in an image of a chessboard pattern. A basic example is described in following lines

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 8#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

# Make a list of calibration images
fname = 'calibration_test.png'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
```

