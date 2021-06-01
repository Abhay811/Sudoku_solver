#importing packages

from numpy.lib.function_base import percentile
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


def find_puzzle(image, debug = False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    #applying adaptive thresholding and then inverting the thres. map
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    #finding contours in the thresholded image and sort them by size in des. order
    cont = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)

    puzzle_count = None

    #loop over contours
    for c in cont:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        #if we found four points, then we have found outline of puzzle
        if len(approx) == 4:
            puzzle_count = approx
            break

        #if puzzle contour is empty then, we should raise error
        if puzzle_count is None:
            raise Exception("Could not find Sudoku puzzle, Debug threshold and contour")
        
        if debug:
            #drawing contour of the puzzle on the image and displaying it
            out = image.copy()
            cv2.drawContours(out, [puzzle_count], -1, (255, 0, 0), thickness = 2)
            cv2.imshow("Puzzle outlining", out)
            cv2.waitKey(0)

    #applying four point perspective transform to original and gray
    puzzle = four_point_transform(image, puzzle_count.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_count.reshape(4, 2))

    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    #returning original and gray images
    return(puzzle, warped)


#extracting digits
def ext_digit(cell, debug = False):
    #applying automatic thresholding to the cell and clearing connected borders
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    #visualizing
    if debug:
        cv2.imshow("After thresh", thresh)
        cv2.waitKey(0)

    #finding contours in the thresholded cell
    conts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)

    #if no contours found
    if len(conts) == 0:
        return None

    #otherwise, find the largest contour in the cell and creating a mask for the contoru
    c = max(conts, key = cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    per_filled = cv2.countNonZero(mask) / float(w * h)

    #if < 3% of the mask filled then we are looking at noise and ignore the contour
    if per_filled < 0.03:
        return None
    
    #applying mask to the thresholded cell
    digit = cv2.bitwise_and (thresh, thresh, mask=mask)

    #visualizing
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    
    #return
    return digit
    