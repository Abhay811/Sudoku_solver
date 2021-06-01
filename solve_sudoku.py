#imporing packages


from pyimage.sudoku import ext_digit
from pyimage.sudoku import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
# import imutlis
import cv2

#constructing the arg parser and parsing the args
arg = argparse.ArgumentParser()
arg.add_argument("-m", "--model", required=True, help="path to trained digit classifier")
arg.add_argument("-i", "--image", required=True, help="path to input Sudoku puzzle image")
arg.add_argument("-d", "--debug", type=int, default=-1,
 help = "whether or not we are visualizing each sted of the pipeline")

args = vars(arg.parse_args())

#load the digit classifier
print("[INFO] loading digit classifier...")
model = load_model(args["model"])

#load the input and resize it
print("[INFO] processing image...")
image = cv2.imread(args["image"])
image = imutlis.resize(image, width = 600)

#find the puzzle in the image
(puzzle_img, warped) = find_puzzle(image, debug=args["debug"] > 0)

#initialize 9 x 9 Sudoku board
board = np.zeros((9, 9), dtype="int")

#location of each cell
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

#intializing a list to store the coordinate of each cell location
cell_loc = []

for y in range(0, 9):
    row = []
    for x in range(0, 9):
        #starting and ending coordinates of the current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY

        #add the coordinate to our cell location list
        row.append((startX, startX, endX, endY))

        #cropping the cell from warped transorm image and extracting the digit
        cell = warped[startY:endY, startX:endX]
        digit = ext_digit(cell, debug = args["debug"] > 0)

        #verifying the digit is not empty
        if digit is not None:
            #resize the cell to 28 x 28 for classification
            d = cv2.resize(digit, (28, 28))
            d = d.astype("float") / 255.0
            d = img_to_array(d)
            d = np.expand_dims(d, axis=0)

            #classifying digit and update the sudoku board with the prediction
            pred = model.predict(d).argmax(axis=1)[0]
            board[y, x] = pred
    cell_loc.append(row)

#constructing a sudoku puzzle from the board
print("[INFO] OCR'd Sudoku board:")
puzzle = Sudoku(3, 3, board = board.tolist())
puzzle.show()

#solving sudoku
print("[INFO] solving puzzle...")
sol = puzzle.solve()
sol.show_full()

#loop over cell and board
for (cell_row, board_row) in zip(cell_loc, sol.board):
    #loop over individual cell in the row
    for (box, digit) in zip(cell_row, board_row):
        startX, startY, endX, endY = box        #unpack the coordinates
        
        #computing coordinate, where the digit will be drawn on output puzzle
        textX = int((endX - startX) * 0.33)
        textY = int((endY - startY) * -0.2)
        textX += startX
        textY += endY
        
        #drawing the result digit on the sudoku puzzle image
        cv2.putText(puzzle_img, str(digit),(textX, textY), 
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 255, 255), thickness=2)

cv2.imshow("Sudoku Result", puzzle_img)
cv2.waitKey(0)
