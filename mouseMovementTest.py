# no she-bang in Windows

from Keyboard_And_Mouse_Controls import *
import math

y_distance = 20
x_distance = 75

while True:
    # input("Wait at prompt ")

    # yCoord = 0
    # xCoord = 0

    # yCoord = int(input("Enter yCoord"))
    # xCoord = int(input("Enter xCoord"))

    # if yCoord == None:
    #     break

    for i in range(1,6):
        print(i)
        time.sleep(1)

    # target = [yCoord, xCoord]

    # # AimMouse(target)

    # offsetY, offsetX = target
    # y_distance = -1 * (450 - offsetY)
    # x_distance = -1 * (800 - offsetX)
    # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_distance, y_distance, 0, 0)

    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_distance, y_distance, 0, 0)

    y_distance = -1 * y_distance
    x_distance = -1 * x_distance


