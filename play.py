import cv2
import numpy as np
from random import choice
import time
from tensorflow import keras
import h5py
model = keras.models.load_model("rps.h5")
dict = {0:'Rock',1:'Paper',2:'Scissors',3:'None'}


def getWinner(a, b):
    if a == 'None':
        return 'Please make a Move'

    if a == b:
        return "It's a Tie"

    if a == "Rock":
        if b == "Scissors":
            return "You have Won this Round" 
        if b == "Paper":
            return "Computer has Won this Round"
            
    if a == "Paper":
        if b == "Rock":
            return "You have Won this Round"
        if b == "Scissors":
            return "Computer has Won this Round"

    if a == "Scissors":
        if b == "Paper":
            return "You have Won this Round"
        if b == "Rock":
            return "Computer has Won this Round"

class Colors():
    red = (0,0,255)
    cyan = (225, 203, 71)
    green = (0,255,0)
    blue  =  (153, 110, 0)
    black = (0,0,0)
    
def getUserMove(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))
    pred = model.predict(np.array([img]))
    className = pred.argmax(axis=-1)
    return dict[className[0]]


window = cv2.VideoCapture(0)
compsMove = 'None'
userMove = 'None'
winner = 'Please Make a Move'
u,c,i,t=0,0,0,0

while True:
    i+=1
    val, frame = window.read()
    if not val:
        continue
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, dsize=None, fx=1.6, fy=1.5)
    cv2.rectangle(frame, (50, 150), (450, 550), Colors.red, 3)
    cv2.rectangle(frame, (570, 150), (970, 550), Colors.green, 3)
    k = cv2.waitKey(10)
    if k == ord('a'):
        t=2.5
        i=0
        userMove = getUserMove(frame[150:550, 570:970])
        compsMove = choice(['Rock', 'Paper', 'Scissors'])
        path = 'Images/'+compsMove+'.png'
        compimg = cv2.imread(path)
        compimg = cv2.resize(compimg,(400, 400) )
        frame[150:550, 50:450]= compimg
        winner = getWinner(userMove, compsMove)
        if(winner == "You have Won this Round") : u+=1
        if(winner == "Computer has Won this Round") : c+=1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Press 'a' to start a new round , 'q' to Quit and 'r' to Restart", (200, 660), font,
                0.6, (255, 255, 225), 1, cv2.LINE_AA)
    cv2.putText(frame, "Y O U  :  " + str(u), (570, 142), font,
                0.6,Colors.green, 2, cv2.LINE_AA)
    cv2.putText(frame, "C O M P U T E R  :  "+str(c), (50, 142),
                font, 0.6,Colors.red, 2, cv2.LINE_AA)
    cv2.putText(frame, "Your Move: " + userMove,
                (570, 580), font, 1, Colors.black, 2, cv2.LINE_AA)
    cv2.putText(frame, "Comp's Move: " + compsMove,
                (50, 580), font, 1,Colors.black, 2, cv2.LINE_AA)
    cv2.putText(frame, str(winner),
                (20, 60), font, 2, Colors.cyan, 4, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissors', frame)
    if(i==1):
        time.sleep(t)
        i=0
        t=0
        compsMove = 'None'
        userMove = 'None'
        winner = 'Please make a Move'
    if k == ord('r'):
        u,c = 0,0
    if k == ord('q'):
        break
window.release()
cv2.destroyAllWindows()
