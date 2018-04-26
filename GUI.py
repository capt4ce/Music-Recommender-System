import os
# from tkFileDialog import askdirectory
 
from tkinter import filedialog
from tkinter import messagebox


import requests
import pygame
import os
# from Tkinter import *
# import ttk
from tkinter import *
from tkinter import ttk
import numpy as np
import cv2
import time

import pandas
import numpy as np
from Recommender.DeepContent.Recommender import ContentRecommender

current_username = None
selected_track_id = None
lbox_current_selected = None
file = None

subscription_key = "dc7f34ef1a614988be991c7f3fedb615"
assert subscription_key
face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

songs_dir = 'dataset/song_mp3/'

Recommender = ContentRecommender('dataset/deep_learning/label_map(boolean_mood).csv', 'dataset/main_song_labels.csv', 'dataset/main_labels.csv', 'dataset/main_user_rating.csv')
Recommender.prepareRecommendation(False)

# songs=[
#     ("Avicii", "D:/music/pymusic/Avicii.mp3",0),
#     ("Coldplay Adventure","D:/music/pymusic/Avicii.mp3",1),
#     ("Yuna","D:/music/pymusic/Yuna.mp3",0),
#     ("Calvin Harris","D:/music/pymusic/CalvinHarris.mp3",1),
#     ]
songs = pandas.DataFrame()


root = Tk()
root.minsize(500,500)
root.configure(background="#1c1c1c")
frame = Frame(root,bg="#1c1c1c",padx=10, pady=10)
frame.pack(padx=10, pady=10,side=BOTTOM)
frame4 = Frame(root,bg="#1c1c1c",padx=10, pady=10)
frame4.pack(side=BOTTOM)
frame1 = Frame(root,bg="#1c1c1c",padx=10, pady=10)
frame1.pack(padx=10, pady=10,side=TOP)
frame2 = Frame(root,bg="#1c1c1c",padx=10, pady=10)
frame2.pack(padx=10, pady=10,side=TOP)
frame3 = Frame(root,bg="#1c1c1c",padx=10, pady=10)
frame3.pack(padx=10, pady=10,side=TOP)
name = StringVar()
title = StringVar()
context = StringVar()
reaction = StringVar()


label1 = ttk.Label(frame1,text='User Name Recommendation',background="#1c1c1c",foreground="white")
label1.pack(padx=10, pady=10,side=LEFT)
mentry1=ttk.Entry(frame1,textvariable = name)
mentry1.pack(padx=10, pady=10, ipadx=70,ipady=4,side=LEFT)

lbox = Listbox(root,width=50,height=10)
# for i in range(len(songs)):
# 	lbox.insert(i,songs[i][0])

def lbox_clicked(e):
    global songs, selected_track_id, popupMenu, lbox_current_selected
    try:
        print(lbox.curselection()[0])
        popupMenu.current(int(songs.loc[lbox.curselection()[0]]['rating']))
        selected_track_id = songs.loc[lbox.curselection()[0]]['track_id']
        print(selected_track_id)
        lbox_current_selected = lbox.curselection()[0]
    except:
        return

lbox.bind('<<ListboxSelect>>',lbox_clicked)

def resetGlobal():
    global file, selected_track_id
    file = selected_track_id = None

def getRecommendation():
    global songs, Recommender, lbox, current_username
    userId = mentry1.get()
    current_username=userId
    songs = Recommender.recommend(userId)
    print(songs)
    lbox.delete(0,'end')
    for i in range(len(songs)):
        lbox.insert(i,songs.loc[i]['title'])
    resetGlobal()

def titleSearch():
    global songs
    songs=Recommender.titleSearch(mentry2.get())
    print(songs)
    lbox.delete(0,'end')
    for i in range(len(songs)):
        lbox.insert(i,songs.loc[i]['title'])
    resetGlobal()  


def labelSearch():
    global songs
    songs=Recommender.labelSearch(mentry3.get())
    lbox.delete(0,'end')
    for i in range(len(songs)):
        lbox.insert(i,songs.loc[i]['title']) 
    resetGlobal() 
        

getbutton = ttk.Button(frame1, text = 'Get Recommendation', command=getRecommendation)
getbutton.pack(padx=10, pady=10,side=LEFT)

label2 = ttk.Label(frame2,text='Title Search',background="#1c1c1c",foreground="white")
label2.pack(padx=45,pady=10,side=LEFT)
mentry2=ttk.Entry(frame2,textvariable = title)
mentry2.pack(padx=10, pady=10, ipadx=70,ipady=4,side=LEFT)
searchbutton = ttk.Button(frame2, text = 'Search', command=titleSearch)
searchbutton.pack(padx=20, pady=10,side=LEFT)

label3 = ttk.Label(frame3,text='Context Search',background="#1c1c1c",foreground="white")
label3.pack(padx=40,pady=10,side=LEFT)
mentry3=Entry(frame3,textvariable = context)
mentry3.pack(padx=10, pady=10, ipadx=70,ipady=4,side=LEFT)
searchbutton1 = ttk.Button(frame3, text = 'Search', command=labelSearch)
searchbutton1.pack(padx=10, pady=10,side=LEFT)

label4 = ttk.Label(root,text='Song List (Recommendation/Search Result)',background="#1c1c1c",foreground="white")
label4.pack(padx=10,pady=10)

label5=Label(root,text=None,background="#1c1c1c",foreground="white")


lbox.pack(padx=10, pady=10,ipadx=50)

def mPause():
    pygame.mixer.music.pause()

def mPlay():
    global file, songs, lbox, songs_dir, player
    idx = lbox.curselection()[0]
    if idx == None:
        idx = 0
    file = songs_dir + songs.loc[idx]['mp3_file']
    print(file)
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()

def add():
    global file,label4
    file = filedialog.askopenfilename()
    label5.config(text='processing '+file)
    #label4.pack()
    print(label5.labelText)

def TakeSnapshotAndSave():
    # access the webcam (every webcam has a number, the default is 0)
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # to detect faces in video
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

        x = 0
        y = 20
        text_color = (0,255,0)
        # write on the live stream video
        cv2.putText(frame, "Press q when ready", (x,y), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=2)


        # if you want to convert it to gray uncomment and display gray not fame
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        # press the letter "q" to save the picture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # write the captured image with this name
            cv2.imwrite('try.jpg',frame)
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    a = return_emotion()

 
def return_emotion():
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream" }

    params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
    }

    image_path = "C:/Users/Kapil/Desktop/basic-image-processing-master/basic-image-processing-master/try.jpg"
    # image_path = "C:/Users/Kapil/Downloads/main007.jpg"
    image_data = open(image_path, 'rb').read()

    response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
    faces = response.json()
    print(faces)
    return faces          

def rateSong():
    global current_username, selected_track_id, lbox
    rating_val = popupMenu.get()
    if rating_val.lower()=='good' and selected_track_id!=None:
        rating = 1
    elif rating_val.lower()=='bad' and selected_track_id!=None:
        rating = 0
    else:
        messagebox.showinfo('Incorrect Rating Value', 'Incorrect Rating Value')
        return
    songs.rating.loc[lbox_current_selected] = rating
    lbox.delete(0,'end')
    for i in range(len(songs)):
        lbox.insert(i,songs.loc[i]['title']) 

    Recommender.rateSong(current_username, selected_track_id, rating)


label6 = ttk.Label(frame4,text='Select',background="#1c1c1c",foreground="white")
label6.grid(row=0, column=0, padx=10, pady=10)
# label6.pack(padx=10, pady=10,side=LEFT)
popupMenu = ttk.Combobox(frame4, width=12, textvariable=reaction)
popupMenu['values']=('Bad', 'Good')
popupMenu.grid(row=0, column=1, padx=10, pady=10)
# popupMenu.pack(ipadx=20,ipady=5)
popupMenu.set('')
# popupMenu.current(0)

rateBtn = ttk.Button(frame4, text = 'Rate',command=rateSong)
rateBtn.grid(row=0, column=2, padx=10, pady=10)
# rateBtn.pack(padx=10,pady=10,side=LEFT)


playbutton = ttk.Button(frame, text = 'Play',command=mPlay)
playbutton.pack(padx=10,pady=10,side=LEFT)

stopbutton = ttk.Button(frame,text='Stop Music',command=mPause)
stopbutton.pack(padx=10, pady=10,side=LEFT)

songadd=ttk.Button(frame, text='Add Music',command=add)
songadd.pack(padx=10, pady=10, side=LEFT)

cap = ttk.Button(frame3, text = 'Detect Face',command=TakeSnapshotAndSave)
cap.pack(padx=10, pady=10,side=LEFT)

label5.pack(padx=10)

root.mainloop()
