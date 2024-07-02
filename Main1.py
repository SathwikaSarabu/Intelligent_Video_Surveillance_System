from tkinter import Tk, Label, Button, messagebox
from tkinter import *

import tkinter
from tkinter import ttk
from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import glob
import os 
import cv2
from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import imutils

from keras.models import load_model
from PIL import Image
from tkinter import filedialog
from tkinter.filedialog import askopenfilename


class Login_Page:

    def __init__(self,login = Tk()):
        self.login = login
        login.title("Login - Video Surveillance")
        login.geometry("450x230+450+170")
        login.configure(bg='lightblue')
        
        self.username = Label(login, text="Username:")
        self.username.place(relx=0.285, rely=0.298, height=20, width=55)

        self.password = Label(login, text="Password:")
        self.password.place(relx=0.285, rely=0.468, height=20, width=55)

        # Creating Buttons

        self.login_button = Button(login, text="Login")
        self.login_button.place(relx=0.440, rely=0.638, height=30, width=60)
        self.login_button.configure(command=self.login_user)

        self.login_completed = IntVar()

        self.exit_button = Button(login, text="Exit")  # , command=master.quit)
        self.exit_button.place(relx=0.614, rely=0.638, height=30, width=60)
        self.exit_button.configure(command=self.exit_login)

        # Creating entry boxes

        self.username_box = Entry(login)
        self.username_box.place(relx=0.440, rely=0.298, height=20, relwidth=0.35)

        self.password_box = Entry(login)
        self.password_box.place(relx=0.440, rely=0.468, height=20, relwidth=0.35)
        self.password_box.configure(show="*")
        self.password_box.configure(background="white")

        # Creating checkbox

        self.var = IntVar()
        self.show_password = Checkbutton(login)
        self.show_password.place(relx=0.285, rely=0.650, relheight=0.100, relwidth=0.125)
        self.show_password.configure(justify='left')
        self.show_password.configure(text='''Show''')
        self.show_password.configure(variable=self.var, command=self.cb)


    def cb(self, ):
        if self.var.get() == True:
            self.password_box.configure(show="")
        else:
            self.password_box.configure(show="*")

    # Giving function to login process

    def login_user(self):
        name = self.username_box.get()
        password = self.password_box.get()
        login_completed = self.login_completed.get()

        if name == "user" and password == "1234":
            #messagebox.showinfo("Login page", "Login successful!")
            self.login.destroy()  #Removes the toplevel window
            #self.main_win.deiconify() #Unhides the root window
            self.login_completed == 1

        else:
            messagebox.showwarning("Login Failed - Acess Denied", "Username or Password incorrect!")
            #return

    def exit_login(self):
        msg = messagebox.askyesno("Exit login page", "Do you really want to exit?")
        if (msg):
            exit()

    def mainloop_window(self): #This is the class function that helps me to mainloop the window
        self.login.mainloop()


login_page = Login_Page() # I dont need to pass the root now since its initialized inside the class
login_page.mainloop_window() # Just mainlooping the authentication window



##### Main Window #####

global filename
global model
images=[]


class Main_Win:

    def __init__(self,main= Tk()): 
        self.main= main
        main.title("Intelligent Video Surveillance System")
        main.geometry("1300x1200")
        

        def readImages(path):
            img = load_img(path)
            img = img_to_array(img)
            img = cv2.resize(img, (227,227), interpolation = cv2.INTER_AREA)
            gray = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
            images.append(gray)

        def upload():
            global filename
            filename = filedialog.askdirectory(initialdir=".")
            text.delete('1.0', END)
            text.insert(END,filename+" loaded\n");
    
        def datasetPreprocess():
            global filename, images
            images.clear()
            text.delete('1.0', END)
            img_list = os.listdir(filename)
            for img in img_list:
                print("Dataset/"+img)
                readImages("Dataset/"+img)
            images = np.array(images)
            testImage = images[0]
            height,width,color = images.shape
            images.resize(width,color,height)
            images = (images - images.mean()) / (images.std())
            images = np.clip(images, 0, 1)
            text.insert(END,"Total images found in dataset: "+ str(images.shape[0]))
            cv2.imshow("Process Images", testImage/255)
            cv2.waitKey(0)

        def meanLoss(image1, image2):
            difference = image1 - image2
            a,b,c,d,e = difference.shape
            n_samples = a*b*c*d*e
            sq_difference = difference**2
            Sum = sq_difference.sum()
            distance = np.sqrt(Sum)
            mean_distance = distance/n_samples
            return mean_distance
        
        def trainModel():
            global model
            text.delete('1.0', END)
            if os.path.exists('model/survey_model.h5'):
                model = load_model("model/survey_model.h5")
            else:
                stae_model=Sequential()
                stae_model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))
                stae_model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
                stae_model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))
                stae_model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))
                stae_model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))
                stae_model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
                stae_model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))
                stae_model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
                frames = images.shape[2]
                frames = frames-frames%10
                training_data = images[:,:,:frames]
                training_data = training_data.reshape(-1,227,227,10)
                training_data = np.expand_dims(training_data,axis=4)
                target_data = training_data.copy()
                callback_save = ModelCheckpoint("model/survey_model.h5", monitor="mean_squared_error", save_best_only=True)
                callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)
                stae_model.fit(training_data,target_data, batch_size = 1, epochs=5, callbacks = [callback_save,callback_early_stopping])
                stae_model.save("model/survey_model.h5")
            text.insert(END,"Auto Encoder Stae model generated and saved inside model folder")

        def abnormalDetection():
            global model
            text.delete('1.0', END)
            filename = filedialog.askopenfilename(initialdir="testVideos")
            cap = cv2.VideoCapture(filename)
            print(cap.isOpened())
            while cap.isOpened():
                imagedump=[]
                ret,frame=cap.read()
                for i in range(10):
                    ret,frame=cap.read()
                    if frame is not None:
                        image = imutils.resize(frame,width=700,height=600)
                        frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
                        gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
                        gray=(gray-gray.mean())/gray.std()
                        gray=np.clip(gray,0,1)
                        imagedump.append(gray)
                imagedump=np.array(imagedump)
                imagedump.resize(227,227,10)
                imagedump=np.expand_dims(imagedump,axis=0)
                imagedump=np.expand_dims(imagedump,axis=4)
                output=model.predict(imagedump)
                loss=meanLoss(imagedump,output)
                if frame is not None:
                    if frame.any()==None:
                        print("none")
                else:
                    break
                if cv2.waitKey(10) & 0xFF==ord('q'):
                    break
                print(str(frame)+" "+str(loss))
                if loss>0.00068:
                    print('Abnormal Event Detected')
                    cv2.putText(image,"Abnormal Event",(100,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
                else:
                    cv2.putText(image,"Normal Event",(100,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),4)
                cv2.imshow("video",image)
            cap.release()
            cv2.destroyAllWindows()

        def close():
            global main
            main.destroy()

        
        font = ('times', 16, 'bold')
        title = Label(main, text='Intelligent Video Surveillance System')
        title.config(bg='black', fg='white')  
        title.config(font=font)           
        title.config(height=3, width=120)       
        title.place(x=0,y=5)

        font1 = ('times', 12, 'bold')
        text=Text(main,height=30,width=100)
        scroll=Scrollbar(text)
        text.configure(yscrollcommand=scroll.set)
        text.place(x=400,y=100)
        text.config(font=font1)


        font1 = ('times', 12, 'bold')
        uploadButton = Button(main, text="Upload Video Frames Dataset", command=upload)
        uploadButton.place(x=50,y=100)
        uploadButton.config(font=font1)  

        processButton = Button(main, text="Dataset Preprocessing", command=datasetPreprocess)
        processButton.place(x=50,y=150)
        processButton.config(font=font1) 

        trainButton = Button(main, text="Train Spatial Temporal AutoEncoder Model", command=trainModel)
        trainButton.place(x=50,y=200)
        trainButton.config(font=font1) 

        testButton = Button(main, text="Test Video Surveillance", command=abnormalDetection)
        testButton.place(x=50,y=250)
        testButton.config(font=font1) 

        exitButton = Button(main, text="Exit", command=close)
        exitButton.place(x=50,y=300)
        exitButton.config(font=font1)
        
        main.config(bg='lightblue')

        

    def mainloop_window(self): #This is the class function that helps me to mainloop the window
        self.main.mainloop()



main_win = Main_Win() # I dont need to pass the root now since its initialized inside the class

main_win.mainloop_window() # Just mainlooping the authentication window