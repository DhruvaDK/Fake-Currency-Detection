import tkinter as tk
from tkinter import Button, Frame, Label, Text, END
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from glob import glob
from numpy import load
from glob import glob
import keras
from tensorflow.keras.layers import Input, Convolution2D, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import os
import torch
from PIL import Image
from io import BytesIO


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.config(bg="skyblue")

        # changing the title of our master widget
        self.master.title(" Fake Currency Detection ")

        self.pack(fill=tk.BOTH, expand=1)

        w = tk.Label(root,
                     text="Real Time Fake Currency Detection System ",
                     fg="black",
                     bg="#FFFFFF",
                     font="Helvetica 20 bold italic")
        w.pack()
        w.place(x=400, y=0)

        # creating a button instance
        quitButton = Button(self, command=self.cap, text="Open Camera", bg="#FFFF00", fg="#4C0099",
                            activebackground="dark red", width=20)
        quitButton.place(x=50, y=150, anchor="w")

        load = Image.open(r"C:/Users/Dhruva D K/Desktop/Fake currency/logo.jfif")
        render = ImageTk.PhotoImage(load)

        t = tk.Label(root, text="Captured Currency", fg="black", bg='#FFFFFF',
                     font=("Times New Roman", 13, "bold italic"), width=12)
        t.pack()
        t.place(x=285, y=50)

        image1 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=150, width=150)
        image1.image = render
        image1.place(x=250, y=90)

        image2 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=150, width=150)
        image2.image = render
        image2.place(x=450, y=90)

    def cap(self, event=None):
        filename = 'temp.png'
        cap = cv2.VideoCapture(0)
        cnt = 0
        while True:
            cap_not, image1 = cap.read()
            cnt += 1
            cv2.imshow("Output", image1)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cv2.imwrite(filename, image1)
        cv2.destroyWindow("Output") 

        # global variables 

        # strings at index 0 is not used, it
        # is to make array indexing simple
        one = [ "", "one ", "two ", "three ", "four ",
                "five ", "six ", "seven ", "eight ",
                "nine ", "ten ", "eleven ", "twelve ",
                "thirteen ", "fourteen ", "fifteen ",
                "sixteen ", "seventeen ", "eighteen ",
                "nineteen "];
         
        # strings at index 0 and 1 are not used,
        # they is to make array indexing simple
        ten = [ "", "", "twenty ", "thirty ", "forty ",
                "fifty ", "sixty ", "seventy ", "eighty ",
                "ninety "];

        class CurrencyNotesDetection:
            """
            Class implements Yolo5 model to make inferences on a source provided/youtube video using Opencv2.
            """

            def __init__(self, model_name):
                """
                Initializes the class with youtube url and output file.
                :param url: Has to be as youtube URL,on which prediction is made.
                :param out_file: A valid output file name.
                """
                self.model = self.load_model(model_name)
                # similar to coco.names contains ['10Rupees','20Rupees',...]
                self.classes = self.model.names
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print("Using Device: ", self.device)

            def load_model(self, model_name):
                """
                Loads Yolo5 model from pytorch hub.
                :return: Customed Trained Pytorch model.
                """
                # Custom Model
                # model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt',force_reload=True)  # default
                # model = torch.hub.load('ultralytics/yolov5','custom', path=model_name, force_reload=True, device='cpu')
                # model = torch.hub.load('/home/gowtham/MajorProject/yolov5_custom/yolov5', 'custom', path=model_name, source='local')  # local repo
                model = torch.hub.load('./yolov5', 'custom', path=model_name, source='local')  # local repo
                
                # Yolo Model from Web
                # for file/URI/PIL/cv2/np inputs and NMS
                # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

                return model

            def class_to_label(self, x):
                """
                For a given label value, return corresponding string label.
                :param x: numeric label
                :return: corresponding string label
                """
                return self.classes[int(x)]

            def numToWords(self,n, s):
         
                str = ""
                
                # if n is more than 19, divide it
                if (n > 19):
                    str += ten[n // 10] + one[n % 10]
                else:
                    str += one[n]
            
                # if n is non-zero
                if(n != 0):
                    str += s
            
                return str

            def convertToWords(self,n):
                # stores word representation of given
                # number n
                out = ""

                # handles digits at ten millions and
                # hundred millions places (if any)
                out += self.numToWords((n // 10000000),"crore ")

                # handles digits at hundred thousands
                # and one millions places (if any)
                out += self.numToWords(((n // 100000) % 100),"lakh ")

                # handles digits at thousands and tens
                # thousands places (if any)
                out += self.numToWords(((n // 1000) % 100),"thousand ")

                # handles digit at hundreds places (if any)
                out += self.numToWords(((n // 100) % 10),"hundred ")

                if (n > 100 and n % 100):
                    out += "and "

                # handles digits at ones and tens
                # places (if any)
                out += self.numToWords((n % 100), "")

                return out

            def get_text(self,labelCnt):
                text = "Image contains"
                noOfLabels,counter = len(labelCnt),0
                for k,v in labelCnt.items():
                    text += " {}{} {} ".format(self.convertToWords(v),k,"Notes" if v>1 else "Note")
                    if(counter != noOfLabels-1):
                        text += 'and'
                    counter += 1

                return text


            def get_detected_image(self,img):
                # Images
                imgs = [img]  # batched list of images

                # Inference
                results = self.model(imgs, size=416)  # includes NMS
                #print(results)
                #print(results .shape)
                # Results
                results.print()  # print results to screen
                #results.show()  # display results
                results.save()  # save as results1.jpg, results2.jpg... etc. in runs directory
                # print(results)  # models.common.Detections object, used for debugging
                results.crop()
                labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
                print(cord)
                n = len(labels)
                labelCnt = {}
                for i in range(n):
                    classLabel = self.classes[int(labels[i])]
                    row = cord[i]
                    # row[4] is conf score
                    print("{} is detected with {} probability.".format(classLabel, row[4]))
                    if classLabel in labelCnt:
                        labelCnt[classLabel] += 1
                    else:
                        labelCnt[classLabel] = 1

                text = self.get_text(labelCnt)
                print("{} This is from yolo_detection.py".format(text))
                # call gTTS (Google Text To Speech)
                

                # Data
                print('\n', results.xyxy[0])  # print img1 predictions
                #          x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence        class
                # tensor([[7.47613e+02, 4.01168e+01, 1.14978e+03, 7.12016e+02, 8.71210e-01, 0.00000e+00],
                #         [1.17464e+02, 1.96875e+02, 1.00145e+03, 7.11802e+02, 8.08795e-01, 0.00000e+00],
                #         [4.23969e+02, 4.30401e+02, 5.16833e+02, 7.20000e+02, 7.77376e-01, 2.70000e+01],
                #         [9.81310e+02, 3.10712e+02, 1.03111e+03, 4.19273e+02, 2.86850e-01, 2.70000e+01]])

                # Transform images with predictions from numpy arrays to base64 encoded images
                # array of original images (as np array) passed to model for inference
                results.imgs
                results.render()  # updates results.imgs with boxes and labels, returns nothing

                #for testing, display results using opencv
                """
                for img in results.imgs:
                    cv2.imshow("YoloV5 Detection", cv2.resize(img, (416, 416))[:, :, ::-1])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                """
                crops = results.crop(save=True) 
                return crops,text#results.imgs[0],text


        img = cv2.imread(filename )


        obj = CurrencyNotesDetection(
            model_name='./yolov5/runs/train/exp/weights/best.pt'
        )
        detected_labels_text = ""
        detected_img,detected_labels_text = obj.get_detected_image(img)
        if len(detected_img)>0:
            xc=detected_img[0]['im']

            xc1=xc[30:xc.shape[0]-30,:]

            xc1=xc1[:,20:xc1.shape[1]-30]


            filename='temp1.png'
            #cv2.imshow("Input ", xc1)
            #print( detected_labels_text)
            cv2.imwrite(filename ,xc1)
        else:
            
            filename='temp1.png'
            #cv2.imshow("Input ", xc1)
            #print( detected_labels_text)
            cv2.imwrite(filename ,img)
        filename='temp1.png'  
        img = cv2.imread(filename)
##        cv2.imshow("Input Image", image1)

        img_resized = cv2.resize(img, (200, 200))
        img = Image.fromarray(img_resized)

        render = ImageTk.PhotoImage(img)
        image1 = Label(image=render, borderwidth=15, highlightthickness=5, height=150, width=150, bg='pink')
        image1.image = render
        image1.place(x=250, y=90)

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        lower_green = np.array([25, 0, 20])
        upper_green = np.array([100, 255, 255])
        mask = cv2.inRange(hsv_img, lower_green, upper_green)
        result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
##        cv2.imshow("Resulted", result)

        img_resized = cv2.resize(result, (200, 200))
        img = Image.fromarray(img_resized)

        render = ImageTk.PhotoImage(img)
        image2 = Label(image=render, borderwidth=15, highlightthickness=5, height=150, width=150, bg='pink')
        image2.image = render
        image2.place(x=450, y=90)

        

        model = load_model('trained_model.h5')

        test_tensors = paths_to_tensor(filename) / 255
        pred = model.predict(test_tensors)
        print(np.max(pred))
        if np.max(pred)>.9:
            print('Given Currency is Predicted as: ' + str(clas1[np.argmax(pred)]))
            res = 'Given Currency is Predicted as: ' + str(clas1[np.argmax(pred)])
        else:
            print('Currency is not detected try again: ' )
            res = 'Currency is not detected try again: '
            # Display the predicted class in the GUI
        T = Text(self, height=5, width=40)
        T.place(x=250, y=400)
        T.insert(END, res)


    def close_window(self):
        self.master.destroy()


def path_to_tensor(img_path, width=224, height=224):
    img = image.load_img(img_path, target_size=(width, height))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)


clas1 = [item[10:-1] for item in sorted(glob("./dataset/*/"))]

root = tk.Tk()
root.geometry("1400x720")
app = Window(root)
root.mainloop()
