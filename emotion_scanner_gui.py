# Code from https://realpython.com/pysimplegui-python/
# Modified by Yalu Ouyang

#? Documentation for PySimpleGUI: https://www.pysimplegui.org/en/latest/cookbook/#recipe-theme-browser

from ultralytics import YOLO
import PySimpleGUI as sg
import cv2
import numpy as np
import os
import PIL
import base64
import io

model = YOLO("C:/Users/yaluo/Desktop/Emotion Scanner/train40_gpu.pt")

path = 'C:/Users/yaluo/Desktop/Emotion Scanner/saved_img'

local_img_column = [
    [sg.Text("File selection", size=(60, 1), justification="center")],
    
    [
        sg.Text("Image File"),
        sg.In(size=(25, 2), enable_events=True, key="-FILE-"),
        sg.FileBrowse(),
    ],    
    
    [sg.Text(size=(60, 1), key="-IMAGE NAME-")],

    [sg.Image(key="-LOCAL IMAGE-", size=(300,300))],
    
    [sg.Button("Image file process", key="-LOCAL PROCESS-", disabled=True, size=(15, 2))]
]

def convert_to_bytes(file_or_bytes, resize=None):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    '''
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


def capture_window(frame):
    capture_layout = [
        # [sg.Text("Captured Snapshot", size=(60, 1), justification="center")],
        
        [sg.Image(filename="", key="-Snapshot-", size=(300,300))],
        
        [sg.InputText()],
        
        [sg.Button("Save Image", size=(15, 2))],
        
        [sg.Button("Process", size=(15, 2))],
        
        [sg.Button("Back to Main Page", size=(15, 2))],
    ]
    
    capture_window = sg.Window("Snapshot", capture_layout, modal=True) #location=(600, 100), 
    
    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    
    while True:

        event, values = capture_window.read(timeout=20)        
                
        if event == "Back to Main Page" or event == sg.WIN_CLOSED:

            break
        
        capture_window["-Snapshot-"].update(data=imgbytes)

        if event == "Save Image":
            
            if not values[0]:
                sg.popup("Enter a valid name")
                continue
            
            #! Tkinter and PySimpleGUI wants to work with .png and .gif by default
            #! Possible to accept more format but lots of more code
            
            img_name = values[0] + ".png"
            # cv2.imwrite("./saved_img/" + img_name + ".png", frame)
            
            cv2.imwrite(os.path.join(path , img_name), frame)
            print(os.path.join(path, img_name))
            sg.popup("Image saved")
            
                                
        if event == "Process":
            
            processed_results = model(frame)
            # processed_results = model("C:/Users/yaluo/Desktop/Emotion Scanner/saved_img/#face1.png")
            
            for r in processed_results:
                im_array = r.plot()
                message = r.verbose()                     
                cv2.imshow("The boxed result", im_array)
                sg.popup(message)
                # print(r)                     
                cv2.waitKey(0)
                                                        
    capture_window.close()


def main():

    sg.theme("DarkBlue")


    # Define the window layout

    layout = [
        [
        [

            sg.Column([
                [sg.Text("OpenCV Webcam footage", size=(60, 1), justification="center")],

                [sg.Image(filename="", key="-IMAGE-", size=(300,300))],

                [sg.Button("Capture", size=(10, 2))],

                [sg.Button("Exit", size=(10, 2))],
            ]),
        
            sg.VSeparator(),

            sg.Column(local_img_column)
        ]            
            
        ]
    ]
    
    # Create the window and show it without the plot

    window = sg.Window("OpenCV Integration", layout) #, location=(800, 200)

    cap = cv2.VideoCapture(1) #! 0 is internal webcam, 1 works for usb cam
    
    while True:

        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:

            break

        ret, frame_raw = cap.read()

        #! flips the frame so that it matches movement in front of webcam
        frame = cv2.flip(frame_raw, 1)

        #* Resizing the camera feed to make it look better on the layout
        #* original size is 640 x 480
        frame = cv2.resize(frame, (576,432))
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()

        window["-IMAGE-"].update(data=imgbytes)
        
        if event == "Capture":
                                
            capture_window(frame)
            
        elif event == "-FILE-":
            chosen_file = values["-FILE-"]
            
            print(chosen_file)
            
            # find acceptable image files        
            
            try:                    
            
                window["-IMAGE NAME-"].update(chosen_file)
                # window["-LOCAL IMAGE-"].update(data=convert_to_bytes(chosen_file, (400,400))) #576,432
                window["-LOCAL IMAGE-"].update(data=convert_to_bytes(chosen_file, (576,432)))
                window["-LOCAL PROCESS-"].update(disabled=False)
            except:
                pass                
            
        elif event == "-LOCAL PROCESS-":
            print(values)
            local_process_name = values["-FILE-"]
            
            # print(local_process_name)
            # print("Stuck at this step:")
            # print("-------------------")
            # print("-------------------")
            # print("-------------------")
                        
            processed_results = model(local_process_name)
            
            
            for r in processed_results:
                im_array = r.plot()
                message = r.verbose()                     
                cv2.imshow("The boxed result", im_array)
                sg.popup(message)
                # print(r)                     
                cv2.waitKey(0)

    window.close()


main()

"""
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
"""