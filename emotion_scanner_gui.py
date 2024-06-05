#? Documentation for PySimpleGUI: https://www.pysimplegui.org/en/latest/cookbook/#recipe-theme-browser

from ultralytics import YOLO, solutions
# reminds me of the football club
import PySimpleGUI as psg
import cv2
import numpy as np
import os
import PIL
import base64
import io

#! Default model
model = YOLO("C:/Users/yaluo/Desktop/Emotion Scanner/train40_gpu.pt")

path = 'C:/Users/yaluo/Desktop/Emotion Scanner/saved_img'

CAM_DEVICE = 0

local_img_column = [
    [psg.Text("File selection", size=(60, 1), justification="center")],
    
    [
        psg.Text("Image File"),
        psg.In(size=(25, 2), enable_events=True, key="-FILE-"),
        psg.FileBrowse(),
    ],    
    
    [psg.Text(size=(60, 1), key="-MODEL NAME-")],

    [psg.Image(key="-LOCAL IMAGE-", size=(300,300))],
    
    [psg.Button("Image file process", key="-LOCAL PROCESS-", disabled=True, size=(15, 2))]
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
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.LANCZOS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


def capture_window(frame):
    """This window is dedicated to capturing snapshots from camera devices.
    
    Has the ability to save those snapshots as actual image files for processing.

    Args:
        frame (_type_): _description_
    """
    capture_layout = [
        # [psg.Text("Captured Snapshot", size=(60, 1), justification="center")],
        
        [psg.Image(filename="", key="-Snapshot-", size=(300,300))],
        
        [psg.InputText()],
        
        [psg.Button("Save Image", size=(15, 2))],
        
        [psg.Button("Process", size=(15, 2))],
        
        [psg.Button("Back to Scan", size=(15, 2))],
    ]
    
    capture_window = psg.Window("Snapshot", capture_layout, modal=True) #location=(600, 100), 
    
    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    
    while True:

        event, values = capture_window.read(timeout=20)        
                
        if event == "Back to Scan" or event == psg.WIN_CLOSED:

            break
        
        capture_window["-Snapshot-"].update(data=imgbytes)

        if event == "Save Image":
            
            if not values[0]:
                psg.popup("Enter a valid name")
                continue
            
            #! Tkinter and PySimpleGUI wants to work with .png and .gif by default
            #! Possible to accept more format but lots of more code
            
            img_name = values[0] + ".png"
            # cv2.imwrite("./saved_img/" + img_name + ".png", frame)
            
            cv2.imwrite(os.path.join(path , img_name), frame)
            print(os.path.join(path, img_name))
            psg.popup("Image saved")
            
                                
        if event == "Process":
            
            processed_results = model(frame)
            # processed_results = model("C:/Users/yaluo/Desktop/Emotion Scanner/saved_img/#face1.png")
            
            for r in processed_results:
                im_array = r.plot()
                message = r.verbose()                     
                cv2.imshow("The boxed result", im_array)
                psg.popup(message)
                # print(r)                     
                cv2.waitKey(0)
                                                        
    capture_window.close()


def scan_window():

    # Define the window layout

    layout = [
        psg.Column([
                [psg.Text("Cam footage", size=(60, 1), justification="center")],

                [psg.Image(filename="", key="-SCAN IMAGE-", size=(300,300))],

                [psg.Button("Capture", size=(10, 2))],

                [psg.Button("Back to Main Page", size=(10, 2))],
        ])
    ]
    
    # Create the window and show it without the plot

    window = psg.Window("Inventory Footage", layout) #, location=(800, 200)

    #! 0 is internal webcam, 1 works for usb cam
    #! usb cam is 1280x720 (16:9), webcam is 4:3.
    # usb cam takes longer than internal cam
    cap = cv2.VideoCapture(CAM_DEVICE)
    
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define line points (w, h)
    line_points = [(20, 400), (500, 400)]

    # Init Object Counter
    counter = solutions.ObjectCounter(
        view_img=True,
        reg_pts=line_points,
        classes_names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    
    while True:

        event, values = window.read(timeout=20)

        if event == "Back to Main Page" or event == psg.WIN_CLOSED:

            break

        ret, frame_raw = cap.read()
        
        if not ret:
            print("frame isn't available")
            break

        #! flips the frame so that it matches movement in front of webcam
        # not needed if using external camera
        frame = cv2.flip(frame_raw, 1)
        # frame = frame_raw

        #* Resizing the camera feed to make it look better on the layout
        #* original size is 640 x 480
        frame = cv2.resize(frame, (600,450))
        
        tracks = model.track(frame, agnostic_nms=True, persist=True, show=False)
        tracked_im = tracks[0].plot()
        # frame = counter.start_counting(frame, tracks)
        
        imgbytes = cv2.imencode(".png", tracked_im)[1].tobytes()
        # imgbytes = cv2.imencode(".png", frame)[1].tobytes()

        window["-SCAN IMAGE-"].update(data=imgbytes)
        
        if event == "Capture":
                                
            capture_window(frame)
            

    window.close()


def main():

    psg.theme("DarkBlue")


    # Define the window layout

    layout = [
        [

            psg.Column([
                [psg.Text("Cam footage", size=(60, 1), justification="center")],

                [psg.Image(filename="", key="-IMAGE-", size=(300,300))],

                [psg.Button("Capture", size=(10, 2))],

                [psg.Button("Exit", size=(10, 2))],
            ]),
        
            psg.VSeparator(),

            psg.Column(local_img_column)
        ]            

    ]
    
    # Create the window and show it without the plot

    window = psg.Window("Inventory Footage", layout) #, location=(800, 200)

    #! 0 is internal webcam, 1 works for usb cam
    #! usb cam is 1280x720 (16:9), webcam is 4:3.
    # usb cam takes longer than internal cam
    cap = cv2.VideoCapture(0)
    
    while True:

        event, values = window.read(timeout=20)

        if event == "Exit" or event == psg.WIN_CLOSED:

            break

        ret, frame_raw = cap.read()
        
        if not ret:
            print("frame isn't available")
            break

        #! flips the frame so that it matches movement in front of webcam
        # not needed if using external camera
        frame = cv2.flip(frame_raw, 1)
        # frame = frame_raw

        #* Resizing the camera feed to make it look better on the layout
        #* original size is 640 x 480
        frame = cv2.resize(frame, (600,450))
            
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()

        window["-IMAGE-"].update(data=imgbytes)
        
        if event == "Capture":
                                
            capture_window(frame)
            
        elif event == "-FILE-":
            chosen_file = values["-FILE-"]
            
            print(f"chosen_file: {chosen_file}")
            print(f"chosen_file type: {type(chosen_file)}")
            
            # find acceptable image files        
            
            try:                    
            
                window["-MODEL NAME-"].update(chosen_file)
                # window["-LOCAL IMAGE-"].update(data=convert_to_bytes(chosen_file, (400,400))) #576,432
                window["-LOCAL IMAGE-"].update(data=convert_to_bytes(chosen_file, (576,432)))
                window["-LOCAL PROCESS-"].update(disabled=False)
            except Exception as e:
                psg.popup(e)
                pass
            
        elif event == "-LOCAL PROCESS-":
            
            local_process_name = values["-FILE-"]                        
            processed_results = model(local_process_name)
                        
            for r in processed_results:
                im_array = r.plot()
                message = r.verbose()                     
                cv2.imshow("The boxed result", im_array)
                psg.popup(message)
                # print(r)                     
                cv2.waitKey(0)

    window.close()


main()