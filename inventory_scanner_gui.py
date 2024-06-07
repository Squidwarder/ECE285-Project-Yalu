#? Documentation for PySimpleGUI: https://www.pysimplegui.org/en/latest/cookbook/#recipe-theme-browser

from ultralytics import YOLO, solutions
import PySimpleGUI as psg   # reminds me of the football club
import cv2
import numpy as np
import os

#! Default model
model = YOLO("models/model1/best_models_labels/model1_train6_best.pt")
path = 'C:/Users/yaluo/UCSD_Course/ECE285/Project/saved_img'

CAM_DEVICE = 0
#! 0 is internal webcam, 1 works for usb cam
#! usb cam is 1280x720 (16:9), webcam is 4:3.
# usb cam takes longer than internal cam
resize_dct = {0:(400, 300), 1:(400, 25)}

def capture_window(frame):
    """This window is dedicated to capturing snapshots from camera devices.
    
    Has the ability to save those snapshots as actual image files for processing.
    """
    capture_layout = [
        # [psg.Text("Captured Snapshot", size=(60, 1), justification="center")],
        
        [psg.Image(filename="", key="-Snapshot-", size=(300,300))],
        
        [psg.InputText(default_text="Image name without extension")],
        
        [psg.Button("Save Image", size=(15, 2))],
        
        [psg.Button("Process", size=(15, 2))],
        
        [psg.Button("Back to Scan", size=(15, 2))],
    ]
    
    capture_window = psg.Window("Snapshot", capture_layout, modal=True) #location=(600, 100), 
    
    #! PySimpleGUI only displays images in PNG, GIF, PPM/PGM formats, so NEED to
    #! convert frames taken using cv2.VideoCapture() into supported encodings.
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
            
            cv2.imwrite(os.path.join(path , img_name), frame)
            print(f"saved to : {os.path.join(path, img_name)}")
            psg.popup("Image saved")
            
                                
        if event == "Process":
            
            processed_results = model(frame)
            
            for r in processed_results:
                im_array = r.plot()
                message = r.verbose()                     
                cv2.imshow("NN results on Snapshot", im_array)
                psg.popup(message)
                # print(r)                     
                cv2.waitKey(0)
                                                        
    capture_window.close()


def scan_window(cap):
    #! cap is videoCapture passed on from main()

    layout = [
        
        [psg.Text("Cam footage", size=(60, 1), justification="center")],

        [psg.Image(filename="", key="-SCAN IMAGE-", size=(300,300))],

        [psg.Button("Capture", size=(10, 2))],

        [psg.Button("Back to Main Page", size=(10, 2))]
        
    ]
    
    # Create the window and show it without the plot
    scan_window = psg.Window("Scanning Inventory", layout) #, location=(800, 200)

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

        event, values = scan_window.read(timeout=20)

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
        frame = cv2.resize(frame, resize_dct[CAM_DEVICE])
        
        tracks = model.track(frame, agnostic_nms=True, persist=True, show=False)
        tracked_im = tracks[0].plot()
        # frame = counter.start_counting(frame, tracks)
        
        imgbytes = cv2.imencode(".png", tracked_im)[1].tobytes()
        # imgbytes = cv2.imencode(".png", frame)[1].tobytes()

        scan_window["-SCAN IMAGE-"].update(data=imgbytes)
        
        if event == "Capture":
                                
            capture_window(frame)
            

    scan_window.close()

local_img_column = [
    [psg.Text("Image selection", size=(60, 1), justification="center")],
    
    [
        psg.Text("Image File"),
        psg.In(size=(25, 2), enable_events=True, key="-FILE-"),
        psg.FileBrowse(),
    ],    
    
    [psg.Text(size=(60, 1), key="-LOC IMG NAME-")],

    [psg.Image(key="-LOCAL IMAGE-", size=(300,300))],
    
    [psg.Button("Image file process", key="-LOCAL PROCESS-", disabled=True, size=(15, 2))]
]

def main():

    psg.theme("DarkBlue")

    # Define the window layout
    layout = [
        [
            psg.Column([
                [psg.Text("Cam footage", size=(60, 1), justification="center")],
                [psg.Image(filename="", key="-IMAGE-", size=(200,200))],
                
                [
                    psg.Text("Choose Model (.pt)"),
                    psg.In(size=(25, 2), enable_events=True, key="-NN NAME-"),
                    psg.FileBrowse(),
                ],
                [psg.Multiline("""Model file can be found under /models/model#/best_models_labels/model#.pt. Corresponding labels will be displayed here""",
                    size=(50, 10), key="-NN LABELS-")],

                [psg.Button("Scan", size=(10, 2))],
                [psg.Button("Exit", size=(10, 2))],
            ]),        
            psg.VSeparator(),
            psg.Column(local_img_column)
        ]            

    ]
    
    # Create the window and show it without the plot
    window = psg.Window("Inventory Scanner", layout) #, location=(800, 200)

    cap = cv2.VideoCapture(CAM_DEVICE)
    
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
        frame = cv2.resize(frame, resize_dct[CAM_DEVICE])
            
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()

        window["-IMAGE-"].update(data=imgbytes)
        
        if event == "Scan":
            scan_window(cap)
            
        elif event == "-FILE-":
            chosen_file = values["-FILE-"]
            
            print(f"chosen_file: {chosen_file}")
            print(f"chosen_file type: {type(chosen_file)}")
            
            # find acceptable image files
            chosen_file_img = cv2.imread(chosen_file)
            chosen_file_img = cv2.resize(chosen_file_img, resize_dct[CAM_DEVICE])
            chosen_file_img = cv2.imencode(".png", chosen_file_img)[1].tobytes()
            
            try:                    
            
                window["-LOC IMG NAME-"].update(chosen_file)
                window["-LOCAL IMAGE-"].update(data=chosen_file_img)
                window["-LOCAL PROCESS-"].update(disabled=False)
            except Exception as e:
                psg.popup(e)
                pass
            
        elif event == "-NN NAME-":
            chosen_model = values["-NN NAME-"]
            
            model_dir = os.path.dirname(chosen_model)
            label_files = [f for f in os.listdir(model_dir) if f.endswith('label.txt')]
            label_file = os.path.join(model_dir, label_files[0])
            global model    #! Updating global variable model
            model = YOLO(chosen_model)
            
            # print(f"chosen_model: {chosen_model}")
            # print(f"chosen_model dir: {model_dir}")
            # print(f"label files[0]: {label_files[0]}")
            # print(f"label file: {label_file}")
            
            try:                    
                with open(label_file, 'r') as f2:
                    window["-NN LABELS-"].update(f2.read())
            except Exception as e:
                psg.popup(e)
                pass
        
        elif event == "-LOCAL PROCESS-":
            
            local_process_name = values["-FILE-"]
            processed_results = model(local_process_name)
                        
            for r in processed_results:
                im_array = r.plot()
                message = r.verbose()
                im_array = cv2.resize(im_array, resize_dct[CAM_DEVICE])
                cv2.imshow("NN Detection results", im_array)
                psg.popup(message)
                # print(r)                     
                cv2.waitKey(0)

    window.close()


main()