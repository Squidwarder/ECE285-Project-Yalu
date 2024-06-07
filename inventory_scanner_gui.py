#? Documentation for PySimpleGUI: https://www.pysimplegui.org/en/latest/cookbook/#recipe-theme-browser

from ultralytics import YOLO, solutions
import PySimpleGUI as psg   # reminds me of the football club
import cv2
import numpy as np
import os

#! Default model
model = YOLO("models/model1/best_models_labels/model1_train6_best.pt")
model_path = 'C:/Users/yaluo/UCSD_Course/ECE285/Project/models'
path = 'C:/Users/yaluo/UCSD_Course/ECE285/Project/saved_img'

CAM_DEVICE = "usb_cam"
#! usb cam is 1280x720 (16:9), webcam is 4:3.
#! dimensions in (w, h)
# usb cam takes longer than internal cam
resize_dct = {"webcam":(400, 300), "usb_cam":(400, 225)}

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
        
        [psg.Text("Cam footage", size=(30, 1), justification="center", font=("Arial", 16, "bold"))],

        [psg.Image(filename="", key="-SCAN IMAGE-", size=(300,300))],
        
        [psg.Multiline(size=(30, 12), key="-SCAN MSG-")],

        [psg.Button("Capture", size=(10, 2))],

        [psg.Button("Back to Main Page", size=(10, 2))]
        
    ]
    
    # Create the window and show it without the plot
    scan_window = psg.Window("Scanning Inventory", layout) #, location=(800, 200)
    
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
        if CAM_DEVICE == "webcam":
            frame = cv2.flip(frame_raw, 1)
        else:
            frame = frame_raw

        #* Resizing the camera feed to make it look better on the layout
        #* original size is 640 x 480
        frame = cv2.resize(frame, resize_dct[CAM_DEVICE])
        
        tracks = model.track(frame, agnostic_nms=True, persist=True, show=False)
        tracked_im = tracks[0].plot()
        message = tracks[0].verbose()
        
        imgbytes = cv2.imencode(".png", tracked_im)[1].tobytes()

        scan_window["-SCAN IMAGE-"].update(data=imgbytes)
        message = message.replace(",", "\n")
        scan_window["-SCAN MSG-"].update(value=message, text_color="#99ffee", font=("Arial", 16, "bold"))
        
        if event == "Capture":
                                
            capture_window(frame)
            

    scan_window.close()

local_img_column = [
    [psg.Text("Image selection", size=(30, 1), justification="center", font=("Arial", 16, "bold"))],
    
    [
        psg.Text("Image File"),
        psg.In(size=(25, 2), enable_events=True, key="-FILE-"),
        psg.FileBrowse(file_types=(("PNG", ".png"), ("JPG1", ".JPG"), ("jpg2", ".jpg"), ("JPEG", ".jpeg")),
                       initial_folder=path),
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
                [psg.Text("Cam footage", size=(30, 1), justification="center", font=("Arial", 16, "bold"))],
                [psg.Image(filename="", key="-START CAM-", size=(200,200))],
                
                [
                    psg.Text("Choose Model (.pt)"),
                    psg.In(size=(25, 2), enable_events=True, key="-NN NAME-"),
                    psg.FileBrowse(file_types=(("PT files", ".pt"),), initial_folder=model_path+"/model1/best_models_labels"),
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

    #! whenever the usb camera is plugged in, it always to device 0.
    #! webcam also takes device 0 when by itself.
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
        if CAM_DEVICE == "webcam":
            frame = cv2.flip(frame_raw, 1)
        else:
            frame = frame_raw

        #* Resizing the camera feed to make it look better on the layout
        #* original size is 640 x 480
        frame = cv2.resize(frame, resize_dct[CAM_DEVICE])
            
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()

        window["-START CAM-"].update(data=imgbytes)
        
        if event == "Scan":
            scan_window(cap)
            
        elif event == "-FILE-":
            chosen_file = values["-FILE-"]
            
            # print(f"chosen_file: {chosen_file}")
            # print(f"chosen_file type: {type(chosen_file)}")
            
            # find acceptable image files
            chosen_file_img = cv2.imread(chosen_file)
            # always display local image in 4:3 aspect ratio
            chosen_file_img = cv2.resize(chosen_file_img, resize_dct["webcam"])
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
            # print(f"processed_results: {processed_results}")
            
            for r in processed_results:
                im_array = r.plot()
                message = r.verbose()
                message = message.replace(",", "\n")
                im_array = cv2.resize(im_array, resize_dct[CAM_DEVICE])
                cv2.imshow("NN Detection results", im_array)                
                psg.popup(message)
                # print(r)                     
                cv2.waitKey(0)

    window.close()


main()