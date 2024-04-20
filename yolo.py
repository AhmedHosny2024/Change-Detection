from ultralytics import YOLO
import cv2
import os

model=YOLO('yolov8s.pt')

def detect_building(path):
  image = cv2.imread(path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # apply gaussian blure to the image
  image = cv2.GaussianBlur(image, (3, 3), 0)

  directory = "output"
  filename = os.path.splitext(os.path.basename(path))[0]


  result = model(source=image, show=True, conf=0.4, save=True, project=directory, name=filename)


image_dir='trainval/B'
i=0
with os.scandir(image_dir) as entries:
  image_dir = [entry.path for entry in entries if entry.is_file()]
  for image in image_dir:
    if i==50:
      break
    detect_building(image)
    i+=1

# detect_building("trainval/B/0053.png")
# fix the path to the image
#loop on the output take the folder inside it and get theimage then rename it with the folder name and save it as file in the output folder
# then delete the folder
output = "output"
with os.scandir(output) as entries:
  output_dir = [entry.path for entry in entries if entry.is_dir()]
  for folder in output_dir:
    foldername=folder
    with os.scandir(folder) as entries:
      folder = [entry.path for entry in entries if entry.is_file()]
      for image in folder:
        filename = os.path.basename(folder[0])
        os.rename(image, f"{foldername}"+".png")

    os.rmdir(foldername)
    
        