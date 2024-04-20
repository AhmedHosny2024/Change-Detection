import torch
import torchvision 
from torchvision import transforms as T
import cv2

coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , 
"truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" ,
"bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , 
"hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" , 
"frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
"baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
"plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
"banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
"pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
"mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
"laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
"oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
"clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]

# Load the model
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

# Load the image
image = cv2.imread('trainval/B/0069.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = T.ToTensor()(image)

# Get the prediction
with torch.no_grad():
    prediction = model([image])

for element in prediction:
    boxes, labels, scores = element['boxes'], element['labels'], element['scores']
    res=torch.argwhere(scores > 0.1)

    boxes=boxes[res]
    labels=labels[res]
    scores=scores[res]
    # Draw the bounding boxes
    image_np = image.mul(255).permute(1, 2, 0).byte().numpy()
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    for box, label, score in zip(boxes, labels, scores):
        x, y, x2, y2 = box[0]
        #convert the tensor to numpy
        x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
        cv2.rectangle(image_np, (x, y), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_np, coco_names[label], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("image", image_np)
    cv2.waitKey(0)
