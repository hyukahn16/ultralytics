from ultralytics import YOLO
from ultralytics.yolo.v8.detect.train import Loss
import cv2
import torch
import numpy as np

'''
Execute on CLI with "python -W ignore::UserWarning main.py"
'''

# import os
# print(os.getcwd())

print("Is CUDA available? " + str(torch.cuda.is_available()) + "\n\n")


model = YOLO("yolov8n.pt")

# Gather zidane.jpg ground truth labels
# img_dir = "./ultralytics/assets/zidane.jpg"
# img = cv2.imread(img_dir) # / 255.0
# result, _ = model(img)
# res_plotted = result[0].plot()

# exit()

# Prepare image
img_dir = "./ultralytics/assets/zidane.jpg"
img_dim = 320
img = cv2.imread(img_dir) # / 255.0
img = cv2.resize(img, (img_dim, img_dim))
img = torch.tensor([img], dtype=torch.float).permute(0, 3, 1, 2)
img.requires_grad_()
# img = img.to("cuda")
# print(img.device)

# Run YOLO
model = YOLO("yolov8n.pt")
result, preds = model(img)
print("# of classes: 80")
print("type of preds: ", str(type(preds)))
print("length of preds list: ", str(len(preds)))
print("shape of preds[0] tensor: ", str(preds[0].shape))
print("type of preds[1]: ", str(type(preds[1])))
print("length of preds[1] list: ", str(len(preds[1])))
print("shape of preds[1][0]", str(preds[1][0].shape))
print("shape of preds[1][1]", str(preds[1][1].shape))
print("shape of preds[1][2]", str(preds[1][2].shape))
print("---------------------------")
print("# of predicted bboxes: " + str(len(result[0].boxes)))
print(result[0].boxes[0].xywh)
print(result[0].boxes[0].cls)
print(result[0].boxes[1].cls)
print(result[0].boxes[0].conf)
print("---------------------------")
print(preds[0][0,:,0])
print(preds[0][0,4:,0])
print(torch.sum(preds[0][0,0:3,0]))
preds[0].backward()
loss = Loss(model.model)
# loss(preds, )
exit()

# Attempting to visualize bboxes
result[0].orig_img = result[0].orig_img.detach().permute(0, 2, 3, 1).numpy()[0]
# cv2.imshow("image", result[0].orig_img)
# cv2.waitKey(0)
res_plotted = result[0].plot()
cv2.imshow("image", res_plotted)
cv2.waitKey(0)

