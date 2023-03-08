from ultralytics import YOLO
import cv2
import torch
import numpy as np

'''
Execute on CLI with "python -W ignore::UserWarning main.py"
'''

# import os
# print(os.getcwd())

print("Is CUDA available? " + str(torch.cuda.is_available()) + "\n\n")


# Prepare image
img_dir = "./ultralytics/assets/zidane.jpg"
img = cv2.imread(img_dir) # / 255.0
img = cv2.resize(img, (640, 640))
img = torch.tensor([img], dtype=torch.float).permute(0, 3, 1, 2)
img.requires_grad_()
# img = img.to("cuda")
# print(img.device)

# Run YOLO
model = YOLO("yolov8n.pt")
result, preds = model(img)
print(preds[0])
print(result[0].boxes)
print(result[0].orig_img.shape)
# print(result[0].orig_img)
result[0].orig_img = result[0].orig_img.detach().permute(0, 2, 3, 1).numpy()[0]
# cv2.imshow("image", result[0].orig_img)
# cv2.waitKey(0)


# Visualize bboxes
res_plotted = result[0].plot()
cv2.imshow("image", res_plotted)
cv2.waitKey(0)

