import glob
import numpy as np
import torch
import os
import cv2
from model.unet import unet

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = unet(n_channels=3, n_classes=1)

    net.to(device=device)

    net.load_state_dict(torch.load('best_model.pth', map_location=device))

    net.eval()

    tests_path = glob.glob('data/test/*.png')

    for test_path in tests_path:
        save_output_path = test_path.split('.')[0] + '_output.png'
        img = cv2.imread(test_path)
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= -1.1137] = 255
        pred[pred < -1.1137] = 0
        cv2.imwrite(save_output_path, pred)
