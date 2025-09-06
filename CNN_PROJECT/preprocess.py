import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

def img_convert(data: np.ndarray) -> torch.Tensor:
    # 轉灰階（必要時）
    if len(data.shape) == 3:
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    else:
        gray = data
    
    
    mean_val = np.mean(gray)
    if(mean_val) > 140 :
        gray = cv2.bitwise_not(gray)

    # 使用 bilateral filter 去雜訊，保留邊緣
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 直方圖均衡化，增強對比度（讓黑白更分明）
    enhanced = cv2.equalizeHist(blurred)
    
    # 使用自適應二值化（根據區域亮度判斷黑白）


    # 可選：細小膨脹，補回細線段（如果文字很淡很細）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(enhanced, kernel, iterations=1)

    _, binary_image = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY)

    # 轉為 PIL 圖片以便 torchvision 處理
    rgb_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(rgb_image)

    # 圖片轉換流程（Resize → ToTensor → Grayscale）
    img_size = (28, 28)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Grayscale(1)  # 保留單通道
    ])

    # 可視化預覽（Debug 用）
    # image.show()

    return transform(image)
