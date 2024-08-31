# 此处逻辑会开机自动启动这个代码，启动代码后五分钟后自动停止代码，同时识别的数字会记录到日志中，识别的视频会保存下来以供后续发现问题
# 此处还会建立文件夹使得识别到的天井照片储存下来
# 处理过后的天井照片也会储存下来`
import logging
import time
import datetime
import os
from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO

origin_num = 0
rotate_num = 0
crop_num = 0

now = datetime.datetime.now()

folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists(f"/home/amov/Desktop/well{folder_name}"):
    os.mkdir(f"/home/amov/Desktop/well{folder_name}")
    os.mkdir(f"/home/amov/Desktop/well{folder_name}/origin")
    os.mkdir(f"/home/amov/Desktop/well{folder_name}/rotate")
    os.mkdir(f"/home/amov/Desktop/well{folder_name}/crop")


# 定义红色在HSV色彩空间的范围
lower_red1 = np.array([0, 50, 50])  # np.array([0, 120, 70])  # np.array([0, 20, 50])  #
upper_red1 = np.array([15, 255, 255])  # np.array([10, 255, 255])  # np.array([20, 255, 255])  #
lower_red2 = np.array([150, 50, 50])  # np.array([170, 120, 70])  # np.array([150, 20, 50])  #
upper_red2 = np.array([180, 255, 255])  # np.array([180, 255, 255])  # np.array([180, 255, 255])  #

# 定义更明亮的粉色区间
lower_pink = np.array([320, 150, 150])  # 提高饱和度和亮度的下限
upper_pink = np.array([360, 255, 255])  # 保持色调上限

# 定义保存视频的格式和编码，这里使用MP4编码
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 创建VideoWriter对象，指定输出文件名、编码器、帧率和分辨率
out = cv2.VideoWriter(f"/home/amov/Desktop/well{folder_name}/output.mp4", fourcc, 5.0, (1920, 1080))

logging.basicConfig(filename=f"/home/amov/Desktop/well{folder_name}/output.log", level=logging.INFO, format='')
start_time = time.time()

# 有两处调节亮度的操作 搜索clip
classes_names = {0: '00', 1: '01', 2: '02', 3: '03', 4: '04', 5: '05', 6: '06', 7: '07', 8: '08', 9: '09', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15', 16: '16', 17: '17', 18: '18', 19: '19', 20: '20', 21: '21', 22: '22', 23: '23', 24: '24', 25: '25', 26: '26', 27: '27', 28: '28', 29: '29', 30: '30', 31: '31', 32: '32', 33: '33', 34: '34', 35: '35', 36: '36', 37: '37', 38: '38', 39: '39', 40: '40', 41: '41', 42: '42', 43: '43', 44: '44', 45: '45', 46: '46', 47: '47', 48: '48', 49: '49', 50: '50', 51: '51', 52: '52', 53: '53', 54: '54', 55: '55', 56: '56', 57: '57', 58: '58', 59: '59', 60: '60', 61: '61', 62: '62', 63: '63', 64: '64', 65: '65', 66: '66', 67: '67', 68: '68', 69: '69', 70: '70', 71: '71', 72: '72', 73: '73', 74: '74', 75: '75', 76: '76', 77: '77', 78: '78', 79: '79', 80: '80', 81: '81', 82: '82', 83: '83', 84: '84', 85: '85', 86: '86', 87: '87', 88: '88', 89: '89', 90: '90', 91: '91', 92: '92', 93: '93', 94: '94', 95: '95', 96: '96', 97: '97', 98: '98', 99: '99'}
maxdet = 4
max_num = 4
# 定义yolo权重文件路径
MODELOBB = "/home/amov/ultralytics-main/yolo_obb_red.pt"
MODELCLASSIFY = "/home/amov/ultralytics-main/yolo_cls.pt"
print("loading obb model")
modelObb = YOLO(MODELOBB)  # 通常是pt模型的文件
print("loading classify model")
modelClassify = YOLO(MODELCLASSIFY)

# 定义调用摄像头索引
CAMERAINDEX = 0

streamInput = True

# 在模块顶层声明全局变量
num_list = []

# 拉普拉斯算子，用来锐化图像
kernel1 = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])

def auto_rotate(img):  # 此处image是numpy类型
    image = img.copy()

    # 调节图像亮度
    image = (image * 0.7).clip(0, 255).astype(np.uint8)

    # 对图像进行锐化的处理
    # image = cv2.filter2D(image, -1, kernel=kernel1)

    crop_img = image[420:480, 140:180]
    # 将图像转换为HSV色彩空间
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # 创建红色五边形的掩膜
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # 创建更明亮的粉色的掩膜
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

    # 将红色和粉色的掩膜合并
    mask = cv2.bitwise_or(mask_red, mask_pink)

    # 进行形态学操作以去除噪声
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 进行开运算（先腐蚀后膨胀）通过判断上边是白色或黑色来翻转图片

    if np.sum(np.sum(mask)) < 60 * 40 * 255 * 0.6:
        return image
    else:
        image = cv2.rotate(image, cv2.ROTATE_180)
        return image



def apply_num_rec_package(rawImage):
    if rawImage is not None:
        # 对图像进行裁切
        x1, y1 = 30, 280  # 80, 390
        x2, y2 = 320, 600  # 220, 520
        crop_image = rawImage[y1: y2, x1: x2]
        # 调节图像亮度(需要较亮的环境)
        # crop_image = (crop_image * 0.5).clip(0, 255).astype(np.uint8)
        # cv2.imshow("crop image", crop_image)
        # 腐蚀函数
        '''kernel = np.ones((10, 10), np.uint8)
        binary_img = cv2.dilate(binary_img, kernel, iterations=3)'''
        global crop_num
        crop_num += 1
        cv2.imwrite(f"/home/amov/Desktop/well{folder_name}/crop/{crop_num}.png", crop_image)
        # print(rawImage.shape[0], rawImage.shape[1])  # 打印的信息就是函数croptarget的宽度和高度
        results_classify = modelClassify.predict(
                            source=crop_image,
                            imgsz=640,
                            device='0',
                            half=True,
                            iou=0.4,
                            conf=0.5,
                            save=False
                            )
        num = classes_names[results_classify[0].probs.top1]
        # print(results_classify[0])
        print("Classify num is:" + num + '   ' + str(time.time() - start_time))
        
        logging.info("Classify num is:" + num + '   ' + str(time.time() - start_time))
        num_list.append(num)
        return rawImage


def cropTarget(rawImage, cropTensor, width, height):
    # 将Tensor转换为列表(该列表内有四个元素，每一个元素是一个坐标)
    cropTensorList = cropTensor.tolist()

    # 检查列表长度是否为4，如果不是，则可能存在问题
    if len(cropTensorList) != 4:
        raise ValueError("cropTensor must contain exactly 4 elements")

    # 根据条件选择不同的点集合
    if (cropTensorList[0][0] - cropTensorList[1][0]) ** 2 + (cropTensorList[0][1] - cropTensorList[1][1]) ** 2 > (
            cropTensorList[1][0] - cropTensorList[2][0]) ** 2 + (cropTensorList[1][1] - cropTensorList[2][1]) ** 2:
        rectPoints = np.array([cropTensorList[0], cropTensorList[1], cropTensorList[2], cropTensorList[3]],
                              dtype=np.float32)
    else:
        rectPoints = np.array([cropTensorList[3], cropTensorList[0], cropTensorList[1], cropTensorList[2]],
                              dtype=np.float32)

    dstPoints = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

    affineMatrix = cv2.getAffineTransform(rectPoints[:3], dstPoints[:3])

    return cv2.warpAffine(rawImage, affineMatrix, (width, height))


def most_common_four_strings(strings):
    # 使用Counter计算每个字符串的出现次数
    count = Counter(strings)
    # 获取出现次数最多的四个字符串及其次数
    most_common = count.most_common(max_num)
    # 提取字符串
    result = [item[0] for item in most_common]
    return result

def obb_predict(frame):
    results_obb = modelObb.predict(
        source=frame,
        imgsz=640,  # 此处可以调节
        half=True,
        iou=0.4,
        conf=0.7,
        device='0',  # '0'使用GPU运行
        max_det=maxdet,
        save=False
        # augment = True
    )
    return results_obb[0]

def plot(result):
    try:
        annotatedFrame = result.plot()  # 获取框出的图像
        cropTensors = result.obb.xyxyxyxy.cpu()  # 矩形的四个坐标
        # cv2.imshow("target", annotatedFrame)
        # 将帧写入视频文件
        out.write(annotatedFrame)
    except AttributeError:
        print("No result.obb, maybe you have used a classify model")
    return cropTensors

def cls_predict():
    cropTensors = plot(result)
    for j, cropTensor in enumerate(cropTensors):
        framet = cropTarget(result.orig_img, cropTensor, 320, 640)
        if framet is not None and framet.size != 0:
            global origin_num
            origin_num += 1
            cv2.imwrite(f"/home/amov/Desktop/well{folder_name}/origin/{origin_num}.png", framet)
            # cv2.imshow("five", framet)
            framet = auto_rotate(framet)
            if framet is not None and framet.size != 0:
                global rotate_num
                rotate_num += 1
                cv2.imwrite(f"/home/amov/Desktop/well{folder_name}/rotate/{rotate_num}.png", framet)
                img_num = apply_num_rec_package(framet)
                #cv2.imshow("img_num" + str(j), img_num)
            else:
                continue
        else:
            continue

        cv2.waitKey(1)

skip_interval_no_object = 2
skip_interval_with_object = 2
i = 0 # i记录总帧数
flag_skip = True
threshold = 10
threshold_all = 200
threshold_num = 0

if streamInput:

    for index in range (0, 99):
        cap = cv2.VideoCapture(index)
        if cap.isOpened() is True:
            break
        else:
            print(f"index{index} doesn't exit")
    
    cap.set(cv2.CAP_PROP_FPS, 330)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frameFps = cap.get(cv2.CAP_PROP_FPS)  # 帧率

    print(frameWidth, frameHeight, frameFps)

    while cap.isOpened():
        success, frame = cap.read()
        end_time = time.time()
        if success == True:
            i += 1
            if end_time - start_time > 480:
                break
            # print("the threshold_num is" + str(threshold_num))
            if flag_skip is True and i % skip_interval_no_object == 0:  # and threshold_all >= threshold_num:
                result = obb_predict(frame)
                if result.obb.cls.numel() > 0:
                    # print("flag_skip is False")
                    flag_skip = False
                if result.obb.cls.numel() == 0:
                    threshold_num += 1
                # print(result)
                # 按 'q' 键退出循环(必须有要不然没有画面 目前不知道为什么)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                cls_predict()  # 实现数字摆正与数字识别

            elif flag_skip is False and i % skip_interval_with_object == 0:
                result = obb_predict(frame)
                if result.obb.cls.numel() == 0:
                    threshold_num += 1
                    # print("threshold += 1")
                if threshold_num > threshold:
                    flag_skip = True
                    # print("flag_skip is True")
                    threshold_num = 0
                # print(result)
                # 按 'q' 键退出循环(必须有要不然没有画面 目前不知道为什么)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                cls_predict()  # 实现数字摆正与数字识别
            else:
                continue


        else:
            break

                # print("sleeping……")

    cap.release()
    cv2.destroyAllWindows()

# end_time = time.time()
# print(f"方法执行时间: {end_time - start_time:.6f} 秒") # 输出执行时间，保留6位小数
print("the final number is" + str(most_common_four_strings(num_list)))
logging.info("the final number is" + str(most_common_four_strings(num_list)))
print("finished!")

