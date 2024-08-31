import cv2

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, currentFrame = cap.read()
        print("Trying to read frame...")  # 更改打印信息以更清晰地表明操作
        if success:
            cv2.imshow("current frame", currentFrame)

            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to read frame")
            break  # 如果无法读取帧，则退出循环


    # 释放摄像头并关闭所有OpenCV窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()