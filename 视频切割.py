import cv2
import os
import time
i=0
#请确认当前文件夹照片的数量
k=188
# video_path="source_video"
# save_paht="source_img"

video_path="source_video"
save_paht="source_img"
video_list=os.listdir(video_path)
#开启摄像头请注释下列
print(video_list)
choose=int(input("choose:"))-1

if choose <len(video_list):
    video=video_list[choose]
    cap = cv2.VideoCapture(f"source_video\\{video}")
    while True:
        try:
            ret,image=cap.read()
            cv2.imshow("img",image)
            #每张照片的相隔时间
            if i==10:
                print(1)
                cv2.imwrite(f"{save_paht}\\{k}.jpg",image)
                k+=1
                i=0
            else:
                i+=1
                print(i)
            key = cv2.waitKey(1)
            if key == 27:
                break
        except Exception:
            print(k)
            break




