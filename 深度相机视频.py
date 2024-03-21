import pyrealsense2 as rs
import cv2
import numpy as np

if __name__ == "__main__":
    KEY = cv2.waitKey(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    pipeline = rs.pipeline()  # 定义流程pipeline
    config = rs.config()  # 定义配置config
    config.enable_stream(rs.stream.color, 1280, 720 , rs.format.bgr8, 30)  # 配置color流
    pipeline.start(config)  # 流程开始
    align_to = rs.stream.color  # 与color流对齐
    align = rs.align(align_to)
    out = cv2.VideoWriter("source_video/RGB1.avi", fourcc, 30, (1280, 720))
    while True:
        frames = pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = align.process(frames)  # 获取对齐帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
        ############### 相机参数的获取 #######################
        color_image = np.asanyarray(color_frame.get_data())
        out.write(color_image)
        cv2.imshow("image", color_image)
        KEY = cv2.waitKey(1)
        if KEY == 27:
            break
