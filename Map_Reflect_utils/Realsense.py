import pyrealsense2 as rs
import cv2
import numpy as np
import random


class realsense(object):
    def __init__(self, num=0):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        print(num)
        ctx = rs.context()
        serial_rs = ctx.devices[num].get_info(rs.camera_info.serial_number)
        self.config.enable_device(serial_rs)
        self.width, self.height = int(), int()

    def cam_init(self, ppi=640):
        if ppi == 1080:
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.width, self.height = 1080, 720
        else:
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
            self.width, self.height = 640, 480
        self.pipeline.start(self.config)  # 流程开始
        align_to = rs.stream.color  # 与color流对齐
        align = rs.align(align_to)
        return align

    def cam_run(self, align):
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
        ############### 相机参数的获取 #######################
        # intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位），由于对齐RGB图因此变为8位
        color_image = np.asanyarray(color_frame.get_data())  # RGB图

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)  # 转换为色彩深度图

        return color_image.astype(np.uint8), depth_colormap.astype(np.uint8), depth_intrin, aligned_depth_frame

    # deal with depth frame data
    def depth_to_data(self, depth_intrin, aligned_depth_frame, mid_pos, min_val=20):
        distance_list = []
        # print(box,)
        randnum = 80
        for i in range(randnum):
            bias = random.randint(-min_val // 20, min_val // 20)
            pos1, pos2 = int(mid_pos[0] + bias), int(mid_pos[1] + bias)
            dist = aligned_depth_frame.get_distance(pos1 if pos1 < self.width else self.width,
                                                    pos2 if pos2 < self.height else self.height)
            if dist:
                distance_list.append(dist)
                if np.var(distance_list) > 3:  # 添加方差，以至于不会造成值突变
                    distance_list.pop()
        distance_list = np.array(distance_list)
        distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]
        if len(distance_list) > 0:
            mean_distance = np.nanmean(distance_list)
        else:
            mean_distance = 0
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, mid_pos,
                                                            mean_distance)  # 相机内参和深度逆矩阵运算，可得出基于相机坐标系下的，target三维信息

        x_cl = float(camera_coordinate[2])
        y_cl = float(camera_coordinate[0])
        z_cl = float(camera_coordinate[1])

        return x_cl, y_cl, z_cl


# (int(xywh[0]+xywh[2]/2),int(xywh[1]+xywh[3]/2))
if __name__ == "__main__":
    real = realsense()
    align = real.cam_init()
    cv2.namedWindow("real")
    # usage intrduction
    while True:
        color_image, depth_colormap, depth_intrin, aligned_depth_frame = real.cam_run(align=align)
        images_real = np.hstack((color_image, depth_colormap))
        mid_pos = (int(color_image.shape[1] / 2) + 50, int(color_image.shape[0] / 2))

        x, y, z = real.depth_to_data(depth_intrin, aligned_depth_frame, mid_pos, 20)
        cv2.circle(images_real, mid_pos, 2, (0, 255, 0), 2)
        cv2.imshow("real", images_real)
        print("y=%.2f m" % y)
        print("x=%.2f m" % x)
        print("z=%.2f m" % z)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyWindow("real")
