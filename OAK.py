import depthai as dai
import cv2
import numpy as np

class OAK(object):
    def __init__(self):
        self.pipeline=dai.Pipeline()#oak init pipeline
        self.size_gobal=(1920,1440)

    def getFrame(self,queue):
        frame = queue.get()
        # cv2.imshow("frame1",frame.getCvFrame().astype(np.uint16))
        # Convert frame to OpenCV format and return
        return frame.getCvFrame().astype(np.uint16)

    def getMonoCamera(self,isLeft):
        # Configure mono camera
        mono = self.pipeline.createMonoCamera()  # creat camera local variables
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

        # Set Camera Resolution
        # mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        if isLeft:
            # Get left camera
            mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
        else:
            # Get right camera
            mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        return mono #camera cofig

    def getStereoPair(self, monoLeft, monoRight,Size):

        # Configure stereo pair for depth estimation
        stereo = self.pipeline.createStereoDepth()

        #zoom detect
        # spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)

        # Checks occluded pixels and marks them as invalid
        stereo.setLeftRightCheck(True)

        # Configure left and right cameras to work as a stereo pair

        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        monoRight.setFps(35)
        monoLeft.setFps(35)

        return stereo


    def Run_RGB(self,device):
        self.video = device.getOutputQueue(name="video", maxSize=1, blocking=True)
        videoIn = self.video .get()
        videoIn=videoIn.getCvFrame()

        return videoIn



    def Run_depth(self,device,stereo,coord):

            depthQueue = device.getOutputQueue(name="depth",
                                                       maxSize=1, blocking=False)
            depthMultiplier = 255 / stereo.initialConfig.getMaxDisparity()
            # while True:
            depth_frame = self.getFrame(depthQueue)
            depth_frame_n = np.where(depth_frame == 65535, np.nan, depth_frame)
            new_frme = depth_frame_n[coord[0],coord[1]]
            depth_frame = (depth_frame *
                         depthMultiplier).astype(np.uint8)#8byte
            depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
            return depth_frame.astype(np.uint8),np.nanmean(new_frme)

    def cam_init(self,size=1080):
        # RGB channel
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        xoutVideo = self.pipeline.create(dai.node.XLinkOut)

        xoutVideo.setStreamName("video")

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        if size == 1080:
            self.size_gobal = (1920, 960)
            camRgb.setVideoSize(1920, 960)

        else:
            self.size_gobal=(1440, 720)
            camRgb.setVideoSize(1440, 720)
        camRgb.setFps(35)
        xoutVideo.input.setBlocking(False)
        xoutVideo.input.setQueueSize(1)

        camRgb.video.link(xoutVideo.input)

        #depth channel
        monoLeft = self.getMonoCamera(isLeft=True)
        monoRight = self.getMonoCamera(isLeft=False)
        stereo = self.getStereoPair(monoLeft, monoRight,size)  # mix two camera 相机匹配
        #深度与rgb对齐
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        xoutDepth = self.pipeline.createXLinkOut()  # depth channel
        xoutDepth.setStreamName("depth")  # make sure name
        stereo.depth.link(xoutDepth.input)
        #对元组进行解包
        stereo.setOutputSize(*self.size_gobal)
        device = dai.Device(self.pipeline)
        return device,stereo





if __name__=="__main__":
    oak=OAK()
    # device1,stereo = oak.depth_init()
    device,stereo=oak.cam_init(720)
    coord=[320,240]
    while True:
        img_RGB=oak.Run_RGB(device)
        img_depth,depth_data=oak.Run_depth(device, stereo,coord)
        # print(img_depth.shape)
        # img_depth=cv2.resize(img_depth,img_RGB.shape[1::-1])
        # print(np.nanmean(depth_data),'mm')
        cv2.imshow("video1", img_depth)
        cv2.imshow("video2", img_RGB)
        if cv2.waitKey(1) == ord('q'):
            break
