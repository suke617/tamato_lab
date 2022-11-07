#!/usr/bin/env python
# -*- coding: utf-8 -*-

#共通のライブラリ
import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np 
from PIL import Image

#モデル用のライブラリ
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression,scale_coords, strip_optimizer
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel


#realsense用ライブラリ
import pyrealsense2 as rs
#ROS周りのライブラリ
import rclpy  # ROS2のPythonモジュールをインポート
from rclpy.node import Node # rclpy.nodeモジュールからNodeクラスをインポート
from std_msgs.msg import String ,Bool# トピック通信に使うStringメッセージ型をインポート
# from geometry_msgs.msg import Twist # トピック通信に使うTwistメッセージ型をインポート
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from cv_bridge import CvBridge, CvBridgeError
import math
#時間管理用ライブラリ
import time
import threading



"""
@author yoshida keisuke

YOLO7をROS2で動かせるように変更
(応急処置的なコード、とりあえず動く)

suscribe_messege
/image_raw

publish_messege
/bbox

"""


"""
yolo7を用いた画像処理パート

・実行内容
  yolo7で検知→座標取得→アームの座標系での移動距離に変換→データ送信

・各種設定パラメータ
  WEIGHTS : weight_fileの置き場所
  IMGSZ : 学習した際の画像サイズ
  Device  - GPU使用時 : 使用するGPUのデバイス番号を指定
          - CPU使用時 : cpuと指定
          #str型で与えないとエラーになるので注意 
  IOU_THRES : つの領域がどれぐらい重なっているかを表す指標

"""

class Image_Processing(Node):  
    #定数パラメータ
    WEIGHTS = "runs/train/yolov7-tiny35/weights/last.pt"
    IMGSZ =640
    Device = str("cpu") #str(0)   
    IOU_THRES = int(0.45)
    CONF_THRES=0.9
    Realsense_Mode = True #False

    def __init__(self):
        super().__init__('sim_node')  #Nodeクラスのコンストラクタを呼び出し,ノード作成
        #load_model
        self.model,self.names,self.colors,self.stride,self.imgsz,self.device= self.prepare_model()
        #publisher
        #データ型,ノード名、キューサイズ
        self.image_pub = self.create_publisher(Image,'/result', 10)         
        self.Fruit_Pos = self.create_publisher(Int64,'Fruit_Pos', 10) 
        self.Ontarget = self.create_publisher(Image,'Ontarget', 10) 
        self.create_timer(1.0, self.node_callback)
        
        #opencv→rosdata
        self.bridge = CvBridge()
        self.half = False
        self.cam = realsense_module()
 
    def node_callback(self):
        if self.Realsense_Mode :
            start=time.time()
            color_img,depth_img = self.get_img()        
            print("cam_time",start-time.time())
            detect_fruit=self.detect(color_img,depth_img,self.Realsense_Mode)
            print("estimate_time",start-time.time())
            self.cam.shutdown()
        else :
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            start=time.time()
            detect_fruit=self.detect(frame,False,self.Realsense_Mode)
            print(start-time.time())
            print("果実を発見しました") if detect_fruit else print("発見できませんでした")


    def get_img(self) :
        color_image,depth_image,img_flag=self.cam.obtain_cam_image()
        lim_colorimage,lim_depth_image=self.cam.limit_area(color_image,depth_image)
        return lim_colorimage,lim_depth_image
    
    
    def convert_to_real_distance(self) :
        #座標変換作る　場所は今のキネクトの位置
        pass

    def publish_data(self,data) :
        pass
        # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))


    def prepare_model(self):
        device = select_device(self.Device)
        half = device.type != 'cpu'  
        self.half = half
        # Load model
        model = attempt_load(self.WEIGHTS, map_location=device)  
        stride = int(model.stride.max())  
        imgsz = check_img_size(self.IMGSZ, s=stride)   
        model = TracedModel(model, device, self.IMGSZ)
        if half:
            model.half()  
        
        # class_names
        names = model.names
        # bboxs_colors
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        return model,names,colors,stride,imgsz,device
    

    def detect(self,color_img,depth_img,realsense_mode):
        im0s = color_img
        detect_fruit=False
        # Padded resize
        img = self.letterbox(im0s, self.imgsz , self.stride)[0]
        # Convert
        #realsense_mode
        if realsense_mode :
           img = img.transpose(2, 0, 1)
        else :
           img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  
        img /= 255.0  # 0 - 255 to 0.0 - 1.0 #正規化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.CONF_THRES, self.IOU_THRES, classes=None, agnostic=False)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #検知できた際の処理
            if len(det): 
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                #描画用(bbox_class_conf)   
                for *xyxy, conf, cls in reversed(det):
                    r_x,r_y,l_x,l_y = xyxy
                    #c_x, c_y = lambda right_px,left_px : (right_px+left_px)/2 (100,0.08)
                    #r_x,r_y,l_x,l_y =x.to('cpu').detach().numpy().copy(),c.to('cpu').detach().numpy().copy(),v.to('cpu').detach().numpy().copy(),b.to('cpu').detach().numpy().copy()
                    #target_center = .transpose(2, 0, 1)
                    if display_bbox :  
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                detect_fruit=True
            cv2.imwrite(f"{i}_detect_img.jpg", im0)
        return detect_fruit
            


    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

"""
realsense関係のモジュール
"""
class realsense_module():
    def __init__(self) :
        #RGBとdepthの初期設定
        self.conf = rs.config()
        #解像度はいくつか選択できる
        self.conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.conf.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #stream開始
        self.pipe = rs.pipeline()
        self.profile = self.pipe.start(self.conf)
        #Alignオブジェクト生成(位置合わせのオブジェクト)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def obtain_cam_image(self) :
        try :
            #フレーム待ち(これがないとデータの取得にエラーが出ることがあるらしい）
            frames = self.pipe.wait_for_frames()
            # frameデータを取得
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not depth_frame or not color_frame:
                return
            #dataがunit16の形で入っているのでnumpy配列に変更
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            img_flag=True
            return color_image,depth_image,img_flag
        except Exception as e :
            print(e)
            color_image=None
            depth_image=None
            img_flag=False
            return color_image,depth_image,img_flag

    def limit_area(self,color_image,depth_image):
        left=0
        right=600
        top=0
        bottom=500
        lim_colorimage=color_image[left:right,top:bottom,:]
        lim_depth_image=depth_image[left:right,top:bottom]
        return lim_colorimage,lim_depth_image

    def shutdown(self):
        self.pipe.stop()

    def save_data():  
        #画像を保存用(ある周期ごとにdataを保存したい)
        pass
    
if __name__ == '__main__':
    display_bbox=True
    with torch.no_grad():
        #rosの初期設定
        rclpy.init() # rclpyモジュールの初期化
        run=Image_Processing() # ノードの作成
        try :
            rclpy.spin(run) #sabscribeするまで待つ
            print("sim2_state")
            # 制御周期
            
        except KeyboardInterrupt :
            print("Ctrl+Cが入力されました")  
            print("プログラム終了")  
        rclpy.shutdown() # rclpyモジュールの終了処理   

  





