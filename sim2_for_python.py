#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
import time
import cv2

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

"""
@author 吉田圭佑

"""

"""
外部プログラムからの入力




本プログラムからの出力
Image -画像処理後のデータを送信

"""

class Image_Processing(Node):  
    #定数パラメータ

    def __init__(self):
        super().__init__('image_node')  #Nodeクラスのコンストラクタを呼び出し,ノード作成

        #publisher
        self.publisher = self.create_publisher(Image,'cmd_vel', 10) #データ型,ノード名、キューサイズ
        self.Fruit_Pos = self.create_publisher(Int64,'Fruit_Pos', 10) 
        self.Ontarget = self.create_publisher(Image,'Ontarget', 10) 

        #opencv→rosdata
        self.bridge = CvBridge()
        self.create_subscription(Bool, "/Analysys_Flag",self.callback,10) 

    def callback(self,flag) :
        if flag.data :
            #realsense初期設定
            cam=realsense_module()
            color_image,depth_image,img_flag=cam.obtain_cam_image()
            lim_colorimage,lim_depth_image=cam.limit_area(color_image,depth_image)
            mask,detect_flag=self.mask_process(color_image)
            if detect_flag :
                print("トマト発見")
                tmt_list=self.detection(color_image,mask,depth_image)
                # pos=self.convert_to_real_distance(tmt_list)
                # self.publish_data() 
            else :
                print("トマトを発見できませんでした")
        else :
            pass

    """
    misscountどのようにpubするか？
    """
    def main_loop(self,color_image,depth_image):
        print(self.analysys_falg)
        
    def mask_process(self,img):
        # img = np.array(img)
        #R値のほうが強いものをトマト領域とする
        # mask=np.where(img[:,:,1] > 1.4*img[:,:,2] , 255 , 0)
        # ret2, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
        mask=img[:,:,1] < 1.4*img[:,:,2]
        mask=mask.astype('uint8')
        #cv2.imshow("mask", img)
        #カーネルを作成する。
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #収縮→膨張
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        #膨張→収縮
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        tom_mask = mask[0 < mask]
        tom_area = len(tom_mask)
        if tom_area > 0 : 
            tomato_detect_flag=True 
        else :
            tomato_detect_flag=False
        return mask,tomato_detect_flag
        
       
    #@author 藤本
    #k-meansによる物体認識
    def createmask(self,rgb_im):
        sigmaI = [85.8997609774271,-39.1553459604252,-38.6963626815188,
        -39.1553459604252,133.189825077682,-61.8149107865788,
        -38.6963626815188,-61.8149107865787,85.4460573367766]

        centroid = [0.377317984497057,0.51866805610024,0.304385088726629,
        0.511377845841187,0.682406674784993,0.311679884061617,
        0.258554224598926,0.343365128798657,0.218927575291707,
        0.692867936509784,0.385189498292858,0.222930547134729,
        0.75326742738383,0.783867898551208,0.769989835131869]

        w,h,c=rgb_im.shape
        bw=np.zeros(w,h)
        for x in range(w):
            for y in range(h):
                if rgb_im[x, y, 1] + rgb_im[x,y ,2] + rgb_im[x, y, 3] != 0:
                    r = rgb_im[x,y,1] * 0.0039216
                    g = rgb_im[x,y,2] * 0.0039216
                    b = rgb_im[x,y,3] * 0.0039216
                    minV = 1000000.0
                    bestCluster = -1
                    for i in range(5):
                        d1 = r - centroid[i, 1]
                        d2 = g - centroid[i, 2]
                        d3 = b - centroid[i, 3]
                        e1 = d1 * sigmaI[1, 1] + d2 * sigmaI[2, 1] + d3 * sigmaI[3, 1]
                        e2 = d1 * sigmaI[1, 2] + d2 * sigmaI[2, 2] + d3 * sigmaI[3, 2]
                        e3 = d1 * sigmaI[1, 3] + d2 * sigmaI[2, 3] + d3 * sigmaI[3, 3]
                        v = (e1 * d1 + e2 * d2 + e3 * d3)
                        if v<minV:
                            minV=v
                            bestCluster=i
                        #4:fruits, 2:green fruits, 1,3:leaf,steam, 5:others
                    if bestCluster == 4 :
                        bw[y, x] = 1
                    elif bestCluster == 2 :
                        bw[y, x] = 1
                    elif bestCluster == 1 and bestCluster == 3 :
                        bw[x, y] = 0
                    else :
                        bw[x, y] = 0
                        
        #th, bw = cv2.threshold(bw, 128, 192, cv2.THRESH_OTSU)
        # カーネルを作成する。
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 2値画像を収縮する。
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)
        # 2値画像を収縮する。
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

        return bw

    def detection(self,rgb,bw,depth):
        """
        変数説明
        tmtlist=[トマトのx座標、トマトのy座標             
        
        """

        #検出する個数、最大の個数
        listSize = 20           
        #検出する範囲の深度
        limitDepth = 999
        #半径(近似する領域)
        r = 23
        r2 = r * 2                  
        cnt = 0.0
        cnt2 = 0
        a=0.0
        r3 = 12
        #s_y, s_x BWの画角取得
        [s_x, s_y] = bw.shape
        #トマト領域チェック確認用配列
        lock = np.zeros(bw.shape)
        #左右入れ替え(？)
        depth = np.fliplr(depth)
        #[[y, x, depth, ?, ?]が入る配列生成
        tmtList = np.zeros([listSize, 5])
        output = np.zeros([listSize, 5])
        
        targetPos = [-1 , -1 ]
        targetDepth = -1 
        Ontarget=1
        view = rgb
        
        
        """
        径仮定法
        RGB値で求めたbwの中から真円率の高いピクセルだけを抽出
        
        """

        for count in range(listSize) : #20まで検出
            #とりあえず初期値
            limitDepth=999
            minDepth = 999 
            # cX = 1
            # cY = 1
            #画像の全画素探索
            for i in range(s_x) :
                for j in range(s_y) :
                    #未捜索かつ果実部かつ距離測定正しいかつ一番近い物体
                    if lock[i, j] == 0 and bw[i, j] == 1 and depth[i, j]!= 0 and depth[i, j] < minDepth :
                        minDepth = depth[i, j] #最小のdepthより小さければok!書き換え
                        #そのピクセル周りの情報を探す
                        cX = i
                        cY = j  
                    
            cnt = 0.0
            cnt2 = 0.0
            #アームの届く範囲でトマトを見つけれたか
            if minDepth < limitDepth :
                # 検出した最小のdepthから正方形で±rの範囲に対して捜索
                for i2 in range(cX-r,cX+r):
                    for j2 in range(cY-r,cY+r):
                        if i2 > 0 and j2 > 0 and i2 < s_x and j2 < s_y :
                            # 見ている点が原点から半径r内にあるか計算
                            if math.sqrt((i2-cX)*(i2-cX)+(j2-cY)*(j2-cY))<=r :
                                lock[i2, j2] = 1 #探索領域としてフラグ付
                                #遮蔽率
                                if rgb[i2, j2, 1] > rgb[i2, j2, 2] * 2 and depth[i2, j2] != 0 :
                                    cnt += 1.0
                                cnt2 += 1.0
                                a = cnt/cnt2
                            tmtList[count, :] = [cX, cY, depth[cX, cY], a*100, 0]
      
            #トマトを見つけれてもアームが届かない位置ならば除外
            else :
                break

        # """
        # ソートpart

        # 処理内容
        # ・距離測定ミスっているのは弾く
        # ・距離順で並び替え→ここは遮蔽率でやるのはどうか？

        # """
        
        # tom_count=0

        # #距離測定ミスっているのは弾く
        # for i in range(1,listSize):
        #     #depthが0（検出失敗？)の場合弾く
        #     if tmtList[i, 2] == 0 :
        #         tmtList[i, :] = []
        #     else :
        #         tom_count+=1

    
        # # #深度順でソート（いるのか？）
        tmtList = sorted(tmtList, reverse=False , key=lambda x:x[2])
        
        # #一番右はじの物を取りに行く 
        # # tmtList = sorted(tmtList, reverse=False , key=lambda x:x[1])
     
        # """
        # TODO 円じゃないやつは消さないと行けない

        # """

        # """
        # 収穫順番
        # """
        
        # sort_tom_list=[]
        # for i in  range(1,tom_count):
        #     tmtList[i, :] = new_tmtList[i, :]
        
        return tmtList
  
    
    """
    使ってない（画像処理だけで検出できないかと思っただけ）
    """
    # def detect_sub(self,mask,color_image,depth_image):
    #     mask[depth_image[:,:]>=999]=0 #一定距離以下は切り捨て
    #     #丸の検出
    #     mask = cv2.Canny(mask,100,200)
    #     circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,param1=120,param2=15,minRadius=10,maxRadius=30)
    #     if circles is not None and len(circles) > 0:
    #         #型をfloat32からunit16に変更：整数のタイプになるので、後々トラブルが減る。
    #         circles = np.uint16(np.around(circles))
        
    #     #表示用の画像トマト部分だけ抽出
    #     show_image = cv2.bitwise_and(color_image, mask)

    
    #     tom_list=[]
    #     tom_num=1
    #     for i in circles[0,:]:
    #         #tom_list=[番号,x座標、ｙ座標,半径,深度情報]
    #         if i[2]*i[2]<130 : #除外処理(一定以下の大きさの範囲は除外)
    #             if not tom_list : #最初リストが空のときだけこの処理
    #                 tom_list=[tom_num,i[0],i[1],i[2],depth_image[i[0],i[1]]]
    #                 tom_num+=1
    #             else :
    #                 tom_list=np.stack(tom_list,[tom_num,i[0],i[1],i[2],depth_image[i[0],i[1]]])
    #                 tom_num+=1 
    #             # 中心の座標を示す円を描く
    #             cv2.circle(color_image,(i[0],i[1]),2,(0,0,255),2)
    #         else :   
    #             pass
    #     return tom_list,show_image
    # def ripeness_judge(self,color_image):
    #     #配列の作成
    #     mask = color_image.copy()
    #     #収穫対象はRGBで考えた際にRedの値がBlueの1.4倍以上であればいい  
    #     mask[color_image[:,:,1]  < color_image[:,:,2]*1.4] = 0
    #     mask[color_image[:,:,1] >= color_image[:,:,2]*1.4] = 255
    #     #グレースケール変換(0.299 * R + 0.587 * G + 0.114 * B)
    #     mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    #     _,mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     kernel = np.ones((5,5),np.uint8)
    #     opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #     mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #     return mask
        
    """
    ここ見ないと
    """
    def convert_to_real_distance(self,tmt_list):
        #カメラの取り付け位置
        cam_pos_x=0.3
        cam_pos_y=0.5
        #カメラの画角
        img_center_x=720
        img_height=1280
        for i in range(tmt_list) :
            pos_x=tmt_list[i,0]
            pos_y=tmt_list[i,1]
            distanse=tmt_list[i,2]
            real_pose_x=cam_pos_x
            real_pose_y=cam_pos_y
            pos[i]=[real_pose_x,real_pose_y,distanse]
        return pos


    def publish_data(self,data) :
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))


"""
realsense関係のモジュール
"""
class realsense_module():
    def __init__(self) -> None:
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
        left=200
        right=400
        top=400
        bottom=900
        lim_colorimage=color_image[left:right,top:bottom,:]
        lim_depth_image=depth_image[left:right,top:bottom]
        return lim_colorimage,lim_depth_image

    def shutdown(self):
        self.pipe.stop()

    def save_data():  
        #画像を保存用(ある周期ごとにdataを保存したい)
        pass


def main():
    #rosの初期設定
    rclpy.init() # rclpyモジュールの初期化
    while (True):
        run=Image_Processing() # ノードの作成
        rclpy.spin(run) #sabscribeするまで待つ
        print("sim2_state")
        rclpy.sleep(1)
    cam.shutdown()
    rclpy.shutdown() # rclpyモジュールの終了処理



if __name__=='__main__':
    main()
   


"""
TO DOリスト
収穫可能範囲測定(realsense)
"""

