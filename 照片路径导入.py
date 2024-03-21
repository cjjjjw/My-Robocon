import os
import random
import shutil
import numpy as np


def copy(file_path,path_list):
    filelist=os.listdir(file_path)
    for i in path_list:
        for file in filelist:
            # print(file)
            src =os.path.join(file_path,file)
            dst=i
            # print(src,dst)
            shutil.copy(src,dst)
        print("照片转移完成")

test_num=0

if __name__=="__main__":
    file_path= "source_img"
    images=os.path.exists(file_path)
    if  images:
        file_name=str(input("输入新创建照片储存文件夹的名称:"))
        if os.path.exists(f'datasets\\{file_name}'):
            pass
        else:
            os.mkdir(f'datasets\\{file_name}')
            os.makedirs(f'datasets\\{file_name}\\images\\train')
            os.makedirs(f'datasets\\{file_name}\\images\\val')
            os.makedirs(f'datasets\\{file_name}\\labels\\train')
            os.makedirs(f'datasets\\{file_name}\\labels\\val')
            os.mkdir(f'datasets\\{file_name}\\test_imgs')
            os.mkdir(f'datasets\\{file_name}\\test_videos')
            # 传入的是照片的路径
        #复制照片到目标文件夹

        path_list=[f'datasets\\{file_name}\\images\\train',f'datasets\\{file_name}\\images\\val']
        try:
          copy(file_path,path_list)
        except Exception:
            print(f"没有文件权限，或是文件夹被占用，请手动在D:\\yolov7-main\\datasets\\{file_name}\\images\\train，放入文件")
         # 获取绝对路径
        try:
            path1 = f'datasets\\{file_name}\\images\\train'
            path2 = f'datasets\\{file_name}\\images\\val'
            files1 = os.listdir(path1)
            files2 = os.listdir(path2)
            # open中的mode只是读取方式，不能直接调用
            # 先初始化是防止txt过于繁琐
            #todo请把yolov7改成你的yolo文件夹
            with open(f'datasets\\{file_name}\\train_list.txt', mode='w', encoding='utf-8') as f:
                f.write("")
            with open(f'datasets\\{file_name}\\val_list.txt', mode='w', encoding='utf-8') as f:
                f.write("")
            for i in files1:
                with open(f'datasets\\{file_name}\\train_list.txt', mode='a', encoding='utf-8') as f:
                    f.write(f"{path1}\\{i}\n")
            for i in files2:
                with open(f'datasets\\{file_name}\\val_list.txt', mode='a', encoding='utf-8') as f:
                    f.write(f"{path2}\\{i}\n")
            #test path create
            for i in np.random.randint(0, len(files1), size=[1, np.random.randint(10,50)])[0]:
                with open(f'datasets\\{file_name}\\test_list.txt', mode='a', encoding='utf-8') as f:
                    f.write(f"{path1}\\{files1[i]}\n")
                    test_num+=1
            print(f"随机测试数据有{test_num}个")
            print("成功导入txt图片或视频路径到")
        except Exception:
            print('请再次启动')
    else:
        print("没有照片或视频目标文件夹")



#先判断文件夹是否存在

# src=file_name
# dst='D:\\yolov7-main\\datasets\\{file_name}'
# shutil.copy(src, dst)
# work_dir = os.getcwd() # 获取绝对路径
# src = os.path.join()
# dst = os.path.join()
 #复制图片
