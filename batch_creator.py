import os
import tensorflow as tf
import random
import numpy as np
from PIL import Image

def image_preprocessing(image_file = '', size = 100):
    '''返回2维矩阵(1,3*size*size)'''
    im = Image.open(image_file)
    #图片转化为(size,size)
    w,h = im.size
    if (w <= h):
        p = float(size)/w*h
        im = im.resize((size, int(p)))
        w,h = im.size
        x = (h-size)/2
        im = im.crop((0,x,100,100+x))
    else:
        p = float(size)/h*w
        im = im.resize((int(p), size))
        w,h = im.size
        x = int((w-size)/2)
        im = im.crop((x,0,100+x,100))
    resized = np.array(im)
    channels = []
    channels.append(resized[:,:,0])
    channels.append(resized[:,:,1])
    channels.append(resized[:,:,2])
    channels_reshaped = []
    for i in range(3):
        channels_reshaped.append(np.reshape(channels[i],[1,-1]))
    channels_in_one = np.concatenate(channels_reshaped,axis = 1)
    data = channels_in_one
    print (len(data[0]))
    return data



def randomRead(root):
    '''随机读取一个文件'''
    import os
    file_names = []
    for parent, dirnames, filenames in os.walk(root):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        file_names = filenames
        # for filename in filenames:                        #输出文件信息
        #     print("parent is" + parent)
        #     print("filename is:" + filename)
        #     print("the full name of the file is:" + os.path.join(parent, filename))
    x = random.randint(0, len(file_names)-1)
    print(file_names[x])
    return file_names[x]


class BatchCreator(object):
    def __init__(self, batch_size, file_name,shape):
        self._batch_size = batch_size
        self._file_name = file_name
        self._shape = list(shape)

        pass
    def getBatch(self):
        sample = np.zeros(self._shape, int)
        batch  = [sample]
        for i in range(1,self._batch_size):
            batch = np.concatenate([batch,[sample]],axis = 0)
            pass
        batch = np.stack(batch,axis=0)
        return batch

class ImgBatchCreator(BatchCreator):
    def __init__(self, batch_size, file_name,shape,n_classes ):
        BatchCreator.__init__(self, batch_size, file_name,shape)
        self._n_classes = n_classes
        pass
    def getBatch(self):
        class_str = '0'
        sample_path = os.path.join(self._file_name,'train',class_str)
        sample_file = randomRead(sample_path)
        sample_file = os.path.join(sample_path,sample_file)
        print('正在读取：',sample_file)
        data = image_preprocessing(sample_file)
        batch  = [data]
        for i in range(1,self._batch_size):
            if class_str == str(self._n_classes -1 ):
                class_str = '0'
            else:
                class_str = str(int(class_str)+1)
            sample_path = os.path.join(self._file_name,'train',class_str)
            sample_file = randomRead(sample_path)
            sample_file = os.path.join(sample_path,sample_file)
            print('正在读取：',sample_file)
            data = image_preprocessing(sample_file)
            batch = np.concatenate([batch,[data]],axis = 0)
        batch = np.stack(batch,axis=0)
        return batch


def main():
    a = BatchCreator(5,'',[2,3])
    b = a.getBatch()

    print(b)
    print(b.shape)
    # randomRead('D:\\python\\lstm')
    pass

    image_preprocessing('D:\\python\\lstm\\train\\0\\aeroplane_s_000004.png')

    c = ImgBatchCreator(25,'D:\\python\\lstm',[50,50],10)
    d = c.getBatch()
    print(d.shape)
if __name__ == '__main__':
    main()