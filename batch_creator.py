import os
import tensorflow as tf
import random
import numpy as np
import re

def image_preprocessing(image_file = '', size = 100):
    '''返回2维矩阵(1,3*size*size)'''
    im_file = tf.gfile.FastGFile(image_file, 'rb').read()
    if re.findall(r'jpg$', image_file):
        img_data = tf.image.decode_jpeg(im_file)
    if re.findall(r'png$', image_file):
        img_data = tf.image.decode_png(im_file)
    
    #图片转化为(size,size)
    image_shape = img_data.eval().shape
    if image_shape[0] <= image_shape[1]:
        p = size/(image_shape[0])*(image_shape[1])
        resized = tf.image.resize_images(img_data, (size, int(p)), method=1)
    else:
        p = size/(image_shape[1])*(image_shape[0])
        resized = tf.image.resize_images(img_data, (int(p), size), method=1)
    croped = tf.image.resize_image_with_crop_or_pad(resized,size,size).eval()
    channels = tf.unstack(croped,axis= 2)
    channels_reshaped = []
    for i in range(3):
        channels_reshaped.append(tf.reshape(channels[i],[1,-1]))
    channels_in_one = tf.concat(channels_reshaped,axis = 1)
    data = channels_in_one
    # print (data.eval().shape)
    return data
def image_preprocessing_v2(image_file = '', size = 100):
    '''返回2维矩阵(1,3*size*size)'''
    im_file = tf.gfile.FastGFile(image_file, 'rb').read()
    # if re.findall(r'jpg$', image_file):
    #     img_data = tf.image.decode_jpeg(im_file)
    # if re.findall(r'png$', image_file):
    #     img_data = tf.image.decode_png(im_file)
    #图片转化为(size,size)
    image = tf.image.decode_png(im_file,3)
    # 对图片进行裁剪和正则化（将数值[0,255]转化为[-1,1]）  
    image = tf.image.resize_images(image, size=[size, size])  
    image = image*1.0/127.5 - 1.0  
    channels = tf.unstack(image,axis= 2)
    channels_reshaped = []
    for i in range(3):
        channels_reshaped.append(tf.reshape(channels[i],[1,-1]))
    channels_in_one = tf.concat(channels_reshaped,axis = 1)
    # print (data.eval().shape)
    return channels_in_one


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
    # print(file_names[x])
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
        # print('正在读取：',sample_file)
        data = image_preprocessing(sample_file,self._shape[0])
        batch_x  = [data]
        batch_y = np.zeros((self._batch_size,self._n_classes))
        batch_y[0,0] = 1
        for i in range(1,self._batch_size):
            if class_str == str(self._n_classes -1 ):
                class_str = '0'
            else:
                class_str = str(int(class_str)+1)
            sample_path = os.path.join(self._file_name,'train',class_str)
            sample_file = randomRead(sample_path)
            sample_file = os.path.join(sample_path,sample_file)
            # print('正在读取：',sample_file)
            data = image_preprocessing(sample_file,self._shape[0])
            batch_x = tf.concat([batch_x,[data]],axis = 0)
            batch_y[i,int(class_str)] = 1
        with tf.Session() as sess:
            batch_x_array = sess.run(tf.stack(batch_x,axis=0))
        return batch_x_array,batch_y


class ImgBatchCreator_v2(BatchCreator):
    def __init__(self, batch_size, file_name,shape,n_classes ):
        BatchCreator.__init__(self, batch_size, file_name,shape)
        self._n_classes = n_classes
        pass
        imagepaths = []  
        labels = []  
        for c in range(n_classes):
            c_dir = os.path.join(self._file_name,'train', str(c))  
            for parent, dirnames, filenames in os.walk(c_dir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
                filenames = filenames
                parent = parent
                for filename in filenames:                        #输出文件信息
                    imagepaths.append(os.path.join(parent, filename))
                    label = np.zeros(n_classes)
                    label[c] = 1
                    labels.append(label.tolist())
        imagepaths = tf.convert_to_tensor(imagepaths, tf.string)  
        labels = tf.convert_to_tensor(labels, tf.float32)
        # 建立 Queue  
        imagepath, self.label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)  
    
        # 读取图片，并进行解码  
        image = tf.read_file(imagepath)  
        # if re.findall(r'jpg$', imagepath):
        #     image = tf.image.decode_jpeg(image)
        # if re.findall(r'png$', imagepath):
        #     image = tf.image.decode_png(image)
        image = tf.image.decode_png(image,3)
        # 对图片进行裁剪和正则化（将数值[0,255]转化为[-1,1]）  
        image = tf.image.resize_images(image, size=[self._shape[0], self._shape[1]])  
        image = image*1.0/127.5 - 1.0  
        channels = tf.unstack(image,axis= 2)
        channels_reshaped = []
        for i in range(3):
            channels_reshaped.append(tf.reshape(channels[i],[1,-1]))
        self.channels_in_one = tf.concat(channels_reshaped,axis = 1)
        
    def getBatch(self):
        # 创建 batch  
        X, Y = tf.train.batch([self.channels_in_one, self.label], shapes = [[1,self._shape[0]*self._shape[1]*3],[self._n_classes]],batch_size=self._batch_size, num_threads=4, capacity=self._batch_size*8)
        return X,Y

def main():
    with tf.Session() as sess:
        a = BatchCreator(5,'',[2,3])
        b = a.getBatch()

        print(b)
        print(b.shape)
        # randomRead('D:\\python\\lstm')
        image_preprocessing('D:\\python\\lstm\\train\\0\\aeroplane_s_000004.png')

        # c = ImgBatchCreator(25,'D:\\python\\lstm',[10,10],10)
        # d ,y= c.getBatch()
        # print(d.shape)
        # print(y)
        e = ImgBatchCreator_v2(128,'D:\\python\\lstm',[28,28],10)
       
        coord = tf.train.Coordinator()
        
        
        threads = tf.train.start_queue_runners(sess=sess,coord =coord)
        x,y = e.getBatch()
        tf.global_variables_initializer().run()
        x = sess.run(x)
        y = sess.run(y)
        print(x.shape)
        print(y.shape)
        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    main()