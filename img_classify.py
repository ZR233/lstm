import tensorflow as tf
import batch_creator
import numpy as np
class Config(object):
    #时间层数
    time_steps = 28*3
    #隐含层节点数
    num_units=128
    #输入维数
    n_input=28
    #学习率
    learning_rate=0.001
    #类别数
    n_classes=10
    #一批的数量
    batch_size=128


class Classifier(object):
    def __init__(self, Config):
        self.__is_train = True
        #weights and biases of appropriate shape to accomplish above task
        self.__out_weights=tf.Variable(tf.random_normal([Config.num_units,Config.n_classes]),name= 'w')
        self.__out_bias=tf.Variable(tf.random_normal([Config.n_classes]),name='b')
        #defining placeholders
        #input image placeholder	
        self.__x=tf.placeholder("float",[None,Config.time_steps,Config.n_input],name='x')
        #input label placeholder
        self.__y=tf.placeholder("float",[None,Config.n_classes],name='y')

        #processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
        self.__inputs=tf.unstack(self.__x ,Config.time_steps,1)

        #defining the network
        self.__lstm_layer = tf.contrib.rnn.BasicLSTMCell(Config.num_units,forget_bias=1)
        self.__outputs,_ = tf.contrib.rnn.static_rnn(self.__lstm_layer,self.__inputs,dtype="float32")

        #converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
        self.__prediction = tf.matmul(self.__outputs[-1],self.__out_weights) + self.__out_bias

    def train(self):
        """inputs: []"""
        #loss_function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.__prediction,labels=self.__y))
        #optimization
        opt=tf.train.AdamOptimizer(learning_rate=Config.learning_rate).minimize(loss)
        #model evaluation
        correct_prediction=tf.equal(tf.argmax(self.__prediction,1),tf.argmax(self.__y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #initialize variables
        init=tf.global_variables_initializer()
        saver = tf.train.Saver() 
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            sess.run(init)
            iter=1
            imc = batch_creator.ImgBatchCreator_v2(Config.batch_size,'D:\\python\\lstm',[int(Config.time_steps/3),Config.n_input],Config.n_classes)
            coord = tf.train.Coordinator()
            batch_x,batch_y = imc.getBatch()
            tf.global_variables_initializer().run()
            threads = tf.train.start_queue_runners(sess=sess,coord =coord)
            while iter<800:
                x = sess.run(batch_x)
                y = sess.run(batch_y)

                batch_x_r = np.reshape(x,[Config.batch_size,Config.time_steps,Config.n_input])
                sess.run(opt, feed_dict={self.__x: batch_x_r, self.__y: y})
                print('训练一次')
                saver.save(sess, "Model/model.ckpt")
                print("保存model")
                if iter %10==0:
                    acc=sess.run(accuracy,feed_dict={self.__x:batch_x_r,self.__y:y})
                    los=sess.run(loss,feed_dict={self.__x:batch_x_r,self.__y:y})
                    print("For iter ",iter)
                    print("Accuracy ",acc)
                    print("Loss ",los)
                    print(y[0])
                    print(sess.run(self.__prediction, feed_dict={self.__x:batch_x_r}))
                    print("__________________")
                    
                iter=iter+1
            coord.request_stop()
            coord.join(threads)
        pass
    def classify(self,file):
        saver = tf.train.Saver({'b':self.__out_bias,'w':self.__out_weights}) 
        with tf.Session() as sess:
            im_data = batch_creator.image_preprocessing_v2(file,Config.n_input).eval()
            im_data = np.reshape(im_data,(1,Config.time_steps,Config.n_input))
            saver.restore(sess, "./Model/model.ckpt")
            x=tf.placeholder("float",[None,Config.time_steps,Config.n_input],name='x')
            #input label placeholder
            y=tf.placeholder("float",[None,Config.n_classes],name='y')

            #processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
            inputs=tf.unstack(x ,Config.time_steps,1)

            outputs,_ = tf.contrib.rnn.static_rnn(self.__lstm_layer,inputs,dtype="float32")

            #converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
            prediction = tf.matmul(outputs[-1],self.__out_weights) + self.__out_bias
            tf.global_variables_initializer().run()
            y = sess.run(prediction, feed_dict={x: im_data})
            print(y)
        pass
def main():
    config = Config()
    ls = Classifier(config)
    ls.train()

if __name__ == '__main__':
    main()