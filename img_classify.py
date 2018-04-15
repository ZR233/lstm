import tensorflow as tf


class Config(object):
    #时间层数
    time_steps = 28
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
        self.__out_weights=tf.Variable(tf.random_normal([Config.num_units,Config.n_classes]))
        self.__out_bias=tf.Variable(tf.random_normal([Config.n_classes]))
        #defining placeholders
        #input image placeholder	
        self.__x=tf.placeholder("float",[None,Config.time_steps,Config.n_input])
        #input label placeholder
        self.__y=tf.placeholder("float",[None,Config.n_classes])

        #processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
        self.__inputs=tf.unstack(__x ,Config.time_steps,1)

        #defining the network
        self.__lstm_layer = tf.contrib.rnn.BasicLSTMCell(Config.num_units,forget_bias=1)
        self.__outputs,_ = tf.contrib.rnn.static_rnn(__lstm_layer,__inputs,dtype="float32")

        #converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
        self.__prediction = tf.matmul(__outputs[-1],__out_weights) + __out_bias
    def batchCreator(self, file_name):
        
        pass
    def train(self, inputs):
        """inputs: []"""
        #loss_function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=__prediction,labels=__y))
        #model evaluation
        correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #initialize variables
        init=tf.global_variables_initializer()
        with tf.Session() as sess:
        sess.run(init)
        iter=1
        while iter<800:
            batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
            batch_x=batch_x.reshape((batch_size,time_steps,n_input))
            sess.run(opt, feed_dict={x: batch_x, y: batch_y})
            if iter %10==0:
                acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
                los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
                print("For iter ",iter)
                print("Accuracy ",acc)
                print("Loss ",los)
                print("__________________")
            iter=iter+1

        pass
    pass