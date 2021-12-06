import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras import Model

class LeNet(Model):
    """
    LeNet is an early and famous CNN architecture for image classfication task.
    It is proposed by Yann LeCun. Here we use its architecture as the startpoint
    for your CNN practice. Its architecture is as follow.

    input >> Conv2DLayer >> Conv2DLayer >> flatten >>
    DenseLayer >> AffineLayer >> softmax loss >> output

    Or

    input >> [conv2d-avgpooling] >> [conv2d-avgpooling] >> flatten >>
    DenseLayer >> AffineLayer >> softmax loss >> output

    http://deeplearning.net/tutorial/lenet.html
    """

    def __init__(self, input_shape, output_size=10):
        '''
        input_shape: The size of the input. (img_len, img_len, channel_num).
        output_size: The size of the output. It should be equal to the number of classes.
        '''
        super(LeNet, self).__init__()
        #############################################################
        # TODO: Define layers for your custom LeNet network         
        # Hint: Try adding additional convolution and avgpool layers
        #############################################################
        
        print('./utils/neuralnets/cnn/my_model_LeNet.__init__() not implemented!') # delete me
        
        #############################################################
        #                          END TODO                         #                                              
        #############################################################

    
    def call(self, x):
        '''
        x: input to LeNet model.
        '''
        #call function returns forward pass output of the network
        #############################################################
        # TODO: Implement forward pass for custom network defined 
        # in __init__ and return network output
        #############################################################
        
        print('./utils/neuralnets/cnn/my_model_LeNet.call() not implemented!') # delete me
        
        #############################################################
        #                          END TODO                         #                                              
        #############################################################
        

