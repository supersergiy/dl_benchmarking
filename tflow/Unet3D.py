import tensorflow as tf
import numpy as np

class Unet(object):
    def __init__(self, merge = True, #Otherwise sum
                       symmetric = False,
                       activation ="Relu",
                       kernel_size = 3,
                       batchnorm = True,
                       upsample = True,
                       data_format = 'channels_first',
                       kernel_shape =  [[3,8],
                                        [8,16],
                                        [16,32],
                                        [32,64]]):
        super(Unet, self).__init__()
        self.merge = merge
        self.kshp = kernel_shape
        self.levels = len(kernel_shape)
        self.factor = 3 if merge else 1
        self.fused = True
        self.data_format = data_format # 'channels_first'
        self.batchnorm = True
        self.activation = tf.nn.relu if activation=="Relu" else tf.nn.elu
        self.padding = 'SAME' if symmetric else 'VALID'
        self.kernel_size = 3
        self.upsample = upsample
        self.upsample_layer = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format= self.data_format)

    def upconv(self, inputs, filters, stride=2, padding=1,
                output_padding=1, upsample=False):

        if self.upsample:
            #print('Upsample', inp, out)
            inputs = self.upsample_layer(inputs)#tf.keras.layers.upsample(inputs, size=2, data_format= self.data_format)
            inputs = tf.layers.conv3d(
                  inputs=inputs, filters=filters, kernel_size=self.kernel_size, strides=1,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)
        else:
            #print('ConvTranpose', inp, out)
            inputs = tf.layers.conv3d_transpose(
                  inputs, filters=filters, kernel_size=self.kernel_size, strides=2,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)

        if self.batchnorm:
            inputs = tf.layers.batch_normalization(
                inputs, fused=self.fused)
        return self.activation(inputs)

    def block(self, inputs, n=1, filters=[16,32], strides=1):
        for f in filters:
            inputs = tf.layers.conv3d(
                  inputs=inputs, filters=f, kernel_size=self.kernel_size, strides=strides,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)

            if self.batchnorm:
                inputs = tf.layers.batch_normalization(
                    inputs, fused=self.fused)
        return  self.activation(inputs)

    def forward(self, x):
        layers = [x]
        for i in range(self.levels-1):
            x = self.block(x, filters=[int(self.kshp[i][1]/2), self.kshp[i][1]])
            layers.append(x)
            x = tf.layers.max_pooling3d(x, pool_size=2, strides=2, padding='SAME', data_format=self.data_format)

        x = self.block(x, filters=[self.kshp[-1][0], self.kshp[-1][1]])
        layers.append(x)

        for i in range(1, self.levels):
            j = self.levels-i
            x = self.upconv(x, filters=min(self.factor,2)*self.kshp[j][0])
            x_enc = layers[(self.levels-1)-(i-1)]

            start = int((x_enc.get_shape().as_list()[2]-x.get_shape().as_list()[2])/2)

            if self.data_format=="channels_first":
                x_enc = x_enc[:,:,start:start+x.get_shape().as_list()[2],
                                  start:start+x.get_shape().as_list()[2],
                                  start:start+x.get_shape().as_list()[2]]

            else:
                x_enc = x_enc[:,  start:start+x.get_shape().as_list()[2],
                                  start:start+x.get_shape().as_list()[2],
                                  start:start+x.get_shape().as_list()[2], :]

            if self.merge:
                ax = 1 if self.data_format=="channels_first" else 4
                x = tf.concat((x_enc, x),axis=ax)
            else:
                x = x+x_enc

            x = self.block(x, [self.kshp[j][0],self.kshp[j][0]])

        x = tf.layers.conv3d(
              inputs=x, filters=self.kshp[0][0], kernel_size=self.kernel_size, strides=1,
              padding=self.padding, use_bias=True,
              kernel_initializer=tf.variance_scaling_initializer(),
              data_format=self.data_format)

        return x


#Simple test
if __name__ == "__main__":
    unet = Unet()
    shp = (1,3,128,128,128)
    inp = np.ones(shp, dtype=np.float32)
    x = tf.placeholder(tf.float32, shape=shp)
    out = unet.forward(x)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(out, feed_dict={x: inp})
    print(output.shape) #expected output (1,3,32,32,32)
