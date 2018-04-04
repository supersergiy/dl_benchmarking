import tensorflow as tf
import numpy as np

class Unet(object):
    def __init__(self, merge = True, #Otherwise sum
                       symmetric = False,
                       activation ="relu",
                       kernel_size = 3,
                       batchnorm = True,
                       upsample = False,
                       data_format = 'channels_first',
                       residual = False,
                       kernel_shape =  [[3,24],
                                        [32,48],
                                        [48,72],
                                        [72,104],
                                        [104, 144]]):
        super(Unet, self).__init__()

        if residual:
            self.kshp = [[3,24], [24,32], [32,48], [48,72], [72, 104], [104, 144]]
            merge = False
            symmetric = True
            upsample = True

        elif symmetric:
            self.kshp = [[3,32], [32,64], [64,128], [128,256]]
        else:
            self.kshp = [[3,64], [64,128], [128,256], [256,512]]

        self.merge = merge
        self.levels = len(self.kshp)
        self.factor = 3 if merge else 1
        self.fused = True
        self.data_format = data_format # 'channels_first'
        self.batchnorm = batchnorm
        def activ(x):
            x = tf.nn.relu(x) if activation=="relu" else tf.nn.elu(x)
            print(activation)
            return x

        self.activation = activ

        self.kernel_size = 3
        self.upsample = upsample
        self.residual = residual

        self.padding = 'SAME' if symmetric else 'VALID'
        self.upsample_layer = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format= self.data_format)

    def upconv(self, inputs, filters, stride=2, padding=1,
                output_padding=1, upsample=False):

        if self.upsample:
            print('~Upsample')
            inputs = self.upsample_layer(inputs)#tf.keras.layers.upsample(inputs, size=2, data_format= self.data_format)
            print('Conv3D', filters)
            inputs = tf.layers.conv3d(
                  inputs=inputs, filters=filters, kernel_size=1, strides=1,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)
        else:
            print('~ConvTranpose', filters)
            inputs = tf.layers.conv3d_transpose(
                  inputs, filters=filters, kernel_size=2, strides=2,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)

        return inputs

    def block(self, inputs, filters=[16,32], strides=1):

        for f in filters:
            print('Conv3D', f)
            inputs = tf.layers.conv3d(
                  inputs=inputs, filters=f, kernel_size=self.kernel_size, strides=strides,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)

            if self.batchnorm:
                print('Batchnorm', f)
                inputs = tf.layers.batch_normalization(
                    inputs, fused=self.fused)
            inputs = self.activation(inputs)
        return inputs

    def res_block(self, inputs, filters=[32,32], strides=1):
        for i in range(len(filters)):
            print('Conv3D', filters[i])
            inputs = tf.layers.conv3d(
                  inputs=inputs, filters=filters[i], kernel_size=self.kernel_size, strides=strides,
                  padding=self.padding, use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)
            if self.batchnorm:
                print("Batchnorm")
                inputs = tf.layers.batch_normalization(
                    inputs, fused=self.fused)

            inputs = self.activation(inputs)
            if i == 0:
                skip = inputs
            else:
                print('Conv3D', filters[i])
                inputs = tf.layers.conv3d(
                      inputs=inputs, filters=filters[i], kernel_size=self.kernel_size, strides=strides,
                      padding=self.padding, use_bias=False,
                      kernel_initializer=tf.variance_scaling_initializer(),
                      data_format=self.data_format)
        print("Sum")
        inputs = skip + inputs
        if self.batchnorm:
            print("Batchnorm")
            inputs = tf.layers.batch_normalization(
                inputs, fused=self.fused)
        inputs = self.activation(inputs)

        return inputs

    def forward(self, x):
        if self.residual:
            print('Conv3D', self.kshp[0][1])
            x = tf.layers.conv3d(
                  inputs=x, filters=self.kshp[0][1], kernel_size=[1,5,5], strides=1,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)
            self.activation(x)

        layers = [x]

        for i in range(self.levels-1):
            if self.residual:
                x = self.res_block(x, filters=[self.kshp[i][1], self.kshp[i][1]])
            else:
                x = self.block(x, filters=[int(self.kshp[i][1]/2), self.kshp[i][1]])
            layers.append(x)
            print("~Max_pooling3D")
            x = tf.layers.max_pooling3d(x, pool_size=2, strides=2, padding='SAME', data_format=self.data_format)

        if self.residual:
            x = self.res_block(x, filters=[self.kshp[-1][1], self.kshp[-1][1]])
        else:
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
                print("Merge")
                ax = 1 if self.data_format=="channels_first" else 4
                x = tf.concat((x_enc, x),axis=ax)
            else:
                print("Sum")
                x = x+x_enc
                if self.batchnorm and self.residual:
                    print('Batchnorm')
                    x = tf.layers.batch_normalization(x, fused=self.fused)

            x = self.block(x, filters=[self.kshp[j][0],self.kshp[j][0]])

        if self.residual:
            print('Conv3D', self.kshp[0][1])
            x = tf.layers.conv3d(
                  inputs=x, filters=self.kshp[0][1], kernel_size=[1,5,5], strides=1,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)
            self.activation(x)

            print('Conv3D', 12)
            x = tf.layers.conv3d(
                  inputs=x, filters=12, kernel_size=1, strides=1,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)
        else:
            print('Conv3D', self.kshp[0][0])
            x = tf.layers.conv3d(
                  inputs=x, filters=self.kshp[0][0], kernel_size=1, strides=1,
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
