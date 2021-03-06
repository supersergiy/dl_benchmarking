import tensorflow as tf
import numpy as np

class Unet(object):
    def __init__(self, merge = False, #Otherwise sum
                       symmetric = True,
                       activation ="relu",
                       kernel_size = 3,
                       batchnorm = True,
                       upsample = True,
                       data_format = 'channels_first',
                       residual = False,
                       block = False,
                       dim = 3,
                       kernels = [[3,32], [32,64], [64,128], [128,256]]):
        super(Unet, self).__init__()

        self.kshp = kernels
        if residual:
            merge = False
            symmetric = True
            upsample = True

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
        self.is_block = block
        self.kernel_size = 3
        self.upsample = upsample
        self.residual = residual

        self.padding = 'SAME' if symmetric else 'VALID'
        #3D vs 2D
        self._conv = tf.layers.conv3d if dim == 3 else tf.layers.conv2d
        self._conv_transpose = tf.layers.conv3d_transpose if dim==3 else tf.layers.conv2d_transpose
        self._pool = tf.layers.max_pooling3d if dim == 3 else tf.layers.max_pooling2d
        self._crop = self.crop3d if dim == 3 else self.crop2d
        self.dim = dim

        if dim == 3:
            factor = (1, 2, 2) if residual else (2, 2, 2)
            self.upsample_layer = tf.keras.layers.UpSampling3D(size=factor, data_format= self.data_format)
        else:
            factor = (2,2)
            self.upsample_layer = tf.keras.layers.UpSampling2D(size=factor, data_format= self.data_format)

    def upconv(self, inputs, filters, stride=2, padding=1,
                output_padding=1, upsample=False):

        if self.upsample:
            print('~Upsample')
            inputs = self.upsample_layer(inputs)#tf.keras.layers.upsample(inputs, size=2, data_format= self.data_format)
            print('Conv', filters)
            inputs = self._conv(
                  inputs=inputs, filters=filters, kernel_size=1, strides=1,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)
        else:
            print('~ConvTranpose', filters)
            inputs = self._conv_transpose(
                  inputs, filters=filters, kernel_size=2, strides=2,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)

        return inputs

    def block(self, inputs, filters=[16,32], strides=1):

        for f in filters:
            print('Conv', f)
            inputs = self._conv(
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
            print('Conv', filters[i])
            inputs = self._conv(
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
                print('Conv', filters[i])
                inputs = self._conv(
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

    def crop3d(self, x_enc, x):
        if self.data_format=="channels_first":
            start = (int((x_enc.get_shape().as_list()[2]-x.get_shape().as_list()[2])/2),
                     int((x_enc.get_shape().as_list()[3]-x.get_shape().as_list()[3])/2),
                     int((x_enc.get_shape().as_list()[4]-x.get_shape().as_list()[4])/2))
            x_enc = x_enc[:,:,start[0]:start[0]+x.get_shape().as_list()[2],
                              start[1]:start[1]+x.get_shape().as_list()[3],
                              start[2]:start[2]+x.get_shape().as_list()[4]]

        else:
            start = (int((x_enc.get_shape().as_list()[1]-x.get_shape().as_list()[1])/2),
                     int((x_enc.get_shape().as_list()[2]-x.get_shape().as_list()[2])/2),
                     int((x_enc.get_shape().as_list()[3]-x.get_shape().as_list()[3])/2))
            x_enc = x_enc[:,  start[0]:start[0]+x.get_shape().as_list()[1],
                              start[1]:start[1]+x.get_shape().as_list()[2],
                              start[2]:start[2]+x.get_shape().as_list()[3], :]
        return x_enc

    def crop2d(self, x_enc, x):
        if self.data_format=="channels_first":
            start = (int((x_enc.get_shape().as_list()[2]-x.get_shape().as_list()[2])/2),
                     int((x_enc.get_shape().as_list()[3]-x.get_shape().as_list()[3])/2))
            x_enc = x_enc[:,:,start[0]:start[0]+x.get_shape().as_list()[2],
                              start[1]:start[1]+x.get_shape().as_list()[3]]

        else:
            start = (int((x_enc.get_shape().as_list()[1]-x.get_shape().as_list()[1])/2),
                     int((x_enc.get_shape().as_list()[2]-x.get_shape().as_list()[2])/2))
            x_enc = x_enc[:,  start[0]:start[0]+x.get_shape().as_list()[1],
                              start[1]:start[1]+x.get_shape().as_list()[2] :]
        return x_enc

    def forward(self, x):
        if self.is_block:
            print(x.get_shape())
            sph = int(list(x.get_shape())[1])
            x = self.res_block(x, filters=[sph, sph])
            return x
        if self.residual:
            print('Conv', self.kshp[0][1])
            x = self._conv(
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
            if self.residual:
                x = self._pool(x, pool_size=(1,2,2), strides=(1,2,2), padding='SAME', data_format=self.data_format)
            else:
                x = self._pool(x, pool_size=2, strides=2, padding='SAME', data_format=self.data_format)

        if self.residual:
            x = self.res_block(x, filters=[self.kshp[-1][1], self.kshp[-1][1]])
        else:
            x = self.block(x, filters=[self.kshp[-1][0], self.kshp[-1][1]])
        layers.append(x)

        for i in range(1, self.levels):
            j = self.levels-i
            x = self.upconv(x, filters=min(self.factor,2)*self.kshp[j][0])
            x_enc = layers[(self.levels-1)-(i-1)]
            print(x.shape, x_enc.shape)
            #exit()

            x_enc = self._crop(x_enc, x)

            if self.merge:
                print("Merge")

                ax = 1 if self.data_format=="channels_first" else (4 if self.dim == 3 else 3)
                x = tf.concat((x_enc, x), axis=ax)
            else:
                print("Sum")
                x = x+x_enc
                if self.batchnorm and self.residual:
                    print('Batchnorm')
                    x = tf.layers.batch_normalization(x, fused=self.fused)

            x = self.block(x, filters=[self.kshp[j][0],self.kshp[j][0]])

        if self.residual:
            print('Conv', self.kshp[0][1])
            x = self._conv(
                  inputs=x, filters=self.kshp[0][1], kernel_size=[1,5,5], strides=1,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)
            self.activation(x)

            print('Conv', 12)
            x = self._conv(
                  inputs=x, filters=12, kernel_size=1, strides=1,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)

            print("Sigmoid")
            x = tf.sigmoid(x)
        else:
            print('Conv', self.kshp[0][0])
            x = self._conv(
                  inputs=x, filters=self.kshp[0][0], kernel_size=1, strides=1,
                  padding=self.padding, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)

        return x


#Simple test
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    unet = Unet(block=False, dim=2)
    #shp = (1,64,192,192,192)
    shp = (1,3,192, 192)
    inp = np.ones(shp, dtype=np.float32)
    x = tf.placeholder(tf.float32, shape=shp)
    out = unet.forward(x)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    output = sess.run(out, feed_dict={x: inp})
    print(output.shape) #expected output (1,3,32,32,32)
