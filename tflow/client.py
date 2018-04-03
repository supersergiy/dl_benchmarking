from TF import TF
from optparse import OptionParser
parser = OptionParser()

parser.add_option("--sym_unet", action="store_true", dest="sym_unet", default=False)
parser.add_option("--box_size", dest="box_size", default=128)
parser.add_option("--warmup", dest="num_warmup", default=5, type="int")
parser.add_option("--iter", dest="num_iter", default=20, type="int")

if __name__ == "__main__":
	(options, args) = parser.parse_args()
	net =  TF()
        #Unet(symmetric=False, merge=True, activation="Relu", batchnorm=False, 
	#					 upsample=False, data_format='channels_first', kernel_shape=[[3, 64], [64, 128], [128, 256], [256, 512]])
  box_size = options.box_size 
  shp = (1,3,box_size, box_size, box_size)
  warmups    = options.num_warmup 
  iterations = options.num_iter
  
  durations = []
  for i in range(warmups + iterations):
		t = net.process()
		if i > warmpus:
			durations.append(t)
  average = sum(durations) / len(durations)
		
