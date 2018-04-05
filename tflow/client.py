from TF import TF
from optparse import OptionParser
parser = OptionParser()

#parser.add_option("--sym_unet", action="store_true", dest="sym_unet", default=False)
parser.add_option("--gpu", action="store_true", dest="gpu", default=False)
parser.add_option("--batchnorm", action="store_true", dest="batchnorm", default=False)
#parser.add_option("--box_size", dest="box_size", default=128)
parser.add_option("--warmup", dest="num_warmup", default=7, type="int")
parser.add_option("--iter", dest="num_iter", default=30, type="int")
parser.add_option("--activ", dest="activ", default="relu")
parser.add_option("--optimize", action="store_true", dest="optimize", default=False)
parser.add_option("--threads", dest="threads", default=16, type="int")
#parser.add_option("--merge", action="store_true", dest="merge", default=False)

parser.add_option("--model", dest="model", default="original") # symmetric, residual

if __name__ == "__main__":
	(options, args) = parser.parse_args()

	warmups    = options.num_warmup
	iterations = options.num_iter

	if options.model=="original":
		merge, symmetric, residual = True, False, False
		shp = (1, 3, 108, 108, 108)
		threads = 16 # 2 cores
	if options.model=="symmetric":
		merge, symmetric, residual = False, True, False
		shp = (1, 3, 128, 128, 128)
		threads = 16 # 4 cores
	if options.model=="residual":
		shp = (1, 3, 16, 192, 192)
		threads = 16 # 2 cores
		merge, symmetric, residual = False, True, True

	net =  TF(merge = merge,
			  batchnorm=options.batchnorm,
			  shape=shp,
			  activation=options.activ,
			  gpu=options.gpu,
			  symmetric=symmetric,
			  optimize= options.optimize,
			  residual=residual,
			  threads=threads)

	durations = []

	for i in range(warmups + iterations):
		t = net.process()
		print(t)
		if i > warmups:
			durations.append(t)
	average = sum(durations) / len(durations)
	print(average)
