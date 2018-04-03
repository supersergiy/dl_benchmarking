from TF import TF
from optparse import OptionParser
parser = OptionParser()

parser.add_option("--sym_unet", action="store_true", dest="sym_unet", default=False)
parser.add_option("--gpu", action="store_true", dest="gpu", default=False)
parser.add_option("--batchnorm", action="store_true", dest="batchnorm", default=False)
parser.add_option("--box_size", dest="box_size", default=128)
parser.add_option("--warmup", dest="num_warmup", default=5, type="int")
parser.add_option("--iter", dest="num_iter", default=20, type="int")
parser.add_option("--activ", dest="activ", default="relu")

if __name__ == "__main__":
	(options, args) = parser.parse_args()
	box_size = options.box_size
	shp = (1,3,box_size, box_size, box_size)
	warmups    = options.num_warmup
	iterations = options.num_iter
	net =  TF(batchnorm=options.batchnorm,
						shape=shp,
						activation=options.activ,
						gpu=options.gpu,
						symmetric=options.sym_unet)

	durations = []

	for i in range(warmups + iterations):
		t = net.process()
		if i > warmups:
			durations.append(t)
	average = sum(durations) / len(durations)
	print(average)
