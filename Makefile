SHELL := /bin/bash

.PHONY: all
all:
	@echo -e "Usage:\n\
	$$ make train         # train the network on the CPU \n\"

.PHONY: train
train: utils/stitch_wrapper.so data
	@echo Training...
	python lib/nnet.py --config config.json --gpu 0


.PHONY: train_cpu
train_cpu:
	python lib/nnet.py --config config.json 

.PHONY: test
test:
	python lib/nnet.py --config config.json --gpu 0 --test True

.PHONY: test_cpu
test_cpu:
	python lib/nnet.py --config config.json  --test True

.PHONY: clean
clean:
	rm -f utils/stitch_wrapper.so

utils/stitch_wrapper.so:
	cd utils && makecython++ stitch_wrapper.pyx "" "stitch_rects.cpp ./hungarian/hungarian.cpp"

