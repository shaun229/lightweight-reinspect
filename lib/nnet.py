"""nnet.py is used to generate and train the ReInspect deep network architecture and 
process test images for detection"""

import time
import cv2
import numpy as np
import json
import os
import random
from scipy.misc import imread
import apollocaffe
from apollocaffe.models import googlenet
from apollocaffe.layers import (Power, LstmUnit, Convolution, NumpyData,
                                Transpose, Filler, SoftmaxWithLoss,
                                Softmax, Concat, Dropout, InnerProduct)

from utils import (annotation_jitter, image_to_h5,
                   annotation_to_h5, load_data_mean, Rect, stitch_rects)
from utils.annolist import AnnotationLib as al


def load_idl(idlfile, data_mean, net_config, jitter=False, train=False):
    """Take the idlfile, data mean and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""

    annolist = al.parse(idlfile)
    annos = [x for x in annolist]
    for anno in annos:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
    while True:
        #in video mode, we can't randomly shuffle the inputs
        #random.shuffle(annos)
        for anno in annos:
            if jitter:
                jit_image, jit_anno = annotation_jitter(
                    anno, target_width=net_config["img_width"],
                    target_height=net_config["img_height"])
            else:
                jit_image = imread(anno.imageName)
                jit_anno = anno
            image = image_to_h5(jit_image, data_mean, image_scaling=1.0)
            boxes, box_flags = annotation_to_h5(
                jit_anno, net_config["grid_width"], net_config["grid_height"],
                net_config["region_size"], net_config["max_len"])
            yield {"imname": anno.imageName, "raw": jit_image, "image": image,
                   "boxes": boxes, "box_flags": box_flags, "anno": jit_anno}
        #when the training video is done, we have to rest the memory seeds to empty again
        if train:
            global LSTM_HIDDEN_SEED
            global LSTM_MEM_SEED
            LSTM_HIDDEN_SEED = None
            LSTM_MEM_SEED = None

def generate_decapitated_googlenet(net, net_config):
    """Generates the googlenet layers until the inception_5b/output.
    The output feature map is then used to feed into the lstm layers."""

    google_layers = googlenet.googlenet_layers()
    google_layers[0].p.bottom[0] = "image"
    for layer in google_layers:
        if "loss" in layer.p.name:
            continue
        if layer.p.type in ["Convolution", "InnerProduct"]:
            for p in layer.p.param:
                p.lr_mult *= net_config["googlenet_lr_mult"]

        #try reducing image size more agressively during first pooling phase
        if layer.p.name == "pool1/3x3_s2":
           net.f('''name: "pool1/3x3_s2"
                type: "Pooling"
                bottom: "conv1/7x7_s2"
                top: "pool1/3x3_s2"
                pooling_param {
                  pool: MAX
                  kernel_size: 3
                  stride: 4
                }''')
           continue #skip that layer

        if layer.p.name == "pool2/3x3_s2":
            net.f('''
            name: "pool2/3x3_s2"
            type: "Pooling"
            bottom: "conv2/norm2"
            top: "pool2/3x3_s2"
            pooling_param {
              pool: MAX
              kernel_size: 5
              stride: 4
            }''')
            continue

        if layer.p.name == "inception_3a/3x3_reduce":
           net.f("""
            name: "inception_3a/3x3_reduce"
            type: "Convolution"
            bottom: "pool2/3x3_s2"
            top: "inception_3a/3x3_reduce"
            param {
            lr_mult: 1
            decay_mult: 1
            }
            param {
            lr_mult: 2
            decay_mult: 0
            }
            convolution_param {
            num_output: 96
            kernel_size: 1
            weight_filler {
              type: "xavier"
              std: 0.09
            }
            bias_filler {
              type: "constant"
              value: 0.2
            }
            }""")
           continue
 
        if layer.p.name == "inception_3a/3x3":
           net.f("""
            name: "inception_3a/3x3"
            type: "Convolution"
            bottom: "inception_3a/3x3_reduce"
            top: "inception_3a/3x3"
            param {
            lr_mult: 1
            decay_mult: 1
            }
            param {
            lr_mult: 2
            decay_mult: 0
            }
            convolution_param {
            num_output: 96
            pad: 1
            kernel_size: 3
            weight_filler {
              type: "xavier"
              std: 0.03
            }
            bias_filler {
              type: "constant"
              value: 0.2
            }
            }""")
           continue

        if "5x5" in layer.p.name and "3a" in layer.p.name:
            continue

        if layer.p.name == "inception_3a/output":
            net.f('''
            name: "inception_3a/output"
            type: "Concat"
            bottom: "inception_3a/1x1"
            bottom: "inception_3a/3x3"
            bottom: "inception_3a/pool_proj"
            top: "inception_3a/output"
            ''')
            break

        net.f(layer)


def generate_intermediate_layers(net):
    """Takes the output from the decapitated googlenet and transforms the output
    from a NxCxWxH to (NxWxH)xCx1x1 that is used as input for the lstm layers.
    N = batch size, C = channels, W = grid width, H = grid height."""

    net.f(Convolution("post_fc7_conv", bottoms=["inception_3a/output"],
                      param_lr_mults=[1., 2.], param_decay_mults=[0., 0.],
                      num_output=1024, kernel_dim=(1, 1),
                      weight_filler=Filler("gaussian", 0.005),
                      bias_filler=Filler("constant", 0.)))
    net.f(Power("lstm_fc7_conv", scale=0.01, bottoms=["post_fc7_conv"]))
    net.f(Transpose("lstm_input", bottoms=["lstm_fc7_conv"]))

def generate_ground_truth_layers(net, box_flags, boxes):
    """Generates the NumpyData layers that output the box_flags and boxes
    when not in deploy mode. box_flags = list of bitstring (e.g. [1,1,1,0,0])
    encoding the number of bounding boxes in each cell, in unary,
    boxes = a numpy array of the center_x, center_y, width and height
    for each bounding box in each cell."""

    old_shape = list(box_flags.shape)
    new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
    net.f(NumpyData("box_flags", data=np.reshape(box_flags, new_shape)))

    old_shape = list(boxes.shape)
    new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
    net.f(NumpyData("boxes", data=np.reshape(boxes, new_shape)))

def generate_lstm_seeds(net, num_cells):
    """Generates the lstm seeds that are used as
    global LSTM_MEM_SEED 
    input to the first lstm layer."""


    global LSTM_HIDDEN_SEED
    global LSTM_MEM_SEED
    #Test always forgetting
    if True: #LSTM_HIDDEN_SEED == None: 
        LSTM_HIDDEN_SEED = np.zeros((net.blobs["lstm_input"].shape[0], num_cells)) 
    if True: #LSTM_MEM_SEED == None: 
        LSTM_MEM_SEED = np.zeros((net.blobs["lstm_input"].shape[0], num_cells))
    net.f(NumpyData("lstm_hidden_seed", LSTM_HIDDEN_SEED))
    net.f(NumpyData("lstm_mem_seed", LSTM_MEM_SEED))

def get_lstm_params(step):
    """Depending on the step returns the corresponding
    hidden and memory parameters used by the lstm."""

    if step == 0:
        return ("lstm_hidden_seed", "lstm_mem_seed")
    else:
        return ("lstm_hidden%d" % (step - 1), "lstm_mem%d" % (step - 1))

def generate_lstm(net, step, lstm_params, lstm_out, dropout_ratio):
    """Takes the parameters to create the lstm, concatenates the lstm input
    with the previous hidden state, runs the lstm for the current timestep
    and then applies dropout to the output hidden state."""

    hidden_bottom = lstm_out[0]
    mem_bottom = lstm_out[1]
    num_cells = lstm_params[0]
    filler = lstm_params[1]
    net.f(Concat("concat%d" % step, bottoms=["lstm_input", hidden_bottom]))
    try:
        lstm_unit = LstmUnit("lstm%d" % step, num_cells,
                       weight_filler=filler, tie_output_forget=True,
                       param_names=["input_value", "input_gate",
                                    "forget_gate", "output_gate"],
                       bottoms=["concat%d" % step, mem_bottom],
                       tops=["lstm_hidden%d" % step, "lstm_mem%d" % step])
    except:
        # Old version of Apollocaffe sets tie_output_forget=True by default
        lstm_unit = LstmUnit("lstm%d" % step, num_cells,
                       weight_filler=filler,
                       param_names=["input_value", "input_gate",
                                    "forget_gate", "output_gate"],
                       bottoms=["concat%d" % step, mem_bottom],
                       tops=["lstm_hidden%d" % step, "lstm_mem%d" % step])
    net.f(lstm_unit)
    net.f(Dropout("dropout%d" % step, dropout_ratio,
                  bottoms=["lstm_hidden%d" % step]))

def generate_inner_products(net, step, filler):
    """Inner products are fully connected layers. They generate
    the final regressions for the confidence (ip_soft_conf),
    and the bounding boxes (ip_bbox)"""

    net.f(InnerProduct("ip_conf%d" % step, 2, bottoms=["dropout%d" % step],
                       output_4d=True,
                       weight_filler=filler))
    net.f(InnerProduct("ip_bbox_unscaled%d" % step, 4,
                       bottoms=["dropout%d" % step], output_4d=True,
                       weight_filler=filler))
    net.f(Power("ip_bbox%d" % step, scale=100,
                bottoms=["ip_bbox_unscaled%d" % step]))
    net.f(Softmax("ip_soft_conf%d" % step, bottoms=["ip_conf%d"%step]))

def generate_losses(net, net_config):
    """Generates the two losses used for ReInspect. The hungarian loss and
    the final box_loss, that represents the final softmax confidence loss"""

    net.f("""
          name: "hungarian"
          type: "HungarianLoss"
          bottom: "bbox_concat"
          bottom: "boxes"
          bottom: "box_flags"
          top: "hungarian"
          top: "box_confidences"
          top: "box_assignments"
          loss_weight: %s
          hungarian_loss_param {
            match_ratio: 0.5
            permute_matches: true
          }""" % net_config["hungarian_loss_weight"])
    net.f(SoftmaxWithLoss("box_loss",
                          bottoms=["score_concat", "box_confidences"]))

def forward(net, input_data, net_config, deploy=False):
    """Defines and creates the ReInspect network given the net, input data
    and configurations."""

    net.clear_forward()
    if deploy:
        image = np.array(input_data["image"])
    else:
        image = np.array(input_data["image"])
        box_flags = np.array(input_data["box_flags"])
        boxes = np.array(input_data["boxes"])

    net.f(NumpyData("image", data=image))
    tic = time.time()
    generate_decapitated_googlenet(net, net_config)
#    print "decap pass", time.time() - tic
    generate_intermediate_layers(net)
    if not deploy:
        generate_ground_truth_layers(net, box_flags, boxes)
    generate_lstm_seeds(net, net_config["lstm_num_cells"])

    filler = Filler("uniform", net_config["init_range"])
    concat_bottoms = {"score": [], "bbox": []}
    lstm_params = (net_config["lstm_num_cells"], filler)
    for step in range(net_config["max_len"]):
        lstm_out = get_lstm_params(step)
        generate_lstm(net, step, lstm_params,
                      lstm_out, net_config["dropout_ratio"])
        generate_inner_products(net, step, filler)

        concat_bottoms["score"].append("ip_conf%d" % step)
        concat_bottoms["bbox"].append("ip_bbox%d" % step)

    net.f(Concat("score_concat", bottoms=concat_bottoms["score"], concat_dim=2))
    net.f(Concat("bbox_concat", bottoms=concat_bottoms["bbox"], concat_dim=2))

    if not deploy:
        generate_losses(net, net_config)

    if deploy:
        bbox = [np.array(net.blobs["ip_bbox%d" % j].data)
                for j in range(net_config["max_len"])]
        conf = [np.array(net.blobs["ip_soft_conf%d" % j].data)
                for j in range(net_config["max_len"])]
        return (bbox, conf)
    else:
        return None

def test(config):
    """Test the model and output to test/output."""

    net = apollocaffe.ApolloNet()

    net_config = config["net"]
    data_config = config["data"]
    solver = config["solver"]

    image_mean = load_data_mean(
        data_config["idl_mean"], net_config["img_width"],
        net_config["img_height"], image_scaling=1.0)

    input_gen_test = load_idl(data_config["test_idl"],
                                   image_mean, net_config, jitter=False)

    forward(net, input_gen_test.next(), config["net"])

    try:
        net.load(solver["weights"])
    except:
        pass

    net.phase = 'test'
    test_loss = []
    for i in range(solver["test_iter"]):
        input_test = input_gen_test.next()
        image = input_test["raw"]
        print input_test["anno"]
        tic = time.time()
        bbox, conf = forward(net, input_test, config["net"], True)
        print "forward deploy time", time.time() - tic
        bbox_list = bbox
        conf_list = conf
        pix_per_w = 32
        pix_per_h = 32

        all_rects = [[[] for x in range(net_config['grid_width'])] for y in range(net_config['grid_height'])]
        for n in range(len(bbox_list)):
            for k in range(net_config['grid_width'] * net_config['grid_height']):
                y = int(k / net_config['grid_width'])
                x = int(k % net_config['grid_width'])
                bbox = bbox_list[n][k]
                conf = conf_list[n][k,1].flatten()[0]
                abs_cx = pix_per_w/2 + pix_per_w*x + int(bbox[0,0,0])
                abs_cy = pix_per_h/2 + pix_per_h*y+int(bbox[1,0,0])
                w = bbox[2,0,0]
                h = bbox[3,0,0]
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))
        acc_rects = stitch_rects(all_rects)
        #print acc_rects

        for idx, rect in enumerate(acc_rects):
            if rect.true_confidence < 0.8:
                print 'rejected', rect.true_confidence
                continue
            else:
                print 'found', rect.true_confidence
                cv2.rectangle(image, (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                                   (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                                   (255,0,0),
                                   2)
        cv2.imwrite("test_output2/img_out%s.jpg" % i, image)

def train(config):
    """Trains the ReInspect model using SGD with momentum
    and prints out the logging information."""

    net = apollocaffe.ApolloNet()

    net_config = config["net"]
    data_config = config["data"]
    solver = config["solver"]
    logging = config["logging"]

    image_mean = load_data_mean(
        data_config["idl_mean"], net_config["img_width"],
        net_config["img_height"], image_scaling=1.0)

    input_gen = load_idl(data_config["train_idl"],
                              image_mean, net_config, jitter=False, train=True)
    input_gen_test = load_idl(data_config["test_idl"],
                                   image_mean, net_config, jitter=False)

    forward(net, input_gen.next(), config["net"])

    try:
        net.load(solver["weights"])
    except:
        pass

    loss_hist = {"train": [], "test": []}
    loggers = [
        apollocaffe.loggers.TrainLogger(logging["display_interval"],
                                        logging["log_file"]),
        apollocaffe.loggers.TestLogger(solver["test_interval"],
                                       logging["log_file"]),
        apollocaffe.loggers.SnapshotLogger(logging["snapshot_interval"],
                                           logging["snapshot_prefix"]),
        ]
    for i in range(solver["start_iter"], solver["max_iter"]):
        if i % solver['test_interval'] == 0:
            net.save(solver["weights"])
            print "WEIGHTS SAVED"

        #disable testing to prevent lstm memory corruption
        if i % solver["test_interval"] == -1:
            net.phase = 'test'
            test_loss = []
            #save the weights
            net.save(solver["weights"])

            for x in range(solver["test_iter"]):
                test_input_data = input_gen_test.next()
                tic = time.time()
                forward(net, test_input_data, config["net"], False)
                print "Forward pass", time.time() - tic
                test_loss.append(net.loss)

            loss_hist["test"].append(np.mean(test_loss))
            net.phase = 'train'

        forward(net, input_gen.next(), config["net"])
        loss_hist["train"].append(net.loss)
        net.backward()
        learning_rate = (solver["base_lr"] *
                         (solver["gamma"])**(i // solver["stepsize"]))
        net.update(lr=learning_rate, momentum=solver["momentum"],
                   clip_gradients=solver["clip_gradients"])
        for logger in loggers:
            logger.log(i, {'train_loss': loss_hist["train"],
                           'test_loss': loss_hist["test"],
                           'apollo_net': net, 'start_iter': 0})


def process_frame(frame):
    '''process uinsg reinspect'''
    config = json.load(open('config.json', 'r'))
    
    net = apollocaffe.ApolloNet()

    net_config = config["net"]
    data_config = config["data"]
    solver = config["solver"]

    image_mean = load_data_mean(
        data_config["idl_mean"], net_config["img_width"],
        net_config["img_height"], image_scaling=1.0)

    input_gen_test = load_idl(data_config["test_idl"],
                                   image_mean, net_config, jitter=False)

    image  = image_to_h5(frame, image_mean, image_scaling=1.0)

    input_test = {"imname": '', "raw": frame, "image": image}


    forward(net, image, config["net"], True)

    net.load(solver["weights"])

    net.phase = 'test'
    bbox, conf = forward(net, input_test, config["net"], True)
    bbox_list = bbox
    conf_list = conf
    pix_per_w = net_config['img_width']/net_config['grid_width']
    pix_per_h = net_config['img_height']/net_config['grid_height']

    all_rects = [[[] for x in range(net_config['grid_width'])] for y in range(net_config['grid_height'])]
    for n in range(len(bbox_list)):
        for k in range(net_config['grid_width'] * net_config['grid_height']):
            y = int(k / net_config['grid_width'])
            x = int(k % net_config['grid_width'])
            bbox = bbox_list[n][k]
            conf = conf_list[n][k,1].flatten()[0]
            abs_cx = pix_per_w/2 + pix_per_w*x + int(bbox[0,0,0])
            abs_cy = pix_per_h/2 + pix_per_h*y+int(bbox[1,0,0])
            w = bbox[2,0,0]
            h = bbox[3,0,0]
            all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))
    acc_rects = stitch_rects(all_rects)
    #print acc_rects

    for idx, rect in enumerate(acc_rects):
        if rect.true_confidence < 0.8:
            print 'rejected', rect.true_confidence
            continue
        else:
            print 'found', rect.true_confidence
            cv2.rectangle(image, (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                               (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                               (255,0,0),
                               2)
    cv2.imwrite("test_output2/img_out%s.jpg" % i, image)

def main():
    """Sets up all the configurations for apollocaffe, and ReInspect
    and runs the trainer."""
    parser = apollocaffe.base_parser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--test')
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    if args.weights is not None:
        config["solver"]["weights"] = args.weights
    config["solver"]["start_iter"] = args.start_iter
    apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(args.gpu)
    apollocaffe.set_cpp_loglevel(args.loglevel)

    global LSTM_HIDDEN_SEED 
    global LSTM_MEM_SEED 
    LSTM_HIDDEN_SEED = None
    LSTM_MEM_SEED = None

    if args.test is not None:
        test(config)
    else:
        train(config)

if __name__ == "__main__":
    main()
