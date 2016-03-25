Uses ReInspect (https://github.com/Russell91/ReInspect) to enable tracking in videos. 

Overview:
-config.json: contains parameters for the nnet to run. 
Important fields:
	-train/test_idl: idl file is the dataset, the file contains the path to an image and the bounding boxes
	-idl_mean: npy file that contains the mean values of the dataset, used to center the input 
	-weights: path to the weights file, this is tied to the idl_mean file you used while generating the weights file
	-grid_height: image_height/32, ex: current images are 320x480x3, thus grid_height = 320/32 = 10. update this if you change the resolution
	-grid_width: image_width/32
-lib/nnet.py: contains all code interacting with the nnet
-utils/: contains helper functions for lib/nnet.py
-weights/: stores the different weights, remember that each weights file must match the mean you used while training it in config.json
-data/: contains the idl files, the images, and the mean npy files
-bin/process_video.py: contains tracking that uses the nnet to track people, used to process a local video
-bin/process_day.py: uses process_video to process an entire day's worth of video from azure storage using a specified placement name
-bin/create_mean.py: takes an idl file as input and generates a mean.npy for centering the data set (currenlty I dont center, i.e. use a zero-mean)

To train use, make train. To generate test outputuse make test, it will write to a folder named test_output2.
To process a local video use process_video, it can write the processed frames to test_output, which you can convert to video using ffmpeg.
ex: ffmpeg -framerate 10 -i test_output/frame%000d.jpg output.mp4

Depdencies:
-nnet: http://apollocaffe.com/#Installation
-others: cv2, numpy, scipy
-for process_day.py pip install azure

Control Parameters:
-threshold in process_image can be tuned according to desired presicion and recall, higher threshold higher precision, lower recall & vice-versa
-resolution of image, higher is better! I belive it will improve overlapping bbox cases, i.e. when ppl are close to each other
-training set, adding images even for different stores can improve results

TODO:
-Integrate into the pipeline:
Kafka, database access, and supervisor

-Look at porting the code to Caffe:
Caffe has a branch that supports OpenCL. The only concern is that there are some important nnet units that are
not supported by Caffe. Mainly the LSTM units. It should be possible to implement these units ourselves into 
the Caffe branch by studying the implementation in ApolloCaffe.

-Optimize code:
Python is very slow, perhaps some of it can be ported to faster lanugaues. Lei has experience working with Go to improve runtime
perhaps he can lead that effort. 

-Using motion detection with nnet:
Use background substraction to determine if a frame is worth processing, if nothing moves, we can save computation time

-Enable self learning:
For example, if the nnet detects a human w 95% conf, then it is a good image for a +ve sample
Another example, if nnet detects a bbox w 60% conf and has only moved a few pixels then its most likely a door frame
or a light pole, etc which means its a good image for a -ve sample

I tried doing this but always resulted in worse results, because even one bad image mistaken for a good +ve 
or good -ve sample can degrade the nnet.

Conclusion:
Let nnet detect a good sample image for leanring, launch a MTURK request to have the image labelled correctly before
learning from it (ie. calling train_single_frame)`.

This will allow the nnet to adapt to changes such as summer to winter, shadows, new obstructions in the view, etc automatically
without requring any input from engineering. It also proactively prevents drops in accuracy over time.

-Automated Correction of Bad Counts:
When we detect counts that are way off, there needs to be an automated way to correct the counts. 

High Level Steps:
1. Use the nnet or any other tracking algo to process the video first. When a walkin/walkout event is detected, pull a few frames 
around that event from the video and add it an MTURK job.
2. Once the entire video is processed, send the images to be labelled by MTURK. Add the images back to the training set and 
retrain the weights for 2-3 hours. If you are centering the images, make sure to update the mean before retraining. Use the 
retrained weights to reprocess the video. 
3. Repeat the process until the counts in the database are accurate enough

Advantages: 
-No human interaction == engineering time saved

Disadvantages:
-MTURK costs
-Takes hours to fix the counts
-May not fix all errors, such as obstructions in the view, incorrect config files, etc

We can also extend this idea to automate onboarding for new stores. Use the current weights file for a new store, and let
the accuracy improve gradually.

