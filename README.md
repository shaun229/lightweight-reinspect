Depdencies:
for process_day.py pip install azure

Control Parameters:
-threshold in process_image can be tuned according to desired presicion and recall, higher threshold higher precision, lower recall & vice-versa
-resolution of image, higher is better! I belive it will improve overlapping bbox cases, i.e. when ppl are close to each other
-training set, adding images even for different stores can improve results

TODO:

-Enable self learning:
For example, if the nnet detects a human w 95% conf, then it is a good image for a +ve sample
Another example, if nnet detects a bbox w 60% conf and has only moved a few pixels then its most likely a door frame
or a light pole, etc which means its a good image for a -ve sample

I tried doing this but always resulted in worse results, because even one bad image mistaken for a good +ve 
or good -ve sample can degrade the nnet.

conclusion = let nnet detect a good sample image for leanring but always require human confirmation (MTURK)  before adding to model


