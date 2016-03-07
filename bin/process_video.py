import cv2
from lib import nnet
import argparse


DIST_THRESHOLD = 50

def detect_line_trip(bbox1, bbox2, x1, x2, y):
    '''detect if a horizontal line was tripped'''
    if (bbox1[1] >= y and bbox2[1] <= y) or \
            (bbox1[1] <= y and bbox2[1] >= y) and \
            (x1 <= bbox1[0] and bbox1[0] <= x2) and \
            (x1 <= bbox2[0] and bbox2[0] <= x2):
        return True
    else:
        return False

def detect_loi_line_trip(new_bbox, prev_bbox, LOI_BOX_IN, LOI_BOX_OUT, INOUT):
    '''returns tuple of loi trip status, walkin and walkout'''
    in_status = 'None'
    out_status = 'None'
    if INOUT[0]: #vertical orientation
        #walkin status
        in_line1_status = detect_line_trip(new_bbox, prev_bbox, LOI_BOX_IN[0], LOI_BOX_IN[0]+LOI_BOX_IN[2], LOI_BOX_IN[1])
        in_line2_status = detect_line_trip(new_bbox, prev_bbox, LOI_BOX_IN[0], LOI_BOX_IN[0]+LOI_BOX_IN[2], LOI_BOX_IN[1]+LOI_BOX_IN[3])
        
        #walkout status
        out_line1_status = detect_line_trip(new_bbox, prev_bbox, LOI_BOX_OUT[0], LOI_BOX_OUT[0]+LOI_BOX_OUT[2], LOI_BOX_OUT[1])
        out_line2_status = detect_line_trip(new_bbox, prev_bbox, LOI_BOX_OUT[0], LOI_BOX_OUT[0]+LOI_BOX_OUT[2], LOI_BOX_OUT[1]+LOI_BOX_OUT[3])

    else: #horizontal orientation
        temp_new_bbox = [new_bbox[1], new_bbox[0]]
        temp_prev_bbox = [prev_bbox[1], prev_bbox[0]]

        #walkin status
        in_line1_status = detect_line_trip(new_bbox, prev_bbox, LOI_BOX_IN[1], LOI_BOX_IN[1]+LOI_BOX_IN[3], LOI_BOX_IN[0])
        in_line2_status = detect_line_trip(new_bbox, prev_bbox, LOI_BOX_IN[1], LOI_BOX_IN[1]+LOI_BOX_IN[3], LOI_BOX_IN[0]+LOI_BOX_IN[2])
        
        #walkout status
        out_line1_status = detect_line_trip(new_bbox, prev_bbox, LOI_BOX_OUT[1], LOI_BOX_OUT[1]+LOI_BOX_OUT[3], LOI_BOX_OUT[0])
        out_line2_status = detect_line_trip(new_bbox, prev_bbox, LOI_BOX_OUT[1], LOI_BOX_OUT[1]+LOI_BOX_OUT[3], LOI_BOX_OUT[0]+LOI_BOX_OUT[2])        

    #update status
    if in_line1_status or in_line2_status:
        if in_line1_status == in_line2_status: #both lines tripped
            in_status = 'Both'
        elif in_line1_status:
            in_status = 'Top'
        else:
            in_status = 'Bottom'
        
    if out_line1_status or out_line2_status:
        if out_line1_status == out_line2_status: #both lines tripped
            out_status = 'Both'
        elif out_line1_status:
            out_status = 'Top'
        else:
            out_status = 'Bottom'

    return [in_status, out_status]

def getDistance(bbox1, bbox2):
    '''return euclidean distance b/w bbox centers'''
    return ((bbox1[0] - bbox2[0])**2 + (bbox1[1] - bbox2[1])**2)**0.5

def getClosestBBoxPair(new_bboxes, prev_bboxes):
    '''finds closest new bboxes to the old bboxes'''
    closest_new_bbox = None
    closest_prev_bbox = None
    min_dist = -1
    for new_bbox in new_bboxes:
        for prev_bbox in prev_bboxes:
            distance = getDistance(new_bbox, prev_bbox)
            if min_dist == -1 or min_dist > distance:
                closest_new_bbox = new_bbox
                closest_prev_bbox = prev_bbox
                min_dist = distance
    return [closest_new_bbox, closest_prev_bbox, min_dist]

def process_loi_status(status, dir_bool, prev_bbox, prev_status, LOI_BOX):
    '''detect if a walkin/walkout event occured and update loi status'''

    assert status in ['Both', 'Top', 'Bottom', 'None']
    assert prev_status in ['Both', 'Top', 'Bottom', 'None']

    traffic_event = 0 #can be a walkin or walkout depending on dir_bool & LOI_BOX
    if status == 'Both':
        if dir_bool and (prev_bbox[1] >= LOI_BOX[1]+LOI_BOX[3] or
                prev_bbox[0] <= LOI_BOX[0]):
            traffic_event+=1
        elif not dir_bool and (prev_bbox[1] <= LOI_BOX[1] or
                prev_bbox[0] >= LOI_BOX[0]+LOI_BOX[2]):
            traffic_event+=1
        status = 'None'
    elif dir_bool and status == 'Top' and prev_status == 'Bottom':
        traffic_event+=1
        status = 'None'
    elif not dir_bool and status == 'Bottom' and prev_status == 'Top':
        traffic_event+=1
        status = 'None'
    elif status == 'None':
        status = prev_status

    assert traffic_event <= 1 
    assert status in ['Both', 'Top', 'Bottom', 'None']

    return [status, traffic_event]
        

def process_frame(frame, prev_bboxes, prev_loi_status, LOI_BOX_IN, LOI_BOX_OUT, INOUT):
    '''Detect walkin/walkout if two lines are tripped and update loi status'''

    new_bboxes = nnet.process_frame(frame)
    new_loi_status = {}
    new_prev_bboxes = []

    walkin = 0
    walkout = 0

    while len(new_bboxes) != 0:
        closest_new_bbox, closest_prev_bbox, dist = getClosestBBoxPair(new_bboxes, prev_bboxes)

        if closest_prev_bbox == None or dist > DIST_THRESHOLD:            
            break #closest_new_bbox is a new bbox/newly detected person

        #determine which lines were tripped -> loi status    
        in_status, out_status = detect_loi_line_trip(closest_new_bbox, closest_prev_bbox, LOI_BOX_IN, LOI_BOX_OUT, INOUT)

        prev_in_status = prev_loi_status[closest_prev_bbox][0] if closest_prev_bbox in prev_loi_status else 'None'
        prev_out_status = prev_loi_status[closest_prev_bbox][1] if closest_prev_bbox in prev_loi_status else 'None'
        #process loi status
        in_count, in_status = process_loi_status(in_status, INOUT[1], closest_prev_bbox, prev_in_status, LOI_BOX_IN)
        out_count, out_status = process_loi_status(out_status, not INOUT[1], closest_prev_bbox, prev_out_status, LOI_BOX_OUT)
        
        #update walkin and walkout
        walkin += in_count
        walkout += out_count    
            
        #update loi status for next frame
        new_loi_status[closest_new_bbox] = [in_status, out_status]
        #update prev_bboxes for next frame
        new_prev_bboxes.append(closest_new_bbox)

        #remove bboxes 
        new_bboxes.remove(closest_new_bbox)
        prev_bboxes.remove(closest_prev_bbox)
    
    #process bboxes which are new
    for bbox in new_bboxes:
        new_loi_status[bbox] = ['None', 'None']
    #add any remaining bboxes
    new_prev_bboxes.extend(new_bboxes)

    return (walkin, walkout, new_prev_bounding_boxes, new_loi_status)


def process_video(LOI_BOX_IN, LOI_BOX_OUT, INOUT, video_file):
    """
        Core function.  Performs in-out counting on video
    ARGS:
        LOI_BOX_IN, LOI_BOX_OUT: 4-tuples consisting of x- and y-coordinates of the
            top-left corners of LoI boxes along with their widths and heights.
        INOUT: 2-tuple of booleans indicating direction and orientation of the LoI.
            First entry is True if in/out are vertical and False if in/out are horizontal.
            Second entry is True if in is up or right and False if in is down or left.
    RETURNS:
        Tuple of in_count and out_count
    """

    walkin = 0
    walkout = 0

    vidcap = cv2.VideoCapture(video_file)
    success, frame = vidcap.read()
    prev_bboxes = {}
    prev_loi_status = {}

    while success:
        win, wout, prev_bboxes, prev_loi_status = process_frame(frame, prev_bboxes, prev_loi_status, LOI_BOX_IN, LOI_BOX_OUT, INOUT)
        walkin += win
        walkout += wout
        sucess, frame = vidcap.read()
        print walkin, walkout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse video using nnet")
    parser.add_argument('video_file')
    parser.add_argument('-loi_in',  nargs='+', type=int)    
    parser.add_argument('-loi_out', nargs='+', type=int)        
    parser.add_argument('-inout', nargs='+', type=bool)
    args = parser.parse_args()    
    print args.loi_in   
    process_video(args.loi_in, args.loi_out, args.inout, args.video_file)


