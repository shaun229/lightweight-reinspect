'''process a day's worth of video for a given placement name and date'''
'''rm test_output/*.jpg; python bin/process_day.py 8100224 2016-02-09 -loi_in 260 150 170 40 -loi_out 260 150 170 40 -inout True False'''
from dateutil.parser import parse
from azure.storage.blob import BlobService
import argparse
from process_video import process_video

account_name = 'percolatastorage'
account_key = 'zKpwZUgkPNcZeFPo1JKDzTDU85AmuUPxfo2MQsJLXlibfVi9aNg3admwRL8YpJJS/DA2XPjjlZP6Xgf7TKH+Aw=='



def download_video(video_name, filename):
    '''download video for a given device to file, filename'''
    blob_service = BlobService(account_name, account_key)
    blob = blob_service.get_blob_to_path(
       'percolata-data',
        video_name,
        filename,
        max_connections=8,
    )
        

def get_video_names(placement_name, date):
    '''return list of video names for a given placement name and day, ex: 8100224, 2016-02-09'''
    blob_service = BlobService(account_name, account_key)
    gen =  blob_service.list_blobs('percolata-data', 'data/combined/video/' + \
            str(placement_name) + '/' + date.strftime('%Y-%m-%d'))

    video_names = [x.name for x in gen]
    return video_names


def process_day(placement_name, date, loi_in, loi_out, inout):
    '''download videos & process videos squentially'''
    video_names = sorted(get_video_names(placement_name, date))

    #skip the first 17*4 + 2 = 70
    video_names = video_names[70:]

    for video_name in video_names:
        print 'Processing video: ' + video_name
        download_video(video_name, 'temp_video.mp4')
        walkin, walkout, _ = process_video(loi_in, loi_out, inout, 'temp_video.mp4')
        f = open('counts.txt', 'a')
        f.write(video_name + ':' + str(walkin) + ',' + str(walkout) + '\n')
        f.close()

def str2boolINOUT(INOUT):
    '''convert str INOUT to bool INOUT'''
    INOUT[0] = True if INOUT[0] == 'True' else False
    INOUT[1] = True if INOUT[1] == 'True' else False
    return INOUT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse video using nnet")
    parser.add_argument('placement_name')
    parser.add_argument('date')
    parser.add_argument('-loi_in',  nargs='+', type=int)    
    parser.add_argument('-loi_out', nargs='+', type=int)        
    parser.add_argument('-inout', nargs='+')
    args = parser.parse_args()    
    process_day(args.placement_name, parse(args.date), args.loi_in, args.loi_out, str2boolINOUT(args.inout))
