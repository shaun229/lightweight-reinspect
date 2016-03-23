'''create the mean file from a folder of images'''
from cv2 import imread, imwrite
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="create image mean from idl file")
    parser.add_argument('idl_file')
    idl_file = parser.parse_args().idl_file

    path = '/'.join(idl_file.split('/')[:-1])

    images = []
    for line in open(idl_file):
        print 'line',line
        res = '/'.join(line.split('/')[-2:])#.split('"')[0]
        if res.split('"')[0] == '':
           res = res.split('"')[1]
        else:
	   res = res.split('"')[0]

        images.append(res)
        print res
        img = imread(path+'/'+res)
        print path+'/'+res
        if img == None:
            print 'ERROR'
            raw_input()

    avg_img = np.zeros((320,480,3))
    avg_img = avg_img.astype(float)
    for image in images:
        img= imread(path+'/'+image)
        nd_img = img.astype(float)
        avg_img += nd_img
        

    avg_img /= len(images)
    print avg_img
    np.save('mean.npy', avg_img)
    imwrite('mean.jpg', avg_img.astype('uint8'))    
        

 
main()

