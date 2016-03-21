'''create the mean file from a folder of images'''
import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="create image mean from folder of images for nnet")
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
        img = cv2.imread(path+'/'+res)
        print path+'/'+res
        if img == None:
            print 'ERROR'
            raw_input()

    avg_img = np.zeros((320,480,3))
    avg_img = avg_img.astype(float)
    for image in images:
        img= cv2.imread(path+'/'+image)
        nd_img = img.astype(float)
        print avg_img[0][0][0], img[0][0][0]
        avg_img += nd_img
        
        print avg_img[0][0][0]

    avg_img /= len(images)
    print avg_img
    np.save('mean.npy', avg_img)
        
        

 
main()

