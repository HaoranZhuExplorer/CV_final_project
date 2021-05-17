import numpy as np
import matplotlib.pyplot as plt
from skimage import io,filters,feature
from skimage.color import rgb2gray
import skimage.transform as st
import cv2 as cv

#find edges in the region of interests(mask)
def mask_edges(im_edges,vertices):
    mask = np.zeros((im_edges.shape));
    #fill the region bounded by the 4 vertices with 255(white)
    #other parts of the image remain to be 0
    cv.fillPoly(mask, vertices, 255);
    #find edges with in the region of interest bounded by vertices
    mask_edge_im = np.zeros((im_edges.shape));
    for i in range(im_edges.shape[0]):
        for j in range(im_edges.shape[1]):
            if(mask[i,j]!=0 and im_edges[i,j]!=0):
                mask_edge_im[i,j]=255;
            else:
                mask_edge_im[i,j]=0;
    return mask_edge_im;
    

#use hough transform to detect lines in mask_edge_im
def hough_trans_lines(mask_edge_im,threshold=10,line_length=50,line_gap=3):
    return st.probabilistic_hough_line(mask_edge_im,threshold,line_length,line_gap);
    

def main():
    #read the 2 images, turn them into gray images, and do a gaussain blur
    im1_path = 'lane1.jpg';
    im1 = io.imread(im1_path);
    im1 = rgb2gray(im1);
    im1 = filters.gaussian(im1,sigma=0.5);
    im2_path = 'lane2.jpg';
    im2 = io.imread(im2_path);
    im2 = rgb2gray(im2);
    im2 = filters.gaussian(im2,sigma=0.5);

    #use Canny edge detection to extract edges
    im1_edges = feature.canny(im1,sigma=3);
    im2_edges = feature.canny(im2,sigma=3);

    #Create the 4 vertices for the mask for the image,
    #which highlights the region we interest in for the image
    #(mask is the region that we hope to detect the lane)
    vertices1 = np.array([[(363,667),(0,787),(1037,850),(792,663)]]);
    vertices2 = np.array([[(450,458),(0,631),(1030,783),(900,455)]]);

    mask_edges1 = mask_edges(im1_edges,vertices1);
    mask_edges2 = mask_edges(im2_edges,vertices2);

    hough_lines1 = hough_trans_lines(mask_edges1,line_gap=15);
    hough_lines2 = hough_trans_lines(mask_edges2,line_length=130,line_gap=20);
                      
    fig1 = plt.figure(figsize=(7,7));
    plt.imshow(mask_edges1,cmap='gray');
    plt.title("Images1 edges detected manually");
    fig2 = plt.figure(figsize=(7,7));
    plt.imshow(mask_edges2,cmap='gray');
    plt.title("Images2 edges detected manually");
    
    #plot the detected lines on the original image
    fig3 = plt.figure(figsize=(7,7));
    plt.imshow(im1,cmap='gray');
    for line in hough_lines1:
        p0,p1 = line;
        plt.plot((p0[0],p1[0]),(p0[1],p1[1]),c='r');
    plt.title("Probabilistic Hough Transform(im1)");
    
    fig4 = plt.figure(figsize=(7,7));
    plt.imshow(im2,cmap='gray');
    for line in hough_lines2:
        p0,p1 = line;
        plt.plot((p0[0],p1[0]),(p0[1],p1[1]),c='r');
    plt.title("Probabilistic Hough Transform(im2)");

    plt.show();

main();
