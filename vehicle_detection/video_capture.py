import os.path as osp
import os
import cv2
from PIL import Image

    
def main():
    video_in = cv2.VideoCapture("vehicles_video.mp4");

    fps = int(video_in.get(cv2.CAP_PROP_FPS));
    
    #create the output video
    #fourcc = cv2.VideoWriter_fourcc(*'XVID');
    #video_out = cv2.VideoWriter('video_out.mp4', fourcc, fps, size);

    #read a frame
    success, frame = video_in.read();
    frames_detected = 1;
    out_frame_num = 0;
    #continue reading frames until there is no frame to read
    while success:
        frames_detected+=1;
        #save a frame as png per second
        if(frames_detected == fps):
            out_frame_num +=1;
            frame_num = str(out_frame_num);
            im = Image.fromarray(frame);
            im.save(frame_num+".png");
            frames_detected = 0;          
        #video_out.write(frame);
        success, frame = video_in.read();

main();
