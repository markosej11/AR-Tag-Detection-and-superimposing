ENPM673: Perception for Autonomous Robotics Project 1

#### Instructions to run the program for Project 1
File name: ENPM673_project_1.py

Two data directories are used: ReferenceImages and VideoDataset.
Python 3 is required.

ReferenceImages contains:
1) Lena.png
2) ref_marker.png
3) ref_marker_grid.png

VideoDataset contains:
1) Tag0.mp4
2) Tag1.mp4
3) Tag2.mp4
4) multipleTags.mp4

Running Instructions:
1) Make sure the directories containing the Lena image (ReferenceImages) and videos (VideoDataset) are in the same folder as the python program.
2) When you run the video the program will ask you which video you would like to play. Enter 1 for Tag0, enter 2 for Tag1, enter 3 for Tag2 or enter 4 for multiple tags. 
3) The program will destroy all the windows once the video has finished playing.

We imported the following two libraries:
1) numpy 
2) cv2

Results:
Based on your choice of video, you will see a window with the tag ID, superimposed Lena image and 3D projection of a cube.


Video file:
The video output of our program can be found in the below link:
https://drive.google.com/drive/folders/19Blu8EAS0JE6sqs-EkyWMdnui6AJ861P

We recommend using VLC player to play the videos.
