# AVS1 

This branch contains implementation of our user interface for the paper "Semantic Segmentation of Golf Courses for Course Rating Assistance"
## Folder Structure

"course_rating" handles calculation for different course metrics such as green size, hole length etc.

"user_interface" is a user interface for calculating metrics for a given course. The user interface uses files from "course_rating" to calculate the metrics.

Prediction images must be created for the user interface to open. This is done using the "cuda_predict_image" file using a cuda capable GPU.

The user interface can be opened by running the main file in the folder.
