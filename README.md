# udacity_sensor_fusion_camera_project_one
udacity_sensor_fusion_camera_project_one_works

## MP1. Data Buffer Optimazation
In each time of new element come, I check the size of dataBuffer.
If the dataBuffer size is equal with dataBufferSize, then remove the first element in the dataBuffer.
After that, i will try to add the new element to the vector end.

## MP2. Keypoint Detection
I search the google and opencv offcial website. Then i implement all the necessary alogrithm in the the code matching2D_Student.cpp.

For Detection Alogrithm:
First of all, i compile the original version of code with Shi-Tomasi and BRISK.
Then i replace detector one by one, try to find the best for us.
The result is shown in below:
1. Shi-Tomasi detect 1300 keypoints in 17ms.
2. Harris detect 150 keypoints in 17ms. 
3. FAST detect 1300 keypoints in 0.7ms.
4. BRISK detect 2700 keypoints in 40ms.
5. ORB detect 500 keypoints in 7ms. ORB detect 2000 keypoints in 40ms.
6. AKAZE detect 1300 keypoints in 75ms.
7. SIFT detect 1400 keypoints in 125ms.

In my opinion, FAST/BRISK/ORB is good for us. Because they are fast, and they can get a lot of keypoints.

## MP3. Keypoint Removal
I loop all the keypoints, and check their postion one by one.
The keypoint which is not inside the pre-defined rectangle will be deleted.
    x range: rect.x + rect.x+width
    y range: rect.y + rect.y+height

As the result, i find there is a few keypoint on the ground or beyond the vehicle boundary.
And it seems FAST will have more keypoint out the the boundary of real car.
It seems ORB performance is better.

## MP4. Keypoint Descriptors
After searching the google and opencv offcial website, I implement all the necessary alogrithm in the the code matching2D_Student.cpp.
I test their time one by one with FAST (1300 keypoints)
1. BRISK use 2ms.
2. BRIEF use 0.7ms.
3. ORB use 1ms.
4. FREAK use 40ms.
5. AKAZE use 70ms.
6. SIFT use 80ms.

## MP5. Descriptor Matching

    if (descSource.type() != CV_32F)
    {
      descSource.convertTo(descSource, CV_32F);
      descRef.convertTo(descRef, CV_32F);
    }
    matcher = cv::FlannBasedMatcher::create();

## MP6. Descriptor Distance Ratio
The descriptor distance ratio is compute as follow:

    float threshold_ratio=0.8;
    int count_before=knn_match_list.size();

    for(int k=0 ; k< knn_match_list.size() ; k++)
    {
        float ratio = knn_match_list[k][0].distance / knn_match_list[k][1].distance;

        if(ratio < threshold_ratio){
            matches.push_back(knn_match_list[k][0]);
        }
    }

## For the following three tasks:
  MP7. Performance Evaluation : the number of keypoints in preceding vehicle for detectors
  MP.8 Performance Evaluation : the number of matched keypoints for detectors & descriptors
  MP.9 Performance Evaluation : the used time for detectors & descriptors

  I have make the test with all possible detector / descriptor combinations.
  And I record the result of the combinations in the table below:

  Note: for each combination, we record following performance:
  1. The number of keypoints found in all 10 images (inside the rectangle)
  2. The number of matched points in all 10 images (inside the rectangle)
  3. detector time for one image
  4. descriptor time for one image

  In the following table, each cell means one combination. the performance item will list one by one (from 1 to 4).
  
  
|c:detector&r:descriptor | SHITOMASI | HARRIS | FAST | BRISK | ORB | AKAZE | SIFT |
|-|-|-|-|-|-|-|-|
BRISK|1189 points<br/>771 matched<br/>17ms<br/>2ms<br/>| 359 points<br/>179 matched<br/>20ms<br/>1.5ms<br/> |1169 points<br/>736 matched<br/>1ms<br/>2ms<br/> |2714 points<br/>1545 matched<br/>40ms<br/>3ms<br/>|3213 points<br/>2013 matched<br/>10ms<br/>3.5ms<br/>|1655 points<br/>1204 matched<br/>80ms<br/>3ms<br/>|1371 points<br/>586 matched<br/>120ms<br/>1ms<br/>|
BRIEF|
ORB|
FREAK|
AKAZE|
SIFT|





