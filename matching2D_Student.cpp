#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    
    // configure matcher
    bool crossCheck = false; //note: use knnMatch, may can not use crossCheck=true;


    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {

        //One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. 
        //L1 and L2 norms are preferable choices for SIFT and SURF descriptors
        //NORM_HAMMING should be used with ORB, BRISK and BRIEF
        //NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description). 
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        //... TODO : implement FLANN matching
        matcher =   cv::FlannBasedMatcher::create();
        cout << "FLANN matching";
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { 
        // k nearest neighbors (k=2)

        // TODO : implement k-nearest-neighbor matching
        double t = (double)cv::getTickCount();
        std::vector< std::vector< cv::DMatch >> knn_match_list; //the result list for knnMatch
        matcher->knnMatch(descSource, descRef, knn_match_list, 2);

        // TODO : filter matches using descriptor distance ratio test
        float threshold_ratio=0.8;
        int count_before=knn_match_list.size();
        
        for(int k=0 ; k< knn_match_list.size() ; k++)
        {
            float ratio = knn_match_list[k][0].distance / knn_match_list[k][1].distance;
            
            if(ratio < threshold_ratio){
                matches.push_back(knn_match_list[k][0]);
            }
        }

        int count_after=matches.size();

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
        cout << " Before NN ratio threshold , match count is "<<count_before<<", after the count is "<<count_after<<", delete percent is "<<1- (count_after+0.0)/count_before<<endl;
    }

}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// -> BRISK , BRIEF, ORB, FREAK, AKAZE, SIFT
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {
        if(descriptorType.compare("BRIEF")==0){
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        }
        else if(descriptorType.compare("ORB")==0){
            extractor = cv::ORB::create();
        }
        else if(descriptorType.compare("FREAK")==0){
            extractor = cv::xfeatures2d::FREAK::create();
        }
        else if(descriptorType.compare("AKAZE")==0){
            //AKAZE descriptor can be used only with AKAZE detector
            //or may get error. 
            //AKAZE descriptor do have some requirment on keypoints.
            extractor = cv::AKAZE::create();
        }
        else if(descriptorType.compare("SIFT")==0){
            extractor = cv::xfeatures2d::SIFT::create();
        }

    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

//// -> FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    double t;
    if(detectorType.compare("FAST")==0)
    {
        cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(40,true);

        t = (double)cv::getTickCount();
        fast->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        cout << "Fast with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if(detectorType.compare("BRISK")==0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();

        t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        cout << "BRISK with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if(detectorType.compare("ORB")==0)
    {
        cv::Ptr<cv::ORB> detector = cv::ORB::create(2000);

        t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        cout << "ORB with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if(detectorType.compare("AKAZE")==0)
    {
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
        
        t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        cout << "AKAZE with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if(detectorType.compare("SIFT")==0)
    {
        cv::Ptr<cv::xfeatures2d::SIFT> detectorSIFT= cv::xfeatures2d::SIFT::create();
        t = (double)cv::getTickCount();
        detectorSIFT->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

        cout << "SIFT detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }


    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType.append(" Detector Results");
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


//// -> HARRIS
bool myfunction (const cv::KeyPoint& i, const cv::KeyPoint& j) { return (i.response > j.response); }
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    double t = (double)cv::getTickCount();

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    //cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    std::vector<cv::KeyPoint> keypoints_pre;

    //find local max in dst_norm as keypoint_pre
    
    int range=blockSize;
    //you must use dst_norm , not dst_norm_scaled. 
    //the dst_norm_scaled value maybe very large. Not inside [0,255]
    for(int r=0; r< dst_norm.rows; r++)
    {
        //check index boundary
        if(r-range<0 || r+range >dst_norm.rows-1)
            continue;
        
        for(int c=0; c<dst_norm.cols; c++)
        {
            //check col index boundary
            if(c-range<0 || c+range > dst_norm.cols-1)
            continue;
            
            // compute if dst_norm_scaled [r, c] is the local max 
            // check range: rectangle, center point (r,c)
            // min : r - range , c - range
            // max : r + range , c + range
            int cur_value = (int)(dst_norm.at<float>(r,c));
            
            //skip the value too low, important. must have this limit. 
            //or the corner list found will be every large.
            if(cur_value< minResponse)
                continue;


            bool flag=true;
            int min_value=255;
            //start to loop neighborhood
            for (int i = r-range ; i <= r + range ; i++)
            {
                for (int j= c-range ; j <= c + range ; j++)
                {
                    //skip point does not in the circle;
                    float distance= sqrt((i-r)*(i-r)+(j-c)*(j-c));
                    if(distance > range)
                        continue;

                    int search_value= (int)(dst_norm.at<float>(i,j));
                    
                    if(min_value>search_value)
                    {
                        min_value=search_value;
                    }


                    if(cur_value < search_value)
                    {
                        flag = false;
                        break;
                    }
                }
            }
            
            //cur_value is local max (and not all value equal with each other)
            if(flag == true && min_value!=cur_value)
            {

                //std::cout << "current r is "<<r<<", c is "<<c <<", value is "<< cur_value<< std::endl;
                //restore r,c to KeyPoints
                cv::KeyPoint point(c, r, range, -1, cur_value);
                keypoints_pre.push_back(point);
            }
            
        }
    }

    //use nms to minimize the overlap corner point
    float threshold_overlap=0.4;
    sort(keypoints_pre.begin(),keypoints_pre.end(),myfunction);

    //int loop=0;
    while(keypoints_pre.size()!=0)
    {
        cv::KeyPoint pointA = keypoints_pre.back();
        keypoints_pre.pop_back();
        keypoints.push_back(pointA);
        
        //++loop;
        
        auto iter=keypoints_pre.begin();
        while(iter!=keypoints_pre.end())
        {
            cv::KeyPoint pointB= *iter;

            float IOU = cv::KeyPoint::overlap(pointA, pointB);

            if(IOU>threshold_overlap){

                //std::cout << "nms found "<<IOU <<" "<<loop<< std::endl;
                //erase method will return next iterator. 
                //in this way, delete item is safe.
                iter=keypoints_pre.erase(iter);

            }
            else{

                ++iter;
            }
            
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}