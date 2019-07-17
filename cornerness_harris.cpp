#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

bool myfunction (const cv::KeyPoint& i, const cv::KeyPoint& j) { return (i.response > j.response); }

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)


    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);


    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    std::cout << "dst_norm_scaled size is "<<dst_norm.rows<<","<<dst_norm.cols << std::endl;
    //std::cout << dst_norm_scaled<<endl;
  
    
    vector<cv::KeyPoint> keypoints_result;
    vector<cv::KeyPoint> keypoints_pre;
  
  //int range =10 or 5 or 2 or 1, with no nms, looks good.
    int range=3;
  
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
            int min_value = 255;
            
            //skip the value too low, important. must have this limit. 
            //or the corner list found will be every large.
            if(cur_value< minResponse)
                continue;


            bool flag=true;
            //start to loop neighborhood
            for (int i = r-range ; i <= r + range ; i++)
            {
                for (int j= c-range ; j <= c + range ; j++)
                {
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
            
            if(flag == true && min_value!=cur_value)
            {

                //std::cout << "current r is "<<r<<", c is "<<c <<", value is "<< cur_value<< std::endl;
                //restore r,c to KeyPoints
                cv::KeyPoint point(c, r, range, -1, cur_value);
                keypoints_pre.push_back(point);
            }
            
        }
    }
  
    std::cout << "keypoints_pre size is "<<keypoints_pre.size() << std::endl;
    
    sort(keypoints_pre.begin(),keypoints_pre.end(),myfunction);
  
    int loop=0;
    while(keypoints_pre.size()!=0)
    {
        cv::KeyPoint pointA = keypoints_pre.back();
        keypoints_pre.pop_back();
        keypoints_result.push_back(pointA);
        
        ++loop;
        //std::cout << "keypoints_pre size is "<<keypoints_pre.size() <<" loop "<<loop<< std::endl;

        /*delete vector item method:
        auto iter = a.begin();
        while (iter != a.end()) {
            if (*iter > 30) {
                iter = a.erase(iter);
            }
            else {
                ++iter;
            }
        } */
        auto iter=keypoints_pre.begin();
        while(iter!=keypoints_pre.end())
        {
            cv::KeyPoint pointB= *iter;

            /* 
            if(abs(pointA.pt.x-pointB.pt.x)>range)
                continue;

            if(abs(pointA.pt.y-pointB.pt.y)>range)
                continue;	
                
            //get pointA's Box
            int ax1= pointA.pt.x-range;
            int ay1= pointA.pt.y-range;
            int ax2= pointA.pt.x+range;
            int ay2= pointA.pt.y+range;
                
            //get pointB's Box
            int bx1= pointB.pt.x-range;
            int by1= pointB.pt.y-range;
            int bx2= pointB.pt.x+range;
            int by2= pointB.pt.y+range;
                
            //Compute IOU
            int delta_x = min(bx2,ax2)-max(ax1,bx1);
            int delta_y = min(by2,ay2)-max(ay1,by1);

                
            float intersection= delta_x* delta_y;

            float IOU = intersection / ((2*range+1)*(2*range+1)-intersection);*/
            
            float IOU = cv::KeyPoint::overlap(pointA, pointB);

            //threshold should > 0.5
            float threshold = 0.2;
            if(IOU>threshold){

                std::cout << "nms found "<<IOU <<" "<<loop<< std::endl;

                //erase method will return next iterator. 
                //in this way, delete item is safe.
                iter=keypoints_pre.erase(iter);

            }
            else{

                ++iter;
            }
            
        }
    }
    std::cout << "keypoints_result size is "<<keypoints_result.size() << std::endl;


//    for(auto it = keypoints_result.begin(); it!= keypoints_result.end(); ++it)
//    {
//        cv::KeyPoint point = *it;
//        cv::circle (dst_norm_scaled, cv::Point(point.pt.x, point.pt.y), range, cv::Scalar(255,0,0),1,cv::LINE_AA);
//    }
   // visualize results
    cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypoints_result, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
   
    //for(auto it = keypoints_pre.begin(); it!= keypoints_pre.end(); ++it)
    //{
    //    cv::KeyPoint point = *it;
    //    cv::circle (dst_norm_scaled, cv::Point(point.pt.x, point.pt.y), 3, cv::Scalar(255,0,0),-1);
    //}

	 // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    //cv::imshow(windowName, dst_norm_scaled);
    cv::imshow(windowName,visImage);
    cv::waitKey(0);
}



int main()
{
    cornernessHarris();
}