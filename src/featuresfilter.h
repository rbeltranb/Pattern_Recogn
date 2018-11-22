#ifndef FEATURESFILTER_H
#define FEATURESFILTER_H

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;



/**
 * @brief The FeaturesFilter class
 */
class FeaturesFilter
{

public:
    static vector<int> loadWekaSelectionFile(cv::String wekaFilePath);
    static vector<int> loadGimpSelectionBMP(cv::String gimpBMPPath);
    static cv::Mat applyFilter(cv::Mat featuresVec, vector<int> _pixelsSel, float pixelThreshold);
    static void overlayPixelsSelection(cv::String wekaFilePath, vector<int> pixelsSel, float pixelThreshold);

    static cv::HOGDescriptor getHog(int imgH, int imgW);

};


#endif // FEATURESFILTER_H
