#include "bclassifierhog.h"


/**
 * @brief BClassifierHOG::calcMeanDiff
 * @param imagePath
 * @return
 */
cv::Mat BClassifierHOG::calcMeanDiff(cv::String imagePath) {
    cv::Mat image = cv::imread(imagePath, 0);
    cv::HOGDescriptor hog = FeaturesFilter::getHog(image.rows, image.cols);
    vector<float> descriptors;
    hog.compute(image, descriptors);

    cv::Mat difference;
    cv::subtract(FeaturesFilter().applyFilter(cv::Mat(descriptors).t(), _pixelsSel, _pixelThreshold),
                 _meanMatrix, difference, cv::Mat(), _meanMatrix.type());
    return difference;
}
