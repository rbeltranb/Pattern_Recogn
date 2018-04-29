#include "trainerhog.h"


/**
 * @brief TrainerHOG::imread
 * @param imagePath
 * @return
 */
cv::Mat TrainerHOG::imread(cv::String imagePath) {
    return getDescriptors(cv::imread(imagePath, CV_8UC1), FeaturesFilter::getHog(_imgH, _imgW));
}


/**
 * @brief TrainerHOG::getNumFeatures
 * @param image
 * @return
 */
int TrainerHOG::getNumFeatures(cv::Mat image) {
    cv::Mat descriptors = getDescriptors(image, FeaturesFilter::getHog(_imgH, _imgW));
    return descriptors.cols;
}


/**
 * @brief TrainerHOG::getDescriptors
 * @param image
 * @param hog
 * @return
 */
cv::Mat TrainerHOG::getDescriptors(cv::Mat image, cv::HOGDescriptor hog) {
    vector<float> descriptors;
    hog.compute(image, descriptors);

    return cv::Mat(descriptors).t();
}


/**
 * @brief TrainerHOG::getFeatureVal
 * @param image
 * @param i
 * @return
 */
cv::Scalar TrainerHOG::getFeatureVal(cv::Mat image, int i) {
    return image.at<float>(0,i);
}
