#include "bclassifier.h"


/**
 * @brief BClassifier::BClassifier
 */
BClassifier::BClassifier() {}


/**
 * @brief BClassifier::loadMeanCovarianceVectors
 * @param meanFilePath
 * @param covaFilePath
 */
void BClassifier::loadMeanCovarianceVectors(cv::String meanFilePath, cv::String covaFilePath) {
    cv::FileStorage fileM(meanFilePath, cv::FileStorage::READ);
    fileM["pixel_threshold"] >> _pixelThreshold;
    fileM["mean_matrix"] >> _meanMatrix; fileM.release();
    cv::FileStorage fileC(covaFilePath, cv::FileStorage::READ);
    fileC["num_samples"] >> _numSamples;
    fileC["covariance_matrix"] >> _covarianceMatrix; fileC.release();

    _meanMatrix.convertTo(_meanMatrix, CV_64F);
    _covarianceMatrix.convertTo(_covarianceMatrix, CV_64F);

    // Turns squared if it is a Diagonal Matrix:
    if (_covarianceMatrix.rows == 1) {
         cv::Mat covarianceMatrix = cv::Mat::zeros(_covarianceMatrix.cols, _covarianceMatrix.cols, CV_64F);
         for (int i=0; i < _covarianceMatrix.cols; i++) {
             covarianceMatrix.at<double>(i,i) = _covarianceMatrix.at<double>(0,i);
         }
         _covarianceMatrix = covarianceMatrix;
         isCovarianceDiag = true;
    }
}


/**
 * @brief BClassifier::~BClassifier
 */
BClassifier::~BClassifier() {
    _meanMatrix.~Mat();
    _covarianceMatrix.~Mat();
}
