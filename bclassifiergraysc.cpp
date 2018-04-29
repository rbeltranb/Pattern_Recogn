#include "bclassifiergraysc.h"


/**
 * @brief BClassifierGraySc::BClassifierGraySc
 * @param isMultivariate - Uses Variance or Covariance.
 */
BClassifierGraySc::BClassifierGraySc(bool isMultivariate) {
    _isMultivariate = isMultivariate;
}


/**
 * @brief BClassifierGraySc::calcProbability
 * @param imagePath
 * @return
 */
double BClassifierGraySc::calcProbability(cv::String imagePath) {

    if (_isMultivariate) {
        return probabilityMultivariateND(imagePath);
    } else {
        return probabilityNormalD(imagePath);
    }
}


/**
 * @brief BClassifierGraySc::calcMeanDiff
 * @param imagePath
 * @return
 */
cv::Mat BClassifierGraySc::calcMeanDiff(cv::String imagePath) {
    cv::Mat image = cv::imread(imagePath, 0);
    image.convertTo(image, _meanMatrix.type());
    cv::Mat diffImageMean = (FeaturesFilter().applyFilter(
                                 image.reshape(0, 1), _pixelsSel, _pixelThreshold) - _meanMatrix) / 255; // Normalized.

    return diffImageMean;
}


/**
 * @brief BClassifierGraySc::probabilityMultivariateND
 * @param imagePath
 * @return - The estimator or exponent to the Gaussian function.
 */
double BClassifierGraySc::probabilityMultivariateND(cv::String imagePath) {
    cv::Mat diffImageMean = calcMeanDiff(imagePath);

    cv::Mat exponentM = -(diffImageMean * ((isCovarianceDiag)? 1/_covarianceMatrix :
                                                                   _covarianceMatrix.inv()) * diffImageMean.t()) / 2;
    /*double gaussianPDF = exp(exponentM.at<double>(0,0)) / sqrt(pow(2*CV_PI, diffImageMean.total())
                                                               * ((isCovarianceDiag)? getDetDiagonal(_covarianceMatrix)
                                                                                    : cv::determinant(_covarianceMatrix)));*/

    // Returns the exponent as an estimator; gaussianPDF is mostly zero due to big number in the quotient.
    return -exponentM.at<double>(0,0);//gaussianPDF;
}


/**
 * @brief BClassifierGraySc::probabilityNormalD
 * @param imagePath
 * @return
 */
double BClassifierGraySc::probabilityNormalD(cv::String imagePath) {
    double probability = 1;

    cv::Mat diffImageMean = calcMeanDiff(imagePath);
    cv::Mat variance = _covarianceMatrix.diag().t();

    // The multiplicatory of probabilities is almost zero.
    for (int i = 0; i < diffImageMean.cols; i++) {
        probability *= probabilityNormalD(diffImageMean.at<double>(i), 0, variance.at<double>(i));
    }

    return probability;
}


/**
 * @brief BClassifierGraySc::probabilityNormalD
 * @param x
 * @param mean
 * @param variance
 * @return - In testing HOG shows the exponential factor too close to zero.
 */
double BClassifierGraySc::probabilityNormalD(double x, double mean, double variance) {
    return exp(-(x-mean)*(x-mean) / (2*variance)) / sqrt(2*CV_PI * variance);
}


/**
 * @brief BClassifierGraySc::getDetDiagonal
 * @param matrix
 * @return
 */
double BClassifierGraySc::getDetDiagonal(cv::Mat matrixDiag) {
    double determinant = 1;

    for (int i = 0; i < matrixDiag.rows; i++) {
        determinant *= matrixDiag.at<double>(i,i);
    }

    return determinant;
}
