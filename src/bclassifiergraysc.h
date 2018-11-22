#ifndef BCLASSIFIERGRAYSC_H
#define BCLASSIFIERGRAYSC_H

#include "bclassifier.h"


/**
 * @brief The BClassifierGraySc class
 */
class BClassifierGraySc : public BClassifier
{

public:
    explicit BClassifierGraySc(bool isMultivariate);
    double calcProbability(cv::String imagePath);
    double probabilityNormalD(double x, double mean, double variance);

protected:
    virtual cv::Mat calcMeanDiff(cv::String imagePath);

private:
    bool _isMultivariate;

    double probabilityNormalD(cv::String imagePath);
    double probabilityMultivariateND(cv::String imagePath);
    double getDetDiagonal(cv::Mat matrixDiag);

};


#endif // BCLASSIFIERGRAYSC_H
