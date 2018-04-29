#ifndef BCLASSIFIERHOG_H
#define BCLASSIFIERHOG_H

#include "bclassifiergraysc.h"


/**
 * @brief The BClassifierHOG class
 */
class BClassifierHOG : public BClassifierGraySc
{

public:
    BClassifierHOG(bool isMultivariate) : BClassifierGraySc(isMultivariate) {};

protected:
    cv::Mat calcMeanDiff(cv::String imagePath);

};

#endif // BCLASSIFIERHOG_H
