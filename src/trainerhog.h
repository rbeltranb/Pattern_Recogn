#ifndef TRAINERHOG_H
#define TRAINERHOG_H

#include "trainer.h"


/**
 * @brief The TrainerHOG class
 */
class TrainerHOG : public Trainer
{

protected:
    cv::Mat imread(cv::String imagePath);
    int getNumFeatures(cv::Mat image);
    cv::Scalar getFeatureVal(cv::Mat image, int i);

private:
    cv::Mat getDescriptors(cv::Mat image, cv::HOGDescriptor hog);

};


#endif // TRAINERHOG_H
