#ifndef TRAINERGREYSC_H
#define TRAINERGREYSC_H

#include "trainer.h"


/**
 * @brief The TrainerGreySc class
 */
class TrainerGreySc : public Trainer
{

protected:
    inline cv::Mat imread(cv::String imagePath) {
        return cv::imread(imagePath, CV_8UC1).reshape(0,1);
    }
    inline int getNumFeatures(cv::Mat image) {
        return image.cols*image.rows;
    }
    inline cv::Scalar getFeatureVal(cv::Mat image, int i) {
        return image.at<uchar>(0,i);
    }

};


#endif // TRAINERGREYSC_H
