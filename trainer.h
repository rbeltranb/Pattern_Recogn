#ifndef TRAINER_H
#define TRAINER_H

#include "featuresfilter.h"



/**
 * @brief The Trainer class
 */
class Trainer
{

public:
    const int PEDESTRIAN = 1;
    const int NON_PEDESTRIAN = 0;

    Trainer();
    ~Trainer();

    cv::Mat calcMeanVector(cv::String imagesPath);
    void constructWekaFile(cv::String imagesPath, int idClass, cv::String classDiff);
    void calcMeanCovarianceVectors(cv::String imagesPath, float pixelThreshold);
    void saveMeanCovarianceVectors(cv::String destPath, int idClass, cv::String classDiff);
    inline void loadWekaSelectionFile(cv::String wekaFilePath) {
        _pixelsSel = FeaturesFilter().loadWekaSelectionFile(wekaFilePath);
    }
    inline void loadGimpSelectionBMP(cv::String gimpBMPPath) {
        _pixelsSel = FeaturesFilter().loadGimpSelectionBMP(gimpBMPPath);
    }

protected:
    int _imgH, _imgW, _numFeatures, _numSamples;
    cv::Mat _meanMatrix;
    cv::Mat _covarianceMatrix;
    vector<int> _pixelsSel;
    float _pixelThreshold = 1.0f;

    vector<cv::String> calcImagesDimension(cv::String imagesPath);
    virtual cv::Mat imread(cv::String imagePath) = 0;
    virtual int getNumFeatures(cv::Mat image) = 0;
    virtual cv::Scalar getFeatureVal(cv::Mat image, int i) = 0;

};


#endif // TRAINER_H
