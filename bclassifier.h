#ifndef BCLASSIFIER_H
#define BCLASSIFIER_H

#include "featuresfilter.h"



/**
 * @brief The BClassifier class
 */
class BClassifier
{

public:
    int _numSamples;

    BClassifier();
    ~BClassifier();

    virtual double calcProbability(cv::String imagePath) = 0;
    inline bool isMultivariate() {return !isCovarianceDiag;}

    inline void loadWekaSelectionFile(cv::String wekaFilePath) {
        _pixelsSel = FeaturesFilter().loadWekaSelectionFile(wekaFilePath);
    }
    inline void loadGimpSelectionBMP(cv::String gimpBMPPath) {
        _pixelsSel = FeaturesFilter().loadGimpSelectionBMP(gimpBMPPath);
    }
    void loadMeanCovarianceVectors(cv::String meanFilePath, cv::String covaFilePath);

protected:
    cv::Mat _meanMatrix;
    cv::Mat _covarianceMatrix;
    /** @brief isCovarianceDiag - Used to reduce calculations. */
    bool isCovarianceDiag = false;
    vector<int> _pixelsSel;
    float _pixelThreshold = 1.0f;

};


#endif // BCLASSIFIER_H
