#include "trainer.h"


/**
 * @brief Trainer::Trainer
 */
Trainer::Trainer() {}


/**
 * @brief Trainer::calcMeanVector
 * @param imagesPath
 * @return
 */
cv::Mat Trainer::calcMeanVector(cv::String imagesPath) {
    vector<cv::String> imagesPathVector = calcImagesDimension(imagesPath);
    cv::Mat meanMatrix = cv::Mat::zeros(1, _numFeatures, CV_32FC1);

    for (int i = 0; i < _numSamples; i++) {
        cv::accumulate(imread(imagesPathVector[i]), meanMatrix);
    }

    return meanMatrix;
}


/**
 * @brief Trainer::calcMeanCovarianceVectors
 * @param imagesPath
 * @param pixelThreshold
 */
void Trainer::calcMeanCovarianceVectors(cv::String imagesPath, float pixelThreshold) {
    _pixelThreshold = pixelThreshold;
    vector<cv::Mat> imagesVector;
    vector<cv::String> imagesPathVector = calcImagesDimension(imagesPath);

    for (int i = 0; i < _numSamples; i++) {
        imagesVector.push_back(FeaturesFilter().applyFilter(
                                   imread(imagesPathVector[i]), _pixelsSel, _pixelThreshold));
    }

    cv::calcCovarMatrix(imagesVector.data(), _numSamples, _covarianceMatrix, _meanMatrix,
                        CV_COVAR_NORMAL | CV_COVAR_SCALE);
    _covarianceMatrix = _covarianceMatrix/(_numSamples-1);
}


/**
 * @brief Trainer::saveMeanCovarianceVectors
 * @param destPath
 * @param idClass
 * @param classDiff
 */
void Trainer::saveMeanCovarianceVectors(cv::String destPath, int idClass, cv::String classDiff) {
    classDiff = "_"+to_string(_imgW)+"x"+to_string(_imgH)+"_c"+to_string(idClass)+"_"+classDiff;

    // Saving data as BMP:
    if (_pixelsSel.size() == 0) {
        classDiff += "_all";
        cv::imwrite(destPath+"/meanMatrix"+classDiff+".bmp", (_meanMatrix.reshape(0, _imgH)));
    } else try {
        FeaturesFilter().overlayPixelsSelection(destPath+"/meanMatrix"+classDiff+"_all.bmp",
                                                _pixelsSel, _pixelThreshold);
        classDiff += "_weka"; // @TODO: Should be Gimp selection file!
    } catch (exception& ex) {
        cout << "\nThe mean image with the overlaid pixels couldn't be created.";
        cout << "The file " << "meanMatrix"<<classDiff<<"_all.bmp" << " must be in " << destPath;
    }
    cv::imwrite(destPath+"/covarianceMatrix"+classDiff+".bmp", _covarianceMatrix*255); // Scaling to view as image.
    cv::Mat covarianceDiag = _covarianceMatrix.diag().t();
    if (_pixelsSel.size() == 0)
    cv::imwrite(destPath+"/covarianceDiag"+classDiff+".bmp", (covarianceDiag.reshape(0, _imgH))*255);

    // Saving data as XML:
    cv::FileStorage fileM(destPath+"/meanMatrix"+classDiff+".xml", cv::FileStorage::WRITE);
    fileM << "pixel_threshold" << _pixelThreshold;
    fileM << "mean_matrix" << _meanMatrix; fileM.release();
    cv::FileStorage fileC(destPath+"/covarianceMatrix"+classDiff+".xml", cv::FileStorage::WRITE);
    fileC << "num_samples" << _numSamples;
    fileC << "covariance_matrix" << _covarianceMatrix; fileC.release();
    cv::FileStorage fileD(destPath+"/covarianceDiag"+classDiff+".xml", cv::FileStorage::WRITE);
    fileD << "num_samples" << _numSamples;
    fileD << "covariance_matrix" << covarianceDiag; fileD.release();
}


/**
 * @brief Trainer::calcImagesDimension
 * @param imagesPath
 * @return
 */
vector<cv::String> Trainer::calcImagesDimension(cv::String imagesPath) {
    // Vector with the individual path to each image.
    vector<cv::String> imagesPathVector;
    cv::glob(imagesPath, imagesPathVector);
    _numSamples = imagesPathVector.size();

    if (_numFeatures == 0 || _imgH == 0 || _imgW == 0) {
        // First image is used only to calculate dimensions.
        cv::Mat image = cv::imread(imagesPathVector[0]);
        // It is assumed that in the training set all images have the same dimension.
        _imgH = image.rows; _imgW = image.cols;

        _numFeatures = getNumFeatures(image);
    }

    return imagesPathVector;
}


/**
 * @brief Trainer::constructWekaFile
 * @param imagesPath
 * @param idClass
 * @param classDiff
 */
void Trainer::constructWekaFile(cv::String imagesPath, int idClass, cv::String classDiff) {
    cv::Mat image;
    cv::Scalar valPixel;
    vector<cv::String> imagesPathVector = calcImagesDimension(imagesPath);
    classDiff = "_"+to_string(_imgW)+"x"+to_string(_imgH)+"_c"+to_string(idClass)+"_"+classDiff;

    ofstream wekaFile(imagesPath+"/weka_vectors"+classDiff+".arff");
    wekaFile << "@RELATION Pedestrians" << endl;

    for (int i = 0; i < _numFeatures; i++) {
        wekaFile<< "@ATTRIBUTE Pixel" << i << " NUMERIC" << endl;
    }

    wekaFile << "@ATTRIBUTE Klasse {0,1}\n% 0=NonPedestrian, 1=Pedestrian\n@DATA" << endl;

    int k;
    try {
        for (k = 0; k < _numSamples; k++) {
            image = imread(imagesPathVector[k]);

            for (int i = 0; i < _numFeatures; i++) {
                valPixel = getFeatureVal(image, i);
                wekaFile << valPixel.val[0] << ",";
            }

            wekaFile << to_string(idClass) << endl;
        }
    } catch(cv::Exception ex) {
        cout << "Fail to load image number " << k << ".\n" << ex.err;
    }

    wekaFile.close();
}


/**
 * @brief Trainer::~Trainer
 */
Trainer::~Trainer() {
    _meanMatrix.~Mat();
    _covarianceMatrix.~Mat();
}
