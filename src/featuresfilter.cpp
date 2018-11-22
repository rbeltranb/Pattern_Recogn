#include "featuresfilter.h"


/**
 * @brief FeaturesFilter::loadWekaSelectionFile
 * @param wekaFilePath
 * @return
 */
vector<int> FeaturesFilter::loadWekaSelectionFile(cv::String wekaFilePath) {
    vector<int> pixelsSel;
    ifstream wekaFile(wekaFilePath);
    string word;

    while(wekaFile >> word) {
        if(word == "Selected") {
            wekaFile >> word; // "attributes:"
            wekaFile >> word;

            stringstream ss(word);
            string item;
            while (getline(ss, item, ',')) {
                pixelsSel.push_back(atoi(item.c_str())-1);
            }

            break;
        }
    }

    wekaFile.close();
    return pixelsSel;
}


/**
 * @brief FeaturesFilter::loadGimpSelectionBMP
 * @param gimpBMPPath
 * @return
 */
vector<int> FeaturesFilter::loadGimpSelectionBMP(cv::String gimpBMPPath) {
    vector<int> pixelsSel;
    cv::Mat thresholdImage = cv::imread(gimpBMPPath, 0);
    cv::Scalar intensity;
    vector<int> pixelsRemain;

    for (int i = 0; i < thresholdImage.rows; i++) {
        for (int j = 0; j < thresholdImage.cols; j++) {
            intensity = thresholdImage.at<uchar>(i,j);
            if (intensity[0] < 255) { // White threshold.
                pixelsSel.push_back(i*thresholdImage.cols+j);
            } else {
                pixelsRemain.push_back(i*thresholdImage.cols+j);
            }
        }
    }

    reverse(pixelsRemain.begin(), pixelsRemain.end()); // Ending pixels have less divergence.
    pixelsSel.insert(pixelsSel.end(), pixelsRemain.begin(), pixelsRemain.end());
    return pixelsSel;
}


/**
 * @brief FeaturesFilter::applyFilter
 * @param featuresVec - Has only one row.
 * @param pixelsSel
 * @param pixelThreshold
 * @return
 */
cv::Mat FeaturesFilter::applyFilter(cv::Mat featuresVec, vector<int> pixelsSel, float pixelThreshold) {
    uint n = pixelsSel.size()*pixelThreshold;
    cv::Mat newFeaturesVec = featuresVec;

    if (n != pixelsSel.size()) {
        newFeaturesVec = featuresVec.col(pixelsSel[0]);

        for (uint i = 1; i < n; i++) {
            cv::hconcat(newFeaturesVec, featuresVec.col(pixelsSel[i]), newFeaturesVec);
        }
    }

    return newFeaturesVec;
}


/**
 * @brief FeaturesFilter::overlayPixelsSelection - Draw over the Mean BMP file the selected pixels set.
 * @param imageMeanPath
 * @param pixelsSel
 * @param pixelThreshold
 */
void FeaturesFilter::overlayPixelsSelection(cv::String imageMeanPath, vector<int> pixelsSel, float pixelThreshold) {
    cv::Mat meanImage = cv::imread(imageMeanPath);
    cv::String imageOverlaidPath = imageMeanPath.substr(0, imageMeanPath.rfind('.'))
            + "_thr"+to_string((int)(pixelThreshold*100))+".bmp";

    for (int i = 0; i < pixelsSel.size()*pixelThreshold; i++) {
        int posX = pixelsSel[i]/meanImage.cols;
        int posY = pixelsSel[i]%meanImage.cols;
        cv::circle(meanImage, cv::Point(posX, posY), 1, cv::Scalar(0, 255, 255));
    }

    cv::imwrite(imageOverlaidPath, meanImage);
}



/**
 * @brief FeaturesFilter::getHog
 * @param imgH
 * @param imgW
 * @return
 */
cv::HOGDescriptor FeaturesFilter::getHog(int imgH, int imgW) {

    if (imgH == 36 and imgW == 18) {
        return cv::HOGDescriptor(cv::Size(18,36), cv::Size(6,6), cv::Size(6,6), cv::Size(3,3), 9);
    } else if (imgH == 96 and imgW == 48) {
        return cv::HOGDescriptor(cv::Size(48,96), cv::Size(8,8), cv::Size(8,8), cv::Size(4,4), 9);
    }

    throw cv::Exception(); // The image size is not soported.
}
