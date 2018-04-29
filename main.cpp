/**
 * Image Processing  and   Pattern Recognition
 * Professorship of Communications Engineering
 * TU-Chemnitz                     ICS-WS17/18
 *
 * Implementation of a Bayesian classifier with Gaussian distribution hypothesis based on the detector approach.
 * As feature vector characteristic is used the greyscale values of the image, as well as the HOG descriptor of
 * such images.
 *
 * Developed by:
 * Raul Beltrán B.	Cod. 447953
 *
 **/
#include "trainergreysc.hpp"
#include "trainerhog.h"
#include "bclassifierhog.h"


void estimatorClassifiersComparison();
void calcConfusionMatrix(vector<cv::String> samplesPaths, BClassifier* classifier,
                         float discThreshold, int numSamples, bool isGrayScale, bool isConfMatrix);

/**
 * @brief main
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[])
{
    if (argc > 1) {

        if (strncmp(argv[1], "-weka", 5)==0 || strncmp(argv[1], "-training", 9)==0) {
            Trainer* trainer;
            if (strncmp(argv[argc-1], "GreySc", 3)==0)
                trainer = new TrainerGreySc();
            else trainer = new TrainerHOG();

            if (strncmp(argv[1], "-weka", 5)==0) {
                trainer->constructWekaFile(
                            argv[2], (strncmp(argv[3], "1", 1)==0? trainer->PEDESTRIAN : trainer->NON_PEDESTRIAN), argv[4]);
                cout << "Please find and move the result .arff file from " << argv[2]<<"\n";
            }
            else { //-training

                int minusPars = 0;
                if (argc == 8) {
                    if (strncmp(argv[2], "Weka", 4)==0) {
                        trainer->loadWekaSelectionFile(argv[3]);
                    } else { //-Gimp
                        trainer->loadGimpSelectionBMP(argv[3]);
                    }

                    trainer->calcMeanCovarianceVectors(argv[5], stof(argv[4]));
                } else {
                    trainer->calcMeanCovarianceVectors(argv[2], 1.0f);
                    minusPars = 3;
                }

                cv::String localPath = cv::String(argv[0]).substr(0, cv::String(argv[0]).rfind('/'));
                trainer->saveMeanCovarianceVectors(
                            localPath, (strncmp(argv[6-minusPars], "1", 1)==0? trainer->PEDESTRIAN : trainer->NON_PEDESTRIAN),
                            argv[7-minusPars]);
            }
            return 0;
        }

        if (strncmp(argv[1], "-classify", 9)==0 || strncmp(argv[1], "-discrimins", 11)==0) {
            BClassifier* classifier;
            if (strncmp(argv[argc-2], "GreySc", 3)==0)
                classifier = new BClassifierGraySc(!strncmp(argv[argc-1], "Multi", 4));
            else classifier = new BClassifierHOG(!strncmp(argv[argc-1], "Multi", 4));

            classifier->loadMeanCovarianceVectors(argv[4], argv[5]);

            int minusPars = 0;
            if ((strncmp(argv[1], "-classify", 9)==0 && argc == 12)
                    || ((strncmp(argv[1], "-discrimins", 11)==0) && argc == 10)) {

                if (strncmp(argv[6], "Weka", 4)==0) {
                    classifier->loadWekaSelectionFile(argv[7]);
                } else { //-Gimp
                    classifier->loadGimpSelectionBMP(argv[7]);
                }
            } else minusPars =  2;

            vector<cv::String> samplesPaths = {argv[2], argv[3]};
            if (strncmp(argv[1], "-classify", 9)==0) {
                calcConfusionMatrix(samplesPaths, classifier, stof(argv[8-minusPars]), stoi(argv[9-minusPars]),
                        !strncmp(argv[argc-2], "GreySc", 3), true /*isConfMatrix*/);
            } else {
                calcConfusionMatrix(samplesPaths, classifier, 1.0f, 1, !strncmp(argv[argc-2], "GreySc", 3), false);
            }

            return 0;
        }
    }

    cout<<"\n-weka\n\tConstructs an .arff file corresponding to the Class to be analyzed in Weka:";
    cout<<"\n\t[path_training_images_directory, class_id (1:Pedestrians|0:NonPedestrians), alias ('GraySc'|'HOG')]";

    cout<<"\n\n-training\n\tLoads the Weka or Gimp selection file and calcs the Mean and Covariance matrices, to save finally these in BMP an XML formats:";
    cout<<"\n\t[filter_type ('Weka'|'Gimp'), path_selection_features_file, features_threshold, path_training_images_directory, class_id (1:Pedestrians|0:NonPedestrians), alias ('GraySc'|'HOG')]";

    cout<<"\n\n\tCalcs the Mean and Covariance matrices with all the features and saves these in BMP an XML formats:";
    cout<<"\n\t[path_training_imagesdirectory, class_id (1:Pedestrians|0:NonPedestrians), alias ('GraySc'|'HOG')]";

    cout<<"\n\n-classify\n\tCreates the Transition Matrix testing the Classifier with sets of given group size images:";
    cout<<"\n\t[path_positive_images_directory, path_negative_images_directory, path_mean_file, path_co-variance_file, filter_type ('Weka'|'Gimp'), path_selection_features_file, discriminator_threshold, size_group_samples, alias ('GraySc'|'HOG'), gaussian_type ('Multi'|'Mono')]";

    cout<<"\n\n\tCreates the Transition Matrix testing the Classifier with sets of given group size images whitout using a features filter:";
    cout<<"\n\t[path_positive_images_directory, path_negative_images_directory, path_mean_file, path_co-variance_file, discriminator_threshold, size_group_samples, alias ('GraySc'|'HOG'), gaussian_type ('Multi'|'Mono')]";

    cout<<"\n\n-discrimins\n\tCalcs all the Discriminators from to the images given in the samples directory:";
    cout<<"\n\t[path_positive_images_directory, path_negative_images_directory, path_mean_file, path_co-variance_file, filter_type ('Weka'|'Gimp'), path_selection_features_file, alias ('GraySc'|'HOG'), gaussian_type ('Multi'|'Mono')]";

    cout<<"\n\n\tCalcs all the Discriminators from to the images given in the samples directory without using a features filter:";
    cout<<"\n\t[path_positive_images_directory, path_negative_images_directory, path_mean_file, path_co-variance_file, alias ('GraySc'|'HOG'), gaussian_type ('Multi'|'Mono')]\n\n";


//COMMAND LINE EXAMPLES:
//
//-weka /Volumes/Intenso_Mac/DaimlerBenchmark/TrainingData/Pedestrians/18x36 1 GreySc
//-weka /Volumes/Intenso_Mac/DaimlerBenchmark/TrainingData/NonPedestrians/18x36 0 GreySc

//-trainning Weka /Volumes/Intenso_Mac/DaimlerBenchmark/ClassifyData/weka_correlationAttributeEval_18x36_GreySc.txt 0.4
//           /Volumes/Intenso_Mac/DaimlerBenchmark/TrainingData/Pedestrians/18x36 1 GreySc

//-classify /Users/Rb2/Downloads/Image_Processing/Pedestrians/18x36 /Users/Rb2/Downloads/Image_Processing/NonPedestrians/18x36
//          /Volumes/Intenso_Mac/DaimlerBenchmark/TrainingData/Pedestrians/18x36-GreySc_weka/meanMatrix_18x36_c1_GreySc_weka.xml
//          /Volumes/Intenso_Mac/DaimlerBenchmark/TrainingData/Pedestrians/18x36-GreySc_weka/covarianceMatrix_18x36_c1_GreySc_weka.xml
//          Weka /Volumes/Intenso_Mac/DaimlerBenchmark/ClassifyData/weka_correlationAttributeEval_18x36_GreySc.txt 0.4 50 GreySc Multi

//--ALL
//-discrimins /Users/Rb2/Downloads/Image_Processing/Pedestrians/18x36 /Users/Rb2/Downloads/Image_Processing/NonPedestrians/18x36
//            /Volumes/Intenso_Mac/DaimlerBenchmark/TrainingData/Pedestrians/18x36-GreySc_all/meanMatrix_18x36_c1_GreySc_all.xml
//            /Volumes/Intenso_Mac/DaimlerBenchmark/TrainingData/Pedestrians/18x36-GreySc_all/covarianceMatrix_18x36_c1_GreySc_all.xml
//            GreySc Multi

}


/**
 * @brief estimatorClassifiersComparison - Produces results to compare both Clasiffiers, whith two different filters
 * and two variations in the Gaussian function.
 */
void estimatorClassifiersComparison() {
    cv::String pathClassify       = "/Users/Rb2/Downloads/Image_Processing";
    cv::String pathTrainingPed    = "/Volumes/Intenso_Mac/DaimlerBenchmark/TrainingData/Pedestrians";
    cv::String pathTrainingNonPed = "/Volumes/Intenso_Mac/DaimlerBenchmark/TrainingData/NonPedestrians";

    BClassifierGraySc classifier_C1(true), classifier_C0(true);

    vector<BClassifier*> classifiers = {&classifier_C1, &classifier_C0};
    vector<cv::String> samplesPaths = {pathClassify+"/Pedestrians/18x36",
                                       pathClassify+"/NonPedestrians/18x36"};

    int totalTrainingSamples = 0;
    for (unsigned int j = 0; j < classifiers.size(); j++) {
        totalTrainingSamples += classifiers[j]->_numSamples;
    }

    for (int m = 0; m < 2; m++) { // Two tests: Covariance and Variance.
        classifier_C1.loadMeanCovarianceVectors(pathTrainingPed+"/18x36-GreySc_all/meanMatrix_18x36_c1_GreySc_all.xml",
                                                pathTrainingPed+"/18x36-GreySc_all/covarianceMatrix_18x36_c1_GreySc_all.xml");
        classifier_C0.loadMeanCovarianceVectors(pathTrainingNonPed+"/18x36-GreySc_all/meanMatrix_18x36_c0_GreySc_all.xml",
                                                pathTrainingNonPed+"/18x36-GreySc_all/covarianceMatrix_18x36_c0_GreySc_all.xml");

        for (int l = 0; l < 2; l++) { // Two selections: Weka and Gimp.
            cout<<"\n"<<((l==0)? "Weka" : "Gimp");
            cout<<((m==0)? "-Squared(Covariance):" : "-Diagonal(Variance):")<<"\n";

            classifier_C1.loadWekaSelectionFile(pathClassify+"/weka_correlationAttributeEval_18x36_GreySc.txt");
            classifier_C0.loadWekaSelectionFile(pathClassify+"/weka_correlationAttributeEval_18x36_GreySc.txt");

            for (unsigned int i = 0; i < samplesPaths.size(); i++) {
                // Vector with the individual path to each image.
                vector<cv::String> imagesPathVector;
                cv::glob(samplesPaths[i], imagesPathVector);
                string prob_c0, prob_c1;

                for (unsigned int k = 0; k < 100/*imagesPathVector.size()*/; k++) {

                    for (unsigned int j = 0; j < classifiers.size(); j++) {
                        double probability = classifiers[j]->calcProbability(imagesPathVector[k]);
                        // The a priori probability takes in account the porportion of samples per Class.
                        //probability *= (double)classifiers[j]->_numSamples / totalTrainingSamples;

                        if(j==0) { prob_c0+="\t"+to_string(probability); }
                        else { prob_c1+="\t"+to_string(probability); }
                    }
                }

                cout<<"\nSamples "<<((i==0)? "Pos" : "Neg")<<"\nClassPed\t"<<prob_c0;
                cout<<"\nSamples "<<((i==0)? "Pos" : "Neg")<<"\nClassNonP\t"<<prob_c1;
            }

            classifier_C1.loadGimpSelectionBMP(pathClassify+"/gimp_meanMatrix_18x36_GreySc.bmp");
            classifier_C0.loadGimpSelectionBMP(pathClassify+"/gimp_meanMatrix_18x36_GreySc.bmp");
        }

        classifier_C1 = BClassifierGraySc(false); classifier_C0 = BClassifierGraySc(false);
        classifier_C1.loadMeanCovarianceVectors(pathTrainingPed+"/18x36-GreySc_all/meanMatrix_18x36_c1_GreySc_all.xml",
                                                pathTrainingPed+"/18x36-GreySc_all/covarianceDiag_18x36_c1_GreySc_all.xml");
        classifier_C0.loadMeanCovarianceVectors(pathTrainingNonPed+"/18x36-GreySc_all/meanMatrix_18x36_c0_GreySc_all.xml",
                                                pathTrainingNonPed+"/18x36-GreySc_all/covarianceDiag_18x36_c0_GreySc_all.xml");
    }
}


/**
 * @brief calcConfusionMatrix
 * @param samplesPaths - Data order: [0] Positives, [1] Negatives
 * @param classifier
 * @param discThreshold
 * @param numSamples
 * @param isConfMatrix
 * @brief confusionMatrix
 *              ---------------------------
 *     i / j    | ClassPed  | ClassNonPed |
 * ----------------------------------------
 * | SamplesPos | i[0],j[0] | i[0],j[1]   |
 * ----------------------------------------
 * | SamplesNeg | i[1],j[0] | i[1],j[1]   |
 * ----------------------------------------
 */
void calcConfusionMatrix(vector<cv::String> samplesPaths, BClassifier* classifier,
                         float discThreshold, int numSamples, bool isGrayScale, bool isConfMatrix) {

    for (unsigned int i = 0; i < samplesPaths.size(); i++) {
        // Vector with the individual path to each image.
        vector<cv::String> imagesPathVector;
        cv::glob(samplesPaths[i], imagesPathVector);

        if (isConfMatrix)
            cout << "\n                           " << (i==0?"TP  FN\n":"FP  TN\n");

        // Divides the total samples set into groups with size numSamples.
        for (int j = 0; j < ceil((float)imagesPathVector.size()/numSamples); j++) {
            int iLow = j*numSamples;
            int iHigh = (j+1 != ceil((float)imagesPathVector.size()/numSamples))?
                        (j+1)*numSamples : iLow+imagesPathVector.size()%numSamples;
            cv::Mat confusionMatrix = cv::Mat::zeros(1, samplesPaths.size(), CV_32F);
            double maxProbability, probability;

            for (int k = iLow; k < iHigh; k++) {
                double discriminator = classifier->calcProbability(imagesPathVector[k]);

                if (!isConfMatrix) {
                    cout << (classifier->isMultivariate()?"Discriminator":"Probability")
                            << " Samples" << (i==0?"Pos: ":"Neg: ") << discriminator << "\n";

                    /* Best images selection: µ ± ∂
                    discriminator /= 1000000;
                    if ((i==0 && discriminator > 3.43 && discriminator < 3.79)
                    || (i==1 && discriminator > 3.69 && discriminator < 4.06)){
                        cout<<imagesPathVector[k]<<"\n";
                    }*/
                } else {

                    if (isGrayScale) {
                        maxProbability = 70.0f; //@TODO Use constants in a file!
                        probability = discriminator;
                    } else { // isHOG
                        maxProbability = 29.15f; //@TODO Use constants in a file!
                        probability = 100*1000000/discriminator;
                    }

                    confusionMatrix.at<float>(0, (probability > maxProbability*discThreshold)? 0 : 1) += 1;
                }
            }

            if (isConfMatrix) {
                cout << "Threshold:" << maxProbability*discThreshold
                     << (i==0?" | SamplesPos ":" | SamplesNeg ") << confusionMatrix << " = "
                     << confusionMatrix.at<float>(0,i)*100/numSamples << "%\n";
            }
        }
    }
}
