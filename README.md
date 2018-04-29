# Pattern Recognition

Using basic algorithms of image processing to detect pedestrian in greyscale images a classifier is trained with example images. Here the meaningful image characteristics are the so-called features. For each training image, a feature vector is calculated from these characteristics. Training in this case basically means calculating the mean and covariance matrix of the feature vector.

  Load Images  =>  Calculate Features  =>  Select Subset  =>  Calculate Mean & Covariance

So, a Bayesian classifier with Gaussian distribution hypothesis based on the detector approach is implemented. As feature vector characteristic is possible use the greyscale values of the image, as well the Histogram of Oriented Gradients (HOG).

The results for detection and false alarm for different threshold values *T_i = i\*d_max/n* for both characteristics. The calculated decision function is tested with the images from the training and testing dataset and a transition matrix is created. (The images for training and testing are from the Daimler Pedestrian Detection Benchmark)

## --

To use just a well-defined subset of the feature vector with the strongest class specific characteristics is used the Weka tool, therefore the feature vectors from the positive and negative images are created and saved in a ARFF-format files (http://www.cs.waikato.ac.nz/ml/weka/arff.html).
