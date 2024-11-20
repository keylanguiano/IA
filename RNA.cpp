#include <iostream>
#include <filesystem>
#include <fstream>
using namespace std;

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
using namespace cv::ml;
using namespace cv;
namespace fs = std::filesystem;

// For feature extraction
HOGDescriptor hog
(
    Size(1024, 512), // winSize
    Size(128, 128),   // blockSize
    Size(64, 64),   // blockStride
    Size(128, 128),   // cellSize
    9,              // nbins
    1,              // derivAper
    -1,             // winSigma
    cv::HOGDescriptor::HistogramNormType::L2Hys, // histogramNormType
    0.2,            // L2HysThresh
    0,              // gammaCorrection
    64,             // nlevels
    1               // signedGradient
);

void saveMatToCSV(const string& filename, const Mat& matrix);

// CASE 1: FEATURE EXTRACTION AND SAVING TO .CSV
void processImagesAndSaveFeatures();
void extractTextureFeatures(const Mat& image, vector<float>& features);
bool saveToCSV(const string& filename, const vector<vector<float>>& data, const vector<string>& labels);

// CASE 2: LOAD FONT_FEATURES.CSV FILE AND SAVE DATA TO MATRICES
void loadFeaturesFromCSV(const string& filename, Mat& fullFeatures, vector<string>& originalLabels);
Mat shuffleData(Mat fullFeatures);
void splitData(const Mat fullFeatures, int nFolds, Mat& trainMat, Mat& testMat, Mat& trainLabelsMat, Mat& testLabelsMat);

// CASE 3: TRAIN THE ANN MLP WITH MATRICES LOADED INTO THE SYSTEM AND SAVE THE MODEL
int ANN_MLP_CreateBasic(Ptr <ml::ANN_MLP> & ann, int nFeatures, int nClasses);
int ANN_MLP_TrainAndSave(Ptr <ml::ANN_MLP> & ann, Mat & train_data, Mat & trainLabelsMat, int nClasses, char * filename_ANNmodel);

// CASE 5: TEST THE MODEL WITH THE TRAINING MATRIX AND GENERATE A CONFUSION MATRIX
int ANN_MLP_Test(Ptr <ml::ANN_MLP> & ann, Mat & testMat, Mat & testLabelsMat, int nClasses);

// CASE 6: TEST THE MODEL WITH A SINGLE IMAGE
int ANN_MLP_Test_Single(Ptr<ml::ANN_MLP>& annTRAINED, const string& imagePath);

int main(void)
{
    // CASE 2 VARIABLES:
	int nFolds = 10;
    vector<string> originalLabels;
	Mat fullFeatures;
    Mat trainMat, testMat, trainLabelsMat, testLabelsMat;

    // CASE 3 VARIABLES:
    int nFeatures = 0;
    int nClasses = 0;
    Ptr <ml::ANN_MLP> ann;
    char* filename_ANNmodel{};

    // CASE 3 VARIABLES:
    Ptr<ANN_MLP> annTRAINED;

    // CASE 6 VARIABLES:
    string routePath = "../Images/TRAINING DATASET/Berlin Sans FB/";
    string filename;
	string route;
    int pred;

    char overwrite;
    char opcion = ' ';

    do
    {
        cout << "------------------------------------------------------" << endl;
        cout << "IA PRACTICE | ANN MLP RECOGNIZER OF FONT TYPES IN TEXTS. \n" << endl;

        cout << "THIS PROGRAM CAN PERFORM THE FOLLOWING TASKS:" << endl;
        cout << "\t 1. EXTRACT FEATURES AND SAVE TO A .CSV FILE" << endl;
        cout << "\t 2. LOAD THE .CSV FILE TO SEPARATE TRAINING AND TEST MATRICES" << endl;
        cout << "\t 3. TRAIN THE ANN MLP WITH MATRICES LOADED INTO THE SYSTEM AND SAVE THE MODEL" << endl;
        cout << "\t 4. LOAD A TRAINED MODEL" << endl;
        cout << "\t 5. TEST THE MODEL WITH THE TRAINING MATRIX AND GENERATE A CONFUSION MATRIX" << endl;
        cout << "\t 6. TEST THE MODEL WITH A SINGLE IMAGE" << endl;
        cout << "\nPRESS ANY KEY TO EXIT" << endl;

        cout << "OPTION: ";
        cin >> opcion;

        switch (opcion) {
            case '1':
                cout << "------------------------------------------------------" << endl;
                // Check if the font_features.csv file exists in the root folder
				if (fs::exists("FONT_FEATURES.csv")) {
					cout << "\tTHE FONT_FEATURES.csv FILE ALREADY EXISTS IN THE ROOT FOLDER." << endl;
					cout << "\tDO YOU WANT TO OVERWRITE IT? (Y/N): ";
					cin >> overwrite;

					if (overwrite == 'Y' || overwrite == 'y') {
						processImagesAndSaveFeatures();
					}
				}
				else {
					processImagesAndSaveFeatures();
				}

                system("pause");
                break;

            case '2':
                cout << "------------------------------------------------------" << endl;
				
                // Check if the font_features.csv file exists in the root folder, if not, recommend option 1 to create it
                if (!fs::exists("FONT_FEATURES.csv")) {
					cout << "\tTHE FONT_FEATURES.csv FILE DOES NOT EXIST IN THE ROOT FOLDER." << endl;
					cout << "\tPLEASE RUN OPTION 1 TO CREATE IT." << endl;
					system("pause");
					break;
                }
                else {
					loadFeaturesFromCSV("FONT_FEATURES.csv", fullFeatures, originalLabels);
					splitData(fullFeatures, nFolds, trainMat, testMat, trainLabelsMat, testLabelsMat);
                    cout << "\tWARNING: ALL MATRIX SIZES ARE GIVEN IN A [ COLUMNS X ROWS ] FORMAT:" << "\n\n";
                    cout << "\tDESCRIPTOR SIZE:\t" << trainMat.cols << "\n";
                    cout << "\tNUMBER OF CLASSES:\t" << originalLabels.size() << "\n\n";
                    cout << "\tTRAINING MAT SIZE:\t" << trainMat.size() << "\n";
                    cout << "\tTESTING  MAT SIZE:\t" << testMat.size() << "\n\n";
                    cout << "\tTRAIN LABELS MAT SIZE:\t" << trainLabelsMat.size() << "\n";
                    cout << "\tTEST LABELS MAT SIZE:\t" << testLabelsMat.size() << "\n\n";

					system("pause");
                }
                break;

            case '3':
                cout << "------------------------------------------------------" << endl;

				if (trainMat.empty() || testMat.empty() || trainLabelsMat.empty() || testLabelsMat.empty()) {
					cout << "\tERROR: TRAINING AND TEST MATRICES HAVE NOT BEEN LOADED." << endl;
					cout << "\tPLEASE RUN OPTION 2 TO LOAD THEM." << endl;
					cout << "\tOR RUN OPTION 1 TO CREATE THEM." << endl << endl;
					system("pause");
					break;
				}

				if (fs::exists("ANNfontTypesClassifierModel.yml")) {
					cout << "\tTHE FILE ANNfontTypesClassifierModel.yml ALREADY EXISTS IN THE ROOT FOLDER." << endl;
					cout << "\tDO YOU WANT TO OVERWRITE IT? (Y/N): ";
					cin >> overwrite;

                    if (overwrite == 'N' || overwrite == 'n')
                        break;
				}

				nFeatures = trainMat.cols;
                nClasses = static_cast<int>(originalLabels.size());

                ANN_MLP_CreateBasic(ann, nFeatures, nClasses);
                cout << "\n\tTHE NUMBER OF DIFFERENT CLASSES IS " << nClasses << "\n";

				// Filename for saving/loading trained models
				filename_ANNmodel = (char*)"ANNfontTypesClassifierModel.yml";

				// Train and save the model
				ANN_MLP_TrainAndSave(ann, trainMat, trainLabelsMat, nClasses, filename_ANNmodel);
				system("pause");
                break;

            case '4':
                cout << "------------------------------------------------------" << endl;

				if (!fs::exists("ANNfontTypesClassifierModel.yml")) {
					cout << "\tTHE FILE ANNfontTypesClassifierModel.yml DOES NOT EXIST IN THE ROOT FOLDER." << endl;
                    cout << "\tPLEASE RUN OPTION 3 TO CREATE IT." << endl <<  endl;
					system("pause");
					break;
				}

                // CHECK IF A MODEL IS LOADED
				if (!annTRAINED.empty()) {
					cout << "\tTHERE IS A MODEL LOADED IN THE SYSTEM." << endl;
					cout << "\tDO YOU WANT TO LOAD A NEW MODEL? (Y/N): ";
					cin >> overwrite;

					if (overwrite == 'N' || overwrite == 'n')
						break;

                    cout << endl;
				}

				// RELEASE MEMORY FOR LOADED MODEL
				annTRAINED.release();
                cout << "\tLOADING A TRAINED MODEL FROM FILE.\n\n";
                // Now, we can load the saved model
                filename_ANNmodel = (char*)"ANNfontTypesClassifierModel.yml";
                annTRAINED = cv::ml::ANN_MLP::load(filename_ANNmodel);

				annTRAINED.empty() ? cout << "\tERROR: FAILED TO LOAD THE MODEL." << "\n\n" : cout << "\tMODEL SUCCESSFULLY LOADED FROM FILE: " << filename_ANNmodel << "\n\n";
				system("pause");
                break;

            case '5':
                cout << "------------------------------------------------------" << endl;
                
                if (trainMat.empty() || testMat.empty() || trainLabelsMat.empty() || testLabelsMat.empty()) {
                    cout << "\tERROR: TEST MATRICES HAVE NOT BEEN LOADED." << endl;
                    cout << "\tPLEASE RUN OPTION 2 TO LOAD THEM." << endl;
                    cout << "\tOR RUN OPTION 1 TO CREATE THEM." << endl << endl;
                    system("pause");
                    break;
                }

				if (annTRAINED.empty()) {
					cout << "\tERROR: NO TRAINED MODEL HAS BEEN LOADED." << endl;
					cout << "\tPLEASE RUN OPTION 4 TO LOAD IT." << endl;
					cout << "\tOR RUN OPTION 3 TO TRAIN A NEW MODEL." << endl << endl;
					system("pause");
					break;
				}

                nClasses = static_cast<int>(originalLabels.size());
                ANN_MLP_Test(annTRAINED, testMat, testLabelsMat, nClasses);
                system("pause");
                break;

            case '6':
                cout << "------------------------------------------------------" << endl;

                if (annTRAINED.empty()) {
                    cout << "\tERROR: NO TRAINED MODEL LOADED." << endl;
                    cout << "\tPLEASE RUN OPTION 4 TO LOAD IT." << endl;
                    cout << "\tOR RUN OPTION 3 TO TRAIN A NEW MODEL." << endl << endl;
                    system("pause");
                    break;
                }

                // Keyla
                // cout << "\tTHE SAMPLE MUST BE FOUND ON THE ROUTE ../Images/TESTING/" << endl;
                cout << "\nNOTE: THE IMAGE FOR TESTING MUST HAVE DIMENSIONS " << hog.winSize.width << "x" << hog.winSize.height << endl;
                cout << "\tTHE SAMPLE MUST BE FOUND ON THE ROUTE ./Images/TESTING/" << endl;
                cout << "\tENTER FILE NAME: ";
                cin >> filename;
                route = routePath + filename;
                pred = ANN_MLP_Test_Single(annTRAINED, route);
                cout << "\n\tPREDICTION: " << originalLabels[pred] << "\n\n";
                system("pause");
                break;

            default:
                exit(0);
                break;
        }
    } while (1);

    return 0;
}

// CASE 1
// FEATURE EXTRACTION AND SAVING TO .CSV
void extractTextureFeatures(const Mat& image, vector<float>& features) {
    // Convert the image to grayscale
    Mat imgGray;
    cvtColor(image, imgGray, COLOR_BGR2GRAY);

    // Check that the image has the expected size
    if (imgGray.size() != Size(1024, 512)) {
        cout << "\tERROR: IMAGE DOES NOT HAVE THE EXPECTED SIZE OF 1024 x 512." << endl;
        return;
    }

    // Calculate HOG for the complete image
    vector<float> hogFeatures;
    try {
        hog.compute(imgGray, hogFeatures);
    }
    catch (const cv::Exception& e) {
        cout << "\n\tERROR: EXCEPTION WHEN CALCULATING HOG: " << e.what() << endl;
        return;
    }

    // Insert the HOG features into the features vector
    features.insert(features.end(), hogFeatures.begin(), hogFeatures.end());
}

// PROCESS SAMPLES AND SAVE FEATURES
void processImagesAndSaveFeatures() {
// Keyla
// string samplesPath = "../IMAGES/TRAINING DATASET/"; // Path to the samples folder
    string samplesPath = "./images/TRAINING DATASET"; // Path to the samples folder
    vector<vector<float>> data;
    vector<string> labels;

    cout << endl;
    for (const auto& entry : fs::directory_iterator(samplesPath)) {
        cout << "\tPROCESSING SAMPLE: " << entry.path().filename().string() << endl;
        if (fs::is_directory(entry)) {
            string label = entry.path().filename().string();
            for (const auto& imgEntry : fs::directory_iterator(entry.path())) {
                if (imgEntry.path().extension() == ".jpg" || imgEntry.path().extension() == ".png") {
                    Mat image = imread(imgEntry.path().string());
                    if (image.empty()) {
                        cout << "\tERROR: COULD NOT LOAD IMAGE: " << imgEntry.path() << endl;
                        continue;
                    }

                    vector<float> features;
                    extractTextureFeatures(image, features);
                    data.push_back(features);
                    labels.push_back(label);
                }
            }
        }
    }

    bool success = saveToCSV("FONT_FEATURES.csv", data, labels);

    if (success)
        cout << "\n\tFEATURES SAVED TO FILE: FONT_FEATURES.csv" << endl << endl;
}

// SAVE FEATURES TO A .CSV FILE
bool saveToCSV(const string& filename, const vector<vector<float>>& data, const vector<string>& labels) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "\n\tERROR: COULD NOT OPEN FILE TO WRITE: " << filename << endl;
        return false;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        for (const auto& feature : data[i]) {
            file << feature << ",";
        }
        file << labels[i] << "\n";
    }

    file.close();
    return true;
}

// CASE 2
// LOAD FONT_FEATURES.CSV FILE AND STORE DATA IN MATRICES
void loadFeaturesFromCSV(const string& filename, Mat& fullFeatures, vector<string>& originalLabels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "\n\tERROR: COULD NOT OPEN FILE TO READ: " << filename << endl;
        return;
    }

    vector<vector<float>> data;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string feature;
        vector<float> features;
        vector<string> rowElements;

        // Read all elements of the row
        while (getline(ss, feature, ',')) {
            if (!feature.empty()) {
                rowElements.push_back(feature);
            }
        }

        // Process all features except the last column
        for (size_t i = 0; i < rowElements.size() - 1; ++i) {
            features.push_back(stof(rowElements[i]));
        }

        // Process the last column as a label
        string label = rowElements.back();
        auto it = find(originalLabels.begin(), originalLabels.end(), label);
        if (it == originalLabels.end()) {
            originalLabels.push_back(label);
            features.push_back(static_cast<int>(originalLabels.size() - 1));
        } else {
            features.push_back(static_cast<int>(distance(originalLabels.begin(), it)));
        }

        data.push_back(features);
    }

    file.close();

    // Convert data to a feature matrix
    fullFeatures = Mat(static_cast<int>(data.size()), static_cast<int>(data[0].size()), CV_32F);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            fullFeatures.at<float>(static_cast<int>(i), static_cast<int>(j)) = data[i][j];
        }
    }

    cout << "\tFEATURES LOADED SUCCESSFULLY FROM FILE: " << filename << endl;
}

// FUNCTION TO SAVE ANY MATRIX TO A .CSV FILE
void saveMatToCSV(const string& filename, const Mat& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "\n\tERROR: COULD NOT OPEN FILE TO WRITE: " << filename << endl;
        return;
    }

    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            file << matrix.at<float>(i, j) << ",";
        }
        file << "\n";
    }

    file.close();
}

// SHUFFLE THE FEATURE MATRIX AND LABELS
Mat shuffleData(Mat fullFeatures) {
    Mat shuffledFeatures;
    Mat shuffledLabels;
    Mat indices = Mat::zeros(fullFeatures.rows, 1, CV_32S);
    randu(indices, Scalar(0), Scalar(fullFeatures.rows));

    for (int i = 0; i < fullFeatures.rows; ++i) {
        int index = indices.at<int>(i);
        Mat feature = fullFeatures.row(index);
        shuffledFeatures.push_back(feature);
    }

    return shuffledFeatures;
}

// SPLIT FEATURE MATRIX AND LABELS INTO trainMat, testMat, trainLabelsMat, testLabelsMat ACCORDING TO nFolds
void splitData(const Mat fullFeatures, int nFolds, Mat& trainMat, Mat& testMat, Mat& trainLabelsMat, Mat& testLabelsMat) {
    // Check if trainMat, testMat, trainLabelsMat, testLabelsMat are empty; if they have data, release memory
    if (!trainMat.empty()) {
        trainMat.release();
    }
    if (!testMat.empty()) {
        testMat.release();
    }
    if (!trainLabelsMat.empty()) {
        trainLabelsMat.release();
    }
    if (!testLabelsMat.empty()) {
        testLabelsMat.release();
    }

    // Shuffle data
    Mat shuffled = shuffleData(fullFeatures);

    // Calculate the number of samples per fold
    int nSamplesPerFold = shuffled.rows / nFolds;
    int nSamplesLearn = nSamplesPerFold * (nFolds - 1);
    int nSamplesTest = shuffled.rows - nSamplesLearn;
    cout << "\n\tFOR " << nFolds << "-FOLD TEST: " << nSamplesPerFold << " SAMPLES PER FOLD; " << nSamplesLearn << " FOR LEARNING, " << nSamplesTest << " FOR TESTING.\n";

    // Split data into trainMat, testMat, trainLabelsMat, testLabelsMat
    trainMat = shuffled(Range(0, nSamplesLearn), Range(0, shuffled.cols - 1));
    testMat = shuffled(Range(nSamplesLearn, shuffled.rows), Range(0, shuffled.cols - 1));
    trainLabelsMat = shuffled(Range(0, nSamplesLearn), Range(shuffled.cols - 1, shuffled.cols));
    testLabelsMat = shuffled(Range(nSamplesLearn, shuffled.rows), Range(shuffled.cols - 1, shuffled.cols));

    // Convert labels to integers
    trainLabelsMat.convertTo(trainLabelsMat, CV_8UC1);
    testLabelsMat.convertTo(testLabelsMat, CV_8UC1);
}

// CASE 3
// TRAINING THE CLASSIFIER
int ANN_MLP_CreateBasic(Ptr <ml::ANN_MLP> & ann, int nFeatures, int nClasses)
{
    cout << "\tCREATING AN ANN_MLP\n";

    ann = ml::ANN_MLP::create();
    Mat_<int>layers (3, 1);
    layers(0) = nFeatures;              // input
    layers(1) = nFeatures * 2 + 1;     // hidden
    layers(2) = nClasses;               // output, 1 pin per class.
    ann -> setLayerSizes(layers);
    ann -> setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
    ann -> setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001));
    ann -> setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);

    return true;
}

int ANN_MLP_TrainAndSave(Ptr<ml::ANN_MLP>& ann, Mat& train_data, Mat& trainLabelsMat, int nClasses, char* filename_ANNmodel)
{
    Mat train_classes = Mat::zeros(train_data.rows, nClasses, CV_32FC1);

    for (int i = 0; i < train_classes.rows; i++)
    {
        train_classes.at<float>(i, trainLabelsMat.at <uchar>(i)) = 1.0;
    }

    cout << "\n\tTRAIN DATA SIZE:\t" << train_data.size() << "\n\tTRAIN CLASSES SIZE:\t" << train_classes.size() << "\n";
    cout << "\n\tTRAINING THE ANN... (PLEASE WAIT)\n";

    // Set up the training parameters
    TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001);
    ann->setTermCriteria(criteria);
    ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);

    // Train the network
    bool trained = ann->train(train_data, ml::ROW_SAMPLE, train_classes);

    if (trained) {
        cout << "\n\tTRAINING COMPLETED SUCCESSFULLY.\n";
    }
    else {
        cout << "\n\tTRAINING FAILED.\n";
    }

    ann->save(filename_ANNmodel);
    cout << "\n\tTRAINED MODEL SAVED AS: " << filename_ANNmodel << "\n\n";

    return 0;
}


// CASE 5
// CREATE A FUNCTION TO TEST THE MODEL WITH THE TEST MATRIX AND GENERATE A CONFUSION MATRIX
int ANN_MLP_Test(Ptr <ml::ANN_MLP>& ann, Mat& test_data, Mat& testLabelsMat, int nClasses)
{
    cout << "\tANN PREDICTION TEST\n\n";

    Mat confusion(nClasses, nClasses, CV_32S, Scalar(0));
    cout << "\tCONFUSION MATRIX SIZE: " << confusion.size() << "\n\n";

    // Test samples in test_data Mat
    for (int i = 0; i < test_data.rows; i++)
    {
        int pred = ann->predict(test_data.row(i), noArray());
        int truth = testLabelsMat.at <uchar>(i);
        confusion.at <int>(truth, pred)++;
    }

    cout << "\tCONFUSION MATRIX:\n\t" << confusion << endl;

    Mat correct = confusion.diag();
    float accuracy = sum(correct)[0] / sum(confusion)[0];
    cout << "\n\tACCURACY: " << accuracy << "\n\n";
    return 0;
}

// CASE 6: TEST THE MODEL WITH A SINGLE IMAGE
int ANN_MLP_Test_Single(Ptr<ml::ANN_MLP>& annTRAINED, const string& imagePath)
{
    // Load the sample image
    Mat sample = imread(imagePath);
    if (sample.empty())
    {
        cerr << "\tERROR: COULD NOT LOAD IMAGE FROM THE PROVIDED PATH." << endl;
        return -1;
    }

    // Check the image dimensions
    if (sample.cols != hog.winSize.width || sample.rows != hog.winSize.height) {
        cerr << "\tERROR: THE IMAGE MUST HAVE DIMENSIONS " << hog.winSize.width << "x" << hog.winSize.height << " TO BE PROCESSED." << endl;
        return -1;
    }

    // Preprocessing the sample
    Mat preprocSample;
    cvtColor(sample, preprocSample, COLOR_BGR2GRAY); // Convert to grayscale

    // Invert colors if necessary (black background and white letters)
    double minVal, maxVal;
    minMaxLoc(preprocSample, &minVal, &maxVal);
    if (maxVal == 0) {
        cerr << "\tERROR: THE IMAGE IS COMPLETELY BLACK." << endl;
        return -1;
    }
    if (minVal == 0 && maxVal == 255) {
        // The image already has a black background and white letters
    }
    else {
        // Invert colors
        bitwise_not(preprocSample, preprocSample);
    }

    preprocSample.convertTo(preprocSample, CV_8U, 1.0 / 255.0); // Normalize

    vector<float> featureVector;
    hog.compute(preprocSample, featureVector);
    int numFeatures = featureVector.size();

    // Vector to matrix
    Mat underTest = Mat::zeros(1, numFeatures, CV_32FC1);
    for (int k = 0; k < numFeatures; k++)
        underTest.at<float>(0, k) = featureVector[k];

    // Prediction
    Mat response;
    annTRAINED->predict(underTest, response);

    // Get the predicted class and its probability
    Point maxLoc;
    minMaxLoc(response, 0, 0, 0, &maxLoc);
    int predictedClass = maxLoc.x;
    float confidence = response.at<float>(0, predictedClass);

    cout << "Predicted class: " << predictedClass << endl;
    cout << "Confidence: " << confidence << endl;

    return predictedClass;
}
