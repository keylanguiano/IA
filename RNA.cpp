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
    Size(640, 256), // winSize
    Size(64, 64), // blockSize
    Size(32, 32),   // blockStride
    Size(64, 64),   // cellSize
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
Ptr <ml::ANN_MLP> ANN_MLP_CreateBasic (int nFeatures, int nClasses);
int ANN_MLP_CreateBasic(Ptr <ml::ANN_MLP> & ann, int nFeatures, int nClasses);
int ANN_MLP_Train(Ptr <ml::ANN_MLP> & ann, Mat & train_data, Mat & trainLabelsMat, int nClasses);
int ANN_MLP_TrainAndSave(Ptr <ml::ANN_MLP> & ann, Mat & train_data, Mat & trainLabelsMat, int nClasses, char * filename_ANNmodel);

// CASE 5: TEST THE MODEL WITH THE TRAINING MATRIX AND GENERATE A CONFUSION MATRIX
int ANN_MLP_Test(Ptr <ml::ANN_MLP> & ann, Mat & testMat, Mat & testLabelsMat, int nClasses);

// CASE 6: TEST THE MODEL WITH A SINGLE IMAGE
int ANN_MLP_Test_Single(string filename, Ptr <ml::ANN_MLP> & annTRAINED);

// Global variable
// Corresponds to the size of sub-images
int SZ = 20;

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
    string routePath = "./Images/TESTING/";
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
                  cout << "THE FONT_FEATURES.csv FILE ALREADY EXISTS IN THE ROOT FOLDER." << endl;
                  cout << "DO YOU WANT TO OVERWRITE IT? (Y/N): ";
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
                  cout << "THE FONT_FEATURES.csv FILE DOES NOT EXIST IN THE ROOT FOLDER." << endl;
                  cout << "PLEASE RUN OPTION 1 TO CREATE IT." << endl;
                  system("pause");
                  break;
                        }
                        else {
                  loadFeaturesFromCSV("FONT_FEATURES.csv", fullFeatures, originalLabels);
                  splitData(fullFeatures, nFolds, trainMat, testMat, trainLabelsMat, testLabelsMat);
                            cout << "WARNING: ALL MATRIX SIZES ARE GIVEN IN A [ COLUMNS X ROWS ] FORMAT:" << "\n\n";
                            cout << "\tDESCRIPTOR SIZE : " << trainMat.cols << "\n";
                            cout << "\tNUMBER OF CLASSES: " << originalLabels.size() << "\n\n";
                            cout << "\tTRAINING MAT SIZE: " << trainMat.size() << "\n";
                            cout << "\tTESTING  MAT SIZE: " << testMat.size() << "\n\n";
                            cout << "\tTRAIN LABELS MAT SIZE: " << trainLabelsMat.size() << "\n";
                            cout << "\tTEST LABELS MAT SIZE: " << testLabelsMat.size() << "\n\n";
                  system("pause");
                }
                break;

            case '3':
                cout << "------------------------------------------------------" << endl;

                if (trainMat.empty() || testMat.empty() || trainLabelsMat.empty() || testLabelsMat.empty()) {
                  cout << "ERROR: TRAINING AND TEST MATRICES HAVE NOT BEEN LOADED." << endl;
                  cout << "PLEASE RUN OPTION 2 TO LOAD THEM." << endl;
                  cout << "OR RUN OPTION 1 TO CREATE THEM." << endl << endl;
                  system("pause");
                  break;
                }

                if (fs::exists("ANNfontTypesClassifierModel.yml")) {
                  cout << "THE FILE ANNfontTypesClassifierModel.yml ALREADY EXISTS IN THE ROOT FOLDER." << endl;
                  cout << "DO YOU WANT TO OVERWRITE IT? (Y/N): ";
                  cin >> overwrite;

                            if (overwrite == 'N' || overwrite == 'n')
                                break;
                }

                nFeatures = trainMat.cols;
                        nClasses = static_cast<int>(originalLabels.size());

                        ANN_MLP_CreateBasic(ann, nFeatures, nClasses);
                        cout << "\nTHE NUMBER OF DIFFERENT CLASSES IS " << nClasses << "\n";

                // Filename for saving/loading trained models
                filename_ANNmodel = (char*)"ANNfontTypesClassifierModel.yml";

                // Train and save the model
                ANN_MLP_TrainAndSave(ann, trainMat, trainLabelsMat, nClasses, filename_ANNmodel);

                cout << "TRAINED MODEL SAVED AS: " << filename_ANNmodel << "\n";
                system("pause");
                        break;

            case '4':
                cout << "------------------------------------------------------" << endl;

                if (!fs::exists("ANNfontTypesClassifierModel.yml")) {
                  cout << "THE FILE ANNfontTypesClassifierModel.yml DOES NOT EXIST IN THE ROOT FOLDER." << endl;
                            cout << "PLEASE RUN OPTION 3 TO CREATE IT." << endl <<  endl;
                  system("pause");
                  break;
                }

                        // CHECK IF A MODEL IS LOADED
                if (!annTRAINED.empty()) {
                  cout << "THERE IS A MODEL LOADED IN THE SYSTEM." << endl;
                  cout << "DO YOU WANT TO LOAD A NEW MODEL? (Y/N): ";
                  cin >> overwrite;

                  if (overwrite == 'N' || overwrite == 'n')
                    break;

                            cout << endl;
                }

                // RELEASE MEMORY FOR LOADED MODEL
                annTRAINED.release();
                        cout << "LOADING A TRAINED MODEL FROM FILE.\n\n";
                        // Now, we can load the saved model
                        filename_ANNmodel = (char*)"ANNfontTypesClassifierModel.yml";
                        annTRAINED = cv::ml::ANN_MLP::load(filename_ANNmodel);

                annTRAINED.empty() ? cout << "ERROR: FAILED TO LOAD THE MODEL." << "\n\n" : cout << "MODEL SUCCESSFULLY LOADED FROM FILE: " << filename_ANNmodel << "\n\n";
                system("pause");
                break;

            case '5':
                cout << "------------------------------------------------------" << endl;
                
                if (trainMat.empty() || testMat.empty() || trainLabelsMat.empty() || testLabelsMat.empty()) {
                    cout << "ERROR: TEST MATRICES HAVE NOT BEEN LOADED." << endl;
                    cout << "PLEASE RUN OPTION 2 TO LOAD THEM." << endl;
                    cout << "OR RUN OPTION 1 TO CREATE THEM." << endl << endl;
                    system("pause");
                    break;
                }

                if (annTRAINED.empty()) {
                  cout << "ERROR: NO TRAINED MODEL HAS BEEN LOADED." << endl;
                  cout << "PLEASE RUN OPTION 4 TO LOAD IT." << endl;
                  cout << "OR RUN OPTION 3 TO TRAIN A NEW MODEL." << endl << endl;
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
					cout << "ERROR: NO SE HA CARGADO UN MODELO ENTRENADO." << endl;
					cout << "POR FAVOR EJECUTE LA OPCION 4 PARA CARGARLO." << endl;
					cout << "O EJECUTE LA OPCION 3 PARA ENTRENAR UN NUEVO MODELO." << endl << endl;
					system("pause");
					break;
				}

				cout << "LA MUESTRA SE DEBE DE ENCONTRAR EN LA RUTA ./Images/TESTING/" << endl;
				cout << "INGRESE EL NOMBRE DEL ARCHIVO: ";
				cin >> filename;
				route = routePath + filename;
                pred = ANN_MLP_Test_Single(annTRAINED, route);
				cout << "\nPREDICCION: " << originalLabels[pred] << "\n\n";
				system("pause");
                break;

            default:
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
    if (imgGray.size() != Size(640, 256)) {
        cout << "ERROR: IMAGE DOES NOT HAVE THE EXPECTED SIZE OF 640 x 256." << endl;
        return;
    }

    // Calculate HOG for the complete image
    vector<float> hogFeatures;
    try {
        hog.compute(imgGray, hogFeatures);
    }
    catch (const cv::Exception& e) {
        cout << "ERROR: EXCEPTION WHEN CALCULATING HOG: " << e.what() << endl;
        return;
    }

    // Insert the HOG features into the features vector
    features.insert(features.end(), hogFeatures.begin(), hogFeatures.end());
}

// PROCESS SAMPLES AND SAVE FEATURES
void processImagesAndSaveFeatures() {
    string samplesPath = "./images/MUESTRAS"; // Path to the samples folder
    vector<vector<float>> data;
    vector<string> labels;

    cout << endl;
    for (const auto& entry : fs::directory_iterator(samplesPath)) {
        cout << "PROCESSING SAMPLE: " << entry.path().filename().string() << endl;
        if (fs::is_directory(entry)) {
            string label = entry.path().filename().string();
            for (const auto& imgEntry : fs::directory_iterator(entry.path())) {
                if (imgEntry.path().extension() == ".jpg" || imgEntry.path().extension() == ".png") {
                    Mat image = imread(imgEntry.path().string());
                    if (image.empty()) {
                        cout << "ERROR: COULD NOT LOAD IMAGE: " << imgEntry.path() << endl;
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
        cout << "\nFEATURES SAVED TO FILE: FONT_FEATURES.csv" << endl;
}

// SAVE FEATURES TO A .CSV FILE
bool saveToCSV(const string& filename, const vector<vector<float>>& data, const vector<string>& labels) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "\nERROR: COULD NOT OPEN FILE TO WRITE: " << filename << endl;
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
        cout << "\nERROR: COULD NOT OPEN FILE TO READ: " << filename << endl;
        return;
    }

    vector<vector<float>> data;
    vector<string> labels;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string feature;
        vector<float> features;
        while (getline(ss, feature, ',')) {
            if (feature.empty()) {
                continue;
            }

            if (feature.find_first_not_of("0123456789.-") != string::npos) {
                auto it = find(originalLabels.begin(), originalLabels.end(), feature);
                if (it == originalLabels.end()) {
                    originalLabels.push_back(feature);
                    features.push_back(static_cast<int>(originalLabels.size() - 1));
                } else {
                    features.push_back(static_cast<int>(distance(originalLabels.begin(), it)));
                }
            } else {
                features.push_back(stof(feature));
            }
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

    cout << "FEATURES LOADED SUCCESSFULLY FROM FILE: " << filename << endl;
}

// FUNCTION TO SAVE ANY MATRIX TO A .CSV FILE
void saveMatToCSV(const string& filename, const Mat& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: COULD NOT OPEN FILE TO WRITE: " << filename << endl;
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
    cout << "\nFOR " << nFolds << "-FOLD TEST: " << nSamplesPerFold << " SAMPLES PER FOLD; " << nSamplesLearn << " FOR LEARNING, " << nSamplesTest << " FOR TESTING.\n";

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
int ANN_MLP_CreateBasic (Ptr <ml::ANN_MLP> & ann, int nFeatures, int nClasses)
{
    cout << "\nCREATING AN ANN_MLP\n";

    ann = ml::ANN_MLP::create ();
    Mat_ <int> layers (3, 1);
    layers(0) = nFeatures;              // input
    layers (1) = nFeatures * 2 + 1;     // hidden
    layers(2) = nClasses;               // output, 1 pin per class.
    ann -> setLayerSizes (layers);
    ann -> setActivationFunction (ml::ANN_MLP::SIGMOID_SYM, 0, 0);
    ann -> setTermCriteria (TermCriteria (TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001));
    ann -> setTrainMethod (ml::ANN_MLP::BACKPROP, 0.0001);

    return true;
}

Ptr <ml::ANN_MLP> ANN_MLP_CreateBasic (int nFeatures, int nClasses)
{
    cout << "\n\n\nCreating an ANN_MLP\n\n";

    Ptr <ml::ANN_MLP> ann = ml::ANN_MLP::create ();
    Mat_ <int> layers (3, 1);
    layers (0) = nFeatures;
    layers (1) = nFeatures * 2 + 1;
    layers (2) = nClasses;
    ann -> setLayerSizes (layers);
    ann -> setActivationFunction (ml::ANN_MLP::SIGMOID_SYM, 0, 0);
    ann -> setTermCriteria (TermCriteria (TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001));
    ann -> setTrainMethod (ml::ANN_MLP::BACKPROP, 0.0001);

    return ann;
}

int ANN_MLP_Train (Ptr <ml::ANN_MLP> & ann, Mat & train_data, Mat & trainLabelsMat, int nClasses)
{
    Mat train_classes = Mat::zeros (train_data.rows, nClasses, CV_32FC1);

    for (int i = 0; i < train_classes.rows; i ++)
    {
        train_classes.at <float> (i, trainLabelsMat.at <uchar> (i)) = 1.0;
    }

    cout << "\nTrain data size: " << train_data.size () << "\nTrain classes size: " << train_classes.size () << "\n\n";

    cout << "Training the ANN... (please wait)\n";
    ann -> train (train_data, ml::ROW_SAMPLE, train_classes);
    cout << "Done.\n\n";

    return 0;
}

int ANN_MLP_TrainAndSave(Ptr<ml::ANN_MLP> &ann, Mat &train_data, Mat &trainLabelsMat, int nClasses, char *filename_ANNmodel)
{
    Mat train_classes = Mat::zeros(train_data.rows, nClasses, CV_32FC1);

    for(int i = 0; i < train_classes.rows; i ++)
    {
        train_classes.at<float>(i, trainLabelsMat.at <uchar> (i)) = 1.0;
    }

    cout << "\nTRAIN DATA SIZE: " << train_data.size () << "\nTRAIN CLASSES SIZE: " << train_classes.size () << "\n";
    cout << "\nTRAINING THE ANN... (PLEASE WAIT)\n";
    ann -> train(train_data, ml::ROW_SAMPLE, train_classes);

    ann -> save(filename_ANNmodel);
    cout << "\nTRAINED MODEL SAVED AS " << filename_ANNmodel <<  "\n\n";

    return 0;
}

// CASE 5
// CREATE A FUNCTION TO TEST THE MODEL WITH THE TEST MATRIX AND GENERATE A CONFUSION MATRIX
int ANN_MLP_Test(Ptr <ml::ANN_MLP>& ann, Mat& test_data, Mat& testLabelsMat, int nClasses)
{
    cout << "ANN PREDICTION TEST\n\n";

    Mat confusion(nClasses, nClasses, CV_32S, Scalar(0));
    cout << "CONFUSION MATRIX SIZE: " << confusion.size() << "\n\n";

    // Test samples in test_data Mat
    for (int i = 0; i < test_data.rows; i++)
    {
        int pred = ann->predict(test_data.row(i), noArray());
        int truth = testLabelsMat.at <uchar>(i);
        confusion.at <int>(truth, pred)++;
    }

    cout << "CONFUSION MATRIX:\n" << confusion << endl;

    Mat correct = confusion.diag();
    float accuracy = sum(correct)[0] / sum(confusion)[0];
    cout << "\nACCURACY: " << accuracy << "\n\n";
    return 0;
}

// CASE 6: TEST THE MODEL WITH A SINGLE IMAGE
int ANN_MLP_Test_Single(string filename, Ptr <ml::ANN_MLP>& annTRAINED)
{
    // Cargar la imagen de muestra
    Mat sample = imread(imagePath);
    if (sample.empty())
    {
        cerr << "ERROR: NO SE PUDO CARGAR LA IMAGEN DESDE LA RUTA PROPORCIONADA." << endl;
        return -1;
    }

    // Preprocesamiento de la muestra
    Mat preprocSample;
    cvtColor(sample, preprocSample, COLOR_BGR2GRAY); // Convertir a escala de grises

    // Invertir colores si es necesario (fondo negro y letras blancas)
    double minVal, maxVal;
    minMaxLoc(preprocSample, &minVal, &maxVal);
    if (maxVal == 0) {
        cerr << "ERROR: LA IMAGEN ESTÁ COMPLETAMENTE NEGRA." << endl;
        return -1;
    }
    if (minVal == 0 && maxVal == 255) {
        // La imagen ya tiene fondo negro y letras blancas
    }
    else {
        // Invertir colores
        bitwise_not(preprocSample, preprocSample);
    }

    preprocSample.convertTo(preprocSample, CV_8U, 1.0 / 255.0); // Normalizar

    vector<float> featureVector;
    hog.compute(preprocSample, featureVector);
    int numFeatures = featureVector.size();

    // Vector a matriz
    Mat underTest = Mat::zeros(1, numFeatures, CV_32FC1);
    for (int k = 0; k < numFeatures; k++)
        underTest.at<float>(0, k) = featureVector[k];

    // Predicción
    int predictedClass = static_cast<int>(annTRAINED->predict(underTest, noArray()));
    return predictedClass;
}