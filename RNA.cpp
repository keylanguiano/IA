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

// CASE 1: EXTRACCION DE CARACTERISTICAS Y GUARDADO EN .CVS
void processImagesAndSaveFeatures();
void extractTextureFeatures(const Mat& image, vector<float>& features);
void saveToCSV(const string& filename, const vector<vector<float>>& data, const vector<string>& labels);

// CASE 2: CARGAR ARCHIVO FONT_FEATURES.CSV Y GUARDAR LOS DATOS EN LAS MATRICES
void loadFeaturesFromCSV(const string& filename, Mat& fullFeatures);
Mat shuffleData(Mat fullFeatures);
void splitData(const Mat fullFeatures, int nFolds, Mat& trainMat, Mat& testMat, Mat& trainLabelsMat, Mat& testLabelsMat);

// Preprocessing
void CreateDeskewedTrainAndTestImageSets (vector <Mat> & deskewedTrainImages, vector <Mat> & deskewedTestImages, vector <Mat> & trainImages, vector <Mat> & testImages);
Mat deskew (Mat & img);
void CreateTrainAndTestHOGs (vector <vector <float>> & trainHOG, vector <vector <float>> & testHOG, vector <Mat> & deskewedtrainImages, vector <Mat> & deskewedtestImages);

// Feature matrices
int syFeatureMatricesForTestAndTrain (vector <Mat> & trainImages, vector <Mat> & testImages, vector <int> & trainLabels, vector <int> & testLabels, Mat & trainMat, Mat & testMat, Mat & trainLabelsMat, Mat & testLabelsMat);
void ConvertVectorToMatrix (vector <vector <float>> & trainHOG, vector <vector <float>> & testHOG, Mat & trainMat, Mat & testMat);

// Multilayer perceptron
void ANN_train_test (int nclasses, const Mat & train_data, const Mat & trainLabelsMat, const Mat & test_data, const Mat & testLabelsMat, Mat & confusion);
Ptr <ml::ANN_MLP> syANN_MLP_CreateBasic (int nFeatures, int nClasses);
int syANN_MLP_CreateBasic (Ptr <ml::ANN_MLP> & ann,int nFeatures, int nClasses);
int syANN_MLP_Train (Ptr <ml::ANN_MLP> & ann, Mat & train_data, Mat & trainLabelsMat, int nClasses);
int syANN_MLP_TrainAndSave( Ptr <ml::ANN_MLP> & ann, Mat & train_data, Mat & trainLabelsMat, int nClasses, char * filename_ANNmodel);
int syANN_MLP_Test (Ptr <ml::ANN_MLP> & ann, Mat & test_data, Mat & testLabelsMat, int nClasses);
int syANN_MLP_Test_Single (string filename, Ptr <ml::ANN_MLP> & annTRAINED);

// Global variable
// Corresponds to the size of sub-images
int SZ = 20;

int main(void)
{
    // CASE 1 VARIABLES:
    char overwrite;

    // CASE 2 VARIABLES:
	int nFolds = 10;
	Mat fullFeatures;
    Mat trainMat, testMat, trainLabelsMat, testLabelsMat;

    // CASE 3 VARIABLES:
    int nFeatures = 0;
    int nClasses = 0;
    Ptr <ml::ANN_MLP> ann;
    char* filename_ANNmodel{};

    // CASE 4 VARIABLES:
    Ptr<ANN_MLP> annTRAINED;

    // CASE 5 VARIABLES:
    string filename = " ";
    int label = 0;
    string file2 = " ";
    string file3 = " ";
    Mat testImage;

    char opcion = ' ';

    do
    {
        cout << "------------------------------------------------------" << endl;
        cout << "IA PRACTICE | ANN MLP RECOGNIZER OF FONT TYPES IN TEXTS. \n" << endl;

        cout << "ESTE PROGRAMA PUEDE REALIZAR LAS SIGUIENTES TAREAS:" << endl;
        cout << "\t 1. EXTRAER LAS CARACTERISTICAS Y GUARDAR EN UN ARCHIVO .CVS" << endl;
        cout << "\t 2. CARGAR EL ARCHIVO .CVS PARA SEPARAR LAS MATRICES DE ENTRENAMIENTO Y PRUEBA" << endl;
        cout << "\t 3. ENTRENAR LA ANN MLP CON LAS MATRICES CARGADAS EN EL SISTEMA Y GUARDAR EL MODELO" << endl;
        cout << "\t 4. CARGAR MODELO ENTRENADO" << endl;
        cout << "\t 5. PROBAR EL MODELO CON LA MATRIZ DE ENTRENAMIENTO Y GENERAR UNA MATRIZ DE CONFUSION" << endl;
        cout << "\t 6. PROBAR EL MODELO CON UNA IMAGEN" << endl;
        cout << "\nINGRESE CUALQUIER TECLA PARA SALIR" << endl;

        cout << "OPCION: ";
        cin >> opcion;

        switch (opcion) {
            case '1':
                cout << "------------------------------------------------------" << endl;
     
                // Revisar si existe en la carpeta raiz el archivo font_features.csv
				if (fs::exists("FONT_FEATURES.csv")) {
					cout << "EL ARCHIVO FONT_FEATURES.csv YA EXISTE EN LA CARPETA RAIZ." << endl;
					cout << "¿DESEA SOBREESCRIBIRLO? (S/N): ";
					cin >> overwrite;

					if (overwrite == 'S' || overwrite == 's') {
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
				
                // Revisar si existe en la carpeta raiz el archivo font_features.csv, si no existe recomendar la opcion 1 para 
                if (!fs::exists("FONT_FEATURES.csv")) {
					cout << "EL ARCHIVO FONT_FEATURES.csv NO EXISTE EN LA CARPETA RAIZ." << endl;
					cout << "POR FAVOR EJECUTE LA OPCION 1 PARA CREARLO." << endl;
					system("pause");
					break;
                }
                else {
					loadFeaturesFromCSV("FONT_FEATURES.csv", fullFeatures);
					splitData(fullFeatures, nFolds, trainMat, testMat, trainLabelsMat, testLabelsMat);
                    cout << "\nDESCRIPTOR SIZE : " << trainMat.cols << endl;
                    cout << "\nWARNING: ALL MATRIX SIZES ARE GIVEN IN A [ COLUMNS X ROWS ] FORMAT:\n" << endl;
                    cout << "TRAINING MAT SIZE: " << trainMat.size() << "\n";
                    cout << "TESTING  MAT SIZE: " << testMat.size() << "\n\n";
                    cout << "TRAIN LABELS MAT SIZE: " << trainLabelsMat.size() << "\n";
                    cout << "TEST LABELS MAT SIZE: " << testLabelsMat.size() << "\n\n";
                }
                break;

            case '3':
                cout << "------------------------------------------------------" << endl;
                nFeatures = trainMat.cols;
                nClasses = 10;
                cout << "\n\nTHE NUMBER OF DIFFERENT CLASSES IS " << nClasses << "\n\n";

                syANN_MLP_CreateBasic(ann,nFeatures,nClasses);

                // Filename for saving/loading trained models
                filename_ANNmodel = (char*)"ANNdigitsClassifierModel.yml";

                // Train and save the model
                syANN_MLP_TrainAndSave(ann,trainMat,trainLabelsMat,nClasses,filename_ANNmodel);
                break;

            case '4':
                cout << "------------------------------------------------------" << endl;
                cout << "\n\nLOADING A TRAINED MODEL FROM FILE.\n\n";
                // Now, we can load the saved model
                filename_ANNmodel = (char*)"ANNdigitsClassifierModel.yml";
                annTRAINED = cv::ml::ANN_MLP::load(filename_ANNmodel);
                break;

            case '5':
                // and perform the test
                syANN_MLP_Test(annTRAINED, testMat, testLabelsMat, nClasses);
                break;

            case '6':
                cout << "------------------------------------------------------" << endl;
                cout << "\n\nPERFORMING A SINGLE-IMAGE TEST\n\n";

                filename = "../images/digit_recognition/testB.jpg";
                label = syANN_MLP_Test_Single(filename,annTRAINED);
                cout << "\nPREDICTED CLASS FOR \"" << filename << "\" IS: " << label << endl;

                file2 = "../images/digit_recognition/testD.jpg";
                cout << "\nPREDICTED CLASS FOR \"" << file2 << "\" IS: " << syANN_MLP_Test_Single(file2, annTRAINED) << endl;

                file3 =  "../images/digit_recognition/download.png";
                testImage = imread(file3, IMREAD_GRAYSCALE);
                cout << "\nPREDICTED CLASS FOR \"" << file3 << "\" IS: " << syANN_MLP_Test_Single(file3, annTRAINED) << endl;
                break;

            default:
                return 0;
        }
    } while (1);

   //// PART A: LOAD THE SAMPLES
   //// The goal is split the mosaic and conform the vectors trainImages, testImages, trainLabels, testLabels

   // string pathName = "../images/digit_recognition/digits.png";
   // vector <Mat> trainImages;
   // vector <Mat> testImages;
   // vector <int> trainLabels;
   // vector <int> testLabels;
   // loadImagesAndLabelsForTrainAndTest (pathName, trainImages, testImages, trainLabels, testLabels);

   // cout << "Train labels in an int array of size: " << trainLabels.size () << "\n";
   // cout << "Test labels in an int array of size: " << testLabels.size () << "\n\n";

   //// PART B: FEATURE EXTRACTION
   //// The goal is to conform four matrices     [ columns   x    rows       ]:
   //// trainMat:       float-type image of size [ nFeatures x nTrainSamples ]
   //// testMat:        float-type image of size [ nFeatures x nTestSamples  ]
   //// trainLabelsMat: uchar-type image of size [     1     x nTrainSamples ]
   //// testLabelsMat:  uchar-type image of size [     1     x nTestSamples  ]

   // Mat trainMat, testMat, trainLabelsMat, testLabelsMat;
   // syFeatureMatricesForTestAndTrain (trainImages, testImages, trainLabels, testLabels, trainMat, testMat, trainLabelsMat, testLabelsMat);

   // cout << "Descriptor Size : " << trainMat.cols << endl;
   // cout << "\n\nWARNING: All matrix sizes are given in a [ columns x rows ] format:\n\n";
   // cout << "Training Mat size: " << trainMat.size () << "\n";
   // cout << "Testing  Mat size: " << testMat.size () << "\n\n";
   // cout << "Train labels Mat size: " << trainLabelsMat.size () << "\n";
   // cout << "Test labels Mat size: " << testLabelsMat.size () << "\n\n\n";

   //// PART C: TRAINING A CLASSSIFIER
   //// The goal is to create/train/save an ANN specifically a Multilayer Perceptron

   // // ANN creation
   // int nFeatures = trainMat.cols;
   // int nClasses = 10;
   // cout << "\n\nThe number of different classes is " << nClasses << "\n\n";

   // Ptr <ml::ANN_MLP> ann;
   // syANN_MLP_CreateBasic (ann,nFeatures,nClasses);

   // // ANN training (the model is saved)

   // // Filename for saving/loading trained models
   // char *filename_ANNmodel = "ANNdigitsClassifierModel.yml";
   // // Train and save the model
   // syANN_MLP_TrainAndSave(ann,trainMat,trainLabelsMat,nClasses,filename_ANNmodel);

   // // PART D: THE TEST (USING A PRE-TRAINED MODEL)
   // cout << "\n\nLoading a trained model from file.\n\n";
   // // Now, we can load the saved model
   // Ptr <ANN_MLP> annTRAINED = cv::ml::ANN_MLP::load (filename_ANNmodel);
   // // and perform the test
   // syANN_MLP_Test (annTRAINED, testMat, testLabelsMat, nClasses);

   // // PART E: SINGLE-IMAGE TEST
   // // classifying the featues of an image under test
   // cout << "\n\nPerforming a single-image test\n\n";

   // string filename = "../images/digit_recognition/testB.jpg";
   // int label = syANN_MLP_Test_Single(filename,annTRAINED);
   // cout << "\nPredicted class for \"" << filename << "\" is: " << label << endl;

   // string file2 = "../images/digit_recognition/testD.jpg";
   // cout << "\nPredicted class for \"" << file2 << "\" is: " << syANN_MLP_Test_Single (file2, annTRAINED) << endl;

   // string file3 =  "../images/digit_recognition/download.png";
   // Mat testImage = imread (file3, IMREAD_GRAYSCALE);
   // cout << "\nPredicted class for \"" << file3 << "\" is: " << syANN_MLP_Test_Single (file3, annTRAINED) << endl;

    return 0;
}

int syANN_MLP_Test_Single (string filename, Ptr <ml::ANN_MLP> & annTRAINED)
{
    Mat imgTest = imread (filename, cv::IMREAD_GRAYSCALE);

    // Preprocessing
    resize (imgTest, imgTest, Size (20, 20), INTER_LINEAR);
    Mat preprocTest = deskew (imgTest);

    // Feature extraction
    vector <float> featureVector;
    hog.compute (preprocTest, featureVector);
    int numFeatures = featureVector.size ();

    // Vector to matrix
    Mat underTest =  Mat::zeros (1, numFeatures, CV_32FC1);

    for (int k = 0; k < numFeatures; k++)
        underTest.at <float> (0, k) = featureVector [k];

    // Prediction
    return annTRAINED -> predict (underTest, noArray ());
}

// EXTRACCION DE CARACTERISTICAS Y GUARDADO EN .CVS
void extractTextureFeatures(const Mat& image, vector<float>& features) {
    // Convertir la imagen a escala de grises
    Mat imgGray;
    cvtColor(image, imgGray, COLOR_BGR2GRAY);

    // Verificar que la imagen tenga el tamaño esperado
    if (imgGray.size() != Size(640, 256)) {
        cout << "ERROR: LA IMAGEN NO TIENE EL TAMAÑO ESPERADO DE 640 x 256." << endl;
        return;
    }

    // Calcular HOG para la imagen completa
    vector<float> hogFeatures;
    try {
        hog.compute(imgGray, hogFeatures);
    }
    catch (const cv::Exception& e) {
        cout << "ERROR: EXCEPCIÓN AL CALCULAR HOG: " << e.what() << endl;
        return;
    }

    // Insertar las características HOG en el vector de características
    features.insert(features.end(), hogFeatures.begin(), hogFeatures.end());
}

void processImagesAndSaveFeatures() {
    string samplesPath = "./images/MUESTRAS"; // Ruta a la carpeta de muestras
    vector<vector<float>> data;
    vector<string> labels;

    cout << endl;
    for (const auto& entry : fs::directory_iterator(samplesPath)) {
		cout << "PROCESANDO MUESTRA: " << entry.path().filename().string() << endl;
        if (fs::is_directory(entry)) {
            string label = entry.path().filename().string();
            for (const auto& imgEntry : fs::directory_iterator(entry.path())) {
                if (imgEntry.path().extension() == ".jpg" || imgEntry.path().extension() == ".png") {
                    Mat image = imread(imgEntry.path().string());
                    if (image.empty()) {
                        cout << "ERROR: NO SE PUDO CARGAR LA IMAGEN: " << imgEntry.path() << endl;
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

    saveToCSV("FONT_FEATURES.csv", data, labels);
    cout << "\nCARACTERISTICAS GUARDADAS EN EL ARCHIVO: FONT_FEATURES.csv" << endl;
}

void saveToCSV(const string& filename, const vector<vector<float>>& data, const vector<string>& labels) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: NO SE PUDO ABRIR EL ARCHIVO PARA ESCRIBIR: " << filename << endl;
        return;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        for (const auto& feature : data[i]) {
            file << feature << ",";
        }
        file << labels[i] << "\n";
    }

    file.close();
}

// CARGAR ARCHIVO FONT_FEATURES.CSV Y GUARDAR LOS DATOS EN LAS MATRICES
void loadFeaturesFromCSV(const string& filename, Mat& fullFeatures) {
	ifstream file(filename);
	if (!file.is_open()) {
		cout << "ERROR: NO SE PUDO ABRIR EL ARCHIVO PARA LEER: " << filename << endl;
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
				labels.push_back(feature);
			}
			else {
				features.push_back(stof(feature));
			}
		}

		data.push_back(features);
	}

	file.close();

	// Convertir los datos a una matriz de características
	fullFeatures = Mat(data.size(), data[0].size(), CV_32F);
	for (size_t i = 0; i < data.size(); ++i) {
		for (size_t j = 0; j < data[i].size(); ++j) {
			fullFeatures.at<float>(i, j) = data[i][j];
		}
	}

	cout << "\nCARACTERISTICAS CARGADAS CORRECTAMENTE DESDE EL ARCHIVO: " << filename << endl;
}

// MEZCLAR LA MATRIZ DE CARACTERISTICAS Y ETIQUETAS
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

// DIVIDIR LA MATRIZ DE CARACTERISTICAS Y ETIQUETAS EN trainMat, testMat, trainLabelsMat, testLabelsMat TENIENDO EN CUENTA nFolds
void splitData(const Mat fullFeatures, int nFolds, Mat& trainMat, Mat& testMat, Mat& trainLabelsMat, Mat& testLabelsMat) {
	// Verificar si las matrices trainMat, testMat, trainLabelsMat, testLabelsMat estan vacias, si tienen datos, liberar la memoria
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

	// Mezclar los datos
    Mat shuffled = shuffleData(fullFeatures);

	// Calcular el número de muestras por pliegue
    int nSamplesPerFold = shuffled.rows / nFolds;
    int  nSamplesLearn = nSamplesPerFold * (nFolds - 1);
    int nSamplesTest = shuffled.rows - nSamplesLearn;
    cout << "\nFOR " << nFolds << "-FOLD TEST: " << nSamplesPerFold << " SAMPLES PER FOLD; " << nSamplesLearn << " FOR LEARNING, " << nSamplesTest << " FOR TESTING.\n";

	// Dividir los datos en trainMat, testMat, trainLabelsMat, testLabelsMat
    trainMat = shuffled(Range(0, nSamplesLearn), Range(0, shuffled.cols - 1));
    testMat = shuffled(Range(nSamplesLearn, shuffled.rows), Range(0, shuffled.cols - 1));
    trainLabelsMat = shuffled(Range(0, nSamplesLearn), Range(shuffled.cols - 1, shuffled.cols));
    testLabelsMat = shuffled(Range(nSamplesLearn, shuffled.rows), Range(shuffled.cols - 1, shuffled.cols));

	// Convertir las etiquetas a enteros
    trainLabelsMat.convertTo(trainLabelsMat, CV_8UC1);
    testLabelsMat.convertTo(testLabelsMat, CV_8UC1);
}

// CONSTRUCT FEATURE MATRICES
int syFeatureMatricesForTestAndTrain (vector <Mat> & trainImages, vector <Mat> & testImages, vector <int> & trainLabels, vector <int> & testLabels, Mat & trainMat, Mat & testMat, Mat & trainLabelsMat, Mat & testLabelsMat)
{
    // Pre-process the subimages
    vector <Mat> deskewedTrainImages;
    vector <Mat> deskewedTestImages;
    CreateDeskewedTrainAndTestImageSets (deskewedTrainImages, deskewedTestImages, trainImages, testImages);

    // Extract feature vectors
    std::vector <std::vector <float>> trainHOG;
    std::vector <std::vector <float>> testHOG;
    CreateTrainAndTestHOGs (trainHOG, testHOG, deskewedTrainImages, deskewedTestImages);

    int descriptor_size = trainHOG [0].size ();

    // Shaping the feature vectors into feature matrices
    trainMat = Mat::zeros (trainHOG.size (),descriptor_size, CV_32FC1);
    testMat = Mat::zeros (testHOG.size (),descriptor_size, CV_32FC1);

    ConvertVectorToMatrix (trainHOG, testHOG, trainMat, testMat);
    trainLabelsMat = Mat::zeros (trainLabels.size (), 1, CV_8UC1);

    for (int r = 0; r < trainLabelsMat.rows; r ++)
        trainLabelsMat.at <uchar> (r, 0) = trainLabels [r];

    testLabelsMat = Mat::zeros (testLabels.size (), 1, CV_8UC1);

    for (int r = 0; r < testLabelsMat.rows; r ++)
        testLabelsMat.at <uchar> (r, 0) = testLabels [r];

    return true;
}

// Image pre-processing
void CreateDeskewedTrainAndTestImageSets (vector <Mat> & deskewedTrainImages, vector <Mat> & deskewedTestImages, vector <Mat> & trainImages, vector <Mat> & testImages)
{
    for (int i = 0; i < trainImages.size (); i ++)
    {
        Mat deskewedImg = deskew (trainImages [i]);
        deskewedTrainImages.push_back (deskewedImg);
    }

    for (int i = 0; i < testImages.size (); i ++)
    {
        Mat deskewedImg = deskew (testImages [i]);
        deskewedTestImages.push_back (deskewedImg);
    }
}

Mat deskew (Mat & img)
{
    float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;

    Moments m = moments (img);

    if (abs (m.mu02) < 1e-2)
    {
        return img.clone ();
    }

    float skew = m.mu11 / m.mu02;
    Mat warpMat = (Mat_ <float> (2, 3) << 1, skew, -0.5 * SZ * skew, 0, 1, 0);
    Mat imgOut = Mat::zeros (img.rows, img.cols, img.type ());
    warpAffine (img, imgOut, warpMat, imgOut.size (),affineFlags);

    return imgOut;
}

// Feature extraction
void CreateTrainAndTestHOGs (vector <vector <float>> & trainHOG, vector <vector <float>> & testHOG, vector <Mat> & deskewedtrainImages, vector <Mat> & deskewedtestImages)
{
    for (int y = 0; y < deskewedtrainImages.size (); y++)
    {
        vector <float> descriptors;
        hog.compute (deskewedtrainImages [y],descriptors);
        trainHOG.push_back (descriptors);
    }

    for (int y = 0; y < deskewedtestImages.size (); y ++)
    {
        vector <float> descriptors;
        hog.compute (deskewedtestImages [y], descriptors);
        testHOG.push_back (descriptors);
    }
}

void ConvertVectorToMatrix (vector <vector <float>> & trainHOG, vector <vector <float>> & testHOG, Mat & trainMat, Mat & testMat)
{
    int descriptor_size = trainHOG [0].size ();

    for (int i = 0; i < trainHOG.size (); i ++)
    {
        for (int j = 0; j < descriptor_size; j ++)
        {
            trainMat.at <float> (i, j) = trainHOG [i][j];
        }
    }

    for (int i = 0; i < testHOG.size (); i ++)
    {
        for (int j = 0; j < descriptor_size; j ++)
        {
                testMat.at <float> (i, j) = testHOG [i][j];
        }
    }
}

// TRAINING THE CLASSIFIER
int syANN_MLP_CreateBasic (Ptr <ml::ANN_MLP> & ann, int nFeatures, int nClasses)
{
    cout << "\n\n\nCreating an ANN_MLP\n\n";

    ann = ml::ANN_MLP::create ();
    Mat_ <int> layers (3, 1);
    layers( 0) = nFeatures;     // input
    layers (1) = nFeatures * 2 + 1;  // hidden
    layers (2) = nClasses;      // output, 1 pin per class.
    ann -> setLayerSizes (layers);
    ann -> setActivationFunction (ml::ANN_MLP::SIGMOID_SYM, 0, 0);
    ann -> setTermCriteria (TermCriteria (TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001));
    ann -> setTrainMethod (ml::ANN_MLP::BACKPROP, 0.0001);

    return true;
}

Ptr <ml::ANN_MLP> syANN_MLP_CreateBasic (int nFeatures, int nClasses)
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

int syANN_MLP_Train (Ptr <ml::ANN_MLP> & ann, Mat & train_data, Mat & trainLabelsMat, int nClasses)
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

int syANN_MLP_TrainAndSave(Ptr<ml::ANN_MLP> &ann, Mat &train_data, Mat &trainLabelsMat, int nClasses, char *filename_ANNmodel)
{
   // Warning: ann requires "one-hot" encoding of class labels:
   // Class labels in a float-type sparse matrix of Samples x Classes
   // with an '1' in the corresponding correct-label column

    Mat train_classes = Mat::zeros (train_data.rows, nClasses, CV_32FC1);

    for (int i = 0; i < train_classes.rows; i ++)
    {
        train_classes.at <float> (i, trainLabelsMat.at <uchar> (i)) = 1.0;
    }

    cout << "\nTrain data size: " << train_data.size () << "\nTrain classes size: " << train_classes.size () << "\n\n";
    cout << "Training the ANN... (please wait)\n\n\n";
    ann -> train (train_data, ml::ROW_SAMPLE, train_classes);

    ann -> save (filename_ANNmodel);
    cout << "\n\nTrained model saved as " << filename_ANNmodel <<  "\n\n";

    return 0;
}

int syANN_MLP_Test (Ptr <ml::ANN_MLP> & ann, Mat & test_data, Mat & testLabelsMat, int nClasses)
{
    cout << "ANN prediction test\n\n";

    Mat confusion (nClasses, nClasses, CV_32S, Scalar (0));
    cout << "Confusion matrix size: "<< confusion.size () << "\n";

    // Tests samples in test_data Mat
    for (int i = 0; i < test_data.rows; i ++)
    {
        int pred  = ann -> predict (test_data.row (i), noArray ());
        int truth = testLabelsMat.at <uchar> (i);
        confusion.at <int> (truth,pred) ++;
    }

    cout << "Confusion matrix:\n" << confusion << endl;

    Mat correct = confusion.diag ();
    float accuracy = sum (correct) [0] / sum (confusion) [0];
    cout << "\nAccuracy: " << accuracy << "\n\n";
    return 0;
}

void ANN_train_test (int nclasses, const Mat & train_data, const Mat & trainLabelsMat, const Mat & test_data, const Mat & testLabelsMat, Mat & confusion)
{
    int nfeatures = train_data.cols;

    // ANN init
    cout << "\nInitializing ANN\n\n";

    Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create ();
    Mat_ <int> layers (3, 1);
    layers (0) = nfeatures;
    layers (1) = nfeatures * 2 + 1;
    layers (2) = nclasses;
    ann -> setLayerSizes (layers);
    ann -> setActivationFunction (ml::ANN_MLP::SIGMOID_SYM, 0, 0);
    ann -> setTermCriteria (TermCriteria (TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001));
    ann -> setTrainMethod (ml::ANN_MLP::BACKPROP, 0.0001);

    Mat train_classes = Mat::zeros (train_data.rows, nclasses, CV_32FC1);

    for (int i = 0; i < train_classes.rows; i ++)
    {
        train_classes.at <float> (i, trainLabelsMat.at <uchar> (i)) = 1.0;
    }

    cout << "\nTrain data size: " << train_data.size () << "\nTrain classes size: " << train_classes.size () << "\n\n";
    cout << "Training the ANN...\n\n\n";
    ann -> train (train_data, ml::ROW_SAMPLE, train_classes);

    cout << "ANN prediction test\n\n";

    // Tests samples in test_data Mat
    for (int i = 0; i < test_data.rows; i ++)
    {
        int pred  = ann -> predict (test_data.row (i), noArray ());
        int truth = testLabelsMat.at <uchar> (i);
        confusion.at <int> (truth,pred) ++;
    }

    cout << "Confusion matrix:\n" << confusion << endl;

    Mat correct = confusion.diag ();
    float accuracy = sum (correct) [0] / sum (confusion) [0];
    cout << "\nAccuracy: " << accuracy << "\n\n";
}
