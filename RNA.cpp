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
    Size (20,20), //winSize
    Size (8,8), //blocksize
    Size (4,4), //blockStride,
    Size (8,8), //cellSize,
    9,   //nbins,
    1,   //derivAper,
    -1,  //winSigma,
    cv::HOGDescriptor::HistogramNormType::L2Hys, //histogramNormType,
    0.2, //L2HysThresh,
    0,   //gammal correction,
    64,  //nlevels=64
    1
);

// For loading the samples
void processImage(const std::string& imagePath);
void loadImagesAndLabelsForTrainAndTest (string & pathName, vector <Mat> & trainImages, vector <Mat> & testImages, vector <int> & trainLabels, vector <int> & testLabels);

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

//Extract CSV
void extractTextureFeatures(const Mat &image, vector<float> &features);
void saveToCSV(const string &filename, const vector<vector<float>> &data, const vector<string> &labels);
void processImagesAndSaveFeatures();

// Global variable
// Corresponds to the size of sub-images
int SZ = 20;

int main(void)
{
    // CASE 1 VARIABLES:
    string pathName = " ";
    vector <Mat> trainImages;
    vector <Mat> testImages;
    vector <int> trainLabels;
    vector <int> testLabels;

    // CASE 2 VARIABLES:
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

                processImagesAndSaveFeatures();
                /*
                cout << "------------------------------------------------------" << endl;
                pathName = "digits.png";
                loadImagesAndLabelsForTrainAndTest(pathName, trainImages, testImages, trainLabels, testLabels);

                if (!trainImages.empty() && !testImages.empty() && !trainLabels.empty() && !testLabels.empty())
                {
                    cout << "TRAIN LABELS IN AN INT ARRAY OF SIZE: " << trainLabels.size() << "\n";
                    cout << "TEST LABELS IN AN INT ARRAY OF SIZE: " << testLabels.size() << "\n\n";
                }
                */

                system("pause");
                break;

            case '2':
                cout << "------------------------------------------------------" << endl;
                syFeatureMatricesForTestAndTrain(trainImages, testImages, trainLabels, testLabels, trainMat, testMat, trainLabelsMat, testLabelsMat);

                cout << "\nDESCRIPTOR SIZE : " << trainMat.cols << endl;
                cout << "\nWARNING: ALL MATRIX SIZES ARE GIVEN IN A [ COLUMNS X ROWS ] FORMAT:\n" << endl;
                cout << "TRAINING MAT SIZE: " << trainMat.size() << "\n";
                cout << "TESTING  MAT SIZE: " << testMat.size() << "\n\n";
                cout << "TRAIN LABELS MAT SIZE: " << trainLabelsMat.size() << "\n";
                cout << "TEST LABELS MAT SIZE: " << testLabelsMat.size() << "\n\n";
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

// LOAD SAMPLES
void processImage(const std::string& imagePath) {
    // Cargar imagen
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "No se pudo cargar la imagen" << std::endl;
        return;
    }

    // Convertir a escala de grises
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Aplicar umbral para convertir a binaria
    cv::Mat binary;
    cv::threshold(gray, binary, 150, 255, cv::THRESH_BINARY_INV);

    // Aplicar morfología para juntar caracteres cercanos
    cv::Mat morph;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 5));
    cv::morphologyEx(binary, morph, cv::MORPH_CLOSE, kernel);

    // Encontrar contornos
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Crear una copia de la imagen original para dibujar los rectángulos
    cv::Mat result = image.clone();
    std::vector<cv::Mat> textBlocks;
    cv::Mat textBlock;

    // Variables para el rectángulo delimitador
    int minX = image.cols, minY = image.rows, maxX = 0, maxY = 0;

    // Iterar sobre cada contorno para obtener los rectángulos de texto
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);

        // Dibujar rectángulo en la imagen (opcional)
        cv::rectangle(result, rect, cv::Scalar(0, 255, 0), 2);

        // Actualizar los límites del rectángulo delimitador
        minX = std::min(minX, rect.x);
        minY = std::min(minY, rect.y);
        maxX = std::max(maxX, rect.x + rect.width);
        maxY = std::max(maxY, rect.y + rect.height);

        // Recortar la región de texto
        textBlock = image(rect);
        if (textBlock.cols < image.cols) {
            cv::copyMakeBorder(textBlock, textBlock, 0, 0, 0, image.cols - textBlock.cols, cv::BORDER_CONSTANT, cv::Scalar(255));
        }
        textBlocks.push_back(textBlock);
    }

    // Recortar la imagen result al tamaño de los contornos verdes
    cv::Rect boundingRect(minX, minY, maxX - minX, maxY - minY);
    cv::Mat croppedResult = result(boundingRect);

    // Eliminar los contornos verdes de la imagen recortada
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        cv::Mat roi = croppedResult(rect - cv::Point(minX, minY));
        image(rect).copyTo(roi);
    }

    // Crear una máscara para identificar los píxeles verdes
    cv::Mat greenMask;
    cv::inRange(croppedResult, cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), greenMask);

    // Eliminar los píxeles verdes de la imagen recortada
    cv::Mat finalResult = croppedResult.clone();
    finalResult.setTo(cv::Scalar(255, 255, 255), greenMask);

    // Convertir la imagen final a negativo
    cv::Mat negativeResult;
    cv::bitwise_not(finalResult, negativeResult);

    // Guardar o mostrar la imagen recortada sin contornos verdes y en negativo
    cv::imshow("Negative Result", negativeResult);
    cv::imwrite("negative_result.png", negativeResult);

    cv::waitKey(0);
}

void loadImagesAndLabelsForTrainAndTest(string &pathName, vector <Mat> &trainImages, vector <Mat> &testImages, vector <int> &trainLabels, vector <int> &testLabels)
{
    Mat img = imread(pathName, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "\nERROR: IMAGEN NO CARGADA. VERIFICA LA RUTA DE ARCHIVO." << endl;
        cout << "PRESIONE CUALQUIER BOTON PARA CONTINUAR" << endl;
        cin.get();

        return;
    }
    else {
        int ImgCount = 0;
        for(int i = 0; i < img.rows; i = i + SZ)
        {
            for(int j = 0; j < img.cols; j = j + SZ)
            {
                Mat digitImg = (img.colRange(j, j + SZ).rowRange(i, i + SZ)).clone();

                if (j < int(0.9 * img.cols))
                {
                    trainImages.push_back(digitImg);
                }
                else
                {
                    testImages.push_back(digitImg);
                }
                ImgCount++;
            }
        }

        cout << "\nIMAGE COUNT: " << ImgCount << endl;
        float digitClassNumber = 0;

        for (int z = 0; z <int(0.9 * ImgCount); z++)
        {
            if (z % 450 == 0 && z != 0)
            {
                digitClassNumber = digitClassNumber + 1;
            }

            trainLabels.push_back(digitClassNumber);
        }

        digitClassNumber = 0;

        for (int z = 0; z <int(0.1 * ImgCount); z++)
        {
            if (z % 50 == 0 && z != 0)
            {
                digitClassNumber = digitClassNumber + 1;
            }

            testLabels.push_back(digitClassNumber);
        }
    }
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

void extractTextureFeatures(const Mat &image, vector<float> &features) {
    Mat imgGray;
    cvtColor(image, imgGray, COLOR_BGR2GRAY);

    // Calcula histogramas de gradientes X e Y
    Mat gradX, gradY;
    Sobel(imgGray, gradX, CV_32F, 1, 0, 3);  // Gradiente en X
    Sobel(imgGray, gradY, CV_32F, 0, 1, 3);  // Gradiente en Y

    Scalar meanX, stddevX, meanY, stddevY;

    // Asegura que los gradientes no estén vacíos antes de calcular estadísticas
    if (!gradX.empty()) {
        meanStdDev(gradX, meanX, stddevX);
    } else {
        meanX = Scalar(0);
        stddevX = Scalar(0);
    }

    if (!gradY.empty()) {
        meanStdDev(gradY, meanY, stddevY);
    } else {
        meanY = Scalar(0);
        stddevY = Scalar(0);
    }

    // Añade las medias y desviaciones de los gradientes como características
    features.push_back(static_cast<float>(meanX[0]));
    features.push_back(static_cast<float>(stddevX[0]));
    features.push_back(static_cast<float>(meanY[0]));
    features.push_back(static_cast<float>(stddevY[0]));

    // Calcula momentos Hu como características de forma
    Moments moments = cv::moments(imgGray, true);
    double hu[7];
    HuMoments(moments, hu);
    for (int i = 0; i < 7; i++) {
        features.push_back(static_cast<float>(hu[i]));
    }
}

void saveToCSV(const string &filename, const vector<vector<float>> &data, const vector<string> &labels) {
    ofstream file(filename);

    // Escribe el encabezado del archivo CSV
    file << "Label,MeanGradX,StdDevGradX,MeanGradY,StdDevGradY";
    for (int i = 1; i <= 7; i++) file << ",HuMoment" << i;
    file << "\n";

    // Escribe los datos de las características
    for (size_t i = 0; i < data.size(); i++) {
        file << labels[i] << ",";
        for (float f : data[i]) file << f << ",";
        file << "\n";
    }
    file.close();
}

void processImagesAndSaveFeatures() {
    string basePath = "./images/TRAINING";
    vector<vector<float>> allFeatures;
    vector<string> labels;

    for (const auto &entry : fs::directory_iterator(basePath)) {
        if (fs::is_directory(entry)) {
            string fontName = entry.path().filename().string();  // Nombre de la fuente (subcarpeta)
            vector<float> accumulatedFeatures(11, 0.0f);  // Vector para acumular características
            int imgCount = 0;

            for (const auto &imgFile : fs::directory_iterator(entry.path())) {
                Mat img = imread(imgFile.path().string());
                if (!img.empty()) {
                    vector<float> features;
                    extractTextureFeatures(img, features);

                    // Acumula las características
                    for (size_t i = 0; i < features.size(); i++) {
                        accumulatedFeatures[i] += features[i];
                    }
                    imgCount++;
                }
            }

            // Calcula el promedio de características por subcarpeta
            for (float &value : accumulatedFeatures) {
                value /= imgCount;
            }

            allFeatures.push_back(accumulatedFeatures);
            labels.push_back(fontName);
        }
    }

    // Guarda las características en un archivo CSV
    saveToCSV("font_features.csv", allFeatures, labels);

    cout << "Extraccion y guardado de caracteristicas completado." << endl;
}
