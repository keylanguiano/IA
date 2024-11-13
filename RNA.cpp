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

// CASE 1: EXTRACCION DE CARACTERISTICAS Y GUARDADO EN .CVS
void processImagesAndSaveFeatures();
void extractTextureFeatures(const Mat& image, vector<float>& features);
bool saveToCSV(const string& filename, const vector<vector<float>>& data, const vector<string>& labels);

// CASE 2: CARGAR ARCHIVO FONT_FEATURES.CSV Y GUARDAR LOS DATOS EN LAS MATRICES
void loadFeaturesFromCSV(const string& filename, Mat& fullFeatures, vector<string>& originalLabels);
Mat shuffleData(Mat fullFeatures);
void splitData(const Mat fullFeatures, int nFolds, Mat& trainMat, Mat& testMat, Mat& trainLabelsMat, Mat& testLabelsMat);

// CASE 3: ENTRENAR LA ANN MLP CON LAS MATRICES CARGADAS EN EL SISTEMA Y GUARDAR EL MODELO
Ptr <ml::ANN_MLP> ANN_MLP_CreateBasic (int nFeatures, int nClasses);
int ANN_MLP_CreateBasic(Ptr <ml::ANN_MLP> & ann,int nFeatures, int nClasses);
int ANN_MLP_Train(Ptr <ml::ANN_MLP> & ann, Mat & train_data, Mat & trainLabelsMat, int nClasses);
int ANN_MLP_TrainAndSave( Ptr <ml::ANN_MLP> & ann, Mat & train_data, Mat & trainLabelsMat, int nClasses, char * filename_ANNmodel);

// CASE 5: PROBAR EL MODELO CON LA MATRIZ DE ENTRENAMIENTO Y GENERAR UNA MATRIZ DE CONFUSION
int ANN_MLP_Test(Ptr <ml::ANN_MLP> & ann, Mat & testMat, Mat & testLabelsMat, int nClasses);

// CASE 6: PROBAR EL MODELO CON UNA IMAGEN
int ANN_MLP_Test_Single(string filename, Ptr <ml::ANN_MLP> & annTRAINED);

// Global variable
// Corresponds to the size of sub-images
int SZ = 20;

int main(void)
{
    // CASE 1 VARIABLES:
    char overwrite;

    // CASE 2 VARIABLES:
	int nFolds = 5;
    vector<string> originalLabels;
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
        cout << "\t 4. CARGAR UN MODELO ENTRENADO" << endl;
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
					cout << "DESEA SOBREESCRIBIRLO? (S/N): ";
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
					cout << "ERROR: NO SE HAN CARGADO LAS MATRICES DE ENTRENAMIENTO Y PRUEBA." << endl;
					cout << "POR FAVOR EJECUTE LA OPCION 2 PARA CARGARLAS." << endl;
					cout << "O EJECUTE LA OPCION 1 PARA CREARLAS." << endl << endl;
					system("pause");
					break;
				}

				if (fs::exists("ANNfontTypesClassifierModel.yml")) {
					cout << "EL ARCHIVO ANNfontTypesClassifierModel.yml YA EXISTE EN LA CARPETA RAIZ." << endl;
					cout << "DESEA SOBREESCRIBIRLO? (S/N): ";
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

				cout << "MODELO ENTRENADO GUARDADO COMO: " << filename_ANNmodel << "\n";
				system("pause");
                break;

            case '4':
                cout << "------------------------------------------------------" << endl;

				if (!fs::exists("ANNfontTypesClassifierModel.yml")) {
					cout << "EL ARCHIVO ANNfontTypesClassifierModel.yml NO EXISTE EN LA CARPETA RAIZ." << endl;
                    cout << "POR FAVOR EJECUTE LA OPCION 3 PARA CREARLO." << endl <<  endl;
					system("pause");
					break;
				}

                // VERIFICAR SI HAY UN MODELO CARGADO
				if (!annTRAINED.empty()) {
					cout << "HAY UN MODELO CARGADO EN EL SISTEMA." << endl;
					cout << "DESEA CARGAR UN NUEVO MODELO? (S/N): ";
					cin >> overwrite;

					if (overwrite == 'N' || overwrite == 'n')
						break;

                    cout << endl;
				}

				// LIBERA LA MEMORIA DEL MODELO CARGADO
				annTRAINED.release();
                cout << "LOADING A TRAINED MODEL FROM FILE.\n\n";
                // Now, we can load the saved model
                filename_ANNmodel = (char*)"ANNfontTypesClassifierModel.yml";
                annTRAINED = cv::ml::ANN_MLP::load(filename_ANNmodel);

				annTRAINED.empty() ? cout << "ERROR: NO SE PUDO CARGAR EL MODELO." << "\n\n" : cout << "MODELO CARGADO CORRECTAMENTE DESDE EL ARCHIVO: " << filename_ANNmodel << "\n\n";
				system("pause");
                break;

            case '5':
                cout << "------------------------------------------------------" << endl;
                
                if (trainMat.empty() || testMat.empty() || trainLabelsMat.empty() || testLabelsMat.empty()) {
                    cout << "ERROR: NO SE HAN CARGADO LAS MATRICES DE PRUEBA." << endl;
                    cout << "POR FAVOR EJECUTE LA OPCION 2 PARA CARGARLAS." << endl;
                    cout << "O EJECUTE LA OPCION 1 PARA CREARLAS." << endl << endl;
                    system("pause");
                    break;
                }

				if (annTRAINED.empty()) {
					cout << "ERROR: NO SE HA CARGADO UN MODELO ENTRENADO." << endl;
					cout << "POR FAVOR EJECUTE LA OPCION 4 PARA CARGARLO." << endl;
					cout << "O EJECUTE LA OPCION 3 PARA ENTRENAR UN NUEVO MODELO." << endl << endl;
					system("pause");
					break;
				}

                nClasses = static_cast<int>(originalLabels.size());
                ANN_MLP_Test(annTRAINED, testMat, testLabelsMat, nClasses);
                system("pause");
                break;

            case '6':
                cout << "------------------------------------------------------" << endl;
                cout << "\n\nPERFORMING A SINGLE-IMAGE TEST\n\n";

                filename = "../images/digit_recognition/testB.jpg";
                label = ANN_MLP_Test_Single(filename,annTRAINED);
                cout << "\nPREDICTED CLASS FOR \"" << filename << "\" IS: " << label << endl;

                file2 = "../images/digit_recognition/testD.jpg";
                cout << "\nPREDICTED CLASS FOR \"" << file2 << "\" IS: " << ANN_MLP_Test_Single(file2, annTRAINED) << endl;

                file3 =  "../images/digit_recognition/download.png";
                testImage = imread(file3, IMREAD_GRAYSCALE);
                cout << "\nPREDICTED CLASS FOR \"" << file3 << "\" IS: " << ANN_MLP_Test_Single(file3, annTRAINED) << endl;
                break;

            default:
                break;
        }
    } while (1);

    return 0;
}

// CASE 1
// EXTRACCION DE CARACTERISTICAS Y GUARDADO EN .CVS
void extractTextureFeatures(const Mat& image, vector<float>& features) {
    // Convertir la imagen a escala de grises
    Mat imgGray;
    cvtColor(image, imgGray, COLOR_BGR2GRAY);

    // Verificar que la imagen tenga el tama�o esperado
    if (imgGray.size() != Size(640, 256)) {
        cout << "ERROR: LA IMAGEN NO TIENE EL TAMA�O ESPERADO DE 640 x 256." << endl;
        return;
    }

    // Calcular HOG para la imagen completa
    vector<float> hogFeatures;
    try {
        hog.compute(imgGray, hogFeatures);
    }
    catch (const cv::Exception& e) {
        cout << "ERROR: EXCEPCI�N AL CALCULAR HOG: " << e.what() << endl;
        return;
    }

    // Insertar las caracter�sticas HOG en el vector de caracter�sticas
    features.insert(features.end(), hogFeatures.begin(), hogFeatures.end());
}

// PROCESAR LAS MUESTRAS Y GUARDAR LAS CARACTERISTICAS
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

    bool success = saveToCSV("FONT_FEATURES.csv", data, labels);

    if (success)
        cout << "\nCARACTERISTICAS GUARDADAS EN EL ARCHIVO: FONT_FEATURES.csv" << endl;
}

// GUARDAR LAS CARATERISTICAS EN UN ARCHIVO .CSV
bool saveToCSV(const string& filename, const vector<vector<float>>& data, const vector<string>& labels) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "\nERROR: NO SE PUDO ABRIR EL ARCHIVO PARA ESCRIBIR: " << filename << endl;
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
// CARGAR ARCHIVO FONT_FEATURES.CSV Y GUARDAR LOS DATOS EN LAS MATRICES
void loadFeaturesFromCSV(const string& filename, Mat& fullFeatures, vector<string>& originalLabels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "\nERROR: NO SE PUDO ABRIR EL ARCHIVO PARA LEER: " << filename << endl;
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

    // Convertir los datos a una matriz de caracteristicas
    fullFeatures = Mat(static_cast<int>(data.size()), static_cast<int>(data[0].size()), CV_32F);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            fullFeatures.at<float>(static_cast<int>(i), static_cast<int>(j)) = data[i][j];
        }
    }

    cout << "CARACTERISTICAS CARGADAS CORRECTAMENTE DESDE EL ARCHIVO: " << filename << endl;
}

// FUNCION PARA GUARDAR CUALQUIER MATRIZ EN UN ARCHIVO .CVS
void saveMatToCSV(const string& filename, const Mat& matrix) {
	ofstream file(filename);
	if (!file.is_open()) {
		cout << "ERROR: NO SE PUDO ABRIR EL ARCHIVO PARA ESCRIBIR: " << filename << endl;
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

	// Calcular el n�mero de muestras por pliegue
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
// CREA UNA FUNCION PARA PROBAR EL MODELO CON LA MATRIZ DE PRUEBA Y GENERAR UNA MATRIZ DE CONFUSION
int ANN_MLP_Test(Ptr <ml::ANN_MLP>& ann, Mat& test_data, Mat& testLabelsMat, int nClasses)
{
    cout << "ANN prediction test\n\n";

    Mat confusion(nClasses, nClasses, CV_32S, Scalar(0));
    cout << "Confusion matrix size: " << confusion.size() << "\n\n";

    // Tests samples in test_data Mat
    for (int i = 0; i < test_data.rows; i++)
    {
        int pred = ann->predict(test_data.row(i), noArray());
        int truth = testLabelsMat.at <uchar>(i);
        confusion.at <int>(truth, pred)++;
    }

    cout << "Confusion matrix:\n" << confusion << endl;

    Mat correct = confusion.diag();
    float accuracy = sum(correct)[0] / sum(confusion)[0];
    cout << "\nAccuracy: " << accuracy << "\n\n";
    return 0;
}

// CASE 6: PROBAR EL MODELO CON UNA IMAGEN
int ANN_MLP_Test_Single(string filename, Ptr <ml::ANN_MLP>& annTRAINED)
{
    Mat imgTest = imread(filename, cv::IMREAD_GRAYSCALE);

    // Preprocessing
    resize(imgTest, imgTest, Size(20, 20), INTER_LINEAR);
    Mat preprocTest;

    // Feature extraction
    vector <float> featureVector;
    hog.compute(preprocTest, featureVector);
    int numFeatures = featureVector.size();

    // Vector to matrix
    Mat underTest = Mat::zeros(1, numFeatures, CV_32FC1);

    for (int k = 0; k < numFeatures; k++)
        underTest.at <float>(0, k) = featureVector[k];

    // Prediction
    return annTRAINED->predict(underTest, noArray());
}