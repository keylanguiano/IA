#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = std::experimental::filesystem;

void loadImagesAndLabelsForTrainAndTest(const vector<string> &imagePaths, vector<cv::Mat> &trainImages, vector<cv::Mat> &testImages, vector<int> &trainLabels, vector<int> &testLabels, int SZ = 28);

int main() {
    ifstream archivo("Rutas.txt");
    string rutaImagen;
    vector<string> imagePaths;

    if (!archivo.is_open()) {
        cerr << "No se pudo abrir el archivo Rutas.txt.\n";
        return 1;
    }

    // Leer todas las rutas de imágenes del archivo y guardarlas en un vector
    while (getline(archivo, rutaImagen)) {
        if (!rutaImagen.empty()) {
            imagePaths.push_back(rutaImagen);
        }
    }
    archivo.close();

    // Vectores para almacenar imágenes y etiquetas para entrenamiento y prueba
    vector<cv::Mat> trainImages, testImages;
    vector<int> trainLabels, testLabels;

    // Cargar e invertir imágenes y etiquetas
    loadImagesAndLabelsForTrainAndTest(imagePaths, trainImages, testImages, trainLabels, testLabels);

    return 0;
}

void loadImagesAndLabelsForTrainAndTest(const vector<string> &imagePaths, vector<cv::Mat> &trainImages, vector<cv::Mat> &testImages, vector<int> &trainLabels, vector<int> &testLabels, int SZ) {
    int ImgCount = 0;
    float digitClassNumber = 0;

    // Procesar cada imagen en imagePaths
    for (const auto& pathName : imagePaths) {
        cv::Mat img = cv::imread(pathName, cv::IMREAD_GRAYSCALE);

        if (img.empty()) {
            cerr << "No se pudo cargar la imagen: " << pathName << "\n";
            continue;
        }

        ImgCount++;

        // Recorrer la imagen en segmentos de tamaño SZ x SZ
        for (int i = 0; i < img.rows; i += SZ) {
            for (int j = 0; j < img.cols; j += SZ) {
                // Verificar que el ROI esté dentro de los límites de la imagen
                if (i + SZ <= img.rows && j + SZ <= img.cols) {
                    cv::Mat digitImg = img(cv::Rect(j, i, SZ, SZ)).clone();

                    // Invertir colores de la imagen para entrenamiento/prueba
                    cv::Mat invertedDigitImg;
                    cv::bitwise_not(digitImg, invertedDigitImg);

                    // Asignar la imagen a entrenamiento o prueba
                    if (j < static_cast<int>(0.9 * img.cols)) {
                        trainImages.push_back(invertedDigitImg);
                    } else {
                        testImages.push_back(invertedDigitImg);
                    }

                }
            }
        }

        // Generar etiquetas
        int trainSize = static_cast<int>(0.9 * ImgCount);
        int testSize = ImgCount - trainSize;

        for (int z = 0; z < trainSize; ++z) {
            if (z % 450 == 0 && z != 0) {
                digitClassNumber += 1;
            }
            trainLabels.push_back(digitClassNumber);
        }

        digitClassNumber = 0;

        for (int z = 0; z < testSize; ++z) {
            if (z % 50 == 0 && z != 0) {
                digitClassNumber += 1;
            }
            testLabels.push_back(digitClassNumber);
        }
    }

    cout << "Total de imagenes procesadas: " << ImgCount << endl;
}
