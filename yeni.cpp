#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // Directorio que contiene las imágenes del abecedario
    std::string directorio = "../imagenes_abecedario";  // Cambia esta ruta según sea necesario
    //std::string carpeta_destino = "./letras";            // Carpeta donde se guardarán las letras recortadas

    // Verifica si el directorio existe
    if (!fs::exists(directorio) || !fs::is_directory(directorio)) {
        std::cerr << "Error: El directorio especificado no existe o no es un directorio válido." << std::endl;
        return -1;
    }

    // Crea la carpeta de destino si no existe
/*    if (!fs::exists(carpeta_destino)) {
        fs::create_directory(carpeta_destino);
    }*/

    // Contador para el número total de letras recortadas
    int total_letras = 0;

    // Itera sobre cada archivo en el directorio
    for (const auto& entrada : fs::directory_iterator(directorio)) {
        if (entrada.is_regular_file()) {
            // Carga la imagen
            cv::Mat imagen = cv::imread(entrada.path().string(), cv::IMREAD_GRAYSCALE);
            if (imagen.empty()) {
                std::cerr << "Error: No se pudo cargar la imagen en " << entrada.path().string() << std::endl;
                continue;
            }

            // Preprocesar la imagen: Umbralización para convertirla a blanco y negro
            cv::Mat binarizada;
            cv::threshold(imagen, binarizada, 128, 255, cv::THRESH_BINARY_INV); // Invertir para que las letras sean blancas

            // Encontrar contornos en la imagen
            std::vector<std::vector<cv::Point>> contornos;
            cv::findContours(binarizada, contornos, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Variable para numerar las letras recortadas
            int contador = 0;

            // Itera sobre cada contorno encontrado (cada letra)
            for (size_t i = 0; i < contornos.size(); i++) {
                // Obtener la caja delimitadora para cada contorno
                cv::Rect bounding_box = cv::boundingRect(contornos[i]);

                // Filtrar contornos demasiado pequeños que no corresponden a una letra
                if (bounding_box.width < 10 || bounding_box.height < 10) {
                    continue;
                }

                // Recortar la letra de la imagen original usando la caja delimitadora
                cv::Mat letra = imagen(bounding_box);

                // Guardar la letra recortada en la carpeta de destino
                //std::string nombre_imagen = carpeta_destino + "/letra_" + std::to_string(total_letras++) + ".png";
                //cv::imwrite(nombre_imagen, letra);

                //std::cout << "Guardada: " << nombre_imagen << std::endl;
            }
        }
    }

    // Imprimir el total de letras recortadas
    //std::cout << "Total de letras recortadas: " << total_letras << std::endl;

    //std::cout << "Proceso finalizado. Las letras recortadas se guardaron en la carpeta 'letras'." << std::endl;
    return 0;
}