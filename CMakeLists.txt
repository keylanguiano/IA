cmake_minimum_required(VERSION 3.10)
project(RNA_2.3)

# Requiere C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontrar OpenCV
find_package(OpenCV REQUIRED)

# Añadir los encabezados de OpenCV a los directorios de inclusión
include_directories(${OpenCV_INCLUDE_DIRS})

# Crear el ejecutable
add_executable(RNA_2.3 ./RNA_2.3.cpp)

# Vincular las bibliotecas de OpenCV a tu ejecutable
target_link_libraries(RNA_2.3 ${OpenCV_LIBS})
