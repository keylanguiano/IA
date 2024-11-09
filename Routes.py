import os

# Define la carpeta principal donde están todas las subcarpetas de fuentes
directorio_principal = "Images/TRAINING"

# Define el nombre del archivo donde se guardarán las rutas
archivo_rutas = "Rutas.txt"

# Abre el archivo en modo de escritura
with open(archivo_rutas, "w") as archivo:
    # Recorre todas las carpetas y archivos dentro del directorio principal
    for carpeta_raiz, carpetas, archivos in os.walk(directorio_principal):
        for archivo_nombre in archivos:
            # Obtén la ruta relativa del archivo
            ruta_archivo = os.path.join(carpeta_raiz, archivo_nombre)
            # Escribe la ruta en el archivo de texto
            archivo.write(ruta_archivo + "\n")

print(f"Las rutas se han guardado en {archivo_rutas}")
