#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>

// Configuración de Ultra-Alta Resolución (8K UHD: 7680 x 4320)
const int WIDTH = 7680;
const int HEIGHT = 4320;
const std::string OUTPUT_FILE = "mandelbrot_8k_blurred.ppm";

// Estructura para almacenar un píxel en formato RGB
struct Pixel {
    unsigned char r, g, b;
};

// --- TAREA A: Generación del Conjunto de Mandelbrot ---
void generarMandelbrot(std::vector<Pixel>& imagen) {
    const int MAX_ITER = 500;
    
    // Límites del plano complejo para encuadrar el fractal
    const double minX = -2.0, maxX = 0.5;
    const double minY = -1.25, maxY = 1.25;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            // Mapear los píxeles de la pantalla al plano complejo (c = cr + i*ci)
            double cr = minX + (x * (maxX - minX) / WIDTH);
            double ci = minY + (y * (maxY - minY) / HEIGHT);

            double zr = 0.0, zi = 0.0;
            int iter = 0;

            // Algoritmo de escape del tiempo
            while (zr * zr + zi * zi <= 4.0 && iter < MAX_ITER) {
                double temp = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = temp;
                ++iter;
            }

            // Coloreado básico basado en las iteraciones
            int idx = y * WIDTH + x;
            if (iter == MAX_ITER) {
                imagen[idx] = {0, 0, 0}; // El "cuerpo" del Mandelbrot es negro
            } else {
                // Paleta de colores psicodélica/gradual
                imagen[idx].r = static_cast<unsigned char>((iter * 7) % 256);
                imagen[idx].g = static_cast<unsigned char>((iter * 13) % 256);
                imagen[idx].b = static_cast<unsigned char>((iter * 23) % 256);
            }
        }
    }
}

// --- TAREA B: Aplicación de Filtro de Convolución 2D (Desenfoque Gaussiano pesado) ---
void aplicarFiltroGaussiano(const std::vector<Pixel>& origen, std::vector<Pixel>& destino) {
    // Matriz de convolución (Kernel Gaussiano 5x5)
    // Al ser un filtro "pesado" en 8K, un kernel 5x5 secuencial ya exige bastantes operaciones.
    const int K_SIZE = 5;
    const double kernel[5][5] = {
        {1/273.0,  4/273.0,  7/273.0,  4/273.0, 1/273.0},
        {4/273.0, 16/273.0, 26/273.0, 16/273.0, 4/273.0},
        {7/273.0, 26/273.0, 41/273.0, 26/273.0, 7/273.0},
        {4/273.0, 16/273.0, 26/273.0, 16/273.0, 4/273.0},
        {1/273.0,  4/273.0,  7/273.0,  4/273.0, 1/273.0}
    };
    
    int offset = K_SIZE / 2;

    // Convolución píxel por píxel
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            
            // Caso de bordes: mantener el píxel original para simplificar
            if (x < offset || x >= WIDTH - offset || y < offset || y >= HEIGHT - offset) {
                destino[y * WIDTH + x] = origen[y * WIDTH + x];
                continue;
            }

            double sumR = 0.0, sumG = 0.0, sumB = 0.0;

            // Operación de vecindad (Kernel)
            for (int ky = 0; ky < K_SIZE; ++ky) {
                for (int kx = 0; kx < K_SIZE; ++kx) {
                    int pixelX = x + kx - offset;
                    int pixelY = y + ky - offset;
                    
                    const Pixel& p = origen[pixelY * WIDTH + pixelX];
                    double peso = kernel[ky][kx];

                    sumR += p.r * peso;
                    sumG += p.g * peso;
                    sumB += p.b * peso;
                }
            }

            // Guardar resultado con cast seguro
            int destIdx = y * WIDTH + x;
            destino[destIdx].r = static_cast<unsigned char>(sumR);
            destino[destIdx].g = static_cast<unsigned char>(sumG);
            destino[destIdx].b = static_cast<unsigned char>(sumB);
        }
    }
}

// Función auxiliar para guardar en formato PPM P6 (Binario)
void guardarImagenPPM(const std::vector<Pixel>& imagen, const std::string& nombreArchivo) {
    std::ofstream archivo(nombreArchivo, std::ios::binary);
    if (!archivo) {
        std::cerr << "Error al abrir el archivo para escribir." << std::endl;
        return;
    }
    // Cabecera PPM: P6 (color binario), ancho, alto, y valor máximo de color (255)
    archivo << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    archivo.write(reinterpret_cast<const char*>(imagen.data()), imagen.size() * sizeof(Pixel));
    archivo.close();
}

int main() {
    std::cout << "Iniciando procesamiento secuencial (Resolucion 8K)..." << std::endl;
    
    // Reserva de memoria para las imágenes (7680 * 4320 píxeles cada una)
    std::vector<Pixel> imagenOriginal(WIDTH * HEIGHT);
    std::vector<Pixel> imagenFiltrada(WIDTH * HEIGHT);

    // --- Ejecución Tarea A ---
    auto startA = std::chrono::high_resolution_clock::now();
    std::cout << "Ejecutando Tarea A: Generando Mandelbrot..." << std::endl;
    generarMandelbrot(imagenOriginal);
    auto endA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempoA = endA - startA;
    std::cout << "Tarea A finalizada en: " << tiempoA.count() << " segundos." << std::endl;

    // --- Ejecución Tarea B ---
    auto startB = std::chrono::high_resolution_clock::now();
    std::cout << "Ejecutando Tarea B: Aplicando Filtro Gaussiano 5x5..." << std::endl;
    aplicarFiltroGaussiano(imagenOriginal, imagenFiltrada);
    auto endB = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempoB = endB - startB;
    std::cout << "Tarea B finalizada en: " << tiempoB.count() << " segundos." << std::endl;

    // Guardar el resultado final
    std::cout << "Guardando imagen final en '" << OUTPUT_FILE << "'..." << std::endl;
    guardarImagenPPM(imagenFiltrada, OUTPUT_FILE);
    
    std::cout << "Proceso completado con exito." << std::endl;
    return 0;
}