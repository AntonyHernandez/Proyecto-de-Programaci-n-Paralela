#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <omp.h> 
#include <vector>
#include <mutex>

// Configuración de Ultra-Alta Resolución (8K UHD: 7680 x 4320)
const int WIDTH = 7680;
const int HEIGHT = 4320;
const std::string OUTPUT_FILE = "mandelbrot_8k_blurred.ppm";

// Estructura para almacenar un píxel en formato RGB
struct Pixel {
    unsigned char r, g, b;
};

// --- TAREA A: Generación del Conjunto de Mandelbrot ---
// DOCUMENTACIÓN DE CAMBIOS (Línea Base Paralela):
// Se añade la directiva `#pragma omp parallel for schedule(runtime)` en el bucle externo 'y'.
// Se utiliza 'schedule(runtime)' para permitir la modificación dinámica del planificador (static, dynamic, guided)
// y el tamaño del bloque (chunk size) a través de variables de entorno sin necesidad de recompilar.

void generarMandelbrot(std::vector<Pixel>& imagen, bool analizar) {
    const int MAX_ITER = 500;
    
    // Límites del plano complejo para encuadrar el fractal
    const double minX = -2.0, maxX = 0.5;
    const double minY = -1.25, maxY = 1.25;

    const std::string OUTPUT_FILE = "mandelbrot_8k_blurred.ppm";

    const char* eval[] = {"static","dynamic","guided"};
    int limite = 16;
    int max=3;

    if (!analizar){
        limite = 4;
        max=1;
    }
    

    omp_sched_t schedulers[] = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};

    std::vector<int> historial(MAX_ITER + 1, 0);
    std::mutex mutex;

    double start = omp_get_wtime();

    // Parte No.3 Balanceo de Carga (Schedulers):
    for (int s = 0; s < max; s++) {

        for (int chunks = 1; chunks <= limite; chunks=chunks*2) {

            omp_set_schedule(schedulers[s], chunks);

            double start = omp_get_wtime();

            #pragma omp parallel for schedule(runtime)
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
            double end = omp_get_wtime();

            std::cout<< eval[s]<< ", chunk="<< chunks << " -> Tiempo: "<< (end - start)<< " s\n";

        }
    }

}

// --- TAREA B: Aplicación de Filtro de Convolución 2D (Desenfoque Gaussiano pesado) ---
// DOCUMENTACIÓN DE CAMBIOS (Línea Base Paralela):
// Se añade la directiva `#pragma omp parallel for schedule(static)` en el bucle externo 'y'.
// Dado que cada píxel de la convolución realiza exactamente la misma cantidad de operaciones aritméticas (matriz 5x5),
// el planificador estático por defecto distribuye el rango de filas de forma equitativa y con la menor sobrecarga de hilos posible.

void aplicarFiltroGaussiano(const std::vector<Pixel>& origen, std::vector<Pixel>& destino) {
    // Matriz de convolución (Kernel Gaussiano 5x5)
    const int K_SIZE = 5;
    const double kernel[5][5] = {
        {1/273.0,  4/273.0,  7/273.0,  4/273.0, 1/273.0},
        {4/273.0, 16/273.0, 26/273.0, 16/273.0, 4/273.0},
        {7/273.0, 26/273.0, 41/273.0, 26/273.0, 7/273.0},
        {4/273.0, 16/273.0, 26/273.0, 16/273.0, 4/273.0},
        {1/273.0,  4/273.0,  7/273.0,  4/273.0, 1/273.0}
    };
    
    int offset = K_SIZE / 2;

   
    // Convolución píxel por píxel paralelizada
    #pragma omp parallel
    {
        
        int id = omp_get_thread_num();
        int n = omp_get_num_threads();

        //Parte No.5 SPMD y Afinidad:
        //Para implementar el SPMD se repartio las filas entre los diferentes nucleos
        for (int y = id; y < HEIGHT; y += n) {

            // Caso de bordes: mantener el píxel original para simplificar
            if (y < offset || y >= HEIGHT - offset) {
                for (int x = 0; x < WIDTH; ++x) {
                    destino[y * WIDTH + x] = origen[y * WIDTH + x];
                }
                continue; 
            }

            for (int x = 0; x < WIDTH; ++x) {
                
                if (x < offset || x >= WIDTH - offset) {
                    destino[y * WIDTH + x] = origen[y * WIDTH + x];
                    continue;
                }

                double sumR = 0.0, sumG = 0.0, sumB = 0.0;

                // Operación de vecindad (Kernel)
                for (int ky = 0; ky < K_SIZE; ++ky) {
                    int pixelY = y + ky - offset;
                    const int rowOffset = pixelY * WIDTH;

                    // Basicamente, para lograr Forzar la vectorizaciòn se implementò SIMD en el nucle màs interno 
                    #pragma omp simd reduction(+:sumR, sumG, sumB)
                    for (int kx = 0; kx < K_SIZE; ++kx) {
                        int pixelX = x + kx - offset;
                        const Pixel& p = origen[rowOffset + pixelX];
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
}

// Función auxiliar para guardar en formato PPM P6 (Binario)
void guardarImagenPPM(const std::vector<Pixel>& imagen, const std::string& nombreArchivo) {
    std::ofstream archivo(nombreArchivo, std::ios::binary);
    if (!archivo) {
        std::cerr << "Error al abrir el archivo para escribir." << std::endl;
        return;
    }
    archivo << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    archivo.write(reinterpret_cast<const char*>(imagen.data()), imagen.size() * sizeof(Pixel));
    archivo.close();
}



//Funciòn histograma, guarda la colores presentados en la imagen final
void histograma(const std::vector<Pixel>& imagen) {
    long long H_Red[256] = {0};
    long long H_Green[256] = {0};
    long long H_Blue[256] = {0};

    int pixeles = WIDTH * HEIGHT;

    double start = omp_get_wtime();
    #pragma omp parallel
    {
        long long insideH_Red[256] = {0};
        long long insideH_Green[256] = {0};
        long long insideH_Blue[256] = {0};

        #pragma omp for schedule(static)
        for (int i = 0; i < pixeles; ++i) {
            insideH_Red[imagen[i].r]++;
            insideH_Green[imagen[i].g]++;
            insideH_Blue[imagen[i].b]++;
        }

        #pragma omp critical
        {
            for (int c = 0; c < 256; ++c) {
                H_Red[c] += insideH_Red[c];
                H_Green[c] += insideH_Green[c];
                H_Blue[c] += insideH_Blue[c];
            }
        }
    }
    double end = omp_get_wtime();
    std::cout<< "\n\tTiempo con variables locales y critical = "<< (end - start)<< "s\n";

    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:H_Red[:256], H_Green[:256], H_Blue[:256]) schedule(static)
    for (int i = 0; i < pixeles; ++i) {
        H_Red[imagen[i].r]++;
        H_Green[imagen[i].g]++;
        H_Blue[imagen[i].b]++;
    }
    end = omp_get_wtime();

    std::cout<< "\tTiempo reduction = " << (end - start)<< "s\n";

    // Se imprime el histograma en una tablita
    std::cout << "\n -------------- Histograma de Colores -------------- " << std::endl;
    std::cout << "Intensidad\tRojo (R)\tVerde (G)\tAzul (B)" << std::endl;

    int intensidades[] = {0, 32, 64, 128, 192, 255};

    for (int i : intensidades) {
        if (H_Red[i]>=10000000){
            std::cout << i << "\t\t" << H_Red[i] << "\t" << H_Green[i] << "\t" << H_Blue[i] << std::endl;
        }
        else{
            std::cout << i << "\t\t" << H_Red[i] << "\t\t" << H_Green[i] << "\t\t" << H_Blue[i] << std::endl;
        }
        
    }
    std::cout << " --------------------------------------------------- \n\n" << std::endl;
}

int main() {

    omp_set_num_threads(4);
    std::cout << "Iniciando Línea Base Paralela (Resolucion 8K)..." << std::endl;
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Ejecutando activamente con: " << omp_get_num_threads() << " hilos de CPU." << std::endl;
    }
    std::cout << "--------------------------------------------------" << std::endl;
    
    // Imágenes (7680 * 4320 píxeles cada una)
    std::vector<Pixel> imagenOriginal(WIDTH * HEIGHT);
    std::vector<Pixel> imagenFiltrada(WIDTH * HEIGHT);

    // --- Tarea A ---
    auto startA = std::chrono::high_resolution_clock::now();
    std::cout << "Ejecutando Tarea A: Generando Mandelbrot..." << std::endl;

    generarMandelbrot(imagenOriginal,true);

    auto endA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempoA = endA - startA;
    std::cout << "Tarea A finalizada en: " << tiempoA.count() << " segundos." << std::endl<<"\n\n";

    // --- Tarea B ---
    auto startB = std::chrono::high_resolution_clock::now();
    std::cout << "Ejecutando Tarea B: Aplicando Filtro Gaussiano 5x5..." << std::endl;

    aplicarFiltroGaussiano(imagenOriginal, imagenFiltrada);

    auto endB = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tiempoB = endB - startB;
    std::cout << "Tarea B finalizada en: " << tiempoB.count() << " segundos." << std::endl;

    // Parte No.4 Histograma De colores
    histograma(imagenFiltrada);

    // Guardar el resultado final
    std::cout << "Guardando imagen final en '" << OUTPUT_FILE << "'..." << std::endl;
    guardarImagenPPM(imagenFiltrada, OUTPUT_FILE);
    
    return 0;
}