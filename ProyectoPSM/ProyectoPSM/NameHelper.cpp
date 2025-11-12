#include "NameHelper.h"
#include <string>
#include <iomanip>
#include <sstream>

std::string padCeros(int numero, int anchura) {
    std::stringstream ss;
    ss << std::setw(anchura) << std::setfill('0') << numero;
    return ss.str();
}

std::vector<std::string> NameHelper::GenerarNombres()
{
    std::vector<std::string> nameList;

    const int FOTOS_POR_COMBINACION = 5;

    std::vector<std::string> codigos = { "07", "08", "09" };

    const std::vector<std::string> orientaciones = {
        "000", "045", "090", "135", "180", "225", "270", "315"
    };

    const std::vector<std::string> angulosCenitales = { "10", "40", "70", "90" };

    // Bucle 1: Ángulos (verticales)
    for (const std::string& angulo : angulosCenitales) {

        // Bucle 2: Códigos (07, 08, 09)
        for (const std::string& codigo : codigos) {

            // Bucle 3: Orientaciones (horizontales)
            for (const std::string& orientacion : orientaciones) {

                // Bucle 4: Número de secuencia (1 a 5)
                for (int i = 1; i <= FOTOS_POR_COMBINACION; ++i) {

                    std::string numSecuencia = padCeros(i, 3);

                    std::string nombreFinal = codigo + "_" +
                        orientacion + "_" +
                        angulo + "_" +
                        numSecuencia;

                    nameList.push_back(nombreFinal);
                }
            }
        }
    }

    return nameList;
}