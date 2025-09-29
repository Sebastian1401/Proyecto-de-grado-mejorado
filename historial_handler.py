import os

def obtener_datos_pacientes():
    """
    Escanea el directorio de resultados y recopila los datos de cada paciente.
    Devuelve una lista de diccionarios, donde cada diccionario es un paciente.
    """
    # Ruta a la carpeta principal de resultados
    ruta_principal = "resultados_prueba"

    lista_pacientes = []

    # Verificar si la carpeta principal existe para evitar errores
    if not os.path.isdir(ruta_principal):
        print(f"ADVERTENCIA: El directorio '{ruta_principal}' no existe.")
        return []

    # Listar todas las subcarpetas (cada una es un paciente por su cédula)
    for cedula_folder in os.listdir(ruta_principal):
        patient_path = os.path.join(ruta_principal, cedula_folder)

        # Nos aseguramos de que es una carpeta
        if os.path.isdir(patient_path):
            # El nombre de la carpeta es la cédula, lo guardamos
            datos_paciente = {'cedula': cedula_folder}

            # Buscamos y leemos el archivo de datos del paciente
            archivo_datos = os.path.join(patient_path, 'datos_paciente.txt')
            if os.path.exists(archivo_datos):
                with open(archivo_datos, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            # Guardamos la clave en minúsculas para facilitar la búsqueda después
                            datos_paciente[key.strip().lower()] = value.strip()

            lista_pacientes.append(datos_paciente)

    # Ordenar la lista de pacientes por cédula (opcional, pero recomendado)
    lista_pacientes.sort(key=lambda p: p.get('cedula', ''))

    return lista_pacientes

# Esta parte solo se ejecuta si corres el archivo directamente
# Sirve para hacer una prueba rápida
if __name__ == '__main__':
    pacientes = obtener_datos_pacientes()
    if pacientes:
        print("--- Prueba del script: Pacientes encontrados ---")
        for p in pacientes:
            print(f"- Cédula: {p.get('cedula', 'N/A')}, Nombre: {p.get('nombre', 'N/A')}")
    else:
        print("--- Prueba del script: No se encontraron datos de pacientes. ---")