#################################################################
# FASE 4 - Despliegue del modelo de multilateración
#################################################################
# Definimos una función que recibe las mediciones a un destino por parte de los monitores 
# y devuelve la posición estimada del destino. La función se basa en el algoritmo de multilateración, 
# que consiste en calcular la intersección de las circunferencias que representan las distancias a los monitores.

# Importamos las librerías necesarias
import mlflow.sklearn
import pandas as pd
import numpy as np
import geopy.distance


#################################################################
# Función de carga de modelos MLFLOW
#################################################################
def load_mlflow_models():    
    # Configuración de la URI de MLFlow
    mlflow.set_tracking_uri("http://localhost:5000")  # Ajusta esto a tu URI de MLFlow

    # Carga del modelo
    knn_3_model_name = "Arch3_KNN_3"
    knn_5_model_name = "Arch3_KNN_5"

    knn_3_version = 2
    knn_5_version = 2

    knn_3_mlflow_model = mlflow.sklearn.load_model(model_uri=f"models:/{knn_3_model_name}/{knn_3_version}")
    knn_5_mlflow_model = mlflow.sklearn.load_model(model_uri=f"models:/{knn_5_model_name}/{knn_5_version}")

    return knn_3_mlflow_model, knn_5_mlflow_model

#################################################################
# Función de cálculo de la posición estimada para KNN 3
#################################################################
def predict_knn_3(df_3, y_train, knn_3_mlflow_model):
    # Añadimos las columnas necesarias a la tabla de mediciones
    df_3['coords_1'] = None
    df_3['coords_2'] = None
    df_3['coords_3'] = None
    df_3['latitude_pred'] = None
    df_3['longitude_pred'] = None
    # Por cada entrada de la tabla de mediciones, calculamos la posición estimada
    for index, row in df_3.iterrows():
        
        X_valid = row[['latency_m1', 'latency_m2', 'latency_m3', 'latency_m4', 'latency_m5', 'latency_m6']]
        # Convertimos de serie a dataframe
        X_valid = pd.DataFrame(X_valid).transpose()

        # Haciendo uso de knn_3 obtenemos los vecinos más cercanos para la primera entrada de fingerprint
        distances, indices = knn_3_mlflow_model.kneighbors(X_valid, n_neighbors=3)

        # Obtenemos las coordenadas de los vecinos más cercanos
        nearest_neighbors_coords = y_train[indices[0]]
        lat_1, lon_1 = map(float, nearest_neighbors_coords[0].replace('[', '').replace(']', '').split())
        lat_2, lon_2 = map(float, nearest_neighbors_coords[1].replace('[', '').replace(']', '').split())
        lat_3, lon_3 = map(float, nearest_neighbors_coords[2].replace('[', '').replace(']', '').split())
        coords_1 = [lat_1, lon_1]
        coords_2 = [lat_2, lon_2]
        coords_3 = [lat_3, lon_3]

        # Añadimos las coordenadas de los vecinos más cercanos a la tabla de mediciones
        df_3.at[index, 'coords_1'] = coords_1
        df_3.at[index, 'coords_2'] = coords_2
        df_3.at[index, 'coords_3'] = coords_3

        # Calculamos la posición estimada
        distances = [
            geopy.distance.distance((lat_1, lon_1), (lat_2, lon_2)).km,
            geopy.distance.distance((lat_2, lon_2), (lat_3, lon_3)).km,
            geopy.distance.distance((lat_3, lon_3), (lat_1, lon_1)).km
        ]

        # Obtenemos el centroide del triángulo
        lat_pred = (lat_1 + lat_2 + lat_3) / 3
        lon_pred = (lon_1 + lon_2 + lon_3) / 3
        centroid = [
            lat_pred,
            lon_pred
        ]

        # Añadimos la posición estimada a la tabla de mediciones
        df_3.at[index, 'latitude_pred'] = lat_pred
        df_3.at[index, 'longitude_pred'] = lon_pred

    return df_3

#################################################################
# Función de cálculo de la posición estimada para KNN 5
#################################################################
def predict_knn_5(df_5, y_train, knn_5_mlflow_model):
    # Añadimos las columnas de coordenadas de los vecinos más cercanos y de la posición estimada
    df_5['coords_1'] = None
    df_5['coords_2'] = None
    df_5['coords_3'] = None
    df_5['coords_4'] = None
    df_5['coords_5'] = None
    df_5['latitude_pred'] = None
    df_5['longitude_pred'] = None
    # Por cada entrada de la tabla de mediciones, calculamos la posición estimada
    for index, row in df_5.iterrows():
        X_valid = row[['latency_m1', 'latency_m2', 'latency_m3', 'latency_m4', 'latency_m5', 'latency_m6']]
        # Convertimos de serie a dataframe
        X_valid = pd.DataFrame(X_valid).transpose()

        # Haciendo uso de knn_5 obtenemos los vecinos más cercanos para la primera entrada de fingerprint
        distances, indices = knn_5_mlflow_model.kneighbors(X_valid, n_neighbors=5)

        # Obtenemos las coordenadas de los vecinos más cercanos
        nearest_neighbors_coords = y_train[indices[0]]
        lat_1, lon_1 = map(float, nearest_neighbors_coords[0].replace('[', '').replace(']', '').split())
        lat_2, lon_2 = map(float, nearest_neighbors_coords[1].replace('[', '').replace(']', '').split())
        lat_3, lon_3 = map(float, nearest_neighbors_coords[2].replace('[', '').replace(']', '').split())
        lat_4, lon_4 = map(float, nearest_neighbors_coords[3].replace('[', '').replace(']', '').split())
        lat_5, lon_5 = map(float, nearest_neighbors_coords[4].replace('[', '').replace(']', '').split())
        coords_1 = [lat_1, lon_1]
        coords_2 = [lat_2, lon_2]
        coords_3 = [lat_3, lon_3]
        coords_4 = [lat_4, lon_4]
        coords_5 = [lat_5, lon_5]

        # Añadimos las coordenadas de los vecinos más cercanos a la tabla de mediciones
        df_5.at[index, 'coords_1'] = coords_1
        df_5.at[index, 'coords_2'] = coords_2
        df_5.at[index, 'coords_3'] = coords_3
        df_5.at[index, 'coords_4'] = coords_4
        df_5.at[index, 'coords_5'] = coords_5

        # Calculamos la posición estimada
        distances = [
            geopy.distance.distance((lat_1, lon_1), (lat_2, lon_2)).km,
            geopy.distance.distance((lat_2, lon_2), (lat_3, lon_3)).km,
            geopy.distance.distance((lat_3, lon_3), (lat_4, lon_4)).km,
            geopy.distance.distance((lat_4, lon_4), (lat_5, lon_5)).km,
            geopy.distance.distance((lat_5, lon_5), (lat_1, lon_1)).km
        ]

        # Obtenemos el centroide del pentágono
        lat_pred = (lat_1 + lat_2 + lat_3 + lat_4 + lat_5) / 5
        lon_pred = (lon_1 + lon_2 + lon_3 + lon_4 + lon_5) / 5
        centroid = [
            lat_pred,
            lon_pred
        ]

        # Añadimos la posición estimada a la tabla de mediciones
        df_5.at[index, 'latitude_pred'] = lat_pred
        df_5.at[index, 'longitude_pred'] = lon_pred


    return df_5

#################################################################
# Función principal que va a ser llamada para predecir la posición de un objetivo
#################################################################
# Función que recibe las mediciones a un destino por parte de los monitores
# y devuelve la posición estimada del destino
def predict_target_position(df):
    # Directorio de trabajo
    work_dir = 'C:/1_PhD_git_repo/ML-Architectures'

    # Cargamos los modelos de MLFlow
    knn_3_mlflow_model, knn_5_mlflow_model = load_mlflow_models()

    # Cargamos el dataset necesario para obtener la ubicación geográfica de los vecinos más cercanos
    training_data = pd.read_csv(work_dir + '/Arch3-KnnFingerprint_6Monit/tmp/fingerprintKNN_train.csv')    
    
    # Obtenemos la columna objetivo con las coordenadas y la convertimos a numpy arrays
    y_train = training_data[['gps_coord']]
    y_train = np.array([np.array(coord) for coord in y_train['gps_coord']])

    # # Dependiendo del número de vecinos, llamamos a una función u otra
    # if num_neighbors == 3:
    #     # Llamamos a la función predict_knn_3
    #     df_result = predict_knn_3(df, y_train, knn_3_mlflow_model)
    # elif num_neighbors == 5:
    #     # Llamamos a la función predict_knn_5
    #     df_result = predict_knn_5(df, y_train, knn_5_mlflow_model)
    # else:
    #     # Si el número de vecinos no es 3 ni 5, devolvemos un error
    #     return "Error: El número de vecinos debe ser 3 o 5"

    # Copia de df para evitar modificar el original
    df_3 = df.copy()
    df_5 = df.copy()

    df_result_knn_3 = predict_knn_3(df_3, y_train, knn_3_mlflow_model)
    df_result_knn_5 = predict_knn_5(df_5, y_train, knn_5_mlflow_model)
    
    return df_result_knn_3, df_result_knn_5

#################################################################
# Código de prueba
#################################################################
# # Indicamos el directorio de trabajo
# work_dir = 'c:/2_Github_Repositories/TFM-Teleco-Geontology'

# ml_dir = work_dir + '/ML-Architectures/Arch1-GlobalRegression'

# # Cargamos un dataset de prueba
# df = pd.read_csv(ml_dir + '/tmp/test_set.csv')

# # Llamamos a la función predict_target_position
# df_result = predict_target_position(df)

# # Mostramos el resultado
# print(df_result)
