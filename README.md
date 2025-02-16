# Proyecto de Análisis de Datos Cinematográficos y Sistema de Recomendación

<h1 align="center">
  <br>
  <a href="portada"><img src="images/portada.jpeg" alt="portda" width="500"></a>
  <br>
  API de Películas
  <br>
</h1>

<h4 align="center">Bienvenido a la API de Películas</h4>

<p align="center">
  <a href="#Introducción">Introducción</a> •
  <a href="#Instalación y Requisitos">Instalación y Requisitos</a> •
  <a href="#Estructura del Proyecto">Estructura del Proyecto</a> •
  <a href="#Datos y Fuentes">Datos y Fuentes</a> •
  <a href="#Uso y Ejecución">Uso y Ejecución</a> •
  <a href="#EDA">EDA</a> •
  <a href="#API y Sistema de Recomendación">API y Sistema de Recomendación</a> •
  <a href="#Despliegue">Despliegue</a> •
  <a href="#Video">Video</a> •
  <a href="#Contribución">Contribución
</a> •
  <a href="#Contacto">Contacto</a>

</p>

## Introducción

Este proyecto se desarrolla en el contexto de una start-up que provee servicios de agregación de plataformas de streaming. El objetivo es crear un sistema de recomendación de películas, partiendo desde cero en un entorno con datos poco maduros y sin procesar.

El proyecto abarca desde el tratamiento y recolección de los datos (tareas de Data Engineering) hasta el entrenamiento y despliegue de un modelo de Machine Learning para recomendaciones, pasando por el análisis exploratorio de los datos y el desarrollo de una API para consultas.

## 1. Instalación y Requisitos

### Requisitos:
- Python 3.7 o superior
- pandas
- numpy
- matplotlib
- scikit-learn
- FastAPI

### Pasos de instalación:
```bash
# Clonar el repositorio
git clone https://github.com/Adlazz/movies_rs.git

# Ir al directorio del repositorio
cd movies_rs

# Crear un entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar las dependencias
pip install -r requirements.txt
```

## 2. Estructura del Proyecto
```
movies_rs/
│
├── Datasets/                    # Contiene los archivos de datos utilizados
│   ├── movies_dataset.csv (descargar punto 3)
│   ├── credits.csv (descargar punto 3)
|   └── movies_top_10_percent.csv
|
├── Notebooks/                   # Informes y visualizaciones generados
│   ├── 1. ETL_movies.ipynb
|   └── 2. EDA_movies.ipynb
|
├── __pycache__/*           # Almacena archivos de código byte compilado de Python
│
├── images/                      # Imagenes varias y portada
│   ├── boxplot.png
│   ├── eda_dist_presupuesto.png
│   └── (more)
│
├── movie_env/*                  # Entorno virtual
│
├── .gitignore                   # Archivos que no se suben a Github
|
├── README.md                    # Este archivo
|
├── index.html                   # Da estilo a la página principal
|
├── main.py                      # Archivo de ejecución de fastAPI
│
└── requirements.txt             # Lista de dependencias del proyecto
```

## 3. Datos y Fuentes
Los datos utilizados en este proyecto provienen de dos archivos principales:
- `movies_dataset.csv`: Contiene información sobre películas, incluyendo presupuestos, ingresos, fechas de estreno, etc.
- `credits.csv`: Contiene información sobre el elenco y equipo de las películas.
- Los dos archivos los puedes obtener en el siguiente enlace: https://drive.google.com/drive/folders/1X_LdCoGTHJDbD28_dJTxaD4fVuQC9Wt5

## 4. Uso y Ejecución
1. Para ejecutar la carga y la limpieza de los archivos, use el notebook `Notebooks/1. ETL_movies.ipynb`.
2. Para realizar el análisis exploratorio de datos (EDA), abra el notebook `Notebooks/2. EDA_movies.ipynb`.
3. Para iniciar la API, ejecute:
   ```
   uvicorn main:app --reload
   ```

## 5. Análisis Exploratorio de Datos (EDA)

El EDA incluyó:

- Análisis de la distribución de presupuestos y recaudación.
- Estudio de la evolución temporal de la producción cinematográfica.
- Identificación de outliers y anomalías en los datos.
- Creación de nubes de palabras para analizar frecuencias en títulos de películas.

<img src="images\nube.png" alt="nube">


### Resultados del Análisis de Correlación

La matriz de correlación reveló las siguientes relaciones importantes:

- Fuerte correlación positiva entre **budget** y **revenue** (0.75)
- Correlación moderada entre **budget** y **vote_count** (0.61)
- Correlación fuerte entre **revenue** y **vote_count** (0.78)
- Baja correlación entre **budget** y **vote_average** (0.0068)
- Correlación moderada entre **popularity** y **revenue** (0.38)

Estas correlaciones sugieren que mientras el presupuesto y los ingresos están fuertemente relacionados, la calidad percibida (medida por el promedio de votos) no está necesariamente vinculada a estos factores financieros.
<div style="text-align: center;">
<img src="images\matriz.png" alt="matriz" width="400" height="350">
</div>

**Aclaración:**
Para el desarrollo de este análisis exploratorio se optó por reducir el Dataframe según popularidad, promedio de votos y cantidad de votos. Esto es debido a las limitaciones del servidor y a modo de optimizar el proceso reducimos la cantidad de datos de 45000 a 4500 aproximadamente (nos quedamos con el 10% de las mejores películas).  
Ya que el objetivo es recomendar películas, elegimos de la data original aquellas mejores valoradas.

Puede usted hacer uso del `Notebooks/1. ETL_movies.ipynb` y hacer su propio recorte de datos para obtener sus propias conclusiones.

Ver el análisis exploratorio completo en el archivo: `Notebooks/2. EDA_movies.ipynb`

## 6. API y Sistema de Recomendación

<div style="text-align: center;">
<img src="images\Leo-meme.jpeg" alt="leo" width="60%">
</div>

La API desarrollada con FastAPI proporciona los siguientes endpoints:

1. `/cantidad_filmaciones_mes/{mes}`: Devuelve la cantidad de películas estrenadas en un mes específico.
2. `/cantidad_filmaciones_dia/{dia}`: Devuelve la cantidad de películas estrenadas en un día específico.
3. `/score_titulo/{titulo}`: Devuelve el título, año de estreno y score de una película.
4. `/votos_titulo/{titulo}`: Devuelve información sobre los votos de una película (si tiene al menos 2000 valoraciones).
5. `/get_actor/{nombre_actor}`: Devuelve información sobre el éxito de un actor basado en el retorno de sus películas.
6. `/get_director/{nombre_director}`: Devuelve información sobre el éxito de un director y detalles de sus películas.
7. `/recomendacion/{titulo}`: Recomienda 5 películas similares a la película especificada.

## 7. Despliegue

La API se desplegó utilizando Render, permitiendo que sea accesible desde la web. (Alternativamente, se podría usar Railway u otro servicio similar).

https://movies-rs-fa0j.onrender.com


## 8. Video Demostrativo

Enlace: https://drive.google.com/file/d/1P85mGhei5a1uHnj8kxjsfQ6IemVNB23a/view?usp=sharing
  
## 9. Contribución y Colaboración

Las contribuciones son bienvenidas. Por favor, siga estos pasos para contribuir:
1. Haz un fork del repositorio en [https://github.com/Adlazz/movies_rs.git](https://github.com/Adlazz/movies_rs.git).

2. Clona tu fork a tu máquina local:
   ```
   git clone https://github.com/TU_USUARIO/movies_rs.git
   ```

3. Crea una nueva rama para tu contribución:
   ```
   git checkout -b feature/TuNuevaCaracteristica
   ```

4. Realiza tus cambios y mejoras en el código.

5. Asegúrate de que tus cambios cumplan con las guías de estilo del proyecto y que todas las pruebas pasen.

6. Haz commit de tus cambios:
   ```
   git commit -m 'Añade alguna característica increíble'
   ```

7. Sube tus cambios a tu fork en GitHub:
   ```
   git push origin feature/TuNuevaCaracteristica
   ```

8. Abre un Pull Request en el repositorio original [https://github.com/Adlazz/movies_rs.git](https://github.com/Adlazz/movies_rs.git).

9. Espera la revisión y feedback. Es posible que se te pida realizar algunos cambios antes de que tu contribución sea aceptada.

10. Una vez aprobada, tu contribución será fusionada con la rama principal del proyecto.

Gracias por contribuir a movies_rs.

## 10. Contacto y Más Información

Para más información sobre este proyecto, puede contactar a: [Adrián Lazzarini](mailto:adrianlazzarini@gmail.com)

---

Este README es parte de un proyecto académico desarrollado como MVP. Representa un trabajo en progreso y está sujeto a mejoras y actualizaciones continuas.
