from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from typing import Dict, List
import calendar
from fastapi.responses import HTMLResponse
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Cargar el DataFrame limpio
df = pd.read_csv('../Datasets/movies_top_10_percent.csv', parse_dates=['release_date'])

# Función auxiliar para convertir mes y día en español a números
def mes_a_numero(mes: str) -> int:
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8, 
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    return meses.get(mes.lower(), 0)

def dia_a_numero(dia: str) -> int:
    dias = {
        'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3, 
        'viernes': 4, 'sábado': 5, 'domingo': 6
    }
    return dias.get(dia.lower(), -1)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

# Endpoint para cantidad de filmaciones por mes
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str) -> Dict[str, str]:
    mes_num = mes_a_numero(mes)
    if mes_num == 0:
        return {"mensaje": f"Mes {mes} no válido"}
    
    cantidad = df[df['release_date'].dt.month == mes_num].shape[0]
    return {"mensaje": f"{cantidad} cantidad de películas fueron estrenadas en el mes de {mes}"}

# Endpoint para cantidad de filmaciones por día
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str) -> Dict[str, str]:
    dia_num = dia_a_numero(dia)
    if dia_num == -1:
        return {"mensaje": f"Día {dia} no válido"}
    
    cantidad = df[df['release_date'].dt.weekday == dia_num].shape[0]
    return {"mensaje": f"{cantidad} cantidad de películas fueron estrenadas en los días {dia}"}

# Endpoint para obtener score de una película
@app.get("/score_titulo/{titulo_de_la_filmacion}")
async def score_titulo(titulo_de_la_filmacion: str):
    # Buscar la película por título (ignorando mayúsculas y minúsculas)
    pelicula = df[df['title'].str.lower() == titulo_de_la_filmacion.lower()]
    
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    
    # Obtener los datos necesarios
    titulo = pelicula['title'].iloc[0]
    anio = pelicula['release_year'].iloc[0]
    score = pelicula['popularity'].iloc[0]
    
    # Formar la respuesta
    respuesta = f"La película {titulo} fue estrenada en el año {anio} con un score/popularidad de {score:.2f}"
    
    return {"mensaje": respuesta}
# Endpoint para obtener votos de una película
@app.get("/votos_titulo/{titulo_de_la_filmacion}")
async def votos_titulo(titulo_de_la_filmacion: str):
    # Buscar la película por título (ignorando mayúsculas y minúsculas)
    pelicula = df[df['title'].str.lower() == titulo_de_la_filmacion.lower()]
    
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    
    # Obtener los datos necesarios
    titulo = pelicula['title'].iloc[0]
    votos = pelicula['vote_count'].iloc[0]
    promedio_votos = pelicula['vote_average'].iloc[0]
    anio = pelicula['release_year'].iloc[0]
    
    # Verificar si la película tiene al menos 2000 votos
    if votos < 2000:
        return {
            "mensaje": f"La película {titulo} no cumple con la condición de tener al menos 2000 votos. "
                       f"Actualmente tiene {votos} votos."
        }
    
    # Formar la respuesta
    respuesta = (f"La película {titulo} fue estrenada en el año {anio}. "
                 f"La misma cuenta con un total de {votos:.0f} valoraciones, "
                 f"con un promedio de {promedio_votos:.2f}")
    
    return {"mensaje": respuesta}

# Endpoint para obtener información de un actor
def process_casting(casting_str):
    try:
        return ast.literal_eval(casting_str)
    except:
        return []

# Procesar la columna 'casting' una vez al inicio
df['casting_list'] = df['casting'].apply(process_casting)

@app.get("/get_actor/{nombre_actor}")
async def get_actor(nombre_actor: str):
    # Filtrar las películas donde el actor participa
    peliculas_actor = df[df['casting_list'].apply(lambda x: nombre_actor.lower() in [actor.lower() for actor in x])]
    
    if peliculas_actor.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado en el dataset")
    
    # Calcular estadísticas
    cantidad_peliculas = len(peliculas_actor)
    retorno_total = peliculas_actor['return'].sum()
    retorno_promedio = peliculas_actor['return'].mean()
    
    # Formar la respuesta
    respuesta = (f"El actor {nombre_actor} ha participado de {cantidad_peliculas} cantidad de filmaciones, "
                 f"el mismo ha conseguido un retorno de {retorno_total:.2f} con un promedio de {retorno_promedio:.2f} por filmación")
    
    return {"mensaje": respuesta}

# Endpoint para obtener información de un director
# Función auxiliar para procesar la columna 'director'
def process_director(director_str):
    try:
        return ast.literal_eval(director_str)
    except:
        return []

# Procesar la columna 'director' una vez al inicio
df['director_list'] = df['director'].apply(process_director)

class PeliculaInfo(BaseModel):
    titulo: str
    fecha_lanzamiento: str
    retorno: float
    costo: float
    ganancia: float

class DirectorInfo(BaseModel):
    nombre: str
    retorno_total: float
    peliculas: List[PeliculaInfo]

@app.get("/get_director/{nombre_director}", response_model=DirectorInfo)
async def get_director(nombre_director: str):
    # Filtrar las películas donde el director participa
    peliculas_director = df[df['director_list'].apply(lambda x: nombre_director.lower() in [director.lower() for director in x])]
    
    if peliculas_director.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado en el dataset")
    
    # Calcular el retorno total
    retorno_total = peliculas_director['return'].sum()
    
    # Preparar la información de cada película
    peliculas_info = []
    for _, pelicula in peliculas_director.iterrows():
        peliculas_info.append(PeliculaInfo(
            titulo=pelicula['title'],
            fecha_lanzamiento=pelicula['release_date'].strftime('%Y-%m-%d'),
            retorno=pelicula['return'],
            costo=pelicula['budget'],
            ganancia=pelicula['revenue'] - pelicula['budget']
        ))
    
    # Crear y devolver la respuesta
    return DirectorInfo(
        nombre=nombre_director,
        retorno_total=retorno_total,
        peliculas=peliculas_info
    )

# Preparar los datos para el sistema de recomendación
def combinar_caracteristicas(row):
    return ' '.join([str(row['genres']), str(row['director']), str(row['casting'])])

df['combined_features'] = df.apply(combinar_caracteristicas, axis=1)

# Crear la matriz de características
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(df['combined_features'])

# Calcular la similitud del coseno
cosine_sim = cosine_similarity(count_matrix)

# Función para obtener recomendaciones
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 películas similares
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Endpoint para recomendaciones
@app.get("/recomendacion/{titulo}")
async def recomendacion(titulo: str):
    try:
        recomendaciones = get_recommendations(titulo)
        return {"recomendaciones": recomendaciones}
    except IndexError:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
