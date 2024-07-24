from fastapi import FastAPI
import pandas as pd
from typing import Dict
import calendar
from fastapi.responses import HTMLResponse


app = FastAPI()

# Cargar el DataFrame limpio
df = pd.read_csv('movies_top_10_percent.csv', parse_dates=['release_date'])

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
@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str) -> Dict[str, str]:
    pelicula = df[df['title'].str.contains(titulo, case=False, na=False)].iloc[0]
    return {
        "titulo": pelicula['title'],
        "año": pelicula['release_year'],
        "score": pelicula['popularity']
    }

# Endpoint para obtener votos de una película
@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str) -> Dict[str, str]:
    pelicula = df[df['title'].str.contains(titulo, case=False, na=False)].iloc[0]
    if pelicula['vote_count'] < 2000:
        return {"mensaje": "La película no cumple con la condición de tener al menos 2000 valoraciones"}
    
    return {
        "titulo": pelicula['title'],
        "cantidad_votos": pelicula['vote_count'],
        "promedio_votos": pelicula['vote_average']
    }

# Endpoint para obtener información de un actor
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str) -> Dict[str, str]:
    peliculas_actor = df[df['casting'].str.contains(nombre_actor, case=False, na=False)]
    if peliculas_actor.empty:
        return {"mensaje": f"No se encontraron películas para el actor {nombre_actor}"}

    cantidad = peliculas_actor.shape[0]
    retorno_total = peliculas_actor['return'].sum()
    retorno_promedio = retorno_total / cantidad
    
    return {
        "actor": nombre_actor,
        "cantidad_filmaciones": cantidad,
        "retorno_total": retorno_total,
        "retorno_promedio": retorno_promedio
    }

# Endpoint para obtener información de un director
@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str) -> Dict[str, str]:
    peliculas_director = df[df['director'].str.contains(nombre_director, case=False, na=False)]
    if peliculas_director.empty:
        return {"mensaje": f"No se encontraron películas para el director {nombre_director}"}
    
    director_info = {
        "director": nombre_director,
        "cantidad_filmaciones": peliculas_director.shape[0],
        "retorno_total": peliculas_director['return'].sum(),
        "retorno_promedio": peliculas_director['return'].mean(),
        "peliculas": []
    }
    
    for _, row in peliculas_director.iterrows():
        director_info["peliculas"].append({
            "titulo": row['title'],
            "fecha_lanzamiento": row['release_date'].strftime("%Y-%m-%d"),
            "retorno_individual": row['return'],
            "costo": row['budget'],
            "ganancia": row['revenue']
        })
    
    return director_info

# Ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
