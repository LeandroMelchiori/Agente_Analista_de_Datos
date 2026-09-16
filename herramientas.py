import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools import PythonAstREPLTool
from IPython.display import display, Markdown
from langsmith import Client
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool, tool
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

load_dotenv()

# Prompt de las Herramientas

prompt_herramienta_explorador = """
Eres un analista de datos encargado de presentar un resumen informativo sobre un **DataFrame**
a partir de una {pregunta} hecha por el usuario.

A continuación, encontrarás la información general de la base de datos:

================= INFORMACIÓN DEL DATAFRAME =================

Dimensiones: {shape}

Columnas y tipos de datos:
{columns}

Valores nulos por columna:
{nulos}

Cadenas 'nan' (en cualquier capitalización) por columna:
{nans_str}

Filas duplicadas: {duplicados}

============================================================

Con base en esta información, redacta un resumen claro y organizado que contenga:

1. Un título: ## Reporte de información general sobre el dataset,
2. La dimensión total del DataFrame;
3. La descripción de cada columna (incluyendo nombre, tipo de dato y qué representa esa columna);
4. Las columnas que contienen datos nulos, con la respectiva cantidad;
5. Las columnas que contienen cadenas 'nan', con la respectiva cantidad;
6. La existencia (o no) de datos duplicados;
7. Un párrafo sobre los análisis que se pueden realizar con estos datos;
8. Un párrafo sobre los tratamientos que se pueden aplicar a los datos.
"""

prompt_herramienta_estadistica = """ Eres un analista de datos encargado de interpretar resultados estadísticos de una base de datos a partir de una {pregunta} realizada por el usuario.

A continuación, encontrarás las estadísticas descriptivas de la base de datos:

================= ESTADÍSTICAS DESCRIPTIVAS =================

{resumen}

============================================================


Con base en estos datos, elabora un resumen explicativo con un lenguaje claro, accesible y fluido,
destacando los principales puntos de los resultados. Incluye:

1. Un título: ## Informe de estadísticas descriptivas;
2. Una visión general de las estadísticas de las columnas numéricas;
3. Un párrafo sobre cada una de las columnas, comentando información sobre sus valores;
4. Identificación de posibles valores atípicos con base en los valores mínimo y máximo;
5. Recomendaciones de próximos pasos en el análisis en función de los patrones identificados.
"""
prompt_herramienta_visual="""Eres un especialista en visualización de datos. Tu tarea es generar **únicamente el código Python**
para graficar con base en la solicitud del usuario.

## Solicitud del usuario:
"{pregunta}"

## Metadatos del DataFrame:
{columnas}

## Muestra de los datos (3 primeras filas):
{muestra}

## Instrucciones obligatorias:
1. Usa las bibliotecas `matplotlib.pyplot` (como `plt`) y `seaborn` (como `sns`);
2. Define el tema con `sns.set_theme()`;
3. Asegúrate de que todas las columnas mencionadas en la solicitud existan en el DataFrame llamado `df`;
4. Elige el tipo de gráfico adecuado según el análisis solicitado:
- **Distribución de variables numéricas**: `histplot`, `kdeplot`, `boxplot` o `violinplot`
- **Distribución de variables categóricas**: `countplot`
- **Comparación entre categorías**: `barplot`
- **Relación entre variables**: `scatterplot`
- **Series temporales**: `lineplot`, con el eje X formateado como fechas
5. Configura el tamaño del gráfico con `figsize=(8, 4)`;
6. Añade título y etiquetas (`labels`) apropiadas a los ejes;
7. Posiciona el título a la izquierda con `loc='left'`, deja el `pad=20` y usa `fontsize=14`;
8. Mantén los ticks del eje X sin rotación con `plt.xticks(rotation=0)`;
9. Elimina los bordes superior y derecho del gráfico con `sns.despine()`;
10. Finaliza el código con `plt.show()`.

Devuelve ÚNICAMENTE el código Python generado separando cada instrucción con ';',
sin ningún texto adicional ni explicación.

Código Python:
"""

## Herramientas

def crear_herramientas(df):

    @tool("herramienta_informaciones_df", return_direct=True)
    def herramienta_informaciones_df(pregunta: str) -> str:
        """
        Utiliza esta herramienta siempre que el usuario solicite información
        general sobre el DataFrame, incluyendo número de filas y columnas,
        nombres y tipos de datos, valores nulos y duplicados.
        """

        shape = df.shape
        columns = df.dtypes
        nulos = df.isnull().sum()
        duplicados = df.duplicated().sum()

        nans_str = df.apply(
            lambda col: col[~col.isna()]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("nan")
        ).sum()

        plantilla_respuesta = PromptTemplate(
            template=prompt_herramienta_explorador,
            input_variables=[
                "pregunta",
                "shape",
                "columns",
                "nulos",
                "nans_str",
                "duplicados",
            ],
        )

        cadena = plantilla_respuesta | llm | StrOutputParser()

        respuesta = cadena.invoke({
            "pregunta": pregunta,
            "shape": shape,
            "columns": columns,
            "nulos": nulos,
            "nans_str": nans_str,
            "duplicados": duplicados,
        })

        return respuesta


    @tool("herramienta_resumen_estadistico", return_direct=True)
    def herramienta_resumen_estadistico(pregunta: str) -> str:
        """
        Utiliza esta herramienta cuando el usuario solicite un resumen
        estadístico completo y descriptivo del DataFrame.
        """

        resumen = df.describe(
            include="number"
        ).transpose().to_string()

        plantilla_respuesta = PromptTemplate(
            template=prompt_herramienta_estadistica,
            input_variables=[
                "pregunta",
                "resumen",
            ],
        )

        cadena = plantilla_respuesta | llm | StrOutputParser()

        respuesta = cadena.invoke({
            "pregunta": pregunta,
            "resumen": resumen,
        })

        return respuesta


    @tool("herramienta_generar_grafico", return_direct=True)
    def herramienta_generar_grafico(pregunta: str):
        """
        Utiliza esta herramienta siempre que el usuario solicite generar
        o visualizar un gráfico a partir del DataFrame.
        """

        columnas_info = "\n".join(
            f"{col} ({dtype})"
            for col, dtype in df.dtypes.items()
        )

        muestra_datos = df.head(3).to_dict(
            orient="records"
        )

        plantilla_respuesta = PromptTemplate(
            template=prompt_herramienta_visual,
            input_variables=[
                "pregunta",
                "columnas",
                "muestra",
            ],
        )

        cadena = plantilla_respuesta | llm | StrOutputParser()

        script_bruto = cadena.invoke({
            "pregunta": pregunta,
            "columnas": columnas_info,
            "muestra": muestra_datos,
        })

        script_limpio = (
            script_bruto
            .replace("```python", "")
            .replace("```", "")
            .strip()
        )

        exec_globals = {
            "df": df,
            "plt": plt,
            "sns": sns,
        }

        exec_locals = {}

        exec(
            script_limpio,
            exec_globals,
            exec_locals,
        )

        fig = plt.gcf()

        st.pyplot(fig)

        return ""


    herramienta_codigos_python = PythonAstREPLTool(
        name="herramienta_codigos_python",
        locals={"df": df},
        description="""
        Utiliza esta herramienta cuando el usuario solicite cálculos,
        consultas o transformaciones específicas sobre el DataFrame `df`.

        Ejemplos:
        - calcular el promedio de una columna;
        - calcular correlaciones;
        - obtener valores únicos;
        - filtrar datos;
        - realizar agrupaciones.

        No utilizar para información general del DataFrame,
        resúmenes estadísticos completos o generación de gráficos.
        """,
        return_direct=False,
    )

    return [
        herramienta_informaciones_df,
        herramienta_resumen_estadistico,
        herramienta_generar_grafico,
        herramienta_codigos_python,
    ]