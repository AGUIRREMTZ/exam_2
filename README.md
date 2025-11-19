# NSL-KDD Dataset Processing API

API REST con Django para procesamiento de datasets usando transformadores y pipelines personalizados basados en scikit-learn.

## Características

- **División de Datasets**: Split estratificado en train/validation/test
- **Preprocesamiento**: Manejo de valores nulos, imputación
- **Transformadores Personalizados**: 
  - DeleteNanRows: Elimina filas con valores nulos
  - CustomScaler: Escalado robusto de features
  - CustomOneHotEncoding: Codificación one-hot personalizada
- **Pipelines**: Pipelines completos para preprocesamiento
- **Codificación Categórica**: One-Hot y Ordinal Encoding
- **Escalado de Features**: RobustScaler para datos numéricos

## Instalación Local

### Requisitos Previos
- Python 3.9+
- pip

### Pasos

1. **Clonar el repositorio**
\`\`\`bash
git clone <tu-repositorio>
cd <nombre-proyecto>
\`\`\`

2. **Crear entorno virtual**
\`\`\`bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
\`\`\`

3. **Instalar dependencias**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. **Aplicar migraciones**
\`\`\`bash
python manage.py migrate
\`\`\`

5. **Ejecutar servidor de desarrollo**
\`\`\`bash
python manage.py runserver
\`\`\`

La API estará disponible en `http://localhost:8000/api/`

## Endpoints de la API

### 1. Overview
\`\`\`
GET /api/
\`\`\`
Retorna información sobre todos los endpoints disponibles.

### 2. Split Dataset
\`\`\`
POST /api/split-dataset/
\`\`\`

**Body:**
\`\`\`json
{
  "data": [[0, "tcp", "http", ...], ...],
  "columns": ["duration", "protocol_type", "service", ...],
  "stratify_column": "protocol_type",
  "random_state": 42,
  "shuffle": true
}
\`\`\`

**Response:**
\`\`\`json
{
  "message": "Dataset split successfully",
  "train_size": 75583,
  "validation_size": 25195,
  "test_size": 25195,
  "train_sample": [...],
  "validation_sample": [...],
  "test_sample": [...]
}
\`\`\`

### 3. Preprocess Data
\`\`\`
POST /api/preprocess/
\`\`\`

**Body:**
\`\`\`json
{
  "data": [[...], ...],
  "columns": ["col1", "col2", ...],
  "remove_nan": false,
  "impute_strategy": "median"
}
\`\`\`

### 4. Transform Categorical
\`\`\`
POST /api/transform-categorical/
\`\`\`

**Body:**
\`\`\`json
{
  "data": [[...], ...],
  "columns": ["col1", "col2", ...],
  "encoding_type": "onehot"
}
\`\`\`

### 5. Scale Features
\`\`\`
POST /api/scale-features/
\`\`\`

**Body:**
\`\`\`json
{
  "data": [[...], ...],
  "columns": ["col1", "col2", ...],
  "features_to_scale": ["src_bytes", "dst_bytes"]
}
\`\`\`

### 6. Apply Pipeline
\`\`\`
POST /api/apply-pipeline/
\`\`\`

**Body:**
\`\`\`json
{
  "data": [[...], ...],
  "columns": ["col1", "col2", ...],
  "pipeline_type": "full"
}
\`\`\`

### 7. Dataset Info
\`\`\`
POST /api/dataset-info/
\`\`\`

**Body:**
\`\`\`json
{
  "data": [[...], ...],
  "columns": ["col1", "col2", ...]
}
\`\`\`

## Despliegue en Vercel

### Configuración

1. **Instalar Vercel CLI**
\`\`\`bash
npm i -g vercel
\`\`\`

2. **Configurar variables de entorno en Vercel**
\`\`\`bash
vercel env add DJANGO_SECRET_KEY
vercel env add DEBUG
\`\`\`

3. **Desplegar**
\`\`\`bash
vercel --prod
\`\`\`

### Limitaciones en Vercel

⚠️ **Importante**: Vercel tiene limitaciones para Django:
- Funciones serverless con timeout de 10s (plan gratuito)
- No hay persistencia de base de datos SQLite
- Tamaño máximo de función: 50MB

### Alternativas Recomendadas

Para producción con Django, considera:
- **Railway**: https://railway.app
- **Render**: https://render.com
- **PythonAnywhere**: https://www.pythonanywhere.com
- **Heroku**: https://www.heroku.com
- **DigitalOcean App Platform**: https://www.digitalocean.com/products/app-platform

## Ejemplo de Uso con Python

\`\`\`python
import requests
import json

# URL de la API
API_URL = "http://localhost:8000/api"

# Datos de ejemplo
data = {
    "data": [
        [0, "tcp", "http", "SF", 181, 5450, "0", 0, 0, 0, 0, "1", 0, 0, 0, 0, 0, 0, 0, 0, "0", "0", 8, 8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 9, 9, 1.0, 0.0, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0, "tcp", "http", "SF", 239, 486, "0", 0, 0, 0, 0, "1", 0, 0, 0, 0, 0, 0, 0, 0, "0", "0", 8, 8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 19, 19, 1.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "columns": ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
}

# Obtener información del dataset
response = requests.post(f"{API_URL}/dataset-info/", json=data)
print(json.dumps(response.json(), indent=2))

# Aplicar pipeline completo
response = requests.post(f"{API_URL}/apply-pipeline/", json={
    **data,
    "pipeline_type": "full"
})
print(json.dumps(response.json(), indent=2))
\`\`\`

## Estructura del Proyecto

\`\`\`
.
├── ml_api/                 # Configuración principal de Django
│   ├── __init__.py
│   ├── settings.py        # Configuración de Django
│   ├── urls.py            # URLs principales
│   └── wsgi.py            # WSGI para despliegue
├── dataset_api/           # Aplicación principal
│   ├── utils/
│   │   ├── data_loader.py    # Funciones de carga de datos
│   │   ├── transformers.py   # Transformadores personalizados
│   │   └── pipelines.py      # Pipelines de sklearn
│   ├── views.py           # Vistas de la API
│   └── urls.py            # URLs de la aplicación
├── requirements.txt       # Dependencias Python
├── vercel.json           # Configuración de Vercel
├── build_files.sh        # Script de build para Vercel
└── README.md             # Este archivo
\`\`\`

## Tecnologías Utilizadas

- **Django 4.2**: Framework web
- **Django REST Framework**: API REST
- **pandas**: Manipulación de datos
- **scikit-learn**: Machine Learning y preprocesamiento
- **numpy**: Operaciones numéricas
- **liac-arff**: Lectura de archivos ARFF

## Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT.

## Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.
