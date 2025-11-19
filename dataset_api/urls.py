"""
URL configuration for dataset_api app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.api_overview, name='api-overview'),
    path('split-dataset/', views.split_dataset, name='split-dataset'),
    path('preprocess/', views.preprocess_data, name='preprocess-data'),
    path('transform-categorical/', views.transform_categorical, name='transform-categorical'),
    path('scale-features/', views.scale_features, name='scale-features'),
    path('apply-pipeline/', views.apply_pipeline, name='apply-pipeline'),
    path('dataset-info/', views.dataset_info, name='dataset-info'),
]
