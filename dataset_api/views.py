"""
API views for dataset processing endpoints.
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
import io
import json

from .utils.data_loader import train_val_test_split
from .utils.transformers import (
    DeleteNanRows, 
    CustomScaler, 
    CustomOneHotEncoding
)
from .utils.pipelines import create_numeric_pipeline, create_full_pipeline


@api_view(['GET'])
def api_overview(request):
    """
    API overview endpoint.
    """
    api_urls = {
        'overview': '/api/',
        'split_dataset': '/api/split-dataset/',
        'preprocess_data': '/api/preprocess/',
        'transform_categorical': '/api/transform-categorical/',
        'scale_features': '/api/scale-features/',
        'apply_pipeline': '/api/apply-pipeline/',
        'dataset_info': '/api/dataset-info/',
    }
    return Response({
        'message': 'NSL-KDD Dataset Processing API',
        'endpoints': api_urls,
        'description': 'API para procesamiento de datasets con transformadores y pipelines personalizados'
    })


@api_view(['POST'])
def split_dataset(request):
    """
    Split dataset into train, validation, and test sets.
    
    Expected JSON body:
    {
        "data": [...],  # List of records
        "columns": [...],  # List of column names
        "stratify_column": "protocol_type",  # Optional
        "random_state": 42,  # Optional
        "shuffle": true  # Optional
    }
    """
    try:
        data = request.data.get('data')
        columns = request.data.get('columns')
        stratify_column = request.data.get('stratify_column', None)
        random_state = request.data.get('random_state', 42)
        shuffle = request.data.get('shuffle', True)
        
        if not data or not columns:
            return Response(
                {'error': 'Missing required fields: data and columns'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Split dataset
        train_set, val_set, test_set = train_val_test_split(
            df, 
            rstate=random_state, 
            shuffle=shuffle, 
            stratify=stratify_column
        )
        
        return Response({
            'message': 'Dataset split successfully',
            'train_size': len(train_set),
            'validation_size': len(val_set),
            'test_size': len(test_set),
            'train_sample': train_set.head(5).to_dict('records'),
            'validation_sample': val_set.head(5).to_dict('records'),
            'test_sample': test_set.head(5).to_dict('records'),
        })
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def preprocess_data(request):
    """
    Apply preprocessing transformations to data.
    
    Expected JSON body:
    {
        "data": [...],
        "columns": [...],
        "remove_nan": true,  # Optional
        "impute_strategy": "median"  # Optional: "mean", "median", "most_frequent"
    }
    """
    try:
        data = request.data.get('data')
        columns = request.data.get('columns')
        remove_nan = request.data.get('remove_nan', False)
        impute_strategy = request.data.get('impute_strategy', 'median')
        
        if not data or not columns:
            return Response(
                {'error': 'Missing required fields: data and columns'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Get info before preprocessing
        null_counts_before = df.isnull().sum().to_dict()
        
        # Apply transformations
        if remove_nan:
            transformer = DeleteNanRows()
            df = transformer.fit_transform(df)
        else:
            # Apply imputation to numeric columns
            from sklearn.impute import SimpleImputer
            numeric_cols = df.select_dtypes(exclude=['object']).columns
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy=impute_strategy)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        null_counts_after = df.isnull().sum().to_dict()
        
        return Response({
            'message': 'Data preprocessed successfully',
            'rows_before': len(data),
            'rows_after': len(df),
            'null_counts_before': null_counts_before,
            'null_counts_after': null_counts_after,
            'processed_data': df.to_dict('records'),
            'sample': df.head(10).to_dict('records'),
        })
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def transform_categorical(request):
    """
    Transform categorical features using One-Hot Encoding.
    
    Expected JSON body:
    {
        "data": [...],
        "columns": [...],
        "encoding_type": "onehot"  # "onehot" or "ordinal"
    }
    """
    try:
        data = request.data.get('data')
        columns = request.data.get('columns')
        encoding_type = request.data.get('encoding_type', 'onehot')
        
        if not data or not columns:
            return Response(
                {'error': 'Missing required fields: data and columns'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Get categorical columns
        cat_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if encoding_type == 'onehot':
            transformer = CustomOneHotEncoding()
            df_transformed = transformer.fit_transform(df)
        else:
            # Ordinal encoding
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder()
            df_transformed = df.copy()
            if len(cat_columns) > 0:
                df_transformed[cat_columns] = encoder.fit_transform(df[cat_columns])
        
        return Response({
            'message': 'Categorical features transformed successfully',
            'encoding_type': encoding_type,
            'original_columns': columns,
            'transformed_columns': df_transformed.columns.tolist(),
            'categorical_columns_found': cat_columns,
            'sample': df_transformed.head(10).to_dict('records'),
        })
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def scale_features(request):
    """
    Scale numeric features using RobustScaler.
    
    Expected JSON body:
    {
        "data": [...],
        "columns": [...],
        "features_to_scale": ["src_bytes", "dst_bytes"]  # Optional
    }
    """
    try:
        data = request.data.get('data')
        columns = request.data.get('columns')
        features_to_scale = request.data.get('features_to_scale', None)
        
        if not data or not columns:
            return Response(
                {'error': 'Missing required fields: data and columns'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Determine features to scale
        if features_to_scale is None:
            features_to_scale = df.select_dtypes(exclude=['object']).columns.tolist()
        
        # Apply scaling
        scaler = CustomScaler(attributes=features_to_scale)
        df_scaled = scaler.fit_transform(df)
        
        return Response({
            'message': 'Features scaled successfully',
            'scaled_features': features_to_scale,
            'scaler_type': 'RobustScaler',
            'sample_before': df.head(5).to_dict('records'),
            'sample_after': df_scaled.head(5).to_dict('records'),
        })
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def apply_pipeline(request):
    """
    Apply full preprocessing pipeline to data.
    
    Expected JSON body:
    {
        "data": [...],
        "columns": [...],
        "pipeline_type": "full"  # "numeric" or "full"
    }
    """
    try:
        data = request.data.get('data')
        columns = request.data.get('columns')
        pipeline_type = request.data.get('pipeline_type', 'full')
        
        if not data or not columns:
            return Response(
                {'error': 'Missing required fields: data and columns'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        if pipeline_type == 'numeric':
            # Apply numeric pipeline only
            numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()
            pipeline = create_numeric_pipeline()
            df_transformed = pipeline.fit_transform(df[numeric_cols])
            df_transformed = pd.DataFrame(
                df_transformed, 
                columns=numeric_cols
            )
        else:
            # Apply full pipeline
            pipeline = create_full_pipeline(df)
            df_transformed = pipeline.fit_transform(df)
            
            # Get feature names
            feature_names = []
            if hasattr(pipeline, 'get_feature_names_out'):
                feature_names = pipeline.get_feature_names_out().tolist()
            
            df_transformed = pd.DataFrame(
                df_transformed,
                columns=feature_names if feature_names else None
            )
        
        return Response({
            'message': 'Pipeline applied successfully',
            'pipeline_type': pipeline_type,
            'original_shape': df.shape,
            'transformed_shape': df_transformed.shape,
            'transformed_columns': df_transformed.columns.tolist() if hasattr(df_transformed, 'columns') else [],
            'sample': df_transformed.head(10).to_dict('records') if hasattr(df_transformed, 'to_dict') else df_transformed[:10].tolist(),
        })
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def dataset_info(request):
    """
    Get information about the dataset.
    
    Expected JSON body:
    {
        "data": [...],
        "columns": [...]
    }
    """
    try:
        data = request.data.get('data')
        columns = request.data.get('columns')
        
        if not data or not columns:
            return Response(
                {'error': 'Missing required fields: data and columns'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Get dataset info
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(exclude=['object']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'sample': df.head(10).to_dict('records'),
        }
        
        # Add statistics for numeric columns
        numeric_stats = {}
        for col in info['numeric_columns']:
            numeric_stats[col] = {
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
            }
        
        info['numeric_statistics'] = numeric_stats
        
        # Add value counts for categorical columns
        categorical_stats = {}
        for col in info['categorical_columns']:
            categorical_stats[col] = df[col].value_counts().head(10).to_dict()
        
        info['categorical_statistics'] = categorical_stats
        
        return Response(info)
        
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
