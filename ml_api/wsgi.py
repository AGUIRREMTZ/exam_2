"""
WSGI config for ml_api project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_api.settings')

application = get_wsgi_application()

# Vercel needs this
app = application
