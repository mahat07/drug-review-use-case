# myproject/urls.py

from django.contrib import admin
from django.urls import path, include
from myapp.views import home
from django.views.generic.base import RedirectView
from django.contrib import admin
from django.urls import path, include
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),  # Includes default auth URLs
    path('', include('myapp.urls')),  # Include app-specific URLs
    path('myapp/', include('myapp.urls')),  # Include myapp URLs for registration and custom views
    # Feedback paths
    path('feedback/', include('myapp.urls')),
    path('feedback/thanks/', include('myapp.urls')),
    # Prediction paths
    path('predict_condition/', include('myapp.urls')),
    path('predict_rating/', include('myapp.urls')),
    path('predict_sentiment/', include('myapp.urls')),  # Fix the ordering
    # Exploration paths
    path('explore_drug/', include('myapp.urls')), 
    path('more_drug_info/', include('myapp.urls')),
    path('explore_condition/', include('myapp.urls')),
    path('more_condition_info/', include('myapp.urls')),
    # Additional paths
    path('download_report/', include('myapp.urls')),
    path('forecast_form/', include('myapp.urls')),
    path('forecast/', include('myapp.urls')),
    path('region/<str:region_name>/', include('myapp.urls')),
    path('forecast1_form/', include('myapp.urls')),
    path('forecast1/', include('myapp.urls')),
    path('region_forecast/<str:region_name>/', include('myapp.urls')),
    # Recent reviews
    path('recent_reviews/', include('myapp.urls')),  # Add a trailing slash for consistency
    path('hiv/', include('myapp.urls')),
    path('profile/',include('myapp.urls')),
]