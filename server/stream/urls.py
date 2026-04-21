from django.urls import path, include
from rest_framework import routers
from .views import Register, Recognize, Stream

router = routers.DefaultRouter()
router.register(r'register', Register, basename='register')
router.register(r'recognize', Recognize, basename='recognize')
router.register(r'stream', Stream, basename='stream')

urlpatterns = [
    path('', include(router.urls)),
]
