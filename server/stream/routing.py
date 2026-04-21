"""
routing.py — Django Channels WebSocket URL routing.

Add this to your project's asgi.py:

    from channels.routing import ProtocolTypeRouter, URLRouter
    from channels.auth import AuthMiddlewareStack
    from your_app import routing

    application = ProtocolTypeRouter({
        "http":      get_asgi_application(),
        "websocket": AuthMiddlewareStack(
            URLRouter(routing.websocket_urlpatterns)
        ),
    })

The client connects at:
    ws://<host>/ws/stream/
    ws://<host>/ws/stream/?rtsp_url=rtsp://user:pass@host/path
"""

from django.urls import re_path
from .consumers import RTSPProxyConsumer

websocket_urlpatterns = [
    re_path(r"^ws/stream/$", RTSPProxyConsumer.as_asgi()),
]