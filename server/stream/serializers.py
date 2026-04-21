from .models import FaceMap
from rest_framework import serializers
import base64
import io
import cv2
import numpy as np
from PIL import Image

class FaceMapSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceMap
        fields = ['id', 'name', 'array', 'angles', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']


class FaceImageSerializer(serializers.Serializer):
    """Serializer for receiving face images in multiple angles."""
    name = serializers.CharField(max_length=100)
    frontal = serializers.CharField(required=True, help_text="Base64 encoded image")
    top = serializers.CharField(required=True, help_text="Base64 encoded image")
    bottom = serializers.CharField(required=True, help_text="Base64 encoded image")
    cw_45 = serializers.CharField(required=False, allow_blank=True)
    cw_90 = serializers.CharField(required=False, allow_blank=True)
    cw_135 = serializers.CharField(required=False, allow_blank=True)
    cw_225 = serializers.CharField(required=False, allow_blank=True)
    cw_270 = serializers.CharField(required=False, allow_blank=True)
    cw_315 = serializers.CharField(required=False, allow_blank=True)

    ANGLE_MAPPING = {
        'frontal': 'frontal',
        'top': 'top',
        'bottom': 'bottom',
        'cw_45': '45',
        'cw_90': '90',
        'cw_135': '135',
        'cw_225': '225',
        'cw_270': '270',
        'cw_315': '315',
    }

    @staticmethod
    def decode_image(base64_str: str):
        """Decode base64 image string to numpy array (OpenCV format)."""
        try:
            # Remove data URI scheme if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            # Convert PIL Image to OpenCV format (BGR)
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # RGB to BGR
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            return image_np
        except Exception as e:
            raise serializers.ValidationError(f"Failed to decode image: {str(e)}")

    def validate(self, data):
        """Validate that at least frontal, top, and bottom are provided."""
        if not all(data.get(key) for key in ['frontal', 'top', 'bottom']):
            raise serializers.ValidationError(
                "Frontal, top, and bottom face images are required."
            )
        return data


class RecognizeSerializer(serializers.Serializer):
    """Serializer for receiving a single face image for recognition."""
    image = serializers.CharField(required=True, help_text="Base64 encoded image")
    threshold = serializers.FloatField(required=False, default=0.4, min_value=0.0, max_value=1.0,
                                       help_text="Cosine distance threshold for matching (default 0.4)")

    @staticmethod
    def decode_image(base64_str: str):
        """Decode base64 image string to numpy array (OpenCV format)."""
        try:
            # Remove data URI scheme if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            # Convert PIL Image to OpenCV format (BGR)
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # RGB to BGR
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            return image_np
        except Exception as e:
            raise serializers.ValidationError(f"Failed to decode image: {str(e)}")