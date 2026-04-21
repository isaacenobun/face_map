from django.db import models

# Create your models here.
class FaceMap(models.Model):
    ANGLE_CHOICES = [
        ('frontal', 'Frontal'),
        ('top', 'Top (Clockwise 0 deg)'),
        ('bottom', 'Bottom (Clockwise 180 deg)'),
        ('45', 'Clockwise 45 deg'),
        ('90', 'Clockwise 90 deg'),
        ('135', 'Clockwise 135 deg'),
        ('225', 'Clockwise 225 deg'),
        ('270', 'Clockwise 270 deg'),
        ('315', 'Clockwise 315 deg'),
    ]
    
    name = models.CharField(max_length=100, unique=True)  # Unique constraint
    array = models.JSONField(default=list)  # 512-d averaged embedding
    angles = models.JSONField(default=dict)  # {angle: [512-d embedding]}
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name