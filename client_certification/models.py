from django.db import models

# Create your models here.
class ImageUpload(models.Model):
    title = models.CharField(max_length=100)
    imagefile = models.FileField(null=True)
    #imagefile = models.FileField(upload_to='imagefile/%Y/%m/%d', null=True)