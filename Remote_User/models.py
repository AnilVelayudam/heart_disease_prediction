from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class heart_disease_model(models.Model):


    age=models.CharField(max_length=300)
    sex=models.CharField(max_length=300)
    cp=models.CharField(max_length=300)
    trestbps=models.CharField(max_length=300)
    chol=models.CharField(max_length=300)
    fbs=models.CharField(max_length=300)
    resecg=models.CharField(max_length=300)
    thalach=models.CharField(max_length=300)
    exang=models.CharField(max_length=300)
    oldpeak=models.CharField(max_length=300)
    slope=models.CharField(max_length=300)
    ca=models.CharField(max_length=300)
    thal=models.CharField(max_length=300)
    target=models.CharField(max_length=300)



class heart_disease_prediction_model(models.Model):


    age=models.CharField(max_length=300)
    sex=models.CharField(max_length=300)
    cp=models.CharField(max_length=300)
    trestbps=models.CharField(max_length=300)
    chol=models.CharField(max_length=300)
    fbs=models.CharField(max_length=300)
    resecg=models.CharField(max_length=300)
    thalach=models.CharField(max_length=300)
    exang=models.CharField(max_length=300)
    oldpeak=models.CharField(max_length=300)
    slope=models.CharField(max_length=300)
    ca=models.CharField(max_length=300)
    thal=models.CharField(max_length=300)
    target=models.CharField(max_length=300)
    prediction=models.CharField(max_length=300)



class detection_results_model(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio_model(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



