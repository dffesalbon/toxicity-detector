from django.db import models

# Create your models here.


class Message(models.Model):

    input = models.CharField(max_length=200)
    #toxicity = models.CharField(max_length=24)

    def __str__(self) -> str:
        return self.input
