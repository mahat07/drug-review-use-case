# models.py
# from django.db import models
# from django.contrib.auth.models import User

# class Feedback(models.Model):
#     drugname = models.CharField(max_length=255)
#     condition = models.CharField(max_length=255)
#     review = models.TextField()
#     created_at = models.DateTimeField(auto_now_add=True)
#     user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link to User model


from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    drugname = models.CharField(max_length=100)
    condition = models.CharField(max_length=100)
    review = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f'{self.user.username} - {self.drugname}'




# from django.db import models
# from django.contrib.auth.models import User

# class Review(models.Model):
#     user = models.ForeignKey(User, on_delete=models.CASCADE)
#     drug_name = models.CharField(max_length=255)
#     condition = models.CharField(max_length=255)
#     review_text = models.TextField()
#     rating = models.FloatField()
#     date_posted = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return f"{self.drug_name} - {self.user.username}"

