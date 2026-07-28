from django.urls import path
from . import views

app_name = "accounts"

urlpatterns = [
    path("register/", views.register, name="register"),
    path("login/", views.user_login, name="login"),
    path("verify-email/<uidb64>/<token>/",views.verify_email,name="verify_email",),
    
]