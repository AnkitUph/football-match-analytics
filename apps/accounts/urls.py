from django.urls import path
from . import views

app_name = "accounts"

urlpatterns = [
    path("register/", views.register, name="register"),
    path("login/", views.user_login, name="login"),
    path("verify-email/<uidb64>/<token>/",views.verify_email,name="verify_email",),
    path("forgot-password/", views.forgot_password, name="forgot_password"),
    path("reset-password/<uidb64>/<token>/", views.reset_password, name="reset_password"),
    path("change-password/", views.change_password, name="change_password"),
    path("logout/", views.user_logout, name="logout"),
]