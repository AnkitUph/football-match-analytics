from django.contrib import messages
from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect

User = get_user_model()


def register(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        # Validate required fields
        if not username or not email or not password or not confirm_password:
            messages.error(request, "All fields are required.")
            return redirect("accounts:register")

        # Passwords match
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect("accounts:register")

        # Username exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect("accounts:register")

        # Email exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return redirect("accounts:register")

        # Create user
        User.objects.create_user(
            username=username,
            email=email,
            password=password,
        )

        messages.success(request, "Account created successfully.")
        return redirect("accounts:login")

    return render(request, "accounts/register.html")


from django.http import HttpResponse

from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect




def user_login(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password")

        user = authenticate(
            request,
            username=email,   # Pass email here
            password=password,
        )

        if user is not None:
            login(request, user)
            messages.success(request, "Login successful.")
            return redirect("dashboard")

        messages.error(request, "Invalid email or password.")

    return render(request, "accounts/login.html")