from urllib import request

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
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

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            is_active=False,
        )

# Generate verification token
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        token = default_token_generator.make_token(user)

        verification_url = (
            f"http://{get_current_site(request).domain}"
            f"/accounts/verify-email/{uid}/{token}/"
        )

        html_message = render_to_string(
            "accounts/emails/verify_email.html",
            {
                "user": user,
                "verification_url": verification_url,
          },
        )

        email_message = EmailMultiAlternatives(
            subject="Verify your Football Analytics account",
            body=f"Verify your account: {verification_url}",
            to=[user.email],
        )
        email_message.attach_alternative(html_message, "text/html")
        email_message.send()

        messages.success(
            request,
            "Account created successfully. Please check your email to verify your account.",
        )

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

            if not user.is_active:
                messages.error(
                    request,
                    "Please verify your email before logging in."
                )
                return redirect("accounts:login")

            login(request, user)
            messages.success(request, "Login successful.")
            return redirect("dashboard")

        messages.error(request, "Invalid email or password.")

    return render(request, "accounts/login.html")


from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_str
from django.contrib.auth.tokens import default_token_generator


def verify_email(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)

    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()

        messages.success(
            request,
            "Your email has been verified. You can now log in."
        )

        return redirect("accounts:login")

    messages.error(
        request,
        "Verification link is invalid or has expired."
    )

    return redirect("accounts:login")