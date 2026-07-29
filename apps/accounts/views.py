import token
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
            return redirect("dashboard:index")

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

from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError


def forgot_password(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            user = None

        # Always show the same message, whether or not the email exists.
        # This prevents attackers from using this form to discover which
        # emails are registered.
        if user is not None:
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)

            reset_url = (
                f"http://{get_current_site(request).domain}"
                f"/accounts/reset-password/{uid}/{token}/"
            )

            html_message = render_to_string(
                "accounts/emails/reset_password.html",
                {
                    "user": user,
                    "reset_url": reset_url,
                },
            )

            email_message = EmailMultiAlternatives(
                subject="Reset your Football Analytics password",
                body=f"Reset your password: {reset_url}",
                to=[user.email],
            )
            email_message.attach_alternative(html_message, "text/html")
            email_message.send()

        messages.success(
            request,
            "If an account with that email exists, a password reset link has been sent."
        )
        return redirect("accounts:login")

    return render(request, "accounts/forgot_password.html")


def reset_password(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    valid_link = user is not None and default_token_generator.check_token(user, token)

    if not valid_link:
        messages.error(request, "This password reset link is invalid or has expired.")
        return redirect("accounts:forgot_password")

    if request.method == "POST":
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        if not password or not confirm_password:
            messages.error(request, "All fields are required.")
            return redirect("accounts:reset_password", uidb64=uidb64, token=token)

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect("accounts:reset_password", uidb64=uidb64, token=token)

        try:
            validate_password(password, user=user)
        except ValidationError as e:
            for err in e.messages:
                messages.error(request, err)
            return redirect("accounts:reset_password", uidb64=uidb64, token=token)

        user.set_password(password)
        user.save()

        messages.success(
            request,
            "Your password has been reset successfully. You can now log in."
        )
        return redirect("accounts:login")

    return render(
        request,
        "accounts/reset_password.html",
        {"uidb64": uidb64, "token": token},
    )

@login_required
def change_password(request):
    if request.method == "POST":
        old_password = request.POST.get("old_password")
        new_password = request.POST.get("new_password")
        confirm_password = request.POST.get("confirm_password")

        if not old_password or not new_password or not confirm_password:
            messages.error(request, "All fields are required.")
            return redirect("accounts:change_password")

        if not request.user.check_password(old_password):
            messages.error(request, "Old password is incorrect.")
            return redirect("accounts:change_password")

        if new_password != confirm_password:
            messages.error(request, "New passwords do not match.")
            return redirect("accounts:change_password")

        try:
            validate_password(new_password, user=request.user)
        except ValidationError as e:
            for err in e.messages:
                messages.error(request, err)
            return redirect("accounts:change_password")

        request.user.set_password(new_password)
        request.user.save()

        # Keeps the user logged in after password change instead of
        # invalidating their current session hash.
        update_session_auth_hash(request, request.user)

        messages.success(request, "Your password has been changed successfully.")
        return redirect("accounts:change_password")

    return render(request, "accounts/change_password.html")


from django.contrib.auth import logout


def user_logout(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect("accounts:login")