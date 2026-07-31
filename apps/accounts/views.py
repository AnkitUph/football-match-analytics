from django.contrib import messages
from django.contrib.auth import (
    authenticate,
    get_user_model,
    login,
    logout,
    update_session_auth_hash,
)
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth.tokens import PasswordResetTokenGenerator, default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.core.exceptions import ValidationError
from django.core.mail import EmailMultiAlternatives
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode

User = get_user_model()


class EmailVerificationTokenGenerator(PasswordResetTokenGenerator):
    """
    Separate salt + hash payload from the password-reset token generator so a
    verification link can never be replayed to reset a password (and vice
    versa). Including is_active means the token is invalidated the moment
    the account is verified.
    """
    key_salt = "accounts.EmailVerificationTokenGenerator"

    def _make_hash_value(self, user, timestamp):
        return f"{user.pk}{timestamp}{user.is_active}"


email_verification_token = EmailVerificationTokenGenerator()


def _build_absolute_url(request, path):
    scheme = "https" if request.is_secure() else "http"
    return f"{scheme}://{get_current_site(request).domain}{path}"


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

        # Password strength (length, common password, similarity to user attrs, etc.)
        try:
            validate_password(password)
        except ValidationError as e:
            for msg in e.messages:
                messages.error(request, msg)
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

        # Generate verification token (distinct from password-reset tokens)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        token = email_verification_token.make_token(user)

        verification_url = _build_absolute_url(
            request, f"/accounts/verify-email/{uid}/{token}/"
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

        try:
            email_message.send()
        except Exception:
            # Don't leave an orphaned, unverifiable account behind
            user.delete()
            messages.error(
                request,
                "We couldn't send the verification email. Please try registering again.",
            )
            return redirect("accounts:register")

        messages.success(
            request,
            "Account created successfully. Please check your email to verify your account.",
        )
        return redirect("accounts:login")

    return render(request, "accounts/register.html")


def user_login(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password")

        user = authenticate(
            request,
            username=email,  # ModelBackend looks this up via USERNAME_FIELD (email)
            password=password,
        )

        if user is not None:
            if not user.is_active:
                messages.error(
                    request,
                    "Please verify your email before logging in.",
                )
                return redirect("accounts:login")

            login(request, user)
            messages.success(request, "Login successful.")
            return redirect("dashboard:index")

        messages.error(request, "Invalid email or password.")

    return render(request, "accounts/login.html")


def verify_email(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user and email_verification_token.check_token(user, token):
        user.is_active = True
        user.save()

        messages.success(
            request,
            "Your email has been verified. You can now log in.",
        )
        return redirect("accounts:login")

    messages.error(
        request,
        "Verification link is invalid or has expired.",
    )
    return redirect("accounts:login")


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

            reset_url = _build_absolute_url(
                request, f"/accounts/reset-password/{uid}/{token}/"
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

            try:
                email_message.send()
            except Exception:
                # Fail silently to the user (same as "email not found") so we
                # don't leak account existence via error behavior, but this
                # is a good spot to log the exception for yourself.
                pass

        messages.success(
            request,
            "If an account with that email exists, a password reset link has been sent.",
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
            "Your password has been reset successfully. You can now log in.",
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


def user_logout(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect("accounts:login")


 
@login_required
def profile(request):
    user = request.user
 
    if request.method == "POST":
        first_name = request.POST.get("first_name", "").strip()
        last_name = request.POST.get("last_name", "").strip()
        email = request.POST.get("email", "").strip().lower()
        profile_picture = request.FILES.get("profile_picture")
 
        if not email:
            messages.error(request, "Email is required.")
            return redirect("accounts:profile")
 
        # Only complain if a *different* user already owns this email
        if User.objects.exclude(pk=user.pk).filter(email=email).exists():
            messages.error(request, "That email is already in use.")
            return redirect("accounts:profile")
 
        user.first_name = first_name
        user.last_name = last_name
        user.email = email
 
        if profile_picture:
            user.profile_picture = profile_picture
 
        user.save()
 
        messages.success(request, "Your profile has been updated.")
        return redirect("accounts:profile")
 
    return render(request, "accounts/profile.html", {"profile_user": user})
 