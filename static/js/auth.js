/**
 * Authentication & Password Visibility Controller (auth.js)
 */

(function () {
    'use strict';

    function togglePasswordVisibility(buttonId, inputId) {
        const btn = document.getElementById(buttonId);
        const input = document.getElementById(inputId);
        if (!btn || !input) return;

        const isPassword = input.type === 'password';
        input.type = isPassword ? 'text' : 'password';

        // Toggle SVG icon opacity or path if needed
        btn.style.color = isPassword ? '#10b981' : '';
    }

    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('.toggle-password').forEach(btn => {
            btn.addEventListener('click', function () {
                const targetId = this.dataset.target || (this.id === 'togglePassword' ? 'password' : '');
                if (targetId) {
                    togglePasswordVisibility(this.id, targetId);
                }
            });
        });
    });

    window.togglePasswordVisibility = togglePasswordVisibility;
})();
