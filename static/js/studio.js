/**
 * Tactical Operations & Football Match Analytics Studio
 * Global Shared Utilities (studio.js)
 */

(function () {
    'use strict';

    /**
     * Extracts CSRF token from document cookies.
     * @param {string} [name='csrftoken']
     * @returns {string}
     */
    function getCsrfToken(name = 'csrftoken') {
        const match = document.cookie.match(new RegExp('(^|;\\s*)(' + name + ')=([^;]*)'));
        return match ? decodeURIComponent(match[3]) : '';
    }

    /**
     * Displays a tactical floating toast notification.
     * @param {string} message
     * @param {'info'|'success'|'warning'|'error'} [type='info']
     * @param {number} [duration=3500]
     */
    function showToast(message, type = 'info', duration = 3500) {
        let container = document.getElementById('studio-toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'studio-toast-container';
            container.style.cssText = `
                position: fixed;
                bottom: 24px;
                right: 24px;
                z-index: 9999;
                display: flex;
                flex-direction: column;
                gap: 10px;
                pointer-events: none;
            `;
            document.body.appendChild(container);
        }

        const colors = {
            info: { bg: 'rgba(6, 182, 212, 0.92)', text: '#080c0e', border: '#06b6d4', icon: 'ℹ️' },
            success: { bg: 'rgba(16, 185, 129, 0.92)', text: '#080c0e', border: '#10b981', icon: '✓' },
            warning: { bg: 'rgba(245, 158, 11, 0.92)', text: '#080c0e', border: '#f59e0b', icon: '⚠️' },
            error: { bg: 'rgba(244, 63, 94, 0.92)', text: '#ffffff', border: '#f43f5e', icon: '✕' }
        };

        const config = colors[type] || colors.info;

        const toast = document.createElement('div');
        toast.style.cssText = `
            background: ${config.bg};
            color: ${config.text};
            border: 1px solid ${config.border};
            border-radius: 10px;
            padding: 12px 18px;
            font-size: 13.5px;
            font-weight: 600;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            backdrop-filter: blur(12px);
            display: flex;
            align-items: center;
            gap: 10px;
            pointer-events: auto;
            transform: translateY(20px);
            opacity: 0;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        `;

        toast.innerHTML = `<span>${config.icon}</span><span>${message}</span>`;
        container.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => {
            toast.style.transform = 'translateY(0)';
            toast.style.opacity = '1';
        });

        // Dismiss
        setTimeout(() => {
            toast.style.transform = 'translateY(-10px)';
            toast.style.opacity = '0';
            setTimeout(() => {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 250);
        }, duration);
    }

    /**
     * Formats duration in seconds to MM:SS or MM:SS.d
     */
    function formatTime(seconds, includeTenths = false) {
        if (isNaN(seconds) || seconds === null) return "00:00";
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        if (includeTenths) {
            const tenths = Math.floor((seconds % 1) * 10);
            return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${tenths}`;
        }
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    // Expose globals
    window.Studio = {
        getCsrfToken,
        showToast,
        formatTime
    };

    // Also support global getCookie for backwards compatibility
    window.getCookie = getCsrfToken;
    window.getCsrfToken = getCsrfToken;
})();
