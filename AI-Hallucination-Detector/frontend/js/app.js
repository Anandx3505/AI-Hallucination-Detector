/**
 * App.js — Main controller
 * Handles: scroll animations, stat counters, intersection observer
 */

// ---- Animated counter ----
function animateCounter(element, target, suffix = '', duration = 2000) {
    let start = 0;
    const step = target / (duration / 16);
    const isFloat = target % 1 !== 0;

    function update() {
        start += step;
        if (start >= target) {
            start = target;
            element.textContent = (isFloat ? start.toFixed(2) : Math.round(start)) + suffix;
            return;
        }
        element.textContent = (isFloat ? start.toFixed(2) : Math.round(start)) + suffix;
        requestAnimationFrame(update);
    }
    update();
}

// ---- Stat counters (trigger once) ----
let statsCounted = false;
function triggerStats() {
    if (statsCounted) return;
    statsCounted = true;

    animateCounter(document.getElementById('stat-acc'), 86.48, '%');
    animateCounter(document.getElementById('stat-recall'), 95.13, '%');
    animateCounter(document.getElementById('stat-f1'), 0.7268, '', 2000);
    animateCounter(document.getElementById('stat-ofa'), 97.03, '%');
}

// ---- Intersection Observer for scroll animations ----
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');

                // Trigger stat counters when hero enters
                if (entry.target.closest('.hero')) {
                    triggerStats();
                }
            }
        });
    }, { threshold: 0.15, rootMargin: '0px 0px -50px 0px' });

    // Observe pipeline steps
    document.querySelectorAll('.pipeline-step').forEach(el => observer.observe(el));

    // Observe fade-in elements
    document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

    // Observe stat items
    document.querySelectorAll('.stat-item').forEach(el => {
        observer.observe(el);
        el.classList.add('fade-in');
    });

    // Observe hero
    const hero = document.querySelector('.hero-content');
    if (hero) observer.observe(hero);
}

// ---- Navbar scroll effect ----
function initNavbar() {
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(10, 14, 39, 0.95)';
            navbar.style.padding = '10px 0';
        } else {
            navbar.style.background = 'rgba(10, 14, 39, 0.85)';
            navbar.style.padding = '16px 0';
        }
    });
}

// ---- Smooth scroll for nav links ----
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ---- Init all ----
document.addEventListener('DOMContentLoaded', () => {
    initScrollAnimations();
    initNavbar();
    initSmoothScroll();

    // Trigger stats on load (since hero is visible)
    setTimeout(triggerStats, 500);
});
