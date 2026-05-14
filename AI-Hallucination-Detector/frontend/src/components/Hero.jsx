import React from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

export default function Hero() {
    const { scrollY } = useScroll();
    const y1 = useTransform(scrollY, [0, 500], [0, 150]);
    const opacity = useTransform(scrollY, [0, 300], [1, 0]);

    return (
        <section id="hero" style={{
            minHeight: '85vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            overflow: 'hidden',
            backgroundColor: '#ffffff'
        }}>
            {/* Abstract Background Elements (Soft nodes/graph vibe) */}
            <motion.div style={{
                position: 'absolute', top: '10%', left: '15%',
                width: '300px', height: '300px',
                background: 'radial-gradient(circle, rgba(0,102,204,0.05) 0%, transparent 70%)',
                borderRadius: '50%', y: y1
            }} />
            <motion.div style={{
                position: 'absolute', bottom: '10%', right: '15%',
                width: '400px', height: '400px',
                background: 'radial-gradient(circle, rgba(52,199,89,0.05) 0%, transparent 70%)',
                borderRadius: '50%', y: useTransform(scrollY, [0, 500], [0, -100])
            }} />

            <motion.div
                className="container"
                style={{ textAlign: 'center', zIndex: 10, opacity, y: useTransform(scrollY, [0, 300], [0, 50]) }}
            >
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                >
                    <h1 style={{
                        fontSize: 'clamp(3rem, 6vw, 5.5rem)',
                        fontWeight: 800,
                        letterSpacing: '-0.04em',
                        lineHeight: 1.1,
                        color: '#1d1d1f',
                        marginBottom: '1.5rem',
                        maxWidth: '900px',
                        marginLeft: 'auto',
                        marginRight: 'auto'
                    }}>
                        Explainable AI <br />
                        <span style={{ color: '#0066cc' }}>Hallucination Detection.</span>
                    </h1>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
                >
                    <p style={{
                        fontSize: 'clamp(1.2rem, 2vw, 1.5rem)',
                        color: '#86868b',
                        maxWidth: '650px',
                        margin: '0 auto 3rem',
                        fontWeight: 400
                    }}>
                        Detecting factual inconsistencies using semantic graphs, attention networks, and a multi-scorer baseline.
                    </p>
                </motion.div>

                {/* Minimal metrics row */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.8, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
                    style={{
                        display: 'inline-flex',
                        gap: '3rem',
                        padding: '1.5rem 3rem',
                        background: 'rgba(255,255,255,0.7)',
                        backdropFilter: 'blur(20px)',
                        WebkitBackdropFilter: 'blur(20px)',
                        borderRadius: '24px',
                        border: '1px solid rgba(0,0,0,0.06)',
                        boxShadow: '0 20px 40px rgba(0,0,0,0.02)'
                    }}
                >
                    <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '2.5rem', fontWeight: 700, color: '#1d1d1f', letterSpacing: '-0.03em' }}>86.48%</div>
                        <div style={{ fontSize: '0.85rem', color: '#86868b', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Accuracy</div>
                    </div>
                    <div style={{ width: '1px', background: 'rgba(0,0,0,0.06)' }} />
                    <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '2.5rem', fontWeight: 700, color: '#0066cc', letterSpacing: '-0.03em' }}>95.13%</div>
                        <div style={{ fontSize: '0.85rem', color: '#86868b', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Detection Rate</div>
                    </div>
                </motion.div>
            </motion.div>
        </section>
    );
}
