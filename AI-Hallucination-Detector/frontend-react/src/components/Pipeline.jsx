import React, { useRef } from 'react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';

const steps = [
    {
        title: "Response Generation",
        text: "A biomedical question is sent to LLaMA-2 (13B), generating 5 answer candidates — 1 correct, 4 hallucinated.",
        icon: "1"
    },
    {
        title: "Semantic Embedding",
        text: "Each answer is converted into a 768-dimensional dense vector using a pre-trained clinical BERT model.",
        icon: "2"
    },
    {
        title: "Contrastive Learning",
        text: "Embeddings are refined via SimCLR, pushing hallucinated answers apart and pulling correct ones together.",
        icon: "3"
    },
    {
        title: "Graph Construction",
        text: "A semantic similarity graph connects answers using k-NN (Cosine Similarity > 0.85).",
        icon: "4"
    },
    {
        title: "Graph Attention (GAT)",
        text: "GAT applies varying attention weights across edges, isolating trustworthy semantic clusters.",
        icon: "5"
    },
    {
        title: "Hallucination Detection",
        text: "Final node classification categorizes answers as Correct, Partially Correct, or Hallucinated.",
        icon: "6"
    }
];

export default function Pipeline() {
    const containerRef = useRef(null);

    const { scrollYProgress } = useScroll({
        target: containerRef,
        offset: ["start center", "end center"]
    });

    const scrollPhysics = useSpring(scrollYProgress, {
        stiffness: 100,
        damping: 30,
        restDelta: 0.001
    });

    return (
        <section id="pipeline" className="section section-bg-off" ref={containerRef}>
            <div className="container" style={{ maxWidth: '800px' }}>
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                    style={{ textAlign: 'center', marginBottom: '5rem' }}
                >
                    <h2 style={{ fontSize: '2.5rem', color: '#1d1d1f', marginBottom: '1rem' }}>The GAT Pipeline</h2>
                    <p style={{ fontSize: '1.2rem', color: '#86868b' }}>A transparent, 6-stage process turning raw text into geometric certainty.</p>
                </motion.div>

                <div style={{ position: 'relative' }}>
                    {/* Animated Vertical Line */}
                    <div style={{
                        position: 'absolute', top: 0, bottom: 0, left: '28px', width: '2px',
                        background: 'rgba(0,0,0,0.05)', zIndex: 0
                    }} />

                    <motion.div style={{
                        position: 'absolute', top: 0, left: '28px', width: '2px',
                        background: 'linear-gradient(to bottom, #0066cc, #34c759)', zIndex: 1,
                        height: useTransform(scrollPhysics, [0, 1], ['0%', '100%']),
                        transformOrigin: 'top'
                    }} />

                    {/* Steps */}
                    {steps.map((step, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -30 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true, margin: "-100px" }}
                            transition={{ duration: 0.7, delay: idx * 0.1, ease: [0.16, 1, 0.3, 1] }}
                            style={{ display: 'flex', gap: '3rem', marginBottom: idx === steps.length - 1 ? 0 : '4rem', position: 'relative', zIndex: 2 }}
                        >
                            <div style={{
                                width: '58px', height: '58px', minWidth: '58px',
                                background: '#ffffff', borderRadius: '50%',
                                border: '1px solid rgba(0,0,0,0.06)',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                boxShadow: '0 8px 24px rgba(0,0,0,0.04)',
                                fontSize: '1.2rem', fontWeight: 600, color: '#1d1d1f',
                                boxSizing: 'border-box'
                            }}>
                                {step.icon}
                            </div>
                            <div style={{
                                flex: 1, padding: '2rem 2.5rem', background: '#ffffff',
                                borderRadius: '24px', border: '1px solid rgba(0,0,0,0.04)',
                                boxShadow: '0 20px 40px rgba(0,0,0,0.02)'
                            }}>
                                <h3 style={{ fontSize: '1.3rem', color: '#1d1d1f', marginBottom: '0.75rem' }}>{step.title}</h3>
                                <p style={{ fontSize: '1.05rem', color: '#86868b', lineHeight: 1.6 }}>{step.text}</p>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}
