import React from 'react';
import { motion } from 'framer-motion';
import {
    Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend,
    RadialLinearScale, PointElement, LineElement, Filler
} from 'chart.js';
import { Bar, Radar } from 'react-chartjs-2';
import { ArrowUpRight, TrendingUp } from 'lucide-react';

ChartJS.register(
    CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend,
    RadialLinearScale, PointElement, LineElement, Filler
);

const accentColors = {
    gat: '#0066cc',
    gatBg: 'rgba(0, 102, 204, 0.1)',
    others: '#e5e5ea',
    othersBg: 'transparent'
};

export default function Comparison() {
    const barData = {
        labels: ['GAT (Ours)', 'DBSCAN', 'GCN', 'SelfCheckGPT'],
        datasets: [
            {
                label: 'Macro F1 Score',
                data: [72.68, 38.64, 36.71, 28.53],
                backgroundColor: [accentColors.gat, accentColors.others, accentColors.others, accentColors.others],
                borderRadius: 12,
                barPercentage: 0.7,
            }
        ]
    };

    const barOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                padding: 16,
                cornerRadius: 12,
                titleFont: { family: 'Inter', size: 16, weight: '600' },
                bodyFont: { family: 'Inter', size: 15 },
                backgroundColor: 'rgba(29, 29, 31, 0.9)'
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false },
                border: { display: false },
                ticks: { font: { family: 'Inter', size: 14 }, color: '#86868b', padding: 15 }
            },
            x: {
                grid: { display: false },
                border: { display: false },
                ticks: { font: { family: 'Inter', size: 15, weight: '500' }, color: '#1d1d1f', padding: 10 }
            }
        }
    };

    const radarData = {
        labels: ['Accuracy', 'Macro F1', 'Binary Recall', 'Consistency'],
        datasets: [
            {
                label: 'GAT (Ours)',
                data: [86.48, 72.68, 95.13, 97.03],
                borderColor: accentColors.gat,
                backgroundColor: accentColors.gatBg,
                borderWidth: 3,
                pointBackgroundColor: accentColors.gat,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 6,
                pointHoverRadius: 8
            },
            {
                label: 'DBSCAN (Best Baseline)',
                data: [74.91, 38.64, 89.94, 60.0],
                borderColor: '#a1a1a6',
                backgroundColor: 'transparent',
                borderWidth: 2,
                borderDash: [5, 5],
                pointBackgroundColor: '#a1a1a6',
                pointRadius: 4
            }
        ]
    };

    const radarOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: { usePointStyle: true, padding: 30, font: { family: 'Inter', size: 15 }, color: '#1d1d1f' }
            },
            tooltip: {
                padding: 16, cornerRadius: 12, titleFont: { family: 'Inter', size: 16 }, backgroundColor: 'rgba(29, 29, 31, 0.9)'
            }
        },
        scales: {
            r: {
                beginAtZero: true,
                max: 100,
                ticks: { display: false },
                grid: { color: 'rgba(0,0,0,0.06)' },
                angleLines: { color: 'rgba(0,0,0,0.06)' },
                pointLabels: {
                    font: { family: 'Inter', size: 15, weight: '600' },
                    color: '#1d1d1f'
                }
            }
        }
    };

    return (
        <section id="comparison" className="section" style={{ background: '#ffffff', padding: '10rem 0' }}>
            <div className="container" style={{ maxWidth: '1400px' }}>

                {/* Section Header */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                    style={{ marginBottom: '6rem', maxWidth: '800px' }}
                >
                    <h2 style={{ fontSize: 'clamp(3rem, 5vw, 4rem)', letterSpacing: '-0.03em', color: '#1d1d1f', marginBottom: '1.5rem', lineHeight: 1.1 }}>
                        Performance <br /> Beyond Baselines.
                    </h2>
                    <p style={{ fontSize: '1.4rem', color: '#86868b', fontWeight: 400 }}>
                        Graph attention radically outperforms simple clustering and message passing by capturing the geometric density of truth.
                    </p>
                </motion.div>

                {/* Big Bar Chart Layout (Full Width feature) */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                    style={{
                        background: '#f5f5f7', borderRadius: '40px', padding: '4rem',
                        display: 'flex', flexDirection: 'column', gap: '3rem',
                        marginBottom: '4rem', overflow: 'hidden', position: 'relative'
                    }}
                >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '2rem' }}>
                        <div>
                            <h3 style={{ fontSize: '2rem', color: '#1d1d1f', letterSpacing: '-0.02em', marginBottom: '0.5rem' }}>Macro F1 Dominance</h3>
                            <p style={{ fontSize: '1.2rem', color: '#86868b' }}>F1 Score aggregates precision and recall robustly.</p>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: '#ffffff', padding: '1rem 2rem', borderRadius: '100px', boxShadow: '0 10px 30px rgba(0,0,0,0.03)' }}>
                            <TrendingUp color="#0066cc" size={28} />
                            <div>
                                <div style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#86868b', fontWeight: 600 }}>GAT Outperformance</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#1d1d1f' }}>+88.1% <span style={{ fontSize: '1rem', fontWeight: 500, color: '#86868b' }}>vs DBSCAN</span></div>
                            </div>
                        </div>
                    </div>

                    <div style={{ height: '500px', width: '100%' }}>
                        <Bar data={barData} options={barOptions} />
                    </div>
                </motion.div>

                {/* Split Layout for Radar & Insights */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: '4rem' }}>

                    {/* Radar Chart side */}
                    <motion.div
                        initial={{ opacity: 0, x: -40 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                        style={{
                            background: '#ffffff', borderRadius: '40px', padding: '4rem',
                            border: '1px solid rgba(0,0,0,0.06)',
                            display: 'flex', flexDirection: 'column'
                        }}
                    >
                        <h3 style={{ fontSize: '2rem', color: '#1d1d1f', letterSpacing: '-0.02em', marginBottom: '1rem', textAlign: 'center' }}>Multi-Metric Radar</h3>
                        <p style={{ fontSize: '1.1rem', color: '#86868b', textAlign: 'center', marginBottom: '3rem' }}>Comparing multi-dimensional performance against the best baseline.</p>
                        <div style={{ height: '450px', width: '100%', flex: 1 }}>
                            <Radar data={radarData} options={radarOptions} />
                        </div>
                    </motion.div>

                    {/* Insights Text side */}
                    <motion.div
                        initial={{ opacity: 0, x: 40 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
                        style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '2rem 0' }}
                    >
                        <h3 style={{ fontSize: '2.5rem', letterSpacing: '-0.03em', color: '#1d1d1f', marginBottom: '2rem' }}>
                            Why Graph Attention Wins.
                        </h3>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
                            <div>
                                <h4 style={{ fontSize: '1.3rem', color: '#1d1d1f', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                    <div style={{ width: '8px', height: '8px', background: '#0066cc', borderRadius: '50%' }} />
                                    Contextual Edge Weights
                                </h4>
                                <p style={{ fontSize: '1.15rem', color: '#86868b', lineHeight: 1.6 }}>
                                    Unlike GCN which treats all neighbors equally, GAT learns to heavily weight edges between semantically identical correct answers, creating dense "truth clusters".
                                </p>
                            </div>

                            <div>
                                <h4 style={{ fontSize: '1.3rem', color: '#1d1d1f', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                    <div style={{ width: '8px', height: '8px', background: '#0066cc', borderRadius: '50%' }} />
                                    Isolating Outliers
                                </h4>
                                <p style={{ fontSize: '1.15rem', color: '#86868b', lineHeight: 1.6 }}>
                                    Hallucinations often contradict each other. The attention mechanism assigns near-zero weights to edges connecting contradictory nodes, naturally isolating hallucinations as graph outliers.
                                </p>
                            </div>

                            <div>
                                <a href="#pipeline" style={{
                                    display: 'inline-flex', alignItems: 'center', gap: '8px',
                                    fontSize: '1.1rem', color: '#0066cc', fontWeight: 600, textDecoration: 'none',
                                    marginTop: '1rem'
                                }}>
                                    Review the Architecture <ArrowUpRight size={20} />
                                </a>
                            </div>
                        </div>
                    </motion.div>

                </div>

            </div>
        </section>
    );
}
