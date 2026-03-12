import React from 'react';
import { motion } from 'framer-motion';

const team = [
    { name: "Arpita Rani", roll: "221030055", avatar: "A", color: "#0066cc" },
    { name: "Anand Chaudhary", roll: "221030123", avatar: "A", color: "#34c759" },
    { name: "Rishal Rana", roll: "221030004", avatar: "R", color: "#ff3b30" },
    { name: "Arnav Sharma", roll: "221030059", avatar: "A", color: "#5ac8fa" }
];

export default function Team() {
    return (
        <section id="team" className="section section-bg-off">
            <div className="container">
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ duration: 0.8 }}
                    style={{ textAlign: 'center', marginBottom: '4rem' }}
                >
                    <h2 style={{ fontSize: '2.5rem', color: '#1d1d1f', marginBottom: '1rem' }}>The Research Team</h2>
                    <p style={{ fontSize: '1.1rem', color: '#86868b' }}>Department of Computer Science & Engineering, JUIT Waknaghat</p>
                </motion.div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '2rem', marginBottom: '4rem' }}>
                    {team.map((member, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: idx * 0.1, duration: 0.6 }}
                            style={{
                                background: '#ffffff',
                                padding: '2.5rem 1.5rem',
                                borderRadius: '32px',
                                textAlign: 'center',
                                boxShadow: '0 8px 30px rgba(0,0,0,0.02)',
                                border: '1px solid rgba(0,0,0,0.02)'
                            }}
                        >
                            <div style={{
                                width: '80px', height: '80px', borderRadius: '50%',
                                background: `rgba(${parseInt(member.color.slice(1, 3), 16)}, ${parseInt(member.color.slice(3, 5), 16)}, ${parseInt(member.color.slice(5, 7), 16)}, 0.1)`,
                                color: member.color,
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                margin: '0 auto 1.5rem',
                                fontSize: '1.8rem', fontWeight: 700
                            }}>
                                {member.avatar}
                            </div>
                            <h4 style={{ fontSize: '1.1rem', color: '#1d1d1f', marginBottom: '0.25rem' }}>{member.name}</h4>
                            <p style={{ fontSize: '0.9rem', color: '#86868b', fontFamily: 'monospace' }}>{member.roll}</p>
                        </motion.div>
                    ))}
                </div>

                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.6 }}
                    style={{
                        maxWidth: '500px', margin: '0 auto', background: '#ffffff',
                        padding: '2rem', borderRadius: '24px', textAlign: 'center',
                        boxShadow: '0 8px 30px rgba(0,0,0,0.02)',
                        border: '1px solid rgba(0,0,0,0.02)'
                    }}
                >
                    <span style={{ display: 'block', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#86868b', marginBottom: '0.5rem' }}>
                        Supervisor
                    </span>
                    <h4 style={{ fontSize: '1.2rem', color: '#1d1d1f' }}>Prof. Dr. Vivek Kumar Sehgal</h4>
                </motion.div>
            </div>
        </section>
    );
}
