import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Loader2, CheckCircle2, AlertTriangle } from 'lucide-react';

const mockAnswers = [
    { id: 1, text: "The patient presents with consistent symptoms of pericardial effusion, likely secondary to the recent viral infection.", status: "correct" },
    { id: 2, text: "It is a clear case of myocardial infarction based on the elevated troponin levels and ST-segment elevation.", status: "hallucinated" },
    { id: 3, text: "Pericarditis with effusion is the primary diagnosis, supported by the friction rub heard on auscultation.", status: "correct" },
    { id: 4, text: "The symptoms perfectly match acute pulmonary embolism; immediate heparinization is required.", status: "hallucinated" },
    { id: 5, text: "Given the clear chest X-ray, pericardial effusion is the most logical conclusion matching the clinical picture.", status: "correct" },
];

export default function Demo() {
    const [query, setQuery] = useState('');
    const [isSearching, setIsSearching] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const [results, setResults] = useState([]);
    const [graphConfidence, setGraphConfidence] = useState(0);
    const [error, setError] = useState(null);
    const [isModelReady, setIsModelReady] = useState(false);
    const [useRealLLM, setUseRealLLM] = useState(false);

    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

    // Poll Health Check on Mount
    React.useEffect(() => {
        let interval;
        const checkHealth = async () => {
            try {
                const res = await fetch(`${API_URL}/health`);
                if (res.ok) {
                    setIsModelReady(true);
                    clearInterval(interval);
                }
            } catch (err) {
                // Keep polling
            }
        };

        checkHealth(); // initial check
        interval = setInterval(checkHealth, 2000);
        return () => clearInterval(interval);
    }, [API_URL]);

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!query || !isModelReady) return;
        setIsSearching(true);
        setShowResults(false);
        setError(null);

        try {
            const response = await fetch(`${API_URL}/detect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, use_real_llm: useRealLLM })
            });

            if (!response.ok) throw new Error('Failed to reach detection engine');

            const data = await response.json();
            setResults(data.results);
            setGraphConfidence(data.graph_confidence);
            setShowResults(true);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsSearching(false);
        }
    };

    // Helper to map the 4-class logic to visual styles
    const getStatusStyle = (status) => {
        switch (status) {
            case 'correct':
                return { bg: 'rgba(52,199,89,0.1)', text: '#2eaf4e', border: 'rgba(52,199,89,0.2)', label: 'Consistent truth' };
            case 'minor_hallucination':
                return { bg: 'rgba(255,149,0,0.1)', text: '#ff9500', border: 'rgba(255,149,0,0.2)', label: 'Minor Hallucination' };
            case 'moderate_hallucination':
                return { bg: 'rgba(255,100,0,0.1)', text: '#ff6400', border: 'rgba(255,100,0,0.2)', label: 'Moderate Hallucination' };
            case 'hallucinated':
                return { bg: 'rgba(255,59,48,0.1)', text: '#e6352b', border: 'rgba(255,59,48,0.2)', label: 'Complete Hallucination' };
            default:
                return { bg: '#f5f5f7', text: '#86868b', border: '#e5e5ea', label: 'Unknown' };
        }
    };

    return (
        <section id="demo" className="section" style={{ background: '#ffffff' }}>
            <div className="container" style={{ maxWidth: '800px' }}>
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ duration: 0.8 }}
                    style={{ textAlign: 'center', marginBottom: '4rem' }}
                >
                    <h2 style={{ fontSize: '2.5rem', color: '#1d1d1f', marginBottom: '1rem' }}>Live Detection Demo</h2>
                    <p style={{ fontSize: '1.2rem', color: '#86868b' }}>Test the GAT model's ability to isolate factual answers.</p>
                </motion.div>

                {/* Hybrid Demo Toggle */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    style={{ display: 'flex', justifyContent: 'center', marginBottom: '2rem' }}
                >
                    <div style={{
                        background: '#f5f5f7',
                        borderRadius: '30px',
                        padding: '4px',
                        display: 'flex',
                        position: 'relative',
                        boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.02)'
                    }}>
                        <motion.div
                            initial={false}
                            animate={{ x: useRealLLM ? '100%' : '0%' }}
                            transition={{ type: "spring", stiffness: 400, damping: 30 }}
                            style={{
                                position: 'absolute',
                                width: '50%',
                                height: 'calc(100% - 8px)',
                                background: '#ffffff',
                                borderRadius: '26px',
                                boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
                            }}
                        />
                        <button
                            type="button"
                            onClick={() => setUseRealLLM(false)}
                            style={{
                                flex: 1, padding: '10px 20px', border: 'none', background: 'transparent',
                                zIndex: 1, cursor: 'pointer',
                                color: !useRealLLM ? '#1d1d1f' : '#86868b',
                                fontWeight: !useRealLLM ? 600 : 500, fontSize: '0.95rem',
                                transition: 'color 0.3s ease'
                            }}
                        >
                            ⚡ Fast Demo
                        </button>
                        <button
                            type="button"
                            onClick={() => setUseRealLLM(true)}
                            style={{
                                flex: 1, padding: '10px 20px', border: 'none', background: 'transparent',
                                zIndex: 1, cursor: 'pointer',
                                color: useRealLLM ? '#1d1d1f' : '#86868b',
                                fontWeight: useRealLLM ? 600 : 500, fontSize: '0.95rem',
                                transition: 'color 0.3s ease'
                            }}
                        >
                            🧠 Local LLM Inference
                        </button>
                    </div>
                </motion.div>

                {/* Minimal Search Bar */}
                <motion.form
                    initial={{ opacity: 0, scale: 0.95 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    onSubmit={handleSearch}
                    style={{
                        position: 'relative',
                        background: '#f5f5f7',
                        borderRadius: '24px',
                        padding: '8px 8px 8px 24px',
                        display: 'flex',
                        alignItems: 'center',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.03)',
                        marginBottom: '3rem'
                    }}
                >
                    <Search size={22} color="#86868b" />
                    <input
                        type="text"
                        placeholder="Ask a biomedical question..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        style={{
                            border: 'none', background: 'transparent', flex: 1,
                            padding: '16px', fontSize: '1.1rem',
                            color: '#1d1d1f', outline: 'none'
                        }}
                    />
                    <button
                        type="submit"
                        disabled={isSearching || !query || !isModelReady}
                        style={{
                            background: query && isModelReady ? '#0066cc' : '#d1d1d6',
                            color: '#ffffff',
                            border: 'none',
                            borderRadius: '16px',
                            padding: '14px 28px',
                            fontSize: '1rem',
                            fontWeight: 600,
                            cursor: query && isModelReady ? 'pointer' : 'not-allowed',
                            transition: 'all 0.3s ease',
                            display: 'flex', alignItems: 'center', gap: '8px'
                        }}
                    >
                        {!isModelReady ? <Loader2 className="animate-spin" size={20} /> : (isSearching ? <Loader2 className="animate-spin" size={20} /> : 'Generate')}
                        {!isModelReady && <span style={{ fontSize: '0.9rem', marginLeft: '4px' }}>Waking up model...</span>}
                        {(isModelReady && isSearching && useRealLLM) && <span style={{ fontSize: '0.9rem', marginLeft: '4px' }}>Running LLM (Est. 1-3m)...</span>}
                    </button>
                </motion.form>

                {error && (
                    <div style={{ color: '#e6352b', textAlign: 'center', marginBottom: '2rem', padding: '1rem', background: 'rgba(255,59,48,0.1)', borderRadius: '12px' }}>
                        {error}. Ensure Flask backend is running on port 5000.
                    </div>
                )}

                {/* Results */}
                <AnimatePresence>
                    {showResults && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                            style={{ overflow: 'hidden' }}
                        >
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', paddingBottom: '2rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                    <h3 style={{ fontSize: '1.2rem', color: '#1d1d1f' }}>Live Graph Classification Results</h3>
                                    <span style={{ fontSize: '0.9rem', color: '#86868b' }}>Mean Model Confidence: <strong>{graphConfidence}%</strong></span>
                                </div>
                                <p style={{ fontSize: '0.9rem', color: '#a1a1a6', marginTop: '-1rem', marginBottom: '1rem' }}>
                                    *Note: Answer texts are simulated approximations for this local demo; the PyTorch embedding and graph classification are executing exactly as trained.
                                </p>

                                {results.map((ans, idx) => {
                                    const style = getStatusStyle(ans.status);
                                    return (
                                        <motion.div
                                            key={ans.id}
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: idx * 0.1, duration: 0.5 }}
                                            style={{
                                                background: '#ffffff',
                                                border: `1px solid ${style.border}`,
                                                borderRadius: '20px',
                                                padding: '1.5rem',
                                                display: 'flex',
                                                gap: '1.5rem',
                                                alignItems: 'flex-start',
                                                boxShadow: '0 8px 24px rgba(0,0,0,0.02)'
                                            }}
                                        >
                                            <div style={{
                                                minWidth: '40px', height: '40px', borderRadius: '50%',
                                                background: style.bg,
                                                color: style.text,
                                                display: 'flex', alignItems: 'center', justifyContent: 'center'
                                            }}>
                                                {ans.status === 'correct' ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}
                                            </div>
                                            <div style={{ flex: 1 }}>
                                                <p style={{ color: '#1d1d1f', fontSize: '1.05rem', lineHeight: 1.6, marginBottom: '0.75rem' }}>
                                                    {ans.text}
                                                </p>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                    <span style={{
                                                        display: 'inline-block',
                                                        padding: '4px 12px',
                                                        borderRadius: '20px',
                                                        fontSize: '0.8rem',
                                                        fontWeight: 600,
                                                        background: style.bg,
                                                        color: style.text,
                                                        textTransform: 'uppercase',
                                                        letterSpacing: '0.05em'
                                                    }}>
                                                        {style.label}
                                                    </span>
                                                    <span style={{ fontSize: '0.85rem', color: '#86868b', fontWeight: 500 }}>
                                                        Node Confidence: {ans.confidence}%
                                                    </span>
                                                </div>
                                            </div>
                                        </motion.div>
                                    );
                                })}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </section>
    );
}
