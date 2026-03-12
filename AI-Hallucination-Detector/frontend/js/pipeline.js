/**
 * Pipeline Animations — animated visualizations for each GAT pipeline step
 */

// ---- Hero Background: Animated Graph ----
function initHeroCanvas() {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const nodes = [];
    const nodeCount = 60;

    for (let i = 0; i < nodeCount; i++) {
        nodes.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            r: 2 + Math.random() * 3,
            color: Math.random() > 0.3 ? '#00d4ff' : '#00ff88',
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw edges
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.strokeStyle = `rgba(0, 212, 255, ${0.15 * (1 - dist / 150)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        // Draw nodes
        for (const node of nodes) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
            ctx.fillStyle = node.color;
            ctx.globalAlpha = 0.6;
            ctx.fill();
            ctx.globalAlpha = 1;

            // Move
            node.x += node.vx;
            node.y += node.vy;
            if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
            if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
        }

        requestAnimationFrame(draw);
    }
    draw();
}

// ---- Step 1: Answer cards appearing ----
function initStep1Canvas() {
    const canvas = document.getElementById('step1-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = 120;

    let t = 0;
    const labels = ['Answer 1 ✓', 'Answer 2', 'Answer 3', 'Answer 4', 'Answer 5'];
    const colors = ['#00ff88', '#ff4757', '#ff4757', '#ff4757', '#ff4757'];

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const cardW = Math.min(120, (canvas.width - 60) / 5);
        const gap = (canvas.width - cardW * 5) / 6;

        for (let i = 0; i < 5; i++) {
            const delay = i * 15;
            const progress = Math.min(1, Math.max(0, (t - delay) / 30));
            const x = gap + i * (cardW + gap);
            const y = 15 + (1 - progress) * 30;
            const alpha = progress;

            ctx.globalAlpha = alpha;
            ctx.fillStyle = 'rgba(255,255,255,0.05)';
            ctx.strokeStyle = colors[i] + '80';
            ctx.lineWidth = 1;

            const r = 6;
            ctx.beginPath();
            ctx.roundRect(x, y, cardW, 70, r);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = colors[i];
            ctx.font = '11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(labels[i], x + cardW / 2, y + 40);
            ctx.globalAlpha = 1;
        }

        t = (t + 0.5) % 120;
        requestAnimationFrame(draw);
    }
    draw();
}

// ---- Step 2: Dots appearing in embedding space ----
function initStep2Canvas() {
    const canvas = document.getElementById('step2-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = 120;

    let t = 0;
    const points = [
        { x: 0.2, y: 0.4, c: '#00ff88' },
        { x: 0.7, y: 0.3, c: '#ff4757' },
        { x: 0.5, y: 0.7, c: '#ff4757' },
        { x: 0.3, y: 0.6, c: '#ff4757' },
        { x: 0.8, y: 0.8, c: '#ff4757' },
    ];

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Grid
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i < 10; i++) {
            ctx.beginPath();
            ctx.moveTo(i * canvas.width / 10, 0);
            ctx.lineTo(i * canvas.width / 10, canvas.height);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, i * canvas.height / 10);
            ctx.lineTo(canvas.width, i * canvas.height / 10);
            ctx.stroke();
        }

        // Label
        ctx.fillStyle = 'rgba(255,255,255,0.15)';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillText('768-dim → 128-dim', 10, canvas.height - 8);

        // Points
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const wobble = Math.sin(t * 0.03 + i) * 5;
            const x = p.x * canvas.width;
            const y = p.y * canvas.height + wobble;

            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fillStyle = p.c;
            ctx.globalAlpha = 0.8;
            ctx.fill();

            // Glow
            ctx.beginPath();
            ctx.arc(x, y, 12, 0, Math.PI * 2);
            ctx.fillStyle = p.c;
            ctx.globalAlpha = 0.15;
            ctx.fill();
            ctx.globalAlpha = 1;
        }

        t++;
        requestAnimationFrame(draw);
    }
    draw();
}

// ---- Step 3: Contrastive learning — push/pull ----
function initStep3Canvas() {
    const canvas = document.getElementById('step3-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = 120;

    let t = 0;
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;

        // Before → After animation
        const phase = (Math.sin(t * 0.02) + 1) / 2; // 0 to 1 (scattered → clustered)

        // Green cluster (correct)
        const gBase = { x: cx - 80, y: cy };
        const greens = [
            { ox: -30, oy: -20 }, { ox: 10, oy: 15 }, { ox: -15, oy: 25 }
        ];
        for (const g of greens) {
            const x = gBase.x + g.ox * (1 - phase * 0.6);
            const y = gBase.y + g.oy * (1 - phase * 0.6);
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#00ff88';
            ctx.globalAlpha = 0.8;
            ctx.fill();
            ctx.globalAlpha = 1;
        }

        // Red cluster (hallucinated) — pushed apart
        const rBase = { x: cx + 80, y: cy };
        const reds = [
            { ox: 20, oy: -25 }, { ox: -15, oy: 20 }, { ox: 30, oy: 10 }, { ox: -25, oy: -10 }
        ];
        for (const r of reds) {
            const x = rBase.x + r.ox * (0.6 + phase * 0.8);
            const y = rBase.y + r.oy * (0.6 + phase * 0.8);
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#ff4757';
            ctx.globalAlpha = 0.8;
            ctx.fill();
            ctx.globalAlpha = 1;
        }

        // Arrow labels
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Same class → PULL', cx - 80, 15);
        ctx.fillText('Diff class → PUSH', cx + 80, 15);

        t++;
        requestAnimationFrame(draw);
    }
    draw();
}

// ---- Step 4: Graph construction ----
function initStep4Canvas() {
    const canvas = document.getElementById('step4-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = 120;

    let t = 0;
    const nodes = [
        { x: 0.15, y: 0.3, c: '#00ff88' },
        { x: 0.25, y: 0.7, c: '#00ff88' },
        { x: 0.35, y: 0.4, c: '#00ff88' },
        { x: 0.6, y: 0.2, c: '#ff4757' },
        { x: 0.75, y: 0.6, c: '#ff4757' },
        { x: 0.85, y: 0.3, c: '#ff4757' },
        { x: 0.7, y: 0.8, c: '#ff4757' },
    ];
    const edges = [[0, 1], [0, 2], [1, 2], [3, 5], [4, 6]];

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw edges with animation
        const edgeProgress = Math.min(1, (t % 120) / 60);
        for (const [a, b] of edges) {
            const na = nodes[a], nb = nodes[b];
            const x1 = na.x * canvas.width, y1 = na.y * canvas.height;
            const x2 = nb.x * canvas.width, y2 = nb.y * canvas.height;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x1 + (x2 - x1) * edgeProgress, y1 + (y2 - y1) * edgeProgress);
            ctx.strokeStyle = 'rgba(0, 212, 255, 0.4)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        for (const n of nodes) {
            const x = n.x * canvas.width;
            const y = n.y * canvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 7, 0, Math.PI * 2);
            ctx.fillStyle = n.c;
            ctx.globalAlpha = 0.85;
            ctx.fill();
            ctx.globalAlpha = 1;
        }

        t++;
        requestAnimationFrame(draw);
    }
    draw();
}

// ---- Step 5: Attention weights glowing ----
function initStep5Canvas() {
    const canvas = document.getElementById('step5-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = 120;

    let t = 0;
    const nodes = [
        { x: 0.15, y: 0.5, c: '#00ff88' },
        { x: 0.35, y: 0.3, c: '#00ff88' },
        { x: 0.3, y: 0.75, c: '#00ff88' },
        { x: 0.65, y: 0.4, c: '#ff4757' },
        { x: 0.8, y: 0.7, c: '#ff4757' },
        { x: 0.85, y: 0.25, c: '#ff4757' },
    ];
    const edges = [
        { a: 0, b: 1, w: 0.9 },
        { a: 0, b: 2, w: 0.7 },
        { a: 1, b: 2, w: 0.8 },
        { a: 3, b: 4, w: 0.3 },
        { a: 3, b: 5, w: 0.2 },
        { a: 1, b: 3, w: 0.15 },
    ];

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const pulse = (Math.sin(t * 0.05) + 1) / 2;

        for (const e of edges) {
            const na = nodes[e.a], nb = nodes[e.b];
            const x1 = na.x * canvas.width, y1 = na.y * canvas.height;
            const x2 = nb.x * canvas.width, y2 = nb.y * canvas.height;

            const alpha = 0.2 + e.w * 0.6 * (0.7 + pulse * 0.3);
            const width = 1 + e.w * 4;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = e.w > 0.5 ? `rgba(0, 255, 136, ${alpha})` : `rgba(255, 71, 87, ${alpha * 0.5})`;
            ctx.lineWidth = width;
            ctx.stroke();
        }

        for (const n of nodes) {
            const x = n.x * canvas.width;
            const y = n.y * canvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = n.c;
            ctx.globalAlpha = 0.9;
            ctx.fill();

            // Glow
            ctx.beginPath();
            ctx.arc(x, y, 14, 0, Math.PI * 2);
            ctx.fillStyle = n.c;
            ctx.globalAlpha = 0.1 + pulse * 0.1;
            ctx.fill();
            ctx.globalAlpha = 1;
        }

        // Label
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillText('Thick = high attention', 10, canvas.height - 8);

        t++;
        requestAnimationFrame(draw);
    }
    draw();
}

// ---- Initialize all pipeline canvases ----
function initAllPipelines() {
    initHeroCanvas();
    initStep1Canvas();
    initStep2Canvas();
    initStep3Canvas();
    initStep4Canvas();
    initStep5Canvas();
}

document.addEventListener('DOMContentLoaded', initAllPipelines);
