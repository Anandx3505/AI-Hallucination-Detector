/**
 * Charts — Bar chart and Radar chart for baseline comparison
 * Uses Chart.js 4
 */

const RESULTS = {
    labels: ['GAT (Ours)', 'DBSCAN', 'GCN', 'SelfCheckGPT'],
    accuracy: [86.48, 74.91, 73.94, 34.97],
    f1: [72.68, 38.64, 36.71, 28.53],
    binaryRecall: [95.13, 89.94, 50.97, 98.38],
    colors: {
        gat: '#00ff88',
        others: ['#ff9f43', '#7c3aed', '#ff4757'],
    },
};

function initBarChart() {
    const ctx = document.getElementById('barChart');
    if (!ctx) return;

    const bgColors = [RESULTS.colors.gat, ...RESULTS.colors.others];

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: RESULTS.labels,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: RESULTS.accuracy,
                    backgroundColor: bgColors.map(c => c + '99'),
                    borderColor: bgColors,
                    borderWidth: 2,
                    borderRadius: 6,
                    barPercentage: 0.45,
                },
                {
                    label: 'Macro F1 (%)',
                    data: RESULTS.f1,
                    backgroundColor: bgColors.map(c => c + '55'),
                    borderColor: bgColors,
                    borderWidth: 2,
                    borderRadius: 6,
                    borderDash: [4, 4],
                    barPercentage: 0.45,
                }
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#8892b0', font: { family: 'Inter', size: 12 } },
                },
            },
            scales: {
                x: {
                    ticks: { color: '#8892b0', font: { family: 'Inter', size: 11 } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#8892b0',
                        font: { family: 'JetBrains Mono', size: 11 },
                        callback: v => v + '%',
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                },
            },
        },
    });
}

function initRadarChart() {
    const ctx = document.getElementById('radarChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Macro F1', 'Binary Recall', 'Macro Precision', 'Consistency'],
            datasets: [
                {
                    label: 'GAT (Ours)',
                    data: [86.48, 72.68, 95.13, 72.84, 97.03],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.12)',
                    borderWidth: 2.5,
                    pointBackgroundColor: '#00ff88',
                    pointRadius: 4,
                },
                {
                    label: 'DBSCAN',
                    data: [74.91, 38.64, 89.94, 40.0, 60.0],
                    borderColor: '#ff9f43',
                    backgroundColor: 'rgba(255, 159, 67, 0.08)',
                    borderWidth: 1.5,
                    pointBackgroundColor: '#ff9f43',
                    pointRadius: 3,
                },
                {
                    label: 'GCN',
                    data: [73.94, 36.71, 50.97, 48.27, 55.0],
                    borderColor: '#7c3aed',
                    backgroundColor: 'rgba(124, 58, 237, 0.08)',
                    borderWidth: 1.5,
                    pointBackgroundColor: '#7c3aed',
                    pointRadius: 3,
                },
                {
                    label: 'SelfCheckGPT',
                    data: [34.97, 28.53, 98.38, 25.0, 35.0],
                    borderColor: '#ff4757',
                    backgroundColor: 'rgba(255, 71, 87, 0.08)',
                    borderWidth: 1.5,
                    pointBackgroundColor: '#ff4757',
                    pointRadius: 3,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#8892b0', font: { family: 'Inter', size: 11 } },
                    position: 'bottom',
                },
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#5a6380',
                        font: { family: 'JetBrains Mono', size: 9 },
                        backdropColor: 'transparent',
                    },
                    grid: { color: 'rgba(255,255,255,0.08)' },
                    angleLines: { color: 'rgba(255,255,255,0.08)' },
                    pointLabels: { color: '#8892b0', font: { family: 'Inter', size: 11 } },
                },
            },
        },
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initBarChart();
    initRadarChart();
});
