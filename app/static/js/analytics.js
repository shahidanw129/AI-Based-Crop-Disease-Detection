const dataNode = document.querySelector('#analytics-data');
const analytics = dataNode ? JSON.parse(dataNode.textContent) : null;
const chartColors = ['#3f7051', '#d29a48', '#588b9b', '#ae5540', '#86925f', '#6d7190'];
const charts = new Map();

if (analytics && window.Chart) {
  const trendCanvas = document.querySelector('#trend-chart');
  if (trendCanvas) {
    charts.set('trend-chart', new Chart(trendCanvas, {
      type: 'line',
      data: {
        labels: analytics.labels,
        datasets: analytics.datasets.map((dataset, index) => ({
          ...dataset,
          borderColor: chartColors[index % chartColors.length],
          backgroundColor: `${chartColors[index % chartColors.length]}22`,
          pointRadius: 3,
          tension: 0.25,
          fill: false,
        })),
      },
      options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } },
    }));
  }

  const healthCanvas = document.querySelector('#health-chart');
  if (healthCanvas) {
    charts.set('health-chart', new Chart(healthCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Healthy-class signals', 'Other model classes'],
        datasets: [{ data: [analytics.healthy, analytics.other], backgroundColor: ['#668d63', '#d29a48'], borderWidth: 0 }],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } },
    }));
  }
}

document.querySelectorAll('.chart-download').forEach((button) => {
  button.addEventListener('click', () => {
    const chart = charts.get(button.dataset.chart);
    if (!chart) return;
    const link = document.createElement('a');
    link.href = chart.toBase64Image('image/png', 1);
    link.download = `${button.dataset.chart}.png`;
    link.click();
  });
});