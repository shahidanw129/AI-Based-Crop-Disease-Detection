const metricNode = document.querySelector('#model-metrics');
const nameNode = document.querySelector('#model-names');
if (metricNode && nameNode && window.Chart) {
  const metrics = JSON.parse(metricNode.textContent);
  const names = JSON.parse(nameNode.textContent);
  const canvas = document.querySelector('#model-comparison');
  if (canvas && names.length) {
    const accuracy = metrics.map((entry) => Number.isFinite(entry.test_accuracy) ? entry.test_accuracy * 100 : null);
    const f1 = metrics.map((entry) => {
      const value = entry.classification_report?.['macro avg']?.['f1-score'];
      return Number.isFinite(value) ? value * 100 : null;
    });
    new Chart(canvas, {
      type: 'bar',
      data: {
        labels: names,
        datasets: [
          { label: 'Held-out accuracy (%)', data: accuracy, backgroundColor: '#3f7051' },
          { label: 'Macro F1 (%)', data: f1, backgroundColor: '#d29a48' },
        ],
      },
      options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100 } } },
    });
  }
}