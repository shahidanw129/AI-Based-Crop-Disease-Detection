const saveVaultButton = document.querySelector('#save-offline-vault');

function openFieldbook() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('fieldnote-offline', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('fieldbook', { keyPath: 'id' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function storeFieldbook(snapshot) {
  const database = await openFieldbook();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction('fieldbook', 'readwrite');
    transaction.objectStore('fieldbook').put({ ...snapshot, id: 'current' });
    transaction.oncomplete = () => { database.close(); resolve(); };
    transaction.onerror = () => { database.close(); reject(transaction.error); };
  });
}

if (saveVaultButton) {
  saveVaultButton.addEventListener('click', async () => {
    saveVaultButton.disabled = true;
    saveVaultButton.textContent = 'Saving private fieldbook…';
    try {
      const response = await fetch('/api/offline-vault', { credentials: 'same-origin', cache: 'no-store' });
      if (!response.ok) throw new Error('Please sign in again and retry.');
      const snapshot = await response.json();
      const downloads = await Promise.all(snapshot.scans.slice(0, 20).map(async (scan) => {
        const report = await fetch(scan.report_url, { credentials: 'same-origin' });
        const image = await fetch(scan.image_url, { credentials: 'same-origin' });
        return {
          ...scan,
          report_blob: report.ok ? await report.blob() : null,
          image_blob: image.ok ? await image.blob() : null,
        };
      }));
      await storeFieldbook({ ...snapshot, scans: downloads });
      saveVaultButton.textContent = 'Saved on this device ✓';
    } catch (error) {
      saveVaultButton.textContent = error.message || 'Offline save failed';
    } finally {
      saveVaultButton.disabled = false;
    }
  });
}