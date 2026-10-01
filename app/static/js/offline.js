function readQueuedReminders() {
  try { return JSON.parse(localStorage.getItem('fieldnote-reminder-queue') || '[]'); }
  catch { return []; }
}

function writeQueue(queue) {
  localStorage.setItem('fieldnote-reminder-queue', JSON.stringify(queue));
}

function openOfflineBook() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('fieldnote-offline', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('fieldbook', { keyPath: 'id' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function loadSavedBook() {
  try {
    const database = await openOfflineBook();
    const snapshot = await new Promise((resolve, reject) => {
      const request = database.transaction('fieldbook').objectStore('fieldbook').get('current');
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    database.close();
    if (!snapshot) return;
    const scanContainer = document.querySelector('#offline-scans');
    scanContainer.replaceChildren();
    snapshot.scans.forEach((scan) => {
      const row = document.createElement('article');
      row.className = 'offline-scan-row';
      const details = document.createElement('div');
      const title = document.createElement('h3');
      title.textContent = scan.label;
      const meta = document.createElement('p');
      meta.textContent = `${scan.date} · ${scan.confidence}%`;
      details.append(title, meta);
      if (scan.image_blob) {
        const image = document.createElement('img');
        image.src = URL.createObjectURL(scan.image_blob);
        image.alt = 'Saved crop leaf';
        row.append(image);
      }
      row.append(details);
      if (scan.report_blob) {
        const download = document.createElement('a');
        download.href = URL.createObjectURL(scan.report_blob);
        download.download = `fieldnote-scan-${scan.id}.pdf`;
        download.className = 'text-link';
        download.textContent = 'Download saved report ↓';
        row.append(download);
      }
      scanContainer.append(row);
    });
    const tipContainer = document.querySelector('#offline-tips');
    tipContainer.replaceChildren();
    snapshot.tips.forEach((tip) => {
      const article = document.createElement('article');
      article.className = 'offline-tip-row';
      const title = document.createElement('h3');
      title.textContent = `${tip.crop} · ${tip.title}`;
      const body = document.createElement('p');
      body.textContent = tip.body;
      article.append(title, body);
      tipContainer.append(article);
    });
    document.querySelector('#offline-status').textContent = `Saved ${snapshot.saved_at}. ${snapshot.scans.length} scan records and ${snapshot.tips.length} tips are available on this device.`;
  } catch {
    document.querySelector('#offline-status').textContent = 'No saved offline fieldbook was found in this browser.';
  }
}

async function syncReminders() {
  const status = document.querySelector('#offline-status');
  const queue = readQueuedReminders();
  if (!queue.length) {
    status.textContent = 'There are no queued reminders to sync.';
    return;
  }
  if (!navigator.onLine) {
    status.textContent = `${queue.length} reminder(s) are queued until this device reconnects.`;
    return;
  }
  try {
    const tokenResponse = await fetch('/offline/sync-token', { credentials: 'same-origin', cache: 'no-store' });
    if (!tokenResponse.ok || tokenResponse.redirected) throw new Error('Sign in with the account that created these reminders, then retry.');
    const { csrf_token: token } = await tokenResponse.json();
    const pending = [...queue];
    for (const reminder of pending) {
      const response = await fetch('/api/reminders/sync', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'X-CSRFToken': token },
        body: new URLSearchParams(reminder),
      });
      if (!response.ok) throw new Error('A queued reminder could not be saved. Review its due date and retry.');
      queue.shift();
      writeQueue(queue);
    }
    status.textContent = `${pending.length} reminder(s) synced to your account.`;
  } catch (error) {
    status.textContent = error.message || 'Sync is waiting for a connection and active session.';
  }
}

document.querySelector('#offline-reminder-form')?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(event.currentTarget));
  const queue = readQueuedReminders();
  queue.push({ title: values.title, body: values.body, due_at: values.due_at });
  writeQueue(queue);
  event.currentTarget.reset();
  document.querySelector('#offline-status').textContent = 'Reminder saved locally. Attempting account sync if online.';
  if ('serviceWorker' in navigator) {
    const registration = await navigator.serviceWorker.ready.catch(() => null);
    registration?.sync?.register('fieldnote-reminders').catch(() => {});
  }
  await syncReminders();
});

document.querySelector('#sync-queued-reminders')?.addEventListener('click', syncReminders);
window.addEventListener('online', syncReminders);
navigator.serviceWorker?.addEventListener('message', (event) => {
  if (event.data?.type === 'SYNC_REMINDERS') syncReminders();
});
loadSavedBook();