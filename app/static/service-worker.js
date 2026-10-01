const CACHE_NAME = 'fieldnote-shell-v3';
const SHELL_FILES = [
  '/static/css/style.css',
  '/static/js/main.js',
  '/static/js/offline.js',
  '/static/images/pwa-icon.svg',
  '/static/offline-shell.html',
  '/static/manifest.webmanifest',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES)));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  if (request.mode === 'navigate') {
    if (url.pathname === '/' || url.pathname === '/diseases') {
      event.respondWith(
        fetch(request).then((response) => {
          if (response.ok) caches.open(CACHE_NAME).then((cache) => cache.put(request, response.clone()));
          return response;
        }).catch(async () => (await caches.match(request)) || caches.match('/static/offline-shell.html'))
      );
      return;
    }
    event.respondWith(fetch(request).catch(() => caches.match('/static/offline-shell.html')));
    return;
  }

  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(request).then((cached) => cached || fetch(request).then((response) => {
        if (response.ok && (url.pathname.startsWith('/static/css/') || url.pathname.startsWith('/static/js/'))) {
          caches.open(CACHE_NAME).then((cache) => cache.put(request, response.clone()));
        }
        return response;
      }))
    );
  }
});

self.addEventListener('sync', (event) => {
  if (event.tag === 'fieldnote-reminders') {
    event.waitUntil(self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clients) => {
      clients.forEach((client) => client.postMessage({ type: 'SYNC_REMINDERS' }));
    }));
  }
});