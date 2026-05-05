/* QuantPipe Service Worker — v2
   - API calls       → always network
   - / (shell)       → network-first; cache updated as side-effect; falls back to cache offline
   - Known statics   → cache-first (icons, logo, manifest — truly static)
   - Everything else → network only (no stale caching of unknown routes)
*/
const CACHE = 'qp-v2';
const STATIC_ASSETS = [
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/logo.png',
  '/manifest.json',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE)
      .then(c => c.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // API — always network, offline stub on failure
  if (url.pathname.startsWith('/api/')) {
    e.respondWith(
      fetch(e.request).catch(() =>
        new Response(JSON.stringify({ error: 'offline' }), {
          headers: { 'Content-Type': 'application/json' },
        })
      )
    );
    return;
  }

  // App shell — network-first so every deploy is picked up immediately
  if (url.pathname === '/' || url.pathname === '/index.html') {
    e.respondWith(
      fetch(e.request)
        .then(resp => {
          if (resp.ok) {
            const clone = resp.clone();
            caches.open(CACHE).then(c => c.put(e.request, clone));
          }
          return resp;
        })
        .catch(() => caches.match(e.request))  // use e.request so /index.html hits its own cache key
    );
    return;
  }

  // Known static assets — cache-first
  if (STATIC_ASSETS.includes(url.pathname)) {
    e.respondWith(
      caches.match(e.request).then(cached =>
        cached || fetch(e.request).then(resp => {
          if (resp.ok) {
            const clone = resp.clone();
            caches.open(CACHE).then(c => c.put(e.request, clone));
          }
          return resp;
        })
      )
    );
    return;
  }

  // Everything else — network only (no indefinite caching of unknown routes)
  e.respondWith(fetch(e.request));
});
