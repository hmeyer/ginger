// Minimal static server that serves the real web UI (src/bin/web/index.html)
// with the {{BUILD_TIME}} template token filled in, exactly like server.rs
// does. All /api/* traffic is mocked by Playwright route handlers in the
// tests, so this server only needs to hand back the document.
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 8123;
const HTML = path.join(__dirname, '..', 'src', 'bin', 'web', 'index.html');

const server = http.createServer((req, res) => {
  const url = req.url.split('?')[0];
  if (url === '/' || url === '/index.html') {
    const html = fs.readFileSync(HTML, 'utf8').replace('{{BUILD_TIME}}', 'test');
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(html);
    return;
  }
  // Anything else is intercepted by Playwright; if it slips through just 204.
  res.writeHead(204);
  res.end();
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`webui test server on http://127.0.0.1:${PORT}`);
});
