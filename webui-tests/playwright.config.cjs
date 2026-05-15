const { defineConfig, devices } = require('@playwright/test');

const PORT = 8123;

module.exports = defineConfig({
  testDir: './tests',
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    // Desktop layout (>=700px) so the focus/keyboard behaviour is exercised.
    viewport: { width: 1280, height: 800 },
    actionTimeout: 5_000,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: 'node serve.cjs',
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: true,
    timeout: 10_000,
  },
});
