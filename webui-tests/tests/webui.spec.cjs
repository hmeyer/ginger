// @ts-check
const { test, expect } = require('@playwright/test');

const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64'
);

/**
 * Mocks every /api/* endpoint and records POST bodies. Returns the live
 * array of captured requests ({ url, body }) plus a small filter helper.
 */
async function mockApi(page) {
  const posts = [];

  await page.route('**/api/**', async (route) => {
    const req = route.request();
    const url = new URL(req.url()).pathname;

    if (url === '/api/sensors/stream') {
      // One harmless event; a long retry so EventSource doesn't reconnect.
      return route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'retry: 999999\n\n',
      });
    }
    if (url === '/api/webrtc/whep') {
      return route.fulfill({ status: 503, body: 'no camera in tests' });
    }
    if (url === '/api/map') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ robot_gx: 80, robot_gy: 80, robot_heading: 0 }),
      });
    }
    if (url === '/api/map/png') {
      return route.fulfill({ status: 200, contentType: 'image/png', body: TINY_PNG });
    }
    if (url === '/api/slam/map') {
      // The top-down SLAM map snapshot that feeds the #map-stat HUD.
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'tracking: 28/40 inliers',
          model: 'essential',
          points: [], cameras: [], keyframes: [],
          n_points: 120, r_h: 0.4, tracking: true, n_tracked: 28,
          n_keyframes: 7, loop_closures: 2, bow_ready: true, bow_words: 256,
        }),
      });
    }

    if (req.method() === 'POST') {
      let body = null;
      try { body = req.postDataJSON(); } catch (_) { /* no body */ }
      posts.push({ url, body });
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });

  return {
    posts,
    of(path) { return posts.filter((p) => p.url === path); },
    clear() { posts.length = 0; },
  };
}

/** Press the pointer at the center of `selector` and drag to a fractional
 *  point (fx, fy in [0,1] of the element box). Leaves the button down. */
async function dragTo(page, selector, fx, fy) {
  const box = await page.locator(selector).boundingBox();
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy, { steps: 10 });
}

const knobTransform = (page, sel) =>
  page.locator(`${sel} .joy-knob`).evaluate((el) => el.style.transform);

test('slam HUD: map-stat shows tracking state and map counters', async ({ page }) => {
  await mockApi(page);
  await page.goto('/');
  // Polled from /api/slam/map and rendered before the per-mode
  // early-returns, so it is visible regardless of the feature mode.
  const stat = page.locator('#map-stat');
  await expect(stat).toContainText('TRACK');
  await expect(stat).toContainText('kf:7');
  await expect(stat).toContainText('pt:120');
  await expect(stat).toContainText('loop:2');
  await expect(stat).toContainText('bow:256w');
});

test('layout: joysticks replace the D-pad and sliders', async ({ page }) => {
  await mockApi(page);
  await page.goto('/');
  await expect(page.locator('#drive-joy')).toBeVisible();
  await expect(page.locator('#cam-joy')).toBeVisible();
  // Old controls are gone.
  await expect(page.locator('#b-fwd')).toHaveCount(0);
  await expect(page.locator('#b-stop')).toHaveCount(0);
  await expect(page.locator('input#pan[type=range]')).toHaveCount(0);
});

test('drive joystick: forward drag sends both motors forward, release stops', async ({ page }) => {
  const api = await mockApi(page);
  await page.goto('/');

  await dragTo(page, '#drive-joy', 0.5, 0.0); // straight up
  await expect.poll(() => api.of('/api/drive').length).toBeGreaterThan(0);
  const last = api.of('/api/drive').at(-1).body;
  expect(last.left).toBeGreaterThan(1900);
  expect(last.right).toBeGreaterThan(1900);
  expect(Math.abs(last.left - last.right)).toBeLessThan(50);

  await page.mouse.up();
  await expect.poll(() => api.of('/api/stop').length).toBeGreaterThan(0);
  // Spring-back: knob returns to center.
  await expect.poll(() => knobTransform(page, '#drive-joy')).toBe('translate(-50%, -50%)');
});

test('drive joystick: left drag spins left wheel forward, right wheel back', async ({ page }) => {
  const api = await mockApi(page);
  await page.goto('/');

  await dragTo(page, '#drive-joy', 0.0, 0.5); // hard left
  await expect.poll(() => api.of('/api/drive').length).toBeGreaterThan(0);
  const last = api.of('/api/drive').at(-1).body;
  expect(last.left).toBeGreaterThan(1000);
  expect(last.right).toBeLessThan(-1000);
  await page.mouse.up();
});

test('keyboard only drives the car while the joystick is focused', async ({ page }) => {
  const api = await mockApi(page);
  await page.goto('/');

  // Not focused: arrow keys must do nothing.
  await page.locator('header h1').click();
  await page.keyboard.press('ArrowUp');
  await page.waitForTimeout(200);
  expect(api.of('/api/drive').length).toBe(0);

  // Focus the joystick explicitly (no pointer drive).
  await page.locator('#drive-joy').focus();
  await expect(page.locator('#drive-joy')).toHaveClass(/focused/);
  await expect(page.locator('#drive-hint')).toHaveText(/arrow keys active/);

  await page.keyboard.down('ArrowUp');
  await expect.poll(() => api.of('/api/drive').length).toBeGreaterThan(0);
  const cmd = api.of('/api/drive').at(-1).body;
  expect(cmd.left).toBeGreaterThan(1900);
  expect(cmd.right).toBeGreaterThan(1900);

  await page.keyboard.up('ArrowUp');
  await expect.poll(() => api.of('/api/stop').length).toBeGreaterThan(0);

  // Blur releases control.
  await page.locator('#drive-joy').blur();
  await expect(page.locator('#drive-joy')).not.toHaveClass(/focused/);
});

test('camera joystick: sends pan/tilt in degrees and holds position', async ({ page }) => {
  const api = await mockApi(page);
  await page.goto('/');

  // Drag fully right → pan = 180°, tilt unchanged (no tilt POST).
  await dragTo(page, '#cam-joy', 1.0, 0.5);
  await page.mouse.up();
  await expect.poll(() => api.of('/api/pan').length).toBeGreaterThan(0);
  expect(api.of('/api/pan').at(-1).body.angle).toBe(180);
  expect(api.of('/api/tilt').length).toBe(0);
  await expect(page.locator('#pan-v')).toHaveText('180');

  // Sticky: knob stays off-center after release.
  expect(await knobTransform(page, '#cam-joy')).toContain('calc');

  // Double-click recenters to 90/90.
  await page.locator('#cam-joy').dblclick();
  await expect.poll(() => api.of('/api/pan').at(-1).body.angle).toBe(90);
  expect(api.of('/api/tilt').at(-1).body.angle).toBe(90);
  await expect(page.locator('#pan-v')).toHaveText('90');
  await expect(page.locator('#tilt-v')).toHaveText('90');
  expect(await knobTransform(page, '#cam-joy')).toBe('translate(-50%, -50%)');
});

test('camera joystick: down drag tilts and sends tilt degrees', async ({ page }) => {
  const api = await mockApi(page);
  await page.goto('/');
  await dragTo(page, '#cam-joy', 0.5, 1.0); // straight down
  await page.mouse.up();
  await expect.poll(() => api.of('/api/tilt').length).toBeGreaterThan(0);
  expect(api.of('/api/tilt').at(-1).body.angle).toBe(0);
  await expect(page.locator('#tilt-v')).toHaveText('0');
});

test('camera joystick recenters when telemetry reports an auto-center', async ({ page }) => {
  await mockApi(page);
  await page.goto('/');

  // Aim the bracket off-center and let it settle (sticky).
  await dragTo(page, '#cam-joy', 1.0, 0.5);
  await page.mouse.up();
  await expect(page.locator('#pan-v')).toHaveText('180');

  // Within the interaction grace window, telemetry must NOT yank the knob.
  await page.evaluate(() => camSync(90, 90));
  expect(await page.evaluate(() => camPan)).toBe(180);

  // After the grace window, a telemetry update (the supervisor's forward-
  // drive auto-center) snaps the knob back to center.
  await page.waitForTimeout(800);
  await page.evaluate(() => camSync(90, 90));
  expect(await page.evaluate(() => camPan)).toBe(90);
  await expect(page.locator('#pan-v')).toHaveText('90');
  expect(await knobTransform(page, '#cam-joy')).toBe('translate(-50%, -50%)');
});
