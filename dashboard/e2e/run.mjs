// End-to-end check of the dashboard against a running `horizonguard serve`.
// Drives the locally installed Chrome, exercises every page, fails on console
// errors or broken flows, and saves screenshots for the README.
//
//   BASE_URL=http://127.0.0.1:8765 CHROME_PATH="C:/Program Files/Google/Chrome/Application/chrome.exe" npm run e2e

import { mkdir } from "node:fs/promises";
import puppeteer from "puppeteer-core";

const BASE = process.env.BASE_URL ?? "http://127.0.0.1:8765";
const CHROME =
  process.env.CHROME_PATH ??
  (process.platform === "win32"
    ? "C:/Program Files/Google/Chrome/Application/chrome.exe"
    : "/usr/bin/google-chrome");
const SHOTS = process.env.SHOTS_DIR ?? "../docs/screenshots";
const HEADLESS = process.env.HEADED ? false : "new";

const results = [];
const problems = [];

async function check(name, fn) {
  const start = Date.now();
  try {
    await fn();
    results.push(`PASS  ${name} (${Date.now() - start} ms)`);
  } catch (err) {
    problems.push(`${name}: ${err.message}`);
    results.push(`FAIL  ${name}: ${err.message}`);
  }
}

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: HEADLESS,
  defaultViewport: { width: 1440, height: 900, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
const consoleErrors = [];
page.on("console", (msg) => msg.type() === "error" && consoleErrors.push(msg.text()));
page.on("pageerror", (err) => consoleErrors.push(err.message));
page.on("requestfailed", (req) => consoleErrors.push(`request failed: ${req.url()}`));
await mkdir(SHOTS, { recursive: true });

const shot = (name) => page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: false });
const text = () => page.evaluate(() => document.body.innerText);
const waitText = (t, timeout = 60000) =>
  page.waitForFunction((s) => document.body.innerText.includes(s), { timeout }, t);

await check("API health", async () => {
  const res = await page.goto(`${BASE}/api/health`);
  assert(res.ok(), `status ${res.status()}`);
});

await check("overview loads with live numbers and verdicts", async () => {
  await page.goto(BASE, { waitUntil: "networkidle0" });
  await waitText("Research questions and verdicts");
  const body = await text();
  assert(body.includes("Violations, HorizonGuard"), "missing KPI tiles");
  assert(/\d+%/.test(body), "no percentages rendered");
  assert(body.includes("supported"), "missing verdict chips");
  await shot("overview");
});

await check("sidebar navigation reaches every page", async () => {
  for (const [label, expected] of [
    ["Live monitor", "Watch a run, step by step"],
    ["Runs", "Every closed-loop episode"],
    ["A/B test", "Fork-controlled A/B"],
    ["Guarantees", "How much harm can an alert cause?"],
    ["Method", "From 'will it drift?'"],
    ["Overview", "Is the agent still fixable"],
  ]) {
    const [link] = await page.$$(`xpath/.//nav//a[contains(., "${label}")]`);
    assert(link, `no sidebar link ${label}`);
    await link.click();
    await waitText(expected, 20000);
  }
});

await check("live monitor runs a closed-loop simulation and replays it", async () => {
  await page.goto(`${BASE}/live`, { waitUntil: "networkidle0" });
  await page.waitForSelector("select");
  await page.select("select", "support-desk");
  const selects = await page.$$("select");
  await selects[1].select("fjord-plan");
  await selects[2].select("goal-switch");
  await selects[3].select("horizonguard");
  const seed = await page.$("input[type=number]");
  await seed.click({ clickCount: 3 });
  await seed.type("11");
  const [runBtn] = await page.$$("xpath/.//button[contains(., 'Run simulation')]");
  await runBtn.click();
  await waitText("Predicted net benefit", 90000);
  await page.waitForSelector("svg[aria-label='Risk and fidelity over the run'] path");
  await new Promise((r) => setTimeout(r, 2500)); // let the replay animate a little
  const body = await text();
  assert(body.includes("Fix-by countdown") && body.includes("Step log"), "panels missing");
  await shot("live-monitor");
  // scrub with the slider to the last step
  await page.evaluate(() => {
    const slider = document.querySelector("input[type=range]");
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    setter.call(slider, slider.max);
    slider.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const stepLabel = await page.$eval("input[type=range]", (el) => el.max);
  await waitText(`step ${stepLabel}`, 5000);
});

await check("same seed without a monitor gives a comparable run", async () => {
  const selects = await page.$$("select");
  await selects[3].select("none");
  const [runBtn] = await page.$$("xpath/.//button[contains(., 'Run simulation')]");
  await runBtn.click();
  await waitText("0 interventions", 90000);
});

await check("runs list: debounced search, filters and pagination", async () => {
  await page.goto(`${BASE}/runs`, { waitUntil: "networkidle0" });
  await waitText("page 1 of");
  const before = await text();
  const total = Number(before.match(/(\d+) runs · page/)[1]);
  assert(total >= 30, `expected a library of runs, got ${total}`);
  const [next] = await page.$$("xpath/.//button[contains(., 'Next')]");
  await next.click();
  await waitText("page 2 of");
  await page.type("input[placeholder^='run id']", "horizonguard");
  // the debounced query lands about 300 ms after typing stops; wait for the total to change
  await page.waitForFunction(
    (t) => {
      const m = document.body.innerText.match(/(\d+) runs · page 1/);
      return m && Number(m[1]) !== t;
    },
    { timeout: 10000 },
    total,
  );
  const filtered = Number((await text()).match(/(\d+) runs · page/)[1]);
  assert(filtered < total, "search did not filter");
  const outcome = (await page.$$("select"))[1];
  await outcome.select("violated");
  await new Promise((r) => setTimeout(r, 600));
  await shot("runs");
});

await check("run detail opens from the list and switches policy on the same seed", async () => {
  await page.goto(`${BASE}/runs`, { waitUntil: "networkidle0" });
  await page.waitForSelector("tr.clickable");
  await page.click("tr.clickable");
  await waitText("Predicted net benefit", 30000);
  const url1 = page.url();
  const [other] = await page.$$("xpath/.//a[contains(., 'No monitor')][contains(@class,'btn')]");
  await other.click();
  await page.waitForFunction((u) => location.href !== u, { timeout: 10000 }, url1);
  assert(page.url().endsWith("-none"), `unexpected url ${page.url()}`);
  await waitText("Predicted net benefit", 30000);
  await shot("run-detail");
});

await check("A/B page renders every seed and links into runs", async () => {
  await page.goto(`${BASE}/ab`, { waitUntil: "networkidle0" });
  await waitText("Every seed");
  const squares = await page.$$("button[aria-label^='Open seed']");
  assert(squares.length >= 30, `only ${squares.length} seed squares`);
  await shot("ab-test");
  await squares[0].click();
  await waitText("Predicted net benefit", 30000);
});

await check("guarantees page shows the certificate and studies", async () => {
  await page.goto(`${BASE}/guarantees`, { waitUntil: "networkidle0" });
  await waitText("Rescues kept vs calibration size");
  await shot("guarantees");
});

await check("method page and unknown routes", async () => {
  await page.goto(`${BASE}/method`, { waitUntil: "networkidle0" });
  await waitText("Fork engine");
  await shot("method");
  await page.goto(`${BASE}/does-not-exist`, { waitUntil: "networkidle0" });
  await waitText("Page not found");
});

await check("API is compressed, cached and paginated", async () => {
  const headers = await page.evaluate(async () => {
    const r = await fetch("/api/runs?page=1&size=5");
    const body = await r.json();
    return {
      cache: r.headers.get("cache-control"),
      etag: r.headers.get("etag"),
      size: body.items.length,
    };
  });
  assert(headers.etag && headers.cache.includes("max-age"), "missing cache headers");
  assert(headers.size === 5, "pagination ignored");
  const before = consoleErrors.length;
  const bad = await page.evaluate(async () => (await fetch("/api/simulate", {
    method: "POST",
    body: JSON.stringify({ env: "mars" }),
  })).status);
  assert(bad === 400, `bad simulate request returned ${bad}`);
  consoleErrors.length = before; // Chrome logs the deliberate 400; it is expected here
});

await check("mobile layout keeps navigation usable", async () => {
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
  await page.goto(BASE, { waitUntil: "networkidle0" });
  await waitText("Research questions");
  const navVisible = await page.$eval("nav", (n) => n.getBoundingClientRect().height > 0);
  assert(navVisible, "navigation hidden on mobile");
  await page.screenshot({ path: `${SHOTS}/mobile.png` });
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
});

await check("no console errors or failed requests", async () => {
  assert(consoleErrors.length === 0, consoleErrors.slice(0, 5).join(" | "));
});

await browser.close();
console.log(results.join("\n"));
if (problems.length) {
  console.error(`\n${problems.length} check(s) failed`);
  process.exit(1);
}
console.log(`\nAll ${results.length} checks passed`);
