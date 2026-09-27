// Full end-to-end test: two real Chromium instances via Playwright, the actual frontend JS (same files the
// backend serves), a live WebSocket signalling relay, a live WebRTC data channel, and the real FastAPI backend.
// Whisper, the tone tagger and the LLM are swapped for the SIGNBRIDGE_FAKE_MODELS stub (see backend/engines.py
// and backend/smoother.py) since this machine has no GPU and no network to grab their weights - but every other
// line below is the real deployed code: dictionary decoder, sign grammar, rule-based smoother, the browser's
// own letter-typing function, its WAV encoder + VAD, the landing-screen role selection, the actual call.
//
// Each browser picks a role first, like a real caller would - browser A goes "I sign", browser B goes "I speak".
// That locks each one's call screen to a single pipeline, so this also checks the *other* pipeline's tools stay
// hidden on both sides before driving each browser through whatever its own role exposes.
//
// This isn't a substitute for a live-camera test - MediaPipe still needs an actual hand in front of an actual
// camera, which this machine doesn't have (see Section 5.1). What this does prove is that once a role's picked
// and a letter's been typed or spoken, everything downstream - across two separate browser processes and a real
// network stack - actually works together.
//
// Usage:  SIGNBRIDGE_FAKE_MODELS=1 python -m uvicorn backend.app:app --port 8130 &
//         node tests/e2e/e2e_call.mjs http://localhost:8130 [screenshot-dir]
// Needs Playwright (npm i -D playwright) and a Chromium build. Not part of `pytest tests` - that one needs
// neither a browser nor a display.
import { chromium } from "playwright-core";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

const BASE = process.argv[2] || "http://localhost:8130";
const SHOTS = process.argv[3] || null;
if (SHOTS) fs.mkdirSync(SHOTS, { recursive: true });

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const row = (ch, conf = 0.9) => { const a = new Array(26).fill((1 - conf) / 25); a[LETTERS.indexOf(ch)] = conf; return a; };
const flat = () => new Array(26).fill(1 / 26);

let pass = 0, fail = 0;
function check(name, ok, detail) {
  console.log((ok ? "PASS " : "FAIL ") + name + (detail ? "  -  " + detail : ""));
  ok ? pass++ : fail++;
}
async function shot(page, name) { if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png` }); }
async function card(page, heading, name) {
  if (!SHOTS) return;
  const loc = page.locator("h2", { hasText: heading }).locator("xpath=ancestor::div[contains(@class,'card')][1]");
  await loc.screenshot({ path: `${SHOTS}/${name}.png` }).catch((e) => console.log("shot failed", name, e.message));
}
// since the redesign, the whole call screen is one frame: shoot the frame itself, or the nearest ancestor
// of any given element, rather than hunting for a particular card's own heading.
async function cardSel(page, selector, name) {
  if (!SHOTS) return;
  const loc = page.locator(selector).locator("xpath=ancestor::div[contains(@class,'card')][1]");
  await loc.screenshot({ path: `${SHOTS}/${name}.png` }).catch((e) => console.log("shot failed", name, e.message));
}

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM || "/opt/pw-browsers/chromium",
    args: ["--no-sandbox", "--disable-features=WebRtcHideLocalIpsWithMdns", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${path.join(HERE, "fakemic.wav")}`],
  });
  const ctx = await browser.newContext({ permissions: ["camera", "microphone"], viewport: { width: 1500, height: 900 } });
  const open = async (label) => {
    const p = await ctx.newPage();
    p.on("pageerror", (e) => console.log(label, "PAGEERROR", e.message));
    await p.goto(BASE + "/");
    return p;
  };
  const A = await open("A"); // Deaf signer's browser
  const B = await open("B"); // hearing caller's browser
  await A.waitForTimeout(1200);

  check("backend reachable from the page", (await A.textContent("#backendChip")).includes("Engine online"), await A.textContent("#backendChip"));

  // --- the landing screen: each browser picks its own role, before anything else is possible -------------------
  const landingVisibleA = await A.isVisible("#roleSignBtn");
  check("the app opens on the feature/role-selection landing screen", landingVisibleA);
  check("the landing page shows the signing library before any role is picked", await A.isVisible("#landingRefgrid"));
  const landingLetters = await A.locator("#landingRefgrid .refcell").count();
  check("the landing library lists all 26 letters", landingLetters === 26, landingLetters + " letters");
  await shot(A, "00_landing");
  await A.click("#roleSignBtn");
  await B.click("#roleSpeechBtn");
  await A.waitForTimeout(200); await B.waitForTimeout(200);
  check("picking a role switches to the video-call screen", await A.isVisible("#screen-call") && await B.isVisible("#screen-call"));
  check("the role badge reflects the choice", (await A.textContent("#roleBadge")) === "signing" && (await B.textContent("#roleBadge")) === "speaking");

  // --- role selection locks each browser to one pipeline's tools, not both -------------------------------------
  const aHasSignTools = await A.isVisible("#startBtn");
  const aHasSpeechTools = await A.isVisible("#micBtn");
  const bHasSpeechTools = await B.isVisible("#micBtn");
  const bHasSignTools = await B.isVisible("#startBtn");
  check("the signer's screen shows the camera/composer tools", aHasSignTools);
  check("the signer's screen hides the microphone tools", !aHasSpeechTools);
  check("the speaker's screen shows the microphone tools", bHasSpeechTools);
  check("the speaker's screen hides the camera/composer tools", !bHasSignTools);
  check("the avatar is shown on both screens", (await A.isVisible("#avatar")) && (await B.isVisible("#avatar")));
  check("only the signer gets the avatar in the call; the speaker's is a testing preview below it",
    (await A.evaluate(`!!document.querySelector('#avatar').closest('.sidepanel')`)) &&
    (await B.evaluate(`!!document.querySelector('#avatar').closest('.avatar-preview')`)));
  check("the avatar heading says which side's speech it's signing", (await A.textContent("#avatarTitle")) === "Partner's speech, signed for you" && (await B.textContent("#avatarTitle")) === "Your speech, signed");
  // the call frame holds the video, the avatar/conversation side panel and the composer; the avatar must sit in
  // its own side panel next to the video, not over it, and the detection readout lives in the practice window.
  const insideStage = (sel) => `document.querySelector('.callstage').contains(document.querySelector('${sel}'))`;
  check("the avatar sits in the call frame's side panel", await A.evaluate(insideStage("#avatar")) && await A.evaluate(`!!document.querySelector('#avatar').closest('.sidepanel')`));
  check("the avatar panel does not overlap the video", await A.evaluate(`(()=>{const a=document.querySelector('.sidepanel').getBoundingClientRect(), v=document.querySelector('.videoarea').getBoundingClientRect(); return a.left>=v.right || a.top>=v.bottom;})()`));
  check("the letter/confidence detection readout sits in the practice window", await A.evaluate(`!!document.querySelector('#letter').closest('.practice')`));
  check("the typed-message composer sits inside the video-call frame", await A.evaluate(insideStage("#typed")));
  check("the speech transcript sits inside the video-call frame", await B.evaluate(insideStage("#transcript")));

  // --- join the same call room from both browsers -------------------------------------------------------------
  await A.fill("#roomInput", "e2e");
  await B.fill("#roomInput", "e2e");
  await A.click("#joinBtn");
  await A.waitForTimeout(500);
  await B.click("#joinBtn");
  const openedA = await A.waitForFunction("callOpen()", null, { timeout: 15000 }).then(() => true, () => false);
  const openedB = await B.waitForFunction("callOpen()", null, { timeout: 15000 }).then(() => true, () => false);
  check("WebRTC data channel opens on both sides", openedA && openedB);
  const tracks = await B.evaluate('els("remoteVideo").srcObject ? els("remoteVideo").srcObject.getTracks().length : 0');
  check("A's camera video arrives on B over the peer connection", tracks > 0, tracks + " track(s)");
  await card(A, "Video call", "01_call_connected");

  // --- Pipeline A: type a clean word letter by letter, exactly as commitLetter() does, then Speak ---------------
  await A.evaluate('clearMessage()');
  for (const ch of "HELP") await A.evaluate(([c, p]) => appendLetter(c, p), [ch, row(ch)]);
  check("typed letters render in the message bar", (await A.textContent("#typed")) === "HELP", await A.textContent("#typed"));
  await A.click("#speakBtn");
  await A.waitForFunction('els("smoothed").textContent.length>0', null, { timeout: 5000 }).catch(() => {});
  await A.waitForTimeout(600);
  const smoothed = await A.textContent("#smoothed");
  check("decoder + smoother turn HELP into a sentence and speak it", /help/i.test(smoothed), smoothed);
  await B.waitForTimeout(400);
  const bMsgs1 = await B.textContent("#remoteMsgs");
  check("the sentence reaches the hearing caller over the data channel", /help/i.test(bMsgs1), bMsgs1.slice(0, 80));
  await cardSel(A, "#typed", "02_speak_clean_word");
  await A.evaluate(() => window.scrollTo(0, 0));
  await shot(A, "07_call_screen_full"); // the single unified call frame: video, local + avatar PiPs, detection, composer, conversation

  // --- auto-speak: during a call, pausing after signing sends the sentence without pressing anything --------------
  await A.evaluate('clearMessage()');
  for (const ch of "HELP") await A.evaluate(([c, p]) => appendLetter(c, p), [ch, row(ch)]);
  const autoSent = await B.waitForFunction('(els("remoteMsgs").textContent.match(/help/gi)||[]).length>=2', null, { timeout: 8000 }).then(() => true, () => false);
  check("a paused signer's message is sent and spoken to the hearing caller automatically", autoSent);
  check("the composer clears once the sentence is sent", (await A.textContent("#typed")) === "");

  // --- an unclear letter: the decoder must ask to repeat and must not speak ------------------------------------
  await A.evaluate('clearMessage()');
  for (const ch of "QZX") await A.evaluate(([c, p]) => appendLetter(c, p), [ch, flat()]);
  const beforeCount = (await B.textContent("#remoteMsgs")).length;
  await A.click("#speakBtn");
  await A.waitForFunction('els("smoothed").textContent.includes("repeat")', null, { timeout: 5000 }).catch(() => {});
  const repeatMsg = await A.textContent("#smoothed");
  check('an unclear word makes the app ask to repeat, not guess', /please sign word|not clear/i.test(repeatMsg), repeatMsg);
  await B.waitForTimeout(400);
  const afterCount = (await B.textContent("#remoteMsgs")).length;
  check("nothing was sent to the other caller when the word was unclear", afterCount === beforeCount);
  await cardSel(A, "#smoothed", "03_please_repeat");

  // --- Pipeline B: the hearing caller's mic, the real VAD and WAV encoder, the (stubbed) Whisper endpoint ------
  await B.click("#micBtn");
  const gotUtt = await B.waitForFunction("utts.length>0", null, { timeout: 20000 }).then(() => true, () => false);
  check("the browser's own voice-activity detector and WAV encoder produce an utterance", gotUtt);
  if (gotUtt) {
    const u = await B.evaluate("utts[utts.length-1]");
    check("the utterance was transcribed and tagged with a tone", !!u.text && !!u.tone, JSON.stringify(u));
    await A.waitForTimeout(600);
    const aMsgs = await A.textContent("#remoteMsgs");
    check("the transcript reaches the Deaf signer as a captioned, tagged message", aMsgs.toLowerCase().includes(u.text.toLowerCase().split(" ")[0]), aMsgs.slice(0, 100));
    const avatar = await A.textContent("#avStatus");
    check("the signing avatar, in its own panel, plans and announces the caption", /signing|done/i.test(avatar), avatar);
  }
  await B.click("#micBtn");
  await shot(B, "08_speaker_screen_full"); // speaker's single unified call frame: video, mic/caption tools, avatar, conversation - no camera/composer
  await card(B, /^Your speech(\s|$)/, "04_speech_transcribed"); // not "Your speech, signed" - that's the avatar heading now
  await card(A, "Video call", "05_caption_received");
  await cardSel(A, "#avatarTitle", "06_avatar_signs_it");

  // --- learn & practise: "Matched" only for the right sign, and practising on the speaker's screen sends nothing ---
  const beforePractice = (await A.textContent("#remoteMsgs")).length;
  const prac = await B.evaluate(() => {
    document.querySelector('#pracModes [data-mode=phrases]').click();
    practiceShow(PRAC_ITEMS.phrases.findIndex(p => p.key === "How are you?"));
    const t = performance.now();
    onGloss("GOOD", t);                                                  // a different sign: must not match
    const wrong = els("pracFeedback").textContent;
    onGloss("HOW", t + 3000); onGloss("YOU", t + 3600);                  // the target phrase, in ASL order
    return { wrong, right: els("pracFeedback").textContent, typed: message };
  });
  check('practice only says "Matched" once the target phrase is actually signed', !/Matched/.test(prac.wrong) && /Matched/.test(prac.right), JSON.stringify(prac));
  await A.waitForTimeout(600);
  check("practising on the speaker's screen types and sends nothing", prac.typed === "" && (await A.textContent("#remoteMsgs")).length === beforePractice);

  const rtt = await A.textContent("#callRtt");
  check("the data-channel ping/pong round trip is measured", /\d/.test(rtt), rtt);

  // --- change role: leaves the call and returns to the landing screen -------------------------------------------
  await A.click("#changeRoleBtn");
  await A.waitForTimeout(300);
  check('"change role" leaves the call and returns to the feature-selection screen', await A.isVisible("#roleSignBtn"));

  await browser.close();
  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
