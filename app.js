/**
 * Video Copy Generator
 * Transcribes a video clip in-browser (Whisper via Transformers.js),
 * then generates 5 headlines and 3 descriptions using the Claude API.
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';

// Use WASM backend (works without GPU)
env.backends.onnx.wasm.proxy = false;

// ── Element refs ───────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const apiKeyInput        = $('api-key-input');
const apiKeyToggle       = $('api-key-toggle');
const iconEye            = $('icon-eye');
const iconEyeOff         = $('icon-eye-off');

const dropZone           = $('drop-zone');
const fileInput          = $('file-input');
const fileInfo           = $('file-info');
const fileName           = $('file-name');
const fileSize           = $('file-size');
const fileClear          = $('file-clear');
const btnTranscribe      = $('btn-transcribe');

const stepTranscript     = $('step-transcript');
const transcriptionStatus     = $('transcription-status');
const transcriptionStatusText = $('transcription-status-text');
const progressWrap       = $('progress-wrap');
const progressBar        = $('progress-bar');
const transcriptText     = $('transcript-text');
const transcriptHint     = $('transcript-hint');

const stepUrl            = $('step-url');
const urlInput           = $('url-input');
const btnFetch           = $('btn-fetch');
const urlStatus          = $('url-status');
const urlStatusText      = $('url-status-text');
const urlContentWrap     = $('url-content-wrap');
const urlContentText     = $('url-content-text');
const urlFetchBadge      = $('url-fetch-badge');

const stepGenerate       = $('step-generate');
const btnGenerate        = $('btn-generate');
const generateStatus     = $('generate-status');
const generateStatusText = $('generate-status-text');

const resultsSection     = $('results');
const btnCopyAll         = $('btn-copy-all');
const headlinesGrid      = $('headlines-grid');
const descriptionsList   = $('descriptions-list');
const toast              = $('toast');

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile = null;
let whisperPipeline = null;
let toastTimer = null;

// ── Init ──────────────────────────────────────────────────────────────────────
function initApp() {
  // Restore saved API key
  const savedKey = localStorage.getItem('anthropic_api_key');
  if (savedKey) apiKeyInput.value = savedKey;

  // Save key on change
  apiKeyInput.addEventListener('input', () => {
    localStorage.setItem('anthropic_api_key', apiKeyInput.value.trim());
  });

  // Show/hide API key
  apiKeyToggle.addEventListener('click', () => {
    const isHidden = apiKeyInput.type === 'password';
    apiKeyInput.type = isHidden ? 'text' : 'password';
    iconEye.style.display = isHidden ? 'none' : '';
    iconEyeOff.style.display = isHidden ? '' : 'none';
  });

  // Drop zone
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });
  fileClear.addEventListener('click', clearFile);

  // Transcribe
  btnTranscribe.addEventListener('click', handleTranscribe);

  // URL fetch
  btnFetch.addEventListener('click', handleFetchUrl);
  urlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') handleFetchUrl(); });

  // Generate
  btnGenerate.addEventListener('click', handleGenerate);

  // Copy all
  btnCopyAll.addEventListener('click', handleCopyAll);
}

// ── File handling ─────────────────────────────────────────────────────────────
function setFile(file) {
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileInfo.hidden = false;
  btnTranscribe.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  fileInfo.hidden = true;
  btnTranscribe.disabled = true;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ── Audio extraction ──────────────────────────────────────────────────────────
async function extractAudioFloat32(file) {
  const arrayBuffer = await file.arrayBuffer();
  // Create context at 16 kHz — Whisper's required sample rate
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  let audioBuffer;
  try {
    audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  } catch {
    await audioCtx.close();
    throw new Error('Could not decode audio from this file. Try MP4 or WebM.');
  }
  // Mix to mono: average all channels into channel 0
  const numChannels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const mono = new Float32Array(length);
  for (let c = 0; c < numChannels; c++) {
    const channelData = audioBuffer.getChannelData(c);
    for (let i = 0; i < length; i++) mono[i] += channelData[i];
  }
  for (let i = 0; i < length; i++) mono[i] /= numChannels;

  // Resample if AudioContext ignored our sampleRate hint
  const result = audioBuffer.sampleRate !== 16000
    ? resampleLinear(mono, audioBuffer.sampleRate, 16000)
    : mono;

  await audioCtx.close();
  return result;
}

function resampleLinear(input, fromRate, toRate) {
  const ratio = fromRate / toRate;
  const outputLength = Math.round(input.length / ratio);
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const pos = i * ratio;
    const low = Math.floor(pos);
    const high = Math.min(low + 1, input.length - 1);
    const frac = pos - low;
    output[i] = input[low] * (1 - frac) + input[high] * frac;
  }
  return output;
}

// ── Transcription ─────────────────────────────────────────────────────────────
async function handleTranscribe() {
  if (!selectedFile) return;

  // Show transcript step
  stepTranscript.hidden = false;
  transcriptionStatus.hidden = false;
  progressWrap.hidden = false;
  transcriptText.hidden = true;
  transcriptHint.hidden = true;
  setProgress(0);
  setTranscriptStatus('Extracting audio from video…');
  btnTranscribe.disabled = true;

  let audioData;
  try {
    audioData = await extractAudioFloat32(selectedFile);
  } catch (err) {
    showToast(err.message, 'error');
    setTranscriptStatus('Audio extraction failed.');
    btnTranscribe.disabled = false;
    return;
  }

  setTranscriptStatus('Loading Whisper model… (first run downloads ~150 MB, cached after)');
  setProgress(5);

  try {
    if (!whisperPipeline) {
      whisperPipeline = await pipeline(
        'automatic-speech-recognition',
        'Xenova/whisper-small',
        {
          progress_callback: (info) => {
            if (info.status === 'downloading') {
              const pct = info.loaded && info.total
                ? Math.round((info.loaded / info.total) * 60)
                : 0;
              setProgress(5 + pct);
              setTranscriptStatus(`Downloading model… ${pct}%`);
            } else if (info.status === 'loading') {
              setProgress(65);
              setTranscriptStatus('Loading model into memory…');
            }
          }
        }
      );
    }

    setProgress(70);
    setTranscriptStatus('Transcribing audio…');

    const result = await whisperPipeline(audioData, {
      chunk_length_s: 30,
      stride_length_s: 5,
      language: 'english',
      task: 'transcribe',
    });

    setProgress(100);

    const text = (result.text || '').trim();
    transcriptText.value = text;
    transcriptText.hidden = false;
    transcriptHint.hidden = false;
    transcriptionStatus.hidden = true;
    progressWrap.hidden = true;

    // Unlock next steps
    stepUrl.hidden = false;
    stepGenerate.hidden = false;

    showToast('Transcript ready!', 'success');
  } catch (err) {
    console.error(err);
    showToast('Transcription failed: ' + err.message, 'error');
    setTranscriptStatus('Transcription failed. Please try again.');
  } finally {
    btnTranscribe.disabled = false;
  }
}

function setTranscriptStatus(msg) {
  transcriptionStatusText.textContent = msg;
}

function setProgress(pct) {
  progressBar.style.width = `${pct}%`;
  progressBar.setAttribute('aria-valuenow', pct);
}

// ── URL fetch ─────────────────────────────────────────────────────────────────
async function handleFetchUrl() {
  const url = urlInput.value.trim();
  if (!url) { showToast('Please enter a URL first.', 'error'); return; }

  urlStatus.hidden = false;
  urlStatusText.textContent = 'Fetching page…';
  btnFetch.disabled = true;

  try {
    const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`;
    const res = await fetch(proxyUrl, { signal: AbortSignal.timeout(15000) });
    if (!res.ok) throw new Error(`Proxy returned ${res.status}`);
    const json = await res.json();
    if (!json.contents) throw new Error('No content received');

    const text = stripHtml(json.contents).slice(0, 6000);
    urlContentText.value = text;
    urlFetchBadge.textContent = 'auto';
    urlFetchBadge.classList.remove('manual');
    urlContentWrap.hidden = false;
    urlStatus.hidden = true;
    showToast('Page content fetched.', 'success');
  } catch (err) {
    console.warn('URL fetch failed:', err.message);
    urlStatus.hidden = true;
    // Show fallback text area for manual paste
    urlFetchBadge.textContent = 'manual';
    urlFetchBadge.classList.add('manual');
    urlContentText.value = '';
    urlContentText.placeholder = 'Auto-fetch failed (CORS restriction or network error).\nPaste the relevant text from that page here instead.';
    urlContentWrap.hidden = false;
    showToast('Could not auto-fetch the URL. Paste the content manually.', 'error');
  } finally {
    btnFetch.disabled = false;
  }
}

function stripHtml(html) {
  // Remove script/style blocks
  let text = html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s{2,}/g, ' ')
    .trim();
  return text;
}

// ── Generate copy ─────────────────────────────────────────────────────────────
async function handleGenerate() {
  const apiKey = apiKeyInput.value.trim();
  if (!apiKey) {
    showToast('Please enter your Anthropic API key in Step 1.', 'error');
    apiKeyInput.focus();
    return;
  }
  const transcript = transcriptText.value.trim();
  if (!transcript) {
    showToast('Transcript is empty. Please transcribe a video first.', 'error');
    return;
  }

  const url = urlInput.value.trim() || '(not provided)';
  const urlContent = urlContentText.value.trim() || '(not provided)';

  generateStatus.hidden = false;
  generateStatusText.textContent = 'Calling Claude…';
  btnGenerate.disabled = true;
  resultsSection.hidden = true;

  try {
    const result = await callClaude(apiKey, transcript, url, urlContent);
    renderResults(result);
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    showToast('Copy generated!', 'success');
  } catch (err) {
    console.error(err);
    showToast(err.message, 'error');
  } finally {
    generateStatus.hidden = true;
    btnGenerate.disabled = false;
  }
}

async function callClaude(apiKey, transcript, url, urlContent) {
  const prompt = buildPrompt(transcript, url, urlContent);

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-6',
      max_tokens: 2048,
      messages: [{ role: 'user', content: prompt }],
    }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const msg = body?.error?.message || `HTTP ${res.status}`;
    if (res.status === 401) throw new Error('Invalid API key. Check your key in Step 1.');
    if (res.status === 429) throw new Error('Rate limit reached. Wait a moment and try again.');
    throw new Error(`Claude API error: ${msg}`);
  }

  const data = await res.json();
  const rawText = data?.content?.[0]?.text || '';
  return parseCopyResponse(rawText);
}

function buildPrompt(transcript, url, urlContent) {
  return `You are a social media copywriter creating clip captions in the style of TED's Instagram/social team.

TRANSCRIPT OF VIDEO CLIP:
"""
${transcript}
"""

ADDITIONAL CONTEXT FROM: ${url}
"""
${urlContent}
"""

Using the transcript and context above, generate copy in the following specific styles:

---

HEADLINE STYLE GUIDE:
- Short (5–9 words), conversational, social-media-native
- Use sentence case or all-lowercase — never title case
- Lead with question words like "How", "What", "Why", "Do", "Does", "Will", "Can"
- Include "you" or "how" frequently
- Be provocative, direct, and surprising
- Examples of the RIGHT tone:
  * "Do violent lyrics cause violence?"
  * "Want to be happier? Try this."
  * "how humor fights misinformation"
  * "What will people do when AI takes over?"
  * "How to become a K-pop superstar"

DESCRIPTION STYLE GUIDE:
- 2–4 sentences per description
- Structure: Hook or intriguing statement → Speaker name + role → Key insight or takeaway from the clip → Call to action
- Mention specific details from the content (techniques, numbers, named concepts)
- Warm, human, conversational — not corporate
- Always end with one of these CTAs (vary them across the 3 descriptions):
  * "Visit the link in our bio to watch the full talk."
  * "Visit the link in our bio to learn more."
  * "Visit the link in our bio to watch the rest."
- Examples of the RIGHT tone:
  * "Chances are you already measure biomarkers like your steps and sleep, but what if you measured all facets of your life? In his TED Talk, management consultant Chris Musser explains how he uses a simple tracker to monitor progress across nine dimensions — and how it has vastly improved his general outlook on life. Visit the link in our bio to watch the full talk."
  * "We're giving up the beautiful messiness of being human when we depend too much on AI relationships, says sextech expert Bryony Cole. In her TED Talk, she offers 3 questions to ask yourself if you're already intimate with AI — and lays out a playbook for synthetic companionship that doesn't hide you from the friction of human life. Visit the link in our bio to learn more."

---

Format your response EXACTLY as follows (no extra commentary, just this structure):

HEADLINES:
1. [headline]
2. [headline]
3. [headline]
4. [headline]
5. [headline]

DESCRIPTIONS:
1. [paragraph]
2. [paragraph]
3. [paragraph]`;
}

function parseCopyResponse(raw) {
  const headlines = [];
  const descriptions = [];

  const headlineSection = raw.match(/HEADLINES:\s*([\s\S]*?)(?=DESCRIPTIONS:|$)/i);
  const descSection = raw.match(/DESCRIPTIONS:\s*([\s\S]*?)$/i);

  if (headlineSection) {
    const lines = headlineSection[1].split('\n');
    for (const line of lines) {
      const m = line.match(/^\s*\d+\.\s+(.+)/);
      if (m) headlines.push(m[1].trim());
    }
  }

  if (descSection) {
    // Descriptions can span multiple lines — split by numbered item
    const items = descSection[1].split(/\n\s*\d+\.\s+/);
    for (const item of items) {
      const text = item.trim();
      if (text) descriptions.push(text);
    }
  }

  if (!headlines.length || !descriptions.length) {
    throw new Error('Could not parse Claude's response. The raw response has been logged to the console.');
  }

  return { headlines, descriptions };
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults({ headlines, descriptions }) {
  headlinesGrid.innerHTML = '';
  descriptionsList.innerHTML = '';

  headlines.forEach((h) => {
    const card = document.createElement('div');
    card.className = 'headline-card';
    card.innerHTML = `
      <span>${escapeHtml(h)}</span>
      <button class="copy-btn" data-text="${escapeAttr(h)}">Copy</button>
    `;
    headlinesGrid.appendChild(card);
  });

  descriptions.forEach((d) => {
    const card = document.createElement('div');
    card.className = 'description-card';
    card.innerHTML = `
      <p>${escapeHtml(d)}</p>
      <div class="description-card-footer">
        <button class="copy-btn" data-text="${escapeAttr(d)}">Copy</button>
      </div>
    `;
    descriptionsList.appendChild(card);
  });

  // Attach copy button handlers
  resultsSection.querySelectorAll('.copy-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      await copyToClipboard(btn.dataset.text, btn);
    });
  });
}

async function handleCopyAll() {
  const headlines = [...headlinesGrid.querySelectorAll('.headline-card span')].map((el) => el.textContent);
  const descriptions = [...descriptionsList.querySelectorAll('.description-card p')].map((el) => el.textContent);

  const allText = [
    'HEADLINES:',
    ...headlines.map((h, i) => `${i + 1}. ${h}`),
    '',
    'DESCRIPTIONS:',
    ...descriptions.map((d, i) => `${i + 1}. ${d}`),
  ].join('\n');

  await copyToClipboard(allText, btnCopyAll);
}

async function copyToClipboard(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = orig;
        btn.classList.remove('copied');
      }, 1800);
    }
    showToast('Copied to clipboard!', 'success');
  } catch {
    showToast('Copy failed — please select and copy manually.', 'error');
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  if (toastTimer) clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.className = `toast visible${type !== 'info' ? ' ' + type : ''}`;
  toastTimer = setTimeout(() => {
    toast.classList.remove('visible');
  }, 3500);
}

// ── HTML helpers ──────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escapeAttr(str) {
  return str.replace(/"/g, '&quot;').replace(/\n/g, ' ');
}

// ── Start ─────────────────────────────────────────────────────────────────────
initApp();
