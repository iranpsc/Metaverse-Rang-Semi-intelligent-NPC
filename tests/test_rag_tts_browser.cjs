// Run with Playwright available on NODE_PATH and Chromium installed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const { chromium } = require('playwright');

const template = fs.readFileSync('LLM/templates/rag_chat.html', 'utf8');
const urls = {rag_chat_api:'/api/rag-chat/', tts_api:'/api/tts/', vc_convert_api:'/api/vc/',
  voice_samples_api:'/api/voice-samples/', list_vector_stores_api:'/api/vector-stores/list/'};
const render = text => text.replace(/\{% url '([^']+)' %\}/g, (_, name) => urls[name] || '/unused/')
  .replace(/\{\{[\s\S]*?\}\}/g, 'browser-test').replace(/\{%[\s\S]*?%\}/g, '');
const content = template.match(/\{% block content %\}([\s\S]*?)\{% endblock %\}/)[1];
const script = template.match(/\{% block extra_js %\}([\s\S]*?)\{% endblock %\}/)[1];
const base = fs.readFileSync('STT/templates/base.html', 'utf8');
const blocks = {title:'RAG audio test',content,extra_js:script};
const html = render(base.replace(/\{% if messages %\}[\s\S]*?\{% endif %\}/, '')
  .replace(/\{% block (\w+) %\}[\s\S]*?\{% endblock %\}/g, (_,name)=>blocks[name] || ''));
const wav = Buffer.alloc(44 + 4410 * 2);
wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
wav.writeUInt32LE(22050, 24); wav.writeUInt32LE(44100, 28); wav.writeUInt16LE(2, 32);
wav.writeUInt16LE(16, 34); wav.write('data', 36); wav.writeUInt32LE(wav.length - 44, 40);
for (let i=0;i<4410;i++) wav.writeInt16LE(Math.round(1000 * Math.sin(2 * Math.PI * 440 * i / 22050)), 44 + i*2);

let failure = false;
let ttsCalls = 0;
let conversions = 0;
let conversionFailure = '';
let convertedSource = null;
const convertedWav = Buffer.from(wav);
for (let i=0;i<4410;i++) convertedWav.writeInt16LE(Math.round(1000 * Math.sin(2 * Math.PI * 880 * i / 22050)), 44 + i*2);
const server = http.createServer((req, res) => {
  if (req.url === '/') {res.setHeader('Content-Type','text/html; charset=utf-8'); return res.end(html);}
  if (req.url === '/api/rag-chat/') {
    res.setHeader('Content-Type','text/event-stream');
    return res.end([{type:'start'}, {type:'token',token:'سلام.'}, {type:'done',chunk_count:1}]
      .map(event => 'data: ' + JSON.stringify(event) + '\n\n').join(''));
  }
  if (req.url === '/api/tts/' || req.url.startsWith('/api/vc/')) {
    if (req.url === '/api/tts/') ttsCalls++; else conversions++;
    if (failure) {res.statusCode=503; res.setHeader('Content-Type','application/json'); return res.end('{"error":"unavailable"}');}
    if (req.url.startsWith('/api/vc/')) {
      const body=[];
      req.on('data', chunk => body.push(chunk));
      req.on('end', () => {
        convertedSource=Buffer.concat(body);
        if (conversionFailure === 'http') {res.statusCode=503;res.setHeader('Content-Type','application/json');return res.end('{"error":"conversion unavailable"}');}
        if (conversionFailure === 'invalid') {res.setHeader('Content-Type','application/json');return res.end('{}');}
        res.setHeader('Content-Type','audio/wav');res.end(conversionFailure === 'empty' ? Buffer.alloc(0) : convertedWav);
      });
      return;
    }
    res.setHeader('Content-Type','audio/wav'); return res.end(wav);
  }
  res.setHeader('Content-Type','application/json');
  res.end(JSON.stringify(req.url === '/api/voice-samples/' ? {samples:[{id:1,name:'Test voice'}]}
    : {stores:[{path:'test',name:'Test knowledge base'}]}));
});

(async () => {
  await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
  const browser = await chromium.launch({headless:true,args:['--no-sandbox','--autoplay-policy=user-gesture-required']});
  const origin = 'http://127.0.0.1:' + server.address().port;
  async function pageFor(mode = '') {
    const context = await browser.newContext();
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    // Exercise the text fallback as well as audio: no external CDN is required.
    await page.route('https://**', route => route.abort());
    await page.addInitScript(mode => {
      window.__starts = 0; window.__ended = 0; window.__native = 0;
      const start = AudioBufferSourceNode.prototype.start;
      AudioBufferSourceNode.prototype.start = function(...args) {
        window.__starts++;
        this.addEventListener('ended', () => window.__ended++);
        return start.apply(this,args);
      };
      document.addEventListener('playing', event => {if(event.target.tagName === 'AUDIO') window.__native++;}, true);
      if (mode === 'muted') localStorage.setItem('ttsEnabled','off');
      if (mode === 'conversion') {localStorage.setItem('vcEnabled','on');localStorage.setItem('vcSampleId','1');}
      if (mode === 'slow-resume') {
        const Real = window.AudioContext;
        window.AudioContext = class extends Real {
          constructor(...args) {super(...args);this.pending = true;}
          get state() {return this.pending ? 'suspended' : super.state;}
          resume() {const result=super.resume();return new Promise(resolve=>setTimeout(()=>{this.pending=false;resolve(result);},250));}
        };
      }
      if (mode === 'native' || mode === 'blocked') {
        window.AudioContext = class {constructor(){throw new Error('Web Audio output unavailable');}};
        window.webkitAudioContext = undefined;
      }
      if (mode === 'blocked') {
        const play = HTMLMediaElement.prototype.play;
        let first = true;
        HTMLMediaElement.prototype.play = function() {
          if (first) {first=false;return Promise.reject(new DOMException('User gesture required','NotAllowedError'));}
          return play.call(this);
        };
      }
      if (mode === 'slow-decode') {
        const decode = BaseAudioContext.prototype.decodeAudioData;
        BaseAudioContext.prototype.decodeAudioData = async function(...args) {
          const result=await decode.apply(this,args);window.__decoding=true;
          await new Promise(resolve=>setTimeout(resolve,300));return result;
        };
      }
    }, mode);
    await page.goto(origin);
    await page.waitForFunction(() => document.querySelector('#vector-store-select').value === 'test');
    return {page, context, errors};
  }
  async function send(page) {await page.locator('#message-input').fill('سلام');await page.locator('#send-btn').click();}
  async function played(page) {await page.waitForFunction(()=>window.__ended > 0 || window.__native > 0);}
  async function scenario(name, mode, run) {
    const {page, context, errors} = await pageFor(mode);
    try {await run(page);assert.deepEqual(errors,[]);console.log('Passed:',name);}
    finally {await context.close();failure=false;conversionFailure='';}
  }
  try {
    await scenario('full chat stream plays even when Markdown CDN is unavailable','',async page=>{
      await send(page);await played(page);
      assert.match(await page.locator('#chat-box').textContent(),/سلام\./);
    });
    await scenario('saved mute is explicit; replay enables and reads the existing answer','muted',async page=>{
      const before=ttsCalls;await send(page);
      await page.waitForFunction(()=>!document.querySelector('#voice-replay-btn').disabled);
      assert.equal(ttsCalls,before);
      assert.match(await page.locator('#voice-btn').textContent(),/خاموش/);
      await page.locator('#voice-replay-btn').click();await played(page);
      assert.equal(await page.evaluate(()=>localStorage.getItem('ttsEnabled')),'on');
    });
    await scenario('pending AudioContext.resume is awaited','slow-resume',async page=>{await send(page);await played(page);});
    await scenario('explicit sound test exposes a real native audio player','',async page=>{
      await page.locator('#voice-test-btn').click();await played(page);
      assert.equal(await page.locator('#tts-audio').isVisible(),true);
    });
    await scenario('visible synthesis failure can be retried with the test button','',async page=>{
      failure=true;await send(page);
      await page.waitForFunction(()=>document.querySelector('#tts-status').dataset.state==='error');
      assert.match(await page.locator('#tts-status').textContent(),/503/);
      failure=false;await page.locator('#voice-test-btn').click();await played(page);
    });
    await scenario('native player works when Web Audio is unavailable','native',async page=>{
      await page.locator('#voice-test-btn').click();await played(page);
      assert.equal(await page.locator('#tts-audio').isVisible(),true);
    });
    await scenario('autoplay denial retains the audio for manual playback','blocked',async page=>{
      await send(page);
      await page.waitForFunction(()=>document.querySelector('#tts-status').dataset.state==='blocked');
      assert.equal(await page.locator('#tts-audio').isVisible(),true);
      await page.locator('#tts-audio').press('Space');await played(page);
      await page.waitForFunction(()=>document.querySelector('#tts-status').dataset.state==='ready');
    });
    await scenario('muting during decoding prevents stale speech','slow-decode',async page=>{
      await send(page);await page.waitForFunction(()=>window.__decoding);
      await page.locator('#voice-btn').click();
      await page.evaluate(()=>ttsPlaybackQueue);
      await page.waitForTimeout(400);
      assert.equal(await page.evaluate(()=>window.__starts),0);
    });
    await scenario('selected voice conversion feeds the playback queue','conversion',async page=>{
      const before=conversions;await page.locator('#voice-test-btn').click();await played(page);
      await page.evaluate(()=>ttsPlaybackQueue);
      assert.equal(conversions,before+1);
      assert.deepEqual(convertedSource,wav);
      const bytes=await page.evaluate(async()=>Array.from(new Uint8Array(await (await fetch(document.querySelector('#tts-audio').src)).arrayBuffer())));
      assert.deepEqual(Buffer.from(bytes),convertedWav);
      assert.equal(await page.locator('#tts-status').getAttribute('data-state'),'ready');
    });
    for (const kind of ['http','invalid','empty']) {
      await scenario(`conversion ${kind} failure keeps original speech and displays a warning`,'conversion',async page=>{
        conversionFailure=kind;
        await page.locator('#voice-test-btn').click();await played(page);
        await page.evaluate(()=>ttsPlaybackQueue);
        assert.equal(await page.locator('#tts-status').getAttribute('data-state'),'warning');
        assert.match(await page.locator('#tts-status').textContent(),/تبدیل صدا انجام نشد/);
        const bytes=await page.evaluate(async()=>Array.from(new Uint8Array(await (await fetch(document.querySelector('#tts-audio').src)).arrayBuffer())));
        assert.deepEqual(Buffer.from(bytes),wav);
        conversionFailure='';
        await page.locator('#voice-test-btn').click();await page.evaluate(()=>ttsPlaybackQueue);
        assert.equal(await page.locator('#tts-status').getAttribute('data-state'),'ready');
      });
    }
  } finally {await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});
