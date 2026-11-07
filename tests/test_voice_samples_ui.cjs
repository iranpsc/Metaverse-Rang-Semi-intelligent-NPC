const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
let source = fs.readFileSync('LLM/templates/voice_samples.html', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
source = source.replaceAll(/\{% url '[^']+' %\}/g, '/api/voice-samples/').replace('{{ max_voice_sample_bytes }}', '26214400');
const elements = {};
const document = {getElementById(id) { return elements[id] ||= {value: '', disabled: false, textContent: '', innerHTML: '', classList: {add() {}, remove() {}}, addEventListener() {}, focus() {}, querySelectorAll() {return [];}}; }};
const context = vm.createContext({document, console: {error() {}}, FormData, File, Blob, TypeError, clearInterval, fetch: async () => ({ok: true, json: async () => ({samples: []})})});
vm.runInContext(source, context);
async function saveWith(response) {
  context.fetch = async () => {if (response instanceof Error) throw response; return response;};
  vm.runInContext('nameInput.value = "Narrator"; recordedBlob = new Blob(["recording"]); saveBtn.disabled = false;', context);
  await vm.runInContext('saveSample()', context);
}
(async () => {
  await saveWith({ok: false, status: 413, json: async () => {throw new SyntaxError('HTML');}});
  assert.match(elements['vs-status'].textContent, /server rejected the recording size/);
  assert.equal(elements['vs-save'].disabled, false);
  assert.equal(vm.runInContext('recordedBlob.size', context), 9);
  await saveWith({ok: false, status: 502, json: async () => {throw new SyntaxError('HTML');}});
  assert.match(elements['vs-status'].textContent, /HTTP 502/);
  await saveWith({ok: false, status: 400, json: async () => ({error: 'recording is too large'})});
  assert.match(elements['vs-status'].textContent, /recording is too large/);
  await saveWith(new TypeError('Failed to fetch'));
  assert.match(elements['vs-status'].textContent, /recording is still available/);
  assert.equal(elements['vs-save'].disabled, false);
  let called = false;
  context.fetch = async () => {called = true;};
  vm.runInContext('recordedBlob = new Blob([new Uint8Array(26214401)]);', context);
  await vm.runInContext('saveSample()', context);
  assert.equal(called, false);
  assert.match(elements['vs-status'].textContent, /shorter clip/);
  await saveWith({ok: true, status: 201, json: async () => ({sample: {name: '<test>'}, samples: []})});
  assert.equal(elements['vs-status'].textContent, '✅ Saved <test>.');
  assert.equal(vm.runInContext('recordedBlob', context), null);
  console.log('Passed: HTML 413/502, JSON validation, network failure, size guard, successful save.');
})().catch(error => {console.error(error); process.exitCode = 1;});
