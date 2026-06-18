function mcDropClip(event, label) {
  event.preventDefault();
  const clipId = event.dataTransfer.getData('text/plain');
  fetch('/api/move_clip', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({clip_id: clipId, label})
  }).then(() => window.location.reload());
}
function mcDragClip(event, clipId) {
  event.dataTransfer.effectAllowed = 'move';
  event.dataTransfer.setData('text/plain', clipId);
  event.dataTransfer.setData('application/x-clip-id', clipId);
}
function mcRenameClip(clipId, value) {
  fetch('/api/rename_clip', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({clip_id: clipId, display_name: value})
  });
}
function mcUpdateCategory(categoryId, name, description) {
  fetch('/api/update_category', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({category_id: categoryId, name, description})
  }).then(() => setTimeout(() => window.location.reload(), 120));
}
function mcPlay(clipId) {
  const escapedClipId = window.CSS && CSS.escape ? CSS.escape(clipId) : clipId.replace(/"/g, '\\"');
  const row = document.querySelector(`.clip-row[data-clip-id="${escapedClipId}"]`);
  const url = row && row.dataset.mediaUrl ? row.dataset.mediaUrl : '/media/' + encodeURIComponent(clipId);
  let audio = document.getElementById('mc-global-audio');
  if (!audio) {
    audio = document.createElement('audio');
    audio.id = 'mc-global-audio';
    audio.style.display = 'none';
    audio.addEventListener('ended', mcClearPlaying);
    audio.addEventListener('pause', () => {
      if (audio.dataset.manualPause === 'true') mcClearPlaying();
    });
    document.body.appendChild(audio);
  }
  if (audio.dataset.clipId === clipId && !audio.paused) {
    audio.dataset.manualPause = 'true';
    audio.pause();
    return;
  }
  const playToken = String(Date.now()) + '-' + Math.random().toString(16).slice(2);
  mcClearPlaying();
  audio.dataset.manualPause = 'false';
  audio.dataset.clipId = clipId;
  audio.dataset.playToken = playToken;
  audio.preload = 'auto';
  audio.onerror = () => {
    if (audio.dataset.playToken !== playToken) return;
    const error = audio.error;
    const errorMap = {
      1: 'MEDIA_ERR_ABORTED',
      2: 'MEDIA_ERR_NETWORK',
      3: 'MEDIA_ERR_DECODE',
      4: 'MEDIA_ERR_SRC_NOT_SUPPORTED',
    };
    const code = error ? `${error.code} ${errorMap[error.code] || ''}` : 'unknown';
    alert(
      '音频加载失败：' + code +
      '\\nnetworkState=' + audio.networkState +
      ' readyState=' + audio.readyState +
      '\\n请直接打开 ' + url + ' 检查接口。'
    );
  };
  const absoluteUrl = new URL(url, window.location.href).href;
  const switchingSource = audio.src !== absoluteUrl;
  if (switchingSource) {
    audio.removeAttribute('src');
    audio.load();
    audio.src = absoluteUrl;
    audio.load();
  } else {
    try {
      if (audio.ended || audio.currentTime > 0) audio.currentTime = 0;
    } catch (error) {
      audio.load();
    }
  }
  audio.play()
    .then(() => {
      if (audio.dataset.playToken !== playToken) return;
      if (row) {
        row.classList.add('is-playing');
        const button = row.querySelector('.play-btn');
        if (button) button.textContent = '❚❚';
      }
    })
    .catch(error => {
      if (audio.dataset.playToken !== playToken) return;
      const message = error && error.message ? error.message : '';
      if (error && error.name === 'AbortError') return;
      if (message.includes('interrupted by a new load request')) return;
      if (audio.dataset.retriedPlay !== playToken) {
        audio.dataset.retriedPlay = playToken;
        audio.removeAttribute('src');
        audio.load();
        audio.src = absoluteUrl;
        audio.load();
        audio.play().catch(() => {
          if (audio.dataset.playToken !== playToken) return;
          alert(
            '播放失败：' + message +
            '\\nnetworkState=' + audio.networkState +
            ' readyState=' + audio.readyState +
            '\\nURL=' + url
          );
        });
        return;
      }
      alert(
        '播放失败：' + message +
        '\\nnetworkState=' + audio.networkState +
        ' readyState=' + audio.readyState +
        '\\nURL=' + url
      );
    });
}
function mcClearPlaying() {
  document.querySelectorAll('.clip-row.is-playing').forEach(row => {
    row.classList.remove('is-playing');
    const button = row.querySelector('.play-btn');
    if (button) button.textContent = '▶';
  });
  const audio = document.getElementById('mc-global-audio');
  if (audio) audio.dataset.clipId = '';
}
function mcResetRuntimeState() {
  mcClearPlaying();
  const globalAudio = document.getElementById('mc-global-audio');
  if (globalAudio) {
    globalAudio.pause();
    globalAudio.removeAttribute('src');
    globalAudio.load();
    globalAudio.dataset.clipId = '';
    globalAudio.dataset.playToken = '';
  }
  const editAudio = document.getElementById('mc-edit-audio');
  if (editAudio) {
    editAudio.pause();
    editAudio.removeAttribute('src');
    editAudio.load();
  }
  window.mcEditRegions = null;
  window.mcEditCategories = null;
  document.querySelectorAll('#mc-editor-root').forEach(root => {
    root.dataset.ready = 'false';
  });
}
function mcAddColumn() {
  fetch('/api/add_category', {method: 'POST'}).then(() => window.location.reload());
}
function mcInputKey(event, original) {
  if (event.key === 'Enter') event.target.blur();
  if (event.key === 'Escape') {
    event.target.value = original;
    event.target.blur();
  }
}
function mcSelectedClipIds() {
  return Array.from(document.querySelectorAll('.clip-check:checked')).map(input => input.dataset.clipId);
}
function mcVisibleClipIds() {
  return Array.from(document.querySelectorAll('.clip-row')).map(row => row.dataset.clipId).filter(Boolean);
}
function mcSelectVisible(checked) {
  document.querySelectorAll('.clip-check').forEach(input => input.checked = checked);
}
function mcBatchUpdate(action) {
  let clipIds = mcSelectedClipIds();
  if (action === 'confirm') {
    clipIds = mcVisibleClipIds();
  }
  if (clipIds.length === 0) {
    alert(action === 'confirm' ? '当前没有可完成的片段。' : '请先勾选片段。');
    return;
  }
  fetch('/api/batch_update_clips', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({clip_ids: clipIds, action})
  }).then(() => window.location.reload());
}
function mcDropToEditor(event) {
  event.preventDefault();
  const clipId = event.dataTransfer.getData('application/x-clip-id') || event.dataTransfer.getData('text/plain');
  if (!clipId) return;
  mcStartEditClip(clipId);
}
function mcStartEditClip(clipId) {
  fetch('/api/start_edit_clip', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({clip_id: clipId})
  }).then(async response => {
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || ('HTTP ' + response.status));
    return data;
  }).then(() => window.location.reload())
    .catch(error => alert('进入编辑区失败：' + error.message));
}
async function mcInitWaveEditor() {
  const root = document.getElementById('mc-editor-root');
  if (!root || root.dataset.ready === 'true') return;
  root.dataset.ready = 'true';
  const audio = document.getElementById('mc-edit-audio');
  const canvas = document.getElementById('mc-wave-canvas');
  const layer = document.getElementById('mc-region-layer');
  const table = document.getElementById('mc-region-table');
  const regions = JSON.parse(root.dataset.regions || '[]');
  const categories = JSON.parse(root.dataset.categories || '[]');
  window.mcEditRegions = regions;
  window.mcEditCategories = categories;

  const syncDuration = () => {
    const duration = audio.duration || Math.max(...regions.map(r => Number(r.end_sec || 0)), 1);
    root.dataset.duration = String(duration);
    mcRenderRegions();
    mcRenderRegionTable();
  };
  audio.addEventListener('loadedmetadata', syncDuration);
  if (audio.readyState >= 1) syncDuration();

  try {
    const response = await fetch(root.dataset.sourceUrl);
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const buffer = await response.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(buffer.slice(0));
    mcDrawWaveform(canvas, audioBuffer);
    if (!audio.duration) {
      root.dataset.duration = String(audioBuffer.duration);
      mcRenderRegions();
      mcRenderRegionTable();
    }
    audioContext.close();
  } catch (error) {
    const ctx = canvas.getContext('2d');
    ctx.font = '13px sans-serif';
    ctx.fillStyle = '#b42318';
    ctx.fillText('波形加载失败：' + error.message, 16, 32);
  }

  window.addEventListener('resize', () => {
    if (window.mcEditRegions) mcRenderRegions();
  });
}
function mcDrawWaveform(canvas, audioBuffer) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(600, Math.floor(rect.width * window.devicePixelRatio));
  const height = Math.max(120, Math.floor(rect.height * window.devicePixelRatio));
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#d8dde3';
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
  const data = audioBuffer.getChannelData(0);
  const step = Math.ceil(data.length / width);
  ctx.strokeStyle = '#52606d';
  ctx.beginPath();
  for (let x = 0; x < width; x++) {
    let min = 1;
    let max = -1;
    const start = x * step;
    const end = Math.min(start + step, data.length);
    for (let i = start; i < end; i++) {
      const value = data[i];
      if (value < min) min = value;
      if (value > max) max = value;
    }
    ctx.moveTo(x, (1 + min) * height / 2);
    ctx.lineTo(x, (1 + max) * height / 2);
  }
  ctx.stroke();
}
function mcRenderRegions() {
  const root = document.getElementById('mc-editor-root');
  const layer = document.getElementById('mc-region-layer');
  if (!root || !layer || !window.mcEditRegions) return;
  const duration = Number(root.dataset.duration || 1);
  layer.innerHTML = '';
  window.mcEditRegions.forEach((region, index) => {
    const el = document.createElement('div');
    el.className = 'edit-region';
    el.dataset.index = index;
    mcApplyRegionElementPosition(el, region, duration);
    const color = mcRegionColor(region.label, 0.9);
    el.style.borderColor = color;
    el.style.background = mcRegionColor(region.label, 0.18);
    el.innerHTML = `<div class="region-handle left"></div><div class="region-label">${region.display_name || region.clip_id}</div><div class="region-handle right"></div>`;
    el.querySelectorAll('.region-handle').forEach(handle => handle.style.background = color);
    mcAttachRegionDrag(el, region);
    layer.appendChild(el);
  });
}
function mcApplyRegionElementPosition(el, region, duration) {
  const safeDuration = Math.max(1, Number(duration || 1));
  const left = Math.max(0, Number(region.start_sec || 0) / safeDuration * 100);
  const right = Math.min(100, Number(region.end_sec || 0) / safeDuration * 100);
  el.style.left = left + '%';
  el.style.width = Math.max(0.5, right - left) + '%';
}
function mcRegionColor(label, alpha) {
  const palette = ['37,99,235', '14,165,163', '124,58,237', '217,119,6', '5,150,105', '219,39,119'];
  const categories = window.mcEditCategories || [];
  const index = Math.max(0, categories.findIndex(category => category.name === label));
  const rgb = palette[index % palette.length];
  return `rgba(${rgb}, ${alpha})`;
}
function mcAttachRegionDrag(el, region) {
  let mode = 'move';
  let startX = 0;
  let originalStart = 0;
  let originalEnd = 0;
  el.querySelector('.left').addEventListener('pointerdown', event => { mode = 'left'; begin(event); });
  el.querySelector('.right').addEventListener('pointerdown', event => { mode = 'right'; begin(event); });
  el.addEventListener('pointerdown', event => { if (!event.target.classList.contains('region-handle')) { mode = 'move'; begin(event); } });
  function begin(event) {
    event.preventDefault();
    startX = event.clientX;
    originalStart = Number(region.start_sec);
    originalEnd = Number(region.end_sec);
    el.setPointerCapture(event.pointerId);
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', end);
  }
  function move(event) {
    const root = document.getElementById('mc-editor-root');
    const duration = Number(root.dataset.duration || 1);
    const stage = document.querySelector('.wave-stage');
    const delta = (event.clientX - startX) / stage.clientWidth * duration;
    if (mode === 'left') {
      region.start_sec = Math.max(0, Math.min(originalEnd - 1, originalStart + delta));
    } else if (mode === 'right') {
      region.end_sec = Math.min(duration, Math.max(originalStart + 1, originalEnd + delta));
    } else {
      const length = originalEnd - originalStart;
      const nextStart = Math.max(0, Math.min(duration - length, originalStart + delta));
      region.start_sec = nextStart;
      region.end_sec = nextStart + length;
    }
    region.start_sec = Math.round(region.start_sec * 10) / 10;
    region.end_sec = Math.round(region.end_sec * 10) / 10;
    mcApplyRegionElementPosition(el, region, duration);
    mcRenderRegionTable();
  }
  function end(event) {
    el.releasePointerCapture(event.pointerId);
    el.removeEventListener('pointermove', move);
    el.removeEventListener('pointerup', end);
    mcRenderRegions();
  }
}
function mcRenderRegionTable() {
  const table = document.getElementById('mc-region-table');
  if (!table || !window.mcEditRegions) return;
  table.innerHTML = '';
  window.mcEditRegions.forEach((region, index) => {
    if (region.selected === undefined) region.selected = true;
    const card = document.createElement('div');
    card.className = 'region-card' + (region.selected ? '' : ' is-muted');
    card.style.borderColor = mcRegionColor(region.label, 0.55);
    const options = (window.mcEditCategories || []).map(category => {
      const selected = category.name === region.label ? 'selected' : '';
      return `<option value="${category.name}" ${selected}>${category.name}</option>`;
    }).join('');
    card.innerHTML = `
      <div class="region-card-head">
        <input type="checkbox" ${region.selected ? 'checked' : ''} onchange="window.mcEditRegions[${index}].selected=this.checked; mcRenderRegionTable();">
        <input value="${region.display_name || region.clip_id}" onchange="window.mcEditRegions[${index}].display_name=this.value; mcRenderRegions();">
      </div>
      <div class="region-time-row">
        <input type="number" step="0.1" value="${Number(region.start_sec).toFixed(1)}" onchange="window.mcEditRegions[${index}].start_sec=Number(this.value); mcRenderRegions();">
        <input type="number" step="0.1" value="${Number(region.end_sec).toFixed(1)}" onchange="window.mcEditRegions[${index}].end_sec=Number(this.value); mcRenderRegions();">
      </div>
      <div class="region-category-row">
        <span class="region-color-dot" style="background:${mcRegionColor(region.label, 0.9)}"></span>
        <select onchange="window.mcEditRegions[${index}].label=this.value; mcRenderRegions(); mcRenderRegionTable();">${options}</select>
      </div>
    `;
    table.appendChild(card);
  });
}
function mcAddEditRegion() {
  const root = document.getElementById('mc-editor-root');
  const audio = document.getElementById('mc-edit-audio');
  if (!root || !window.mcEditRegions || window.mcEditRegions.length === 0) {
    alert('当前没有可参考的原曲片段。');
    return;
  }
  const duration = Math.max(1, Number(root.dataset.duration || 1));
  const baseRegion = window.mcEditRegions[0];
  const sorted = [...window.mcEditRegions].sort((a, b) => Number(a.end_sec || 0) - Number(b.end_sec || 0));
  const lastEnd = Number(sorted.at(-1).end_sec || 0);
  const current = audio && Number.isFinite(audio.currentTime) && audio.currentTime > 0 ? audio.currentTime : lastEnd;
  const length = Math.min(12, Math.max(4, duration / 10));
  const start = Math.max(0, Math.min(duration - length, current));
  const end = Math.min(duration, start + length);
  const label = (window.mcEditCategories && window.mcEditCategories[0] && window.mcEditCategories[0].name)
    || baseRegion.label
    || '';
  const index = window.mcEditRegions.length + 1;
  window.mcEditRegions.push({
    clip_id: root.dataset.baseClipId || baseRegion.clip_id,
    display_name: '新增片段_' + String(index).padStart(2, '0'),
    label,
    section: 'unknown',
    start_sec: Math.round(start * 10) / 10,
    end_sec: Math.round(end * 10) / 10,
    selected: true,
    is_new: true,
  });
  mcRenderRegions();
  mcRenderRegionTable();
}
function mcCommitEditRegions() {
  if (!window.mcEditRegions || window.mcEditRegions.length === 0) {
    alert('没有可裁剪的片段。');
    return;
  }
  const selectedRegions = window.mcEditRegions.filter(region => region.selected !== false);
  if (selectedRegions.length === 0) {
    alert('请至少勾选一个片段。');
    return;
  }
  fetch('/api/commit_edit_regions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({regions: selectedRegions})
  }).then(response => {
    if (!response.ok) throw new Error('HTTP ' + response.status);
    return response.json();
  }).then(data => {
    alert('已生成 ' + data.created + ' 个新片段。');
    window.location.reload();
  }).catch(error => alert('重切失败：' + error.message));
}
