const $ = s => document.querySelector(s);

let chosen = 'hd';
let files = [];
let current = null;
let originalImage = null;
let aiCanvas = null;
let showingOriginal = false;
let session = null;
let busy = false;

const MODEL_URL =
  'https://huggingface.co/thisismert/real-esrgan-onnx/resolve/main/onnx/realesr-general-x4v3.onnx';

const names = {
  hd: 'LUME HD · Aprimoramento com inteligência artificial.',
  beauty: 'BEAUTY · Ajuste suave de luz e cor.',
  low: 'LOW LIGHT · Clareamento de sombras.',
  batch: 'BATCH · Selecione uma foto por vez.'
};

document.querySelectorAll('.style').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.style').forEach(b =>
      b.classList.remove('selected')
    );
    button.classList.add('selected');
    chosen = button.dataset.style;
    aiCanvas = null;
    $('#styleNote').textContent = names[chosen];
    if (originalImage) render();
  });
});

$('#start').onclick = () => show('editor');
$('#back').onclick = () => show('home');

function show(id) {
  document.querySelectorAll('.screen').forEach(s =>
    s.classList.remove('active')
  );
  $('#' + id).classList.add('active');
}

function status(message) {
  let el = $('#lumeStatus');

  if (!el) {
    el = document.createElement('p');
    el.id = 'lumeStatus';
    el.className = 'hint';
    $('#editarea').append(el);
  }

  el.textContent = message;
}

$('#photos').addEventListener('change', event => {
  files = [...event.target.files];
  if (!files.length) return;

  $('#selection').innerHTML = '';

  files.forEach((file, index) => {
    const thumb = document.createElement('img');
    thumb.className = 'thumb';
    thumb.src = URL.createObjectURL(file);
    thumb.title = file.name;
    thumb.onclick = () => loadFile(index);
    $('#selection').append(thumb);
  });

  $('#editarea').hidden = false;
  loadFile(0);
});

function loadFile(index) {
  current = files[index];
  aiCanvas = null;
  showingOriginal = false;

  const image = new Image();

  image.onload = () => {
    originalImage = image;
    render();
    status('Foto carregada. Escolha o aprimoramento.');
  };

  image.onerror = () => status('Não foi possível abrir esta foto.');

  image.src = URL.createObjectURL(current);
}

function render() {
  if (!originalImage) return;

  const canvas = $('#preview');
  const ctx = canvas.getContext('2d');

  if (!showingOriginal && chosen === 'hd' && aiCanvas) {
    canvas.width = aiCanvas.width;
    canvas.height = aiCanvas.height;
    ctx.filter = 'none';
    ctx.drawImage(aiCanvas, 0, 0);
    return;
  }

  const image = originalImage;
  const scale = Math.min(
    1,
    1800 / Math.max(image.width, image.height)
  );

  canvas.width = Math.round(image.width * scale);
  canvas.height = Math.round(image.height * scale);

  ctx.filter = 'none';

  if (!showingOriginal && chosen === 'low') {
    ctx.filter = 'brightness(1.24) contrast(1.08) saturate(1.08)';
  } else if (!showingOriginal && chosen === 'beauty') {
    ctx.filter = 'brightness(1.04) saturate(1.06) contrast(.97)';
  } else if (!showingOriginal && chosen === 'hd') {
    ctx.filter = 'contrast(1.08) saturate(1.05)';
  }

  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  ctx.filter = 'none';
}

async function getSession() {
  if (session) return session;

  if (!window.ort) {
    throw new Error('Biblioteca de IA não carregada.');
  }

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

  status('Baixando o modelo de inteligência artificial...');

  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ['wasm']
  });

  return session;
}

async function enhanceWithAI() {
  if (!originalImage || busy) return;

  busy = true;
  $('#aiButton').disabled = true;

  try {
    const ai = await getSession();

    // Limita a imagem de entrada para proteger a memória do iPhone.
    const maxSide = 1024;
    const scale = Math.min(
      1,
      maxSide / Math.max(originalImage.width, originalImage.height)
    );

    const width = Math.max(1, Math.round(originalImage.width * scale));
    const height = Math.max(1, Math.round(originalImage.height * scale));

    const source = document.createElement('canvas');
    source.width = width;
    source.height = height;

    const sourceCtx = source.getContext('2d', {
      willReadFrequently: true
    });

    sourceCtx.drawImage(originalImage, 0, 0, width, height);

    const resultCanvas = document.createElement('canvas');
    resultCanvas.width = width * 4;
    resultCanvas.height = height * 4;

    const resultCtx = resultCanvas.getContext('2d', {
      willReadFrequently: true
    });

    const tileSize = 128;
    const total =
      Math.ceil(width / tileSize) *
      Math.ceil(height / tileSize);

    let completed = 0;

    for (let y = 0; y < height; y += tileSize) {
      for (let x = 0; x < width; x += tileSize) {
        const w = Math.min(tileSize, width - x);
        const h = Math.min(tileSize, height - y);

        const pixels = sourceCtx.getImageData(x, y, w, h).data;
        const plane = w * h;
        const inputData = new Float32Array(plane * 3);

        for (let i = 0; i < plane; i++) {
          inputData[i] = pixels[i * 4] / 255;
          inputData[plane + i] = pixels[i * 4 + 1] / 255;
          inputData[plane * 2 + i] = pixels[i * 4 + 2] / 255;
        }

        const tensor = new ort.Tensor(
          'float32',
          inputData,
          [1, 3, h, w]
        );

        const outputs = await ai.run({
          [ai.inputNames[0]]: tensor
        });

        const output = outputs[ai.outputNames[0]];
        const outWidth = w * 4;
        const outHeight = h * 4;
        const outPlane = outWidth * outHeight;

        const tile = resultCtx.createImageData(outWidth, outHeight);

        for (let i = 0; i < outPlane; i++) {
          tile.data[i * 4] =
            Math.round(Math.max(0, Math.min(1, output.data[i])) * 255);

          tile.data[i * 4 + 1] =
            Math.round(Math.max(0, Math.min(1, output.data[outPlane + i])) * 255);

          tile.data[i * 4 + 2] =
            Math.round(Math.max(0, Math.min(1, output.data[outPlane * 2 + i])) * 255);

          tile.data[i * 4 + 3] = 255;
        }

        resultCtx.putImageData(tile, x * 4, y * 4);

        completed++;
        status(
          `Aprimorando com IA: ${Math.round(completed / total * 100)}%`
        );

        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    aiCanvas = resultCanvas;
    showingOriginal = false;
    render();

    status('Aprimoramento com IA concluído! ✨');

  } catch (error) {
    console.error(error);
    status('Erro no processamento: ' + error.message);
  } finally {
    busy = false;
    $('#aiButton').disabled = false;
  }
}

const aiButton = document.createElement('button');
aiButton.id = 'aiButton';
aiButton.className = 'primary';
aiButton.textContent = '✧ Aprimorar com IA';
aiButton.style.marginTop = '12px';
aiButton.onclick = enhanceWithAI;

$('#editarea').append(aiButton);

$('#original').onclick = () => {
  showingOriginal = !showingOriginal;
  $('#original').textContent =
    showingOriginal ? 'Ver edição' : 'Ver original';
  render();
};

$('#save').onclick = () => {
  if (!current) return;

  $('#preview').toBlob(blob => {
    if (!blob) {
      status('Não foi possível salvar a foto.');
      return;
    }

    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');

    link.href = url;
    link.download =
      'LUME-' + current.name.replace(/\.[^.]+$/, '') + '.jpg';

    link.click();

    setTimeout(() => URL.revokeObjectURL(url), 3000);
  }, 'image/jpeg', 0.96);
};

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  });
}
