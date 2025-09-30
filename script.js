import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.0';

env.allowRemoteModels = true;
env.allowLocalModels = false;
env.useBrowserCache = true;

typeset();

function typeset() {
  const scroller = document.getElementById('scroller');
  const messagesEl = document.getElementById('messages');
  const typingEl = document.getElementById('typing');
  const form = document.getElementById('composer');
  const input = document.getElementById('composerInput');
  const sendButton = form.querySelector('.chat-send');
  const statusEl = document.getElementById('headerStatus');

  const DEFAULT_VERSES = ['Sarai come ossigeno.'];
  let OPTIONS = DEFAULT_VERSES;
  let randomDeck = [];

  const conversation = [];
  let embedder = null;
  let verseEmbeddings = [];
  let modelReady = false;
  const recentVerseIndexes = [];
  const RECENT_LIMIT = 3;

  sendButton.disabled = true;

  function dedupePreserveOrder(array) {
    const seen = new Set();
    const out = [];
    for (const item of array) {
      const key = item.trim();
      if (!key || seen.has(key)) continue;
      seen.add(key);
      out.push(item);
    }
    return out;
  }

  function shuffle(array) {
    const copy = [...array];
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  }

  function resetRandomDeck() {
    randomDeck = shuffle(OPTIONS);
  }

  function drawRandomVerse() {
    if (!randomDeck.length) {
      resetRandomDeck();
    }
    return randomDeck.pop();
  }

  function addMessage(text, sender) {
    const item = document.createElement('li');
    item.className = `message message--${sender}`;

    const bubble = document.createElement('div');
    bubble.className = 'message__bubble';
    bubble.textContent = text;

    const meta = document.createElement('span');
    meta.className = 'message__meta';
    const now = new Date();
    meta.textContent = now.toLocaleTimeString('it-IT', {
      hour: '2-digit',
      minute: '2-digit'
    });

    bubble.appendChild(meta);
    item.appendChild(bubble);
    messagesEl.appendChild(item);

    conversation.push({ text, sender });

    requestAnimationFrame(() => {
      scroller.scrollTo({ top: scroller.scrollHeight, behavior: 'smooth' });
    });
  }

  function setTyping(isTyping) {
    typingEl.setAttribute('aria-hidden', String(!isTyping));
  }

  function buildContext(promptText) {
    const historyContext = conversation.slice(-6).map(entry => {
      const prefix = entry.sender === 'user' ? 'Utente' : 'Bot';
      return `${prefix}: ${entry.text}`;
    });
    historyContext.push(`Utente: ${promptText}`);
    return historyContext.join('\n');
  }

  function cosineSimilarity(normalizedA, normalizedB) {
    let score = 0;
    for (let i = 0; i < normalizedA.length; i++) {
      score += normalizedA[i] * normalizedB[i];
    }
    return score;
  }

  function addToRecent(index) {
    recentVerseIndexes.push(index);
    if (recentVerseIndexes.length > RECENT_LIMIT) {
      recentVerseIndexes.shift();
    }
  }

  function pickFromRanked(ranked) {
    for (const candidate of ranked) {
      if (!recentVerseIndexes.includes(candidate.index)) {
        addToRecent(candidate.index);
        return candidate.index;
      }
    }
    const fallback = ranked[0];
    addToRecent(fallback.index);
    return fallback.index;
  }

  async function chooseVerse(promptText) {
    if (!modelReady || !embedder || verseEmbeddings.length !== OPTIONS.length) {
      return drawRandomVerse();
    }
    const contextText = buildContext(promptText);
    const { data } = await embedder(contextText, { pooling: 'mean', normalize: true });
    const ranked = verseEmbeddings
      .map((vector, index) => ({
        index,
        score: cosineSimilarity(data, vector)
      }))
      .sort((a, b) => b.score - a.score);
    const selectedIndex = pickFromRanked(ranked);
    return OPTIONS[selectedIndex];
  }

  async function reply(promptText) {
    setTyping(true);
    const intentDelay = 380 + Math.random() * 850;
    await new Promise(resolve => setTimeout(resolve, intentDelay));
    const verse = await chooseVerse(promptText);
    setTyping(false);
    addMessage(verse, 'bot');
  }

  form.addEventListener('submit', event => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) {
      return;
    }
    addMessage(text, 'user');
    input.value = '';
    input.focus();
    reply(text).catch(err => {
      console.error('Errore nella risposta AI:', err);
      setTyping(false);
      addMessage(drawRandomVerse(), 'bot');
    });
  });

  window.addEventListener('load', () => {
    input.focus();
  });

  async function loadVerses() {
    const response = await fetch('verses.txt');
    if (!response.ok) {
      throw new Error(`Impossibile recuperare verses.txt (status ${response.status})`);
    }
    const text = await response.text();
    return text
      .split(/\r?\n/) // gestione CRLF/LF
      .map(line => line.trim())
      .filter(Boolean);
  }

  async function warmupEmbeddings() {
    try {
      statusEl.textContent = 'Carico il modello di embedding…';
      embedder = await pipeline('feature-extraction', 'Snowflake/snowflake-arctic-embed-m');
      statusEl.textContent = 'Indicizzo i versi…';
      verseEmbeddings = [];
      for (let i = 0; i < OPTIONS.length; i++) {
        statusEl.textContent = `Indicizzo i versi… ${i + 1}/${OPTIONS.length}`;
        const { data } = await embedder(OPTIONS[i], { pooling: 'mean', normalize: true });
        verseEmbeddings.push(Float32Array.from(data));
      }
      modelReady = true;
      sendButton.disabled = false;
      statusEl.textContent = 'Online · risponde con versi contestualizzati';
    } catch (error) {
      console.error('Impossibile inizializzare il modello:', error);
      modelReady = false;
      sendButton.disabled = false;
      statusEl.textContent = 'Offline · risposte casuali';
    }
  }

  async function bootstrap() {
    try {
      statusEl.textContent = 'Carico i versi…';
      const fileVerses = await loadVerses();
      const unique = dedupePreserveOrder(fileVerses);
      if (unique.length) {
        OPTIONS = unique;
      }
    } catch (error) {
      console.error('Errore caricando i versi da verses.txt:', error);
      statusEl.textContent = 'Versi offline · uso fallback casuale';
    }

    resetRandomDeck();
    await warmupEmbeddings();
  }

  bootstrap();
}
