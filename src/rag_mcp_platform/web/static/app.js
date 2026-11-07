const chat = document.getElementById("chat");
const empty = document.getElementById("empty");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const examplesEl = document.getElementById("examples");

const EXAMPLES = [
  "How do I get started?",
  "What configuration options are available?",
  "How can I troubleshoot a failed task?",
];

function init() {
  EXAMPLES.forEach((text) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "example";
    chip.textContent = text;
    chip.onclick = () => {
      input.value = text;
      form.requestSubmit();
    };
    examplesEl.appendChild(chip);
  });
  loadHealth();
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    statusEl.textContent = `${h.embedding_provider}:${h.embedding_model} · LLM ${h.llm_provider}:${h.llm_model}`;
  } catch {
    statusEl.textContent = "backend unreachable";
    statusEl.classList.add("error");
  }
}

function autoresize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

input.addEventListener("input", autoresize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

function addMessage(role, text) {
  if (empty) empty.remove();
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role === "user" ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (text instanceof Node) bubble.appendChild(text);
  else bubble.textContent = text;

  msg.appendChild(roleEl);
  msg.appendChild(bubble);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return { msg, bubble };
}

function typingIndicator() {
  const wrap = document.createElement("span");
  wrap.className = "typing";
  wrap.innerHTML = "<span></span><span></span><span></span>";
  return wrap;
}

function renderSources(container, sources) {
  if (!sources || sources.length === 0) return;
  const box = document.createElement("div");
  box.className = "sources";

  const title = document.createElement("div");
  title.className = "sources-title";
  title.textContent = "Sources";
  box.appendChild(title);

  sources.forEach((s) => {
    const row = document.createElement("div");
    row.className = "source";

    const label = document.createElement("span");
    if (s.url) {
      const a = document.createElement("a");
      a.href = s.url;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = s.citation;
      label.appendChild(a);
    } else {
      label.textContent = s.citation;
    }

    const score = document.createElement("span");
    score.className = "score";
    score.textContent = s.score.toFixed(3);

    row.appendChild(label);
    row.appendChild(score);
    box.appendChild(row);
  });

  container.appendChild(box);
}

async function ask(query) {
  addMessage("user", query);
  const { msg, bubble } = addMessage("assistant", typingIndicator());

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    bubble.textContent = data.answer;
    renderSources(msg, data.sources);
  } catch (err) {
    bubble.textContent = `Something went wrong: ${err.message}`;
    bubble.classList.add("error");
  } finally {
    chat.scrollTop = chat.scrollHeight;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  input.value = "";
  autoresize();
  ask(query);
});

init();
