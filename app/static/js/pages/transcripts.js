let state = {
  transcripts: [],
  activeTranscript: null,
  mapData: { people: [], locations: [], claims: [], events: [] },
};

const entityLabels = new Map();

function indexEntityLabels(data) {
  entityLabels.clear();
  (data.people || []).forEach((person) => entityLabels.set(person.id, person.display_name));
  (data.locations || []).forEach((location) => entityLabels.set(location.id, location.name));
  (data.claims || []).forEach((claim) => entityLabels.set(claim.id, claim.statement));
  (data.events || []).forEach((event) => entityLabels.set(event.id, event.title));
}

function renderTranscriptList() {
  const list = document.getElementById("transcript-list");
  const count = document.getElementById("transcript-count");
  const query = document.getElementById("transcript-search").value.trim().toLowerCase();
  const transcripts = state.transcripts.filter((transcript) => {
    const haystack = `${transcript.title} ${transcript.source_type} ${transcript.url}`.toLowerCase();
    return !query || haystack.includes(query);
  });

  count.textContent = `${transcripts.length} source${transcripts.length === 1 ? "" : "s"} shown`;
  list.innerHTML = "";
  if (transcripts.length === 0) {
    list.innerHTML = '<p class="muted">No transcripts match the current search.</p>';
    return;
  }

  transcripts.forEach((transcript) => {
    const button = document.createElement("button");
    button.className = `transcript-card${state.activeTranscript && state.activeTranscript.source_id === transcript.id ? " is-active" : ""}`;
    button.type = "button";
    button.innerHTML = `
      <strong>${escapeHtml(transcript.title)}</strong>
      <span class="meta">${escapeHtml(transcript.source_type)} · ${escapeHtml((transcript.source_stats && transcript.source_stats.lines) || 0)} lines</span>
    `;
    button.addEventListener("click", () => loadTranscript(transcript.id));
    list.appendChild(button);
  });
}

async function loadTranscript(sourceId) {
  const transcript = await window.HaymarketApi.getTranscript(sourceId);
  state.activeTranscript = transcript;
  renderTranscriptList();
  renderTranscript();
}

function renderTranscript() {
  const transcript = state.activeTranscript;
  const title = document.getElementById("transcript-title");
  const meta = document.getElementById("transcript-meta");
  const body = document.getElementById("transcript-body");
  const teiLink = document.getElementById("tei-link");
  const mentionSummary = document.getElementById("mention-summary");

  if (!transcript) {
    body.innerHTML = '<p class="muted">No transcript selected.</p>';
    return;
  }

  const kindFilter = document.getElementById("mention-kind-filter").value;
  const mentions = (transcript.mentions || []).filter((mention) => !kindFilter || mention.kind === kindFilter);
  title.textContent = transcript.title;
  meta.textContent = `${transcript.source_type} · ${mentions.length} highlighted mention${mentions.length === 1 ? "" : "s"}`;
  mentionSummary.textContent = `${mentions.length} visible mention${mentions.length === 1 ? "" : "s"} from ${transcript.mentions.length} total`;
  teiLink.hidden = false;
  teiLink.href = `/api/transcripts/${encodeURIComponent(transcript.source_id)}/tei`;

  body.innerHTML = "";
  (transcript.segments || []).forEach((segment) => {
    const paragraph = document.createElement("p");
    paragraph.className = `transcript-segment segment-${escapeAttribute(segment.kind || "text")}`;
    paragraph.id = `segment-${segment.index}`;
    if (segment.page_ref) {
      paragraph.dataset.pageRef = segment.page_ref;
    }
    paragraph.innerHTML = renderSegmentText(segment, mentions);
    body.appendChild(paragraph);
  });
}

function renderSegmentText(segment, mentions) {
  const segmentMentions = mentions
    .filter((mention) => mention.start < segment.end && mention.end > segment.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);
  if (segmentMentions.length === 0) {
    return `${renderPageBadge(segment)}${escapeHtml(segment.text || "")}`;
  }

  let cursor = segment.start;
  const parts = [renderPageBadge(segment)];
  segmentMentions.forEach((mention) => {
    const start = Math.max(mention.start, segment.start);
    const end = Math.min(mention.end, segment.end);
    if (start < cursor) return;
    parts.push(escapeHtml((segment.text || "").slice(cursor - segment.start, start - segment.start)));
    parts.push(renderMention((segment.text || "").slice(start - segment.start, end - segment.start), mention));
    cursor = end;
  });
  parts.push(escapeHtml((segment.text || "").slice(cursor - segment.start)));
  return parts.join("");
}

function renderPageBadge(segment) {
  if (!segment.page_ref) return "";
  return `<span class="page-badge">p. ${escapeHtml(segment.page_ref)}</span>`;
}

function renderMention(text, mention) {
  const label = entityLabels.get(mention.entity_id) || mention.entity_id || mention.kind;
  return `<mark class="mention mention-${escapeAttribute(mention.kind || "unknown")}" title="${escapeAttribute(label)}">${escapeHtml(text)}</mark>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

async function boot() {
  document.getElementById("transcript-search").addEventListener("input", renderTranscriptList);
  document.getElementById("mention-kind-filter").addEventListener("change", renderTranscript);

  try {
    const [transcripts, mapData] = await Promise.all([window.HaymarketApi.getTranscripts(), window.HaymarketApi.getMapData()]);
    state.transcripts = transcripts;
    state.mapData = mapData;
    indexEntityLabels(mapData);
    renderTranscriptList();
    if (transcripts.length > 0) {
      const requestedSource = new URLSearchParams(window.location.search).get("source");
      const initialSource = transcripts.some((transcript) => transcript.id === requestedSource) ? requestedSource : transcripts[0].id;
      await loadTranscript(initialSource);
    }
  } catch (error) {
    document.getElementById("transcript-count").textContent = "Unable to load transcripts.";
    document.getElementById("transcript-body").innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`;
  }
}

boot();
