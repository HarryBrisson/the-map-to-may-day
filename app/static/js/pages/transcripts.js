let state = {
  transcripts: [],
  activeTranscript: null,
  activeView: "sources",
  expandedSources: new Set(),
  expandedEvents: new Set(),
  pendingPageRef: null,
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

function renderNavigator() {
  document.querySelectorAll("[data-view-tab]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.viewTab === state.activeView);
  });
  document.getElementById("event-filter-wrap").hidden = state.activeView !== "sources";
  if (state.activeView === "events") {
    renderEventTimeline();
  } else if (state.activeView === "document-events") {
    renderDocumentEventTimeline();
  } else {
    renderSourceTimeline();
  }
}

function renderSourceTimeline() {
  const list = document.getElementById("transcript-list");
  const count = document.getElementById("transcript-count");
  const transcripts = filteredSources().sort(compareSources);
  count.textContent = `${transcripts.length} source${transcripts.length === 1 ? "" : "s"} shown`;
  list.innerHTML = "";

  if (transcripts.length === 0) {
    list.innerHTML = '<p class="muted">No sources match the current filters.</p>';
    return;
  }

  groupedBy(transcripts, sourceGroupKey).forEach((items, group) => {
    const section = document.createElement("section");
    section.className = "timeline-group";
    section.innerHTML = `<h3>${escapeHtml(group)}</h3>`;
    items.forEach((transcript) => section.appendChild(renderSourceCard(transcript)));
    list.appendChild(section);
  });
}

function renderSourceCard(transcript) {
  const nav = navigation(transcript);
  const expanded = state.expandedSources.has(transcript.id);
  const card = document.createElement("article");
  card.className = `document-card${state.activeTranscript && state.activeTranscript.source_id === transcript.id ? " is-active" : ""}`;
  const people = nav.primary_people || [];
  const locations = nav.primary_locations || [];
  const events = nav.referenced_events || [];
  const documentEvents = nav.document_events || [];
  card.innerHTML = `
    <div class="document-card-main" role="button" tabindex="0" data-source-open="${escapeAttribute(transcript.id)}">
      <div>
        <h4>${escapeHtml(nav.brief_title || transcript.title)}</h4>
        <div class="meta">${escapeHtml(sourceMeta(transcript))}</div>
      </div>
      <span class="count-chip">${escapeHtml(nav.document_role || transcript.source_type)}</span>
    </div>
    <div class="chip-row">
      ${people.slice(0, 3).map((item) => renderChip(item.label, "person")).join("")}
      ${locations.slice(0, 2).map((item) => renderChip(item.label, "location")).join("")}
      ${(nav.topics || []).slice(0, 3).map((topic) => renderChip(topic, "topic")).join("")}
    </div>
    <div class="card-actions">
      <button type="button" class="text-button" data-source-toggle="${escapeAttribute(transcript.id)}">${expanded ? "Collapse" : "Expand"}</button>
      <button type="button" class="text-button" data-source-load="${escapeAttribute(transcript.id)}">Open Transcript</button>
    </div>
    ${expanded ? renderSourceDetails(transcript, people, locations, events, documentEvents) : ""}
  `;
  card.querySelector("[data-source-open]").addEventListener("click", () => loadTranscript(transcript.id));
  card.querySelector("[data-source-open]").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") loadTranscript(transcript.id);
  });
  card.querySelector("[data-source-toggle]").addEventListener("click", () => toggleSource(transcript.id));
  card.querySelector("[data-source-load]").addEventListener("click", () => loadTranscript(transcript.id));
  card.querySelectorAll("[data-source-page]").forEach((button) => {
    button.addEventListener("click", () => loadTranscript(transcript.id, button.dataset.sourcePage));
  });
  return card;
}

function renderSourceDetails(transcript, people, locations, events, documentEvents) {
  const nav = navigation(transcript);
  return `
    <div class="document-details">
      ${nav.navigation_summary ? `<p>${escapeHtml(nav.navigation_summary)}</p>` : ""}
      <div class="detail-grid">
        <div><strong>People</strong><div class="chip-row">${people.map((item) => renderChip(item.label, "person")).join("") || '<span class="muted">None tagged</span>'}</div></div>
        <div><strong>Places</strong><div class="chip-row">${locations.map((item) => renderChip(item.label, "location")).join("") || '<span class="muted">None tagged</span>'}</div></div>
      </div>
      <div class="support-list">
        ${events.length ? events.map((eventRef) => renderSourceEventRef(transcript, eventRef)).join("") : '<p class="muted">No referenced events in the brief yet.</p>'}
      </div>
      ${documentEvents.length ? `
        <div class="support-list">
          ${documentEvents.map((eventRef) => renderSourceDocumentEventRef(transcript, eventRef)).join("")}
        </div>
      ` : ""}
      <p class="meta">${escapeHtml((nav.claim_count || 0).toString())} claims · ${escapeHtml(((transcript.source_stats || {}).mentions || 0).toString())} mentions</p>
    </div>
  `;
}

function renderSourceEventRef(transcript, eventRef) {
  const pageButtons = (eventRef.page_refs || []).slice(0, 4).map((pageRef) => {
    return `<button type="button" class="page-jump" data-source-page="${escapeAttribute(pageRef)}">p. ${escapeHtml(pageRef)}</button>`;
  }).join("");
  return `
    <article class="support-card">
      <strong>${escapeHtml(eventRef.label || "Referenced event")}</strong>
      <div class="meta">${escapeHtml(eventMeta(eventRef))}</div>
      ${eventRef.summary ? `<p>${escapeHtml(eventRef.summary)}</p>` : ""}
      ${eventRef.supporting_quote ? `<blockquote>${escapeHtml(eventRef.supporting_quote)}</blockquote>` : ""}
      <div class="chip-row">${pageButtons}</div>
    </article>
  `;
}

function renderSourceDocumentEventRef(transcript, eventRef) {
  const pageButtons = (eventRef.page_refs || []).slice(0, 4).map((pageRef) => {
    return `<button type="button" class="page-jump" data-source-page="${escapeAttribute(pageRef)}">p. ${escapeHtml(pageRef)}</button>`;
  }).join("");
  return `
    <article class="support-card">
      <strong>${escapeHtml(eventRef.label || "Document event")}</strong>
      <div class="meta">${escapeHtml([eventKindLabel(eventRef.event_kind), eventMeta(eventRef)].filter(Boolean).join(" · "))}</div>
      ${eventRef.summary ? `<p>${escapeHtml(eventRef.summary)}</p>` : ""}
      ${eventRef.supporting_quote ? `<blockquote>${escapeHtml(eventRef.supporting_quote)}</blockquote>` : ""}
      <div class="chip-row">${pageButtons}</div>
    </article>
  `;
}

function renderEventTimeline() {
  const list = document.getElementById("transcript-list");
  const count = document.getElementById("transcript-count");
  const events = filteredEvents().sort(compareEventGroups);
  count.textContent = `${events.length} event${events.length === 1 ? "" : "s"} shown`;
  list.innerHTML = "";

  if (events.length === 0) {
    list.innerHTML = '<p class="muted">No referenced events match the current filters.</p>';
    return;
  }

  groupedBy(events, eventGroupKey).forEach((items, group) => {
    const section = document.createElement("section");
    section.className = "timeline-group";
    section.innerHTML = `<h3>${escapeHtml(group)}</h3>`;
    items.forEach((eventGroup) => section.appendChild(renderEventCard(eventGroup)));
    list.appendChild(section);
  });
}

function renderDocumentEventTimeline() {
  const list = document.getElementById("transcript-list");
  const count = document.getElementById("transcript-count");
  const events = filteredDocumentEvents().sort(compareEventGroups);
  count.textContent = `${events.length} document event${events.length === 1 ? "" : "s"} shown`;
  list.innerHTML = "";

  if (events.length === 0) {
    list.innerHTML = '<p class="muted">No document events match the current filters.</p>';
    return;
  }

  groupedBy(events, eventGroupKey).forEach((items, group) => {
    const section = document.createElement("section");
    section.className = "timeline-group";
    section.innerHTML = `<h3>${escapeHtml(group)}</h3>`;
    items.forEach((eventGroup) => section.appendChild(renderDocumentEventCard(eventGroup)));
    list.appendChild(section);
  });
}

function renderDocumentEventCard(eventGroup) {
  const expanded = state.expandedEvents.has(eventGroup.key);
  const card = document.createElement("article");
  card.className = "document-card event-document-card";
  card.innerHTML = `
    <div class="document-card-main" role="button" tabindex="0" data-event-toggle="${escapeAttribute(eventGroup.key)}">
      <div>
        <h4>${escapeHtml(eventGroup.label)}</h4>
        <div class="meta">${escapeHtml([eventKindLabel(eventGroup.event_kind), eventMeta(eventGroup)].filter(Boolean).join(" · "))}</div>
      </div>
      <span class="count-chip">${eventGroup.sources.length} source${eventGroup.sources.length === 1 ? "" : "s"}</span>
    </div>
    <p>${escapeHtml(eventGroup.summary || "")}</p>
    <div class="chip-row">
      ${renderChip(eventGroup.location_label, "location")}
      ${(eventGroup.participant_labels || []).slice(0, 4).map((label) => renderChip(label, "person")).join("")}
    </div>
    ${expanded ? renderEventSupport(eventGroup) : ""}
  `;
  card.querySelector("[data-event-toggle]").addEventListener("click", () => toggleEvent(eventGroup.key));
  card.querySelector("[data-event-toggle]").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") toggleEvent(eventGroup.key);
  });
  card.querySelectorAll("[data-support-source]").forEach((button) => {
    button.addEventListener("click", () => loadTranscript(button.dataset.supportSource, button.dataset.supportPage || null));
  });
  return card;
}

function renderEventCard(eventGroup) {
  const expanded = state.expandedEvents.has(eventGroup.key);
  const card = document.createElement("article");
  card.className = "document-card event-document-card";
  card.innerHTML = `
    <div class="document-card-main" role="button" tabindex="0" data-event-toggle="${escapeAttribute(eventGroup.key)}">
      <div>
        <h4>${escapeHtml(eventGroup.label)}</h4>
        <div class="meta">${escapeHtml(eventMeta(eventGroup))}</div>
      </div>
      <span class="count-chip">${eventGroup.sources.length} source${eventGroup.sources.length === 1 ? "" : "s"}</span>
    </div>
    <p>${escapeHtml(eventGroup.summary || "")}</p>
    <div class="chip-row">
      ${renderChip(eventGroup.location_label, "location")}
      ${(eventGroup.participant_labels || []).slice(0, 4).map((label) => renderChip(label, "person")).join("")}
    </div>
    ${expanded ? renderEventSupport(eventGroup) : ""}
  `;
  card.querySelector("[data-event-toggle]").addEventListener("click", () => toggleEvent(eventGroup.key));
  card.querySelector("[data-event-toggle]").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") toggleEvent(eventGroup.key);
  });
  card.querySelectorAll("[data-support-source]").forEach((button) => {
    button.addEventListener("click", () => loadTranscript(button.dataset.supportSource, button.dataset.supportPage || null));
  });
  return card;
}

function renderEventSupport(eventGroup) {
  return `
    <div class="support-list">
      ${eventGroup.sources.map(({ source, ref }) => `
        <article class="support-card">
          <strong>${escapeHtml(navigation(source).brief_title || source.title)}</strong>
          <div class="meta">${escapeHtml(sourceMeta(source))}</div>
          ${ref.supporting_quote ? `<blockquote>${escapeHtml(ref.supporting_quote)}</blockquote>` : ""}
          <button type="button" class="text-button" data-support-source="${escapeAttribute(source.id)}" data-support-page="${escapeAttribute((ref.page_refs || [])[0] || "")}">Open Transcript</button>
        </article>
      `).join("")}
    </div>
  `;
}

function filteredSources(options = {}) {
  const filters = readFilters();
  return state.transcripts.filter((transcript) => sourceMatches(transcript, filters, options));
}

function sourceMatches(transcript, filters, options = {}) {
  const nav = navigation(transcript);
  const haystack = [
    transcript.title,
    transcript.source_type,
    transcript.url,
    nav.brief_title,
    nav.navigation_summary,
    ...(nav.topics || []),
    ...(nav.primary_people || []).map((item) => item.label),
    ...(nav.primary_locations || []).map((item) => item.label),
    ...(nav.referenced_events || []).map((item) => item.label),
    ...(nav.document_events || []).map((item) => item.label),
  ].join(" ").toLowerCase();
  if (filters.query && !haystack.includes(filters.query)) return false;
  if (filters.sourceType && transcript.source_type !== filters.sourceType) return false;
  if (filters.person && !hasEntity(nav.primary_people, filters.person)) return false;
  if (filters.location && !hasEntity(nav.primary_locations, filters.location)) return false;
  if (filters.event && !hasEvent(nav.referenced_events, filters.event)) return false;
  if (filters.topic && !(nav.topics || []).includes(filters.topic)) return false;
  if (!options.skipSourceDates && !dateInRange((nav.document_date || {}).normalized_date, filters.startDate, filters.endDate)) return false;
  return true;
}

function filteredEvents() {
  const filters = readFilters();
  const sourcePool = state.transcripts.filter((source) => sourceMatches(source, filters, { skipSourceDates: true }));
  const groups = new Map();
  sourcePool.forEach((source) => {
    (navigation(source).referenced_events || []).forEach((ref) => {
      if (filters.event && !hasEvent([ref], filters.event)) return;
      const eventDate = ((ref.event_time || {}).start || "").slice(0, 10);
      if (!dateInRange(eventDate || null, filters.startDate, filters.endDate)) return;
      const key = eventKey(ref);
      if (!groups.has(key)) {
        groups.set(key, { ...ref, key, sources: [] });
      }
      groups.get(key).sources.push({ source, ref });
    });
  });
  return Array.from(groups.values());
}

function filteredDocumentEvents() {
  const filters = readFilters();
  const sourcePool = state.transcripts.filter((source) => sourceMatches(source, filters, { skipSourceDates: true }));
  const groups = new Map();
  sourcePool.forEach((source) => {
    (navigation(source).document_events || []).forEach((ref) => {
      const eventDate = ((ref.event_time || {}).start || "").slice(0, 10);
      if (!dateInRange(eventDate || null, filters.startDate, filters.endDate)) return;
      const key = documentEventKey(ref);
      if (!groups.has(key)) {
        groups.set(key, { ...ref, key, sources: [] });
      }
      groups.get(key).sources.push({ source, ref });
    });
  });
  return Array.from(groups.values());
}

function readFilters() {
  return {
    query: document.getElementById("transcript-search").value.trim().toLowerCase(),
    sourceType: document.getElementById("source-type-filter").value,
    person: document.getElementById("person-nav-filter").value,
    location: document.getElementById("location-nav-filter").value,
    event: document.getElementById("event-nav-filter").value,
    topic: document.getElementById("topic-filter").value,
    startDate: document.getElementById("nav-start-date").value,
    endDate: document.getElementById("nav-end-date").value,
  };
}

function populateFilters() {
  fillSelect("source-type-filter", unique(state.transcripts.map((item) => item.source_type)), "All types");
  fillSelect("person-nav-filter", collectEntityOptions("primary_people"), "All people");
  fillSelect("location-nav-filter", collectEntityOptions("primary_locations"), "All places");
  fillSelect("event-nav-filter", collectEventOptions(), "All events");
  fillSelect("topic-filter", unique(state.transcripts.flatMap((item) => navigation(item).topics || [])), "All topics");
}

function fillSelect(id, options, label) {
  const select = document.getElementById(id);
  select.innerHTML = `<option value="">${escapeHtml(label)}</option>`;
  options.forEach((option) => {
    const element = document.createElement("option");
    if (typeof option === "string") {
      element.value = option;
      element.textContent = option;
    } else {
      element.value = option.value;
      element.textContent = option.label;
    }
    select.appendChild(element);
  });
}

function collectEntityOptions(key) {
  const options = new Map();
  state.transcripts.forEach((transcript) => {
    (navigation(transcript)[key] || []).forEach((item) => {
      const value = item.canonical_id || item.label;
      if (value && item.label) options.set(value, item.label);
    });
  });
  return Array.from(options, ([value, label]) => ({ value, label })).sort((a, b) => a.label.localeCompare(b.label));
}

function collectEventOptions() {
  const options = new Map();
  state.transcripts.forEach((transcript) => {
    (navigation(transcript).referenced_events || []).forEach((item) => {
      const value = item.canonical_id || eventKey(item);
      if (value && item.label) options.set(value, item.label);
    });
  });
  return Array.from(options, ([value, label]) => ({ value, label })).sort((a, b) => a.label.localeCompare(b.label));
}

function toggleSource(sourceId) {
  if (state.expandedSources.has(sourceId)) state.expandedSources.delete(sourceId);
  else state.expandedSources.add(sourceId);
  renderNavigator();
}

function toggleEvent(key) {
  if (state.expandedEvents.has(key)) state.expandedEvents.delete(key);
  else state.expandedEvents.add(key);
  renderNavigator();
}

async function loadTranscript(sourceId, pageRef = null) {
  const transcript = await window.HaymarketApi.getTranscript(sourceId);
  state.activeTranscript = transcript;
  state.pendingPageRef = pageRef;
  renderNavigator();
  renderTranscript();
  const url = new URL(window.location.href);
  url.searchParams.set("source", sourceId);
  if (pageRef) url.searchParams.set("page", pageRef);
  else url.searchParams.delete("page");
  window.history.replaceState({}, "", url);
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
  mentionSummary.textContent = `${mentions.length} visible mention${mentions.length === 1 ? "" : "s"} from ${(transcript.mentions || []).length} total`;
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

  if (state.pendingPageRef) {
    scrollToPageRef(state.pendingPageRef);
    state.pendingPageRef = null;
  }
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

function scrollToPageRef(pageRef) {
  const target = Array.from(document.querySelectorAll("[data-page-ref]")).find((element) => element.dataset.pageRef === pageRef);
  if (target) {
    target.scrollIntoView({ block: "start", behavior: "smooth" });
    target.classList.add("is-target");
    window.setTimeout(() => target.classList.remove("is-target"), 1800);
  }
}

function navigation(transcript) {
  return transcript.navigation || {};
}

function sourceMeta(transcript) {
  const nav = navigation(transcript);
  const date = formatDate((nav.document_date || {}).normalized_date) || (nav.document_date || {}).original_text || "Unknown date";
  const order = nav.document_order || {};
  const witness = (transcript.transcript_metadata || {}).witness_name;
  const pageLabel = order.sequence_label ? `pp. ${order.sequence_label}` : "";
  return [date, witness, order.volume ? `Vol. ${order.volume}` : "", pageLabel].filter(Boolean).join(" · ");
}

function eventMeta(eventRef) {
  const time = eventRef.event_time || {};
  const date = formatDate((time.start || "").slice(0, 10)) || time.original_text || "Undated";
  return [date, eventRef.location_label, confidenceLabel(eventRef.confidence)].filter(Boolean).join(" · ");
}

function sourceGroupKey(transcript) {
  const date = (navigation(transcript).document_date || {}).normalized_date;
  return formatDate(date) || "Unknown Date";
}

function eventGroupKey(eventRef) {
  const date = ((eventRef.event_time || {}).start || "").slice(0, 10);
  return formatDate(date) || "Undated Events";
}

function compareSources(a, b) {
  const navA = navigation(a);
  const navB = navigation(b);
  return compareNullable((navA.document_date || {}).normalized_date, (navB.document_date || {}).normalized_date)
    || compareNullable((navA.document_order || {}).volume, (navB.document_order || {}).volume)
    || compareNumbers((navA.document_order || {}).page_start, (navB.document_order || {}).page_start)
    || a.title.localeCompare(b.title);
}

function compareEventGroups(a, b) {
  return compareNullable(((a.event_time || {}).start || "").slice(0, 10), ((b.event_time || {}).start || "").slice(0, 10))
    || a.label.localeCompare(b.label);
}

function compareNullable(a, b) {
  if (!a && !b) return 0;
  if (!a) return 1;
  if (!b) return -1;
  return String(a).localeCompare(String(b));
}

function compareNumbers(a, b) {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return Number(a) - Number(b);
}

function groupedBy(items, keyFunc) {
  const groups = new Map();
  items.forEach((item) => {
    const key = keyFunc(item);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  });
  return groups;
}

function hasEntity(items, value) {
  return (items || []).some((item) => item.canonical_id === value || item.label === value);
}

function hasEvent(items, value) {
  return (items || []).some((item) => item.canonical_id === value || eventKey(item) === value || item.label === value);
}

function dateInRange(value, start, end) {
  if (!start && !end) return true;
  if (!value) return false;
  if (start && value < start) return false;
  if (end && value > end) return false;
  return true;
}

function eventKey(ref) {
  return ref.canonical_id || `${((ref.event_time || {}).start || "undated").slice(0, 10)}|${normalizeText(ref.label)}|${normalizeText(ref.location_label)}`;
}

function documentEventKey(ref) {
  return `${((ref.event_time || {}).start || "undated").slice(0, 10)}|${normalizeText(ref.event_kind)}|${normalizeText(ref.label)}|${normalizeText(ref.location_label)}`;
}

function eventKindLabel(value) {
  return String(value || "").replaceAll("_", " ");
}

function normalizeText(value) {
  return String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
}

function unique(values) {
  return Array.from(new Set(values.filter(Boolean))).sort((a, b) => String(a).localeCompare(String(b)));
}

function renderPageBadge(segment) {
  if (!segment.page_ref) return "";
  return `<span class="page-badge">p. ${escapeHtml(segment.page_ref)}</span>`;
}

function renderMention(text, mention) {
  const label = entityLabels.get(mention.entity_id) || mention.entity_id || mention.kind;
  return `<mark class="mention mention-${escapeAttribute(mention.kind || "unknown")}" title="${escapeAttribute(label)}">${escapeHtml(text)}</mark>`;
}

function renderChip(label, kind) {
  if (!label) return "";
  return `<span class="nav-chip chip-${escapeAttribute(kind)}">${escapeHtml(label)}</span>`;
}

function confidenceLabel(value) {
  if (typeof value !== "number") return "";
  return `${Math.round(value * 100)}% confidence`;
}

function formatDate(value) {
  if (!value) return "";
  return value;
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
  document.getElementById("transcript-search").addEventListener("input", renderNavigator);
  document.getElementById("mention-kind-filter").addEventListener("change", renderTranscript);
  ["source-type-filter", "person-nav-filter", "location-nav-filter", "event-nav-filter", "topic-filter", "nav-start-date", "nav-end-date"].forEach((id) => {
    document.getElementById(id).addEventListener("input", renderNavigator);
  });
  document.querySelectorAll("[data-view-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeView = button.dataset.viewTab;
      renderNavigator();
    });
  });

  try {
    const [transcripts, mapData] = await Promise.all([window.HaymarketApi.getTranscripts(), window.HaymarketApi.getMapData()]);
    state.transcripts = transcripts;
    state.mapData = mapData;
    indexEntityLabels(mapData);
    populateFilters();
    renderNavigator();
    if (transcripts.length > 0) {
      const params = new URLSearchParams(window.location.search);
      const requestedSource = params.get("source");
      const requestedPage = params.get("page");
      const initialSource = transcripts.some((transcript) => transcript.id === requestedSource) ? requestedSource : transcripts[0].id;
      await loadTranscript(initialSource, requestedPage);
    }
  } catch (error) {
    document.getElementById("transcript-count").textContent = "Unable to load transcripts.";
    document.getElementById("transcript-body").innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`;
  }
}

boot();
