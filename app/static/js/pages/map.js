const palette = [
  "#8e3b2f",
  "#2f6f8e",
  "#4d7c3f",
  "#7a4f9a",
  "#b36b1f",
  "#2f7f72",
  "#9a3f68",
  "#5b5f7a",
];

let map;
let markers = [];
let state = {
  people: [],
  events: [],
  claims: [],
  locations: [],
  sources: [],
  activeEventId: null,
};

const peopleById = new Map();
const locationsById = new Map();
const claimsById = new Map();
const sourcesById = new Map();

function colorForPerson(personId) {
  if (!personId) return "#666";
  let hash = 0;
  for (const char of personId) {
    hash = (hash + char.charCodeAt(0)) % palette.length;
  }
  return palette[hash];
}

function initializeMap() {
  map = L.map("map").setView([41.884, -87.644], 14);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(map);
}

function indexData(data) {
  state = {
    people: data.people || [],
    events: data.events || [],
    claims: data.claims || [],
    locations: data.locations || [],
    sources: data.sources || [],
    activeEventId: null,
  };

  peopleById.clear();
  locationsById.clear();
  claimsById.clear();
  sourcesById.clear();
  state.people.forEach((person) => peopleById.set(person.id, person));
  state.locations.forEach((location) => locationsById.set(location.id, location));
  state.claims.forEach((claim) => claimsById.set(claim.id, claim));
  state.sources.forEach((source) => sourcesById.set(source.id, source));
}

function populatePeopleFilter() {
  const select = document.getElementById("person-filter");
  state.people
    .slice()
    .sort((a, b) => a.display_name.localeCompare(b.display_name))
    .forEach((person) => {
      const option = document.createElement("option");
      option.value = person.id;
      option.textContent = person.display_name;
      select.appendChild(option);
    });
}

function eventDate(event) {
  const value = event.time && event.time.start;
  return value ? value.slice(0, 10) : "";
}

function passesFilters(event) {
  const selectedPerson = document.getElementById("person-filter").value;
  const startDate = document.getElementById("start-date").value;
  const endDate = document.getElementById("end-date").value;
  const minConfidence = Number(document.getElementById("confidence-filter").value);
  const date = eventDate(event);

  if (selectedPerson && !(event.participant_person_ids || []).includes(selectedPerson)) {
    return false;
  }
  if (startDate && date && date < startDate) return false;
  if (endDate && date && date > endDate) return false;
  if ((event.confidence || 0) < minConfidence) return false;
  return true;
}

function clearMarkers() {
  markers.forEach((marker) => marker.remove());
  markers = [];
}

function renderMarkers(events) {
  clearMarkers();
  const bounds = [];

  events.forEach((event) => {
    const location = locationsById.get(event.location_id);
    const coords = location && location.coordinates;
    if (!coords || coords.lat == null || coords.lng == null) return;

    const primaryPerson = (event.participant_person_ids || [])[0];
    const marker = L.circleMarker([coords.lat, coords.lng], {
      color: colorForPerson(primaryPerson),
      fillColor: colorForPerson(primaryPerson),
      fillOpacity: 0.78,
      radius: 9,
      weight: 2,
    });

    marker.bindPopup(`<strong>${escapeHtml(event.title)}</strong><br>${escapeHtml(location.name || "")}`);
    marker.on("click", () => showEventDetails(event.id));
    marker.addTo(map);
    markers.push(marker);
    bounds.push([coords.lat, coords.lng]);
  });

  if (bounds.length > 0) {
    map.fitBounds(bounds, { maxZoom: 15, padding: [30, 30] });
  }
}

function renderEventList(events) {
  const list = document.getElementById("event-list");
  const count = document.getElementById("event-count");
  list.innerHTML = "";
  count.textContent = `${events.length} event${events.length === 1 ? "" : "s"} shown`;

  if (events.length === 0) {
    list.innerHTML = '<p class="muted">No events match the current filters.</p>';
    return;
  }

  events.forEach((event) => {
    const card = document.createElement("article");
    card.className = `event-card${state.activeEventId === event.id ? " is-active" : ""}`;
    card.tabIndex = 0;
    card.innerHTML = `
      <h3>${escapeHtml(event.title)}</h3>
      <div class="meta">${escapeHtml((event.time && event.time.display) || eventDate(event) || "Unknown time")}</div>
      <p>${escapeHtml(event.description || "")}</p>
      ${renderPeoplePills(event.participant_person_ids || [])}
    `;
    card.addEventListener("click", () => showEventDetails(event.id));
    list.appendChild(card);
  });
}

function renderPeoplePills(personIds) {
  if (personIds.length === 0) return "";
  const pills = personIds
    .map((personId) => {
      const person = peopleById.get(personId);
      const name = person ? person.display_name : personId;
      return `<span class="pill" style="background:${colorForPerson(personId)}">${escapeHtml(name)}</span>`;
    })
    .join("");
  return `<div class="pill-row">${pills}</div>`;
}

function showEventDetails(eventId) {
  state.activeEventId = eventId;
  const event = state.events.find((item) => item.id === eventId);
  if (!event) return;

  const location = locationsById.get(event.location_id);
  const claims = (event.claim_ids || []).map((claimId) => claimsById.get(claimId)).filter(Boolean);
  const details = document.getElementById("details-panel");
  details.innerHTML = `
    <h2>${escapeHtml(event.title)}</h2>
    <p>${escapeHtml(event.description || "")}</p>
    <p class="meta">${escapeHtml((event.time && event.time.display) || eventDate(event) || "Unknown time")}</p>
    ${renderPeoplePills(event.participant_person_ids || [])}
    <h3>Location</h3>
    <p>${escapeHtml(location ? location.name : "Unknown")}</p>
    <p class="meta">${escapeHtml(location ? location.address_1886 || "" : "")}</p>
    <h3>Claims</h3>
    ${claims.length ? claims.map(renderClaim).join("") : '<p class="muted">No linked claims yet.</p>'}
  `;

  applyFilters();
}

function renderClaim(claim) {
  const reporter = peopleById.get(claim.reported_by_person_id);
  const source = claim.source || sourcesById.get(claim.source_id) || {};
  const sourceLink = source.url
    ? `<a href="${escapeAttribute(source.url)}" target="_blank" rel="noreferrer">source</a>`
    : "";
  const transcriptLink = source.id ? `<a href="/transcripts?source=${escapeAttribute(source.id)}">transcript</a>` : "";

  return `
    <div class="claim">
      <p>${escapeHtml(claim.statement || "")}</p>
      <p class="meta">
        Reported by ${escapeHtml(reporter ? reporter.display_name : claim.reported_by_person_id || "unknown")}
        ${claim.claim_made_at ? `on ${escapeHtml(claim.claim_made_at)}` : ""}
        ${sourceLink}
        ${transcriptLink}
      </p>
      ${claim.quote ? `<blockquote>${escapeHtml(claim.quote)}</blockquote>` : ""}
    </div>
  `;
}

function applyFilters() {
  document.getElementById("confidence-output").value = Number(
    document.getElementById("confidence-filter").value,
  ).toFixed(2);

  const events = state.events.filter(passesFilters);
  renderMarkers(events);
  renderEventList(events);
}

function wireFilters() {
  ["person-filter", "start-date", "end-date", "confidence-filter"].forEach((id) => {
    document.getElementById(id).addEventListener("input", applyFilters);
  });
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
  initializeMap();
  wireFilters();

  try {
    const data = await window.HaymarketApi.getMapData();
    indexData(data);
    populatePeopleFilter();
    applyFilters();
  } catch (error) {
    document.getElementById("event-count").textContent = "Unable to load map data.";
    document.getElementById("event-list").innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`;
  }
}

boot();
