window.HaymarketApi = {
  async getMapData() {
    const response = await fetch("/api/map-data");
    if (!response.ok) {
      throw new Error(`Failed to load map data: ${response.status}`);
    }
    return response.json();
  },

  async getTranscripts() {
    const response = await fetch("/api/transcripts");
    if (!response.ok) {
      throw new Error(`Failed to load transcripts: ${response.status}`);
    }
    return response.json();
  },

  async getTranscript(sourceId) {
    const response = await fetch(`/api/transcripts/${encodeURIComponent(sourceId)}`);
    if (!response.ok) {
      throw new Error(`Failed to load transcript: ${response.status}`);
    }
    return response.json();
  },
};
