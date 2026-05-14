import React, { useCallback, useEffect, useState } from "react";

/**
 * Map generation panel — template selector + natural language input.
 *
 * Props:
 *   onGenerate(result) — callback after successful generation
 */
export default function MapGeneratorPanel({ onGenerate }) {
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState("");
  const [nlPrompt, setNlPrompt] = useState("");
  const [originX, setOriginX] = useState(80);
  const [originY, setOriginY] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("template"); // "template" | "natural"

  // Fetch templates on mount
  useEffect(() => {
    fetch("/api/map/templates")
      .then(r => r.json())
      .then(data => {
        if (data.ok) setTemplates(data.templates);
      })
      .catch(() => {});
  }, []);

  const handleGenerateTemplate = useCallback(() => {
    if (!selectedTemplate || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);

    fetch("/api/map/template", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        template: selectedTemplate,
        origin_x: originX,
        origin_y: originY,
      }),
    })
      .then(r => r.json())
      .then(data => {
        if (data.ok) {
          setResult(data.spec);
          onGenerate?.(data);
        } else {
          setError(data.error || "Generation failed");
        }
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [selectedTemplate, originX, originY, loading, onGenerate]);

  const handleGenerateNL = useCallback(() => {
    if (!nlPrompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);

    fetch("/api/map/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: nlPrompt.trim(),
        origin_x: originX,
        origin_y: originY,
      }),
    })
      .then(r => r.json())
      .then(data => {
        if (data.ok) {
          setResult(data.spec);
          onGenerate?.(data);
        } else {
          setError(data.error || "Generation failed");
        }
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [nlPrompt, originX, originY, loading, onGenerate]);

  const selectedTmpl = templates.find(t => t.key === selectedTemplate);

  return (
    <div className="map-gen-panel">
      <div className="map-gen-header">
        <h3>{"\uD83D\uDDFA\uFE0F"} Map Generator</h3>
      </div>

      {/* Mode toggle */}
      <div className="map-gen-mode-toggle">
        <button
          className={`tab ${mode === "template" ? "active" : ""}`}
          onClick={() => setMode("template")}
        >
          {"\uD83D\uDCC4"} Template
        </button>
        <button
          className={`tab ${mode === "natural" ? "active" : ""}`}
          onClick={() => setMode("natural")}
        >
          {"\u270D\uFE0F"} Natural Language
        </button>
      </div>

      {/* Coordinate picker */}
      <div className="map-gen-coords">
        <label>
          Origin X:
          <input
            type="number"
            value={originX}
            onChange={e => setOriginX(parseInt(e.target.value) || 0)}
            className="coord-input"
          />
        </label>
        <label>
          Origin Y:
          <input
            type="number"
            value={originY}
            onChange={e => setOriginY(parseInt(e.target.value) || 0)}
            className="coord-input"
          />
        </label>
      </div>

      {mode === "template" ? (
        <div className="map-gen-template">
          <select
            className="map-gen-select"
            value={selectedTemplate}
            onChange={e => setSelectedTemplate(e.target.value)}
          >
            <option value="">Select a template...</option>
            {templates.map(t => (
              <option key={t.key} value={t.key}>
                {t.name} ({t.biome}) — {t.location_count} locations
              </option>
            ))}
          </select>
          {selectedTmpl && (
            <div className="map-gen-preview">
              <div className="preview-name">{selectedTmpl.name}</div>
              <div className="preview-desc">{selectedTmpl.description}</div>
              <div className="preview-meta">
                Biome: {selectedTmpl.biome} | {selectedTmpl.width}x{selectedTmpl.height} |{" "}
                {selectedTmpl.location_count} locations
              </div>
            </div>
          )}
          <button
            className="btn btn-primary"
            onClick={handleGenerateTemplate}
            disabled={!selectedTemplate || loading}
          >
            {loading ? "Generating..." : "\u2728 Generate from Template"}
          </button>
        </div>
      ) : (
        <div className="map-gen-nl">
          <textarea
            className="map-gen-textarea"
            rows={3}
            placeholder="Describe the area you want to create... (e.g., 'A bustling harbor town with a lighthouse and fish market')"
            value={nlPrompt}
            onChange={e => setNlPrompt(e.target.value)}
          />
          <button
            className="btn btn-primary"
            onClick={handleGenerateNL}
            disabled={!nlPrompt.trim() || loading}
          >
            {loading ? "AI Generating..." : "\u2728 Generate from Description"}
          </button>
        </div>
      )}

      {error && <div className="map-gen-error">{"\u26A0\uFE0F"} {error}</div>}

      {result && (
        <div className="map-gen-result">
          <div className="result-header">{"\u2705"} Generated: {result.name}</div>
          <div className="result-desc">{result.description}</div>
          <div className="result-meta">
            Biome: {result.biome} | Origin: ({result.origin[0]}, {result.origin[1]}) |{" "}
            {result.locations_placed} locations placed
          </div>
          <div className="result-locations">
            {result.locations?.map((loc, i) => (
              <span className="result-loc" key={i}>
                {loc.name} ({loc.type})
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
