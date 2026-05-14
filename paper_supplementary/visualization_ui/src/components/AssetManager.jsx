import React, { useCallback, useEffect, useState } from "react";

/**
 * Asset management sidebar — upload, list, delete custom assets.
 * Also supports AI art generation for buildings and backgrounds.
 *
 * Props:
 *   onAssetsChange() — callback when assets change (to refresh map)
 */
export default function AssetManager({ onAssetsChange }) {
  const [assets, setAssets] = useState([]);
  const [overrides, setOverrides] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Upload form state
  const [uploadName, setUploadName] = useState("");
  const [uploadType, setUploadType] = useState("building_sprite");
  const [uploadTarget, setUploadTarget] = useState("");
  const [uploadFile, setUploadFile] = useState(null);

  // AI generation state
  const [aiType, setAiType] = useState("building");
  const [aiBuildingType, setAiBuildingType] = useState("cafe");
  const [aiBiome, setAiBiome] = useState("town_center");
  const [aiDesc, setAiDesc] = useState("");
  const [aiLoading, setAiLoading] = useState(false);

  const fetchAssets = useCallback(() => {
    fetch("/api/assets")
      .then(r => r.json())
      .then(data => {
        if (data.ok) {
          setAssets(data.assets || []);
          setOverrides(data.overrides || {});
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchAssets();
  }, [fetchAssets]);

  const handleUpload = useCallback(() => {
    if (!uploadFile || !uploadName.trim() || loading) return;
    setLoading(true);
    setError(null);

    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(",")[1];
      fetch("/api/assets/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: uploadName.trim(),
          asset_type: uploadType,
          target: uploadTarget || `${uploadType}:default`,
          filename: uploadFile.name,
          file_data: base64,
        }),
      })
        .then(r => r.json())
        .then(data => {
          if (data.ok) {
            fetchAssets();
            onAssetsChange?.();
            setUploadName("");
            setUploadFile(null);
            setUploadTarget("");
          } else {
            setError(data.error || "Upload failed");
          }
        })
        .catch(err => setError(err.message))
        .finally(() => setLoading(false));
    };
    reader.readAsDataURL(uploadFile);
  }, [uploadFile, uploadName, uploadType, uploadTarget, loading, fetchAssets, onAssetsChange]);

  const handleDelete = useCallback((assetId) => {
    fetch(`/api/assets/${assetId}`, { method: "DELETE" })
      .then(r => r.json())
      .then(data => {
        if (data.ok) {
          fetchAssets();
          onAssetsChange?.();
        }
      })
      .catch(() => {});
  }, [fetchAssets, onAssetsChange]);

  const handleAiGenerate = useCallback(() => {
    if (aiLoading) return;
    setAiLoading(true);
    setError(null);

    const endpoint = aiType === "building" ? "/api/ai-art/building" : "/api/ai-art/background";
    const body = aiType === "building"
      ? { building_type: aiBuildingType, biome: aiBiome, name: aiDesc || aiBuildingType }
      : { biome: aiBiome, description: aiDesc, season: "spring" };

    fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(r => r.json())
      .then(data => {
        if (data.ok) {
          fetchAssets();
          onAssetsChange?.();
        } else {
          setError(data.error || "AI generation failed");
        }
      })
      .catch(err => setError(err.message))
      .finally(() => setAiLoading(false));
  }, [aiType, aiBuildingType, aiBiome, aiDesc, aiLoading, fetchAssets, onAssetsChange]);

  return (
    <div className="asset-manager">
      <div className="asset-header">
        <h3>{"\uD83C\uDFA8"} Asset Manager</h3>
      </div>

      {/* Upload section */}
      <div className="asset-section">
        <h4>Upload Custom Asset</h4>
        <div className="asset-upload-form">
          <input
            type="text"
            className="asset-input"
            placeholder="Asset name"
            value={uploadName}
            onChange={e => setUploadName(e.target.value)}
          />
          <select
            className="asset-select"
            value={uploadType}
            onChange={e => setUploadType(e.target.value)}
          >
            <option value="building_sprite">Building Sprite</option>
            <option value="character_sprite">Character Sprite</option>
            <option value="background">Background</option>
            <option value="tilemap">Tilemap</option>
          </select>
          <input
            type="text"
            className="asset-input"
            placeholder="Target (e.g., building:cafe)"
            value={uploadTarget}
            onChange={e => setUploadTarget(e.target.value)}
          />
          <input
            type="file"
            accept="image/*"
            onChange={e => setUploadFile(e.target.files[0] || null)}
            className="asset-file-input"
          />
          <button
            className="btn btn-primary btn-sm"
            onClick={handleUpload}
            disabled={!uploadFile || !uploadName.trim() || loading}
          >
            {loading ? "Uploading..." : "\u2B06 Upload"}
          </button>
        </div>
      </div>

      {/* AI Generation section */}
      <div className="asset-section">
        <h4>{"\u2728"} AI Art Generation</h4>
        <div className="asset-ai-form">
          <select
            className="asset-select"
            value={aiType}
            onChange={e => setAiType(e.target.value)}
          >
            <option value="building">Building Sprite</option>
            <option value="background">Background</option>
          </select>
          {aiType === "building" && (
            <select
              className="asset-select"
              value={aiBuildingType}
              onChange={e => setAiBuildingType(e.target.value)}
            >
              {["cafe", "temple", "library", "shop", "bakery", "home", "teahouse",
                "pharmacy", "school", "gallery", "townhall", "park", "square"].map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          )}
          <select
            className="asset-select"
            value={aiBiome}
            onChange={e => setAiBiome(e.target.value)}
          >
            {["town_center", "farmland", "riverside", "forest", "mountain"].map(b => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
          <input
            type="text"
            className="asset-input"
            placeholder="Description (optional)"
            value={aiDesc}
            onChange={e => setAiDesc(e.target.value)}
          />
          <button
            className="btn btn-primary btn-sm"
            onClick={handleAiGenerate}
            disabled={aiLoading}
          >
            {aiLoading ? "Generating..." : "\uD83E\uDD16 Generate"}
          </button>
        </div>
      </div>

      {error && <div className="asset-error">{"\u26A0\uFE0F"} {error}</div>}

      {/* Asset list */}
      <div className="asset-section">
        <h4>Custom Assets ({assets.length})</h4>
        <div className="asset-list">
          {assets.length === 0 && (
            <div className="asset-empty">No custom assets yet.</div>
          )}
          {assets.map(asset => (
            <div className="asset-item" key={asset.id}>
              <div className="asset-item-thumb">
                <img
                  src={asset.url}
                  alt={asset.name}
                  onError={e => { e.target.style.display = "none"; }}
                />
              </div>
              <div className="asset-item-info">
                <div className="asset-item-name">{asset.name}</div>
                <div className="asset-item-meta">
                  {asset.asset_type} | {asset.target}
                </div>
              </div>
              <button
                className="btn btn-sm btn-danger"
                onClick={() => handleDelete(asset.id)}
                title="Delete asset"
              >
                {"\u2715"}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
