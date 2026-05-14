import { useEffect, useState } from "react";

/**
 * Preloads the tilemap spritesheet and returns { loaded, image }.
 * The tilemap is a packed 192×176 PNG (12×11 tiles of 16×16).
 */
export default function useTileLoader() {
  const [state, setState] = useState({ loaded: false, image: null });

  useEffect(() => {
    const img = new Image();
    img.onload = () => setState({ loaded: true, image: img });
    img.onerror = () => {
      console.warn("Failed to load tilemap, falling back to colored rects");
      setState({ loaded: true, image: null });
    };
    img.src = "/tiles/tilemap.png";
    return () => { img.onload = null; img.onerror = null; };
  }, []);

  return state;
}
