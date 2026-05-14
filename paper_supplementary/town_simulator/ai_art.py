"""AI art generation for AURA Town (building sprites, backgrounds)."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AIArtGenerator:
    """Generate pixel-art style sprites and backgrounds using a configurable image API."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._api_url = api_url
        self._api_key = api_key

    @property
    def available(self) -> bool:
        """True if image generation API is configured."""
        return bool(self._api_url)

    def generate_building_sprite(
        self,
        building_type: str,
        biome: str,
        name: str,
        width: int = 64,
        height: int = 64,
    ) -> Optional[bytes]:
        """Generate a pixel-art building sprite.

        Args:
            building_type: Type of building (cafe, temple, etc.)
            biome: Biome context (forest, riverside, etc.)
            name: Building name for thematic context
            width, height: Desired image dimensions

        Returns:
            PNG image bytes, or None if generation fails.
        """
        if not self.available:
            return None

        prompt = (
            f"Pixel art sprite of a {building_type} building called '{name}', "
            f"in a {biome.replace('_', ' ')} setting. "
            f"Chinese/Asian-inspired architecture, top-down RPG style, "
            f"16-bit retro game aesthetic, clean pixel art, "
            f"transparent background, {width}x{height} pixels."
        )

        return self._call_api(prompt, width, height)

    def generate_background(
        self,
        biome: str,
        season: str,
        description: str,
        width: int = 640,
        height: int = 512,
    ) -> Optional[bytes]:
        """Generate a region background image.

        Args:
            biome: Biome type (forest, mountain, riverside, etc.)
            season: Current season
            description: Additional description for the scene
            width, height: Desired image dimensions

        Returns:
            PNG image bytes, or None if generation fails.
        """
        if not self.available:
            return None

        season_desc = {
            "spring": "cherry blossoms, fresh green, warm light",
            "summer": "lush foliage, bright sun, vibrant colors",
            "autumn": "golden leaves, warm tones, misty atmosphere",
            "winter": "snow-covered, bare branches, soft blue light",
        }

        biome_desc = {
            "town_center": "a bustling Chinese town with traditional buildings",
            "farmland": "peaceful rice paddies and farmhouses",
            "riverside": "a scenic river with willow trees and bridges",
            "forest": "a mystical bamboo forest with dappled light",
            "mountain": "misty mountain peaks with monasteries",
        }

        prompt = (
            f"Pixel art background scene: {biome_desc.get(biome, biome)}, "
            f"{season_desc.get(season, season)} season. "
            f"{description}. "
            f"Top-down RPG style tilemap background, 16-bit retro game aesthetic, "
            f"Chinese/Asian-inspired landscape art, "
            f"{width}x{height} pixels."
        )

        return self._call_api(prompt, width, height)

    def _call_api(self, prompt: str, width: int, height: int) -> Optional[bytes]:
        """Call the configured image generation API.

        This method supports common image generation API formats.
        Override this for custom API integrations.
        """
        if not self._api_url:
            logger.warning("No image generation API URL configured")
            return None

        try:
            import urllib.request
            import json

            payload = json.dumps({
                "prompt": prompt,
                "width": width,
                "height": height,
                "n": 1,
            }).encode("utf-8")

            headers = {
                "Content-Type": "application/json",
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            req = urllib.request.Request(
                self._api_url,
                data=payload,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            # Try common response formats
            # Format 1: {"data": [{"url": "..."}]} (OpenAI-style)
            if "data" in result and isinstance(result["data"], list):
                image_url = result["data"][0].get("url")
                if image_url:
                    with urllib.request.urlopen(image_url, timeout=30) as img_resp:
                        return img_resp.read()

            # Format 2: {"images": ["base64..."]} (SD-style)
            if "images" in result and isinstance(result["images"], list):
                import base64
                return base64.b64decode(result["images"][0])

            # Format 3: {"image": "base64..."} (simple)
            if "image" in result and isinstance(result["image"], str):
                import base64
                return base64.b64decode(result["image"])

            # Format 4: raw bytes in response
            if isinstance(result, bytes):
                return result

            logger.warning("Unrecognized image API response format")
            return None

        except Exception as e:
            logger.error("Image generation API call failed: %s", e)
            return None
