"""Custom asset management for AURA Town."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CustomAsset:
    """A user-uploaded custom asset."""

    id: str
    name: str
    asset_type: str  # tilemap, character_sprite, background, building_sprite
    filename: str
    target: str  # what this overrides (e.g., "building:cafe", "character:Lin Wei", "background:forest")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "asset_type": self.asset_type,
            "filename": self.filename,
            "target": self.target,
            "url": f"/custom-assets/{self.filename}",
        }


class AssetManager:
    """Manages custom asset uploads and overrides."""

    ALLOWED_TYPES = {"tilemap", "character_sprite", "background", "building_sprite"}
    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

    def __init__(self, assets_dir: Optional[str] = None) -> None:
        if assets_dir is None:
            # Default: visualization-ui/public/custom-assets/
            assets_dir = str(
                Path(__file__).resolve().parent.parent
                / "visualization-ui" / "public" / "custom-assets"
            )
        self._assets_dir = Path(assets_dir)
        self._assets_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._assets_dir / "manifest.json"
        self._assets: Dict[str, CustomAsset] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load asset manifest from disk."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("assets", []):
                    asset = CustomAsset(**item)
                    self._assets[asset.id] = asset
            except Exception as e:
                logger.warning("Failed to load asset manifest: %s", e)

    def _save_manifest(self) -> None:
        """Persist asset manifest to disk."""
        data = {"assets": [a.to_dict() for a in self._assets.values()]}
        # Remove 'url' from saved data (it's computed)
        for item in data["assets"]:
            item.pop("url", None)
        try:
            with open(self._manifest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save asset manifest: %s", e)

    def upload_asset(
        self,
        name: str,
        asset_type: str,
        file_data: bytes,
        filename: str,
        target: str,
    ) -> Optional[CustomAsset]:
        """Upload a new custom asset.

        Args:
            name: Display name for the asset
            asset_type: One of ALLOWED_TYPES
            file_data: Raw file bytes
            filename: Original filename
            target: Override target (e.g., "building:cafe", "character:Lin Wei")

        Returns:
            CustomAsset on success, None on failure.
        """
        # Validate type
        if asset_type not in self.ALLOWED_TYPES:
            logger.warning("Invalid asset type: %s", asset_type)
            return None

        # Validate extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            logger.warning("Invalid file extension: %s", ext)
            return None

        # Validate size
        if len(file_data) > self.MAX_FILE_SIZE:
            logger.warning("File too large: %d bytes", len(file_data))
            return None

        # Generate unique filename
        asset_id = uuid.uuid4().hex[:12]
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        stored_filename = f"{asset_id}_{safe_name}{ext}"

        # Write file
        filepath = self._assets_dir / stored_filename
        try:
            with open(filepath, "wb") as f:
                f.write(file_data)
        except Exception as e:
            logger.error("Failed to write asset file: %s", e)
            return None

        asset = CustomAsset(
            id=asset_id,
            name=name,
            asset_type=asset_type,
            filename=stored_filename,
            target=target,
        )
        self._assets[asset_id] = asset
        self._save_manifest()
        logger.info("Uploaded asset: %s (%s) -> %s", name, asset_type, target)
        return asset

    def delete_asset(self, asset_id: str) -> bool:
        """Delete an asset by ID."""
        asset = self._assets.get(asset_id)
        if asset is None:
            return False

        # Delete file
        filepath = self._assets_dir / asset.filename
        try:
            if filepath.exists():
                filepath.unlink()
        except Exception as e:
            logger.warning("Failed to delete asset file: %s", e)

        del self._assets[asset_id]
        self._save_manifest()
        logger.info("Deleted asset: %s", asset_id)
        return True

    def list_assets(self, asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all assets, optionally filtered by type."""
        assets = self._assets.values()
        if asset_type:
            assets = [a for a in assets if a.asset_type == asset_type]
        return [a.to_dict() for a in assets]

    def get_asset_overrides(self) -> Dict[str, str]:
        """Get target → URL mapping for all custom assets.

        Used by the frontend to override default rendering.
        """
        return {
            asset.target: f"/custom-assets/{asset.filename}"
            for asset in self._assets.values()
        }
