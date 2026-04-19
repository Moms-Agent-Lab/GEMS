"""
SkillStore — Version-controlled skill storage with CRUD operations.

Supports creating, updating, merging, and deleting skills with automatic
snapshot-based versioning so mutations can be rolled back if they degrade
benchmark performance.

Snapshots are stored as timestamped copies under a ``.versions/`` directory
inside the skills root::

    skills/
    ├── .versions/
    │   ├── spatial__v1__20260413T120000.md
    │   └── spatial__v2__20260413T130000.md
    ├── spatial/
    │   └── SKILL.md
    └── ...
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SkillStore:
    """Manages skill lifecycle with version history.

    Parameters
    ----------
    skills_dir : Root directory containing skill sub-directories.
    """

    def __init__(self, skills_dir: str | Path) -> None:
        self.root = Path(skills_dir).resolve()
        self._versions_dir = self.root / ".versions"
        self._versions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_skill(
        self,
        name: str,
        description: str,
        body: str,
        metadata: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> Path:
        """Create a new skill directory with a SKILL.md file.

        The ``tags`` list is written into the frontmatter verbatim (after
        deduplication and ``"agent"`` auto-injection, see :meth:`_normalize_tags`).
        Raises ``FileExistsError`` if the skill already exists.
        """
        skill_dir = self.root / name
        if skill_dir.exists():
            raise FileExistsError(f"Skill {name!r} already exists at {skill_dir}")

        skill_dir.mkdir(parents=True)
        content = self._build_skill_md(
            name, description, body, metadata, self._normalize_tags(tags)
        )
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(content, encoding="utf-8")
        self._snapshot(name, "v1")
        return skill_md

    def update_skill(
        self,
        name: str,
        description: str | None = None,
        body: str | None = None,
        metadata: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> Path:
        """Update an existing skill, snapshotting the old version first.

        Only the provided fields are updated; ``None`` means keep existing.
        Tag semantics: when ``tags`` is given, the union of existing tags and
        new tags is written back (tags accumulate, never silently removed).
        """
        skill_dir = self.root / name
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"Skill {name!r} not found at {skill_md}")

        version = self._next_version(name)
        self._snapshot(name, version)

        existing = self._parse_existing(skill_md)
        new_desc = description if description is not None else existing["description"]
        new_body = body if body is not None else existing["body"]
        new_meta = metadata if metadata is not None else existing.get("metadata")

        existing_tags = existing.get("tags") or []
        if tags is None:
            merged_tags = existing_tags or None
        else:
            merged_tags = self._normalize_tags(list(existing_tags) + list(tags))

        content = self._build_skill_md(name, new_desc, new_body, new_meta, merged_tags)
        skill_md.write_text(content, encoding="utf-8")
        return skill_md

    def delete_skill(self, name: str) -> None:
        """Delete a skill directory (snapshot first for recovery)."""
        skill_dir = self.root / name
        if not skill_dir.exists():
            raise FileNotFoundError(f"Skill {name!r} not found at {skill_dir}")

        version = self._next_version(name)
        self._snapshot(name, version, tag="deleted")
        shutil.rmtree(skill_dir)

    def merge_skills(
        self,
        names: list[str],
        merged_name: str,
        merged_description: str,
        merged_body: str,
        delete_originals: bool = True,
        tags: list[str] | None = None,
    ) -> Path:
        """Merge multiple skills into one new skill.

        The merged skill's tags default to the union of the originals' tags
        (so model/bench partitioning is preserved across merges) plus any
        explicit ``tags`` argument.
        """
        for name in names:
            skill_dir = self.root / name
            if not skill_dir.exists():
                raise FileNotFoundError(f"Skill {name!r} not found")

        # Collect union of existing tags before snapshotting so a merge
        # doesn't silently lose the model:*/bench:* partitioning tags.
        union_tags: list[str] = list(tags or [])
        for name in names:
            md = self.root / name / "SKILL.md"
            if md.exists():
                union_tags.extend(self._parse_existing(md).get("tags") or [])

        for name in names:
            version = self._next_version(name)
            self._snapshot(name, version, tag="pre-merge")

        # The merged destination may collide with one of the originals; if so,
        # remove it first so create_skill can write cleanly.
        if (self.root / merged_name).exists():
            shutil.rmtree(self.root / merged_name, ignore_errors=True)

        result = self.create_skill(
            merged_name, merged_description, merged_body, tags=union_tags
        )

        if delete_originals:
            for name in names:
                if name != merged_name:
                    shutil.rmtree(self.root / name, ignore_errors=True)

        return result

    def rollback_skill(self, name: str, version: str | None = None) -> Path:
        """Restore a skill from a previous snapshot.

        If *version* is ``None``, restores the most recent snapshot.
        """
        snapshots = self.list_versions(name)
        if not snapshots:
            raise FileNotFoundError(f"No snapshots found for skill {name!r}")

        if version:
            target = None
            for snap in snapshots:
                if snap["version"] == version:
                    target = snap
                    break
            if not target:
                raise FileNotFoundError(
                    f"Version {version!r} not found for skill {name!r}"
                )
        else:
            target = snapshots[-1]

        snapshot_path = Path(target["path"])
        skill_dir = self.root / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"

        # Snapshot current state before rollback
        if skill_md.exists():
            rb_version = self._next_version(name)
            self._snapshot(name, rb_version, tag="pre-rollback")

        shutil.copy2(snapshot_path, skill_md)
        return skill_md

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_skills(self) -> list[str]:
        """Return sorted list of skill names."""
        return sorted(
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (d / "SKILL.md").exists()
        )

    def list_versions(self, name: str) -> list[dict[str, str]]:
        """Return snapshot history for a skill, oldest first."""
        snapshots: list[dict[str, str]] = []
        prefix = f"{name}__"
        for f in sorted(self._versions_dir.iterdir()):
            if f.name.startswith(prefix) and f.suffix == ".md":
                parts = f.stem.split("__")
                version = parts[1] if len(parts) >= 2 else "unknown"
                tag = parts[2] if len(parts) >= 3 else ""
                snapshots.append({
                    "name": name,
                    "version": version,
                    "tag": tag,
                    "path": str(f),
                })
        return snapshots

    def get_skill_content(self, name: str) -> str:
        """Read current SKILL.md content."""
        skill_md = self.root / name / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"Skill {name!r} not found")
        return skill_md.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tags(tags: list[str] | None) -> list[str]:
        """Deduplicate, strip, and guarantee the ``agent`` tag is present.

        Accepts messy input (``None``, whitespace, duplicates, tuples) and
        returns a stable, sorted list.  ``"agent"`` is always included so the
        skill is discoverable by :meth:`SkillManager.build_available_skills_xml`
        which filters on ``include_tags={"agent", …}``.
        """
        seen: set[str] = set()
        out: list[str] = []
        for t in tags or []:
            if t is None:
                continue
            t_clean = str(t).strip()
            if not t_clean or t_clean in seen:
                continue
            seen.add(t_clean)
            out.append(t_clean)
        if "agent" not in seen:
            out.append("agent")
        return sorted(out)

    def _build_skill_md(
        self,
        name: str,
        description: str,
        body: str,
        metadata: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Generate SKILL.md content with YAML frontmatter."""
        lines = ["---", f"name: {name}"]
        if "\n" in description or len(description) > 80:
            lines.append("description: >-")
            for dline in description.split("\n"):
                lines.append(f"  {dline.strip()}")
        else:
            lines.append(f"description: \"{description}\"")
        lines.append("license: MIT")
        if metadata:
            lines.append("metadata:")
            for k, v in sorted(metadata.items()):
                lines.append(f"  {k}: \"{v}\"")
        effective_tags = tags if tags else ["agent"]
        lines.append(f"tags: [{', '.join(effective_tags)}]")
        lines.append("---")
        lines.append("")
        lines.append(body)
        return "\n".join(lines)

    def _snapshot(self, name: str, version: str, tag: str = "") -> Path:
        """Copy the current SKILL.md to the versions directory."""
        skill_md = self.root / name / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"Cannot snapshot: {skill_md} does not exist")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        parts = [name, version, ts]
        if tag:
            parts.append(tag)
        snap_name = "__".join(parts) + ".md"
        snap_path = self._versions_dir / snap_name
        shutil.copy2(skill_md, snap_path)
        return snap_path

    def _next_version(self, name: str) -> str:
        """Compute the next version number for a skill."""
        existing = self.list_versions(name)
        max_v = 0
        for snap in existing:
            m = re.match(r"v(\d+)", snap["version"])
            if m:
                max_v = max(max_v, int(m.group(1)))
        return f"v{max_v + 1}"

    def _parse_existing(self, skill_md: Path) -> dict[str, Any]:
        """Parse an existing SKILL.md into components (incl. tags list)."""
        import yaml

        content = skill_md.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return {"description": "", "body": content, "metadata": None, "tags": []}

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {"description": "", "body": content, "metadata": None, "tags": []}

        try:
            fm = yaml.safe_load(parts[1]) or {}
        except Exception:
            fm = {}

        raw_tags = fm.get("tags")
        tags = list(raw_tags) if isinstance(raw_tags, list) else []

        return {
            "description": str(fm.get("description", "")),
            "body": parts[2].strip(),
            "metadata": fm.get("metadata"),
            "tags": tags,
        }
