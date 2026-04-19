"""End-to-end coverage for the two new cross-cutting features:

1. ``tags`` flow-through — ``SkillEvolver.auto_tags`` → ``SkillStore.create_skill``
   → SKILL.md frontmatter → ``SkillManager.build_available_skills_xml(include_tags=…)``.
2. ``ARCH_REGISTRY`` is assembled from per-skill ``arch.yaml`` files rather
   than a hard-coded Python dict.  Adding/removing an ``arch.yaml`` must flip
   the registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from comfyclaw.agent import ARCH_REGISTRY, load_arch_registry
from comfyclaw.evolve import MutationProposal, SkillEvolver
from comfyclaw.skill_manager import SkillManager
from comfyclaw.skill_store import SkillStore


# ---------------------------------------------------------------------------
# 1. Tag flow: SkillStore <-> evolver <-> SkillManager filter
# ---------------------------------------------------------------------------


class TestSkillStoreTags:
    def test_create_skill_writes_tags_into_frontmatter(self, tmp_path: Path) -> None:
        store = SkillStore(tmp_path)
        skill_md = store.create_skill(
            name="counting-helper",
            description="Counting-specific tips",
            body="# Counting\nUse regional prompting.",
            tags=["model:longcat", "bench:geneval2", "topic:counting"],
        )
        content = skill_md.read_text(encoding="utf-8")
        # Every written skill must carry the partitioning tags + auto "agent".
        assert "tags: [" in content
        for expected in (
            "agent",
            "bench:geneval2",
            "model:longcat",
            "topic:counting",
        ):
            assert expected in content, f"{expected!r} missing from {content!r}"

    def test_update_skill_unions_new_tags_with_existing(
        self, tmp_path: Path
    ) -> None:
        store = SkillStore(tmp_path)
        store.create_skill(
            name="s1",
            description="d",
            body="b",
            tags=["model:qwen", "bench:geneval2"],
        )
        store.update_skill(name="s1", body="b v2", tags=["topic:counting"])
        content = (tmp_path / "s1" / "SKILL.md").read_text(encoding="utf-8")
        # Existing partition tags must be preserved across updates.
        for expected in (
            "agent",
            "bench:geneval2",
            "model:qwen",
            "topic:counting",
        ):
            assert expected in content

    def test_auto_agent_tag_present_even_when_not_supplied(
        self, tmp_path: Path
    ) -> None:
        store = SkillStore(tmp_path)
        store.create_skill(name="s2", description="d", body="b", tags=[])
        content = (tmp_path / "s2" / "SKILL.md").read_text(encoding="utf-8")
        assert "agent" in content.split("tags:")[1]


class TestEvolverAutoTags:
    def test_apply_mutation_tags_skill_with_model_and_bench(
        self, tmp_path: Path
    ) -> None:
        evolver = SkillEvolver(
            evolved_skills_dir=tmp_path,
            llm_model="anthropic/claude-test",
            auto_tags=["model:longcat", "bench:geneval2"],
        )

        mutation = MutationProposal(
            mutation_type="create",
            target_skills=[],
            rationale="unit test",
            failure_cluster="counting-objects",
            proposed_changes={
                "name": "counting-objects-skill",
                "description": "Help with counting.",
                "body": "# Counting\n...",
                # LLM-proposed tag should also flow through.
                "tags": ["topic:counting"],
            },
        )

        evolver._apply_mutation(mutation)

        content = (
            tmp_path / "counting-objects-skill" / "SKILL.md"
        ).read_text(encoding="utf-8")
        for expected in (
            "agent",
            "bench:geneval2",
            "model:longcat",
            "topic:counting",
        ):
            assert expected in content


class TestSkillManagerIncludeTagsFilter:
    def test_include_tags_filters_by_model_and_bench(
        self, tmp_path: Path
    ) -> None:
        store = SkillStore(tmp_path)
        store.create_skill(
            name="longcat-only",
            description="LongCat-specific trick",
            body="body",
            tags=["model:longcat", "bench:geneval2"],
        )
        store.create_skill(
            name="qwen-only",
            description="Qwen-specific trick",
            body="body",
            tags=["model:qwen", "bench:geneval2"],
        )
        store.create_skill(
            name="generic",
            description="Works everywhere",
            body="body",
            tags=[],
        )

        sm = SkillManager(skills_dir=None, evolved_skills_dir=str(tmp_path))
        # Force-mark them as 'agent'-visible so the filter logic kicks in.
        xml = sm.build_available_skills_xml(
            include_tags={"agent", "model:longcat", "bench:geneval2"}
        )

        assert "longcat-only" in xml
        assert "generic" in xml           # no model:* tag → any-tag-overlap path
        assert "qwen-only" not in xml     # model:* tag but not selected


# ---------------------------------------------------------------------------
# 2. ARCH_REGISTRY is sourced from per-skill arch.yaml files
# ---------------------------------------------------------------------------


class TestArchYamlLoader:
    def test_builtin_registry_has_expected_entries(self) -> None:
        assert set(ARCH_REGISTRY) == {"qwen_image", "z_image", "longcat_image"}

    def test_longcat_entry_matches_yaml(self) -> None:
        cfg = ARCH_REGISTRY["longcat_image"]
        assert cfg.skill_name == "longcat-image"
        assert cfg.lora_supported is False
        assert cfg.lora_node == ""
        assert "longcat" in cfg.unet_keywords
        assert "LongCatImageModelLoader" in cfg.node_classes

    def test_qwen_entry_matches_yaml(self) -> None:
        cfg = ARCH_REGISTRY["qwen_image"]
        assert cfg.skill_name == "qwen-image-2512"
        assert cfg.lora_supported is True
        assert cfg.lora_node == "LoraLoaderModelOnly"
        assert cfg.lora_needs_clip is False

    def test_z_image_entry_matches_yaml(self) -> None:
        cfg = ARCH_REGISTRY["z_image"]
        assert cfg.skill_name == "z-image-turbo"
        assert "lumina2" in cfg.clip_type_keywords
        assert "qwen3_4b" in cfg.clip_type_keywords

    def test_adding_arch_yaml_registers_new_arch(self, tmp_path: Path) -> None:
        """Dropping a new ``skills/<slug>/arch.yaml`` MUST register a new arch."""
        skill_dir = tmp_path / "my-new-model"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-new-model\n---\n")
        (skill_dir / "arch.yaml").write_text(
            "registry_name: my_new_model\n"
            "description: My New Model\n"
            "detection:\n"
            "  unet_keywords: ['my_new_model']\n"
            "  node_classes: ['MyNewModelLoader']\n"
            "  clip_type_keywords: []\n"
            "lora:\n"
            "  node: LoraLoaderModelOnly\n"
            "  needs_clip: false\n"
            "  supported: true\n",
            encoding="utf-8",
        )

        reg = load_arch_registry(tmp_path)
        assert "my_new_model" in reg
        cfg = reg["my_new_model"]
        assert cfg.skill_name == "my-new-model"   # defaulted from directory
        assert cfg.unet_keywords == ("my_new_model",)
        assert cfg.node_classes == frozenset({"MyNewModelLoader"})
        assert cfg.lora_supported is True
        assert cfg.lora_node == "LoraLoaderModelOnly"

    def test_malformed_arch_yaml_is_skipped_not_fatal(
        self, tmp_path: Path, recwarn: pytest.WarningsChecker
    ) -> None:
        skill_dir = tmp_path / "broken"
        skill_dir.mkdir()
        (skill_dir / "arch.yaml").write_text(
            "this: is: not: valid: yaml:\n", encoding="utf-8"
        )
        # Must not raise, just warn + skip.
        reg = load_arch_registry(tmp_path)
        assert reg == {}

    def test_missing_registry_name_is_skipped(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "no-registry"
        skill_dir.mkdir()
        (skill_dir / "arch.yaml").write_text(
            "description: Missing registry_name\n", encoding="utf-8"
        )
        reg = load_arch_registry(tmp_path)
        assert reg == {}

    def test_empty_dir_returns_empty_registry(self, tmp_path: Path) -> None:
        assert load_arch_registry(tmp_path) == {}

    def test_nonexistent_dir_returns_empty_registry(
        self, tmp_path: Path
    ) -> None:
        assert load_arch_registry(tmp_path / "does-not-exist") == {}
