import textwrap

import pytest

from src.utils.config import load_config


@pytest.fixture
def base_yaml(tmp_path):
    p = tmp_path / "base.yaml"
    p.write_text(textwrap.dedent("""
        digra:
          alpha: 0.2
          early_stopping:
            unchanged_rounds_threshold: 2
        datasets:
          nq:
            n_questions: 150
        baselines:
          - cot
          - digra
    """))
    return p


@pytest.fixture
def override_yaml(tmp_path):
    p = tmp_path / "override.yaml"
    p.write_text(textwrap.dedent("""
        digra:
          alpha: 0.5
    """))
    return p


def test_dot_access(base_yaml):
    cfg = load_config(base_yaml)
    assert cfg.digra.alpha == 0.2
    assert cfg.digra.early_stopping.unchanged_rounds_threshold == 2


def test_dict_access(base_yaml):
    cfg = load_config(base_yaml)
    assert cfg["digra"]["alpha"] == 0.2


def test_missing_field_raises(base_yaml):
    cfg = load_config(base_yaml)
    with pytest.raises(AttributeError):
        _ = cfg.digra.nonexistent_field


def test_list_of_scalars_preserved(base_yaml):
    cfg = load_config(base_yaml)
    assert cfg.baselines == ["cot", "digra"]


def test_nested_dataset_access(base_yaml):
    cfg = load_config(base_yaml)
    assert cfg.datasets.nq.n_questions == 150


def test_override_merges_and_overwrites(base_yaml, override_yaml):
    cfg = load_config(base_yaml, overrides=override_yaml)
    # overridden value wins
    assert cfg.digra.alpha == 0.5
    # non-overridden nested value survives the merge untouched
    assert cfg.digra.early_stopping.unchanged_rounds_threshold == 2
    # untouched top-level section survives entirely
    assert cfg.datasets.nq.n_questions == 150


def test_to_dict_returns_plain_dict(base_yaml):
    cfg = load_config(base_yaml)
    plain = cfg.digra.to_dict()
    assert isinstance(plain, dict)
    assert plain["alpha"] == 0.2


def test_full_scale_override_restores_seeds_and_agent_counts():
    # regression test tied to the real project config files (not the tmp
    # fixtures above): confirms configs/full_scale.yaml actually restores
    # the full sweep on top of the reduced-scope configs/base.yaml.
    cfg_base = load_config("configs/base.yaml")
    assert cfg_base.project.seeds == [0]
    assert cfg_base.debate.agent_counts == [3]

    cfg_full = load_config("configs/base.yaml", overrides="configs/full_scale.yaml")
    assert cfg_full.project.seeds == [0, 1, 2, 3]
    assert cfg_full.debate.agent_counts == [3, 5]

    # everything else from base.yaml survives the merge untouched
    assert cfg_full.digra.alpha == cfg_base.digra.alpha
    assert cfg_full.datasets.nq.n_questions == cfg_base.datasets.nq.n_questions
