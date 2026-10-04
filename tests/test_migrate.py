"""架构迁移单测 — 恒等加深/专家加宽精确性、裁剪形状、SVD 界、CLI 烟雾."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from src.llm.local.config import tiny_test_config
from src.llm.local.migrate import (
    add_identity_layers,
    subselect_layers,
    svd_split,
    widen_experts,
)
from src.llm.local.model import TinyLLM


def _tiny_eval():
    """eval 模式 tiny 模型（精确性对比用，关 dropout 噪声）."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    m.eval()
    return m


def test_add_identity_layers_exact():
    """加恒等层：输出逐位一致（pre-norm + 零输出投影 = 恒等）."""
    m = _tiny_eval()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        before = m(x)["logits"]
    add_identity_layers(m, 2)
    assert len(m.layers) == 6 and m.config.n_layers == 6
    with torch.no_grad():
        after = m(x)["logits"]
    assert torch.equal(before, after)


def test_widen_experts_exact():
    """专家加宽：复制平分后输出逐位一致."""
    m = _tiny_eval()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        before = m(x)["logits"]
    widen_experts(m, 96, seed=0)
    assert m.config.expert_hidden == 96
    with torch.no_grad():
        after = m(x)["logits"]
    # 除法舍入带来 ulp 级误差（数学恒等，浮点近似），用紧公差
    assert torch.allclose(before, after, atol=1e-5), (before - after).abs().max()
    with pytest.raises(ValueError, match="必须大于"):
        widen_experts(m, 64)


def test_subselect_layers_shape():
    """层裁剪：层数收缩，前向可跑."""
    m = _tiny_eval()
    subselect_layers(m, [0, 2])
    assert len(m.layers) == 2 and m.config.n_layers == 2
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = m(x)
    assert out["logits"].shape == (1, 8, 256)


def test_svd_split_bound():
    """SVD：满秩误差 ~0，低秩误差有上界且可复原形状."""
    torch.manual_seed(0)
    w = torch.randn(64, 48)
    a, b, err = svd_split(w, 48)
    assert err < 1e-4
    assert torch.allclose(a @ b, w, atol=1e-3)
    a2, b2, err2 = svd_split(w, 8)
    assert a2.shape == (64, 8) and b2.shape == (8, 48)
    assert 0.0 < err2 < 1.0


def test_migrate_cli_smoke(tmp_path):
    """CLI 烟雾：源 ckpt -> 加深 -> 输出目录可用 + 输出一致."""
    import migrate_model as mig

    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    src = tmp_path / "src"
    src.mkdir()
    torch.save(m.state_dict(), src / "model.pt")
    from dataclasses import asdict

    (src / "hparams.json").write_text(
        json.dumps({"config": asdict(tiny_test_config())}), encoding="utf-8")
    (src / "vocab.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    out = tmp_path / "out"
    assert mig.main(["--src", str(src), "--out", str(out), "--add-layers", "1"]) == 0
    assert (out / "model.pt").exists()
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["n_layers"] == 5
    assert (out / "vocab.json").exists()
    # 新权重可被同配置模型加载
    from src.llm.local.config import SmallLLMConfig

    m2 = TinyLLM(SmallLLMConfig(**cfg))
    m2.load_state_dict(torch.load(out / "model.pt"))
    m2.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        assert torch.equal(m(x)["logits"][:, :, :], m2(x)["logits"][:, :, :])
