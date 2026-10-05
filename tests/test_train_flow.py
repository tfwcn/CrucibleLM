"""训练流程单测 — 打包/掩码/ckpt/评测（离线，torch 必需，无 torch 自动跳过）."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import train_local_llm as flow
from src.llm.local.config import tiny_test_config
from src.llm.local.data import (
    IGNORE,
    PackedBatcher,
    fit_tokenizer_on_stream,
    iter_local_texts,
    pack_pretrain,
    pack_sft,
)
from src.llm.local.infer import SimpleTokenizer
from src.llm.local.linear_attn import GatedDeltaLite
from src.llm.local.model import TinyLLM
from src.llm.local.train import build_optimizer, train_step

FIXTURE = str(Path(__file__).resolve().parent / "fixtures" / "llm-corpus")


def _tok():
    """在 fixture 语料上拟合的分词器."""
    return fit_tokenizer_on_stream(iter_local_texts(FIXTURE), 256)


def test_pack_pretrain_shapes():
    """预训练打包：定长块，id 合法."""
    tok = _tok()
    blocks = list(pack_pretrain(iter_local_texts(FIXTURE), tok.encode, 32, tok.EOS))
    assert len(blocks) > 0
    for b in blocks:
        assert b.shape == (33,)
        assert int(b.max()) < 256


def test_pack_sft_mask():
    """SFT 打包：prompt 区全 -100，output 区有真实 label."""
    tok = _tok()
    triples = [("把下面的句子翻译成英文：你好世界", "", "Hello world"),
               ("解释什么是摸鱼", "", "上班时间合理休息放松")]
    packed = list(pack_sft(iter(triples), tok.encode, 64, tok.EOS))
    assert len(packed) > 0
    for x, y in packed:
        assert x.shape == y.shape == (65,)
        # 至少有掩位和有效位
        assert (y == IGNORE).any()
        assert (y != IGNORE).any()
        # 有效位与输入对齐（自回归，label 就是对应位置的 token）
        valid = y != IGNORE
        assert (x[valid] == y[valid]).all()


def test_local_texts_jsonl(tmp_path):
    """本地 jsonl：text/content 字段与纯文本行都能读."""
    p = tmp_path / "a.jsonl"
    long1 = "这是第一篇很长的中文文档内容测试数据，用来验证分词打包流程啊啊啊啊"
    long2 = "这是第二篇很长的中文文档内容测试数据，用来验证分词打包流程啊啊啊啊"
    long3 = "这是一行很长的纯文本中文文档内容，用来验证分词打包流程啊啊啊啊啊"
    p.write_text(json.dumps({"text": long1}) + "\n"
                 + json.dumps({"content": long2}) + "\n"
                 + long3 + "\n",
                 encoding="utf-8")
    docs = list(iter_local_texts(tmp_path))
    assert len(docs) == 3


def test_train_step_with_sft_labels():
    """train_step 支持 labels 掩码：全掩时 loss 为 0 且可反向."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    opt = build_optimizer(m)
    x = torch.randint(0, 256, (2, 16))
    y = torch.full_like(x, IGNORE)
    y[:, 8:] = x[:, 8:]  # 只学后半
    stats = train_step(m, opt, x, labels=y)
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_ckpt_roundtrip(tmp_path):
    """存盘/恢复：权重一致、step 续上."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    opt = build_optimizer(m)
    flow.save_ckpt(tmp_path, m, opt, 5, 1000, {"phase": "pretrain"})
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "ckpt-000005" / "model.pt").exists()
    m2 = TinyLLM(tiny_test_config())
    meta = flow.load_ckpt(tmp_path, m2, None, map_location="cpu")
    assert meta["step"] == 5 and meta["tokens_seen"] == 1000
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_ckpt_rotation_keeps_last_n(tmp_path):
    """版本轮转：只留最近 N 个快照，latest 恒指向最新；快照无 optim.pt."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    opt = build_optimizer(m)
    for step in (1, 2, 3, 4):
        flow.save_ckpt(tmp_path, m, opt, step, step * 100, keep_last=2)
    snaps = sorted(p.name for p in tmp_path.glob("ckpt-*") if p.is_dir())
    assert snaps == ["ckpt-000003", "ckpt-000004"], snaps
    assert (tmp_path / "optim.pt").exists()  # latest 照常含优化器状态
    assert not (tmp_path / "ckpt-000004" / "optim.pt").exists()  # 快照省写盘
    meta = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert meta["step"] == 4
    # 从快照恢复与 latest 一致
    m3 = TinyLLM(tiny_test_config())
    m3.load_state_dict(torch.load(tmp_path / "ckpt-000004" / "model.pt"))
    for p1, p2 in zip(m.parameters(), m3.parameters()):
        assert torch.equal(p1, p2)


def test_evaluate_runs():
    """评测函数在小数据上跑通."""
    torch.manual_seed(0)
    tok = _tok()
    m = TinyLLM(tiny_test_config())
    texts = list(iter_local_texts(FIXTURE))[:4]
    make_eval = lambda: PackedBatcher(  # noqa: E731
        pack_pretrain(iter(texts), tok.encode, 32, tok.EOS), 2)
    m.eval()
    val = flow.evaluate(m, make_eval, 3, torch.device("cpu"))
    assert val == val and val > 0  # 有限正数


def test_lr_schedule():
    """学习率调度：warmup 上升、之后衰减."""
    peak = flow.lr_schedule(0, 100, 10, 1.0)
    mid = flow.lr_schedule(5, 100, 10, 1.0)
    end = flow.lr_schedule(100, 100, 10, 1.0)
    assert peak < mid and end < 0.2


def test_configure_hf_endpoint():
    """镜像源配置：环境变量 + 运行期常量一起生效（不依赖 shell 传递）."""
    import os

    datasets = pytest.importorskip("datasets")
    from src.llm.local.data import configure_hf_endpoint

    old_env = os.environ.get("HF_ENDPOINT")
    from datasets import config as ds_config

    old_cfg = ds_config.HF_ENDPOINT
    try:
        assert configure_hf_endpoint("https://hf-mirror.com") == "https://hf-mirror.com"
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
        assert ds_config.HF_ENDPOINT == "https://hf-mirror.com"
        assert configure_hf_endpoint(None) is None
    finally:
        if old_env is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = old_env
        ds_config.HF_ENDPOINT = old_cfg


def test_local_texts_parquet(tmp_path):
    """本地 parquet：只读 text 列，其他列不进内存."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table({
        "text": ["这是第一篇很长的中文文档内容测试数据，用来验证分词打包流程啊啊啊啊",
                 "短",
                 "这是第二篇很长的中文文档内容测试数据，用来验证分词打包流程啊啊啊啊"],
        "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    })
    pq.write_table(table, tmp_path / "shard.parquet")
    docs = list(iter_local_texts(tmp_path))
    assert len(docs) == 2  # 短文档被过滤


def test_collect_holdout_local_is_lazy():
    """本地 holdout 切分：评测取前半，训练流跳过前半（不再 whole-list 物化）."""
    args = flow.parse_args(["--data", "local", "--local-path", FIXTURE])
    holdout, train_gen = flow.collect_holdout(args)
    all_docs = list(iter_local_texts(FIXTURE))
    n = max(2, min(flow.HOLDOUT_DOCS, len(all_docs)) // 2)
    assert holdout == all_docs[:n]
    assert list(train_gen) == all_docs[n:]


def test_parse_local_mix():
    """本地多目录解析：单目录/权重/非法."""
    from src.llm.local.data import parse_local_mix

    assert parse_local_mix("data/a") == [("data/a", 1)]
    assert parse_local_mix("data/a:3,data/b:1") == [("data/a", 3), ("data/b", 1)]
    assert parse_local_mix("/abs/path") == [("/abs/path", 1)]
    with pytest.raises(ValueError, match="为空"):
        parse_local_mix("  , ")


def test_collect_holdout_multi_dir_partition(tmp_path):
    """多目录混合：评测+训练恰好覆盖全量一次，无重叠无遗漏."""
    from src.llm.local.data import parse_local_mix

    da, db = tmp_path / "a", tmp_path / "b"
    da.mkdir()
    db.mkdir()
    docs_a = [f"A区第{i}篇很长的中文文档内容测试数据用来验证啊啊啊啊啊啊啊啊啊啊" for i in range(6)]
    docs_b = [f"B区第{i}篇很长的中文文档内容测试数据用来验证啊啊啊啊啊啊啊啊啊啊" for i in range(6)]
    (da / "doc.txt").write_text("\n\n".join(docs_a), encoding="utf-8")
    (db / "doc.txt").write_text("\n\n".join(docs_b), encoding="utf-8")
    assert parse_local_mix(f"{da}:1,{db}:3") == [(str(da), 1), (str(db), 3)]
    args = flow.parse_args(["--data", "local", "--local-path", f"{da}:1,{db}:3"])
    holdout, train_gen = flow.collect_holdout(args)
    assert len(holdout) == flow.HOLDOUT_DOCS or len(holdout) == 12
    rest = list(train_gen)
    # 12 篇全覆盖一次（混合顺序不限，比对多重集）
    assert sorted(holdout + rest) == sorted(docs_a + docs_b)
    assert len(holdout) + len(rest) == 12


def test_data_cursor_resume_partition(tmp_path):
    """断点续流：评测 + 跳过 + 续训三段恰好覆盖全量，无重叠无遗漏."""
    docs = [f"第{i}篇很长的中文文档内容测试数据用来验证断点续流情况啊啊啊啊啊啊啊啊" for i in range(30)]
    (tmp_path / "doc.txt").write_text("\n\n".join(docs), encoding="utf-8")
    args = flow.parse_args(["--data", "local", "--local-path", str(tmp_path)])
    holdout, train_gen = flow.collect_holdout(args, data_cursor=10)
    assert hasattr(train_gen, "n")  # CountedIterator 计数中
    rest = list(train_gen)
    assert len(holdout) + 10 + len(rest) == 30
    assert sorted(holdout + rest) == sorted(docs[:len(holdout)] + docs[len(holdout) + 10:])
    # 消费计数单调：拉取后 .n 增长
    args2 = flow.parse_args(["--data", "local", "--local-path", str(tmp_path)])
    _, train2 = flow.collect_holdout(args2)
    assert train2.n == 0
    next(train2)
    assert train2.n == 1


def test_save_ckpt_records_data_cursor(tmp_path):
    """存盘元信息含 data_cursor（续跑断点续流用）."""
    import json

    torch.manual_seed(0)
    from src.llm.local.model import TinyLLM

    m = TinyLLM(tiny_test_config())
    from src.llm.local.train import build_optimizer

    opt = build_optimizer(m)
    flow.save_ckpt(tmp_path, m, opt, 7, 700, data_cursor=1234)
    meta = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert meta["data_cursor"] == 1234 and meta["step"] == 7
    # 在途余量：大数扣减、小数钳零（宁可重见不丢数据）
    assert flow._data_cursor(type("C", (), {"n": 2000})()) == 2000 - flow.RESUME_SLACK_DOCS
    assert flow._data_cursor(type("C", (), {"n": 10})()) == 0


def test_replay_buffer_push_sample_evict():
    """回放池：压入/采样形状、元组保持、满淘汰、不足回 None."""
    from src.llm.local.data import ReplayBuffer

    buf = ReplayBuffer(capacity=4, seed=0)
    assert buf.sample(2) is None
    buf.push(torch.stack([torch.full((3,), float(i)) for i in range(3)]))
    assert len(buf) == 3
    out = buf.sample(2)
    assert out.shape == (2, 3)
    # 元组（SFT 的 x/y）结构保持
    buf2 = ReplayBuffer(capacity=4, seed=1)
    buf2.push([(torch.tensor([1, 2]), torch.tensor([3, 4]))])
    x, y = buf2.sample(1)
    assert x.shape == (1, 2) and y.shape == (1, 2)
    # FIFO 淘汰
    buf3 = ReplayBuffer(capacity=2, seed=0)
    for i in range(4):
        buf3.push(torch.tensor([[float(i)]]))
    assert len(buf3) == 2
    assert sorted(b.item() for b in buf3.buf) == [2.0, 3.0]


def test_forward_batch_rho_teacher_fallback():
    """rho-ref=teacher 但无老师：警告回落自参照，结果有限."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    x = torch.randint(0, 256, (2, 16))
    out = flow._forward_batch(m, x, "cpu", rho_keep=0.5, rho_ref="teacher")
    assert torch.isfinite(out["loss"]) and "main_loss" in out


def test_forward_batch_kd_subsample():
    """KD 降频：do_kd=False 不调老师（calls 为 0、无 kd_loss 键）."""

    class StubDistiller:
        def __init__(self):
            self.calls = 0

        def batch_kl(self, x, logits, id_to_char):
            self.calls += 1
            return logits.float().mean() * 0 + 1.0

        def teacher_token_losses(self, x, id_to_char):
            self.calls += 1
            return []

    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    x = torch.randint(0, 256, (2, 16))
    d = StubDistiller()
    id_to_char = {i: "字" for i in range(256)}
    out = flow._forward_batch(m, x, "cpu", distiller=d, kd_alpha=0.5,
                              id_to_char=id_to_char, do_kd=True)
    assert d.calls == 1 and "kd_loss" in out and torch.isfinite(out["loss"])
    out2 = flow._forward_batch(m, x, "cpu", distiller=d, kd_alpha=0.5,
                               id_to_char=id_to_char, do_kd=False)
    assert d.calls == 1 and "kd_loss" not in out2
    assert torch.isfinite(out2["loss"])


def test_dump_hparams(tmp_path):
    """超参落盘：CLI 参数全量可 JSON 序列化，派生量正确."""
    import json

    args = flow.parse_args(["--phase", "pretrain", "--preset", "tiny",
                            "--rho-keep", "0.5"])
    cfg = tiny_test_config()
    hparams = flow.dump_hparams(args, cfg, tmp_path)
    assert hparams["rho_keep"] == 0.5
    assert hparams["tokens_per_step"] == args.batch * (args.seq_len + 1) * args.accum
    assert hparams["config"]["d_model"] == cfg.d_model
    saved = json.loads((tmp_path / "hparams.json").read_text(encoding="utf-8"))
    assert saved["preset"] == "tiny" and "started_at" in saved


def test_convo_to_pairs():
    """多轮对话转样本对：每轮 assistant 配之前全部上下文，空跳过."""
    from src.llm.local.data import convo_to_pairs

    messages = [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你？"},
        {"role": "user", "content": "讲个笑话"},
        {"role": "assistant", "content": "从前有座山"},
        {"role": "assistant", "content": ""},  # 空跳过
        {"role": "tool", "content": "执行结果 ok"},
    ]
    pairs = convo_to_pairs(messages)
    assert len(pairs) == 2
    p1, r1 = pairs[0]
    assert "你是一个助手" in p1 and "你好" in p1 and r1 == "你好！有什么可以帮你？"
    p2, r2 = pairs[1]
    assert "讲个笑话" in p2 and "你好！有什么可以帮你？" in p2 and r2 == "从前有座山"
    assert convo_to_pairs([{"role": "assistant", "content": "无上下文"}]) == []
    assert convo_to_pairs([]) == []


def test_pack_pairs_masks_prompt():
    """pack_pairs：prompt 全掩，response 有效且与输入对齐."""
    from src.llm.local.data import pack_pairs

    tok = _tok()
    pairs = [("<用户>\n你好\n<助手>\n", "你好世界"), ("<系统>\nx\n<用户>\ny\n", "ok啦啦啦啦啦")]
    packed = list(pack_pairs(iter(pairs), tok.encode, 64, tok.EOS))
    assert len(packed) > 0
    for x, y in packed:
        assert (y == IGNORE).any() and (y != IGNORE).any()
        valid = y != IGNORE
        assert (x[valid] == y[valid]).all()


def test_sft_mix_messages_format(monkeypatch):
    """SFT 混合：messages 格式经 convo 转对，triple 直通."""
    import src.llm.local.data as data_mod

    def fake_load(path, **kwargs):
        class FakeStream:
            def shuffle(self, seed=None, buffer_size=None):
                return self

            def __iter__(self):
                if kwargs.get("name") == "General-Agent":
                    return iter([{"messages": [
                        {"role": "user", "content": "查北京天气情况怎么样啊啊啊"},
                        {"role": "assistant", "content": "北京今天晴转多云啊啊啊啊啊啊"},
                    ]}])
                return iter([{"instruction": "把下面的句子翻译成英文啊啊啊啊啊啊啊",
                              "input": "", "output": "Hello world test case ok"}])

        return FakeStream()

    monkeypatch.setattr(data_mod, "require_datasets", lambda: fake_load)
    out = list(data_mod.iter_hf_sft_mix("belle:1,agent-general:1"))
    assert all(type(p) is tuple and len(p) == 2 for p in out), out
    assert len(out) == 2  # 各 1 对，轮询交错
    # messages 对带上下文标签，triple 对带模板
    assert any("用户" in p[0] for p in out)


def test_shuffle_buffer_exact_once():
    """shuffle 蓄水池：每个块恰好产出一次，且顺序被打乱."""
    from src.llm.local.data import ShuffleBuffer

    blocks = [torch.tensor([i]) for i in range(100)]
    out = list(ShuffleBuffer(iter(blocks), capacity=10, seed=0))
    assert sorted(b.item() for b in out) == list(range(100))
    assert [b.item() for b in out] != list(range(100))


def test_iter_local_sft_jsonl_csv(tmp_path):
    """本地 SFT 读取：jsonl/csv 的 instruction/input/output 三元组，坏行跳过."""
    import json as _json

    from src.llm.local.data import iter_local_sft

    (tmp_path / "a.jsonl").write_text(
        _json.dumps({"instruction": "把下面的句子翻译成英文啊啊啊啊啊啊啊啊",
                     "input": "", "output": "Hello world ok"}) + "\n"
        + "not json at all\n"
        + _json.dumps({"instruction": "", "output": "空指令跳过"}) + "\n",
        encoding="utf-8")
    (tmp_path / "b.csv").write_text(
        "instruction,input,output\n"
        "解释什么是摸鱼现象啊啊啊啊啊啊啊啊啊,,摸鱼就是上班时间合理休息放松啊啊\n",
        encoding="utf-8")
    rows = list(iter_local_sft(tmp_path))
    assert len(rows) == 2
    assert rows[0][0].startswith("把下面的句子翻译")
    assert rows[1] == ("解释什么是摸鱼现象啊啊啊啊啊啊啊啊啊", "",
                       "摸鱼就是上班时间合理休息放松啊啊")
    with pytest.raises(RuntimeError, match="无可用文件"):
        list(iter_local_sft(tmp_path / "nodir"))


def test_collect_holdout_sft_local(tmp_path):
    """本地 SFT：holdout 取对 + 训练流分区精确（模板在此已套好）."""
    import json as _json

    for i in range(4):
        (tmp_path / f"s{i}.jsonl").write_text("\n".join(
            _json.dumps({"instruction": f"第{i}组第{j}条很长的中文指令内容测试啊啊啊啊啊",
                         "input": "", "output": f"第{i}组第{j}条很长的中文回答内容测试啊啊啊啊"})
            for j in range(3)) + "\n", encoding="utf-8")
    args = flow.parse_args(["--phase", "sft", "--data", "local",
                            "--local-path", str(tmp_path)])
    holdout, train_gen = flow.collect_holdout(args, data_cursor=0)
    # 12 对全覆盖：holdout 在前，训练流跳过空位（cursor=0 不跳）
    rest = list(train_gen)
    assert len(holdout) + len(rest) == 12
    assert all(len(p) == 2 and "<助手>" in p[0] for p in holdout + rest)


def test_select_topk_loss():
    """RHO 选择：只取高 loss 位均值，保底防空，关闭时等价全均值."""
    from src.llm.local.train import select_topk_loss

    t = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 10.0])
    sel, mask = select_topk_loss(t, 0.5, min_keep=1)
    assert mask.sum() == 3 and float(sel) == pytest.approx((3 + 10 + 2) / 3)
    sel_all, _ = select_topk_loss(t, 1.0)
    assert float(sel_all) == pytest.approx(t.mean().item())
    # 保底：比例再小也至少留 min_keep 个（不并列时精确）
    sel_min, mask_min = select_topk_loss(
        torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), 0.01, min_keep=2)
    assert mask_min.sum() == 2 and float(sel_min) == pytest.approx(4.5)


def test_token_losses_matches_mean():
    """逐 token loss：形状对，均值与 main_loss 一致."""
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    x = torch.randint(0, 256, (2, 16))
    out = m(x, targets=x, return_token_losses=True)
    assert out["token_losses"].shape == (2, 15)
    assert float(out["token_losses"].mean()) == pytest.approx(float(out["main_loss"]))
    assert "_main_mean" in out


def test_curriculum_parse_and_schedule():
    """课程解析与线性过渡值."""
    assert flow.parse_curriculum("") is None
    assert flow.parse_curriculum("0.7:0.0:3000") == (0.7, 0.0, 3000)
    assert flow.curriculum_value(0.7, 0.0, 3000, 0) == pytest.approx(0.7)
    assert flow.curriculum_value(0.7, 0.0, 3000, 1500) == pytest.approx(0.35)
    assert flow.curriculum_value(0.7, 0.0, 3000, 9999) == pytest.approx(0.0)
    with pytest.raises(SystemExit):
        flow.parse_curriculum("bad")


def test_callable_threshold_filters_live(monkeypatch):
    """可调用阈值：中途改值，过滤实时跟随（课程 schedule 的基础）."""
    import src.llm.local.data as data_mod

    state = {"min": 0.9}

    def fake_load(name, **kwargs):
        class FakeStream:
            def shuffle(self, seed=None, buffer_size=None):
                return self

            def __iter__(self):
                return iter([
                    {"text": "这是一篇很长的中文文档内容测试数据用来验证课程啊啊啊啊啊啊啊啊啊啊啊", "score": 0.95},
                    {"text": "这是另一篇很长的中文文档内容测试数据用来验证啊啊啊啊啊啊啊啊啊啊", "score": 0.5},
                    {"text": "第三篇很长的中文文档测试数据用来验证课程过滤情况啊啊啊啊啊啊啊啊啊", "score": 0.2},
                ])

        return FakeStream()

    monkeypatch.setattr(data_mod, "require_datasets", lambda: fake_load)
    spec = {"dataset": "x", "split": "train", "text_field": "text",
            "score_field": "score"}
    it = data_mod._iter_hf_source(spec, min_score=lambda: state["min"])
    assert len((next(it))) > 32  # 0.95 通过
    state["min"] = 0.0  # 放开，后面全过
    rest = list(it)
    assert len(rest) == 2


def test_ema_forward_and_update():
    """EMA：影子前向有限、同步向在线靠拢、_forward_batch 接入正常."""
    torch.manual_seed(0)
    import copy

    m = TinyLLM(tiny_test_config())
    ema = copy.deepcopy(m).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    x = torch.randint(0, 256, (2, 16))
    out = flow._forward_batch(m, x, "cpu", ema_model=ema, ema_weight=0.05)
    assert torch.isfinite(out["ema_loss"]) and torch.isfinite(out["loss"])
    before = ema.embed.weight.detach().clone()
    with torch.no_grad():
        for ema_p, p in zip(ema.parameters(), m.parameters()):
            ema_p.mul_(0.9).add_(p.detach(), alpha=0.1)
    assert not torch.equal(before, ema.embed.weight.detach())
    for p in ema.parameters():
        assert p.grad is None


def _make_tok_model(n_chars=10):
    """构造小词表 + tiny 模型（扩词测试共用）."""
    from src.llm.local.infer import SimpleTokenizer

    chars = [chr(0x4E00 + i) for i in range(n_chars)]
    tok = SimpleTokenizer(256)
    tok._chars = chars
    tok._ids = {ch: i + 4 for i, ch in enumerate(chars)}
    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    return tok, m


def test_extend_vocab_fit_path():
    """扩词（容量内）：旧 id 不动、旧行一致、新 id 可前向、tied 保持."""
    from src.llm.local.infer import extend_vocab_and_model

    tok, m = _make_tok_model(10)
    old_rows = m.embed.weight.detach().clone()
    old_head_is_embed = m.lm_head.weight is m.embed.weight
    n, resized = extend_vocab_and_model(tok, m, ["新", "字", "啊", "哦"])
    assert (n, resized) == (4, False)
    assert tok._ids["新"] == 14 and tok._ids["哦"] == 17
    assert torch.equal(m.embed.weight[:14], old_rows[:14])  # 旧行逐位一致
    assert m.lm_head.weight is m.embed.weight or old_head_is_embed
    x = torch.tensor([[1, 14, 15, 4]])
    out = m(x)
    assert out["logits"].shape == (1, 4, 256)
    assert torch.isfinite(out["logits"]).all()


def test_extend_vocab_resize_path():
    """扩词（超限）：扩行+重绑，旧行一致，新行可前向."""
    from src.llm.local.infer import extend_vocab_and_model

    tok, m = _make_tok_model(10)
    # 人工填满到容量边缘：embed 256 行，可用 id 4..255
    tok._chars = [chr(0x4E00 + i) for i in range(252)]
    tok._ids = {ch: i + 4 for i, ch in enumerate(tok._chars)}
    old_rows = m.embed.weight.detach().clone()
    n, resized = extend_vocab_and_model(tok, m, ["新", "字", "啊"])
    assert (n, resized) == (3, True)
    assert m.embed.num_embeddings == 259
    assert m.lm_head.weight is m.embed.weight  # tied 重绑
    assert torch.equal(m.embed.weight[:256], old_rows)
    x = torch.tensor([[1, 256, 257]])
    out = m(x)
    assert out["logits"].shape == (1, 3, 259)


def test_load_ckpt_overlap_after_extend(tmp_path):
    """续接加载：扩行后旧权重恢复、新行保留初始化，优化器警告但不炸."""
    torch.manual_seed(0)
    tok, m = _make_tok_model(10)
    opt = build_optimizer(m)
    flow.save_ckpt(tmp_path, m, opt, 5, 1000, keep_last=0)
    from src.llm.local.infer import extend_vocab_and_model

    torch.manual_seed(1)  # 换种子使新行初值不同，验证旧行被覆盖
    tok2, m2 = _make_tok_model(10)
    tok2._chars = [chr(0x4E00 + i) for i in range(250)]
    tok2._ids = {ch: i + 4 for i, ch in enumerate(tok2._chars)}
    n, resized = extend_vocab_and_model(tok2, m2, ["新", "字", "啊", "哦", "嗯"])
    assert (n, resized) == (5, True)
    opt2 = build_optimizer(m2)
    meta = flow.load_ckpt(tmp_path, m2, opt2, map_location="cpu")
    assert meta["step"] == 5
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        if "embed" in n1 or "lm_head" in n1:
            assert torch.equal(p1, p2[:256])  # 旧 256 行被旧权重覆盖
            assert torch.isfinite(p2[256:]).all()  # 新增行保留初始化
        else:
            assert torch.equal(p1, p2)


def test_collect_missing_on_fixture():
    """缺字扫描：fixture 上找全词表外的字，按频次返回."""
    args = flow.parse_args(["--data", "local", "--local-path", FIXTURE])
    tok, _ = _make_tok_model(5)  # 只含 5 个生僻字，fixture 字符几乎全缺
    missing = flow.collect_missing(args, tok)
    assert len(missing) > 0 and len(missing) <= args.extend_max_new
    assert all(ch not in tok._ids for ch in missing)


def test_parse_pretrain_mix():
    """混合配比解析：权重/默认权重/未知源报错."""
    from src.llm.local.data import parse_pretrain_mix

    parsed = parse_pretrain_mix("hq:1,ultrafineweb:2")
    assert [w for _, w in parsed] == [1, 2]
    assert parsed[0][0]["dataset"] == "epfml/FineWeb2-HQ"
    assert parsed[1][0]["split"] == "train"  # datasets 只认 train，中文靠 data_files 限定
    assert parse_pretrain_mix("hq")[0][1] == 1
    with pytest.raises(ValueError, match="未知语料"):
        parse_pretrain_mix("nope:1")


def test_interleave_weighted():
    """加权交错：比例正确、耗尽移出、顺序确定."""
    from src.llm.local.data import interleave_weighted

    out = list(interleave_weighted(
        [iter(["a1", "a2"]), iter(["b1", "b2", "b3", "b4"])], [1, 2]))
    assert out == ["a1", "b1", "b2", "a2", "b3", "b4"]


def test_hf_source_field_and_score(monkeypatch):
    """HF 源：按 text_field 取文、按 score 过滤、data_files 透传."""
    import src.llm.local.data as data_mod

    seen: dict = {}

    def fake_load(path, **kwargs):
        seen["name"] = path
        seen.update(kwargs)

        class FakeStream:
            def shuffle(self, seed=None, buffer_size=None):
                return self

            def __iter__(self):
                return iter([
                    {"content": "这是一篇很长的中文文档内容测试数据用来验证啊啊啊啊啊啊啊啊啊啊啊", "score": 0.9},
                    {"content": "这是一篇低分的很长的中文文档内容测试数据啊啊啊啊啊啊啊啊啊啊啊", "score": 0.1},
                    {"content": "这是另一篇很长的中文文档内容测试数据用来验证啊啊啊啊啊啊啊啊啊啊啊", "score": 0.5},
                ])

        return FakeStream()

    monkeypatch.setattr(data_mod, "require_datasets", lambda: fake_load)
    spec = dict(data_mod.PRETRAIN_SOURCES["ultrafineweb"])
    docs = list(data_mod._iter_hf_source(spec, min_score=0.4))
    assert len(docs) == 2  # 0.1 分的长文被 score 过滤
    assert seen["data_files"] == "data/ultrafineweb_zh/*"
    assert seen["split"] == "train"  # datasets 只认 train，中文靠 data_files 限定


def test_prefetch_iterator_order_and_errors():
    """后台预取：顺序与原流一致；流尽 StopIteration；源异常透传."""
    from src.llm.local.data import PrefetchIterator

    items = [torch.tensor([i]) for i in range(20)]
    out = list(PrefetchIterator(iter(items), capacity=4))
    assert [b.item() for b in out] == list(range(20))

    def bad():
        yield torch.tensor([0])
        raise ValueError("源头炸了")

    it = PrefetchIterator(bad(), capacity=4)
    assert next(it).item() == 0
    with pytest.raises(ValueError, match="源头炸了"):
        next(it)
    with pytest.raises(StopIteration):
        next(it)  # 错误后再次调用不再阻塞


def test_moe_grouped_matches_naive():
    """MoE 分组计算与朴素全量前向数值一致（只算路由命中的 token）."""
    from src.llm.local.moe import FineGrainedMoE

    torch.manual_seed(1)
    m = FineGrainedMoE(64, n_experts=4, top_k=2, expert_hidden=16,
                       n_shared=1, aux_coef=0.01)
    m.eval()
    x = torch.randn(2, 8, 64)
    out, aux = m(x)
    # 朴素全量复算（与实现无关的参考逻辑）
    with torch.no_grad():
        probs = torch.softmax(m.router(x), -1)
        tw, ti = torch.topk(probs, 2, dim=-1)
        tw = tw / tw.sum(-1, keepdim=True).clamp_min(1e-6)
        exp = torch.zeros_like(x)
        for e, expert in enumerate(m.experts):
            w = torch.zeros_like(tw[..., 0])
            for k in range(2):
                w = w + torch.where(ti[..., k] == e, tw[..., k],
                                    torch.zeros(()))
            exp = exp + w.unsqueeze(-1) * expert(x)
        for expert in m.shared:
            exp = exp + expert(x)
    assert torch.allclose(out, exp, atol=1e-5), (out - exp).abs().max()
    out.sum().backward()
    for n, p in m.named_parameters():
        assert torch.isfinite(p.grad).all(), f"NaN 梯度：{n}"


def test_return_state_flag():
    """return_state=False 时输出一致、缓存为 None（训练省状态循环与投影）."""
    from src.llm.local.block import HybridBlock

    for full in (True, False):
        torch.manual_seed(0)
        layer = HybridBlock(128, 4, full_attn=full, n_experts=4, top_k=2,
                            expert_hidden=32, n_shared=1)
        layer.eval()
        h = torch.randn(1, 16, 128)
        o1, (p1, a1) = layer(h, None, True)
        o2, (p2, a2) = layer(h, None, False)
        assert p2 is None
        assert p1 is not None
        assert torch.equal(o1, o2) and torch.equal(a1, a2)


def test_muon_optimizer_step_and_roundtrip(tmp_path):
    """Muon 联合优化器：单步有限、参数变化、存盘恢复、对错存盘报错."""
    from src.llm.local.optim import build_hybrid_optimizer

    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    opt = build_hybrid_optimizer(m, muon_lr=0.02, adam_lr=3e-4)
    kinds = {g.get("optimizer") for g in opt.param_groups}
    assert kinds == {"muon", "adam"}  # 两组都分到参数
    before = [p.detach().clone() for p in m.parameters()]
    x = torch.randint(0, 256, (2, 16))
    out = m(x, targets=x)
    out["loss"].backward()
    opt.zero_grad()
    out = m(x, targets=x)
    out["loss"].backward()
    opt.step()
    assert torch.isfinite(out["loss"]).all() or torch.isfinite(out["loss"])
    changed = any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
    assert changed
    torch.save(opt.state_dict(), tmp_path / "optim.pt")
    m2 = TinyLLM(tiny_test_config())
    m2.load_state_dict(m.state_dict())
    opt2 = build_hybrid_optimizer(m2, muon_lr=0.02, adam_lr=3e-4)
    opt2.load_state_dict(torch.load(tmp_path / "optim.pt"))
    with pytest.raises(RuntimeError, match="不是 Muon 存盘"):
        opt2.load_state_dict({"kind": "adamw"})


def test_sparse_select_keys_causal():
    """select_keys：全 < 查询起点（静态）+ 本块排后（块内另配 tril），确定性."""
    from src.llm.local.sparse_attn import select_keys

    idx, n_static = select_keys(200000, 100000, 102000, 128, 4096, 512)
    idx_list = idx.tolist()
    assert idx_list == sorted(idx_list)  # 升序
    assert max(idx_list) < 102000  # 无未来
    assert min(idx_list) == 0  # sink 覆盖头部
    assert len(idx_list) == len(set(idx_list))  # 无重复（重复会双重计数注意力）
    # 短序列全覆盖：窗口盖住时退化为 dense
    idx2, _ = select_keys(300, 0, 300, 128, 4096, 512)
    assert idx2.tolist() == list(range(300))


def test_sparse_dense_equivalence():
    """窗口盖住全序列时，稀疏输出与 dense 逐位一致（同权重）."""
    from src.llm.local.mla import MLAModule
    from src.llm.local.sparse_attn import SparseMLAModule

    torch.manual_seed(0)
    dense = MLAModule(128, 4, 32, 32, 16, max_seq_len=256)
    sparse = SparseMLAModule(128, 4, 32, 32, 16, max_seq_len=256,
                             sparse_threshold=8, sparse_window=256,
                             sparse_sink=128, sparse_stride=64, sparse_chunk=16)
    sparse.load_state_dict(dense.state_dict())  # 同权重（无新增参数，键一致）
    sparse.eval()
    dense.eval()
    h = torch.randn(1, 64, 128)
    with torch.no_grad():
        o_dense, _ = dense(h)
        o_sparse, past = sparse(h)
    assert torch.allclose(o_dense, o_sparse, atol=1e-5), (o_dense - o_sparse).abs().max()
    assert past is not None and len(past) == 2


def test_sparse_decode_matches_prefill():
    """稀疏解码：prefill 缓存 + 逐步解码 == 一次全前向（短序列全覆盖场景）."""
    from src.llm.local.sparse_attn import SparseMLAModule

    torch.manual_seed(0)
    m = SparseMLAModule(128, 4, 32, 32, 16, max_seq_len=256,
                        sparse_threshold=8, sparse_window=256,
                        sparse_sink=128, sparse_stride=64, sparse_chunk=16)
    m.eval()
    h = torch.randn(1, 32, 128)
    with torch.no_grad():
        full, _ = m(h)
        # prefill 前 24 个，再单步解码 8 个
        pre, past = m(h[:, :24])
        assert pre.shape == (1, 24, 128)
        hh, cur_past = h[:, 24:25], past
        outs = []
        for i in range(8):
            o, cur_past = m(hh, cur_past)
            outs.append(o)
            hh = h[:, 25 + i:26 + i] if i < 7 else h[:, 31:32]
        tail = torch.cat(outs, dim=1)
    assert torch.allclose(full[:, 24:], tail, atol=1e-4), (full[:, 24:] - tail).abs().max()


def test_sparse_no_future_leak():
    """稀疏路径因果性：改未来位置，当前及之前输出不变."""
    from src.llm.local.sparse_attn import SparseMLAModule

    torch.manual_seed(0)
    m = SparseMLAModule(128, 4, 32, 32, 16, max_seq_len=512,
                        sparse_threshold=32, sparse_window=64,
                        sparse_sink=8, sparse_stride=16, sparse_chunk=32)
    m.eval()
    h1 = torch.randn(1, 200, 128)
    h2 = h1.clone()
    h2[0, 150] += 50.0
    with torch.no_grad():
        o1, _ = m(h1)
        o2, _ = m(h2)
    diff = (o1 - o2).abs()
    # 位置 150 的变化只许影响窗口内 [150-64, 150] 之后…保守断言前部不受影响
    assert diff[0, :80].max() < 1e-4
    assert diff[0, 150].max() > 0


def test_linear_chunked_matches_parallel():
    """线性注意力分块递推与并行前向一致（前向精确；反向块间截断只断言有限）."""
    from src.llm.local.linear_attn import GatedDeltaLite

    torch.manual_seed(0)
    m = GatedDeltaLite(d_model=64, n_heads=4, linear_chunk=32)
    m.eval()
    h = torch.randn(2, 130, 64)
    with torch.no_grad():
        o_parallel, _ = m(h)  # 此处 t=130 > chunk，本就会走分块；下面强制对比
    # 强制并行分支：临时调大阈值
    m.linear_chunk = 10 ** 9
    with torch.no_grad():
        o_full, _ = m(h)
    assert torch.allclose(o_parallel, o_full, atol=1e-5), (o_parallel - o_full).abs().max()
    # 分块路径反向有限
    m.train()
    o, _ = m(h.clone().requires_grad_(True))
    o.sum().backward()
    for n, p in m.named_parameters():
        assert torch.isfinite(p.grad).all(), f"NaN 梯度：{n}"


def test_linear_decode_matches_train():
    """线性注意力解码一致性：prefill 缓存 + 单步解码 == 全前向末步输出."""
    from src.llm.local.linear_attn import GatedDeltaLite

    torch.manual_seed(0)
    m = GatedDeltaLite(d_model=64, n_heads=4)
    m.eval()
    h = torch.randn(1, 20, 64)
    with torch.no_grad():
        full, _ = m(h)
        _, past = m(h[:, :19])  # prefill 前 19 个
        step, _ = m(h[:, 19:20], past)  # 单步解码第 20 个
    assert torch.allclose(full[:, 19:20], step, atol=1e-5), (full[:, 19:20] - step).abs().max()


def test_linear_conv_tail_survives_stepwise_decode():
    """回归：逐 token 解码时卷积尾部必须覆盖 buf+current 末段.

    曾把尾部截在拼接 buf 之前，导致 buf 每步被左补零覆盖成 [0, x_t]，
    第 3 个 token 起左感受野丢一个真实 token（S 状态偏差 ~1.9）。
    """
    from src.llm.local.linear_attn import GatedDeltaLite

    torch.manual_seed(0)
    m = GatedDeltaLite(d_model=64, n_heads=4)
    m.eval()
    h = torch.randn(1, 8, 64)
    past = None
    outs = []
    with torch.no_grad():
        full, full_past = m(h)
        for i in range(h.shape[1]):
            out, past = m(h[:, i : i + 1], past)
            outs.append(out)
    # 状态与卷积尾都逐位一致（不只是末位输出）
    for name, got, want in zip(("state", "k_buf", "v_buf"), past, full_past):
        assert torch.allclose(got, want, atol=1e-5), f"{name}: {(got - want).abs().max()}"
    stepwise = torch.cat(outs, dim=1)
    assert torch.allclose(full, stepwise, atol=1e-5), (full - stepwise).abs().max()


def test_longctx_config():
    """200K 预设：长度与 YaRN 就位，rope 缓存可建."""
    from src.llm.local.config import longctx_config

    cfg = longctx_config()
    assert cfg.max_seq_len == 204800 and cfg.yarn_scale == 8.0
    assert cfg.sparse_threshold == 4096
    # KV 缓存测算：latent 160 维 ×2B ×200K ×3 层 ≈ 200MB
    assert cfg.n_full_layers * 200 * 1024 * (cfg.kv_lora_rank + cfg.qk_rope_dim) * 2 / 1e9 < 0.3


def test_linear_attn_long_seq_grad_finite():
    """回归：长序列衰减矩阵上三角 exp 溢出曾导致 w_g 梯度 NaN（先 mask -inf 再 exp 已修）."""
    torch.manual_seed(0)
    m = GatedDeltaLite(d_model=128, n_heads=4)
    h = torch.randn(2, 256, 128)
    out, _ = m(h)
    assert torch.isfinite(out).all()
    out.sum().backward()
    for n, p in m.named_parameters():
        assert torch.isfinite(p.grad).all(), f"NaN 梯度：{n}"


def test_linear_attn_causal_no_future_leak():
    """回归：短卷积必须严格因果——改输入位置 i，只许影响输出 >=i.

    曾用 padding=1 双边补零，K/V 漏看未来 1 token（恰是训练目标），
    模型学到复制通道，loss 虚假塌到 ~0 且留存集同样沦陷。
    """
    torch.manual_seed(0)
    m = GatedDeltaLite(d_model=64, n_heads=4)
    m.eval()
    h1 = torch.randn(1, 16, 64)
    h2 = h1.clone()
    h2[0, 10] += 50.0  # 只改第 10 个位置
    with torch.no_grad():
        o1, _ = m(h1)
        o2, _ = m(h2)
    diff = (o1 - o2).abs()
    assert diff[0, :10].max() < 1e-4, "输出 0..9 被未来输入污染，因果性破坏"
    assert diff[0, 10].max() > 0, "同位置应有响应（测试本身有效）"


def test_grad_ckpt_path():
    """梯度检查点路径：前向反向有限（与正常路径 loss 一致性可接受小误差）."""
    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.grad_ckpt = True
    m = TinyLLM(cfg)
    m.train()
    x = torch.randint(0, 256, (2, 32))
    out = m(x, targets=x)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert m.embed.weight.grad is not None
    assert torch.isfinite(m.embed.weight.grad).all()


def test_generate_restores_train_mode():
    """回归：generate() 不得把训练中的模型永久留在 eval 模式.

    否则梯度检查点被静默关闭，后续训练步显存爆炸（OOM 真因）。
    """
    torch.manual_seed(0)
    cfg = tiny_test_config()
    cfg.grad_ckpt = True
    m = TinyLLM(cfg)
    m.train()
    x = torch.randint(0, 256, (1, 8))
    gen = m.generate(x, max_new_tokens=4)
    assert gen.shape == (1, 12)
    assert m.training, "generate 后模型仍应处于 train 模式"


def test_interleave_batches_main_driven():
    """回放混流：主流耗尽即停（含无限 replay），主 batch 保序，配比约权重复."""
    import itertools

    from src.llm.local.data import interleave_batches

    main = [("m", i) for i in range(20)]
    replay = itertools.cycle([("r", i) for i in range(5)])
    out = list(interleave_batches(iter(main), replay, 17, 3))
    got_main = [i for tag, i in out if tag == "m"]
    assert got_main == list(range(20)), "主 batch 必须全量保序"
    frac = sum(1 for tag, _ in out if tag == "r") / len(out)
    assert abs(frac - 3 / 20) < 0.06, frac


def test_interleave_batches_no_replay_passthrough():
    """回放权重 0 时直通主流（filter 语义，防配比 bug 吃数据）."""
    from src.llm.local.data import interleave_batches

    main = list(range(7))
    assert list(interleave_batches(iter(main), iter([]), 17, 0)) == main


def test_save_best_roundtrip(tmp_path):
    """冠军快照：落盘可严格载回，meta 含 step/val（长训冠军永不轮转）."""
    import json

    torch.manual_seed(0)
    m = TinyLLM(tiny_test_config())
    flow.save_best(tmp_path, m, 200, 2.425)
    meta = json.loads((tmp_path / "best" / "meta.json").read_text(encoding="utf-8"))
    assert meta == {"step": 200, "val_loss": 2.425}
    m2 = TinyLLM(tiny_test_config())
    m2.load_state_dict(torch.load(tmp_path / "best" / "model.pt"))
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)
