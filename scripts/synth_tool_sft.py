"""合成工具调用 SFT 数据 —— 把「读工具表 + 填参数」做成可训练的任务.

## 为什么需要

`data/` 下的 SFT 语料里工具调用样本是 **0 条**（grep `tool_call|shell_exec` 只命中
1 处误匹配），真实轨迹也只有 156 次调用、集中在 news/task/trigger 三组。
模型不会凭空学会调用工具，必须先造监督信号。

造之前先想清楚**学什么**。128M / 48M 激活装不下 50 个工具的完整语义，
但也不该装 —— moyume-agents 的运行时本来就是**渐进式发现**：

    ① 模型只有一份「工具目录」（路径 + 一句说明）
    ② 选中一个 → `toolbox_browse` 拿到它的参数表
    ③ 再 `toolbox_call` 填参数

所以真正要学的是两个可分离的小任务，而不是「背下 50 个工具的调用方式」：

| 任务 | prompt 里有什么 | 要输出什么 | 考的是 |
|---|---|---|---|
| **SELECT** | 目录（5~8 个候选，一行一个） | `{"path":"workspace/fs_manage","arguments":{}}` | 从描述判断该用哪个 |
| **FILL** | 目录 + 目标工具的完整参数表 | `{"path":…,"arguments":{"action":"read","path":"docs/x.md"}}` | 按 schema 填出合法参数 |

两步串起来就是一次完整调用，且**能力在 prompt 里**、模型只负责「读表 + 对齐」，
这正是小模型能学会的形状。

## 防「背答案」

如果每条样本的工具组合、参数值、用户措辞都固定，模型会记住「这串字 → 这段输出」，
换个请求就崩。三处随机化：

1. **候选集**：每次从 50 个工具里抽 4~7 个干扰项（同组优先，区分度更高）；
2. **参数值**：从池子里随机取（`$file` / `$cmd` 等模板变量），请求里的词与参数值
   来自同一份抽样，不能各说各话；
3. **措辞**：每个场景 3~6 条请求变体，从口语到正式都有。

另外 **10% 的样本带 QQ 通道前缀**（`[群: …] [用户: …]`），因为真实通道会带；
不带前缀的样本占多数，否则 Web 通道会被忽略。

## 负样本

候选集里没有能完成请求的工具时，模型应输出 `{"path":"","arguments":{}}`（空 path =
不发调用），而不是硬套一个。负样本请求必须是**任何工具都做不到**的事
（翻译、写诗、算术、闲聊），不能用「候选集里恰好没有」的事 —— 那是候选集噪声，
会让模型学成「看不到就不做」，把泛化能力也一并掐掉。

## 用法

    # 1) 先导真实 schema（用 agents 自己的 venv，见 dump_tool_schemas.py）
    cd moyume-agents && .venv/bin/python ../CrucibleLM/scripts/dump_tool_schemas.py

    # 2) 合成
    python scripts/synth_tool_sft.py \\
        --schemas data/tool_schemas.json \\
        --out data/sft-tool/tool_call.jsonl \\
        --n 6000 --real data/sandbox

产物是 `messages` 格式的 jsonl（`<系统>/<用户>/<助手>` 标签与推理端逐字节一致），
训练时直接 `--phase sft --local-path data/sft-tool`。
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.local.toolcall import (  # noqa: E402
    NO_TOOL,
    PERSONAS,
    render_catalog,
    render_params,
    render_system,
)
from src.llm.local.toolcall import (
    format_call as dump_call,
)

VERSION = 1

# ---------------------------------------------------------------- 值池
FILES = [
    "docs/ARCHITECTURE.md", "src/llm/local/infer.py", "src/llm/local/data.py",
    "scripts/train_local_llm.py", "README.md", "config.yaml",
    "tests/test_subword_vocab.py", "docs/pitfalls/local-llm-training.md",
    "data/llm-ckpt/vocab.json", "pyproject.toml", "src/llm/local/resume.py",
    "src/llm/local/structured.py", "scripts/build_subwords.py",
    "scripts/synth_tool_sft.py", "tests/test_holdout.py", "Makefile",
    "docs/details/subword-vocab.md", "src/tools/registry.py",
    "src/llm/tool.py", "data/tool_schemas.json", "requirements.txt",
    ".gitignore", "src/main.py", "tests/test_train_flow.py",
]
DIRS = ["src/llm/local", "tests", "docs", "scripts", "data/sft-zh",
        "src/tools/toolbox", "docs/pitfalls", "data/llm-ckpt", "src/llm",
        "tests/fixtures"]
CMDS = [
    "pytest tests/ -q", "git status", "ls -la", "wc -l docs/ARCHITECTURE.md",
    "grep -rn tool_calls src/", "python scripts/build_subwords.py --help",
    "git log --oneline -5", "du -sh data/*", "nvidia-smi",
    "pytest tests/test_tool_sft.py -q", "git diff --stat",
    "python -c 'import torch; print(torch.__version__)'",
    "find . -name '*.pyc' -delete", "cat data/llm-ckpt/latest.json",
    "ps aux | grep train_local_llm", "df -h /root/tmp",
    "ruff check src/llm/local", "git stash list",
]
QUERIES = [
    "工具调用 微调", "子智能体 派发", "公会战排名", "AI 资讯 日报",
    "flash attention 实现", "Chinchilla 最优参数", "Python 异步超时",
    "LoRA 与全参微调 区别", "RAG 检索增强生成", "MoE 负载均衡 损失",
    "BNB 量化 4bit 显存", "Docker 多阶段构建", "Postgres 索引优化",
    "Vue 3 响应式原理", "JWT 刷新令牌 方案",
]
URLS = [
    "https://example.com/blog/posts/llm.html",
    "https://github.com/opencode-ai/opencode",
    "https://www.bilibili.com/video/BV1xx411c7mD",
    "https://rss.example.com/ai/feed.xml",
    "https://docs.python.org/3/library/asyncio.html",
    "https://arxiv.org/abs/2401.00001",
    "https://blog.example.com/2026/03/moe.html",
    "https://news.ycombinator.com/item?id=12345",
]
GLOBS = ["*.py", "**/*.md", "test_*.py", "src/**/*.py", "*.json", "*.yaml",
         "docs/**/*.md", "scripts/*.py"]
CODE_PATTERNS = ["def encode", "tool_calls", "subword", "TODO", "class TinyLLM",
                 "raise SystemExit", "async def", "IGNORE", "cuda"]
AGENTS = ["公主连结小助手", "新闻小助手", "情报搜索官", "代码审查官", "文档整理员"]
TASKS = [
    "查询当前公会战D面的作业阵容，返回可用的阵容搭配",
    "抓取今日 AI 资讯并总结三条要点",
    "检查 src/llm/local 下的类型标注是否齐全",
    "把 docs/pitfalls 里的条目按主题归类",
    "跑一遍全量测试并汇总失败用例",
]
KEYS = ["项目偏好", "常用命令", "部署地址", "用户称呼", "训练环境",
        "代码风格", "数据库连接", "发布流程"]
MEMORY_TEXT = [
    "用户喜欢简洁的回答，不要长篇大论",
    "训练脚本固定用 /root/.venvs/llm-train/bin/python",
    "测试必须先设 TMPDIR=/root/tmp，否则 tmpfs 满盘",
    "代码注释一律用中文，命名保持英文",
    "服务端口 5212，前端代理走 /server-api",
]
TRIGGER_NAMES = ["每日资讯推送", "周末提醒写周报", "新人入群欢迎",
                 "每天早上播报天气", "每周汇总训练曲线"]
CRONS = ["0 8 * * *", "0 9 * * 1", "*/30 * * * *", "0 22 * * *"]
BLOGS = [
    ("Meta 用人工呼叫中心伪装 AI Agent", "Meta 的 AI 助手背后有真人…", "AI,大模型"),
    ("OpenAI 向顶尖数学家求助", "推理能力瓶颈的另一种解法…", "AI,OpenAI"),
    ("a16z 创办 AI 学院", "投资机构开始做教育…", "AI,行业"),
]
GROUPS = ["软乎乎的喵呜小窝", "摸鱼技术交流群", "公主连结公会群", "AI 资讯群"]
USERS = ["喵之追忆", "打工人甲", "群友乙", "夜猫子", "小透明"]

#: 负样本请求：必须是「任何工具都做不到」的事（见模块说明）
NEGATIVE_REQUESTS = [
    "帮我把这句话翻译成法语：今天天气不错",
    "写一首关于秋天落叶的七言绝句",
    "1234 乘以 5678 等于多少",
    "给我讲个冷笑话吧",
    "你觉得人生的意义是什么",
    "随便陪我聊聊天，我有点无聊",
    "把「摸鱼」两个字倒过来写",
]

# ---------------------------------------------------------------- 场景库
# 每条：tool=目标工具路径，req=请求变体（$var 引用 vars），args=参数模板。
# args 里出现的变量必须在 vars 里，否则渲染会抛（宁可报错也不静默错样本）。
SCENARIOS: list[dict] = [
    # ---- workspace：文件读写 ----
    {"tool": "workspace/fs_manage",
     "req": ["看一下 $file 里写了什么", "把 $file 的内容读出来",
             "$file 有多少行？先读一下", "帮我打开 $file 看看"],
     "vars": {"file": FILES},
     "args": {"action": "read", "path": "$file"}},
    {"tool": "workspace/fs_manage",
     "req": ["$dir 目录下有哪些文件", "列一下 $dir", "看看 $dir 里都有啥"],
     "vars": {"dir": DIRS},
     "args": {"action": "list", "path": "$dir"}},
    {"tool": "workspace/fs_manage",
     "req": ["把 $text 写进 $file", "新建 $file，内容写 $text"],
     "vars": {"file": ["docs/notes.md", "TODO.md", "src/llm/local/scratch.py"],
              "text": ["待办：先跑通冒烟测试", "记录：子词表 2.20x 压缩"]},
     "args": {"action": "write", "path": "$file", "content": "$text"}},
    {"tool": "workspace/shell_exec",
     "req": ["跑一下 $cmd", "执行 $cmd", "在终端里跑 $cmd 看结果",
             "$cmd 的结果是什么"],
     "vars": {"cmd": CMDS},
     "args": {"command": "$cmd"}},
    {"tool": "workspace/glob_files",
     "req": ["找一下所有 $glob 文件", "项目里有哪些 $glob",
             "按 $glob 匹配列出文件"],
     "vars": {"glob": GLOBS},
     "args": {"pattern": "$glob"}},
    {"tool": "workspace/grep_code",
     "req": ["搜一下代码里哪里用到了 $pat", "$pat 出现在哪些文件里",
             "grep 一下 $pat"],
     "vars": {"pat": CODE_PATTERNS},
     "args": {"pattern": "$pat"}},
    {"tool": "workspace/edit_file",
     "req": ["把 $file 里的 $old 改成 $new", "$file 中 $old 替换为 $new",
             "这行写错了，$old 应该是 $new"],
     "vars": {"file": FILES,
              "old": ["tool_calls\": None", "max_steps 10000", "lr=3e-4"],
              "new": ["tool_calls\": parsed", "max_steps 14000", "lr=2e-4"]},
     "args": {"path": "$file", "old_string": "$old", "new_string": "$new"}},
    {"tool": "workspace/apply_patch",
     "req": ["把这个补丁打上去：$patch", "应用这段 diff 到项目",
             "帮我 patch 一下，改动如下 $patch"],
     "vars": {"patch": ["--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-旧标题\n+新标题"]},
     "args": {"patch": "$patch"}},
    {"tool": "workspace/undo_changes",
     "req": ["刚才那个改动撤销掉", "回退一步", "改错了，撤回上一次修改"],
     "vars": {},
     "args": {}},
    {"tool": "workspace/skill_list",
     "req": ["工作区里装了哪些技能", "列一下技能", "有哪些技能可以用"],
     "vars": {},
     "args": {}},

    # ---- web：联网 ----
    {"tool": "web/web_search",
     "req": ["搜一下「$q」", "帮我查查「$q」", "「$q」是什么，联网查一下",
             "搜索「$q」的最新资料"],
     "vars": {"q": QUERIES},
     "args": {"query": "$q"}},
    {"tool": "web/web_fetch",
     "req": ["读一下 $url 这个网页", "把 $url 的正文抓下来",
             "$url 讲了什么"],
     "vars": {"url": URLS},
     "args": {"url": "$url"}},
    {"tool": "web/github_search",
     "req": ["在 GitHub 上搜「$q」的仓库", "找找「$q」相关的开源项目",
             "GitHub 上有没有「$q」"],
     "vars": {"q": ["agent framework", "llm training", "vim config"]},
     "args": {"query": "$q", "target": "repositories"}},
    {"tool": "web/bilibili_search",
     "req": ["B 站搜一下「$q」", "找几个「$q」的视频", "B站有没有「$q」的教程"],
     "vars": {"q": ["PyTorch 教程", "公主连结 攻略", "摸鱼 日常"]},
     "args": {"keyword": "$q"}},
    {"tool": "web/rss_fetch",
     "req": ["拉一下这个 RSS：$url", "订阅源 $url 有什么新内容",
             "$url 的最新条目"],
     "vars": {"url": ["https://rss.example.com/ai/feed.xml",
                      "https://blog.example.com/atom.xml"]},
     "args": {"url": "$url"}},

    # ---- task / system ----
    {"tool": "task/taskplan_write",
     "req": ["帮我列个计划：$goal", "把任务拆成几步：$goal",
             "为「$goal」写个 todo"],
     "vars": {"goal": ["重构分词器并补测试", "跑通工具调用 SFT 冒烟",
                       "整理 pitfalls 文档"]},
     "args": {"steps": '[{"step": "$goal", "status": "in_progress"}]'}},
    {"tool": "task/taskplan_get",
     "req": ["现在任务进度到哪了", "看一下 todo", "计划还剩几步"],
     "vars": {},
     "args": {}},
    {"tool": "task/ask_user",
     "req": ["这一步要删数据，先问我确认", "不确定要不要继续，问一下我",
             "让我确认一下再动手"],
     "vars": {},
     "args": {"questions": [{"question": "确认要删除 data/llm-ckpt 下的旧快照吗？",
                             "options": ["确认删除", "先备份"], "type": "single"}]}},
    {"tool": "system/context_summary",
     "req": ["现在上下文占了多少", "看看对话上下文统计", "上下文多长了"],
     "vars": {},
     "args": {}},
    {"tool": "system/context_get_recent",
     "req": ["把最近几条消息翻出来看看", "刚才聊的什么来着",
             "看下最近 5 条上下文"],
     "vars": {},
     "args": {"count": 5}},
    {"tool": "task/pending_list",
     "req": ["现在有挂起待确认的事吗", "有没有待确认清单",
             "看看 pending 列表"],
     "vars": {},
     "args": {}},

    # ---- news ----
    {"tool": "news/news_fetch",
     "req": ["拉一下最新的 AI 资讯", "看看今天有什么 AI 新闻",
             "采集最新资讯"],
     "vars": {},
     "args": {}},
    {"tool": "news/news_detail",
     "req": ["看看这条资讯的详情 $url", "$url 这篇讲了什么",
             "展开 $url 的正文"],
     "vars": {"url": URLS},
     "args": {"url": "$url"}},
    {"tool": "news/news_publish",
     "req": ["把这篇发布成博客：$title", "写好了，发到论坛：$title",
             "发布博客「$title」"],
     "vars": {"title": [b[0] for b in BLOGS],
              "tags": [b[2] for b in BLOGS]},
     "args": {"title": "$title", "content": "正文内容（Markdown）",
              "summary": "一句话摘要", "tags": "$tags"}},
    {"tool": "news/news_recent",
     "req": ["最近几天发过哪些资讯", "查一下已发布记录去重",
             "这两天发过什么"],
     "vars": {},
     "args": {"days": 2}},

    # ---- trigger ----
    {"tool": "trigger/trigger_create",
     "req": ["每天早上 8 点提醒我 $name", "建个定时任务：$name",
             "帮我创建一个触发器，每天 8 点跑 $name"],
     "vars": {"name": TRIGGER_NAMES},
     "args": {"name": "$name", "trigger_type": "cron",
              "cron_expr": "0 8 * * *", "action_message": "该 $name 啦"}},
    {"tool": "trigger/trigger_list",
     "req": ["我有哪些定时任务", "列一下触发器", "看看现在挂了哪些定时"],
     "vars": {},
     "args": {}},
    {"tool": "trigger/trigger_delete",
     "req": ["把 $name 这个定时删了", "取消 $name 的任务",
             "不要 $name 了，删掉触发器"],
     "vars": {"name": TRIGGER_NAMES,
              "tid": ["a6e1333f567e", "9a4222603852"]},
     "args": {"trigger_id": "$tid"}},
    {"tool": "trigger/trigger_send_message",
     "req": ["给 $group 群发条消息：$msg", "通知一下 $group 群，$msg"],
     "vars": {"group": GROUPS, "msg": ["今晚八点开会", "公会战开始了"]},
     "args": {"target_type": "group", "target_id": "177128821",
              "message": "$msg"}},
    {"tool": "trigger/trigger_run",
     "req": ["手动跑一次 $name", "马上触发 $name 看看效果",
             "不用等定时，现在就执行 $name"],
     "vars": {"name": TRIGGER_NAMES, "tid": ["a6e1333f567e", "9a4222603852"]},
     "args": {"trigger_id": "$tid"}},

    # ---- memory / experience ----
    {"tool": "memory/memory_save",
     "req": ["记住：$fact", "帮我记一下 $fact", "存个记忆，$fact"],
     "vars": {"fact": MEMORY_TEXT, "key": KEYS},
     "args": {"key": "$key", "content": "$fact"}},
    {"tool": "memory/memory_load",
     "req": ["我之前说的 $key 是什么来着", "读一下 $key 那条记忆",
             "回忆一下 $key"],
     "vars": {"key": KEYS},
     "args": {"key": "$key"}},
    {"tool": "memory/memory_search",
     "req": ["搜一下关于$key的记忆", "记忆里有没有$key相关的",
             "找找$key的记录"],
     "vars": {"key": KEYS},
     "args": {"query": "$key"}},
    {"tool": "memory/memory_list",
     "req": ["我存了哪些记忆", "列一下所有记忆", "记忆里都有什么"],
     "vars": {},
     "args": {}},
    {"tool": "memory/memory_delete",
     "req": ["把 $key 那条记忆删了", "忘掉 $key", "不需要记住 $key 了"],
     "vars": {"key": KEYS},
     "args": {"key": "$key"}},
    {"tool": "experience/experience_get",
     "req": ["看看这条经验的全文：$eid", "展开 $eid 的具体步骤",
             "$eid 这条经验怎么做的"],
     "vars": {"eid": ["exp_a1b2c3", "exp_d4e5f6"]},
     "args": {"exp_id": "$eid"}},
    {"tool": "experience/experience_search",
     "req": ["之前遇到$q是怎么解决的", "有没有$q的经验",
             "查查经验库里「$q」的做法"],
     "vars": {"q": ["显存不足", "训练 loss 突增", "tmpfs 满盘"]},
     "args": {"query": "$q"}},
    {"tool": "experience/experience_save",
     "req": ["把这个解决方法存成经验：$q", "记录一下$q的处理方法",
             "这个问题解决了，存进经验库"],
     "vars": {"q": ["CUDA OOM 先降 batch", "loss 突增先查 lr 和动量"]},
     "args": {"problem": "$q", "method_steps": ["复现", "定位", "验证"],
              "tools_used": ["shell_exec"]}},

    # ---- agent ----
    {"tool": "agent/subagent_run",
     "req": ["派 $agent 去做：$task", "让 $agent 处理这件事",
             "叫 $agent 来干这个活"],
     "vars": {"agent": AGENTS, "task": TASKS},
     "args": {"agent": "$agent", "task": "$task"}},
    {"tool": "agent/get_subagent_output",
     "req": ["把子智能体的输出翻到下一页", "看一下 sub_ef455126 的后续内容",
             "子任务结果还有吗，继续翻"],
     "vars": {},
     "args": {"subtask_id": "sub_ef455126", "offset": 0, "limit": 50}},

    # ---- pcr ----
    {"tool": "pcr/pcr_homework",
     "req": ["给我 D 面作业", "查一下公会战作业", "这期公会战用什么阵容"],
     "vars": {},
     "args": {"stage": "D", "limit": 5}},
    {"tool": "pcr/pcr_ranking",
     "req": ["查一下公会战排名", "我们公会排第几", "看看当前排名"],
     "vars": {},
     "args": {}},
    {"tool": "pcr/pcr_announcements",
     "req": ["看看公主连结最新公告", "游戏有什么新活动",
             "翻一下官方动态"],
     "vars": {},
     "args": {"page_num": 1, "page_size": 10}},

    # ---- 技能 ----
    {"tool": "workspace/skill_validate",
     "req": ["校验一下 $skill 这个技能目录", "看看 $skill 结构对不对",
             "$skill 能正常加载吗"],
     "vars": {"skill": ["skills/ai-news-collector", "skills/web-fetch",
                        ".agents/.system-skills/bilibili_search"]},
     "args": {"path": "$skill"}},
    {"tool": "workspace/skill_init",
     "req": ["帮我新建一个技能叫 $skill", "初始化技能目录 $skill",
             "创建一个 $skill 技能模板"],
     "vars": {"skill": ["weekly-report", "log-analyzer", "doc-summarizer"]},
     "args": {"skill_name": "$skill"}},

    # ---- toolbox 自指 ----
    {"tool": "toolbox_search",
     "req": ["有没有能搜 B 站视频的工具", "找一下发邮件的工具",
             "哪个工具可以查天气"],
     "vars": {},
     "args": {"query": "B 站 视频 搜索"}},
    {"tool": "toolbox_browse",
     "req": ["看看工具箱里都有什么", "列一下工具分组", "工具目录在哪"],
     "vars": {},
     "args": {}},
]


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数."""
    p = argparse.ArgumentParser(description="合成工具调用 SFT 数据")
    p.add_argument("--schemas", default="data/tool_schemas.json",
                   help="dump_tool_schemas.py 的产物")
    p.add_argument("--out", default="data/sft-tool/tool_call.jsonl", help="输出 jsonl")
    p.add_argument("--n", type=int, default=6000, help="正样本条数（不含真实轨迹）")
    p.add_argument("--fill-ratio", type=float, default=0.6,
                   help="FILL（填参数）占比，其余是 SELECT（选工具）")
    p.add_argument("--neg-ratio", type=float, default=0.12, help="负样本占比")
    p.add_argument("--real", default="", help="moyume-agents sandbox 根（有则掺真实轨迹）")
    p.add_argument("--real-repeat", type=int, default=3, help="每条真实轨迹重复几遍（措辞不变）")
    p.add_argument("--min-distract", type=int, default=3, help="最少干扰工具数")
    p.add_argument("--max-distract", type=int, default=6, help="最多干扰工具数")
    p.add_argument("--qq-prefix-ratio", type=float, default=0.1,
                   help="带 [群:…][用户:…] 前缀的样本占比（真实通道会带）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--report", type=int, default=6, help="打印几条样例")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 参数
def sample_args(tool: dict, template: dict, values: dict) -> dict:
    """按模板渲染参数（`$var` 引用 values）；变量缺失直接抛，不静默出错样本."""
    out = {}
    for key, val in template.items():
        if isinstance(val, str):
            out[key] = Template(val).substitute(values)
        else:
            out[key] = val
    return out


def validate_args(tool: dict, args: dict) -> None:
    """校验参数是否满足 schema 的必填与类型（合成期自检，防止造出坏样本）."""
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    required = list(params.get("required") or [])
    for key in required:
        if key not in args:
            raise ValueError(f"{tool['path']} 缺必填参数 {key}：{list(args)}")
    type_map = {"string": str, "integer": int, "boolean": bool,
                "array": list, "object": dict}
    for key, val in args.items():
        if key not in props:
            raise ValueError(f"{tool['path']} 出现未声明参数 {key}")
        want = (props[key] or {}).get("type")
        py = type_map.get(want)
        if py is None:
            continue
        # bool 是 int 的子类，必须单独挡掉
        if py is int and isinstance(val, bool):
            raise ValueError(f"{tool['path']}.{key} 应为 integer，实际 bool")
        if not isinstance(val, py):
            raise ValueError(f"{tool['path']}.{key} 应为 {want}，"
                             f"实际 {type(val).__name__}")


# ---------------------------------------------------------------- 组样本
def pick_distractors(tools: list[dict], target: dict, rng: random.Random,
                     lo: int, hi: int) -> list[dict]:
    """抽干扰工具：同组优先（区分度最高），不够再全局补.

    目标工具一定在返回列表里（位置随机），避免模型学到「答案总在第一个」。
    """
    same = [t for t in tools if t["group"] == target["group"] and t["path"] != target["path"]]
    other = [t for t in tools if t["group"] != target["group"]]
    k = rng.randint(lo, hi)
    want_same = min(len(same), max(1, k // 2))
    picked = rng.sample(same, want_same) if want_same else []
    rest = k - len(picked)
    if rest > 0:
        picked += rng.sample(other, min(rest, len(other)))
    picked.append(target)
    rng.shuffle(picked)
    return picked


def qq_prefix(rng: random.Random) -> str:
    """QQ 通道前缀（真实消息会长这样，抽查一部分样本带上以增加鲁棒性）."""
    return f"[群: {rng.choice(GROUPS)}] [用户: {rng.choice(USERS)}]  "


def build_select(tool: dict, request: str, tools: list[dict],
                 rng: random.Random, lo: int, hi: int) -> dict:
    """SELECT 样本：只选工具，不填参数（参数表由运行时再 browse 给）."""
    candidates = pick_distractors(tools, tool, rng, lo, hi)
    system = render_system(rng.choice(PERSONAS), render_catalog(candidates))
    response = dump_call(tool["path"], {})
    return _to_sample(system, request, response)


def build_fill(tool: dict, request: str, args: dict, tools: list[dict],
               rng: random.Random, lo: int, hi: int) -> dict:
    """FILL 样本：目录 + 目标工具参数表 → 填出合法参数."""
    candidates = pick_distractors(tools, tool, rng, lo, hi)
    system = render_system(rng.choice(PERSONAS),
                           render_catalog(candidates, exclude=tool["path"]),
                           render_params(tool))
    response = dump_call(tool["path"], args)
    return _to_sample(system, request, response)


def build_negative(request: str, tools: list[dict],
                   rng: random.Random, lo: int, hi: int) -> dict:
    """负样本：列表里的事都做不到 → 空 path（不发调用）."""
    k = rng.randint(lo, hi + 2)
    candidates = rng.sample(tools, min(k, len(tools)))
    system = render_system(rng.choice(PERSONAS), render_catalog(candidates))
    return _to_sample(system, request, dump_call(NO_TOOL, {}))


def _to_sample(system: str, user: str, response: str) -> dict:
    """统一成 messages 格式（与推理端 `_prepare_ids` 逐字节一致）."""
    return {"version": VERSION, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]}


# ---------------------------------------------------------------- 真实轨迹
def iter_real_calls(base: Path) -> list[tuple[str, str, dict]]:
    """从 sandbox 会话库抽 (用户请求, 工具路径, 参数)，只留合法叶子路径.

    真实轨迹的价值在**措辞与参数值的真实感**，不在数量（156 次）；所以它只作
    补充，主体仍是合成场景。
    """
    out: list[tuple[str, str, dict]] = []
    for db in sorted(base.rglob("conversation.db")):
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except sqlite3.Error:
            continue
        try:
            rows = conn.execute(
                "SELECT role, content, tool_calls FROM messages ORDER BY rowid"
            ).fetchall()
        except sqlite3.Error:
            conn.close()
            continue
        conn.close()
        last_user = ""
        used = True          # 当前 user 是否已采过调用（每个请求只取第一次）
        for role, content, tool_calls in rows:
            if role == "user" and content:
                last_user = content.strip()
                used = False
                continue
            if role != "assistant" or not tool_calls or not last_user or used:
                continue
            try:
                calls = json.loads(tool_calls)
            except (TypeError, ValueError):
                continue
            if not isinstance(calls, list) or not calls:
                continue
            fn = (calls[0] or {}).get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError):
                continue
            path = args.get("path") if isinstance(args, dict) else None
            inner = args.get("arguments") if isinstance(args, dict) else None
            # 真实轨迹是 toolbox_call 包一层：剥出被调用的工具与它的参数
            if fn.get("name") != "toolbox_call" or not isinstance(path, str):
                continue
            if not isinstance(inner, dict):
                continue
            out.append((last_user, path, inner))
            used = True
    return out


# ---------------------------------------------------------------- 主流程
def load_schemas(path: Path) -> list[dict]:
    """读 schema 表并做最低限度的字段校验."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tools = data.get("tools") if isinstance(data, dict) else None
    if not isinstance(tools, list) or not tools:
        raise SystemExit(f"{path} 没有 tools 列表（先跑 dump_tool_schemas.py）")
    for t in tools:
        for key in ("path", "description", "parameters"):
            if key not in t:
                raise SystemExit(f"{path} 的工具缺字段 {key}：{t}")
    return tools


def scenario_tools(scenarios: list[dict], schema_paths: set[str]) -> list[dict]:
    """场景引用的工具必须在 schema 里（防止手写路径写错却静默跳过）."""
    missing = sorted({s["tool"] for s in scenarios} - schema_paths)
    if missing:
        raise SystemExit(f"场景引用了 schema 里不存在的工具：{missing}")
    return scenarios


def check_scenarios(scenarios: list[dict], by_path: dict[str, dict]) -> None:
    """静态自检：变量出处 + 参数名/类型，全部在生成前一次查完.

    随机抽样只能"撞到"缺失变量，撞不到就一直静默；这里一次性全扫，
    新增场景写错立刻报错，而不是等跑到那一轮才炸。
    """
    import re

    # 必须与 string.Template 的 idpattern 一致（ASCII 标识符）：
    # 用 \w 会把中文也算进变量名，`$key的记忆` 会被误读成 `key的记忆`
    pattern = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
    problems: list[str] = []
    for sc in scenarios:
        declared = set(sc.get("vars") or {})
        used: set[str] = set()
        for text in sc.get("req") or []:
            used |= set(pattern.findall(text))
        args = dict(sc.get("args") or {})
        for val in args.values():
            if isinstance(val, str):
                used |= set(pattern.findall(val))
        if used - declared:
            problems.append(f"{sc['tool']}：引用了未声明的变量 {sorted(used - declared)}")
        if declared - used:
            problems.append(f"{sc['tool']}：声明了没用到的变量 {sorted(declared - used)}")

        tool = by_path[sc["tool"]]
        props = (tool.get("parameters") or {}).get("properties") or {}
        required = set((tool.get("parameters") or {}).get("required") or [])
        for key, val in args.items():
            if key not in props:
                problems.append(f"{sc['tool']}：参数 {key} 不在 schema 里")
                continue
            want = (props[key] or {}).get("type")
            if isinstance(val, str):
                # 模板值最终渲染成字符串，只允许填 string 型参数
                if want != "string":
                    problems.append(f"{sc['tool']}.{key} 是 {want}，不能用 $模板")
            else:
                py = {"integer": int, "boolean": bool, "array": list,
                      "object": dict, "string": str}.get(want)
                if py and not isinstance(val, py):
                    problems.append(f"{sc['tool']}.{key} 应为 {want}，"
                                    f"实际 {type(val).__name__}")
        missing = required - set(args)
        if missing:
            problems.append(f"{sc['tool']}：缺必填参数 {sorted(missing)}")
    if problems:
        raise SystemExit("场景定义有问题：\n  " + "\n  ".join(problems))


def generate(args) -> tuple[list[dict], Counter]:
    """生成全部样本（返回样本与统计）."""
    rng = random.Random(args.seed)
    tools = load_schemas(Path(args.schemas))
    by_path = {t["path"]: t for t in tools}
    scenario_tools(SCENARIOS, set(by_path))
    check_scenarios(SCENARIOS, by_path)

    stats: Counter = Counter()
    samples: list[dict] = []

    n_neg = int(args.n * args.neg_ratio)
    n_pos = args.n - n_neg

    for i in range(n_pos):
        sc = SCENARIOS[i % len(SCENARIOS)]
        tool = by_path[sc["tool"]]
        # 变量抽样：同一份 values 同时喂给请求与参数，保证"说的"和"填的"一致
        values = {k: rng.choice(v) for k, v in (sc.get("vars") or {}).items()}
        request = Template(rng.choice(sc["req"])).substitute(values)
        if rng.random() < args.qq_prefix_ratio:
            request = qq_prefix(rng) + request

        if rng.random() < args.fill_ratio:
            try:
                call_args = sample_args(tool, sc.get("args") or {}, values)
                validate_args(tool, call_args)
            except (KeyError, ValueError) as e:
                raise SystemExit(f"场景 {sc['tool']} 生成了坏样本：{e}") from e
            samples.append(build_fill(tool, request, call_args, tools, rng,
                                      args.min_distract, args.max_distract))
            stats["fill:" + sc["tool"]] += 1
        else:
            samples.append(build_select(tool, request, tools, rng,
                                        args.min_distract, args.max_distract))
            stats["select:" + sc["tool"]] += 1

    for _ in range(n_neg):
        samples.append(build_negative(rng.choice(NEGATIVE_REQUESTS), tools, rng,
                                      args.min_distract, args.max_distract))
        stats["negative"] += 1

    if args.real:
        real = iter_real_calls(Path(args.real))
        kept = 0
        for request, path, inner in real:
            tool = by_path.get(path)
            if tool is None:
                stats["real:未知工具"] += 1
                continue
            try:
                validate_args(tool, inner)
            except ValueError:
                stats["real:参数不合 schema"] += 1
                continue
            for _ in range(max(1, args.real_repeat)):
                samples.append(build_fill(tool, request, inner, tools, rng,
                                          args.min_distract, args.max_distract))
                kept += 1
        stats["real"] = kept
        print(f"真实轨迹：抽到 {len(real)} 次调用，采用 {kept} 条"
              f"（未知工具 {stats['real:未知工具']}，参数不合 schema "
              f"{stats['real:参数不合 schema']}）")

    rng.shuffle(samples)
    return samples, stats


def preview(samples: list[dict], count: int) -> None:
    """打印样例，便于人眼检查 prompt 是否合理."""
    for s in samples[:count]:
        msgs = s["messages"]
        print("=" * 70)
        print("[系统]\n" + msgs[0]["content"])
        print("[用户] " + msgs[1]["content"])
        print("[助手] " + msgs[2]["content"])


def main(argv=None) -> int:
    """主流程：生成 → 自检 → 落盘 → 打报告."""
    args = parse_args(argv)
    samples, stats = generate(args)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"已写出 {len(samples)} 条 → {path}")
    kinds = Counter(k.split(":")[0] for k in stats.elements())
    print("构成：" + "，".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    chars = [len(json.dumps(s["messages"], ensure_ascii=False)) for s in samples]
    chars.sort()
    if chars:
        print(f"单条长度（字符）：中位 {chars[len(chars) // 2]}，"
              f"P90 {chars[int(len(chars) * 0.9)]}，最长 {chars[-1]}")
    if args.report:
        preview(samples, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
