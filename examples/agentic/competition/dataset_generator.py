"""Generate synthetic diary entries for the competition agentic example."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Diary templates covering different types of days
DIARY_TEMPLATES = [
    {
        "diary": "今天跑通了AReno的GSPO训练，中间遇到CUDA OOM，调了好久batch-size才解决。下午改简历，改了三个多小时总觉得不够好。晚上看了一集动漫放松。",
        "mood": "充实但有点累",
    },
    {
        "diary": "今天主要在写工单系统的后端代码，完成了工单分配和状态流转的逻辑。感觉权限校验那块写得很顺。中午和同学吃了饭，下午回来继续写测试。",
        "mood": "高效",
    },
    {
        "diary": "今天状态不太好，刷了一上午手机，下午才开始看AReno的源码。看了run_agent.py和reward.py的实现，感觉理解了一些但还有很多不懂。晚上打了游戏。",
        "mood": "有点愧疚",
    },
    {
        "diary": "今天面试准备了一天，背了八股文，模拟了几道算法题。感觉自己准备得还可以但表达的时候会紧张。晚上和家人通了电话，心情好了一些。",
        "mood": "焦虑但 hopeful",
    },
    {
        "diary": "今天把AReno的agentic examples全部读了一遍，写了笔记。然后开始设计自己的agent场景，画了流程图。感觉对agentic RL的理解深了很多。下午去跑了步。",
        "mood": "充实",
    },
    {
        "diary": "今天一直在debug一个多线程的数据竞争问题，从早上搞到下午五点才定位到根因。虽然浪费时间但学到很多。晚上吃了个好的犒劳自己。",
        "mood": "疲惫但有成就感",
    },
    {
        "diary": "今天看了很多技术博客，了解了一些AI Infra的前沿方向。感觉自己要学的东西太多了，有点overwhelmed。下午试着写了点代码但心不在焉。",
        "mood": "迷茫",
    },
    {
        "diary": "今天完成了工单系统的前端对接，用AI编码工具快速搭好了页面。虽然前端不太熟但有AI辅助效率很高。晚上继续看AReno的论文。",
        "mood": "有收获",
    },
    {
        "diary": "今天主要在改简历，根据AI Infra实习岗位的JD调整了项目描述。把AReno的训练实践写了进去。下午投了几个实习。晚上有点焦虑睡不着。",
        "mood": "焦虑",
    },
    {
        "diary": "今天跑了AReno的SFT训练，看到loss输出的时候很有成就感。然后读了agentic.py的源码，理解了AgentTrajectory怎么转成训练数据。晚上和朋友聊了会天。",
        "mood": "开心",
    },
    {
        "diary": "今天在Kaggle上配置AReno环境，编译CUDA算子花了很久。中间遇到了好几个报错，都自己查文档解决了。感觉排查问题的能力在提升。",
        "mood": "有耐心",
    },
    {
        "diary": "今天没怎么做正事，上午睡到很晚才起。下午看了两集番剧，然后刷了很久B站。晚上有点后悔浪费了一天。",
        "mood": "懈怠",
    },
    {
        "diary": "今天把竞争式agent的需求文档写完了，仔细想了三明治反馈的reward设计。感觉这个方向很有意思。下午去健身房练了一小时。",
        "mood": "有动力",
    },
    {
        "diary": "今天研究了张量并行和CUDA算子融合的原理，看了AReno的setup.py和accel代码。虽然很多看不太懂但感觉打开了新世界。晚上继续啃论文。",
        "mood": "兴奋但 overwhelmed",
    },
    {
        "diary": "今天和同学讨论了各自的实习准备情况，发现自己进度还行但深度不够。下午回来认真梳理了一下AReno的架构，写了思维导图。",
        "mood": "平和",
    },
    {
        "diary": "今天改简历改了一整天，反复斟酌每一条经历的描述。改了至少七八版，总觉得不够完美。晚上突然想到还可以加上开源贡献的部分。",
        "mood": "纠结",
    },
]


def generate_records(count: int, *, seed: int = 2026) -> list[dict]:
    """Generate deterministic diary records."""
    rng = random.Random(seed)
    records = []
    for idx in range(count):
        template = dict(rng.choice(DIARY_TEMPLATES))
        records.append({
            "id": idx,
            "diary": template["diary"],
            "mood": template["mood"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic diary entries for the AReno competition agentic example."
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--count", type=int, default=64, help="Number of records.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
