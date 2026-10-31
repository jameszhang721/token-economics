"""观察文本进入 decoder-only LLM 的全过程。

先只看分词（只需 tokenizer 文件）：
    python scripts/inspect_token_pipeline.py --stage tokenizer

再查看 embedding、Transformer 输出和下一个 token（需下载模型）：
    python scripts/inspect_token_pipeline.py --stage model
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hf_config import report as hf_report, setup_hf_endpoint  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_MAX_TOKENS = 256


def resolve_model_source(model_name, revision=None, local_only=False):
    """让 tokenizer、config、权重从同一个不可变快照读取。"""
    if os.path.isdir(model_name):
        if revision is not None:
            raise ValueError("本地模型目录不能同时指定 --revision")
        return model_name, {}

    if local_only:
        from huggingface_hub import snapshot_download
        try:
            source = snapshot_download(model_name, revision=revision,
                                       local_files_only=True)
        except Exception as exc:
            raise ValueError("本地找不到该模型修订版本的缓存；请联网下载或传 --model 本地目录") from exc
        print("使用本地模型快照: {}".format(os.path.basename(source)))
        return source, {}

    from huggingface_hub import HfApi
    info = HfApi(endpoint=os.environ.get("HF_ENDPOINT")).model_info(
        model_name, revision=revision)
    if not info.sha:
        raise ValueError("无法解析模型的 commit SHA，请传 --revision 固定版本")
    print("模型 commit SHA: {}".format(info.sha))
    # 传相同 commit SHA，避免 main 在两次加载之间漂移。
    return model_name, {"revision": info.sha}


def check_token_limit(token_ids, config_limit, max_tokens, truncate):
    """先验证模型硬上限和教学预算；只有显式允许时才裁剪。"""
    if max_tokens < 1:
        raise ValueError("--max-tokens 必须大于 0")
    if config_limit is None:
        raise ValueError("模型 config 未声明 max_position_embeddings，无法验证上下文上限")
    if not isinstance(config_limit, int) or config_limit < 1:
        raise ValueError("模型配置的上下文上限无效: {}".format(config_limit))
    limit = min(max_tokens, config_limit)
    if limit < max_tokens:
        print("--max-tokens={} 大于模型上下文上限 {}，实际限制为 {}。".format(
            max_tokens, config_limit, limit))
    original_count = len(token_ids)
    if original_count > limit:
        if not truncate:
            raise ValueError(("输入有 {} 个 token，模型演示限制为 {}；缩短 --text，或加 --truncate "
                              "明确仅观察前 {} 个 token").format(original_count, limit, limit))
        print("输入有 {} 个 token，仅保留前 {} 个 token；展示和模型前向都使用截断后的 ID。".format(
            original_count, limit))
        token_ids = token_ids[:limit]
    return token_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="你好，token！", help="待观察的原始文本")
    parser.add_argument("--stage", choices=["tokenizer", "model"], default="tokenizer")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF 模型 ID 或本地模型目录")
    parser.add_argument("--revision", help="HF commit SHA / tag；远程会解析成不可变 SHA")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="模型阶段的 token 预算（默认 256；有效上限不超过模型上下文）")
    parser.add_argument("--truncate", action="store_true",
                        help="超出预算时仅观察输入的前 --max-tokens 个 token")
    parser.add_argument("--mirror", default="__unset__",
                        help="默认使用 HF 镜像；official 使用官方；也可传自定义 URL")
    parser.add_argument("--local-only", action="store_true", help="只使用本地缓存，不访问网络")
    args = parser.parse_args()

    # 必须先设置端点，再导入 transformers/huggingface_hub。
    setup_hf_endpoint(None if args.mirror == "official" else args.mirror)
    hf_report()
    from transformers import AutoTokenizer

    try:
        model_path, load_kwargs = resolve_model_source(args.model, args.revision,
                                                       args.local_only)
    except ValueError as exc:
        parser.error(str(exc))
    # 远程来源复用相同 SHA；本地目录 / 离线快照均只用同一路径。
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_only,
                                               **load_kwargs)

    # ① 原文 -> tokenizer 切分 -> token ID。关闭 special tokens 便于观察原文自身。
    token_ids = tokenizer.encode(args.text, add_special_tokens=False)
    if not token_ids:
        parser.error("输入文本没有编码出 token，请传非空文本")

    truncated = False
    if args.stage == "model":
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, local_files_only=args.local_only,
                                            **load_kwargs)
        config_limit = getattr(config, "max_position_embeddings", None)
        print("模型上下文上限: {}；本演示 token 预算: {}".format(
            config_limit if config_limit is not None else "未在 config 中声明", args.max_tokens))
        print("输入编码得到 {} 个 token。".format(len(token_ids)))
        try:
            checked_ids = check_token_limit(token_ids, config_limit,
                                            args.max_tokens, args.truncate)
        except ValueError as exc:
            parser.error(str(exc))
        truncated = len(checked_ids) < len(token_ids)
        token_ids = checked_ids
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"\n① 原始文本: {args.text!r}")
    if truncated:
        print("   ②～⑥ 仅对应展示的 token 前缀；不代表处理了完整原始文本。")
    print("② tokenizer 切分后的片段 / token ID:")
    for index, (piece, token_id) in enumerate(zip(pieces, token_ids)):
        print(f"   位置 {index:>2}: 原始词表表示={piece!r:<20} ID={token_id:<8} "
              f"单独解码={tokenizer.decode([token_id])!r}")
    print(f"③ token IDs: {token_ids}")
    print(f"   整体解码: {tokenizer.decode(token_ids)!r}")
    if truncated:
        print("   上述解码仅为保留的前缀，不应与完整原文比较。")
    print("   注意：单个 token 可能只是 UTF-8 字节片段；单独解码不一定可读，请看整体解码。")

    if args.stage == "tokenizer":
        print("\n到此只下载 tokenizer；加 --stage model 继续观察模型权重与 Transformer。")
        return

    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("\n未检测到 CUDA：会在 CPU 上运行，首次下载和前向计算可能较慢。")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=args.local_only, **load_kwargs
    ).to(device).eval()
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)  # [batch=1, seq_len]
    embedding_layer = model.get_input_embeddings()

    # ④ 用整数 ID 在模型权重表 E 中取对应行：x_i = E[token_id_i]。
    with torch.inference_mode():
        vectors = embedding_layer(input_ids)
        first_id = input_ids[0, 0]
        first_row = embedding_layer.weight[first_id]
        assert torch.equal(vectors[0, 0], first_row)

        # ⑤ model(input_ids=...) 会在内部再次查同一张 embedding 表；
        #    上面的 vectors 只用于观察，并没有作为额外输入传入。
        #    Qwen2.5 在注意力计算中使用 RoPE 位置信息。
        outputs = model(input_ids=input_ids, output_hidden_states=True,
                        use_cache=False, return_dict=True)
        contextual = outputs.hidden_states[-1]
        next_logits = outputs.logits[0, -1]
        probabilities = torch.softmax(next_logits.float(), dim=-1)
        top_probabilities, top_ids = torch.topk(probabilities, k=min(5, probabilities.numel()))

    print(f"\n④ 输入 ID 张量: shape={tuple(input_ids.shape)}, dtype={input_ids.dtype}")
    print(f"   模型 embedding 表: shape={tuple(embedding_layer.weight.shape)} "
          f"(词表行数 × 向量维度), device={embedding_layer.weight.device}")
    print(f"   查表后的向量: shape={tuple(vectors.shape)} "
          f"(batch × token 数 × 向量维度)")
    print(f"   第一个 token 的前 8 维: {vectors[0, 0, :8].float().tolist()}")
    print("   验证：查表所得向量 == embedding.weight[第一个 token ID] → True")
    print(f"⑤ Transformer 最后一层输出: shape={tuple(contextual.shape)}")
    print(f"   第一个位置的上下文化表示（前 8 维）: {contextual[0, 0, :8].float().tolist()}")
    print(f"⑥ 末位置预测下一个 token：logits shape={tuple(outputs.logits.shape)}")
    for token_id, probability in zip(top_ids.tolist(), top_probabilities.tolist()):
        print(f"   ID={token_id:<8} P={probability:.4f} "
              f"解码={tokenizer.decode([token_id])!r}")
    print("   上述概率对应输入文本之后的下一个 token，不是逐字翻译。")


if __name__ == "__main__":
    main()
