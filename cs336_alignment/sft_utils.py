import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """

    batch_size = len(prompt_strs)
    
    all_input_ids = []
    all_response_masks = []
    all_lengths = []

    # 1. 分别对每一对 prompt_str 和 output_str 进行分词
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        # add_special_tokens=False 只编码原始文本本身, 不额外添加特殊 token.
        # 在做训练数据拼接, mask, labels 对齐时, 通常更推荐先用 False.
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_ids = tokenizer.encode(output_str, add_special_tokens=False)

        # 拼接序列
        combined_ids = prompt_ids + output_ids
        all_input_ids.append(combined_ids)

        # 计算长度
        all_lengths.append(len(combined_ids))

        # 构造初始 mask: prompt 部分为 0, response 部分为 1
        mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        all_response_masks.append(mask)

    # 2. 确定 batch 的最大长度用于 padding
    max_len = max(all_lengths)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 3. 进行填充 (Padding)
    # 先创建分配空间, 初始值为 pad_id
    padded_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    padded_masks = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (ids, masks) in enumerate(zip(all_input_ids, all_response_masks)):
        length = len(ids)

        # 使用右填充 (Right Padding)
        padded_input_ids[i, :length] = torch.tensor(ids)
        padded_masks[i, :length] = torch.tensor(masks)

    # 4. 执行 shift 操作
    # input_ids: 取前 N - 1 个
    # labels: 取后 N - 1 个
    # response_mask: 对应 labels 的位置, 也取后 N - 1 个

    input_ids = padded_input_ids[:, :-1]
    # 这里的 .clone() 表示复制一份新的 tensor 数据
    # 因为切片通常得到的是原 tensor 的一个视图. 它和原 tensor 可能共享底层内存.
    # 后续只有 labels 中的数值需要原地修改 (例如下面说的设置特殊值), 所以需要 clone
    labels = padded_input_ids[:, 1:].clone()
    response_mask = padded_masks[:, 1:]

    # 5. 可选: 将 labels 中非 response 部分及 padding 部分设为特殊值 (如 -100)
    # 这样在 F.cross_entropy 中可以直接使用 ignore_index=-100
    # 但根据题目要求, 我们只需要返回 response_mask

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension).
    READ notes/compute_entropy.md to learn more.

    计算模型在每个位置预测得有多不确定, 通常不是 SFT loss 必需项, 更多用于监控分析或者后续 RL/探索相关训练.

    Args:
        logits: torch.Tensor, A tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits over the vocabulary.

    Returns:
        torch.Tensor:
            A tensor of shape (batch_size, sequence_length), where each value is 
            the entropy of the next-token distribution at that position.
    """

    # 1. 计算LSE
    # (batch_size, sequence_length)
    lse = torch.logsumexp(logits, dim=-1)

    # 2. 计算概率 p = softmax(logits)
    # (batch_size, sequence_length, vocab_size)
    probs = F.softmax(logits, dim=-1)

    # 3. 计算期望 E[logits] = sum(p_i * z_i)
    # (batch_size, sequence_length)
    exp_logits = torch.sum(probs * logits, dim=-1)

    # 4. 熵 H = LSE - sum(p_i * z_i)
    entory = lse - exp_logits

    return entory

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    计算模型给标准答案 token 分配了多大概率, 这是 SFT loss 的核心.
    使用 log 概率的原因:
    1. 语言模型序列概率是很多 token 概率的乘积, 取 log 后连乘变连加, 数值更稳定
    2. -log(p) 正好形成交叉熵 loss, 并且对低概率正确答案有更强的训练惩罚.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    
    # 1. 获取模型的输出 logits
    outputs = model(input_ids)
    logits = outputs.logits # (batch_size, sequence_length, vocab_size)

    # 2. 计算所有 token 的 log_softmax
    # softmax 输出概率, log_softmax 输出 log 概率
    # 数学上 log_softmax(x) = log(softmax(x)), 但 F.log_softmax 更稳定, 训练语言模型时通常更常用.
    log_probs_all = F.log_softmax(logits, dim=-1) # (batch_size, sequence_length, vocab_size)

    # 3. 提取对应 label 的对数概率
    # 因为在训练时我们并不关系除了对应 label 的概率
    # gather: 按照 index 从指定维度上取值. 要求 index 与输入张量维度一致
    # unsqueeze(-1): 在最后增加一个维度
    # squeeze(-1): 删除最后一个大小为 1 的维度.
    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # 4. 构建返回字典
    results = {"log_probs": log_probs}

    if return_token_entropy:
        results["token_entropy"] = compute_entropy(logits)

    return results

    
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    把 prompt / padding token 的 loss 屏蔽掉, 只对 response token 的 loss 求和, 
    并按指定分母归一化, 从而得到正确的 SFT loss. normalize_constant 通常是 response token 的数量.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    
    # 1. 将 tensor 与 mask 相乘, 排除 mask == 0 的元素
    masked_tensor = tensor * mask

    # 2. 根据 dim 进行求和
    if dim is None:
        # 对张量中的所有元素求和, 返回一个标量
        total_sum = torch.sum(masked_tensor)
    else:
        # 沿着指定维度求和
        total_sum = torch.sum(masked_tensor, dim=dim)

    # 3. 除以归一化常数
    return total_sum / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss and backprop its gradients for a microbatch.

    microbatch 可以理解成把一个大 batch 拆成几个小 batch, 每次只在显存里跑其中一小块, 它主要是为了解决 GPU 显存不够 的问题.
    假设 global batch = 64, microbatch = 8, 那么每一个 microbatch 就有8条样本. 
    每个 microbatch 都 forward 一次, 算 loss, backward 一次, 但暂时不更新参数. 
    等 8 个 microbatch 的梯度都累积完, 再执行一次 optimizer.step() 根据梯度更新参数 optimizer.zero_grad() 清空旧梯度.
    这叫 gradient accumulation, 即梯度累积. gradient_accumulation_steps 这个参数就是表示被拆分为几个 microbatch.
    """
    pass

