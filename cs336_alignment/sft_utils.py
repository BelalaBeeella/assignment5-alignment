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
    

    

