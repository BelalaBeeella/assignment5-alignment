import argparse
import json

from pathlib import Path
from typing import Any, Callable
from vllm import LLM, SamplingParams
from tqdm import tqdm

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    path = Path(path)
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    
    return examples

def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save rows to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_prompt_template(path: str | Path) -> str:
    """Load the prompt template."""
    path = Path(path)
    return path.read_text(encoding="utf-8")



def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    gold_answers: list[str],
    eval_sampling_params: SamplingParams,
) -> list[dict[str, Any]]:
    """
    Evaluate a language model on a list of prompts.

    This function:
    1. Generates model outputs using vLLM.
    2. Computes reward metrics using reward_fn.
    3. Returns per-example results.
    """

    assert len(prompts) == len(gold_answers), f"len(prompts)={len(prompts)} but len(gold_answers)={len(gold_answers)}"

    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []

    for i, output in enumerate(tqdm(outputs, desc="Scoring generations")):
        prompt = prompts[i]
        gold_answer = gold_answers[i]

        generation = output.outputs[0].text

        reward_dict = reward_fn(generation, gold_answer)

        results.append(
            {
                "index": i,
                "prompt": prompt,
                "gold_answer": gold_answer,
                "generation": generation,
                "reward": reward_dict
            }
        )

    return results

def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:

    """
    Summarize format reward and answer reward categories.
    Expected reward keys are usually:
    - format_reward
    - answer_reward
    But this function is slightly robust to naming differences.
    """

    def get_reward_value(reward: dict[str, float], possible_keys: list[str]) -> float:
        for key in possible_keys:
            if key in reward:
                return float(reward[key])
        raise KeyError(
            f"Cannot find any of keys {possible_keys} in reward dict: {reward}"
        )

    total = len(results)
    format_1_answer_1 = 0
    format_1_answer_0 = 0
    format_0_answer_0 = 0
    other = 0
    answer_correct = 0
    format_correct = 0

    for row in results:
        reward = row["reward"]
        format_reward = get_reward_value(
            reward,
            ["format_reward", "format", "format_correct"],
        )
        answer_reward = get_reward_value(
            reward,
            ["answer_reward", "answer", "answer_correct"],
        )
        if format_reward == 1:
            format_correct += 1
        if answer_reward == 1:
            answer_correct += 1
        if format_reward == 1 and answer_reward == 1:
            format_1_answer_1 += 1
        elif format_reward == 1 and answer_reward == 0:
            format_1_answer_0 += 1
        elif format_reward == 0 and answer_reward == 0:
            format_0_answer_0 += 1
        else:
            other += 1

    return {
        "total": total,
        "answer_accuracy": answer_correct / total if total > 0 else 0.0,
        "format_accuracy": format_correct / total if total > 0 else 0.0,
        "format_1_answer_1": format_1_answer_1,
        "format_1_answer_0": format_1_answer_0,
        "format_0_answer_0": format_0_answer_0,
        "other": other,
    }

def print_examples_by_category(
    results: list[dict[str, Any]],
    max_examples: int = 10,
) -> None:

    """
    Print examples for analysis required in part (b).
    """

    def get_reward_value(reward: dict[str, float], possible_keys: list[str]) -> float:
        for key in possible_keys:
            if key in reward:
                return float(reward[key])
        raise KeyError(
            f"Cannot find any of keys {possible_keys} in reward dict: {reward}"
        )

    categories = {
        "format=1, answer=1": [],
        "format=1, answer=0": [],
        "format=0, answer=0": [],
    }

    for row in results:
        reward = row["reward"]
        format_reward = get_reward_value(
            reward,
            ["format_reward", "format", "format_correct"],
        )
        answer_reward = get_reward_value(
            reward,
            ["answer_reward", "answer", "answer_correct"],
        )

        if format_reward == 1 and answer_reward == 1:
            categories["format=1, answer=1"].append(row)
        elif format_reward == 1 and answer_reward == 0:
            categories["format=1, answer=0"].append(row)
        elif format_reward == 0 and answer_reward == 0:
            categories["format=0, answer=0"].append(row)

    for category_name, rows in categories.items():
        print("\n" + "=" * 100)
        print(f"Category: {category_name}")
        print(f"Count: {len(rows)}")
        print("=" * 100)
        for row in rows[:max_examples]:
            print(f"\nIndex: {row['index']}")
            print(f"Gold answer: {row['gold_answer']}")
            print(f"Reward: {row['reward']}")
            print("Generation:")
            print(row["generation"])
            print("-" * 100)

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/Qwen2.5-Math-1.5B",
        help="Path or HuggingFace name of the model.",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/gsm8k/test.jsonl",
        help="Path to MATH validation jsonl.",
    )

    parser.add_argument(
        "--prompt-path",
        type=str,
        default="cs336_alignment/prompts/r1_zero.prompt",
        help="Path to r1_zero prompt template.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/math_baseline/qwen2.5_math_1.5b_validation_results.jsonl",
        help="Where to save per-example results.",
    )

    parser.add_argument(
        "--summary-path",
        type=str,
        default="outputs/math_baseline/qwen2.5_math_1.5b_validation_summary.json",
        help="Where to save summary metrics.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="For debugging. Evaluate only the first N examples.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum model context length.",
    )

    args = parser.parse_args()

    print(f"Loading data from: {args.data_path}")
    examples = load_jsonl(args.data_path)

    if args.max_examples is not None:
        examples = examples[:args.max_examples]

    print(f"Number of examples: {len(examples)}")

    print(f"Laoding prompt template from: {args.prompt_path}")
    prompt_template = load_prompt_template(args.prompt_path)

    prompts = []
    gold_answers = []

    for example in examples:
        question = example["question"]
        answer = example["answer"]

        prompts.append(prompt_template.format(question=question))
        gold_answers.append(answer)

    print(f"Loading vLLM model from: {args.model_path}")
    vllm_model = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    eval_sampling_paras = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["# Query:", "```", "\n\n#", "User:", "Assistant:"],
    )

    print("Generating and evaluating...")
    results = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        gold_answers=gold_answers,
        eval_sampling_params=eval_sampling_paras,
    )

    print(f"Saving per-example result to: {args.output_path}")
    save_jsonl(args.output_path, results)

    summary = summarize_results(results)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print_examples_by_category(results, max_examples=10)

if __name__ == "__main__":
    main()


