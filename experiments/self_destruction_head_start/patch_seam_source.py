"""Apply the LN-395 execution patches to the pinned SEAM source.

The patches change device placement, logging and output recording only. They
do not change SEAM's objective, data, optimizer or attack.
"""

from pathlib import Path
import sys


def replace_once(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, (path, old[:60])
    path.write_text(text.replace(old, new))


def main():
    root = Path(sys.argv[1]).resolve()

    # Single-GPU placement: SEAM stores auxiliary gradients on the last two
    # visible GPUs; with fewer than three GPUs, keep them on cuda:0.
    trainer = root / "src/core/trainer.py"
    replace_once(
        trainer,
        '            else:\n                print("Available GPU less than three")\n        else:',
        '            else:\n                return [0], ["cuda:0", "cuda:0"]\n        else:')
    replace_once(
        trainer,
        '            else:\n                print("Available GPU less than three")\n',
        '            else:\n                return [0], ["cuda:0", "cuda:0"]\n')

    train = root / "src/train.py"
    replace_once(train, "        return self._n_gpu - 2\n", "        return max(self._n_gpu - 2, 1)\n")
    replace_once(
        train,
        '    trainer.train()\n    wandb.log({"Output dir": training_args.output_dir})',
        '    trainer.train()\n    trainer.save_model(training_args.output_dir)\n'
        '    tokenizer.save_pretrained(training_args.output_dir)\n'
        '    wandb.log({"Output dir": training_args.output_dir})')

    evaluate = root / "src/eval.py"
    replace_once(
        evaluate,
        '    lora_attack: bool = field(',
        '    save_dir: str = field(default="", metadata={"help": "Save the post-attack model here"})\n'
        '    output_json: str = field(default="", metadata={"help": "Write scores here"})\n'
        '    lora_attack: bool = field(')
    replace_once(
        evaluate,
        '    harmful_score_gpt = eval_harmfulness_gpt(model, test_dataloader, tokenizer)\n'
        '    print("Harmful score GPT: ", harmful_score_gpt)\n'
        '    wandb.log({"Harmful score GPT": harmful_score_gpt})\n',
        '    record = {"model_name": eval_args.model_name, "attack": eval_args.attack,\n'
        '              "learning_rate": training_args.learning_rate, "attack_size": eval_args.attack_size,\n'
        '              "harmful_score": float(harmful_score)}\n'
        '    if eval_args.pre_utility:\n'
        '        record["pre_utility"] = {t: utility_score[t]["acc,none"] for t in eval_args.utility_tasks}\n'
        '    if eval_args.save_dir:\n'
        '        model.save_pretrained(eval_args.save_dir)\n'
        '        tokenizer.save_pretrained(eval_args.save_dir)\n'
        '    if os.environ.get("OPENAI_API_KEY"):\n'
        '        harmful_score_gpt = eval_harmfulness_gpt(model, test_dataloader, tokenizer)\n'
        '        print("Harmful score GPT: ", harmful_score_gpt)\n'
        '        wandb.log({"Harmful score GPT": harmful_score_gpt})\n')
    replace_once(
        evaluate,
        '            print("Post utility score: ", utility_score)\n'
        '            wandb.log({"Post utility score": utility_score})\n',
        '            print("Post utility score: ", utility_score)\n'
        '            wandb.log({"Post utility score": utility_score})\n'
        '        record["post_utility"] = {t: utility_score[t]["acc,none"] for t in eval_args.utility_tasks}\n'
        '    if eval_args.output_json:\n'
        '        import json\n'
        '        with open(eval_args.output_json, "w") as handle:\n'
        '            json.dump(record, handle, indent=2)\n')
    text = evaluate.read_text()
    if "\nimport os\n" not in text:
        evaluate.write_text("import os\n" + text)

    datasets = root / "src/data_processing/_datasets.py"
    assert 'open("data/beavertails_with_refusals_train.json"' in datasets.read_text()
    print("patched", root)


if __name__ == "__main__":
    main()
