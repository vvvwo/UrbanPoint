"""Train UrbanSeg or predict labels for unlabelled point clouds."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--config", help="Module name under configs (without .py)")
    parser.add_argument("--input", nargs="+", help="Input PLY/TXT point clouds for inference")
    parser.add_argument("--weight", help="Checkpoint containing state_dict and epoch")
    parser.add_argument("--exp-name", default="urban_seg")
    parser.add_argument("--seed", type=int, default=14133471)
    args = parser.parse_args()
    if args.mode == "test":
        if not args.input or not args.weight:
            parser.error("test requires --input and --weight; weights are not included")
        for filename in args.input + [args.weight]:
            if not Path(filename).is_file():
                parser.error("File does not exist: " + filename)

    import torch
    from utils.config import get_config
    from utils.misc import set_seed
    if not torch.cuda.is_available():
        parser.error("UrbanSeg requires a CUDA-enabled PyTorch environment")
    config = args.config or ("urban_seg_train" if args.mode == "train" else "urban_seg")
    result_dir = Path("result") / args.exp_name
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "model").mkdir(exist_ok=True)
    cfg = get_config(config, {
        "save_path": str(result_dir), "weight": args.weight,
        "seed": args.seed, "data_path_list": args.input, "resume": False,
    })
    set_seed(args.seed)
    if args.mode == "train":
        from trainer import Trainer
        Trainer(cfg).train()
    else:
        from tester import Tester
        Tester(cfg).test()


if __name__ == "__main__":
    main()
