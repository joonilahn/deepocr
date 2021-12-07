import argparse

from deepocr.apis import inference_recognizer, init_recognizer
from deepocr.datasets import build_converter


def parse_args():
    parser = argparse.ArgumentParser(description="DeepOCR Inference.")
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='a checkpoint file')
    parser.add_argument("--img", help="input image file")
    parser.add_argument(
        "--eos-index",
        type=int,
        default=1,
        help="Index number for the 'eos' token. Default to 1."
    )
    args = parse.parse_args()
    return args

def main():
    """Inference a single image file."""
    args = parse_args()

    # build the model and load checkpoint
    model = init_recognizer(args.config, args.checkpoint, device="cuda:0")

    result = inference_recognizer(
        model, args.img, color_type=model.cfg.inference_pipeline[0].color_type
    )

    # decode the result
    converter = build_converter(model.cfg.data.test.converter)
    result_str = converter.decode(result, remove_eos=True, eos_index=args.eos_index)[0]

    print("Inference result: {}".format(result_str))


if __name__ == "__main__":
    main()