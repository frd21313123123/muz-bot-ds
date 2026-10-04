"""Prepare CPU INT8 weights once on a machine with the optional Torch environment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('tiny', 'base', 'small'), default='small')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from ctranslate2.converters import TransformersConverter
    converter = TransformersConverter('openai/whisper-' + args.model, load_as_float16=True,
                                      copy_files=['tokenizer.json', 'preprocessor_config.json'])
    converter.convert(str(args.output), quantization='int8')
    (args.output / 'conversion.json').write_text(json.dumps({
        'source': 'openai/whisper-' + args.model, 'quantization': 'int8', 'intermediate_dtype': 'float16',
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
