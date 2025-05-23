#!/usr/bin/env python3

# ===-- normalize_json.py - Tool to normalize JSON files ------------------=== #
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------=== #

import argparse
import json
import sys
from typing import Iterator, IO

import json5


# Reads the given `file`` line by line until a stripped line is exactly equal to
# the `separator` and yields every chunk between the separators (or the
# beginning and end of the file).
def split_file(file: IO, separator: str) -> Iterator[str]:
  chunk: list[str] = []
  for line in file:
    if line.strip() == separator:
      yield ''.join(chunk)
      chunk = []
    else:
      chunk.append(line)
  yield ''.join(chunk)


def main(argv: list[str]):
  # Set up argument parser.
  parser = argparse.ArgumentParser(
      prog='normalize_python',
      description='Brings a possibly split JSON file into a normalized file',
  )
  parser.add_argument('-i',
                      '--input',
                      default='-',
                      help='Input JSON file. Use - for stdin.')
  parser.add_argument('-s',
                      '--separator',
                      default='// -----',
                      help='String that separates "splits" in the input file.')
  args = parser.parse_args()

  # Open file, then read, normalize, and print each chunk.
  input_file = open(args.input, 'r') if args.input != '-' else sys.stdin
  is_first_chunk = True
  for chunk in split_file(input_file, args.separator):
    if not is_first_chunk:
      print(args.separator)
    is_first_chunk = False
    normalized = json.dumps(json5.loads(chunk), sort_keys=True, indent=2)
    print(normalized)


if __name__ == '__main__':
  main(sys.argv)
