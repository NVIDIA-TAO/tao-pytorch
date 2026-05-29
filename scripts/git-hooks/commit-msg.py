#!/usr/bin/python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import re
from submodules.rules import rules
from nvidia_tao_pytorch.core.path_utils import expand_path

def main():
	file_path = expand_path(sys.argv[1])
	with open(file_path, "r") as fp:
		lines = fp.readlines()

		for idx, line in enumerate(lines):

			if line.strip() == "# ------------------------ >8 ------------------------":
				break

			if line[0] == "#":
				continue

			if not line_valid(idx, line):
				print(f"line# {idx} failed")
				show_rules()
				sys.exit(1)

	sys.exit(0)

def line_valid(idx, line):
	if idx == 0:
		#return re.match("^[A-Z].{,48}[0-9A-z \t]$", line)
		return re.match("^\[((?!\s*$).{0,15})\][ \t].*?[A-Z].{0,48}[0-9A-z \t]$", line)
	else:
		return len(line.strip()) <= 72

def show_rules():
	print(rules)

if __name__ == "__main__":
	main()