# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

rules = """
# Failing to add message in the mentioned format will
# cause your local commit fail.
#
# Please follow these rules for commit messages:
# ==============================================
# 1. Commit message format - first line is mandatory
#    [YOUR_MODULE_NAME] Subject line here not exceeding 50 characters
#    * Optional line entry with detail not exceeding 72 characters
#    * Optional line entry with detail not exceeding 72 characters
#    * Optional line entry with detail not exceeding 72 characters
# 2. Limit the module name (YOUR_MODULE_NAME) to 15 characters length 
# 3. Limit the subject(Text part after [YOUR_MODULE_NAME]) line to max
#    50 characters
# 4. Start subject (Text part after [YOUR_MODULE_NAME]) with a Capital
#    letter and don't end with a period '.'
# 5. Wrap the body lines (if any) at 72 characters
"""