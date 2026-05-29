#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

# Select NGC CLI type based on command line arg
BATCH_CLI='ngccli_bat_linux.zip'
REG_CLI='ngccli_reg_linux.zip'

# Installing NGC CLI type based on env variable.
if [ "x$NGC_INSTALL_CLI" == 'xBATCH' ]; then
    CLI="$BATCH_CLI"
elif [ "x$NGC_INSTALL_CLI" == 'xREGISTRY' ]; then
    CLI="$REG_CLI"
else
    echo "Invalid NGC_INSTALL_CLI asked for. Exiting"
    exit 1
fi

## Download and install
mkdir -p /opt/ngccli && \
wget "https://ngc.nvidia.com/downloads/$CLI" -P /opt/ngccli && \
unzip -u "/opt/ngccli/$CLI" -d /opt/ngccli/ && \
rm /opt/ngccli/*.zip && \
chmod u+x /opt/ngccli/ngc

## Running passed command
if [[ "$1" ]]; then
	eval "$@"
fi