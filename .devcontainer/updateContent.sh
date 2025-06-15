#!/usr/bin/env bash
set -e
# 1. Enable HTTPS auth via Personal Access Token
git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf "git@github.com:"
# 2. Sync any changes in .gitmodules to .git/config
git submodule sync
# 3. Clone and update all submodules recursively
git submodule update --init --recursive
