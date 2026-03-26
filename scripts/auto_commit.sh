#!/bin/bash
# Auto-commit and push Empire changes every 2 hours
# Installed via: crontab -e → 0 */2 * * * /Users/asd/Downloads/Empireisgay-main/scripts/auto_commit.sh

cd /Users/asd/Downloads/Empireisgay-main || exit 1

# Check for changes (excluding .claude/ and .env)
changes=$(git status --porcelain | grep -v "^.. .claude/" | grep -v "^.. .env")

if [ -z "$changes" ]; then
    exit 0  # Nothing to commit
fi

# Stage everything except .claude/ and .env
git add -A
git reset HEAD .claude/ 2>/dev/null
git reset HEAD .env 2>/dev/null

# Generate commit message from changed files
file_count=$(git diff --cached --name-only | wc -l | tr -d ' ')
summary=$(git diff --cached --name-only | head -5 | tr '\n' ', ' | sed 's/,$//')

git commit -m "Auto-commit: ${file_count} file(s) updated — ${summary}

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

git push origin main 2>&1
