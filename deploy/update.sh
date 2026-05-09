#!/usr/bin/env bash
# Legacy script — not used by the current Docker deployment.
# Deployment is handled by GitHub Actions: git pull + docker compose up -d --build quantpipe
# See: .github/workflows/deploy.yml and github.com/micahabanschick/Banschick_Toolset
#
# Kept for reference only.
echo "This script is not used in the Docker deployment."
echo "To redeploy: cd /opt/banschick-toolset && docker compose up -d --build quantpipe"
