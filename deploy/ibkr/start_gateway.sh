#!/usr/bin/env bash
# Headless IB Gateway launcher for systemd
# Overrides the default paths in gatewaystart.sh to match our server layout,
# then calls it with -inline (runs in the foreground — no xterm window needed).
set -euo pipefail

export DISPLAY=:99

# ── IBC / Gateway paths ────────────────────────────────────────────────────────
export IBC_PATH=/opt/ibc
export TWS_PATH=/opt/ibgateway          # where IB Gateway was installed
export TWS_SETTINGS_PATH=/opt/ibgateway # where Gateway stores its settings
export IBC_INI=/opt/ibc/config.ini      # credentials + IBC settings
export LOG_PATH=/var/log/quantpipe      # redirect logs here
export APP=GATEWAY

# IB Gateway version installed: 10.37 → major version = 1037
export TWS_MAJOR_VRSN=1037

# Read TradingMode from config.ini (paper or live)
export TRADING_MODE=

# Use system Java (OpenJDK 21 installed by setup_ibkr.sh)
export JAVA_PATH=

# Verify required files exist before attempting launch
[[ -f "$IBC_INI" ]]              || { echo "ERROR: $IBC_INI not found"; exit 1; }
[[ -d "$TWS_PATH" ]]             || { echo "ERROR: $TWS_PATH not found"; exit 1; }
[[ -f "$IBC_PATH/IBC.jar" ]]    || { echo "ERROR: IBC.jar not found in $IBC_PATH"; exit 1; }

IBLOGIN=$(grep -E '^IbLoginId=' "$IBC_INI" | cut -d= -f2)
[[ "$IBLOGIN" == "REPLACE_WITH_YOUR_IBKR_USERNAME" || -z "$IBLOGIN" ]] && {
    echo "ERROR: IBKR credentials not set in $IBC_INI"
    echo "  Run: nano $IBC_INI"
    echo "  Set IbLoginId and IbPassword, then restart the service."
    exit 1
}

exec "$IBC_PATH/gatewaystart.sh" -inline
