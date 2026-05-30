#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

HERMES_ENV="$HOME/.hermes/.env"
API_KEY="${HERMES_API_KEY:-hermes-voice-key}"
NEED_RESTART=false

# ── 自动配置 Hermes API Server ──────────────────────
if [ -f "$HERMES_ENV" ]; then
    # 检查 API Server 是否已启用
    if ! grep -q "API_SERVER_ENABLED=true" "$HERMES_ENV" 2>/dev/null; then
        echo "==> ⚠️  Hermes API Server 未启用，自动配置 …"
        {
            echo ""
            echo "# Hermes API Server (auto-configured by hermes-voice)"
            echo "API_SERVER_ENABLED=true"
            echo "API_SERVER_KEY=$API_KEY"
        } >> "$HERMES_ENV"
        NEED_RESTART=true
    fi

    # 检查 API_SERVER_KEY 是否一致
    if ! grep -q "API_SERVER_KEY=$API_KEY" "$HERMES_ENV" 2>/dev/null; then
        echo "==> ⚠️  API_SERVER_KEY 不一致，自动修正 …"
        if grep -q "API_SERVER_KEY=" "$HERMES_ENV" 2>/dev/null; then
            sed -i '' "s/^API_SERVER_KEY=.*/API_SERVER_KEY=$API_KEY/" "$HERMES_ENV"
        else
            echo "API_SERVER_KEY=$API_KEY" >> "$HERMES_ENV"
        fi
        NEED_RESTART=true
    fi

    # 同步 config.yaml 的 hermes_api_key
    CURRENT_CFG_KEY=$(grep "hermes_api_key:" config.yaml | sed 's/.*"\(.*\)".*/\1/' 2>/dev/null || true)
    if [ -n "$CURRENT_CFG_KEY" ] && [ "$CURRENT_CFG_KEY" != "$API_KEY" ]; then
        echo "==> ⚠️  config.yaml 的 hermes_api_key 不匹配，自动修正 …"
        sed -i '' "s/hermes_api_key:.*/hermes_api_key: \"$API_KEY\"/" config.yaml
        NEED_RESTART=true
    fi

    # 有改动才重启 gateway
    if [ "$NEED_RESTART" = true ]; then
        echo "==> 重启 Hermes Gateway 使配置生效 …"
        if hermes gateway restart 2>/dev/null; then
            echo "==> Gateway 重启完成"
            sleep 3
        else
            echo "==> Gateway restart 不可用（非 service 模式），跳过重启"
        fi
    fi
fi

# ── 确保 Hermes Gateway 在运行 ──────────────────────
if hermes gateway status &>/dev/null; then
    echo "==> Hermes Gateway already running, reuse it"
else
    echo "==> Starting Hermes Gateway …"
    hermes gateway start &>/dev/null || hermes gateway run --replace &>/dev/null &
    sleep 3
fi

# ── 验证 API Server ─────────────────────────────────
if curl -sf -o /dev/null http://localhost:8642/v1/health 2>/dev/null; then
    echo "==> ✅ API Server 正常"
else
    echo "==> ⚠️  API Server 无响应，检查 ~/.hermes/logs/gateway.log"
fi

echo "==> Starting Hermes Voice …"
python -m voice.main
