#!/bin/bash
set -euo pipefail

WRAPPER_DIR="${EXPERIMENT_AGENT_MCP_WRAPPER_DIR:-$HOME/.cache/researchagent_mcp/bin}"
mkdir -p "$WRAPPER_DIR"

run_warmup() {
  "$@" || echo "warmup failed: $*"
}

run_warmup npx -y -p @modelcontextprotocol/server-github node -e "process.exit(0)"
run_warmup npx -y -p @modelcontextprotocol/server-filesystem node -e "process.exit(0)"
run_warmup npx -y -p @kazuph/mcp-fetch node -e "process.exit(0)"
run_warmup npx -y -p @upstash/context7-mcp node -e "process.exit(0)"
run_warmup uvx --from minimax-coding-plan-mcp python -c "import sys; sys.exit(0)"

cat > "$WRAPPER_DIR/mcp-github" <<'EOF'
#!/bin/bash
set -euo pipefail
exec npx -y @modelcontextprotocol/server-github "$@"
EOF

cat > "$WRAPPER_DIR/mcp-filesystem" <<'EOF'
#!/bin/bash
set -euo pipefail
exec npx -y @modelcontextprotocol/server-filesystem "$@"
EOF

cat > "$WRAPPER_DIR/mcp-fetch" <<'EOF'
#!/bin/bash
set -euo pipefail
exec npx -y @kazuph/mcp-fetch "$@"
EOF

cat > "$WRAPPER_DIR/mcp-context7" <<'EOF'
#!/bin/bash
set -euo pipefail
exec npx -y @upstash/context7-mcp "$@"
EOF

cat > "$WRAPPER_DIR/mcp-minimax" <<'EOF'
#!/bin/bash
set -euo pipefail
exec uvx --from minimax-coding-plan-mcp minimax-coding-plan-mcp "$@"
EOF

chmod +x \
  "$WRAPPER_DIR/mcp-github" \
  "$WRAPPER_DIR/mcp-filesystem" \
  "$WRAPPER_DIR/mcp-fetch" \
  "$WRAPPER_DIR/mcp-context7" \
  "$WRAPPER_DIR/mcp-minimax"

echo "installed wrappers at $WRAPPER_DIR"
