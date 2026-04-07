#!/usr/bin/env bash
#=============================================================================
# Codex Code 一键启动脚本（内网服务器使用）
#
# 功能：
#   1. 通过 Xray (VMess/TCP) 建立本地代理
#   2. 启动 Codex Code（自动走 VMess 节点 IP 直出）
#   3. 提示本地电脑完成首次 OAuth 登录
#
# 使用方法：
#   codex-start               # Xray 模式启动（默认）
#   codex-start -P 54200      # 多开时指定端口，每窗口不同端口
#   codex-start --update      # 先更新再启动
#   codex-start --help        # 查看帮助
#
# 前提条件：
#   - 首次运行自动安装配置 Xray 客户端（无需 sudo）
#=============================================================================

set -euo pipefail

#--- 配置（按实际情况修改）---
LOCAL_PROXY_PORT="${LOCAL_PROXY_PORT:-54199}"   # 本地代理端口，可 -P/--port 或环境变量覆盖（多开时每窗口不同端口）
PORT_SEARCH_LIMIT="${PORT_SEARCH_LIMIT:-20}"    # 默认端口被占用时，最多向上探测的端口数量

# Xray 协议类型: vmess | ss | vmess-ws
XRAY_PROTOCOL="ss"

# ==================================== VMess 配置（TCP 直连）====================================
# 用于 香港Y09 等节点（type: vmess，network 未定义则默认 tcp）
XRAY_SERVER_IP="b.s.2.0.p.j.j.9.6.hk09-vm5.entry.v51124-3.qpon"
XRAY_SERVER_PORT=15266
XRAY_UUID="e0ca35ac-cdcc-3035-b340-5bf253bc9bb1"
XRAY_ALTER_ID=1
XRAY_SECURITY="auto"
XRAY_NETWORK="tcp"

# ==================================== VMess WebSocket 配置 ===================================
# 用于 香港Y10 / 香港Y11 等（network: ws）
XRAY_WS_SERVER_IP=""
XRAY_WS_SERVER_PORT=""
XRAY_WS_UUID=""
XRAY_WS_ALTER_ID=1
XRAY_WS_NETWORK="ws"
XRAY_WS_PATH="/"
XRAY_WS_HOST=""         # WebSocket Host 头

# ==================================== Shadowsocks 配置 =====================================
# 用于 日本Y01 等节点（type: ss）
XRAY_SS_SERVER_IP="5.5.2.u.p.v.k.9.6.jp01-ae5.entry.v51124-3.qpon"
XRAY_SS_SERVER_PORT=474
XRAY_SS_CIPHER="aes-256-gcm"
XRAY_SS_PASSWORD="e0ca35ac-cdcc-3035-b340-5bf253bc9bb1"

# Xray 本地路径（用户级安装，无需 sudo）
XRAY_BIN="${HOME}/.local/bin/xray"
XRAY_CONFIG_DIR="${HOME}/.config/xray"
XRAY_CONFIG="${XRAY_CONFIG_DIR}/client.json"
XRAY_LOG="${XRAY_CONFIG_DIR}/xray.log"
#-------------------------------

XRAY_PID=""    # Xray 客户端进程 PID
DETECTED_TZ=""  # 从出口 IP 自动检测的时区（check_proxy 中填充）
CLEANED_UP=false  # 防止 cleanup 重复执行（INT + EXIT 连续触发）

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    cat << 'EOF'
用法: codex-start [选项] [codex 参数...]

选项:
  --help        显示此帮助
  --check       检查代理连通性
  --update      更新 Codex Code 到最新版（通过代理，使用 npm 官方源）
  -P, --port N  使用本地端口 N（多开时每窗口用不同端口，避免冲突）

示例:
  codex-start                      # Xray Shadowsocks 模式（默认），端口 54199
  codex-start --update             # 先更新再启动
  codex-start -P 54200             # 多开：窗口 1
  codex-start -P 54201             # 多开：窗口 2
  codex-start --port 54202 -p proj # 多开并传参给 codex
  codex-start -P 54200 --check     # 检查指定端口代理

说明:
  支持 VMess/TCP、VMess/WebSocket、Shadowsocks 三种协议。
  切换协议: 修改顶部 XRAY_PROTOCOL 变量（ss / vmess / vmess-ws）。
  默认端口被占用时，脚本会自动向上寻找附近可用端口。
EOF
    exit 0
}

# ========================== 代理引用计数 ==========================
#
# 多窗口共享同一代理端口时，通过引用计数避免先退出的窗口关闭代理：
#   不同启动器（codex/claude）只要复用同一端口，也必须共用同一份引用计数。
#   启动器升级过渡期同时维护旧 registry，避免新旧脚本混跑时互相误杀。

proxy_ref_files() {
    printf '%s\n' \
        "/tmp/agent-proxy-${LOCAL_PROXY_PORT}.refs" \
        "/tmp/codex-proxy-${LOCAL_PROXY_PORT}.refs" \
        "/tmp/claude-proxy-${LOCAL_PROXY_PORT}.refs"
}

proxy_port_select_lock_file() {
    printf '%s\n' "/tmp/agent-proxy-port-select.lock"
}

port_listener_pid() {
    local port="${1:-$LOCAL_PROXY_PORT}"

    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -1 || true
    fi
}

port_is_listening() {
    local port="${1:-$LOCAL_PROXY_PORT}"
    local pid
    pid=$(port_listener_pid "$port")
    if [ -n "${pid:-}" ]; then
        return 0
    fi

    if command -v ss >/dev/null 2>&1; then
        ss -ltn "( sport = :${port} )" 2>/dev/null | tail -n +2 | grep -q .
        return $?
    fi

    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | grep -q .
        return $?
    fi

    # 兜底：通过 /proc/net/tcp 检测监听状态（port 需转十六进制）
    local hex_port
    hex_port=$(printf '%x' "$port")
    awk -v p="$hex_port" 'tolower($2) ~ p"$" && $4=="0A" {found=1} END {exit found ? 0 : 1}' /proc/net/tcp 2>/dev/null
    return $?
}

port_listener_cmd() {
    local pid="${1:-}"
    if [ -z "$pid" ]; then
        return 0
    fi

    ps -p "$pid" -o comm= 2>/dev/null || true
}

find_nearby_available_port() {
    local base_port="${1:-$LOCAL_PROXY_PORT}"
    local offset=0
    local candidate

    while [ $offset -le "$PORT_SEARCH_LIMIT" ]; do
        candidate=$((base_port + offset))
        if ! port_is_listening "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
        offset=$((offset + 1))
    done

    return 1
}

ensure_xray_port_ready() {
    local requested_port="$LOCAL_PROXY_PORT"
    local existing_pid
    local existing_cmd
    local next_port

    if ! port_is_listening "$requested_port"; then
        return 1
    fi

    existing_pid=$(port_listener_pid "$requested_port")
    if [ -z "${existing_pid:-}" ]; then
        echo -e "${YELLOW}  端口 $requested_port 已被其他进程占用，自动寻找附近可用端口...${NC}"
    else
        existing_cmd=$(port_listener_cmd "$existing_pid")
        if [[ "${existing_cmd:-}" == *"xray"* ]]; then
            echo -e "${GREEN}  复用已有 Xray 进程 (PID: ${existing_pid}，端口: ${requested_port})${NC}"
            return 0
        fi

        echo -e "${YELLOW}  端口 $requested_port 被非 Xray 进程占用 (PID: ${existing_pid}, ${existing_cmd:-未知})，自动寻找附近可用端口...${NC}"
    fi

    next_port=$(find_nearby_available_port "$requested_port") || {
        echo -e "${RED}  从端口 ${requested_port} 起向上 ${PORT_SEARCH_LIMIT} 个端口内都没有可用端口。${NC}"
        exit 1
    }

    LOCAL_PROXY_PORT="$next_port"
    echo -e "${GREEN}  自动切换到可用端口: ${LOCAL_PROXY_PORT}${NC}"

    return 1
}

handle_xray_start_failure() {
    local existing_pid
    local existing_cmd

    if port_is_listening; then
        existing_pid=$(port_listener_pid)
        if [ -n "${existing_pid:-}" ]; then
            existing_cmd=$(port_listener_cmd "$existing_pid")
            if [[ "${existing_cmd:-}" == *"xray"* ]]; then
                XRAY_PID=""
                echo -e "${GREEN}  检测到已有 Xray 进程已占用端口，改为复用 (PID: ${existing_pid})${NC}"
                return 0
            fi
            echo -e "${RED}  端口 $LOCAL_PROXY_PORT 已被进程占用 (PID: ${existing_pid}, ${existing_cmd:-未知})${NC}"
        else
            echo -e "${RED}  端口 $LOCAL_PROXY_PORT 已被其他进程占用，但当前用户无法识别具体进程。${NC}"
        fi
    fi

    echo -e "${RED}  Xray 客户端启动失败！最近日志:${NC}"
    tail -5 "$XRAY_LOG" 2>/dev/null || true
    exit 1
}

# 注册当前进程为代理消费者（flock 保证多进程并发安全）
proxy_ref_add() {
    local ref_file
    while IFS= read -r ref_file; do
        {
            flock -x 9
            echo $$ >> "$ref_file"
        } 9>"${ref_file}.lock"
    done < <(proxy_ref_files)
}

# 注销当前进程并判断是否为最后一个存活的消费者
# 同时清理已死亡的僵尸 PID，返回 0 = 最后一个（应清理），1 = 还有其他（保留代理）
proxy_ref_is_last() {
    local ref_file
    local tmp
    local pid
    declare -A live_pids=()

    while IFS= read -r ref_file; do
        tmp="${ref_file}.tmp"
        {
            flock -x 9
            if [ -f "$ref_file" ]; then
                : > "$tmp"
                while IFS= read -r pid; do
                    [ -z "$pid" ] && continue
                    [ "$pid" = "$$" ] && continue
                    if kill -0 "$pid" 2>/dev/null; then
                        echo "$pid" >> "$tmp"
                        live_pids["$pid"]=1
                    fi
                done < "$ref_file"
                if [ -s "$tmp" ]; then
                    mv "$tmp" "$ref_file"
                else
                    rm -f "$tmp" "$ref_file"
                fi
            fi
        } 9>"${ref_file}.lock"
    done < <(proxy_ref_files)

    [ "${#live_pids[@]}" -eq 0 ]
}

# ========================== 清理函数 ==========================

# 关闭 Xray 客户端
cleanup_xray() {
    if [ -n "$XRAY_PID" ] && kill -0 "$XRAY_PID" 2>/dev/null; then
        echo ""
        echo -e "${CYAN}正在关闭 Xray 客户端 (PID: $XRAY_PID)...${NC}"
        kill "$XRAY_PID" 2>/dev/null || true
        wait "$XRAY_PID" 2>/dev/null || true
        echo -e "${GREEN}Xray 客户端已关闭。${NC}"
    else
        # 复用场景：XRAY_PID 为空但我们是最后消费者，按端口查找关闭
        local pid
        pid=$(port_listener_pid)
        if [ -n "${pid:-}" ]; then
            local cmd
            cmd=$(port_listener_cmd "$pid")
            if [[ "${cmd:-}" == *"xray"* ]]; then
                echo ""
                echo -e "${CYAN}正在关闭 Xray 客户端 (PID: ${pid})...${NC}"
                kill "$pid" 2>/dev/null || true
                echo -e "${GREEN}Xray 客户端已关闭。${NC}"
            fi
        fi
    fi
}

# 统一清理入口（引用计数 + Xray 清理 + 兜底杀子进程）
cleanup() {
    # 防止重复执行（INT + EXIT 连续触发）
    if $CLEANED_UP; then
        return 0
    fi
    CLEANED_UP=true

    if proxy_ref_is_last; then
        cleanup_xray
        # 兜底：杀掉本脚本的所有残留子进程
        pkill -P $$ 2>/dev/null || true
    else
        echo ""
        echo -e "${CYAN}其他窗口仍在使用代理，保留代理进程。${NC}"
    fi
}

# ========================== Xray 安装与配置 ==========================

# 确保 Xray 二进制可用（自动检测/下载/安装）
ensure_xray() {
    # 优先检查本地用户安装
    if [ -x "$XRAY_BIN" ]; then
        return 0
    fi

    # 其次检查 PATH 中是否已有 xray
    if command -v xray >/dev/null 2>&1; then
        XRAY_BIN="$(command -v xray)"
        return 0
    fi

    echo -e "${CYAN}[*] Xray 未安装，正在自动安装...${NC}"

    # 检测架构
    local arch
    arch=$(uname -m)
    local xray_file
    case "$arch" in
        x86_64|amd64)    xray_file="Xray-linux-64.zip" ;;
        aarch64|arm64)   xray_file="Xray-linux-arm64-v8a.zip" ;;
        armv7l|armhf)    xray_file="Xray-linux-arm32-v7a.zip" ;;
        *)
            echo -e "${RED}  不支持的架构: $arch${NC}"
            echo "  请手动安装 Xray 到 ${XRAY_BIN} 或 PATH 中"
            exit 1
            ;;
    esac

    local download_url="https://github.com/XTLS/Xray-core/releases/latest/download/${xray_file}"
    local tmp_dir
    tmp_dir=$(mktemp -d)
    local tmp_zip="${tmp_dir}/${xray_file}"

    echo -e "  下载 ${xray_file}..."
    if ! curl -fSL --connect-timeout 15 --max-time 180 -o "$tmp_zip" "$download_url" 2>/dev/null; then
        echo -e "${RED}  下载 Xray 失败！请手动下载并安装:${NC}"
        echo "    下载: $download_url"
        echo "    解压 xray 二进制到: ${XRAY_BIN}"
        echo "    chmod +x ${XRAY_BIN}"
        rm -rf "$tmp_dir"
        exit 1
    fi

    # 解压 xray 二进制
    echo -e "  解压安装到 ${XRAY_BIN}..."
    mkdir -p "$(dirname "$XRAY_BIN")"

    if command -v unzip >/dev/null 2>&1; then
        unzip -o -q "$tmp_zip" xray -d "$tmp_dir"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "
import zipfile
with zipfile.ZipFile('${tmp_zip}') as z:
    z.extract('xray', '${tmp_dir}')
"
    else
        echo -e "${RED}  需要 unzip 或 python3 来解压，请先安装其中之一。${NC}"
        rm -rf "$tmp_dir"
        exit 1
    fi

    mv "${tmp_dir}/xray" "$XRAY_BIN"
    chmod +x "$XRAY_BIN"
    rm -rf "$tmp_dir"

    echo -e "${GREEN}  Xray 安装完成: ${XRAY_BIN}${NC}"
    "$XRAY_BIN" version 2>/dev/null | head -1 || true
}

# 生成 Xray 客户端配置（每次启动强制重写，确保与脚本配置同步）
ensure_xray_config() {
    local config_port="$LOCAL_PROXY_PORT"

    mkdir -p "$XRAY_CONFIG_DIR"

    # 根据协议类型生成不同的 outbound 配置
    local outbound_json=""
    if [ "$XRAY_PROTOCOL" = "vmess-ws" ]; then
        # VMess WebSocket（用于 香港Y10/香港Y11 等 ws 节点）
        outbound_json=$(cat << OUTBOUND
{
  "tag": "vmess-ws-out",
  "protocol": "vmess",
  "settings": {
    "vnext": [
      {
        "address": "${XRAY_WS_SERVER_IP}",
        "port": ${XRAY_WS_SERVER_PORT},
        "users": [
          {
            "id": "${XRAY_WS_UUID}",
            "alterId": ${XRAY_WS_ALTER_ID},
            "security": "auto"
          }
        ]
      }
    ]
  },
  "streamSettings": {
    "network": "${XRAY_WS_NETWORK}",
    "wsSettings": {
      "path": "${XRAY_WS_PATH}",
      "headers": {
        "Host": "${XRAY_WS_HOST}"
      }
    },
    "security": "none"
  }
}
OUTBOUND
)
    elif [ "$XRAY_PROTOCOL" = "ss" ]; then
        # Shadowsocks（用于 日本Y01 等 ss 节点）
        outbound_json=$(cat << OUTBOUND
{
  "tag": "ss-out",
  "protocol": "shadowsocks",
  "settings": {
    "servers": [
      {
        "address": "${XRAY_SS_SERVER_IP}",
        "port": ${XRAY_SS_SERVER_PORT},
        "method": "${XRAY_SS_CIPHER}",
        "password": "${XRAY_SS_PASSWORD}",
        "ota": false,
        "level": 0
      }
    ]
  },
  "streamSettings": {
    "network": "tcp",
    "security": "none"
  }
}
OUTBOUND
)
    else
        # VMess TCP（默认，用于 香港Y09 等 vmess tcp 节点）
        outbound_json=$(cat << OUTBOUND
{
  "tag": "vmess-out",
  "protocol": "vmess",
  "settings": {
    "vnext": [
      {
        "address": "${XRAY_SERVER_IP}",
        "port": ${XRAY_SERVER_PORT},
        "users": [
          {
            "id": "${XRAY_UUID}",
            "alterId": ${XRAY_ALTER_ID},
            "security": "${XRAY_SECURITY}"
          }
        ]
      }
    ]
  },
  "streamSettings": {
    "network": "${XRAY_NETWORK}",
    "security": "none",
    "tcpSettings": {
      "header": {
        "type": "none"
      }
    }
  }
}
OUTBOUND
)
    fi

    cat > "$XRAY_CONFIG" << XEOF
{
  "log": {
    "loglevel": "warning"
  },
  "inbounds": [
    {
      "tag": "http-in",
      "listen": "127.0.0.1",
      "port": ${config_port},
      "protocol": "http"
    }
  ],
  "outbounds": [
    ${outbound_json}
  ]
}
XEOF

    echo -e "${GREEN}  Xray 客户端配置已生成: ${XRAY_CONFIG}${NC}"
}

# ========================== 连接建立 ==========================

# 启动 Xray 客户端
setup_xray() {
    echo -e "${CYAN}[1/3] 启动 Xray 客户端 (${XRAY_PROTOCOL^^})...${NC}"

    ensure_xray

    local selection_lock
    selection_lock=$(proxy_port_select_lock_file)

    {
        flock -x 9

        # 串行化端口选择与启动过程，避免多窗口同时抢占同一端口
        if ensure_xray_port_ready; then
            return 0
        fi

        ensure_xray_config

        # 后台启动 Xray 客户端（setsid 完全脱离终端，防止退出时进程被 kill）
        setsid "$XRAY_BIN" run -c "$XRAY_CONFIG" </dev/null >> "$XRAY_LOG" 2>&1 &
        XRAY_PID=$!

        # 等待端口就绪（SS/WS 首次建连可能较慢，最多等 30 秒）
        local retries=0
        while [ $retries -lt 60 ]; do
            if ! kill -0 "$XRAY_PID" 2>/dev/null; then
                handle_xray_start_failure
                return 0
            fi
            if port_is_listening; then
                break
            fi
            sleep 0.5
            retries=$((retries + 1))
        done

        if [ $retries -ge 60 ]; then
            echo -e "${RED}  Xray 客户端启动超时！最近日志:${NC}"
            tail -5 "$XRAY_LOG" 2>/dev/null || true
            kill "$XRAY_PID" 2>/dev/null || true
            exit 1
        fi

        local xray_dest
        case "$XRAY_PROTOCOL" in
            ss)
                xray_dest="${XRAY_SS_SERVER_IP}:${XRAY_SS_SERVER_PORT}"
                ;;
            vmess-ws)
                xray_dest="${XRAY_WS_SERVER_IP}:${XRAY_WS_SERVER_PORT} (ws)"
                ;;
            *)
                xray_dest="${XRAY_SERVER_IP}:${XRAY_SERVER_PORT}"
                ;;
        esac
        echo -e "${GREEN}  Xray 客户端已启动 (PID: ${XRAY_PID}): 127.0.0.1:${LOCAL_PROXY_PORT} → ${xray_dest}${NC}"
    } 9>"$selection_lock"
}

# 验证代理连通性
check_proxy() {
    echo -e "${CYAN}[2/3] 验证代理连通性...${NC}"

    # 使用 HTTP 检测（TLS 握手在部分节点上不稳定，HTTP 更可靠）
    local ip
    ip=$(curl -s --max-time 15 --proxy "http://127.0.0.1:${LOCAL_PROXY_PORT}" "http://ip-api.com/line?fields=query" 2>/dev/null || true)

    if [ -z "$ip" ]; then
        echo -e "${RED}  代理连通性检查失败！${NC}"
        echo "  可能原因："
        echo "    1. Xray 客户端未正常启动（检查 PID: ${XRAY_PID:-无}）"
        echo "    2. 节点不可达或配置不匹配"
        echo "    3. 网络连接异常"
        echo "  排查命令："
        echo "    curl -v --proxy http://127.0.0.1:${LOCAL_PROXY_PORT} http://ip-api.com/line?fields=query"
        echo "    cat ${XRAY_LOG}"
        exit 1
    fi

    echo -e "${GREEN}  出口 IP: ${ip}${NC}"

    # 显示运营商信息
    local org
    org=$(curl -s --max-time 8 --proxy "http://127.0.0.1:${LOCAL_PROXY_PORT}" "http://ip-api.com/line?fields=org" 2>/dev/null || true)
    if [ -n "$org" ]; then
        echo -e "${GREEN}  运营商: ${org}${NC}"
    fi

    # 自动检测出口 IP 所在时区
    local country
    country=$(curl -s --max-time 8 --proxy "http://127.0.0.1:${LOCAL_PROXY_PORT}" "http://ip-api.com/line?fields=timezone" 2>/dev/null || true)
    if [ -n "$country" ]; then
        DETECTED_TZ="$country"
        echo -e "${GREEN}  时区: ${country}${NC}"
    fi

    # 额外验证 HTTPS（可选，用于确认 TLS 链路正常；失败不阻断）
    local https_ok=false
    if curl -s --max-time 10 --proxy "http://127.0.0.1:${LOCAL_PROXY_PORT}" \
           "http://www.google.com/generate_204" -o /dev/null 2>/dev/null; then
        https_ok=true
    fi
    if [ "$https_ok" = true ]; then
        echo -e "${GREEN}  HTTPS 代理: 正常${NC}"
    else
        echo -e "${YELLOW}  HTTPS 代理: 不稳定（HTTP 代理正常，Codex 仍可使用）${NC}"
    fi
}

# 显示 OAuth 登录说明
show_oauth_instructions() {
    echo ""
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  首次登录说明（已登录过可忽略）                              ║${NC}"
    echo -e "${YELLOW}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${YELLOW}║                                                              ║${NC}"
    echo -e "${YELLOW}║  Codex Code 启动后会显示 OAuth URL。                         ║${NC}"
    echo -e "${YELLOW}║                                                              ║${NC}"
    echo -e "${YELLOW}║  操作步骤：                                                   ║${NC}"
    echo -e "${YELLOW}║    1. 复制 OAuth URL                                          ║${NC}"
    echo -e "${YELLOW}║    2. 在本地电脑浏览器打开（如需代理，请使用同一节点）      ║${NC}"
    echo -e "${YELLOW}║    3. 完成网页登录                                            ║${NC}"
    echo -e "${YELLOW}║    4. 网页显示 Authentication Code → 复制                    ║${NC}"
    echo -e "${YELLOW}║    5. 回到此终端，粘贴 Code 并回车                           ║${NC}"
    echo -e "${YELLOW}║    6. 登录完成！                                              ║${NC}"
    echo -e "${YELLOW}║                                                              ║${NC}"
    echo -e "${YELLOW}║  ※ 支持 VMess/TCP · VMess/WS · Shadowsocks，修改顶部 XRAY_PROTOCOL 切换 ║${NC}"
    echo -e "${YELLOW}║                                                              ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ========================== Codex Code 启动 ==========================

# 代理环境变量（供 update_codex / start_codex 共用）
proxy_env() {
    HTTPS_PROXY="http://127.0.0.1:${LOCAL_PROXY_PORT}" \
    HTTP_PROXY="http://127.0.0.1:${LOCAL_PROXY_PORT}" \
    NO_PROXY="127.0.0.1,localhost" \
    no_proxy="127.0.0.1,localhost" \
    npm_config_registry="https://registry.npmjs.org" \
    "$@"
}

# 确保 npm 全局目录可写（用户级安装，无需 sudo）
ensure_npm_prefix() {
    # 检查当前 npm prefix 是否可写
    local npm_prefix
    npm_prefix=$(npm config get prefix 2>/dev/null || echo "/usr/local")
    if [ -w "$npm_prefix" ]; then
        return 0
    fi

    # prefix 不可写（如 /usr/local），设置为用户目录
    local user_prefix="${HOME}/.npm-global"
    echo -e "${YELLOW}  npm 全局目录 ${npm_prefix} 不可写，配置 prefix 到 ${user_prefix}${NC}"
    mkdir -p "$user_prefix"
    npm config set prefix "$user_prefix" 2>/dev/null

    # 确保 ~/.npm-global/bin 在 PATH 中（当前 session）
    if [[ ":$PATH:" != *":${user_prefix}/bin:"* ]]; then
        export PATH="${user_prefix}/bin:$PATH"
    fi

    # 持久化到 .bashrc（如果尚未添加）
    if ! grep -q 'npm-global/bin' "${HOME}/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "${HOME}/.bashrc"
        echo -e "${YELLOW}  已将 ${user_prefix}/bin 添加到 ~/.bashrc 的 PATH${NC}"
    fi
}

# 更新 Codex Code
update_codex() {
    echo -e "${CYAN}[*] 更新 Codex Code...${NC}"

    ensure_npm_prefix

    local current_ver=""
    current_ver=$(codex --version 2>/dev/null || echo "未安装")
    echo -e "  当前版本: ${current_ver}"

    echo -e "  通过代理安装最新版（npm 官方源）..."
    if proxy_env npm install -g @openai/codex@latest; then
        local new_ver=""
        new_ver=$(codex --version 2>/dev/null || echo "未知")
        echo -e "${GREEN}  更新完成: ${new_ver}${NC}"
    else
        echo -e "${RED}  更新失败！请检查网络或手动运行:${NC}"
        echo "    HTTPS_PROXY=http://127.0.0.1:${LOCAL_PROXY_PORT} npm install -g @openai/codex@latest"
        return 1
    fi
}

# 确保 Codex Code 已安装（新设备首次运行时自动安装）
ensure_codex() {
    if command -v codex >/dev/null 2>&1; then
        return 0
    fi

    echo -e "${YELLOW}[*] Codex Code 未安装，自动安装...${NC}"
    update_codex
}

# 启动 Codex Code
start_codex() {
    ensure_codex

    echo -e "${CYAN}[3/3] 启动 Codex Code...${NC}"
    echo ""

    # 使用自动检测的时区，回退到 America/Phoenix
    local tz="${DETECTED_TZ:-America/Phoenix}"

    TZ="$tz" \
    CODEX_TELEMETRY=disabled \
    DISABLE_TELEMETRY=1 \
    DISABLE_ERROR_REPORTING=1 \
    CODEX_CODE_DISABLE_FEEDBACK_SURVEY=1 \
    proxy_env codex "$@"
}

# ========================== 自注册到 PATH ==========================

# 首次运行时自动将 codex-start 注册到 PATH（后续可直接输入 codex-start 启动）
ensure_in_path() {
    # 如果已经在 PATH 中，跳过
    if command -v codex-start >/dev/null 2>&1; then
        return 0
    fi

    local script_path
    script_path="$(readlink -f "$0")"
    local bin_dir="${HOME}/.local/bin"
    local link_path="${bin_dir}/codex-start"

    mkdir -p "$bin_dir"

    # 创建符号链接（指向脚本实际位置）
    ln -sf "$script_path" "$link_path"

    # 确保 ~/.local/bin 在当前 session 的 PATH 中
    if [[ ":$PATH:" != *":${bin_dir}:"* ]]; then
        export PATH="${bin_dir}:$PATH"
    fi

    # 持久化到 .bashrc（如果尚未添加）
    if ! grep -q '\.local/bin' "${HOME}/.bashrc" 2>/dev/null; then
        echo '' >> "${HOME}/.bashrc"
        echo '# codex-start: ensure ~/.local/bin in PATH' >> "${HOME}/.bashrc"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"
    fi

    echo -e "${GREEN}[*] codex-start 已注册到 PATH，下次可直接运行: codex-start${NC}"
}

# ========================== 参数解析 ==========================

# --- 解析 -P/--port（多开时每窗口不同端口）---
REMAINING=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -P|--port)
            if [[ -n "${2:-}" && "$2" =~ ^[0-9]+$ ]]; then
                LOCAL_PROXY_PORT="$2"
                shift 2
            else
                echo -e "${RED}错误: -P/--port 需要端口号${NC}" >&2
                exit 1
            fi
            ;;
        --ssh|--tunnel-only)
            echo -e "${RED}错误: SSH 隧道模式已删除，当前脚本仅支持 Xray。${NC}" >&2
            exit 1
            ;;
        *)
            REMAINING+=("$1")
            shift
            ;;
    esac
done
set -- "${REMAINING[@]}"

# ========================== 主流程 ==========================

case "${1:-}" in
    --help|-h) show_help ;;
esac

ensure_in_path

case "${1:-}" in
    --check)
        setup_xray
        trap cleanup EXIT INT TERM
        proxy_ref_add
        check_proxy
        ;;
    --update)
        setup_xray
        trap cleanup EXIT INT TERM
        proxy_ref_add
        check_proxy
        update_codex || exit 1
        shift  # 移除 --update
        echo ""
        show_oauth_instructions
        start_codex "$@"
        ;;
    *)
        setup_xray
        trap cleanup EXIT INT TERM
        proxy_ref_add
        check_proxy
        show_oauth_instructions
        start_codex "$@"
        ;;
esac
