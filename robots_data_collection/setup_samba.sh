#!/usr/bin/env bash
set -e

PC_USER=$(whoami)
DATA_DIR="/data"

echo "[1/8] 更新 apt 缓存并安装必要工具"
sudo apt update
sudo apt install -y samba smbclient acl

echo "[2/8] 创建数据目录"
sudo mkdir -p "$DATA_DIR"

echo "[3/8] 设置目录基础权限（setgid + 可写）"
sudo chown root:"$PC_USER" "$DATA_DIR"
sudo chmod 2775 "$DATA_DIR"

echo "[4/8] 修复已有目录权限（立即生效）"
sudo setfacl -R -m g::rwx "$DATA_DIR"

echo "[5/8] 设置默认 ACL（新建文件/目录自动继承权限）"
sudo setfacl -d -m g::rwx "$DATA_DIR"

echo "[6/8] 配置 Samba 用户及密码"
# 创建或更新 Samba 用户，统一密码 qlzn1234
echo -e "qlzn1234\nqlzn1234" | sudo smbpasswd -s -a "$PC_USER"

echo "[7/8] 配置 Samba 共享"
SAMBA_CONF="/etc/samba/smb.conf"
# 备份原配置
sudo cp "$SAMBA_CONF" "${SAMBA_CONF}.bak"

# 写入 [data] 共享（如果已经有同名共享可覆盖）
if grep -q "^\[data\]" "$SAMBA_CONF"; then
    echo "[INFO] /data 共享已存在，覆盖配置"
    sudo sed -i '/^\[data\]/,/^\[/d' "$SAMBA_CONF"
fi

sudo tee -a "$SAMBA_CONF" > /dev/null <<EOF

[data]
   path = $DATA_DIR
   browseable = yes
   writable = yes
   valid users = $PC_USER
   force user = $PC_USER
   force group = $PC_USER
   create mask = 0664
   directory mask = 2775
EOF

echo "[8/8] 重启 Samba 服务"
sudo systemctl restart smbd nmbd
sudo systemctl status smbd --no-pager

echo "✅ /data 已配置 Samba 共享，当前用户 $PC_USER 可读写，ACL 永久生效"
