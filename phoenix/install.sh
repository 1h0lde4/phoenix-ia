#!/bin/bash
set -e

echo "🔥 Phoenix AI Installer"
echo "========================"

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed. Aborting."
    exit 1
fi
echo "✅ Python3 found."

# 2. Install Ollama if missing
if ! command -v ollama &> /dev/null; then
    echo "⚙️  Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed."
else
    echo "✅ Ollama already installed."
fi

# 3. Pull required models (adjust as needed)
echo "📥 Pulling LLM and embedding models (this may take a while)..."
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 4. Set up Python virtual environment
cd "$(dirname "$0")"   # assume install.sh is in ~/.phoenix/
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. (Optional) Install as systemd service
read -p "🔧 Do you want to install Phoenix as a systemd service (autostart on boot)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SERVICE_FILE="/etc/systemd/system/phoenix.service"
    sudo tee "$SERVICE_FILE" > /dev/null << SYSTEMD_EOF
[Unit]
Description=Phoenix AI API Server
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${HOME}/.phoenix
ExecStart=${HOME}/.phoenix/venv/bin/python api_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF

    sudo systemctl daemon-reload
    sudo systemctl enable phoenix
    echo "✅ Phoenix service installed and enabled. Start it with: sudo systemctl start phoenix"
fi

echo ""
echo "========================"
echo "✅ Phoenix installation complete!"
echo "Start manually: cd ~/.phoenix && source venv/bin/activate && python api_server.py"
echo "Then use: http://localhost:8000/v1/chat/completions as your local ChatGPT replacement."
