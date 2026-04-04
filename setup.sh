#!/bin/bash

echo "========================================"
echo "Orange Measurement App - Setup Script"
echo "========================================"
echo ""

echo "Step 1: Checking Python version..."
python3 --version || { echo "Error: Python 3 not found"; exit 1; }
echo "✓ Python 3 found"
echo ""

echo "Step 2: Installing dependencies..."
if pip3 install -r requirements.txt --user --quiet; then
    echo "✓ Dependencies installed"
else
    echo "! Warning: Some dependencies may have failed to install"
    echo "  This is normal in some environments (Replit, etc.)"
    echo "  The system may already have required packages."
fi
echo ""

echo "Step 3: Checking bcrypt installation..."
if python3 -c "import bcrypt" 2>/dev/null; then
    echo "✓ bcrypt is available"
else
    echo "✗ bcrypt is NOT installed"
    echo ""
    echo "IMPORTANT: bcrypt is required for secure password hashing."
    echo "Please install it manually:"
    echo "  pip install bcrypt --user"
    echo "  OR"
    echo "  pip3 install bcrypt --user"
    echo ""
fi

echo "Step 4: Checking other dependencies..."
python3 -c "
import sys
missing = []
for module in ['gradio', 'numpy', 'cv2', 'open3d', 'PIL', 'pandas', 'scipy']:
    try:
        __import__(module)
    except ImportError:
        missing.append(module)

if missing:
    print(f'✗ Missing: {', '.join(missing)}')
    sys.exit(1)
else:
    print('✓ All core dependencies available')
" || { echo "Error: Missing dependencies"; exit 1; }
echo ""

echo "Step 5: Verifying module structure..."
if python3 -c "import config; import utils; import db" 2>/dev/null; then
    echo "✓ All modules can be imported"
else
    echo "✗ Module import failed"
    exit 1
fi
echo ""

echo "Step 6: Creating necessary directories..."
mkdir -p uploads
echo "✓ uploads/ directory ready"
echo ""

echo "Step 7: Checking configuration..."
if [ -f "config.py" ]; then
    echo "✓ config.py found"
else
    echo "✗ config.py not found"
    exit 1
fi
echo ""

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. If bcrypt is missing, install it:"
echo "   pip install bcrypt --user"
echo ""
echo "2. (Optional) Set admin password:"
echo "   export ADMIN_PASSWORD='your-secure-password'"
echo ""
echo "3. Run the application:"
echo "   python3 app.py"
echo ""
echo "4. Open your browser to:"
echo "   http://localhost:5000"
echo ""
echo "For detailed documentation, see:"
echo "  - README.md (user guide)"
echo "  - UPGRADE_GUIDE.md (technical details)"
echo "  - CHANGES.md (what's new)"
echo ""
echo "========================================"
