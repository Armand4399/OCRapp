#!/bin/bash
echo ""
echo "=== Manuscript OCR Uninstaller ==="
echo ""
echo "This will remove:"
echo "  • /Applications/Manuscript OCR.app"
echo "  • ~/Library/Application Support/ManuscriptOCR/ (sessions, training data, models)"
echo ""
read -p "Are you sure? This cannot be undone. (y/N) " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Removing app..."
rm -rf "/Applications/Manuscript OCR.app" 2>/dev/null && echo "  ✓ App removed" || echo "  - App not found in /Applications"

read -p "Also remove all session data and models? (y/N) " remove_data
if [ "$remove_data" = "y" ] || [ "$remove_data" = "Y" ]; then
    rm -rf "$HOME/Library/Application Support/ManuscriptOCR" 2>/dev/null && echo "  ✓ Data removed" || echo "  - Data folder not found"
else
    echo "  - Data folder kept at ~/Library/Application Support/ManuscriptOCR/"
fi

echo ""
echo "Manuscript OCR has been uninstalled."
echo ""
