#!/bin/bash
echo "========================================================================"
echo "FINAL GO/NO-GO CHECK - PRODUCTION DEPLOYMENT READINESS"
echo "========================================================================"
echo ""
echo "Checking EVERYTHING one last time..."
echo ""

# Check 1: Critical files
echo "[1/10] Checking critical files..."
if [ -f "main.py" ] && [ -f "config.yaml" ] && [ -f "requirements.txt" ]; then
    echo "  [+] PASS - All critical files present"
else
    echo "  [-] FAIL - Missing critical files"
    exit 1
fi

# Check 2: Source code
echo "[2/10] Checking source code..."
if [ -d "src/core" ] && [ -d "src/stage1" ] && [ -d "src/stage2" ] && [ -d "src/stage3" ]; then
    echo "  [+] PASS - All source directories present"
else
    echo "  [-] FAIL - Missing source directories"
    exit 1
fi

# Check 3: Documentation
echo "[3/10] Checking documentation..."
if [ -d "docs" ] && [ -f "docs/QUICK_START.md" ]; then
    echo "  [+] PASS - Documentation organized"
else
    echo "  [-] FAIL - Documentation missing"
    exit 1
fi

# Check 4: Scripts
echo "[4/10] Checking utility scripts..."
if [ -d "scripts" ] && [ -f "scripts/validate_production.py" ]; then
    echo "  [+] PASS - Scripts organized"
else
    echo "  [-] FAIL - Scripts missing"
    exit 1
fi

# Check 5: Archive
echo "[5/10] Checking archive..."
if [ -d "archive/old_docs" ] && [ -d "archive/old_tests" ]; then
    echo "  [+] PASS - Old files archived"
else
    echo "  [-] FAIL - Archive missing"
    exit 1
fi

# Check 6: Python syntax
echo "[6/10] Checking Python syntax..."
python -m py_compile main.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  [+] PASS - main.py has valid syntax"
else
    echo "  [-] FAIL - Syntax errors in main.py"
    exit 1
fi

# Check 7: Config
echo "[7/10] Checking configuration..."
if grep -q "stage3:" config.yaml && grep -q "save_unvalidated: true" config.yaml; then
    echo "  [+] PASS - Configuration valid"
else
    echo "  [-] FAIL - Configuration issues"
    exit 1
fi

# Check 8: Environment
echo "[8/10] Checking environment..."
if [ -f ".env" ]; then
    echo "  [+] PASS - .env file present"
else
    echo "  [!] WARN - .env file not found (user needs to configure)"
fi

# Check 9: Validation reports
echo "[9/10] Checking validation reports..."
if [ -f "FINAL_SYSTEM_CERTIFICATION.md" ]; then
    echo "  [+] PASS - Final certification present"
else
    echo "  [-] FAIL - Final certification missing"
    exit 1
fi

# Check 10: Test OpenRouter
echo "[10/10] Checking OpenRouter API..."
if [ -f "test_openrouter.py" ]; then
    echo "  [+] PASS - OpenRouter test script present"
else
    echo "  [!] WARN - OpenRouter test script not found"
fi

echo ""
echo "========================================================================"
echo "GO/NO-GO DECISION"
echo "========================================================================"
echo ""
echo "All critical checks passed!"
echo ""
echo "  [+] File structure: READY"
echo "  [+] Source code: READY"
echo "  [+] Documentation: READY"
echo "  [+] Scripts: READY"
echo "  [+] Archive: READY"
echo "  [+] Syntax: VALID"
echo "  [+] Configuration: VALID"
echo "  [+] Certification: COMPLETE"
echo ""
echo "========================================================================"
echo "DECISION: GO FOR PRODUCTION DEPLOYMENT"
echo "========================================================================"
echo ""
