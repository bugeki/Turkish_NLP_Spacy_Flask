#!/usr/bin/env python
"""Test script to verify all imports work"""

import sys

print("Testing imports...")

try:
    print("1. Testing Flask...")
    from flask import Flask
    print("   ✓ Flask OK")
except Exception as e:
    print(f"   ✗ Flask FAILED: {e}")
    sys.exit(1)

try:
    print("2. Testing Flask-Bootstrap4...")
    from flask_bootstrap import Bootstrap4
    print("   ✓ Flask-Bootstrap4 OK")
except Exception as e:
    print(f"   ✗ Flask-Bootstrap4 FAILED: {e}")
    sys.exit(1)

try:
    print("3. Testing spaCy...")
    import spacy
    print("   ✓ spaCy OK")
except Exception as e:
    print(f"   ✗ spaCy FAILED: {e}")
    sys.exit(1)

try:
    print("4. Testing Turkish model...")
    nlp = spacy.load('tr_core_news_sm')
    print("   ✓ Turkish model OK")
except Exception as e:
    print(f"   ✗ Turkish model FAILED: {e}")
    sys.exit(1)

try:
    print("5. Testing TextBlob...")
    from textblob import TextBlob
    print("   ✓ TextBlob OK")
except Exception as e:
    print(f"   ✗ TextBlob FAILED: {e}")
    sys.exit(1)

try:
    print("6. Testing matplotlib...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("   ✓ matplotlib OK")
except Exception as e:
    print(f"   ✗ matplotlib FAILED: {e}")
    sys.exit(1)

try:
    print("7. Testing app import...")
    from app import app
    print("   ✓ app import OK")
except Exception as e:
    print(f"   ✗ app import FAILED: {e}")
    print(f"   Error details: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All imports successful!")
print("App should be ready to run.")
