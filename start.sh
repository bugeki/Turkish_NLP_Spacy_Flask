#!/bin/bash
set -e

echo "========================================="
echo "Starting Turkish NLP App"
echo "========================================="

echo "Python version:"
python --version

echo ""
echo "Checking if wsgi.py exists..."
ls -la wsgi.py

echo ""
echo "Checking if app.py exists..."
ls -la app.py

echo ""
echo "Testing Python import..."
python -c "import wsgi; print('wsgi import: OK')"

echo ""
echo "Testing app loading..."
python -c "from wsgi import app; print('app loaded: OK'); print('App name:', app.name)"

echo ""
echo "Starting Gunicorn..."
exec gunicorn --bind 0.0.0.0:5000 \
    --workers 1 \
    --timeout 120 \
    --log-level debug \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance \
    wsgi:app
