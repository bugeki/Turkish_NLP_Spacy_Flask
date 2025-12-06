import sys
import os

print(f"Starting WSGI - Python {sys.version}", file=sys.stderr)
print(f"PORT: {os.environ.get('PORT', 'NOT SET')}", file=sys.stderr)
print(f"Working directory: {os.getcwd()}", file=sys.stderr)

try:
    from app import app
    print("App imported successfully", file=sys.stderr)
except Exception as e:
    print(f"ERROR importing app: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    app.run()