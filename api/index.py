from app import create_app

app = create_app()

# Vercel serverless functions automatically expect a WSGI/ASGI app variable named `app`
