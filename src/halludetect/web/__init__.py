"""The web page served at `/`: a browser client for `/v1/verify`.

Deploys with the API (including on Vercel, which can't host the Streamlit
UI). Presentation only - it calls the same public endpoint as any other
caller and never sees a server-side key: visitors enter their own access
key, kept in their browser.
"""
