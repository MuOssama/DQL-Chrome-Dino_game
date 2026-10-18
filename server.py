"""python server.py  -> open http://localhost:8000"""
import http.server, socketserver, os, webbrowser, functools
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"))
with socketserver.TCPServer(("", 8000), http.server.SimpleHTTPRequestHandler) as s:
    print("Dino at http://localhost:8000"); webbrowser.open("http://localhost:8000"); s.serve_forever()
