#!/usr/bin/env python
"""
Simple CORS-enabled web server for serving local files to Neuroglancer.

This server supports HTTP Range requests, which are required for
neuroglancer's sharded precomputed format.

WARNING: Because this web server permits cross-origin requests, it exposes any
data in the directory that is served to any web page running on a machine that
can connect to the web server.
"""

import argparse
import os
import sys
import socket
import mimetypes
from pathlib import Path
from flask import Flask, send_from_directory, after_this_request, request, Response
from werkzeug.exceptions import NotFound


def get_local_ip_addresses():
    """Get all local IP addresses for this machine."""
    ip_addresses = []
    try:
        # Get the hostname and try to resolve it
        hostname = socket.gethostname()
        # Get all addresses associated with the hostname
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ip_addresses and not ip.startswith('127.'):
                ip_addresses.append(ip)
    except socket.gaierror:
        pass
    
    # Also try connecting to an external address to find the default route IP
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            if ip not in ip_addresses:
                ip_addresses.insert(0, ip)
    except OSError:
        pass
    
    return ip_addresses if ip_addresses else ['127.0.0.1']


def generate_directory_listing(dir_path, url_path):
    """Generate an HTML directory listing for the given directory."""
    entries = []
    
    # Add parent directory link if not at root
    if url_path:
        parent = str(Path(url_path).parent)
        if parent == '.':
            parent = ''
        entries.append(f'<li><a href="/{parent}">../</a></li>')
    
    # List directory contents
    try:
        items = sorted(Path(dir_path).iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        for item in items:
            name = item.name
            if item.is_dir():
                name += '/'
            href = f"/{url_path}/{name}" if url_path else f"/{name}"
            entries.append(f'<li><a href="{href}">{name}</a></li>')
    except PermissionError:
        entries.append('<li>Permission denied</li>')
    
    display_path = f"/{url_path}" if url_path else "/"
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Directory listing for {display_path}</title>
    <style>
        body {{ font-family: monospace; margin: 2em; }}
        h1 {{ font-size: 1.2em; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ padding: 0.2em 0; }}
        a {{ text-decoration: none; color: #0066cc; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Directory listing for {display_path}</h1>
    <hr>
    <ul>
        {''.join(entries)}
    </ul>
    <hr>
</body>
</html>"""
    return html


def create_app(directory):
    """Create a Flask app that serves files from the given directory."""
    app = Flask(__name__)
    
    # Add custom MIME types for neuroglancer
    mimetypes.add_type('application/json', '')  # Files without extension
    mimetypes.add_type('application/octet-stream', '.shard')
    
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_file(path):
        """Serve files with CORS headers. Flask automatically handles Range requests."""
        # Log the request
        range_header = request.headers.get('Range', 'no range')
        print(f"Request: {path} [{range_header}]", file=sys.stderr)
        
        @after_this_request
        def add_cors_headers(response):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Range'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Content-Range, Accept-Ranges'
            print(f"Response: {response.status_code} for {path}", file=sys.stderr)
            return response
        
        # Build full path and check if it's a directory
        full_path = Path(directory) / path if path else Path(directory)
        
        if full_path.is_dir():
            # Return directory listing
            html = generate_directory_listing(full_path, path)
            return Response(html, mimetype='text/html')
        
        try:
            # send_from_directory automatically handles Range requests
            return send_from_directory(directory, path if path else '.')
        except NotFound:
            raise
    
    return app


def main():
    """Entry point for the cors-webserver command."""
    ap = argparse.ArgumentParser(
        description="Serve local files with CORS and Range request support for Neuroglancer"
    )
    ap.add_argument(
        "-p", "--port", type=int, default=9000, help="TCP port to listen on"
    )
    ap.add_argument("-a", "--bind", default="0.0.0.0", help="Bind address")
    ap.add_argument("-d", "--directory", default=".", help="Directory to serve")

    args = ap.parse_args()
    
    directory = os.path.abspath(args.directory)
    print(f"Serving directory: {directory}")
    
    if args.bind == '0.0.0.0':
        # Show all available addresses
        print("Server available at:")
        print(f"  http://localhost:{args.port}")
        for ip in get_local_ip_addresses():
            print(f"  http://{ip}:{args.port}")
    else:
        print(f"Server available at: http://{args.bind}:{args.port}")
    
    app = create_app(directory)
    
    try:
        app.run(host=args.bind, port=args.port, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)


if __name__ == "__main__":
    main()

