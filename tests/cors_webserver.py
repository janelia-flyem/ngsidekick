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
import mimetypes
from flask import Flask, send_from_directory, after_this_request, request
from werkzeug.exceptions import NotFound


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
        
        try:
            # send_from_directory automatically handles Range requests
            return send_from_directory(directory, path if path else '.')
        except NotFound:
            # Try serving as index
            if not path or path.endswith('/'):
                return send_from_directory(directory, 'index.html')
            raise
    
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Serve local files with CORS and Range request support for Neuroglancer"
    )
    ap.add_argument(
        "-p", "--port", type=int, default=9000, help="TCP port to listen on"
    )
    ap.add_argument("-a", "--bind", default="127.0.0.1", help="Bind address")
    ap.add_argument("-d", "--directory", default=".", help="Directory to serve")

    args = ap.parse_args()
    
    directory = os.path.abspath(args.directory)
    print(f"Serving directory {directory} at http://{args.bind}:{args.port}")
    
    app = create_app(directory)
    
    try:
        app.run(host=args.bind, port=args.port, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
