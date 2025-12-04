"""
Convenience functions for launching the CORS webserver from Python.

Example usage::

    from ngsidekick.cors_server import serve_directory
    
    # Blocking call - runs until interrupted
    serve_directory("/path/to/data", port=9000)
    
    # Non-blocking - returns the subprocess handle and URL
    server = serve_directory("/path/to/data", port=9000, background=True)
    print(f"Server running at {server.url}")
    # ... do other work ...
    server.process.terminate()
"""
import subprocess
import sys
import socket
from pathlib import Path
from collections import namedtuple

ServerInfo = namedtuple('ServerInfo', ['process', 'url', 'log_file'])
ServerInfo.__new__.__defaults__ = (None,)  # log_file defaults to None


def serve_directory(
    directory,
    port=9000,
    bind="0.0.0.0",
    background=False,
    capture_output=False,
    log_to_file=False
):
    """
    Serve files from a directory with CORS support for Neuroglancer.
    
    This launches the cors-webserver script as a subprocess.
    
    Parameters
    ----------
    directory : str or Path
        Directory to serve files from.
    port : int, optional
        TCP port to listen on. Default is 9000.
    bind : str, optional
        Address to bind to. Default is "0.0.0.0" (all interfaces).
    background : bool, optional
        If True, run the server in the background and return the subprocess.Popen
        object immediately. If False (default), block until the server is interrupted.
    capture_output : bool, optional
        If True, capture stdout/stderr instead of letting them print to console.
        Only useful when background=True.
    log_to_file : bool, optional
        If True, redirect all server output to a file in /tmp/.
        The file path will be available in the returned ServerInfo.log_file.
        Only useful when background=True.
        
    Returns
    -------
    ServerInfo or int
        If background=True, returns a ServerInfo namedtuple with 'process' (Popen),
        'url' (str), and 'log_file' (str or None) attributes.
        If background=False, returns the exit code of the server process.
        
    Examples
    --------
    Blocking usage (runs until Ctrl+C):
    
    >>> serve_directory("/path/to/data", port=9000)
    
    Background usage:
    
    >>> server = serve_directory("/path/to/data", port=9000, background=True)
    >>> print(f"Server running at {server.url}")
    >>> # ... do other work ...
    >>> server.process.terminate()
    >>> server.process.wait(timeout=5)
    """
    directory = Path(directory).resolve()
    
    cmd = [
        sys.executable, "-m", "ngsidekick.bin.cors_webserver",
        "--port", str(port),
        "--bind", bind,
        "--directory", str(directory)
    ]
    
    if background:
        kwargs = {}
        log_file_path = None
        log_file_handle = None
        
        if log_to_file:
            log_file_path = f"/tmp/cors_server_{port}.log"
            log_file_handle = open(log_file_path, 'w')
            kwargs['stdout'] = log_file_handle
            kwargs['stderr'] = log_file_handle
        elif capture_output:
            kwargs['stdout'] = subprocess.PIPE
            kwargs['stderr'] = subprocess.PIPE
        
        proc = subprocess.Popen(cmd, **kwargs)
        url = f"http://{_get_local_ip()}:{port}"
        return ServerInfo(proc, url, log_file_path)
    else:
        result = subprocess.run(cmd)
        return result.returncode


def _get_local_ip():
    """Get the local IP address for this machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
    except OSError:
        return '127.0.0.1'

