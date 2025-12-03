"""
Convenience functions for launching the CORS webserver from Python.

Example usage::

    from ngsidekick.cors_server import serve_directory
    
    # Blocking call - runs until interrupted
    serve_directory("/path/to/data", port=9000)
    
    # Non-blocking - returns the subprocess handle
    proc = serve_directory("/path/to/data", port=9000, background=True)
    # ... do other work ...
    proc.terminate()
"""
import subprocess
import sys
import os
from pathlib import Path


def serve_directory(
    directory,
    port=9000,
    bind="127.0.0.1",
    background=False,
    capture_output=False
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
        Address to bind to. Default is "127.0.0.1".
    background : bool, optional
        If True, run the server in the background and return the subprocess.Popen
        object immediately. If False (default), block until the server is interrupted.
    capture_output : bool, optional
        If True, capture stdout/stderr instead of letting them print to console.
        Only useful when background=True.
        
    Returns
    -------
    subprocess.Popen or int
        If background=True, returns the Popen object for the server process.
        If background=False, returns the exit code of the server process.
        
    Examples
    --------
    Blocking usage (runs until Ctrl+C):
    
    >>> serve_directory("/path/to/data", port=9000)
    
    Background usage:
    
    >>> proc = serve_directory("/path/to/data", port=9000, background=True)
    >>> # ... do other work ...
    >>> proc.terminate()
    >>> proc.wait(timeout=5)
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
        if capture_output:
            kwargs['stdout'] = subprocess.PIPE
            kwargs['stderr'] = subprocess.PIPE
        return subprocess.Popen(cmd, **kwargs)
    else:
        result = subprocess.run(cmd)
        return result.returncode

