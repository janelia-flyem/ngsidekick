"""
Basic convenience functions for downloading and parsing neuroglancer states.
"""
import re
import json
import urllib
import logging

logger = logging.getLogger(__name__)


def parse_nglink(link):
    """
    Given a neuroglancer link, return the corresponding JSON state.

    Arg:
        link: str
            A neuroglancer link such as:
            https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v0.9/male-cns-v0.9.json

            Also works with fully expanded neuroglancer links.

    Returns:
        dict
    """
    _, pseudo_json = link.split('#!')
    if pseudo_json.endswith('.json'):
        return download_ngstate(pseudo_json)
    pseudo_json = urllib.parse.unquote(pseudo_json)
    data = json.loads(pseudo_json)
    return data


def download_ngstate(link):
    """
    Given a neuroglancer link which references a JSON state file,
    download the file and return the enclosed JSON data.

    Arg:
        link: str
            A neuroglancer link such as:
            https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v0.9/male-cns-v0.9.json

    Returns:
        dict
    """
    import requests
    if link.startswith('gs://'):
        url = f'https://storage.googleapis.com/{link[len("gs://"):]}'
        return requests.get(url, timeout=10).json()

    if not link.startswith('http'):
        raise ValueError(f"Don't understand state link: {link}")

    if link.count('://') == 1:
        return requests.get(link, timeout=10).json()

    if link.count('://') == 2:
        url = f'https://storage.googleapis.com/{link.split("://")[2]}'
        return requests.get(url, timeout=10).json()

    raise ValueError(f"Don't understand state link: {link}")


def encode_ngstate(ng_server, link_json_settings):
    """
    Produce a fully expanded URL from a neuroglancer server and a JSON state.
    """
    return ng_server + '/#!' + urllib.parse.quote(json.dumps(link_json_settings))


def layer_dict(state):
    """
    Given a neuroglancer JSON state, return a dictionary of the layer states, keyed by layer name.
    """
    return {layer['name']: layer for layer in state['layers']}


def layer_state(state, name_pattern):
    """
    Given a neuroglancer JSON state and a regex pattern,
    return the layer state whose name matches the pattern.
    If more than one layer matches the pattern, raise an error.
    """
    layer = None
    matches = []
    for layer in state['layers']:
        if re.match(name_pattern, layer['name']):
            matches.append(layer)
    if len(matches) > 1:
        matched_names = [match['name'] for match in matches]
        raise RuntimeError(f"Found more than one layer matching to the regex '{name_pattern}': {matched_names}")
    return matches[0]
