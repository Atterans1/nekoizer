"""Outbound link checks for the places where Nekoizer fetches a file the user names.

Only https links to public hosts are followed. Every redirect hop is checked the same way, and the connection itself dials an address
that its own name lookup proved public, so a later DNS answer cannot send it elsewhere (DNS rebinding). Proxies are not used: a proxy
resolves and dials on its own, so the checks could not hold. A download stops at a byte cap or a time budget. Nothing is kept between
calls: each fetch builds its own opener.
"""
from __future__ import annotations

import contextlib
import http.client
import ipaddress
import pathlib
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------- PUBLIC
class GuardError(ValueError):
    """A link or download was refused. status is the HTTP status to answer with; str(exc) is written for the user."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def IsPublicAddress(address: str) -> bool:
    """True only for a globally routable address. An IPv4-mapped IPv6 address is judged by its IPv4 part."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_global and not ip.is_multicast and not ip.is_unspecified and not ip.is_loopback)


def ResolvePublic(hostname: str, port: int = 443) -> list[str]:
    """Resolve hostname to its addresses. Refuses the whole name when any one address is not public."""
    if not hostname:
        raise GuardError(400, "the link has no host name")
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except UnicodeError as exc:
        raise GuardError(400, "the link has a bad host name") from exc
    except OSError as exc:
        raise GuardError(502, f"could not resolve {hostname}: {exc}") from exc
    addresses = list(dict.fromkeys(info[4][0] for info in infos))
    if not addresses:
        raise GuardError(502, f"could not resolve {hostname}: no address")
    if any(not IsPublicAddress(address) for address in addresses):
        raise GuardError(400, "the link points at a private or local address")
    return addresses


def CheckUrl(url: str) -> None:
    """Refuse anything but an https link to a public host, and refuse hosts that need a confirm step Nekoizer cannot follow."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port or 443
    except ValueError as exc:
        raise GuardError(400, "the link is not a valid URL") from exc
    if parts.scheme != "https":
        raise GuardError(400, "the link must start with https://")
    host = (parts.hostname or "").lower()
    if not host:
        raise GuardError(400, "the link has no host name")
    if any(host == drive or host.endswith("." + drive) for drive in _DRIVE_HOSTS):
        raise GuardError(400, "Google Drive and Mega links need a confirm step that Nekoizer cannot follow: download the file yourself and drop it on a slot")
    ResolvePublic(host, port)


def FetchToFile(url: str, dest: pathlib.Path, max_bytes: int, timeout_s: float = 30.0, total_s: float = 180.0, max_redirects: int = 5) -> dict:
    """Download url into dest through checked hops. Any refusal or failure deletes dest and raises GuardError."""
    dest = pathlib.Path(dest)
    try:
        with dest.open("wb") as out:
            return _Fetch(url, out.write, max_bytes, timeout_s, total_s, max_redirects)
    except BaseException:
        _Discard(dest)
        raise


def FetchBytes(url: str, max_bytes: int, timeout_s: float = 20.0, total_s: float = 60.0, max_redirects: int = 5) -> tuple[bytes, dict]:
    """Download url into memory through the same checked hops. Any refusal or failure raises GuardError."""
    buffer = bytearray()
    info = _Fetch(url, buffer.extend, max_bytes, timeout_s, total_s, max_redirects)
    return bytes(buffer), info


# ---------------------------------------------------------------- PRIVATE
_CHUNK = 1024 * 1024
_HEAD = 512
_DRIVE_HOSTS = ("drive.google.com", "docs.google.com", "drive.usercontent.google.com", "mega.nz", "mega.io")


def _Fetch(url: str, sink, max_bytes: int, timeout_s: float, total_s: float, max_redirects: int) -> dict:
    """Stream url through checked hops into sink(chunk), stopping at max_bytes or total_s."""
    CheckUrl(url)
    deadline = time.monotonic() + total_s
    request = urllib.request.Request(url, headers={"User-Agent": "Nekoizer"})
    size = 0
    head = b""
    try:
        with _Opener(max_redirects).open(request, timeout=timeout_s) as response:
            code = response.status
            if not 200 <= code < 300:
                raise GuardError(502, f"the link answered HTTP {code}")
            content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            while True:
                if time.monotonic() > deadline:
                    raise GuardError(504, "the download took too long")
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise GuardError(413, f"the file is larger than {max_bytes // 2**20} MB")
                if len(head) < _HEAD:
                    head += chunk[: _HEAD - len(head)]
                sink(chunk)
            final_url = response.geturl()
    except GuardError:
        raise
    except urllib.error.HTTPError as exc:  # an HTTPError is also an OSError, so it is caught first
        raise GuardError(502, f"the link answered HTTP {exc.code}") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise GuardError(502, f"could not download the link: {exc}") from exc
    except ValueError as exc:  # urllib refuses, for example, control characters in a URL
        raise GuardError(400, "the link is not a valid URL") from exc
    return {"final_url": final_url, "content_type": content_type, "size": size, "head": head}


def _DialPublic(host: str, port: int, timeout, source_address) -> socket.socket:
    """Connect to an address of host that ResolvePublic returned from this very lookup. The name is never resolved a second time."""
    error: OSError | None = None
    for address in ResolvePublic(host, port):
        try:
            return socket.create_connection((address, port), timeout, source_address)
        except OSError as exc:
            error = exc
    raise error


class _GuardedHTTPSConnection(http.client.HTTPSConnection):
    """An HTTPS connection that dials only the address _DialPublic validated. TLS still checks the original host name."""

    def connect(self) -> None:
        self.sock = _DialPublic(self.host, self.port, self.timeout, self.source_address)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


class _GuardedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_GuardedHTTPSConnection, req, context=self._context)


def _Opener(max_redirects: int) -> urllib.request.OpenerDirector:
    """An opener whose redirect handler runs CheckUrl on every hop and stops after max_redirects hops, whose connections dial
    validated addresses only, and which ignores proxy settings."""

    class Redirects(urllib.request.HTTPRedirectHandler):
        def __init__(self) -> None:
            super().__init__()
            self.hops = 0

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            self.hops += 1
            if self.hops > max_redirects:
                raise GuardError(400, "too many redirects")
            CheckUrl(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    return urllib.request.build_opener(Redirects(), _GuardedHTTPSHandler(), urllib.request.ProxyHandler({}))


def _Discard(path: pathlib.Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
