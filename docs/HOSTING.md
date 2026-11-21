# Hosting atterans.shop from this PC

Everything is served from `C:\Users\jeffr\nekoizer-host\`. Cloudflare provides DNS, TLS and DDoS protection and reaches this PC through an outbound tunnel, so no router ports are open and the home IP is not published.

| Piece | What | Where |
|---|---|---|
| Web server | `serve.py` (FastAPI, 127.0.0.1:8790 only): the site at /, installer files at /downloads, clean URLs, range downloads, security headers, path traversal blocked | `nekoizer-host\serve.py` |
| Tunnel | `cloudflared` 2026.11.1 (SHA-256 verified), tunnel `nekoizer`, routes atterans.shop and www.atterans.shop to the server | `nekoizer-host\config.yml`, credentials in `%USERPROFILE%\.cloudflared` |
| Autostart | `Nekoizer-host.cmd` in your user Startup folder starts both when you sign in to Windows | `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup` |
| Files | `downloads\Nekoizer-Setup-1.0.0.exe` (1.1 MB), `Nekoizer-A-1.0.0.msi`, `Nekoizer-B-1.0.0.msi` (hardlinks to `dist_msi`) | `nekoizer-host\downloads` |

The setup exe downloads the two MSI parts from `https://atterans.shop/downloads/`.

## Measured on 2026-11-20

Through https://atterans.shop with Cloudflare in front: pages and images load in about 0.1 s, the setup exe (1.1 MB) is byte-identical to the local file, ranged MSI downloads work (206), and http:// redirects to https:// (301). Throughput for the large files is the problem: 1 MB samples of the MSI ran at 1.7 to 2.3 Mbit/s, which is this connection's upload speed. At 2 Mbit/s, the two MSI parts (2.3 GB) take about 2.5 hours for one user, and users share that bandwidth. Do not point users at the PC for the MSI parts unless the upload speed is much higher.

## Limits you should know about

- The site and downloads are up only while this PC is on, signed in, and online. When it is off, atterans.shop shows a Cloudflare error.
- Every download uses your home upload speed: the installer is 2.3 GB per user. At 20 Mbit/s up, that is about 16 minutes per user, and several users at once share it.
- Cloudflare's self-serve terms restrict serving mostly large non-web files (video, big downloads) through its proxy on the free plan. A 2.3 GB installer is that kind of file. If Cloudflare objects, move the two MSI files to R2 or another file host and rebuild the setup exe (seconds) with the new address.
- DDoS protection: Cloudflare's automatic L3/L4/L7 mitigation is on for proxied hostnames on the free plan. It does not protect the PC's own address if it leaks, and the tunnel keeps it unpublished.

## Operating it

- Restart the server: run `nekoizer-host\start_site.ps1`. Restart the tunnel: `start_tunnel.ps1`. Logs: `tunnel.log`.
- Update the site: copy the new `site` files into `nekoizer-host\site`. No restart is needed.
- Publish a new version: rebuild with `installer\build_installer.ps1 -BaseUrl https://atterans.shop/downloads`, copy the three files into `nekoizer-host\downloads`, and update `version` and the file names in `site\config.js`.
- The Cloudflare Pages project `nekoizer-site` (pages.dev) is an unused backup copy of the site.
