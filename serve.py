"""Tiny static server WITH HTTP Range support (python's http.server lacks it).

Range requests are required for <video> seeking/streaming — without them the
browser can't seek and long videos appear to loop a tiny clip.

    python3 serve.py [port]      # default 8080
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    _rng = None  # default so copyfile never hits AttributeError (e.g. directory/index)

    def send_head(self):
        self._rng = None
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(f.fileno()).st_size
        ctype = self.guess_type(path)
        rng = self.headers.get("Range")
        self._rng = None
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                start = int(m.group(1)) if m.group(1) else 0
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    start = 0
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                self._rng = (start, length)
                return f
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        if self._rng:
            start, remaining = self._rng
            source.seek(start)
            while remaining > 0:
                chunk = source.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    outputfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)
        else:
            super().copyfile(source, outputfile)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Serving with Range support on http://127.0.0.1:{port}/")
    ThreadingHTTPServer(("127.0.0.1", port), RangeHandler).serve_forever()
