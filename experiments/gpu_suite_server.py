"""USB-reverse localhost-only test payload/result server, scoped to this suite."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from urllib.parse import urlparse,unquote
import json,sys
root=Path(__file__).resolve().parents[1]/'experiments/artifacts/gpu-suite'
results=root/'device-results';results.mkdir(exist_ok=True)
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(root),**kwargs)
    def do_GET(self):
        if self.path=='/suite.json':
            items=json.loads((root/'preparation.json').read_text());config=json.loads((root/'run-config.json').read_text()) if (root/'run-config.json').exists() else dict(run='run1')
            body=json.dumps([dict(name=x['name'],run=config['run'],status=x['status'],reason=x.get('reason',''),files=json.loads((root/x['name']/'files.json').read_text()) if x['status']=='PREPARED' else []) for x in items if not config.get('only') or x['name'] in config['only']]).encode();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        else:super().do_GET()
    def do_PUT(self):
        rel=unquote(urlparse(self.path).path).removeprefix('/upload/')
        if not self.path.startswith('/upload/') or '..' in Path(rel).parts or Path(rel).is_absolute():self.send_error(400);return
        dst=results/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        n=int(self.headers.get('Content-Length','0'))
        if n>256*1024*1024:self.send_error(413);return
        with dst.open('wb') as f:
            while n:
                chunk=self.rfile.read(min(n,1<<20))
                if not chunk:break
                f.write(chunk);n-=len(chunk)
        self.send_response(200 if n==0 else 400);self.send_header('Content-Length','0');self.end_headers()
        print('SAVED',rel,flush=True)
    def log_message(self,fmt,*args):
        if self.command=='PUT' or args and str(args[1])!='200':print(fmt%args,flush=True)
class Server(ThreadingHTTPServer):
    def server_bind(self):self.socket.bind(self.server_address);self.server_name='localhost';self.server_port=self.server_address[1]
print('USB suite server 127.0.0.1:8713',flush=True)
Server(('127.0.0.1',8713),Handler).serve_forever()
