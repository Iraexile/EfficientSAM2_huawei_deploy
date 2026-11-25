import gpu from 'libgpu_probe.so';
import { common } from '@kit.AbilityKit';
import { fileIo as fs } from '@kit.CoreFileKit';
import { http } from '@kit.NetworkKit';
import { taskpool } from '@kit.ArkTS';
interface SuiteJob { name: string; run: string; status: string; reason: string; files: string[]; }
@Concurrent
function runGraph(path: string): string { return gpu.run(path); }
async function transfer(url: string, method: http.RequestMethod, data?: ArrayBuffer): Promise<ArrayBuffer> {
  const req = http.createHttp();
  try {
    const response = await req.request(url, { method: method, extraData: data && data.byteLength>0 ? data : undefined, expectDataType: http.HttpDataType.ARRAY_BUFFER, maxLimit: 100*1024*1024, connectTimeout: 30000, readTimeout: 300000 });
    if (response.responseCode !== 200) { throw new Error('HTTP '+response.responseCode+' '+url); }
    return response.result as ArrayBuffer;
  } finally { req.destroy(); }
}
@Entry
@Component
struct GpuSuite {
  @State result: string = 'Preparing GPU suite...';
  async execute(): Promise<void> {
    const context = getContext(this) as common.UIAbilityContext;
    const base = context.filesDir + '/gpu-suite';
    if (!fs.accessSync(base)) { fs.mkdirSync(base); }
    const bundled = await context.resourceManager.getRawFileList('');
    if (bundled.some((name: string) => name.startsWith('suite_memenc_'))) {
      for (const graph of ['memenc_binarize', 'memenc_sigmoid']) {
        const dir=base+'/'+graph;
        if (!fs.accessSync(dir)) { fs.mkdirSync(dir); }
        for (const entry of bundled.filter((name: string) => name.startsWith('suite_'+graph+'__'))) {
          const fileName=entry.substring(('suite_'+graph+'__').length);
          const bytes=await context.resourceManager.getRawFileContent(entry);
          const f=fs.openSync(dir+'/'+fileName,fs.OpenMode.CREATE|fs.OpenMode.WRITE_ONLY|fs.OpenMode.TRUNC);
          fs.writeSync(f.fd,bytes.buffer);fs.closeSync(f);
        }
        if (fs.accessSync(dir+'/retry2-graph-probe.txt')) { continue; }
        this.result='Offline GPU run '+graph;
        const output=await taskpool.execute(runGraph,dir+'/retry2-graph-probe.txt') as string;
        this.result=graph+'\n'+output.slice(-1600);
        return;
      }
      return;
    }
    let raw: ArrayBuffer;
    try { raw = await transfer('http://127.0.0.1:8713/suite.json', http.RequestMethod.GET); }
    catch (_error) {
      for (const name of ['memenc_binarize', 'memenc_sigmoid']) {
        const dir=base+'/'+name;
        if (!fs.accessSync(dir+'/model.param')) { continue; }
        this.result='Offline GPU run '+name;
        const output=await taskpool.execute(runGraph,dir+'/graph-probe.txt') as string;
        this.result=name+'\n'+output.slice(-1600);
      }
      return;
    }
    const json = String.fromCharCode(...new Uint8Array(raw));
    const jobs = JSON.parse(json) as SuiteJob[];
    for (const job of jobs) {
      this.result = 'Running '+job.name;
      if (job.status !== 'PREPARED') { continue; }
      try {
        const dir=base+'/'+job.name;
        if (!fs.accessSync(dir)) { fs.mkdirSync(dir); }
        for (const name of job.files) {
          const bytes=await transfer('http://127.0.0.1:8713/'+job.name+'/'+name,http.RequestMethod.GET);
          const f=fs.openSync(dir+'/'+name,fs.OpenMode.CREATE|fs.OpenMode.WRITE_ONLY|fs.OpenMode.TRUNC);
          fs.writeSync(f.fd,bytes);fs.closeSync(f);
        }
        const text=await taskpool.execute(runGraph,dir+'/graph-probe.txt') as string;
        for (const name of ['graph-probe.txt','graph-probe.txt.stderr']) {
          const f=fs.openSync(dir+'/'+name,fs.OpenMode.READ_ONLY);const stat=fs.statSync(f.fd);const bytes=new ArrayBuffer(stat.size);fs.readSync(f.fd,bytes);fs.closeSync(f);
          await transfer('http://127.0.0.1:8713/upload/'+job.run+'/'+job.name+'/'+name,http.RequestMethod.PUT,bytes);
        }
        for (const name of fs.listFileSync(dir)) {
          if (name.startsWith('gpu_') && name.endsWith('.bin')) {
            const f=fs.openSync(dir+'/'+name,fs.OpenMode.READ_ONLY);const stat=fs.statSync(f.fd);const bytes=new ArrayBuffer(stat.size);fs.readSync(f.fd,bytes);fs.closeSync(f);
            await transfer('http://127.0.0.1:8713/upload/'+job.run+'/'+job.name+'/'+name,http.RequestMethod.PUT,bytes);
          }
        }
        this.result=job.name+'\n'+text.slice(-1600);
      } catch (error) {
        const message=job.name+' APP_ERROR '+JSON.stringify(error);
        this.result=message;
        const bytes=new Uint8Array(message.length);for(let i=0;i<message.length;i++){bytes[i]=message.charCodeAt(i);}
        await transfer('http://127.0.0.1:8713/upload/'+job.run+'/'+job.name+'/app-error.txt',http.RequestMethod.PUT,bytes.buffer as ArrayBuffer);
      }
    }
    this.result='GPU SUITE COMPLETE. Review PC results.';
    const run=jobs.length>0?jobs[0].run:'empty';
    await transfer('http://127.0.0.1:8713/upload/'+run+'/SUITE_COMPLETE.txt',http.RequestMethod.PUT,new Uint8Array([79,75]).buffer as ArrayBuffer);
  }
  aboutToAppear(): void { this.execute().catch((error: Error) => { this.result=error.message; }); }
  build() { Scroll(){Text(this.result).fontSize(12).padding(12)} }
}
