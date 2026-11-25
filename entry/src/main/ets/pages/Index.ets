import { hilog } from '@kit.PerformanceAnalysisKit';
import { fileIo as fs } from '@kit.CoreFileKit';
import { http } from '@kit.NetworkKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { taskpool } from '@kit.ArkTS';
import testNapi from 'libentry.so';
import { cryptoFramework } from '@kit.CryptoArchitectureKit';

const DOMAIN = 0x0000;
const TAG = 'sam2';
@Concurrent
async function workerFlow(base: string, mode: string): Promise<string> {
  // 全部工具函数与常量就地定义（@Concurrent 只允许 import 与局部变量）
  const PC_BASE = 'http://192.0.2.1:8710'; // Example address; configure for your own test host.
  const DOMAIN = 0x0000;
  const TAG = 'sam2';
  const download = (url: string, dst: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      const req = http.createHttp();
      const f = fs.openSync(dst, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE | fs.OpenMode.TRUNC);
      let code = -1;
      let failed = '';
      req.on('dataReceive', (chunk: ArrayBuffer) => {
        try {
          fs.writeSync(f.fd, chunk);
        } catch (e) {
          failed = `write: ${JSON.stringify(e)}`;
        }
      });
      req.on('dataEnd', () => {
        try {
          fs.closeSync(f);
        } catch (e) {
          // ignore
        }
        req.destroy();
        if (failed) {
          reject(new Error(`${url}: ${failed}`));
        } else if (code > 0 && code !== 200) {
          reject(new Error(`${url}: HTTP ${code}`));
        } else {
          resolve();
        }
      });
      req.requestInStream(url, {
        method: http.RequestMethod.GET,
        connectTimeout: 30000,
        readTimeout: 600000,
      }, (err, responseCode: number) => {
        code = responseCode;
        if (err) {
          try {
            fs.closeSync(f);
          } catch (e) {
            // ignore
          }
          req.destroy();
          reject(new Error(`${url}: ${JSON.stringify(err)}`));
        }
      });
    });
  };

  const syncFromPc = async (): Promise<string> => {
    const listDst = `${base}/push_list.txt`;
    await download(`${PC_BASE}/push_list.txt`, listDst);
    const listContent = fs.readTextSync(listDst);
    const stampPath = `${base}/sync_stamp.txt`;
    if (fs.accessSync(stampPath) && fs.readTextSync(stampPath) === listContent) {
      return 'sync: up to date';
    }
    const lines = listContent.split('\n').filter((l) => l.trim().length > 0);
    let skipped = 0;
    for (let i = 0; i < lines.length; i++) {
      const parts = lines[i].replace(/\r/g, '').split(' ');
      const remote = parts[0];
      const local = parts[1];
      const wantSize = parts.length > 2 ? Number(parts[2]) : -1;
      const wantMd5 = parts.length > 3 ? parts[3] : '';
      const dst = `${base}/${local}`;
      // 增量：本地文件存在且 size+md5 双一致才跳过（清单第 3/4 列）
      if (wantSize >= 0 && fs.accessSync(dst)) {
        const cur = fs.statSync(dst);
        if (cur.size === wantSize) {
          let same = wantMd5.length === 0;
          if (!same) {
            // 分块 md5，避免大文件（400MB 模型）一次性读入内存
            const md = cryptoFramework.createMd('MD5');
            const fh = fs.openSync(dst, fs.OpenMode.READ_ONLY);
            const buf = new ArrayBuffer(8 * 1024 * 1024);
            let n = fs.readSync(fh.fd, buf);
            while (n > 0) {
              md.updateSync({ data: new Uint8Array(buf, 0, n) });
              n = fs.readSync(fh.fd, buf);
            }
            fs.closeSync(fh);
            const dig = md.digestSync();
            let hex = '';
            for (let k = 0; k < dig.data.length; k++) {
              hex += dig.data[k].toString(16).padStart(2, '0');
            }
            same = (hex === wantMd5);
          }
          if (same) {
            skipped++;
            continue;
          }
        }
      }
      const parent = dst.substring(0, dst.lastIndexOf('/'));
      if (!fs.accessSync(parent)) {
        fs.mkdirSync(parent, true);
      }
      if (fs.accessSync(dst)) {
        fs.unlinkSync(dst);
      }
      await download(`${PC_BASE}/${remote}`, dst);
      const st = fs.statSync(dst);
      hilog.info(DOMAIN, TAG, 'synced %{public}s size=%{public}d', local, st.size);
      if (st.size === 0) {
        throw new Error(`downloaded empty file: ${local}`);
      }
    }
    const f = fs.openSync(stampPath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE | fs.OpenMode.TRUNC);
    fs.writeSync(f.fd, listContent);
    fs.closeSync(f);
    return `sync: ${lines.length} files done (${skipped} unchanged)`;
  };

  const runSmoke = (): string => {
    const enc = testNapi.runCase(
      `${base}/models/image_encoder.dat`,
      `${base}/vectors/image_encoder_000`,
      `${base}/out/image_encoder_000`, 4);
    hilog.info(DOMAIN, TAG, 'smoke encoder: %{public}s', enc);
    const ma = testNapi.runCase(
      `${base}/models/memattn_k01.dat`,
      `${base}/vectors/memattn_004`,
      `${base}/out/memattn_004`, 4);
    hilog.info(DOMAIN, TAG, 'smoke memattn: %{public}s', ma);
    let bisect = '';
    const cuts = ['tinyconv', 'enc_cut_conv', 'enc_cut_embed', 'enc_cut_blk0'];
    for (const c of cuts) {
      const r = testNapi.runCase(
        `${base}/models/${c}.dat`,
        `${base}/vectors/${c}_000`,
        `${base}/out/${c}_000`, 4);
      hilog.info(DOMAIN, TAG, 'smoke %{public}s: %{public}s', c, r);
      bisect += `\n${c}: ${r}`;
    }
    return `encoder: ${enc}\nmemattn: ${ma}${bisect}`;
  };

  let status = '';
  try {
    status = await syncFromPc();
  } catch (e) {
    const be = e as BusinessError;
    return `sync exception: code=${be?.code} msg=${be?.message ?? JSON.stringify(e)}`;
  }
  try {
    let r: string;
    if (mode === 'all') {
      r = testNapi.runAllCases(`${base}/models`, `${base}/vectors`, `${base}/out`, 4);
    } else if (mode === 'all_npu') {
      r = testNapi.runAllCases(`${base}/models`, `${base}/vectors`, `${base}/out`, 4, 'nnrt');
    } else if (mode === 'all_gpu') {
      r = testNapi.runAllCases(`${base}/models`, `${base}/vectors`, `${base}/out`, 4, 'gpu');
    } else if (mode === 'pipeline') {
      r = testNapi.runPipeline(`${base}/models`, `${base}/vectors`, `${base}/out`, 4);
    } else if (mode === 'pipeline_npu') {
      r = testNapi.runPipeline(`${base}/models`, `${base}/vectors`, `${base}/out`, 4, 'nnrt');
    } else if (mode === 'probe') {
      r = testNapi.runNnrtProbe(`${base}/models`, `${base}/vectors`, `${base}/out`);
    } else {
      r = runSmoke();
    }
    return `${status}\n${r}`;
  } catch (e) {
    const be = e as BusinessError;
    return `${status}\nrun exception: code=${be?.code} msg=${be?.message ?? JSON.stringify(e)}`;
  }
}

// ---------------- UI ----------------
const PC_BASE_UI = 'http://192.0.2.1:8710'; // Example address; configure for your own test host.

@Entry
@Component
struct Index {
  @State status: string = 'idle';
  @State running: boolean = false;

  private filesDir(): string {
    return getContext(this).filesDir;
  }

  private uploadResult(name: string, text: string): Promise<void> {
    return new Promise((resolve) => {
      const req = http.createHttp();
      req.request(`${PC_BASE_UI}/upload?path=${name}`, {
        method: http.RequestMethod.PUT,
        extraData: text,
        connectTimeout: 10000,
        readTimeout: 30000,
      }, (err, data) => {
        req.destroy();
        if (err || !data || data.responseCode !== 200) {
          hilog.warn(DOMAIN, TAG, 'upload %{public}s failed', name);
        }
        resolve();
      });
    });
  }

  private startFlow(mode: string, resultName: string) {
    if (this.running) {
      return;
    }
    this.running = true;
    this.status = `${mode}: syncing + running...`;
    const task = new taskpool.Task(workerFlow, this.filesDir(), mode);
    taskpool.execute(task).then((ret) => {
      this.status = ret as string;
      this.uploadResult(resultName, this.status);
      if (mode === 'all' || mode === 'all_npu' || mode === 'all_gpu') {
        try {
          const report = fs.readTextSync(`${this.filesDir()}/out/report.txt`);
          const name = mode === 'all_npu' ? 'report_npu.txt' : (mode === 'all_gpu' ? 'report_gpu.txt' : 'report.txt');
          this.uploadResult(name, report);
        } catch (e) {
          hilog.warn(DOMAIN, TAG, 'read report.txt failed');
        }
      }
      if (mode === 'pipeline' || mode === 'pipeline_npu') {
        try {
          const report = fs.readTextSync(`${this.filesDir()}/out/pipeline_report.txt`);
          this.uploadResult(mode === 'pipeline_npu' ? 'pipeline_report_npu.txt' : 'pipeline_report.txt', report);
        } catch (e) {
          hilog.warn(DOMAIN, TAG, 'read pipeline_report.txt failed');
        }
      }
      this.running = false;
    }).catch((e: BusinessError) => {
      this.status = `task exception: ${e?.code} ${e?.message}`;
      this.uploadResult(resultName, this.status);
      this.running = false;
    });
  }

  aboutToAppear(): void {
    setTimeout(() => {
      this.startFlow('smoke', 'smoke.txt');
    }, 300);
  }

  build() {
    Column({ space: 16 }) {
      Text('Efficient-SAM2 端侧对拍')
        .fontSize($r('app.float.page_text_font_size'))
        .fontWeight(FontWeight.Bold)

      Button('同步数据 + 冒烟测试')
        .enabled(!this.running)
        .onClick(() => {
          this.startFlow('smoke', 'smoke.txt');
        })

      Button('全部 43 例对拍')
        .enabled(!this.running)
        .onClick(() => {
          this.startFlow('all', 'allcases.txt');
        })

      Button('全流水线 18 帧')
        .enabled(!this.running)
        .onClick(() => {
          this.startFlow('pipeline', 'pipeline.txt');
        })

      Button('全部对拍（GPU）')
        .enabled(!this.running)
        .onClick(() => {
          this.startFlow('all_gpu', 'allcases_gpu.txt');
        })

      Button('全部对拍（NPU）')
        .enabled(!this.running)
        .onClick(() => {
          this.startFlow('all_npu', 'allcases_npu.txt');
        })

      Button('全流水线 18 帧（NPU）')
        .enabled(!this.running)
        .onClick(() => {
          this.startFlow('pipeline_npu', 'pipeline_npu.txt');
        })

      Button('NPU 探针（tinyconv 遍历配置）')
        .enabled(!this.running)
        .onClick(() => {
          this.startFlow('probe', 'probe_npu.txt');
        })

      Text(this.status).fontSize(12).width('90%')
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
  }
}
