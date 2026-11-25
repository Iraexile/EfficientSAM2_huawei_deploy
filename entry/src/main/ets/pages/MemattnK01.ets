import gpu from 'libgpu_probe.so';
import { common } from '@kit.AbilityKit';
import { fileIo as fs } from '@kit.CoreFileKit';
import { taskpool } from '@kit.ArkTS';
@Concurrent
function runK01(path: string): string { return gpu.run(path); }
@Entry
@Component
struct MemattnK01 {
  @State result: string = 'Running GPU-only minimum probe...';
  aboutToAppear(): void {
    setTimeout(async () => {
      const context = getContext(this) as common.UIAbilityContext;
      const files = await context.resourceManager.getRawFileList('');
      for (const name of files) {
        const bytes = await context.resourceManager.getRawFileContent(name);
        const f = fs.openSync(context.filesDir + '/' + name, fs.OpenMode.CREATE | fs.OpenMode.WRITE_ONLY | fs.OpenMode.TRUNC);
        fs.writeSync(f.fd, bytes.buffer); fs.closeSync(f);
      }
      this.result = await taskpool.execute(runK01, context.filesDir + '/k01-probe.txt') as string;
    }, 500);
  }
  build() {
    Scroll() { Text(this.result).fontSize(12).padding(12) }
  }
}
