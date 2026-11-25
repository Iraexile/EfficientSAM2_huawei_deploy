import gpu from 'libgpu_probe.so';
import { common } from '@kit.AbilityKit';
@Entry
@Component
struct GpuProbe {
  @State result: string = 'Running GPU-only minimum probe...';
  aboutToAppear(): void {
    setTimeout(() => {
      const context = getContext(this) as common.UIAbilityContext;
      this.result = gpu.run(context.filesDir + '/gpu-probe.txt');
    }, 500);
  }
  build() {
    Scroll() { Text(this.result).fontSize(12).padding(12) }
  }
}
