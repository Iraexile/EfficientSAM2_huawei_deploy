import { UIAbility } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
export default class GpuSuiteAbility extends UIAbility {
  onWindowStageCreate(stage: window.WindowStage): void {
    stage.getMainWindow().then((main: window.Window) => { main.setWindowKeepScreenOn(true); });
    stage.loadContent('pages/GpuSuite');
  }
}
