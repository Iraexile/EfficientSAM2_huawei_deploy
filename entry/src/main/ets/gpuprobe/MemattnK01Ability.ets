import { UIAbility } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
export default class MemattnK01Ability extends UIAbility {
  onWindowStageCreate(stage: window.WindowStage): void {
    stage.getMainWindow().then((main: window.Window) => { main.setWindowKeepScreenOn(true); });
    stage.loadContent('pages/MemattnK01');
  }
}
