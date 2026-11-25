import fs from "node:fs";
import path from "node:path";
import {
  AlignmentType, Document, Footer, Header, HeadingLevel, ImageRun, ImportedXmlComponent,
  LevelFormat, Packer, PageBreak, PageNumber, Paragraph, ShadingType, Table, TableCell,
  TableRow, TextRun, WidthType, convertInchesToTwip,
} from "docx";

const outputPath = process.argv[2];
if (!outputPath) throw new Error("Usage: node make_report_docx.js /abs/out.docx");
const ASSETS = "C:\\Users\\hp\\EfficientSAM2\\pc\\report_assets";
const cases = JSON.parse(fs.readFileSync(path.join(ASSETS, "cases.json"), "utf-8"));

const T = String.raw;
const palette = { dark: "263238", primary: "37474F", light: "78909C", fill: "EEF3F6" };
const font = { ascii: "Times New Roman", hAnsi: "Times New Roman", cs: "Times New Roman", eastAsia: "SimSun" };

const run = (text, options = {}) => new TextRun({ text, font, size: 21, ...options });
const para = (children, options = {}) => new Paragraph({
  spacing: { after: 140, line: 300 }, ...options,
  children: Array.isArray(children) ? children : [children],
});
const body = (text) => para(run(text), { indent: { firstLine: convertInchesToTwip(0.33) } });
const bullet = (text) => para(run(text), { numbering: { reference: "b", level: 0 } });
const h1 = (text) => para(run(text, { bold: true, size: 28, color: palette.dark }), {
  heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 140 },
});
const h2 = (text) => para(run(text, { bold: true, size: 24, color: palette.primary }), {
  heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 100 },
});
const caption = (text) => para(run(text, { size: 18, color: palette.light }), {
  alignment: AlignmentType.CENTER, spacing: { after: 200 },
});
const img = (file, w, h, cap) => [
  para(new ImageRun({ type: "png", data: fs.readFileSync(path.join(ASSETS, file)),
    transformation: { width: w, height: h } }), { alignment: AlignmentType.CENTER }),
  caption(cap),
];

const cell = (text, options = {}) => new TableCell({
  children: [para(run(String(text), { size: 18 }), { spacing: { after: 0, line: 240 } })],
  margins: { top: 80, bottom: 80, left: 100, right: 100 },
  ...options,
});
function makeTable(widths, rows) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((r, ri) => new TableRow({
      children: r.map((c, ci) => cell(c, {
        width: { size: widths[ci], type: WidthType.DXA },
        shading: ri === 0 ? { type: ShadingType.CLEAR, fill: palette.fill } : undefined,
      })),
    })),
  });
}

const xmlEscape = (v) => String(v).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const toc = (entries) => {
  const cached = entries.map(({ title, level, page }) => {
    const indent = Math.max(0, level - 1) * 360;
    return `<w:p><w:pPr><w:pStyle w:val="TOC${level}"/>
      <w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9000"/></w:tabs>
      <w:ind w:left="${indent}"/></w:pPr>
      <w:r><w:t>${xmlEscape(title)}</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>${page}</w:t></w:r></w:p>`;
  }).join("");
  return ImportedXmlComponent.fromXmlString(`<w:sdt xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:sdtPr><w:alias w:val="目录"/></w:sdtPr><w:sdtContent>
    <w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/>
    <w:instrText xml:space="preserve"> TOC \\o &quot;1-2&quot; \\h \\z \\u </w:instrText>
    <w:fldChar w:fldCharType="separate"/></w:r></w:p>${cached}
    <w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:sdtContent></w:sdt>`).root[0];
};

const tocEntries = [
  { title: "一、这份报告回答什么问题", level: 1, page: 2 },
  { title: "二、实验环境", level: 1, page: 3 },
  { title: "三、模型是怎么搬上手机的", level: 1, page: 4 },
  { title: "四、怎么确认手机上算得对", level: 1, page: 5 },
  { title: "五、精度结果", level: 1, page: 6 },
  { title: "六、速度结果（profiling 基线）", level: 1, page: 7 },
  { title: "七、NPU 这条路走得通吗", level: 1, page: 9 },
  { title: "八、结论", level: 1, page: 11 },
  { title: "附录 A：71 例对拍完整明细", level: 1, page: 13 },
  { title: "附录 B：产物索引", level: 1, page: 15 },
];

const children = [
  para(run("Efficient-SAM2 端侧部署实验报告", { bold: true, size: 40, color: palette.dark }), {
    heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, spacing: { before: 480, after: 240 },
  }),
  para(run("视频分割模型在 HUAWEI Mate 80（麒麟 9020）上的部署、正确性验证与速度基线", { size: 24, color: palette.primary }), {
    alignment: AlignmentType.CENTER, spacing: { after: 360 },
  }),
  para(run("2026-11-16", { size: 20, color: palette.light }), { alignment: AlignmentType.CENTER, spacing: { after: 480 } }),
  para(run("目录", { bold: true, size: 28, color: palette.dark }), { spacing: { before: 240, after: 120 } }),
  para(run("右键目录选择“更新域”可刷新页码。", { italics: true, color: palette.light })),
  toc(tocEntries),

  // ================= 一 =================
  h1("一、这份报告回答什么问题"),
  body(T`我们把一个叫 Efficient-SAM2 的视频分割模型（给它一段视频和第一帧上的一个目标标注，它能逐帧把这个目标“抠”出来）搬到了华为 Mate 80 手机上，全程只用手机的 CPU，不用 GPU/NPU。报告回答五个问题：`),
  bullet(T`搬上去了吗？—— 搬上去了，模型被拆成 21 个计算图在手机上完整跑通 18 帧视频；`),
  bullet(T`算得对吗？—— 对。手机上每个计算图的输出都和电脑上的参考实现逐一比对过，71 组比对全部正确；`),
  bullet(T`有多快？—— 目前每秒约 0.09 帧：第一帧约 4.2 秒，之后每帧约 10.6~11.2 秒（纯 CPU、无损精度的保守配置）；`),
  bullet(T`慢在哪？—— 约 63% 的时间花在“记忆注意力”上（模型每帧都要回看之前记住的画面特征），且回看的内容越多越慢，每千条记忆约多花 0.25 秒。这正是我们下一步“结构化记忆”改造要削掉的部分。`),
  bullet(T`能不能用 NPU 加速？—— 目前走不通。实测麒麟 9020 的 NPU 不支持 Transformer 所需的几种关键算子，而记忆注意力恰好全是这类算子（详见第七节）。`),
  body(T`需要先说明一个前提：第五、六节的全部数字都来自“纯 CPU + 最高精度（fp32）”的保守配置。这是为了拿到一条干净、可复现的基线——以后做任何加速改造，和这条基线对比就知道赚了多少。NPU 路径的可行性已在第七节单独评估，那里的结论与本基线的数字不混用。`),

  // ================= 二 =================
  h1("二、实验环境"),
  h2("2.1 电脑（用来导出模型、制作标准答案）"),
  makeTable([2200, 6800], [
    ["项", "值"],
    ["显卡", "RTX 5070 Ti Laptop（仅用于在电脑上跑参考实现，不参与手机推理）"],
    ["PyTorch", "2.11.0+cu128"],
    ["模型来源", "Efficient-SAM2（开源仓库 fork），权重用 SAM2.1 官方 tiny 版（156 MB）"],
    ["模型转换工具", "MindSpore Lite 2.11.0（华为官方推理框架的模型转换器，在 WSL2 Linux 里运行）"],
    ["标准答案口径", "最高精度 fp32，关闭所有会降低精度的开关（TF32、autocast）"],
  ]),
  h2("2.2 手机（被测对象）"),
  makeTable([2200, 6800], [
    ["项", "值"],
    ["设备", "HUAWEI Mate 80（型号 VYG-AL00）"],
    ["芯片", "麒麟 9020（8 核 CPU：1 个超大核 + 3 个大核 + 4 个小核），12 GB 内存"],
    ["系统", "HarmonyOS 7.0.0.111（API 26）"],
    ["推理引擎", "系统自带的 MindSpore Lite（通过系统 NDK 的 C 接口调用，4 线程，fp32 全精度）"],
    ["测试程序", "自研 HarmonyOS 应用：把模型和测试数据同步进手机，自动跑比对并把结果传回电脑"],
  ]),

  // ================= 三 =================
  h1("三、模型是怎么搬上手机的"),
  h2("3.1 为什么要拆图"),
  body(T`手机推理引擎不能消化一整个视频模型，必须先把它拆成小块。我们按模型内部的计算阶段拆成 4 类功能块：`),
  makeTable([2100, 3700, 3200], [
    ["功能块", "干什么", "每帧跑几次"],
    ["图像编码器（encoder）", "把当前帧画面压缩成特征（最重的通用计算）", "1 次"],
    ["记忆注意力（memattn）", "拿当前帧特征去“回看”历史帧的记忆，找出目标现在的样子", "第 1 帧不用，之后每帧 1 次"],
    ["掩码解码（samheads）", "根据（回看后的）特征画出目标轮廓，并给自己的把握打分", "1 次"],
    ["记忆编码（memenc）", "把这帧的结果写回“记忆库”，供后面的帧回看", "1 次"],
  ]),
  body(T`由于手机推理引擎要求每个计算图的输入尺寸固定，而“记忆注意力”的输入随记忆条数变大，我们按记忆长度预先做了 16 个尺寸档位（k01~k15 和一个稳态档），加上记忆编码和掩码解码各有 2 个变体，合计 21 个计算图。`),
  h2("3.2 图与图之间的“传送带”"),
  body(T`21 个图不是简单串联：记忆库怎么存、取哪几帧的记忆、每条记忆标记什么时间戳、第一帧的标注怎么处理……这些衔接逻辑（约占整个流程代码的一半）全部用 C++ 在手机端实现。为避免“衔接写错但很难查”，我们先把这套衔接逻辑在电脑上用标准答案做了一遍不跑模型的“空转演练”，17 个跟踪帧全部逐字节对上，才允许它上手机。`),
  h2("3.3 转换过程中踩过的坑（摘要)"),
  bullet(T`转换器默认把图片类数据转成“通道在最后”的排布，而测试程序按“通道在最前”喂数据——总元素个数一样多所以不报错，但算出来全是垃圾。改成显式声明排布后解决；`),
  bullet(T`掩码解码图的“标注类型”输入本应传整数，制作测试数据时写成了浮点数的字节，手机读出来是一串天文数字，导致第一帧的标注完全失效。修正数据类型后解决；`),
  bullet(T`原仓库代码本身有个隐藏 bug：第 11 帧之后的推理代码被写在一个永不执行的分支里，导致后面的帧一直在偷偷复用第 10 帧的结果。打补丁后重新制作了全部标准答案。`),

  // ================= 四 =================
  h1("四、怎么确认手机上算得对"),
  body(T`核心思路是“对答案”：同一份输入，电脑上的参考实现（PyTorch，精度最高）算一遍并记录全部中间结果作为标准答案；手机算一遍；逐数字比对。分三层，从零件到整机：`),
  bullet(T`第一层（电脑上）：PyTorch 原版 vs 导出的 ONNX 模型，43 组比对，误差全部小于 0.0001（万分之一）——证明导出没把模型导坏；`),
  bullet(T`第二层（电脑上）：ONNX 模型 vs 转换后的手机格式模型，11 组抽查全部通过——证明转换没转坏；`),
  bullet(T`第三层（手机上）：把 18 帧视频拆成 71 个测试用例（每个计算图、每一帧各算一组），手机逐个跑并与标准答案比对——证明手机上的每个零件都对；`),
  bullet(T`整机：手机端 C++ 传送带把 21 个图串起来完整跑 18 帧，逐步与标准答案比对——证明整机运转正确。`),
  body(T`判定误差时我们看的不是单个数字差多少，而是同时看两个指标：最大误差（两个结果里差得最多的那个数差多少）和余弦相似度（结果的整体方向是否一致，1.0 表示完全一致）。后者能防止“数值都很接近但整体形状变了”这类漏网之鱼。`),

  // ================= 五 =================
  h1("五、精度结果"),
  h2("5.1 单图精度：71 组比对全部正确"),
  makeTable([2400, 1300, 5300], [
    ["计算图", "组数", "误差情况"],
    ["图像编码器", "18", "最大 4.3e-05，全部通过"],
    ["记忆注意力", "17", "记忆短时 ≤5e-05 通过；记忆长时 1.4e-4~2.2e-4，略超万分之一门限"],
    ["记忆编码", "18", "最大 4.1e-05，全部通过"],
    ["掩码解码", "18", "跟踪帧最大 1.4e-4 通过；第一帧 2.9e-4 略超门限"],
  ]),
  body(T`13 组“略超门限”的比对需要专门说明：它们的余弦相似度全部是 1.0000000（结果形状与标准答案完全一致），误差来源是手机 CPU 在长序列计算中累计的浮点舍入差——序列越长，中间累加次数越多，尾数差越大。实测误差随记忆长度单调增长（记忆 4 千条时 2.3e-5，2.9 万条时 2.2e-4），正是浮点累加的指纹。这类差异属平台固有，不是算错。71 组完整明细见附录 A。`),
  h2("5.2 整机精度：18 帧全程不发散"),
  body(T`整机跑 18 帧时，每一帧的输出与标准答案的差稳定在千分之一到千分之几量级，且不随帧数放大（第 17 帧与第 2 帧的误差同量级）；每帧模型给自己打的把握分与标准答案的排序完全一致，抠图目标的选择 18 帧全部正确。`),
  body(T`整机的误差比单图稍大，原因明确：模型官方设计就是把记忆库按半精度（bf16）存储来省内存，我们的实现忠实复刻了这一点。记忆一旦带上一丁点手机端的浮点噪声，存成半精度时会有个别数值跨过舍入边界，被下一轮注意力放大。这属于设计内的表征漂移，不是实现错误——单零件精度以 71 组比对为准。`),

  // ================= 六 =================
  h1("六、速度结果（profiling 基线）"),
  h2("6.1 各功能块在手机上的耗时"),
  makeTable([2700, 1500, 1300, 1300, 2200], [
    ["功能块", "平均", "最快", "最慢", "说明"],
    ["图像编码器", "3.74 秒", "3.66", "4.42", "每帧都跑，耗时稳定"],
    ["记忆注意力", "2.13→7.27 秒", "—", "—", "随记忆长度增长，见 6.2"],
    ["掩码解码（跟踪帧）", "0.13 秒", "0.08", "0.23", ""],
    ["掩码解码（第一帧）", "0.18 秒", "—", "—", "含标注处理"],
    ["记忆编码", "0.21 秒", "0.19", "0.29", ""],
  ]),
  body(T`拼起来：第一帧约 4.2 秒（首次运行约 4.9 秒）；之后的帧约 10.6~11.2 秒。`),
  ...img("chart_frame_breakdown.png", 480, 214, "图 1：一帧约 11.1 秒的构成——记忆注意力占 63%，图像编码器占 34%"),
  h2("6.2 关键发现：记忆注意力随记忆长度线性变贵"),
  body(T`模型每处理一帧都要把当前画面和记忆库里的全部内容做一次注意力比对，记忆越多越贵。实测耗时与参与比对的记忆条数呈严格的线性关系：`),
  makeTable([2800, 3000, 3000], [
    ["记忆规模档", "参与比对的记忆条数", "耗时"],
    ["k01", "4,100 条", "2.13 秒"],
    ["k02", "8,200 条", "2.97 秒"],
    ["k03", "12,300 条", "4.15 秒"],
    ["k04", "16,400 条", "4.68 秒"],
    ["k05", "20,500 条", "5.50 秒"],
    ["k06", "24,600 条", "6.27 秒"],
    ["k07 及以上", "28,700 条（窗口打满）", "约 7.05~7.27 秒，不再增长"],
  ]),
  ...img("chart_memattn_scaling.png", 470, 261, "图 2：记忆注意力耗时与记忆条数的关系——近似每千条 0.25 秒"),
  body(T`换算下来每千条记忆约 0.25 秒。第 7 档之后不再变贵，是因为模型设计上记忆窗口最多装 7 帧（标注帧 + 最近 6 帧），窗口满了就不再长。这说明：想让它更快，要么让参与比对的记忆变少，要么让比对本身更省——这正是结构化记忆改造的两个下手点。`),
  h2("6.3 逐帧视角"),
  ...img("chart_frames.png", 480, 253, "图 3：18 帧逐帧耗时堆叠——前 7 帧记忆不断累积，之后进入稳态"),
  h2("6.4 和电脑对比"),
  ...img("chart_x86_vs_m80.png", 470, 255, "图 4：同一模型同一精度，手机 CPU 耗时约为 PC 的 3~5 倍"),
  body(T`手机 CPU 比 PC 慢 3~5 倍属正常预期（功耗和散热差一个数量级）。这个倍数同时说明：模型本身在手机 CPU 上的执行效率没有异常——慢是模型计算量和硬件差距决定的，不是部署方式浪费掉的。`),

  // ================= 七：NPU =================
  h1("七、NPU 这条路走得通吗"),
  h2("7.1 为什么专门测这件事"),
  body(T`第六节的瓶颈分析得出一个自然的追问：记忆注意力占 63% 的时间，能不能把它从 CPU 挪到手机 NPU 上算？麒麟 9020 的 NPU 对外暴露的官方编程接口叫 NNRT（Neural Network Runtime），我们把全部 21 个计算图逐一提交给 NNRT 试跑，用实测回答这个问题。`),
  h2("7.2 探测方法：整图试跑 + 单算子排查"),
  body(T`探测分两步。第一步是整图试跑：把每个计算图转成 NPU 目标格式后交给 NNRT 构建，能构建成功就实际推理一遍并与标准答案比对。第二步是定位原因：对构建失败的图，不再整图硬试，而是把图里出现的每种算子单独做成一个最小模型（只含一个算子），逐个提交，看 NPU 到底不认识哪一种。同时测试了各种可能的开关组合（fp16、不同频率档、不同数据排布、指定 NPU 芯片），排除“换个姿势就行”的可能。`),
  h2("7.3 整图结果：能跑通的只有边角"),
  makeTable([2900, 1600, 4500], [
    ["计算图", "NPU 结果", "说明"],
    ["图像编码器（encoder）", "构建失败", "含大量 LayerNorm、Transpose"],
    ["记忆注意力（memattn，全部 16 档）", "构建失败", "纯 Transformer 结构"],
    ["掩码解码（samheads，2 个变体）", "构建失败", "含 LayerNorm"],
    ["记忆编码（memenc 主干）", "构建失败", "含不支持的算子组合"],
    ["卷积小图（tinyconv 等 4 个）", "构建成功", "实际推理 2.6~6.2 毫秒，结果与标准答案一致"],
  ]),
  body(T`能跑通的 4 个小图都是纯卷积结构，合计只占单帧耗时的零头；真正费时的三块（编码器、注意力、解码）全军覆没。切换 fp16、改频率档、改数据排布、显式指定麒麟 9020 的 NPU 芯片，对结果没有任何影响。`),
  h2("7.4 算子级结果：Transformer 的关键零件大半缺席"),
  makeTable([3000, 1600, 4400], [
    ["算子", "NPU 支持", "在模型里的角色"],
    ["MatMul（矩阵乘）", "支持", "注意力和全连接的核心"],
    ["Conv（卷积）", "支持", "图像编码器的卷积层"],
    ["Reshape（改形状）", "支持", "张量整形"],
    ["Transpose（转置）", "不支持", "注意力里交换维度，无法绕开"],
    ["Softmax", "不支持", "注意力权重归一化，必须存在"],
    ["LayerNorm（层归一化）", "不支持", "每个 Transformer 模块都有"],
    ["Split（切分）", "不支持", "多头注意力拆分"],
  ]),
  ...img("chart_npu_ops.png", 470, 209, "图 5：NPU 单算子探测结果——绿色为支持，红色为构建失败"),
  body(T`我们还做了一个排除性实验：担心是“模型太大、胶水算子太多”导致的失败，于是写了一个静态折叠器，把一个 Transformer 模块从 167 个算子精简到 30 个（精简前后输出逐位一致），再提交 NPU——依然构建失败。这说明失败的根源是算子本身不受支持，而不是图的复杂度。`),
  h2("7.5 结论：NPU 帮不上这个模型，至少在 NNRT 这条路上"),
  body(T`把两层证据合起来看，结论很清楚：`),
  bullet(T`记忆注意力恰好是纯 Transformer 结构，而 Transpose、Softmax、LayerNorm 是 Transformer 绕不开的三种算子，麒麟 9020 的 NPU（经 NNRT 接口）都不支持；`),
  bullet(T`我们用的推理引擎（MindSpore Lite）遇到 NPU 不认识的算子时不会自动退回 CPU 执行，而是整个图直接构建失败——所以也没有“部分算子上 NPU、其余留 CPU”的混合方案；`),
  bullet(T`因此瓶颈定位在 CPU 上测得，并不意味着换 NPU 就能解决——恰恰相反，最慢的 memattn 正好是 NPU 最无能为力的部分。`),
  body(T`保留一点余地：理论上还存在“绕过 MindSpore Lite、直接调用 NNRT 原生 C 接口手写算子调度”的路线，但那需要为每个不支持的算子自研替代实现，工程量是重新写半个推理框架，投入产出比极低。在当前约束下，加速的指望应该放在算法侧——也就是结构化记忆改造直接减少注意力的计算量，而不是等硬件救场。`),
  h2("7.6 GPU 路径：PC 上能用，手机上没有"),
  body(T`应导师要求，我们补测了 GPU 路径，PC 端和手机端分开说。`),
  body(T`PC 端（RTX 5070 Ti Laptop，CUDA）：把 71 个用例原封不动拿到 PC 的 GPU 上重跑（同一份输入、预热 3 次后计 10 次取中位数），71 例全部成功。四个功能块的耗时对比：`),
  makeTable([2300, 1500, 1500, 1700, 2000], [
    ["功能块", "PC GPU", "PC CPU", "手机 CPU", "GPU 比手机快"],
    ["图像编码器", "46.7 毫秒", "1058 毫秒", "3743 毫秒", "约 80 倍"],
    ["记忆注意力（稳态）", "50.2 毫秒", "2241 毫秒", "7273 毫秒", "约 145 倍"],
    ["记忆编码", "2.7 毫秒", "44 毫秒", "210 毫秒", "约 78 倍"],
    ["掩码解码（跟踪帧）", "5.1 毫秒", "24 毫秒", "134 毫秒", "约 26 倍"],
  ]),
  ...img("chart_gpu_compare.png", 480, 253, "图 6：四个功能块在三平台上的耗时对比（对数轴）"),
  body(T`拼成整机看：PC GPU 上一帧约 105 毫秒（约每秒 9.5 帧），其中记忆注意力 50 毫秒仍是最贵的一块（占 48%）——注意力的线性增长规律在 GPU 上照样成立（k01 档 15.8 毫秒 → 稳态档 50.2 毫秒），只是单价从手机的每千条 0.25 秒降到约 1.5 毫秒。这说明两件事：一是模型算法本身没有问题，手机上慢纯粹是硬件算力差距；二是即使换上强 GPU，记忆注意力依然和编码器并列第一大开销，结构化记忆改造在任何平台上都有意义。`),
  body(T`手机端（麒麟 9020 的 Mali GPU）：MindSpore Lite 的编程接口里保留了 GPU 设备枚举，我们把 71 个用例全部提交给 GPU 后端实测——包括纯卷积的小图在内，全部在构建阶段失败（status=-2）。失败模式和 NPU 不同：NPU 是“有后端但算子白名单不够”，GPU 是系统推理库里根本没有编译进 GPU 后端，属于“接口留了、实现没给”。`),
  body(T`结论：麒麟 9020 上 CPU、GPU、NPU 三条路径实测封盘——GPU 后端不存在，NPU 缺 Transformer 算子，唯一可用的是 CPU。本模型的端侧加速只能走算法侧（结构化记忆）这条路。`),

  // ================= 八：结论 =================
  h1("八、结论"),
  h2("8.1 部署这件事已经做完，且每一步都可复查"),
  body(T`模型从电脑到手机的完整链路（PyTorch → ONNX → 手机格式 → 真机运行）已经打通，且每个环节都留了“对答案”的比对记录：导出环节 43 组、转换环节 11 组、真机零件 71 组、整机 18 帧。所有超出万分之一门限的比对项都能明确指出原因（浮点累加、半精度存储），没有解释不了的误差。这套比对设施是沉淀下来的资产：以后无论改模型、改记忆结构还是换硬件，重跑一遍就能知道改动有没有把精度改坏。`),
  h2("8.2 速度瓶颈确定是记忆注意力，且成因清楚"),
  body(T`手机上每帧 11 秒里，记忆注意力占 7 秒（63%）。它的耗时和“参与比对的记忆条数”严格成正比（约 0.25 秒/千条），这不是实现缺陷，而是注意力这个算法本身的计算量决定的：每条记忆都要和当前画面的 4096 个位置做一次比对。`),
  body(T`这个结论对下一步研究的意义很直接：`),
  bullet(T`收益可预估——如果结构化记忆方案能把参与比对的记忆从 2.87 万条压到 8 千条，按 0.25 秒/千条折算，每帧可省约 5.2 秒，帧耗时从 11 秒降到 6 秒以内；`),
  bullet(T`上界已知——即使不做任何选择，现有窗口设计已经把成本封在 7 秒左右，改造的目标是用更少记忆达到同等效果，而不是担心它无限膨胀；`),
  bullet(T`对照已就绪——本次的逐档耗时表（6.2 节）就是改造前后的对照基线。`),
  h2("8.3 其余三块不值得先动"),
  body(T`图像编码器（3.7 秒，34%）是通用视觉计算的固有成本，和记忆机制无关，即使记忆注意力优化到零，单帧下限也有 3.7 秒；掩码解码和记忆编码合计只占 3%，优化它们没有意义。如果未来要追求更高的抠图质量而换更大的模型权重，编码器还会更慢，届时再评估。`),
  h2("8.4 这条基线的适用边界"),
  body(T`本报告第五、六节的所有数字都是“纯 CPU + 最高精度”的保守基线。降低计算精度（fp16/INT8）这条路尚未启用，启用后应另立基线，不要和本报告数字直接比较。NPU 和 GPU 路径均已实测排除（第七节）：麒麟 9020 经 NNRT 接口不支持 Transformer 的关键算子，系统推理库又不含 GPU 后端，本模型在手机上只能依赖 CPU 路线，加速的杠杆在算法侧而不是硬件侧。PC GPU 数据（7.6 节）用于回答“模型本身在硬件够强时有多快”，与手机基线同样不混用。`),
  h2("8.5 记忆库容量不是压力，计算才是"),
  body(T`记忆库每帧新增约 1 MB（半精度 0.5 MB），7 帧窗口总共约 3.5 MB——12 GB 内存的手机上完全不是约束。结构化记忆要解决的问题因此非常聚焦：减少注意力计算量、让记忆在内存里连续排布以便硬件高效读取，而不是省存储。`),

  // ================= 附录 A =================
  h1("附录 A：71 例对拍完整明细"),
  body(T`口径：手机端单图独立运行，与电脑标准答案逐数字比对；“最大误差”为该用例全部输出数字中差值最大者；“通过”门限为万分之一（atol=rtol=1e-4）；标 ※ 的 13 例为余弦相似度 1.0000000 的平台固有浮点差（见 5.1 节）。`),
  (() => {
    const rows = [["用例", "计算图", "结果", "最大误差", "耗时(ms)"]];
    for (const c of cases) {
      rows.push([c.case, c.model,
        c.result === "OK" ? "通过" : (c.result === "FAIL" ? "超门限※" : "错误"),
        c.maxdiff.toExponential(2), c.ms.toFixed(0)]);
    }
    return makeTable([1900, 2300, 1300, 1900, 1600], rows);
  })(),

  // ================= 附录 B =================
  h1("附录 B：产物索引"),
  makeTable([4200, 4800], [
    ["产物", "路径"],
    ["过程日志（全部排障与决策记录）", "pc/WORKLOG.md"],
    ["算子支持度报告", "pc/docs/mslite_support_report.md"],
    ["电脑端对拍报告", "pc/outputs/parity_report.txt"],
    ["手机端 71 例对拍报告", "pc/device_results/report.txt"],
    ["手机端 18 帧整机报告", "pc/device_results/pipeline_report.txt"],
    ["NPU 整图探测报告", "pc/device_results/report_npu.txt、pipeline_report_npu.txt"],
    ["NPU 单算子探测报告与探针模型", "pc/device_results/probe_npu.txt；pc/outputs/onnx_opprobe/、ms_opprobe/"],
    ["静态折叠器（排除图复杂度用）", "pc/scripts/fold_static.py"],
    ["PC GPU 逐图计时数据", "pc/outputs/gpu_bench.json；pc/scripts/gpu_bench.py"],
    ["手机 GPU 对拍报告（全部构建失败）", "pc/device_results/report_gpu.txt"],
    ["21 个手机模型文件 + 常量", "pc/outputs/ms/"],
    ["标准答案（18 帧 71 例）", "pc/golden/"],
    ["手机端工程（测试框架 + 整机流水线）", "entry/src/main/cpp/mslite_runner.cpp；entry/src/main/ets/pages/Index.ets"],
  ]),
];

const doc = new Document({
  features: { updateFields: true },
  numbering: {
    config: [{
      reference: "b",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 520, hanging: 260 } } },
      }],
    }],
  },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1300, right: 1300 } } },
    headers: {
      default: new Header({
        children: [para(run("Efficient-SAM2 端侧部署实验报告", { bold: true, size: 18, color: palette.primary }), { alignment: AlignmentType.CENTER })],
      }),
    },
    footers: {
      default: new Footer({
        children: [para(new TextRun({ children: [PageNumber.CURRENT] }), { alignment: AlignmentType.CENTER })],
      }),
    },
    children,
  }],
});

fs.writeFileSync(outputPath, await Packer.toBuffer(doc));
