export const runCase: (modelPath: string, caseDir: string, outDir: string, threads: number, device?: string) => string;
export const runAllCases: (modelDir: string, vectorDir: string, outDir: string, threads: number, device?: string) => string;
export const runPipeline: (modelDir: string, vectorDir: string, outDir: string, threads: number, device?: string) => string;
export const runNnrtProbe: (modelDir: string, vectorDir: string, outDir: string) => string;
