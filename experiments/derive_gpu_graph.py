"""Derive the suite executor from the immutable, reviewed K01 implementation."""
from pathlib import Path
root=Path(__file__).resolve().parents[1];native=root/'experiments/native'
text=(native/'memattn_strict.cpp').read_text();text=text[text.index('#include "k01_rope_layout.h"'):]
text='#include <net.h>\n#include <gpu.h>\n#include <fstream>\n#include <cstdio>\n'+text
for a,b in [('k01_rope_layout.h','suite_rope_layout.h'),('K01RoPELayout','SuiteRoPELayout'),('KAlloc','SuiteAlloc'),('KRef','SuiteRef'),('KCompare','SuiteCompare'),('KClock','SuiteClock'),('KMs','SuiteMs'),('RunMemattnStrict','RunGpuGraph'),('K01_','GRAPH_'),('/k01.param','/model.param'),('/k01.bin','/model.bin')]:text=text.replace(a,b)
text=text.replace('/PC_NCNN','/ONNX_REFERENCE')
text=text.replace('std::ifstream mf(base + "/refs-full.txt");\n    if (!mf) mf.open(base + "/refs.txt");','std::ifstream mf(base + "/refs.txt");').replace('refs.size() < 22','refs.empty()')
start=text.index('    const char *names[]');end=text.index('    std::vector<ncnn::VkMat> uploadedPacked',start)
text=text[:start]+r'''
    std::vector<ncnn::Mat> host;
    auto up=SuiteClock::now();ncnn::VkCompute upload(dev);size_t uploaded=0;
    std::ifstream inputs(base+"/inputs.txt");std::string inputName,dtype,file;int dims,w,h,d,c,batch;
    while(inputs>>inputName>>dims>>w>>h>>d>>c>>batch>>dtype>>file){
      printf("GRAPH_INPUT name=%s dims=%d w=%d h=%d d=%d c=%d batch=%d dtype=%s\n",inputName.c_str(),dims,w,h,d,c,batch,dtype.c_str());
      if(dtype!="float32"||batch!=1)throw std::runtime_error("UNSUPPORTED_INPUT_DTYPE_OR_BATCH "+inputName+" "+dtype);
      ncnn::Mat m;if(dims==1)m.create(w,(size_t)4u);else if(dims==2)m.create(w,h,(size_t)4u);else if(dims==3)m.create(w,h,c,(size_t)4u);else if(dims==4)m.create(w,h,d,c,(size_t)4u);else throw std::runtime_error("input_rank");
      std::ifstream f(base+"/"+file,std::ios::binary);size_t plane=size_t(w)*h*d;
      for(int ch=0;ch<c;ch++)if(!f.read(static_cast<char*>(m.channel(ch).data),plane*4))throw std::runtime_error("input_read "+inputName);
      host.push_back(m);uploaded+=plane*c*4;bool found=false;
      for(size_t j=0;j<net.blobs().size();j++)if(net.blobs()[j].name==inputName){upload.record_upload(host.back(),blobs[j],net.opt);found=true;}
      if(!found)throw std::runtime_error("input_name "+inputName);
    }
    if(host.empty())throw std::runtime_error("input_manifest_empty");
''' +text[end:]
text=text.replace('oldplane != m.cstep','(m.c != 1 && oldplane != m.cstep)').replace('newplane % 4','(shape[4] != 1 && newplane % 4)').replace('m.cstep = newplane;','m.cstep = (newplane+3)/4*4;')
# Report every unsupported mapping before stopping, rather than losing the rest.
text=text.replace('    for (size_t i = 0; i < net.layers().size(); i++) {','    int unsupported=0;\n    for (size_t i = 0; i < net.layers().size(); i++) {',1)
text=text.replace('throw std::runtime_error("GPU_UNSUPPORTED " + l->name);','unsupported++;')
text=text.replace('    if (net.load_model','    if(unsupported)throw std::runtime_error("GPU_UNSUPPORTED_COUNT "+std::to_string(unsupported));\n    if (net.load_model',1)
# Only graph-specific final references, no special-case PyTorch file requirement.
a=text.index('        if (blob == "out0"');b=text.index('        std::ofstream f(',a);text=text[:a]+text[b:]
(native/'gpu_graph.cpp').write_text(text)
rope=(native/'k01_rope_layout.h').read_text().replace('K01RoPELayout','SuiteRoPELayout')
rope=rope.replace('524288u','uint(a.length())').replace('x.h != 128 || x.d != 4096 || ','').replace('in[0].h != 128 || in[0].d != 4096 || ','')
(native/'suite_rope_layout.h').write_text(rope)
print('suite executor derived; reviewed K01 source unchanged')


