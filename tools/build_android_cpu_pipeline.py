#!/usr/bin/env python3
"""Build one quality-first Android CPU package from GPT/SoVITS checkpoints.

All checkpoint/version/frontend work happens here. The resulting .gsvm exposes only UTF-8 text to
PCM. No quantization, half conversion, pruning or approximate operator replacement is performed.
"""
import argparse, subprocess, sys
from pathlib import Path
from model_profiles import detect_sovits


def run(*values: object) -> None:
    command=[str(value) for value in values]; print('Running:', ' '.join(command),flush=True)
    subprocess.run(command,check=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpt',required=True,type=Path); p.add_argument('--sovits',required=True,type=Path)
    p.add_argument('--vocoder',type=Path,help='Optional V4 HiFi-GAN checkpoint')
    p.add_argument('--reference',required=True,type=Path); p.add_argument('--prompt',required=True)
    p.add_argument('--name',required=True); p.add_argument('--output',type=Path)
    p.add_argument('--pipeline-output',type=Path); p.add_argument('--model-output',type=Path)
    p.add_argument('--work',required=True,type=Path); p.add_argument('--upstream',type=Path,default=Path('..'))
    p.add_argument('--language',default='all_zh'); p.add_argument('--minimum-ram-mb',type=int,default=6144)
    p.add_argument('--product-version', default='v3.2.0+0', help='User-visible product/conversion version')
    p.add_argument('--runtime-options',action='store_true',help='Mark option-aware graph ABI v1')
    p.add_argument('--reuse-export',action='store_true',help='Reuse t2s.pt/vits.pt/conditioning.safetensors in --work')
    p.add_argument('--validation-report',type=Path,help='Passed hash-bound report from validate_v2pp_cpu_artifacts.py')
    a=p.parse_args()
    if a.pipeline_output is not None and a.model_output is None:
        raise SystemExit('--pipeline-output requires --model-output')
    if a.output is not None and (a.pipeline_output is not None or a.model_output is not None):
        raise SystemExit('--output cannot be combined with standalone outputs')
    if a.output is None and a.model_output is None:
        raise SystemExit('--output is required unless standalone outputs are supplied')
    here=Path(__file__).resolve().parent; upstream=a.upstream.resolve(); work=a.work.resolve(); work.mkdir(parents=True,exist_ok=True)
    profile,_,_=detect_sovits(a.sovits.resolve())
    if profile.id not in {'v2ProPlus','v4'}:
        raise SystemExit(f'Android scope supports only v2ProPlus and v4, got {profile.id}')
    if not a.reuse_export:
        export_args=[sys.executable,here/'export_cpu_artifacts.py','--gpt',a.gpt.resolve(),'--sovits',a.sovits.resolve(),
            '--reference',a.reference.resolve(),'--prompt',a.prompt,'--language',a.language,'--output',work,'--upstream',upstream]
        if a.vocoder: export_args.extend(['--vocoder',a.vocoder.resolve()])
        run(*export_args)
    if profile.id != 'v4':
        ssl=work/'ssl_mobile_fp32.pt'
        if not (a.reuse_export and ssl.is_file()):
            ssl_model=(upstream/'pretrained_models/chinese-hubert-base') if (upstream/'pretrained_models/chinese-hubert-base').exists() else (upstream/'GPT_SoVITS/pretrained_models/chinese-hubert-base')
            run(sys.executable,here/'export_ssl_mobile.py','--model',ssl_model,'--output',ssl)
        run(sys.executable,here/'fuse_pipeline.py','--artifacts',work,'--ssl',ssl,'--output',work/'pipeline_core.pt')
    bert=work/'bert_mobile_fp32_eager.pt'
    if not (a.reuse_export and bert.is_file()):
        bert_model=(upstream/'pretrained_models/chinese-roberta-wwm-ext-large') if (upstream/'pretrained_models/chinese-roberta-wwm-ext-large').exists() else (upstream/'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large')
        run(sys.executable,here/'export_bert_mobile.py','--model',bert_model,'--output',bert)
    phrase_cache=here.parent/'build/frontend-phrases-v2.json'
    if not phrase_cache.is_file():
        run(sys.executable,here/'compile_frontend_phrases.py','--upstream',upstream,'--output',phrase_cache)
    frontend=here.parent/'build/g2pw-mobile-v3'
    frontend_files=(
        'frontend.json','g2pW.onnx','jieba-dict.txt','jieba-pos-hmm.bin',
        'polyphonic.rep','polyphonic-fix.rep','english.json','english-lexicon.tsv',
        'english-homographs.tsv','english-names.tsv','english-tagger.bin','english-g2p.bin',
        'english-unigrams.tsv','english-bigrams.tsv',
    )
    if any(not (frontend/name).is_file() for name in frontend_files):
        run(sys.executable,here/'export_g2pw_mobile.py','--upstream',upstream,'--output',frontend)
    package=[sys.executable,here/'build_cpu_package.py','--artifacts',work,
        '--frontend',frontend,'--name',a.name,'--version',profile.id,'--product-version',a.product_version,
        '--frontend-profile','full-zh-en-g2pw-v3','--minimum-ram-mb',a.minimum_ram_mb]
    if a.output is not None: package.extend(['--output',a.output.resolve()])
    if a.model_output is not None: package.extend(['--model-output',a.model_output.resolve()])
    if a.pipeline_output is not None: package.extend(['--pipeline-output',a.pipeline_output.resolve()])
    # Both in-scope exporters now expose the option-aware graph ABI v1. Keep the explicit flag on
    # the package command so lower-level packaging remains conservative, while the one-command
    # builder always produces a package that can consume the UI controls.
    package.append('--runtime-options')
    package.append('--reference-input')
    if a.validation_report is not None:
        package.extend(['--upstream-equivalent','--validation-report',a.validation_report.resolve()])
    if profile.id == 'v4':
        run(sys.executable,here/'build_bert_stage.py','--bert',bert,'--output',work/'bert_stage.pt')
        package.extend(['--bert-stage','bert_stage.pt','--acoustic-stage','pipeline_core.pt'])
    else:
        package.extend(['--bert-stage',bert.name,'--acoustic-stage','pipeline_core.pt'])
    run(*package)
    print('Created Android CPU package outputs')


if __name__=='__main__': main()
