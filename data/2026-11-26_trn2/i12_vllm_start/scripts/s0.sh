#!/bin/sh
# step 0 inside the vllm image: no compile
echo "== env"; env | grep -i -E "nki|neuron_libtorch|tmpdir" | sort
echo "== ls -la /var/tmp"; ls -la /var/tmp
echo "== id"; id
echo "== versions"; pip show nki libtorch-neuronx-lite libtorch_neuronx_lite vllm-neuron 2>/dev/null | grep -E "^(Name|Version|Location)"
SP=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
echo "SP=$SP"
echo "== setdefault / environ of NKI vars in python packages"
grep -rn -E "NKI_ENABLE_TRACE_CACHE|NKI_DISABLE_TRACE_CACHE|NKI_COMPILE_CACHE_URL|NKI_TRACE_CACHE_URL|NKI_USE_COLZ" "$SP" --include=*.py 2>/dev/null | grep -v "/nki/_config.py" | head -40
echo "== trace cache gate"
grep -n -E "disable_compile_cache.value|disable_trace_cache.value|return _config.enable_trace_cache|nki-trace-cache-" "$SP/nki/compiler/_trace_cache.py"
grep -n -E "nki-intermediate-cache" "$SP/nki/compiler/_disk_cache.py"
echo "== warning source"
grep -rn -l "_warn_trace_cache_once" "$SP" --include=*.py 2>/dev/null
echo "== nki_hop 625-640"
F=$(find "$SP" -name nki_hop.py -path "*libtorch_neuronx_lite*" | head -1); echo "$F"; sed -n 620,640p "$F"
echo "== parallel_trace TMPDIR / var/tmp"
P=$(find "$SP" -name parallel_trace.py -path "*libtorch_neuronx_lite*" | head -1); echo "$P"
grep -n -E "var/tmp|TMPDIR|NKI_|nki" "$P" | head -30
