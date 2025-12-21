sudo jetson_clocks

sudo nsys profile \
  -t nvtx,cuda,osrt,cudnn,cublas \
  -s cpu \
  -o whisper_profile_result \
  --force-overwrite true \
  python3 nsys_profile_whisper.py