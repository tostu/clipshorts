FROM python:3.12-slim

# yt-dlp is pinned in pyproject.toml: Twitch changes often, bump + rebuild deliberately.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tini \
 && rm -rf /var/lib/apt/lists/*

# Fail the build if this ffmpeg cannot do the render: libass (captions) and libx264 + aac.
# Debian's ffmpeg package ships them; a different base image must keep them.
RUN ffmpeg -hide_banner -filters 2>&1 | grep -Eq '^ \S+ +ass +V->V' \
 && ffmpeg -hide_banner -encoders 2>&1 | grep -q ' libx264 ' \
 && ffmpeg -hide_banner -encoders 2>&1 | grep -Eq ' aac '

WORKDIR /srv
COPY pyproject.toml ./
COPY clipshorts ./clipshorts
RUN pip install --no-cache-dir . \
 && mkdir -p /data && chmod 777 /data

ENV DATA_DIR=/data \
    HF_HOME=/data/.cache/huggingface \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-m", "clipshorts.healthcheck"]

# tini: reap zombie ffmpeg children, forward signals
ENTRYPOINT ["tini", "--"]
# exactly one worker: the job worker and event bus hold in-process state
CMD ["uvicorn", "clipshorts.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
