import pytest

from clipshorts.media import MediaError, summarize_probe

RAW = {
    "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
         "avg_frame_rate": "60/1"},
        {"codec_type": "audio", "codec_name": "aac"},
    ],
    "format": {"duration": "29.97", "size": "123456", "bit_rate": "5000000"},
}


def test_summarize_probe():
    s = summarize_probe(RAW)
    assert s["width"] == 1920 and s["height"] == 1080
    assert s["fps"] == 60.0 and s["duration"] == 29.97
    assert s["has_audio"] and s["audio_codec"] == "aac"
    assert s["size_bytes"] == 123456


def test_summarize_probe_no_audio_and_zero_fps_den():
    raw = {"streams": [{"codec_type": "video", "width": 10, "height": 10,
                        "avg_frame_rate": "0/0"}], "format": {}}
    s = summarize_probe(raw)
    assert s["has_audio"] is False and s["fps"] is None and s["duration"] is None


def test_summarize_probe_no_video():
    with pytest.raises(MediaError):
        summarize_probe({"streams": [{"codec_type": "audio"}], "format": {}})
