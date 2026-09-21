import base64

import pytest

from imgproxy.openai_api import RequestError, image_response, parse_size, validate_n


def test_parse_size_rounds_to_32():
    assert parse_size("1000x1000") == (992, 992)
    assert parse_size("1024x1024") == (1024, 1024)
    assert parse_size("33x33") == (32, 32)


def test_parse_size_clamps():
    assert parse_size("99999x10") == (4096, 32)


def test_parse_size_auto_and_default():
    assert parse_size("auto") == (1024, 1024)
    assert parse_size(None) == (1024, 1024)
    assert parse_size("512x512", default=(640, 640)) == (512, 512)


def test_parse_size_rejects_garbage():
    with pytest.raises(RequestError):
        parse_size("big")
    with pytest.raises(RequestError):
        parse_size("0x0")


def test_validate_n():
    assert validate_n(None) == 1
    assert validate_n(3) == 3
    with pytest.raises(RequestError):
        validate_n(0)
    with pytest.raises(RequestError):
        validate_n(9)


def test_image_response_shape():
    raw = b"png-bytes"
    body = image_response([base64.b64encode(raw).decode()])
    assert body["created"] > 0
    assert len(body["data"]) == 1
    assert base64.b64decode(body["data"][0]["b64_json"]) == raw
