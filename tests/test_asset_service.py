"""AssetService: ảnh không tồn tại -> AppError(NOT_FOUND) = 404."""
from pathlib import Path

import pytest

from core.database.assets import AssetService
from core.exception.errors import AppError, ErrorCode


def test_unknown_image_id_raises_not_found():
    with pytest.raises(AppError) as ei:
        AssetService(Path(".")).image_bytes("nope")
    assert ei.value.code is ErrorCode.NOT_FOUND
    assert ei.value.status_code == 404


def test_missing_file_raises_not_found():
    svc = AssetService(Path("."), {"x": "images/missing.png"})
    with pytest.raises(AppError) as ei:
        svc.image_bytes("x")
    assert ei.value.status_code == 404
