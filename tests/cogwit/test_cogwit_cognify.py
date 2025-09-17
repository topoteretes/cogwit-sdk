import pytest
from cogwit_sdk.cogwit.cogwit import cogwit, CogwitConfig, CognifyResponse, CognifyError
from cogwit_sdk.infrastructure.send_api_request import SuccessResponse, ErrorResponse
from unittest.mock import AsyncMock, patch
from uuid import UUID
from pydantic import BaseModel
from typing import Dict, Any, List


@pytest.mark.asyncio
async def test_cogwit_cognify_success():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()
    pipeline_run_id = UUID("87654321-4321-4321-4321-cba987654321")
    mock_send_api_request.return_value = SuccessResponse(
        status=200, data={"pipeline_run_id": str(pipeline_run_id)}
    )

    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_dataset_ids = [
            UUID("12345678-1234-1234-1234-123456789abc"),
            UUID("11111111-2222-3333-4444-555555555555"),
        ]
        result = await cogwit_instance.cognify(dataset_ids=dummy_dataset_ids)

        assert mock_send_api_request.call_args[0] == (
            "/cognify",
            "post",
            {"X-Api-Key": dummy_api_key},
            {"dataset_ids": dummy_dataset_ids},
        )
        assert result == CognifyResponse(status=200, pipeline_run_id=pipeline_run_id)
        assert mock_send_api_request.call_count == 1


@pytest.mark.asyncio
async def test_cogwit_cognify_assert_api_request_args():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )

    class DummyResponse(BaseModel):
        status: int
        error: Dict[str, Any]

    mock_send_api_request = AsyncMock()
    mock_send_api_request.return_value = DummyResponse(
        status=400, error={"error": "Invalid dataset IDs"}
    )
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_dataset_ids = [UUID("12345678-1234-1234-1234-123456789abc")]
        await cogwit_instance.cognify(dataset_ids=dummy_dataset_ids)
        assert mock_send_api_request.call_count == 1
        assert mock_send_api_request.call_args[0] == (
            "/cognify",
            "post",
            {"X-Api-Key": dummy_api_key},
            {"dataset_ids": dummy_dataset_ids},
        )


@pytest.mark.asyncio
async def test_cogwit_cognify_failure():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()
    mock_send_api_request.return_value = ErrorResponse(
        status=400, error={"error": "Dataset not found"}
    )
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_dataset_ids = [UUID("12345678-1234-1234-1234-123456789abc")]
        result = await cogwit_instance.cognify(dataset_ids=dummy_dataset_ids)
        assert result == CognifyError(status=400, error={"error": "Dataset not found"})
        assert mock_send_api_request.call_count == 1


@pytest.mark.asyncio
async def test_cogwit_cognify_any_non_success_response_returns_CognifyError():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()

    class DummyResponse(BaseModel):
        status: int
        error: Dict[str, Any]

    mock_send_api_request.return_value = DummyResponse(
        status=500, error={"error": "Internal server error"}
    )
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_dataset_ids = [
            UUID("12345678-1234-1234-1234-123456789abc"),
            UUID("87654321-4321-4321-4321-cba987654321"),
        ]
        result = await cogwit_instance.cognify(dataset_ids=dummy_dataset_ids)
        assert result == CognifyError(
            status=500, error={"error": "Internal server error"}
        )
        assert mock_send_api_request.call_count == 1


@pytest.mark.asyncio
async def test_cogwit_cognify_send_api_request_throws_error():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()

    class MyException(Exception):
        pass

    mock_send_api_request.side_effect = MyException("Network error")
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_dataset_ids = [UUID("12345678-1234-1234-1234-123456789abc")]
        with pytest.raises(MyException):
            await cogwit_instance.cognify(dataset_ids=dummy_dataset_ids)
        assert mock_send_api_request.call_count == 1


@pytest.mark.asyncio
async def test_cogwit_cognify_with_empty_dataset_ids():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()
    pipeline_run_id = UUID("87654321-4321-4321-4321-cba987654321")
    mock_send_api_request.return_value = SuccessResponse(
        status=200, data={"pipeline_run_id": str(pipeline_run_id)}
    )

    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        empty_dataset_ids: List[UUID] = []
        result = await cogwit_instance.cognify(dataset_ids=empty_dataset_ids)

        assert mock_send_api_request.call_args[0] == (
            "/cognify",
            "post",
            {"X-Api-Key": dummy_api_key},
            {"dataset_ids": empty_dataset_ids},
        )
        assert result == CognifyResponse(status=200, pipeline_run_id=pipeline_run_id)
        assert mock_send_api_request.call_count == 1
