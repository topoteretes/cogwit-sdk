import pytest
from cogwit_sdk.cogwit.cogwit import cogwit, CogwitConfig, AddResponse, AddError
from cogwit_sdk.infrastructure.send_api_request import SuccessResponse, ErrorResponse
from unittest.mock import AsyncMock, patch
from uuid import UUID
from pydantic import BaseModel
from typing import Dict, Any

@pytest.mark.asyncio
async def test_cogwit_add_success():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()
    dataset_id = UUID("12345678-1234-1234-1234-123456789abc")
    pipeline_run_id = UUID("87654321-4321-4321-4321-cba987654321")
    mock_send_api_request.return_value = SuccessResponse(status=200, data={"dataset_id": str(dataset_id), "pipeline_run_id": str(pipeline_run_id)})

    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_data = "Test data"
        dummy_dataset_name = "test_dataset"
        result = await cogwit_instance.add(data=dummy_data, dataset_name=dummy_dataset_name)
        
        assert mock_send_api_request.call_args[0] == ("/add", "post", {"X-Api-Key": dummy_api_key}, {"text_data": dummy_data, "dataset_id": None, "dataset_name": dummy_dataset_name})
        assert result == AddResponse(status=200, dataset_id=dataset_id, pipeline_run_id=pipeline_run_id)
        assert mock_send_api_request.call_count == 1
        
@pytest.mark.asyncio
async def test_cogwit_add_assert_api_request_args():
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
    mock_send_api_request.return_value = DummyResponse(status=400, error={"error": "Invalid API key"})
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_data = "Test data"
        dummy_dataset_name = "test_dataset"
        await cogwit_instance.add(data=dummy_data, dataset_name=dummy_dataset_name)
        assert mock_send_api_request.call_count == 1
        assert mock_send_api_request.call_args[0] == ("/add", "post", {"X-Api-Key": dummy_api_key}, {"text_data": dummy_data, "dataset_id": None, "dataset_name": dummy_dataset_name})

@pytest.mark.asyncio
async def test_cogwit_add_failure():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()
    mock_send_api_request.return_value = ErrorResponse(status=400, error={"error": "Invalid API key"})
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_data = "Test data"
        dummy_dataset_name = "test_dataset"
        result = await cogwit_instance.add(data=dummy_data, dataset_name=dummy_dataset_name)
        assert result == AddError(status=400, error={"error": "Invalid API key"})
        assert mock_send_api_request.call_count == 1

@pytest.mark.asyncio
async def test_cogwit_add_any_non_success_response_returns_AddError():
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

    mock_send_api_request.return_value = DummyResponse(status=400, error={"error": "Invalid API key"})
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_data = "Test data"
        dummy_dataset_name = "test_dataset"
        result = await cogwit_instance.add(data=dummy_data, dataset_name=dummy_dataset_name)
        assert result == AddError(status=400, error={"error": "Invalid API key"})
        assert mock_send_api_request.call_count == 1

@pytest.mark.asyncio
async def test_cogwit_add_send_api_request_throws_error():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()
    class MyException(Exception):
        pass
    mock_send_api_request.side_effect = MyException("Test error")
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_data = "Test data"
        dummy_dataset_name = "test_dataset"
        with pytest.raises(MyException):
            await cogwit_instance.add(data=dummy_data, dataset_name=dummy_dataset_name)
        assert mock_send_api_request.call_count == 1