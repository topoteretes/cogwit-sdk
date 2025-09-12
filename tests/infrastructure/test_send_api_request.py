import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from cogwit_sdk.infrastructure.send_api_request import send_api_request, SuccessResponse


@pytest.mark.asyncio
async def test_send_api_request_success():
    with patch("cogwit_sdk.infrastructure.send_api_request.aiohttp") as mock_aiohttp:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="success")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_aiohttp.ClientSession.return_value = mock_session

        result = await send_api_request(
            "/test", "post", {"X-Api-Key": "test"}, {"message": "test"}
        )

        assert isinstance(result, SuccessResponse)
        assert result.status == 200
        assert result.data == "success"
        mock_aiohttp.ClientSession.assert_called_once()
        mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_send_api_request_doesnt_throw_on_valid_http_methods():
    # This test only tests the first half of the function, this is to keep the rest of the function happy :D
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="success")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    with patch("cogwit_sdk.infrastructure.send_api_request.aiohttp") as mock_aiohttp:
        for method in ["get", "post", "put", "delete", "patch"]:
            mock_session = MagicMock()
            setattr(mock_session, method, MagicMock(return_value=mock_response))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_aiohttp.ClientSession.return_value = mock_session

            result = await send_api_request(
                "/test", method, {"X-Api-Key": "test"}, {"message": "test"}
            )

            assert isinstance(result, SuccessResponse)
            assert result.status == 200
            assert result.data == "success"


@pytest.mark.asyncio
async def test_send_api_request_throws_on_invalid_http_method():
    with patch("cogwit_sdk.infrastructure.send_api_request.aiohttp"):
        with pytest.raises(
            ValueError, match="'invalid_method' is not a valid HttpMethod"
        ):
            await send_api_request(
                "/test", "invalid_method", {"X-Api-Key": "test"}, {"message": "test"}
            )
