import pytest
from cogwit_sdk.cogwit.cogwit import cogwit, CogwitConfig, SearchResponse, SearchError
from cogwit_sdk.infrastructure.send_api_request import SuccessResponse, ErrorResponse
from cogwit_sdk.modules.search.SearchType import SearchType
from unittest.mock import AsyncMock, patch
from pydantic import BaseModel
from typing import Dict, Any


@pytest.mark.asyncio
async def test_cogwit_search_success():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()
    dummy_payload = {"results": ["result1", "result2"], "metadata": {"count": 2}}
    mock_send_api_request.return_value = SuccessResponse(status=200, data=dummy_payload)

    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_query_text = "What is the meaning of life?"
        dummy_query_type = SearchType.RAG_COMPLETION
        result = await cogwit_instance.search(
            query_text=dummy_query_text, query_type=dummy_query_type
        )

        assert mock_send_api_request.call_args[0] == (
            "/search",
            "post",
            {"X-Api-Key": dummy_api_key},
            {
                "query_type": str(dummy_query_type),
                "query_text": dummy_query_text,
                "use_combined_context": False,
            },
        )
        assert result == SearchResponse(status=200, payload=dummy_payload)
        assert mock_send_api_request.call_count == 1


@pytest.mark.asyncio
async def test_cogwit_search_with_combined_context():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()
    dummy_payload = {"graph_results": ["node1", "node2"], "context": "extended"}
    mock_send_api_request.return_value = SuccessResponse(status=200, data=dummy_payload)

    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_query_text = "Find connections in the data"
        dummy_query_type = SearchType.GRAPH_COMPLETION
        result = await cogwit_instance.search(
            query_text=dummy_query_text,
            query_type=dummy_query_type,
            use_combined_context=True,
        )

        assert mock_send_api_request.call_args[0] == (
            "/search",
            "post",
            {"X-Api-Key": dummy_api_key},
            {
                "query_type": str(dummy_query_type),
                "query_text": dummy_query_text,
                "use_combined_context": True,
            },
        )
        assert result == SearchResponse(status=200, payload=dummy_payload)
        assert mock_send_api_request.call_count == 1


@pytest.mark.asyncio
async def test_cogwit_search_different_search_types():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )

    # Test different SearchType values
    test_cases = [
        (SearchType.SUMMARIES, {"summaries": ["summary1", "summary2"]}),
        (SearchType.INSIGHTS, {"insights": ["insight1", "insight2"]}),
        (SearchType.CHUNKS, {"chunks": ["chunk1", "chunk2"]}),
        (SearchType.FEEDBACK, {"feedback": "positive"}),
        (SearchType.TEMPORAL, {"timeline": ["event1", "event2"]}),
    ]

    for search_type, expected_payload in test_cases:
        mock_send_api_request = AsyncMock()
        mock_send_api_request.return_value = SuccessResponse(
            status=200, data=expected_payload
        )

        with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
            dummy_query_text = f"Test query for {search_type.value}"
            result = await cogwit_instance.search(
                query_text=dummy_query_text, query_type=search_type
            )

            assert mock_send_api_request.call_args[0] == (
                "/search",
                "post",
                {"X-Api-Key": dummy_api_key},
                {
                    "query_type": str(search_type),
                    "query_text": dummy_query_text,
                    "use_combined_context": False,
                },
            )
            assert result == SearchResponse(status=200, payload=expected_payload)


@pytest.mark.asyncio
async def test_cogwit_search_assert_api_request_args():
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
        status=400, error={"error": "Invalid query"}
    )
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_query_text = "Invalid query"
        dummy_query_type = SearchType.RAG_COMPLETION
        await cogwit_instance.search(
            query_text=dummy_query_text, query_type=dummy_query_type
        )
        assert mock_send_api_request.call_count == 1
        assert mock_send_api_request.call_args[0] == (
            "/search",
            "post",
            {"X-Api-Key": dummy_api_key},
            {
                "query_type": str(dummy_query_type),
                "query_text": dummy_query_text,
                "use_combined_context": False,
            },
        )


@pytest.mark.asyncio
async def test_cogwit_search_failure():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()
    mock_send_api_request.return_value = ErrorResponse(
        status=404, error={"error": "No results found"}
    )
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_query_text = "Nonexistent data"
        dummy_query_type = SearchType.CHUNKS
        result = await cogwit_instance.search(
            query_text=dummy_query_text, query_type=dummy_query_type
        )
        assert result == SearchError(status=404, error={"error": "No results found"})
        assert mock_send_api_request.call_count == 1


@pytest.mark.asyncio
async def test_cogwit_search_any_non_success_response_returns_SearchError():
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
        status=500, error={"error": "Search service unavailable"}
    )
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_query_text = "Any query"
        dummy_query_type = SearchType.GRAPH_COMPLETION_COT
        result = await cogwit_instance.search(
            query_text=dummy_query_text,
            query_type=dummy_query_type,
            use_combined_context=True,
        )
        assert result == SearchError(
            status=500, error={"error": "Search service unavailable"}
        )
        assert mock_send_api_request.call_count == 1


@pytest.mark.asyncio
async def test_cogwit_search_send_api_request_throws_error():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()

    class MyException(Exception):
        pass

    mock_send_api_request.side_effect = MyException("Search timeout")
    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        dummy_query_text = "Test query"
        dummy_query_type = SearchType.INSIGHTS
        with pytest.raises(MyException):
            await cogwit_instance.search(
                query_text=dummy_query_text, query_type=dummy_query_type
            )
        assert mock_send_api_request.call_count == 1


@pytest.mark.asyncio
async def test_cogwit_search_with_empty_query_text():
    dummy_api_key = "2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc"
    cogwit_instance = cogwit(
        CogwitConfig(
            api_key=dummy_api_key,
        )
    )
    mock_send_api_request = AsyncMock()
    dummy_payload = {"message": "Lorem ipsum"}
    mock_send_api_request.return_value = SuccessResponse(status=200, data=dummy_payload)

    with patch("cogwit_sdk.cogwit.cogwit.send_api_request", mock_send_api_request):
        empty_query_text = ""
        dummy_query_type = SearchType.SUMMARIES
        result = await cogwit_instance.search(
            query_text=empty_query_text, query_type=dummy_query_type
        )

        assert mock_send_api_request.call_args[0] == (
            "/search",
            "post",
            {"X-Api-Key": dummy_api_key},
            {
                "query_type": str(dummy_query_type),
                "query_text": empty_query_text,
                "use_combined_context": False,
            },
        )
        assert result == SearchResponse(status=200, payload=dummy_payload)
        assert mock_send_api_request.call_count == 1
