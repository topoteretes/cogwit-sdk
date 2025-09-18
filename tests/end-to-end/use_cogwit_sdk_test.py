import os
from cogwit_sdk import cogwit, CogwitConfig
from cogwit_sdk.responses import (
    AddResponse,
    CognifyResponse,
    CombinedSearchResult,
    SearchResult,
)

cogwit_config = CogwitConfig(
    api_key=os.getenv("COGWIT_API_KEY", ""),
)

cogwit_instance = cogwit(cogwit_config)


async def main():
    result = await cogwit_instance.add(
        data="Test data",
        dataset_name="test_dataset",
    )
    assert isinstance(result, AddResponse)
    assert result.status == "PipelineRunCompleted"
    assert "test_dataset" == result.dataset_name

    dataset_id = result.dataset_id

    result = await cogwit_instance.cognify(
        dataset_ids=[dataset_id],
    )
    assert isinstance(result, CognifyResponse)
    dataset_result = result[str(dataset_id)]
    assert dataset_result.status == "PipelineRunCompleted"
    assert dataset_result.dataset_name == "test_dataset"

    search_results = await cogwit_instance.search(
        query_text="What is in data?",
        query_type=cogwit_instance.SearchType.GRAPH_COMPLETION,
    )
    assert isinstance(search_results, list)
    assert all(
        isinstance(search_result, SearchResult) for search_result in search_results
    )
    assert any(
        [search_result.dataset_id == dataset_id for search_result in search_results]
    )

    search_result = await cogwit_instance.search(
        query_text="What is in data?",
        query_type=cogwit_instance.SearchType.GRAPH_COMPLETION,
        use_combined_context=True,
    )
    assert isinstance(search_result, CombinedSearchResult)
    assert any([dataset.id == dataset_id for dataset in search_result.datasets])

    search_results = await cogwit_instance.search(
        query_text="What is in data?",
        query_type=cogwit_instance.SearchType.CHUNKS,
    )
    assert isinstance(search_results, list)
    assert all(
        isinstance(search_result, SearchResult) for search_result in search_results
    )
    assert any(
        [search_result.dataset_id == dataset_id for search_result in search_results]
    )
    assert search_results[0].search_result[0]["text"] == "Test data"

    search_result = await cogwit_instance.search(
        query_text="What is in data?",
        query_type=cogwit_instance.SearchType.CHUNKS,
        use_combined_context=True,
    )
    assert isinstance(search_result, CombinedSearchResult)
    assert any([dataset.id == dataset_id for dataset in search_result.datasets])
    assert search_result.result
    assert search_result.result[0]["text"] == "Test data"


import asyncio

asyncio.run(main())
