from uuid import UUID
from pydantic import BaseModel, RootModel
from typing import Dict, List, Optional, Union, Any
from cogwit_sdk.infrastructure.send_api_request import SuccessResponse, send_api_request
from cogwit_sdk.modules.search.SearchType import SearchType


class CogwitConfig(BaseModel):
    api_key: str


class AddResponse(BaseModel):
    status: str
    dataset_id: UUID
    pipeline_run_id: UUID
    dataset_name: str


class AddError(BaseModel):
    status: int
    error: Union[str, Dict]


class CognifyResult(BaseModel):
    status: str
    dataset_id: UUID
    pipeline_run_id: UUID
    dataset_name: str


class CognifyResponse(RootModel[dict[str, CognifyResult]]):
    def __getitem__(self, dataset_id: str) -> CognifyResult:
        return self.root[dataset_id]


class CognifyError(BaseModel):
    status: int
    error: Union[str, Dict]


class MemifyResponse(RootModel[dict[str, CognifyResult]]):
    def __getitem__(self, dataset_id: str) -> CognifyResult:
        return self.root[dataset_id]


class MemifyError(BaseModel):
    status: int
    error: Union[str, Dict]


# class SearchResponse(BaseModel):
#     results: Dict


class SearchResultDataset(BaseModel):
    id: UUID
    name: str


class CombinedSearchResult(BaseModel):
    result: Optional[Any]
    context: Dict[str, Any]
    graphs: Optional[Dict[str, Any]] = {}
    datasets: List[SearchResultDataset]


class SearchResult(BaseModel):
    search_result: Any
    dataset_id: Optional[UUID]
    dataset_name: Optional[str]


SearchResponse = Union[List[SearchResult], CombinedSearchResult, List[Any]]


class SearchError(BaseModel):
    status: int
    error: Union[str, Dict]


class cogwit:
    config: CogwitConfig

    def __init__(self, config: CogwitConfig):
        self.config = config
        self.SearchType = SearchType

    async def add(
        self,
        data: Union[List[str], str],
        dataset_name: str = "main_dataset",
        dataset_id: Optional[UUID] = None,
        node_set: Optional[List[str]] = None,
    ) -> Union[AddResponse, AddError]:
        response_data = await send_api_request(
            "/add",
            "post",
            {
                "X-Api-Key": self.config.api_key,
                "Content-Type": "application/json",
            },
            {
                "text_data": data if isinstance(data, list) else [data],
                "dataset_id": dataset_id or "",
                "dataset_name": dataset_name,
                "node_set": node_set,
            },
        )

        if isinstance(response_data, SuccessResponse):
            return AddResponse(
                status=response_data.data["status"],
                dataset_id=UUID(response_data.data["dataset_id"]),
                pipeline_run_id=UUID(response_data.data["pipeline_run_id"]),
                dataset_name=response_data.data["dataset_name"],
            )
        else:
            return AddError(
                status=response_data.status,
                error=response_data.error,
            )

    async def cognify(
        self,
        datasets: List[str] = ["main_dataset"],
        dataset_ids: List[UUID] = [],
        temporal_cognify: bool = False,
    ) -> Union[CognifyResponse, CognifyError]:
        response_data = await send_api_request(
            "/cognify",
            "post",
            {
                "X-Api-Key": self.config.api_key,
                "Content-Type": "application/json",
            },
            {
                "datasets": datasets,
                "dataset_ids": dataset_ids,
                "temporal_cognify": temporal_cognify,
            },
        )

        if isinstance(response_data, SuccessResponse):
            return CognifyResponse(
                {
                    dataset_id: CognifyResult(
                        status=result["status"],
                        dataset_id=UUID(result["dataset_id"]),
                        pipeline_run_id=UUID(result["pipeline_run_id"]),
                        dataset_name=result["dataset_name"],
                    )
                    for dataset_id, result in response_data.data.items()
                }
            )
        else:
            return CognifyError(
                status=response_data.status,
                error=response_data.error,
            )

    async def memify(
        self, dataset_name: str = "main_dataset"
    ) -> Union[MemifyResponse, MemifyError]:
        response_data = await send_api_request(
            "/memify",
            "post",
            {
                "X-Api-Key": self.config.api_key,
                "Content-Type": "application/json",
            },
            {
                "dataset_name": dataset_name,
            },
        )

        if isinstance(response_data, SuccessResponse):
            return MemifyResponse(
                {
                    dataset_id: CognifyResult(
                        status=result["status"],
                        dataset_id=UUID(result["dataset_id"]),
                        pipeline_run_id=UUID(result["pipeline_run_id"]),
                        dataset_name=result["dataset_name"],
                    )
                    for dataset_id, result in response_data.data.items()
                }
            )
        else:
            return MemifyError(
                status=response_data.status,
                error=response_data.error,
            )

    async def search(
        self,
        query_text: str,
        query_type: SearchType = SearchType.GRAPH_COMPLETION,
        use_combined_context: bool = False,
        save_interaction: bool = False,
    ) -> Union[SearchResponse, SearchError]:
        response_data = await send_api_request(
            "/search",
            "post",
            {
                "X-Api-Key": self.config.api_key,
                "Content-Type": "application/json",
            },
            {
                "search_type": query_type.value,
                "query": query_text,
                "use_combined_context": use_combined_context,
                "save_interaction": save_interaction,
            },
        )

        if isinstance(response_data, SuccessResponse):
            try:
                return CombinedSearchResult(**response_data.data)
            except (ValueError, TypeError):
                try:
                    return [SearchResult(**result) for result in response_data.data]
                except (ValueError, TypeError):
                    return response_data.data
        else:
            return SearchError(
                status=response_data.status,
                error=response_data.error,
            )
