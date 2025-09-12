from uuid import UUID
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
from cogwit_sdk.infrastructure.send_api_request import SuccessResponse, send_api_request
from cogwit_sdk.modules.search.SearchType import SearchType


class CogwitConfig(BaseModel):
    api_key: str


class AddResponse(BaseModel):
    status: int
    dataset_id: UUID
    pipeline_run_id: UUID


class AddError(BaseModel):
    status: int
    error: Dict


class CognifyResponse(BaseModel):
    status: int
    pipeline_run_id: UUID


class CognifyError(BaseModel):
    status: int
    error: Dict


class SearchResponse(BaseModel):
    status: int
    payload: Dict


class SearchError(BaseModel):
    status: int
    error: Dict


class cogwit:
    config: CogwitConfig

    def __init__(self, config: CogwitConfig):
        self.config = config

    async def add(self, data: Union[List[str], str], dataset_name: str, dataset_id: Optional[UUID] = None) -> Union[AddResponse, AddError]:
        response_data = await send_api_request(
            "/add",
            "post",
            {
                "X-Api-Key": self.config.api_key,
            },
            {
                "text_data": data,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
            },
        )

        if isinstance(response_data, SuccessResponse):
            return AddResponse(
                status=response_data.status,
                dataset_id=UUID(response_data.data["dataset_id"]),
                pipeline_run_id=UUID(response_data.data["pipeline_run_id"]),
            )
        else:
            return AddError(
                status=response_data.status,
                error=response_data.error,
            )

    async def cognify(self, dataset_ids: List[UUID]) -> Union[CognifyResponse, CognifyError]:
        response_data = await send_api_request(
            "/cognify",
            "post",
            {
                "X-Api-Key": self.config.api_key,
            },
            {
                "dataset_ids": dataset_ids,
            },
        )

        if isinstance(response_data, SuccessResponse):
            return CognifyResponse(
                status=response_data.status,
                pipeline_run_id=UUID(response_data.data["pipeline_run_id"]),
            )
        else:
            return CognifyError(
                status=response_data.status,
                error=response_data.error,
            )

    async def search(
        self,
        query_text: str,
        query_type: SearchType,
        use_combined_context: bool = False,
    ) -> Union[SearchResponse, SearchError]:
        response_data = await send_api_request(
            "/search",
            "post",
            {
                "X-Api-Key": self.config.api_key,
            },
            {
                "query_type": str(query_type),
                "query_text": query_text,
                "use_combined_context": use_combined_context,
            },
        )

        if isinstance(response_data, SuccessResponse):
            return SearchResponse(
                status=response_data.status,
                payload=response_data.data,
            )
        else:
            return SearchError(
                status=response_data.status,
                error=response_data.error,
            )





if __name__ == "__main__":
    async def main():
        cogwit_instance = cogwit(
            CogwitConfig(
              api_key="2286ec4a1aac7ce7a8176222338841d4ccbd2936111314cc",
            )
        )

        result = await cogwit_instance.add(data="Test data", dataset_name="test_dataset")
        print(result)

        if result.status == "success":
            result = await cogwit_instance.cognify(
                dataset_ids=[result.dataset_id] # type: ignore
            )

            print(result)

            if result.status == "success":
                results = await cogwit_instance.search(
                    "What is in data?",
                    SearchType.GRAPH_COMPLETION,
                    use_combined_context=True,
                )

    import asyncio
    asyncio.run(main())