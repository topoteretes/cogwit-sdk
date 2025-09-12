import json
import aiohttp
from pydantic import BaseModel
from typing import Any, Dict, Generic, Optional, TypeVar, TypedDict, Union


from .json_encoder import json_encoder

methods_with_payload = ["post", "put"]

DataType = TypeVar("DataType")

# api_base = "https://api.cognee.ai"
api_base = "http://localhost:8001"


class SuccessResponse(Generic[DataType], BaseModel):
    status: int
    data: DataType


class ErrorResponse(BaseModel):
    status: int
    error: Dict[str, Any]


async def send_api_request(
    api_endpoint,
    method,
    headers,
    payload: Optional[Any] = None,
    ResponsePayloadType=Union[str, TypedDict],
) -> Union[SuccessResponse, ErrorResponse]:
    async with aiohttp.ClientSession() as session:
        method_func = getattr(session, method)

        if method in methods_with_payload:
            async with method_func(
                f"{api_base}{api_endpoint}",
                json=json.dumps(json_encoder(payload)),
                headers=headers,
            ) as response:
                if response.status >= 200 and response.status < 300:
                    if headers.get("Content-Type", "") == "application/json":
                        response_data = await response.json()
                    else:
                        response_data = await response.text()

                    return SuccessResponse[ResponsePayloadType](
                        status=response.status,
                        data=response_data,
                    )
                else:
                    return ErrorResponse(
                        status=response.status,
                        error=await response.json(),
                    )

        else:
            async with method_func(
                f"{api_base}{api_endpoint}", headers=headers
            ) as response:
                if response.status == 200:
                    if headers.get("Content-Type", "") == "application/json":
                        response_data = await response.json()
                    else:
                        response_data = await response.text()

                    return SuccessResponse[ResponsePayloadType](
                        status=response.status,
                        data=response_data,
                    )
                else:
                    return ErrorResponse(
                        status=response.status,
                        error=await response.json(),
                    )
