import json
import aiohttp
from pydantic import BaseModel
from typing import Any, Dict, Generic, Optional, TypeVar, Union


from .json_encoder import json_encoder
from enum import Enum


class HttpMethod(Enum):
    GET = "get"
    POST = "post"
    PUT = "put"
    DELETE = "delete"
    PATCH = "patch"

    def has_payload(self) -> bool:
        """Returns True if this HTTP method typically includes a request body."""
        return self in {HttpMethod.POST, HttpMethod.PUT}


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
    method: str,
    headers,
    payload: Optional[Any] = None,
) -> Union[SuccessResponse[Union[str, Dict[str, Any]]], ErrorResponse]:
    async with aiohttp.ClientSession() as session:
        http_method = HttpMethod(method.lower())
        method_has_payload = http_method.has_payload()
        method_func = getattr(session, method)

        if method_has_payload:
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

                    return SuccessResponse(
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

                    return SuccessResponse(
                        status=response.status,
                        data=response_data,
                    )
                else:
                    return ErrorResponse(
                        status=response.status,
                        error=await response.json(),
                    )
