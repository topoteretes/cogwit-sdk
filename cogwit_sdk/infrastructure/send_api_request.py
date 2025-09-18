import os
import aiohttp
from pydantic import BaseModel
from typing import Any, Dict, Generic, Optional, TypeVar, Union


from .json_encoder import json_encoder
from enum import Enum


api_base = os.getenv("COGWIT_API_BASE", "https://api.cognee.ai")


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


class SuccessResponse(BaseModel, Generic[DataType]):
    status: int
    data: DataType


class ErrorResponse(BaseModel):
    status: int
    error: Union[str, Dict[str, Any]]


async def send_api_request(
    api_endpoint,
    method: str,
    headers,
    payload: Optional[Any] = None,
) -> Union[SuccessResponse[Any], ErrorResponse]:
    async with aiohttp.ClientSession() as session:
        http_method = HttpMethod(method.lower())
        method_has_payload = http_method.has_payload()
        method_func = getattr(session, method)

        if method_has_payload:
            async with method_func(
                f"{api_base}/api{api_endpoint}",
                json=json_encoder(payload),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120 * 60, sock_connect=30),
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
                        error=await response.json()
                        if response.status != 500
                        else await response.text(),
                    )

        else:
            async with method_func(
                f"{api_base}/api{api_endpoint}", headers=headers
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
