from abc import ABC

import requests

from aicrowd.constants import (
    AICROWD_API_ENDPOINT,
    GITLAB_API_ENDPOINT,
    RAILS_API_ENDPOINT,
)


class APIResponse:
    def __init__(self, r: requests.Response):
        self.ok = r.ok
        self.status_code = r.status_code
        self.reason = r.reason
        self.text = r.text
        self.headers = r.headers

        self.response_object = r

    def json(self):
        return self.response_object.json()


class Request(ABC):
    """
    For sending requests to Rails and AIcrowd API

    Only some of the HTTP methods have been implemented since others aren't being used.
    """

    def __init__(self, base_url: str, api_key: str = None):
        if api_key:
            self.headers = {"Authorization": f"Token {api_key}"}
        else:
            self.headers = {}

        self.api_key = api_key
        self.base_url = base_url

    def get(
        self, url: str, params: dict = None, headers: dict = {}, **kwargs
    ) -> APIResponse:
        """
        sends a GET request

        Args:
            url: the location to which the request must be sent
            params: the query parameters
            headers: http headers

        Returns:
            requests.Response
        """
        return APIResponse(
            requests.get(
                self.base_url + url,
                params=params,
                headers={**self.headers, **headers},
                **kwargs,
            )
        )

    def post(self, url: str, headers: dict = {}, **kwargs) -> APIResponse:
        """
        sends a POST request

        Args:
            url: the location to which the request must be sent
            headers: http headers

        Returns:
            requests.Response
        """
        return APIResponse(
            requests.post(
                self.base_url + url, headers={**self.headers, **headers}, **kwargs
            )
        )


class RailsAPI(Request):
    """
    For sending requests to Rails API
    """

    def __init__(self, api_key: str = None):
        super().__init__(base_url=RAILS_API_ENDPOINT, api_key=api_key)


class AIcrowdAPI(Request):
    """
    For sending requests to AIcrowd API
    """

    def __init__(self, api_key: str = None):
        super().__init__(base_url=AICROWD_API_ENDPOINT, api_key=api_key)


class GitlabAPI(Request):
    """
    For sending requests to Gitlab
    """

    def __init__(self, base_url=GITLAB_API_ENDPOINT):
        super().__init__(base_url=base_url, api_key=None)
