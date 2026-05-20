"""
all api calls related to challenges
"""

from typing import Union

from aicrowd_api.request import AIcrowdAPI, APIResponse, RailsAPI


def get_challenge_details(api_key: str, params: dict) -> APIResponse:
    """
    Gets the challenge details (id, slug, etc) on giving a query filter

    Args:
        api_key: AIcrowd API Key
        params: filter to be placed on challenges

    Returns:
        requests response object
    """
    return AIcrowdAPI(api_key).get("/challenges/", params=params)


def get_challenge_datasets(api_key: str, challenge_id: int) -> APIResponse:
    """
    Gets the datasets for that challenge

    Args:
        api_key: AIcrowd API Key
        challenge_id: challenge id for which datasets are requested

    Returns:
        requests response object
    """
    return AIcrowdAPI(api_key).get(f"/challenges/{challenge_id}/dataset")


def check_registration_status(api_key: str, challenge_id: int) -> Union[bool, None]:
    """
    Checks if the participant has accepted the conditions (has registered) for the challenge

    Args:
        api_key: AIcrowd API Key
        challenge_id: challenge id for which registration status needs to be checked

    Returns:
        whether participant is registered
    """
    r = RailsAPI(api_key).get("/api_user")
    if not r.ok:
        return None

    participant_id = r.json()["id"]

    r = AIcrowdAPI(api_key).get(
        f"/challenges/{challenge_id}/participant",
        params={"participant_id": participant_id},
    )

    if not r.ok:
        return None

    return r.json().get("registered")
