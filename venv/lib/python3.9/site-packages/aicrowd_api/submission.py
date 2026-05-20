"""
all api calls related to submissions
"""

from aicrowd_api.request import APIResponse, RailsAPI


def get_submission_upload_details(api_key: str, challenge_slug: str) -> APIResponse:
    """
    Gets the presigned s3 upload URL and other details needed for uploading a submission file

    Args:
        api_key: AIcrowd API Key
        challenge_slug: challenge on which submission is being made

    Returns:
        requests response object
    """
    return RailsAPI(api_key).get(
        "/submissions", params={"challenge_id": challenge_slug}, allow_redirects=False
    )


def create_rails_submission(api_key: str, payload: dict) -> APIResponse:
    """
    informs AIcrowd about the uploaded file, which in turn creates the submission on rails

    Args:
        api_key: AIcrowd API Key
        payload: submission data

    Returns:
        requests response object
    """
    return RailsAPI(api_key).post("/submissions", json=payload)
