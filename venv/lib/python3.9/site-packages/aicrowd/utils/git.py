import git
from git.exc import GitError


def ls_remote(url):
    remote_refs = {}
    g = git.cmd.Git()

    try:
        for ref in g.ls_remote(url).split("\n"):
            hash_ref_list = ref.split("\t")
            remote_refs[hash_ref_list[1]] = hash_ref_list[0]
    except GitError:
        return {}

    return remote_refs
