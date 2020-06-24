import re
import subprocess


def get_git_info():
    git_branch_name = ''
    try:
        git_branch_name = subprocess.check_output(["git", "branch"], stderr=subprocess.STDOUT)
        git_branch_name = re.search(r'\* (.*)\n', git_branch_name.decode()).group(1)
    except subprocess.CalledProcessError as e:
        print('command {} failed'.format(' '.join(e.cmd)))
        print('mlflow run exception {}'.format(e.output.decode()))
        exit(1)
    git_commit_hash = ''
    try:
        git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
        git_commit_hash = git_commit_hash.decode()
    except subprocess.CalledProcessError as e:
        print('command {} failed'.format(' '.join(e.cmd)))
        print('mlflow run exception {}'.format(e.output.decode()))
        exit(1)
    git_origin_url = ''
    try:
        git_origin_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"],
                                                 stderr=subprocess.STDOUT)
        git_origin_url = re.search(r'git@(.*)\.git', git_origin_url.decode()).group(1)
        git_origin_url = "https://" + git_origin_url.replace(":", "/") + "/tree/" + git_commit_hash
    except subprocess.CalledProcessError as e:
        print('command {} failed'.format(' '.join(e.cmd)))
        print('mlflow run exception {}'.format(e.output.decode()))
        exit(1)
    return git_branch_name, git_origin_url