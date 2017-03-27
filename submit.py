#!/usr/bin/env python2.7
# Statice competition submission script.
#
# The script checks competition entries for basic validty and then submits.
# It tries to warn users about common mistakes and guide them through the
# docker process.
#

from __future__ import print_function, with_statement

import getpass
import logging
import os
import os.path as osp
import subprocess
import sys

input = raw_input  # remove in python3

DEFAULT_COMPETITION_NAME = 'clue'
BUILD_IMAGE_NAME = 'statice.tmp'
EXPECTED_HEADERS = 'user_id,day_in_cycle,symptom,probability'
REGISTRY_ADDR = 'statice.wattx.io:5000'


STATICE_CHECK_FUNCTIONS = []

logging.basicConfig(level=logging.INFO)


def run_checks_and_submit():
    """Run all the STATICE_CHECK_FUNCTIONS in order, stopping on the first error.

    A check function is any 'statice_check' decorated function that returns
    a positive value on failure which is optionally a string.
    """
    logging.info('Running basic checks...')
    for err in (f() for f in STATICE_CHECK_FUNCTIONS):
        if err:
            if isinstance(err, str):
                print(err, file=sys.stderr)
            sys.exit(1)
    logging.info('Successfully submitted.')
    logging.info('Check http://statice.wattx.io/submissions '
                 'for the status of your submission.')


def statice_check(fn):
    """Decorator for adding a function as a submission step/check.

    Functions are run in the order they are added, to facilitate
    pipelining
    """
    STATICE_CHECK_FUNCTIONS.append(fn)


def run_command(args):
    """Run the command plus arguments in list ignoring STDOUT."""
    # For python3:
    # return subprocess.call(args, stdout=subprocess.DEVNULL, timeout=60) == 0
    print ('args:', ' '.join(args))
    with open(os.devnull, 'w') as fnull:
        return subprocess.call(args, stdout=fnull) == 0


def check_file(filepath):
    if not osp.isfile(filepath):
        return (
            'File `%s` not found. Running this command in the right folder?'
            % filepath)


@statice_check
def login():
    logging.info('Testing docker login...')
    login_command = [
        'docker', 'login', REGISTRY_ADDR,
        '-u', competition_username, '-p', competition_password
    ]
    if not run_command(login_command):
        return ('Failed to login to %s.'
                ' Docker running? Correct credentials?' % REGISTRY_ADDR)


@statice_check
def check_base_files():
    logging.info('Checking for base files in container...')
    return (
        check_file('Dockerfile') or
        check_file('run.sh')
    )


@statice_check
def check_docker_installed():
    logging.info('Testing docker installation...')
    if not run_command(['docker', 'info']):
        return 'Failed to connect to docker, is it installed and running?'


@statice_check
def check_docker_build():
    logging.info('Testing docker build...')
    if not run_command(['docker', 'build', '-t', BUILD_IMAGE_NAME, '.']):
        return 'Failed to build submission.'


@statice_check
def test_files_inside_container():
    logging.info('Checking files inside container...')
    return not run_command([
        'docker', 'run', '-i',
        '-v', ('%s/test/check_inside_container.sh:'
               '/check_inside_container.sh') % os.getcwd(),
        '--entrypoint', '/check_inside_container.sh',
        BUILD_IMAGE_NAME
    ])


@statice_check
def test_run_submission():
    with open('test/result.txt', 'w+'):
        pass  # create empty test file

    return 0
    # TODO: python 2 doesn't have a timeout on subprocess call. So
    # if the user creates a Dockerfile without an Entrypoint then
    # it will hang forever.
    logging.info('Test running submission...')
    test_run_command = [
        'docker', 'run', '-i',
        '-v', '%s/data:/data' % os.getcwd(),
        '-v', '%s/test/result.txt:/result.txt' % os.getcwd(),
        BUILD_IMAGE_NAME
    ]
    if not run_command(test_run_command):
            return 'Failed in trial run of submission.'

    # Try and check if the submitted program actually produces meaningful
    # results file.
    logging.info('Checking results file...')
    with open('test/result.txt', 'r') as f:
        contents = f.read().splitlines()
        if not len(contents) > 1:
            return (
                'The result.txt your file produces does not have any actual' +
                ' results. See test/result.txt for what was produced' +
                ' using sample data.'
            )

        header = contents[0]
        if not set(EXPECTED_HEADERS.split(',')) == set(header.split(',')):
            return (
                'The result.txt your file produces does not have a correct' +
                ' CSV header. Got `%s` instead of expected `%s`.' %
                (header.split(','), EXPECTED_HEADERS)
            )


@statice_check
def tag_submission():
    logging.info('Tagging submission...')
    if not run_command(['docker', 'tag', BUILD_IMAGE_NAME, competition_repo]):
        return 'Failed to docker tag image'


@statice_check
def push_submission():
    logging.info('Pushing submission to {}...'.format(competition_repo))
    if not run_command(['docker', 'push', competition_repo]):
        return 'Failed to docker push'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--competition', default=DEFAULT_COMPETITION_NAME,
        help='tag for this submission')
    parser.add_argument(
        '-u', '--statice_username',
        help='login username/email for statice.wattx.io help')
    parser.add_argument(
        '-p', '--statice_password',
        help='login password for statice.wattx.io help')
    parser.add_argument(
        '-t', '--tag', default='latest',
        help='tag for this submission')

    args = parser.parse_args()

    competition_username = (args.statice_username or
                            input('competition username:'))
    competition_password = (args.statice_password or
                            getpass.getpass('competition password:'))
    competition_repo = '%s/%s/%s:%s' % (
        REGISTRY_ADDR, args.competition,
        competition_username.replace('@', '_'), args.tag)

    logging.info('Starting checks and submission...')
    run_checks_and_submit()
