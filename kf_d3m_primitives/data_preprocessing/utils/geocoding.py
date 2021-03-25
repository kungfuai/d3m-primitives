import subprocess
import time
import requests
import logging
import sys

logger = logging.getLogger(__name__)


def check_geocoding_server(address, volumes, timeout=100, interval=10):
    # confirm that server is responding before proceeding
    # the `12g` in the following may become a hyper-parameter in the future
    PopenObj = subprocess.Popen(
        ["java", "-Xms12g", "-Xmx12g", "-jar", "photon-0.3.1.jar"],
        cwd=volumes["photon-db-latest"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    counter = interval
    while counter <= timeout:
        time.sleep(interval)
        try:
            r = requests.get(address + "api?q=berlin")
            if r.status_code == 200:
                return PopenObj
            else:
                logger.debug(
                    f"Basic request does not return status code 200, trying again in {interval} seconds"
                )
                counter += interval
        except (ConnectionRefusedError, requests.exceptions.ConnectionError) as error:
            logger.debug(f"Connected refused, trying again in {interval} seconds")
            counter += interval
    sys.exit("Connection has not been accepted and timeout setting expired, exiting...")