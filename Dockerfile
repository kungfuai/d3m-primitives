FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9

COPY . /yonder-primitives

RUN pip install -e ./yonder-primitives