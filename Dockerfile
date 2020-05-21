FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18

RUN apt update && apt install -y default-jre-headless

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY common-primitives ./common-primitives
RUN pip install -e common-primitives

# RUN pip install -e git+https://github.com/kungfuai/d3m-primitives@summer_submit#egg=kf-d3m-primitives
COPY . kf-d3m-primitives
RUN pip install -e kf-d3m-primitives