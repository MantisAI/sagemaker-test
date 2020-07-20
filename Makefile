.DEFAULT_GOAL := all
PYTHON_VERSION := python3.7
ECR := 203149375586.dkr.ecr.eu-west-1.amazonaws.com

# Set the default location for the virtualenv to be stored

VIRTUALENV := build/virtualenv

# Create the virtualenv by installing the requirements and test requirements

$(VIRTUALENV)/.installed: requirements.txt
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON_VERSION) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	$(VIRTUALENV)/bin/pip3 install -r requirements_extra.txt
	$(VIRTUALENV)/bin/pip3 install -e .
	touch $@

# Update the requirements to latest. This is required because typically we won't
# want to incldue test requirements in the requirements of the application, and
# because it makes life much easier when we want to keep our dependencies up to
# date.

.PHONY: update-requirements-txt
update-requirements-txt: VIRTUALENV := /tmp/update-requirements-virtualenv
update-requirements-txt:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON_VERSION) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r unpinned_requirements.txt
	echo "# Created by 'make update-requirements-txt'. DO NOT EDIT!" > requirements.txt
	$(VIRTUALENV)/bin/pip freeze | grep -v pkg-resources==0.0.0 >> requirements.txt

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed

.PHONY: docker-login
docker-login:
	aws ecr get-login-password --region=eu-west-1 | sudo  docker login \
		--username AWS --password-stdin ${ECR}

.env: .envrc
	sed -e "/export/!d" -e "s/export //g" $< > $@ 

BASE_URL := http://ai.stanford.edu/~amaas/data/sentiment
RAW_DATA := aclImdb_v1.tar.gz

data/raw: 
	mkdir data/raw -p

data/raw/$(RAW_DATA): data/raw
	curl -L $(BASE_URL)/$(RAW_DATA) --output $@

data/raw/aclImdb: data/raw/$(RAW_DATA)
	(cd data/raw/ && tar -xf $(RAW_DATA))

get_raw_data: data/raw/aclImdb

all: virtualenv
