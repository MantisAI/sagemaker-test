.DEFAULT_GOAL := all
PYTHON_VERSION := python3.7
ECR := 203149375586.dkr.ecr.eu-west-1.amazonaws.com
S3_BUCKET := s3://muanalytics-data/sagemaker-test

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

#
# Log in to access ECR repositories
#
# Note that old versions of awscli may experience issues with command.
#

.PHONY: docker-login
docker-login:
	aws ecr get-login-password --region=eu-west-1 | sudo  docker login \
		--username AWS --password-stdin ${ECR}

#
# Transcribe .envrc in .env for use in docker-compose (it will not read .envrc)
#

.env: .envrc
	sed -e "/export/!d" -e "s/export //g" $< > $@ 

#
# Get glove word embedding
#

WE_URL := http://nlp.stanford.edu/data
WE_ARCHIVE := glove.6B.zip
WE_TEXT := glove.6B.50d.txt

data/raw/$(WE_ARCHIVE): data/raw
	curl -L $(WE_URL)/$(WE_TEXT) --output $@

data/raw/$(WE_TEXT): data/raw/$(WE_ARCHIVE) data/raw 
	(cd data/raw/ && unzip $< $@)

get_word_embedding: data/raw/$(WE_TEXT)

#
# Syncing data with s3. Note that DVC is preferred to this blanket approach!
#

.PHONY: sync_data_to_s3
sync_data_to_s3:
	aws s3 sync data/processed $(S3_BUCKET)/data/processed --exclude '*old/*'

.PHONY: sync_data_from_s3
sync_data_from_s3:
	aws s3 sync $(S3_BUCKET)/data/processed data/processed --exclude '*old/*'

all: virtualenv
