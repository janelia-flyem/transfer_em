# transferem_docker 

This directory creates a docker containers for use with transfer em.
It contains a function for fetching data from ng precomputed datasets.

## Local installation instructions (for local testing)

(to install this container using Google please see "Deploying on cloud run" below)

This is a docker container that can be built locally using

	% docker build . -t transferem

and the container launched with

	% docker run -e "PORT=8080" -p 8080:8080 -v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/keys/FILE_NAME.json:ro  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/FILE_NAME.json transferem

This will start up a web client that listens to 127.0.0.1:8080.  The [GOOGLE_APPLICATION_CREDENTIALS](https://cloud.google.com/docs/authentication/production#obtaining_and_providing_service_account_credentials_manually) is an environment variable
that allows you to use google cloud storage locally.  The -v and -e options can be omitted if you are not using this feature.

## Using transferem for cloud headless commands

To run transferem through the web service simply post a JSON (configuration details below):

	% curl -X POST -H "Content-type: application/json" --data-binary @examples/config.json 127.0.0.1:8080/[end point]

The supported endpoints are:

* volume 

```json
{
	"source": "gs precomputed path like blah/jpeg",
	"start": [0,100,0],
	"size": [256,256,256]
}
```

## Deploying on cloud run

Create a google cloud account and install gcloud.

Build the container and store in cloud.

	% gcloud builds submit --tag gcr.io/[PROJECT_ID]/transferem

If a container already exists, one can build faster from cache with this command
([PROJECT_ID] should be replaced in the YAML with the appropriate project id):

	% gcloud builds submit --config cloudbuild.yaml

Alternatively, one can use docker to build and deploy, which is many cases could be
more convenient since the locally tested image can just be uploaded.  The following
links gcloud with docker, builds a container, and uploads:

	% configure docker with gcloud: gcloud auth configure-docker
	$ docker build . -t gcr.io/flyem-private/transferem
	$ docker push  gcr.io/flyem-private/transferem

Once the container is built, deploy to cloud run with the following command.
The instance is created is 2GB of memory and sets the concurrency to 16
per instance.  One cores is specified
as one core does not perform well in limited tests.  One should make
the endpoint private to avoid unauthorized use.

	% gcloud run deploy --memory 2048Mi --concurrency 8 --cpu 2 --image gcr.io/[PROJECT_ID]/transferem --platform managed 

## Invoking cloud run

The resulting endpoint can be invoked through its HTTP api.  One must specify
a bearer token for authentication.  For convenience, when testing the API with curl
one can make an alias that is authenticated with the following:

	% alias gcurl='curl --header "Authorization: Bearer $(gcloud auth print-identity-token)"'

To write a single slice aligned slice (must have load the sample image into a directory dest-bucket/raw/):

% gcurl -H "Content-Type: application/json" -X POST --data-binary @example/align_config.json  https://[CLOUD RUN ADDR]/alignedslice
